from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, Sampler

from manifeel.tactile_ae_v3.sampling import TemporalSamplingPolicyV3
from manifeel.tactile_ae_v4.window_filter import load_window_indices_v4
from manifeel.tactile_ae_v41.config_loader import load_tactile_ae_v41_config
from manifeel.tactile_ae_v41.config import ForceNormalizationV41Config
from manifeel.tactile_ae_v41.normalizer import ForceNormalizerV41
from manifeel.tactile_ae_v41.manifeel_normalizer import ManiFeelForceNormalizerV41, ManiFeelForceNormalizationV41Config
from manifeel.tactile_ae_v41.data_builder import build_robotwin_sources_v41
from manifeel.tactile_ae_v41.data import CompactNativeWindowDatasetV41
from manifeel.tactile_ae_v41.dexjoco_data import DexJoCoWindowDatasetV41
from manifeel.tactile_ae_v41.manifeel_data import ManiFeelWindowDatasetV41, MANIFEEL_TASK_DIRECTORIES
from .layout import surface_slots
from .replay_data import ManifestReplayDataset

STATES = ("no_contact", "onset_release", "dynamic", "stable")
REPO = Path(__file__).resolve().parents[2]


def load_config(path):
    def expand(x):
        if isinstance(x, dict): return {k: expand(v) for k, v in x.items()}
        if isinstance(x, list): return [expand(v) for v in x]
        if isinstance(x, str): return os.path.expandvars(x)
        return x
    config = expand(yaml.safe_load(Path(path).read_text()))
    names = [d["name"] for d in config["domains"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate domain names")
    for d in config["domains"]:
        if not d["enabled"]:
            continue
        if not d.get("root") or not d.get("normalization"):
            raise ValueError(f"{d['name']}: enabled domain requires audited root and normalization")
        if d["normalization"] not in config["normalizations"]:
            raise ValueError(f"{d['name']}: missing independent normalization group")
        if not np.isfinite(d["weight"]) or d["weight"] <= 0:
            raise ValueError("enabled domain weights must be finite and positive")
    return config


def make_normalizer(spec):
    if spec["type"] == "manifeel_native":
        return ManiFeelForceNormalizerV41(ManiFeelForceNormalizationV41Config(**spec["parameters"]))
    if spec["type"] == "positive_normal_si":
        return ForceNormalizerV41(ForceNormalizationV41Config(**spec["parameters"]))
    raise ValueError("unrecognized normalization convention")


class MixedWindowDataset(Dataset):
    """CPU normalization occurs per domain before packing heterogeneous surfaces."""

    def __init__(self, config):
        self.domains = [d for d in config["domains"] if d["enabled"]]
        self.datasets, self.normalizers, self.rows, self.slots = [], [], [], []
        self.fingerprints = {}
        robot_cache = {}
        for d in self.domains:
            norm_spec = config["normalizations"][d["normalization"]]
            self.normalizers.append(make_normalizer(norm_spec))
            kind = d["kind"]
            if kind == "robotwin":
                cache_key = tuple(d[k] for k in ("source_config", "index", "source_manifest", "stats"))
                if cache_key not in robot_cache:
                    cfg = load_tactile_ae_v41_config(REPO / d["source_config"])
                    sources = build_robotwin_sources_v41(cfg, split="train", source_manifest=Path(d["source_manifest"]))
                    indices = load_window_indices_v4(d["index"], stats_path=d["stats"], source_manifest_path=d["source_manifest"])
                    robot_cache[cache_key] = (cfg, sources, indices)
                cfg, sources, indices = robot_cache[cache_key]
                root_key = "clean_root" if d["name"] == "robotwin_clean" else "random_root"
                if Path(d["root"]).resolve() != Path(cfg["robotwin_v41"][root_key]).resolve():
                    raise ValueError("RoboTwin root differs from the audited source config")
                if norm_spec["parameters"] != cfg["normalization_v41"]:
                    raise ValueError("RoboTwin normalization differs from selected V4.1 recipe")
                index = indices[d["name"]]
                policy = TemporalSamplingPolicyV3.from_mapping(cfg["temporal_sampling_v41"])
                dataset = CompactNativeWindowDatasetV41(sources[d["name"]], index, source_id=len(self.datasets), policy=policy.sources[d["name"]])
                rows = [vars(w) for w in index.windows]
            else:
                artifact = torch.load(d["index"], map_location="cpu", weights_only=False)
                if artifact.get("split") != "train" or artifact.get("normalization") != norm_spec["parameters"]:
                    raise ValueError(f"{d['name']}: split or normalization differs from window index")
                if artifact.get("tasks") != d["tasks"]:
                    raise ValueError(f"{d['name']}: task allowlist differs from audited index")
                if artifact.get("window_frames", 18) != 18 or artifact.get("inner_stride", 1) != 1:
                    raise ValueError("all domains require 18 consecutive native frames")
                if kind == "manifeel":
                    if artifact.get("task_mapping") != dict(MANIFEEL_TASK_DIRECTORIES):
                        raise ValueError("ManiFeel directory mapping mismatch")
                    dataset = ManiFeelWindowDatasetV41(artifact, Path(d["root"]))
                elif kind == "dexjoco":
                    if d.get("successful_manifest"):
                        split_digest = hashlib.sha256(Path(d["episode_split"]).read_bytes()).hexdigest()
                        if artifact.get("episode_split_sha256") != split_digest:
                            raise ValueError("replay episode split changed since indexing")
                        dataset = ManifestReplayDataset(artifact, Path(d["root"]), d["successful_manifest"])
                    else:
                        dataset = DexJoCoWindowDatasetV41(artifact, Path(d["root"]), handle_cache_size=4)
                else:
                    raise ValueError(f"unknown domain kind {kind}")
                rows = artifact["windows"]
            self.datasets.append(dataset)
            self.rows.append(rows)
            self.slots.append(surface_slots(kind, hand=d.get("hand", "right"), finger=d.get("finger", "index")))
            for key in ("index", "source_manifest", "stats", "successful_manifest", "episode_split"):
                if d.get(key):
                    digest = hashlib.sha256()
                    with open(d[key], "rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            digest.update(chunk)
                    self.fingerprints[f"{d['name']}/{key}"] = digest.hexdigest()

    def __len__(self):
        return sum(map(len, self.datasets))

    def __getitem__(self, key):
        domain, item = key
        sample = self.datasets[domain][item]
        force = sample["force_physical"].float()
        support = sample["support_mask"].bool()
        if force.shape[0] != 18 or not torch.isfinite(force).all():
            raise ValueError("window is not 18 finite native frames")
        observed = support.flatten(1).any(-1)
        force, support = force[:, observed], support[observed]
        slots = self.slots[domain][observed]
        if not len(slots):
            raise ValueError("training window has no observed physical surface")
        if len(torch.unique(slots[slots >= 0])) != int((slots >= 0).sum()):
            raise ValueError("multiple surfaces write to the same hand slot")
        with torch.no_grad():
            normalized, _, contact = self.normalizers[domain].normalize(force, support[None])
        return {"force": normalized.transpose(0, 1).contiguous(),
                "contact_target": contact.transpose(0, 1).contiguous(),
                "support": support, "slots": slots, "domain": domain,
                "times": sample["frame_times"], "item": item,
                "state": int(sample["state_id"])}

    def report(self):
        from collections import Counter
        return {d["name"]: {"root": d["root"], "windows": len(rows),
                "tasks": len(set(r["task"] for r in rows)),
                "states": dict(Counter(r["state"] for r in rows)),
                "normalization_group": d["normalization"]}
                for d, rows in zip(self.domains, self.rows)}


def collate_windows(samples):
    return {**{key: torch.cat([s[key] for s in samples])
               for key in ("force", "contact_target", "support", "slots")},
            "owner": torch.cat([torch.full((len(s["slots"]),), i, dtype=torch.long)
                                for i, s in enumerate(samples)]),
            "domain": torch.tensor([s["domain"] for s in samples]),
            "state": torch.tensor([s["state"] for s in samples]),
            "item": torch.tensor([s["item"] for s in samples]),
            "times": torch.stack([s["times"] for s in samples]),
            "batch_size": len(samples)}


class MixedBatchSampler(Sampler):
    """Deterministic batch-addressed RNG: prefetch cannot advance resume state.

    Systematic stratification mixes domains inside each local batch. It has
    unbiased configured proportions and then randomly permutes sample order.
    State sampling caps the desired no-contact mass; tasks/episodes are uniform
    inside the chosen state. No rank receives a domain-specific loader.
    """

    def __init__(self, rows, weights, batch_size, batches, *, seed=20260907,
                 rank=0, world=1, start=0, no_contact_probability=0.1):
        if not 0 <= no_contact_probability < 1 or batch_size < 1:
            raise ValueError("invalid batch size or no-contact probability")
        self.batch_size, self.batches, self.seed = batch_size, batches, seed
        self.rank, self.world, self.start = rank, world, start
        self.weights = np.asarray(weights, dtype=float)
        if len(weights) != len(rows) or not len(rows) or not np.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("invalid active domain weights")
        self.weights /= self.weights.sum()
        self.pools = []
        for domain_rows in rows:
            pool = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for i, r in enumerate(domain_rows):
                pool[r["state"]][r["task"]][r["episode_index"]].append(i)
            if not pool or set(pool) - set(STATES):
                raise ValueError("empty index or unknown contact state")
            # A wholly contact-free source is not silently promoted to 100% zeros.
            if not any(s != "no_contact" for s in pool):
                raise ValueError("domain has no contact-bearing windows; audit before mixing")
            converted = {}
            for state, tasks in pool.items():
                converted[state] = [list(episodes.values()) for episodes in tasks.values()]
            states = list(converted)
            p = np.array([0.0 if s == "no_contact" else {"onset_release": .30, "dynamic": .35, "stable": .25}[s] for s in states])
            p /= p.sum()
            if "no_contact" in states:
                p *= 1 - no_contact_probability
                p[states.index("no_contact")] = no_contact_probability
            self.pools.append((states, p, converted))

    def __len__(self):
        return max(0, self.batches - self.start)

    def __iter__(self):
        for step in range(self.start, self.batches):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, self.rank, self.world]))
            positions = (np.arange(self.batch_size) + rng.random()) / self.batch_size
            domains = np.searchsorted(np.cumsum(self.weights), positions, side="right").clip(max=len(self.pools)-1)
            rng.shuffle(domains)
            result = []
            for d in domains:
                states, probabilities, pools = self.pools[d]
                state = states[int(rng.choice(len(states), p=probabilities))]
                tasks = pools[state]
                episodes = tasks[int(rng.integers(len(tasks)))]
                anchors = episodes[int(rng.integers(len(episodes)))]
                result.append((int(d), anchors[int(rng.integers(len(anchors)))]))
            yield result
