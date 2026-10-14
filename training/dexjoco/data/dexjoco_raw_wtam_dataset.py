"""WTAM loader for the aligned 50 Hz DexJoCo raw-data/TFA-1.0 collection."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from bisect import bisect_right
import json
import math
import multiprocessing
from pathlib import Path
import random
from typing import Any, Sequence

import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from safetensors import safe_open

try:
    import zarr
except ModuleNotFoundError:
    # CUDA inference imports TaskT5Cache from this module but never opens raw
    # training arrays. Keep zarr a training/data-loading-only dependency.
    zarr = None

from training.dexjoco.data.dexjoco_wtam_dataset import (
    BIMANUAL_TASKS, SINGLE_ARM_TASKS, DEFAULT_TASK_PROMPTS, RAW_PAD_NAMES,
    DexJoCoWTAMDataset, _pad_right_arm_22_to_44,
    _h5_take_allow_duplicates,
)
from runtime.dexjoco.normalization import load_quantile_normalizer44
from runtime.dexjoco.text_cache import DiskT5Cache
from runtime.dexjoco.contracts import official_state_to_rotvec_proprio


@dataclass(frozen=True)
class RawEpisode:
    task_name: str
    episode_index: int
    source_root: Path
    tactile_path: Path
    crop_start: int
    length: int
    source_embodiment: str
    prompt: str


from runtime.dexjoco.text_cache import TaskT5Cache


class DexJoCoRawAlignedWTAMDataset(Dataset):
    """1080 accepted episodes with exact 50 Hz V/A/T timestamps.

    Condition is native step ``t``. Action and tactile targets are
    ``t+3,...,t+48``; video targets are ``t+6,...,t+48``.
    """

    def __init__(
        self,
        root: str,
        t5_cache_path: str,
        normalization_stats_path: str,
        embodiment: str = "multitask",
        samples_per_episode: int = 32,
        action_count: int = 16,
        action_stride: int = 3,
        num_video_frames: int = 8,
        video_stride: int = 6,
        video_size: tuple[int, int] = (384, 320),
        tasks: Sequence[str] | None = None,
        task_sampling: str = "proportional_to_windows",
        sampling_mode: str = "raw_episode_uniform",
        va_uniform_window_sampling: bool = False,
        sampling_seed: int = 42,
        sampling_global_batch_size: int = 64,
        sequential_video_decode: bool = False,
        preload_task_t5_embeddings: bool = False,
        zarr_group_cache_size: int = 0,
        manifest_filename: str = "accepted_manifest_1080.json",
        manifest_schema: str = "dexjoco_11task_tfa_alignment_v2",
        include_tactile: bool = True,
        condition_only: bool = False,
        val_enabled: bool = False,
        val: bool = False,
        image_aug: bool = False,
    ) -> None:
        # ``val_enabled`` is a trainer-level config switch that is forwarded
        # through the generic dataset factory; the dataset itself only needs
        # the concrete ``val`` split flag.
        del val_enabled
        if image_aug:
            raise ValueError("DexJoCo raw-data V2.5 does not use image augmentation")
        if embodiment != "multitask":
            raise ValueError("The raw-data contract is the unified 44-D multitask model")
        if (action_count, action_stride, num_video_frames, video_stride) != (16, 3, 8, 6):
            raise ValueError("Raw-data contract requires A/T=16x3 and V=8x6")
        self.root = Path(root)
        # The trainer uses this public contract to enable global
        # embodiment-homogeneous sampling before constructing the DataLoader.
        # Without it, random records can mix 5-token single-arm and 10-token
        # bimanual samples in one batch, which cannot share one compact MoT
        # sequence length.
        self.embodiment = embodiment
        self.action_dim = 44
        self.state_dim = 44
        self.region_count = 30
        manifest_path = Path(manifest_filename)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != manifest_schema:
            raise ValueError(f"Unexpected raw-data manifest: {manifest_path}")
        if float(manifest.get("source_fps", -1)) != 50.0:
            raise ValueError("Raw-data manifest must declare 50 Hz")
        self.action_count, self.action_stride = action_count, action_stride
        self.num_video_frames, self.video_stride = num_video_frames, video_stride
        self.video_size = tuple(video_size)
        self.include_tactile = bool(include_tactile)
        # Keep the legacy TFA-1 reader decision immutable for additive
        # subclasses that expose another tactile protocol via the public flag.
        self._include_tactile_v1 = self.include_tactile
        self.condition_only = bool(condition_only)
        self.samples_per_episode = int(samples_per_episode)
        self.val = bool(val)
        self.task_sampling = str(task_sampling)
        if self.task_sampling not in ("proportional_to_windows", "uniform_over_tasks"):
            raise ValueError(
                f"Unsupported task_sampling={self.task_sampling!r}; "
                "expected proportional_to_windows or uniform_over_tasks"
            )
        self.sampling_mode = str(sampling_mode)
        if self.sampling_mode not in ("window_index", "raw_episode_uniform"):
            raise ValueError(
                f"Unsupported sampling_mode={self.sampling_mode!r}; "
                "expected window_index or raw_episode_uniform"
            )
        self.va_uniform_window_sampling = bool(va_uniform_window_sampling)
        self.sampling_seed = int(sampling_seed)
        self.sampling_global_batch_size = int(sampling_global_batch_size)
        self.sequential_video_decode = bool(sequential_video_decode)
        self.preload_task_t5_embeddings = bool(preload_task_t5_embeddings)
        self.zarr_group_cache_size = int(zarr_group_cache_size)
        if self.zarr_group_cache_size < 0:
            raise ValueError("zarr_group_cache_size must be non-negative")
        # Each DataLoader worker owns its own small LRU.  Zarr DirectoryStore
        # objects are deliberately never shared between worker processes.
        self._zarr_group_cache: OrderedDict[str, Any] = OrderedDict()
        if self.va_uniform_window_sampling:
            if self.task_sampling != "uniform_over_tasks":
                raise ValueError(
                    "va_uniform_window_sampling=true requires "
                    "task_sampling=uniform_over_tasks"
                )
            if self.sampling_mode != "window_index":
                raise ValueError(
                    "va_uniform_window_sampling=true requires "
                    "sampling_mode=window_index"
                )
            if self.sampling_global_batch_size <= 0:
                raise ValueError("sampling_global_batch_size must be positive")
        selected_tasks = tuple(tasks or (SINGLE_ARM_TASKS + BIMANUAL_TASKS))
        unknown = set(selected_tasks) - set(SINGLE_ARM_TASKS + BIMANUAL_TASKS)
        if unknown:
            raise ValueError(f"Unknown DexJoCo tasks: {sorted(unknown)}")
        self.task_names = selected_tasks
        records: list[RawEpisode] = []
        for task in selected_tasks:
            for item in manifest["tasks"][task]:
                tactile_path = Path(item["tfa_h5"])
                if not tactile_path.is_absolute():
                    tactile_path = self.root / tactile_path
                if not tactile_path.is_file():
                    raise FileNotFoundError(tactile_path)
                source_root = Path(item["raw_episode_path"])
                if not source_root.is_absolute():
                    source_root = self.root / source_root
                if not source_root.is_dir():
                    raise FileNotFoundError(source_root)
                records.append(RawEpisode(
                    task, int(item["episode_index"]), source_root,
                    tactile_path, int(item["crop_start"]), int(item["length"]),
                    "single" if task in SINGLE_ARM_TASKS else "bimanual",
                    DEFAULT_TASK_PROMPTS[task],
                ))
        self.records = tuple(records)
        self.records_by_task = {
            task: tuple(record for record in records if record.task_name == task)
            for task in selected_tasks
        }
        self.tasks_by_embodiment = {
            "single": tuple(t for t in selected_tasks if t in SINGLE_ARM_TASKS),
            "bimanual": tuple(t for t in selected_tasks if t in BIMANUAL_TASKS),
        }
        self._task_window_records: dict[str, tuple[RawEpisode, ...]] = {}
        self._task_window_ends: dict[str, tuple[int, ...]] = {}
        for task in selected_tasks:
            task_records = tuple(self.records_by_task[task])
            running = 0
            ends: list[int] = []
            for record in task_records:
                running += max(1, int(record.length - self.action_count * self.action_stride))
                ends.append(running)
            self._task_window_records[task] = task_records
            self._task_window_ends[task] = tuple(ends)
        self._va_uniform_windows_per_task: int | None = None
        if self.va_uniform_window_sampling:
            task_count = len(self.task_names)
            quota_multiple = self.sampling_global_batch_size // math.gcd(
                self.sampling_global_batch_size, task_count
            )
            minimum_windows = min(
                self._task_window_ends[task][-1] for task in self.task_names
            )
            quota = minimum_windows // quota_multiple * quota_multiple
            if quota <= 0:
                raise ValueError(
                    "Not enough task windows to form one globally balanced batch: "
                    f"minimum_windows={minimum_windows}, tasks={task_count}, "
                    f"global_batch={self.sampling_global_batch_size}"
                )
            self._va_uniform_windows_per_task = quota
        self._window_ends: tuple[int, ...] = tuple(
            sum(max(1, int(record.length - self.action_count * self.action_stride)) for record in self.records[:i + 1])
            for i in range(len(self.records))
        )
        self._global_homogeneous_batch_size: int | None = None
        self._homogeneous_embodiment_cycle: tuple[str, ...] = ()
        self._sampling_epoch = 0
        # Persistent DataLoader workers retain their Dataset copy.  A shared
        # integer makes main-process set_epoch() visible without respawning the
        # workers, preserving the epoch-specific window permutation.
        self._shared_sampling_epoch = multiprocessing.Value("q", 0)
        self.normalizer, stats = load_quantile_normalizer44(normalization_stats_path)
        if set(selected_tasks) - set(stats.get("tasks", [])):
            raise ValueError("Normalization statistics do not cover every selected task")
        self.task_t5_cache = TaskT5Cache(t5_cache_path) if Path(t5_cache_path).is_dir() else None
        self.t5_cache = None if self.task_t5_cache is not None else DiskT5Cache(t5_cache_path)
        if self.task_t5_cache is not None and self.preload_task_t5_embeddings:
            for task in selected_tasks:
                self.task_t5_cache.embedding(task, DEFAULT_TASK_PROMPTS[task])

    def __len__(self) -> int:
        if self.va_uniform_window_sampling and not self.val:
            assert self._va_uniform_windows_per_task is not None
            return len(self.task_names) * self._va_uniform_windows_per_task
        if self.sampling_mode == "raw_episode_uniform":
            return len(self.records) * self.samples_per_episode
        return sum(
            max(1, record.length - self.action_count * self.action_stride)
            for record in self.records
        )

    @staticmethod
    def _rng_for_index(index: int, epoch: int = 0) -> random.Random:
        # Include epoch so each pass draws new clips/anchors while remaining
        # reproducible for a given worker and epoch.
        seed = (int(index) * 6364136223846793005 + int(epoch) * 1442695040888963407 + 1442695040888963407) & ((1 << 63) - 1)
        return random.Random(seed)

    def configure_global_embodiment_homogeneous_sampling(self, *, per_device_batch_size: int, world_size: int) -> None:
        global_batch = int(per_device_batch_size) * int(world_size)
        if global_batch <= 0:
            raise ValueError("global homogeneous batch size must be positive")
        available = tuple(
            name for name in ("single", "bimanual")
            if self.tasks_by_embodiment[name]
        )
        if not available:
            raise ValueError("No tasks available for embodiment-homogeneous sampling")
        # Keep the 11-task VA schedule when both embodiments are present, but
        # collapse it to the available embodiment for single-arm-only VAT.
        self._homogeneous_embodiment_cycle = available
        self._global_homogeneous_batch_size = global_batch

    configure_global_task_homogeneous_sampling = configure_global_embodiment_homogeneous_sampling

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        self._sampling_epoch = epoch
        with self._shared_sampling_epoch.get_lock():
            self._shared_sampling_epoch.value = epoch

    def _current_sampling_epoch(self) -> int:
        shared_epoch = getattr(self, "_shared_sampling_epoch", None)
        if shared_epoch is not None:
            return int(shared_epoch.value)
        return int(self._sampling_epoch)

    @staticmethod
    def _mix_uint64(value: int) -> int:
        """Stable SplitMix64 finalizer for epoch/task-specific permutations."""
        mask = (1 << 64) - 1
        value = (int(value) + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    def _resolve_va_uniform_window(self, index: int) -> tuple[RawEpisode, int]:
        """Map a balanced logical index to an epoch-specific unique window.

        Every task contributes the same quota.  The affine permutation is
        injective over each task's complete window pool, so the selected
        subset contains no repeated physical window within an epoch.
        """
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        task_position = int(index) % len(self.task_names)
        task = self.task_names[task_position]
        slot = int(index) // len(self.task_names)
        window_count = self._task_window_ends[task][-1]
        seed = self._mix_uint64(
            self.sampling_seed
            ^ (self._current_sampling_epoch() * 0xD1342543DE82EF95)
            ^ (task_position * 0xA24BAED4963EE407)
        )
        offset = seed % window_count
        if window_count == 1:
            multiplier = 1
        else:
            multiplier = ((seed >> 32) % window_count) or 1
            while math.gcd(multiplier, window_count) != 1:
                multiplier += 1
                if multiplier >= window_count:
                    multiplier = 1
        local_index = (multiplier * slot + offset) % window_count
        ends = self._task_window_ends[task]
        record_index = bisect_right(ends, local_index)
        previous = ends[record_index - 1] if record_index else 0
        return self._task_window_records[task][record_index], local_index - previous

    def _record(self, index: int) -> RawEpisode:
        if self.val:
            return self.records[(index // self.samples_per_episode) % len(self.records)]
        if self.sampling_mode == "raw_episode_uniform":
            rng = self._rng_for_index(index, self._current_sampling_epoch())
            if self._global_homogeneous_batch_size:
                cycle = (
                    ("single", "bimanual", "single", "bimanual", "single",
                     "bimanual", "single", "bimanual", "single", "bimanual",
                     "single")
                    if set(self._homogeneous_embodiment_cycle) == {"single", "bimanual"}
                    else self._homogeneous_embodiment_cycle
                )
                embodiment = cycle[index // self._global_homogeneous_batch_size % len(cycle)]
                tasks = self.tasks_by_embodiment[embodiment]
                if not tasks:
                    raise IndexError(f"No tasks available for embodiment {embodiment!r}")
                task = rng.choice(tasks)
                return rng.choice(self.records_by_task[task])
            if self.task_sampling == "uniform_over_tasks":
                task = rng.choice(self.task_names)
                return rng.choice(self.records_by_task[task])
            return rng.choice(self.records)
        if self._global_homogeneous_batch_size:
            cycle = (
                ("single", "bimanual", "single", "bimanual", "single",
                 "bimanual", "single", "bimanual", "single", "bimanual",
                 "single")
                if set(self._homogeneous_embodiment_cycle) == {"single", "bimanual"}
                else self._homogeneous_embodiment_cycle
            )
            embodiment = cycle[index // self._global_homogeneous_batch_size % len(cycle)]
            tasks = self.tasks_by_embodiment[embodiment]
            task = tasks[index % len(tasks)]
            return self._task_window_records[task][0]
        if self.task_sampling == "uniform_over_tasks":
            task = self.task_names[index % len(self.task_names)]
            task_records = self.records_by_task[task]
            return task_records[(index // len(self.task_names)) % len(task_records)]
        return self.records[index % len(self.records)]

    def _resolve_window(self, index: int) -> tuple[RawEpisode, int]:
        """Resolve a deterministic task/window index like the Stage2 loader."""
        if self.va_uniform_window_sampling and not self.val:
            return self._resolve_va_uniform_window(index)
        if self._global_homogeneous_batch_size:
            cycle = (
                ("single", "bimanual", "single", "bimanual", "single",
                 "bimanual", "single", "bimanual", "single", "bimanual",
                 "single")
                if set(self._homogeneous_embodiment_cycle) == {"single", "bimanual"}
                else self._homogeneous_embodiment_cycle
            )
            embodiment = cycle[(index // self._global_homogeneous_batch_size) % len(cycle)]
            tasks = self.tasks_by_embodiment[embodiment]
            task = tasks[index % len(tasks)]
            local_index = (index // len(tasks)) % self._task_window_ends[task][-1]
        elif self.task_sampling == "uniform_over_tasks":
            task = self.task_names[index % len(self.task_names)]
            local_index = (index // len(self.task_names)) % self._task_window_ends[task][-1]
        else:
            record_index = bisect_right(self._window_ends, index)
            previous = self._window_ends[record_index - 1] if record_index else 0
            record = self.records[record_index]
            return record, index - previous
        ends = self._task_window_ends[task]
        record_index = bisect_right(ends, local_index)
        previous = ends[record_index - 1] if record_index else 0
        return self._task_window_records[task][record_index], local_index - previous

    def _zarr_group(self, record: RawEpisode) -> Any:
        if zarr is None:
            raise ModuleNotFoundError(
                "zarr is required to load DexJoCo raw training samples"
            )
        path = str(record.source_root / "replay.zarr")
        if self.zarr_group_cache_size <= 0:
            return zarr.open_group(path, mode="r")
        group = self._zarr_group_cache.pop(path, None)
        if group is None:
            group = zarr.open_group(path, mode="r")
        self._zarr_group_cache[path] = group
        while len(self._zarr_group_cache) > self.zarr_group_cache_size:
            self._zarr_group_cache.popitem(last=False)
        return group

    def _zarr_array(self, record: RawEpisode, key: str) -> np.ndarray:
        return self._zarr_group(record)[f"data/{key}"]

    def _video_directory(self, record: RawEpisode) -> Path:
        """Return the video root for one episode.

        The VAT-v1 contract always resolves to the original demonstration.
        Additive protocol-specific loaders may override this hook without
        changing the legacy dataset's default behavior.
        """
        return record.source_root / "videos"

    def _frames(self, record: RawEpisode, raw_indices: Sequence[int]) -> list[torch.Tensor]:
        videos = self._video_directory(record)
        if record.source_embodiment == "single":
            # click_mouse was recorded with ego_right as its third-person
            # stream; all other single-arm tasks use front.mp4. Keep the
            # model's packed top view aligned with the actual raw camera.
            third_name = "ego_right.mp4" if record.task_name == "click_mouse" else "front.mp4"
            front_path, right_path, left_path = videos / third_name, videos / "wrist.mp4", None
        else:
            front_path, right_path, left_path = videos / "ego.mp4", videos / "wrist_right.mp4", videos / "wrist_left.mp4"
        front = DexJoCoWTAMDataset._decode_video_frames(
            front_path, raw_indices, sequential=self.sequential_video_decode
        )
        right = DexJoCoWTAMDataset._decode_video_frames(
            right_path, raw_indices, sequential=self.sequential_video_decode
        )
        left = {} if left_path is None else DexJoCoWTAMDataset._decode_video_frames(
            left_path, raw_indices, sequential=self.sequential_video_decode
        )
        result = []
        for index in raw_indices:
            third = front[int(index)]
            h, w = third.shape[:2]
            left_image = np.zeros_like(third) if left_path is None else left[int(index)]
            wrists = np.concatenate((
                cv2.resize(left_image, (w // 2, h // 2)),
                cv2.resize(right[int(index)], (w // 2, h // 2)),
            ), axis=1)
            packed = np.concatenate((third, wrists), axis=0)
            packed = DexJoCoWTAMDataset._resize_with_padding(packed, self.video_size)
            result.append(torch.from_numpy(np.ascontiguousarray(packed)).permute(2, 0, 1).float().div_(255))
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self.val and self.sampling_mode == "window_index":
            record, anchor = self._resolve_window(index)
        else:
            record = self._record(index)
            anchor = index % max(1, record.length - self.action_count * self.action_stride) if self.val else self._rng_for_index(index, self._current_sampling_epoch()).randrange(record.length - self.action_count * self.action_stride)
        horizon = self.action_count * self.action_stride
        anchor_count = record.length - horizon
        if anchor_count <= 0:
            raise ValueError(f"Episode is shorter than +48: {record.tactile_path}")
        action_indices = np.arange(anchor + 3, anchor + 49, 3, dtype=np.int64)
        video_indices = np.arange(anchor + 6, anchor + 49, 6, dtype=np.int64)
        raw_anchor = record.crop_start + anchor
        raw_action_indices = record.crop_start + action_indices
        raw_video_indices = record.crop_start + video_indices
        raw_action = np.asarray(self._zarr_array(record, "action_rotvec").oindex[raw_action_indices], dtype=np.float32)
        # Raw task observations are stored as [1,D] and append task-specific
        # object fields after the official 23/46-D proprio prefix.
        raw_state = np.asarray(
            self._zarr_array(record, "state")[raw_anchor], dtype=np.float32
        ).reshape(-1)
        bimanual = record.source_embodiment == "bimanual"
        state = official_state_to_rotvec_proprio(raw_state, bimanual=bimanual)
        if not bimanual:
            raw_action = _pad_right_arm_22_to_44(raw_action)
            state = _pad_right_arm_22_to_44(state)
            valid = torch.cat((torch.ones(22, dtype=torch.bool), torch.zeros(22, dtype=torch.bool)))
        else:
            valid = torch.ones(44, dtype=torch.bool)
        state_tensor = self.normalizer.normalize_state(torch.from_numpy(state.copy()))
        action_tensor = self.normalizer.normalize_action(torch.from_numpy(raw_action.copy()))
        # Padded dimensions describe a missing arm, not a physical zero command.
        state_tensor = torch.where(valid, state_tensor, torch.zeros_like(state_tensor))
        action_tensor = torch.where(valid[None], action_tensor, torch.zeros_like(action_tensor))
        requested_frames = [raw_anchor] if self.condition_only else [
            raw_anchor, *raw_video_indices.tolist()
        ]
        frames = self._frames(record, requested_frames)
        sample: dict[str, Any] = {
            "first_frame": frames[0],
            "video_frames": None if self.condition_only else torch.stack(frames[1:]),
            "initial_state": state_tensor, "action_sequence": action_tensor,
            "action_valid_mask": valid, "language_embedding": (
                self.task_t5_cache.embedding(record.task_name, record.prompt)
                if self.task_t5_cache is not None else self.t5_cache.embedding(record.prompt)
            ),
            "vlm_inputs": None, "task_name": record.task_name,
            "source_embodiment": record.source_embodiment,
            "episode_name": f"episode_{record.episode_index:06d}",
            "condition_frame_idx": anchor, "condition_rgb_frame_idx": anchor,
            "video_indices": video_indices.tolist(),
            "video_source_action_indices": torch.from_numpy(np.r_[anchor, video_indices]),
            "video_alignment_error_steps_50hz": torch.zeros(9, dtype=torch.long),
            "action_indices": action_indices.tolist(),
        }
        if self._include_tactile_v1:
            observed_indices = np.asarray([max(anchor - 1, 0), anchor], dtype=np.int64)
            all_indices = np.r_[observed_indices, action_indices]
            with h5py.File(record.tactile_path, "r") as handle:
                hands = ("right",) if not bimanual else ("right", "left")
                force = np.concatenate([_h5_take_allow_duplicates(handle[f"{hand}/force_canonical"], all_indices, dtype=np.float32) for hand in hands], axis=1)
                masks = np.concatenate([np.asarray(handle[f"{hand}/metadata/taxel_valid"][:], dtype=bool) for hand in hands], axis=0)
                names = tuple(name.decode() if isinstance(name, bytes) else str(name) for hand in hands for name in handle[f"{hand}/metadata/pad_name"][:])
                times = _h5_take_allow_duplicates(handle["frame_time_seconds"], all_indices, dtype=np.float32)
            if not bimanual:
                force = np.pad(force, ((0, 0), (0, 15), (0, 0), (0, 0), (0, 0)))
                masks = np.pad(masks, ((0, 15), (0, 0), (0, 0)))
                names += tuple(f"LEFT_PAD_{name}" for name in RAW_PAD_NAMES)
            sample.update({
                "tactile_observed_source": torch.from_numpy(np.moveaxis(force[:2], 1, 0).copy()),
                "tactile_future_source": torch.from_numpy(np.moveaxis(force[2:], 1, 0).copy()),
                "tactile_grid_mask_source": torch.from_numpy(masks.copy()),
                "tactile_observed_frame_times": torch.from_numpy((times[:2] - times[0]).copy()),
                "tactile_future_query_times": torch.from_numpy((times[2:] - times[0]).copy()),
                "tactile_pad_names": names,
            })
        return sample
