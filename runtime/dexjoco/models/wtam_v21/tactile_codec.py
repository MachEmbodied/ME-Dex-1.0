from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import hashlib
import json
import torch

from .tactile_vae.model import TactileAutoencoder, TactileAutoencoderConfig
from .tactile_vae.normalizer import DomainRobustNormalizer
from .tactile_vae.protocol import FINGER_IDS, HAND_SIDE_IDS, SEGMENT_IDS


EXPECTED_CODEC_SCHEMA = "universal_tactile_autoencoder"
EXPECTED_CODEC_VERSION = 3
EXPECTED_CODEC_GIT_SHA = "b2906e084128bb9a2510eb02e9ad0076016612d5"
EXPECTED_DATA_FINGERPRINT = "f4b036ad44695d5356384e57088757dfaa2325a2f4ca1665951f001d95c178a4"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class UniversalTactileTaskConfig:
    name: str
    domain_id: int
    normalization_group: int
    source_regions: int
    slot_mapping: tuple[int, ...]
    region_semantics: tuple[tuple[str, str, str], ...]
    frame_interval_seconds: float | None = None
    inference_frame_interval_seconds: float | None = None
    source_force_frame: str = "canonical_sensor_surface_local"
    channel_order: tuple[str, str, str] = ("normal", "shear_u", "shear_v")
    channel_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "UniversalTactileTaskConfig":
        semantics = []
        for item in values["region_semantics"]:
            semantics.append((str(item["hand_side"]), str(item["finger"]), str(item["segment"])))
        result = cls(
            name=str(values["name"]),
            domain_id=int(values["domain_id"]),
            normalization_group=int(values["normalization_group"]),
            source_regions=int(values["source_regions"]),
            slot_mapping=tuple(int(value) for value in values["slot_mapping"]),
            region_semantics=tuple(semantics),
            frame_interval_seconds=(
                None
                if values.get("frame_interval_seconds") is None
                else float(values["frame_interval_seconds"])
            ),
            inference_frame_interval_seconds=(
                None
                if values.get("inference_frame_interval_seconds") is None
                else float(values["inference_frame_interval_seconds"])
            ),
            source_force_frame=str(
                values.get("source_force_frame", "canonical_sensor_surface_local")
            ),
            channel_order=tuple(values.get("channel_order", ("normal", "shear_u", "shear_v"))),
            channel_signs=tuple(float(value) for value in values.get("channel_signs", (1, 1, 1))),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.source_force_frame != "canonical_sensor_surface_local":
            raise ValueError(
                "The support-aware VAE accepts only canonical sensor-surface forces; "
                f"got {self.source_force_frame!r}"
            )
        if self.channel_order != ("normal", "shear_u", "shear_v"):
            raise ValueError(f"Unexpected tactile channel order: {self.channel_order}")
        if self.channel_signs != (1.0, 1.0, 1.0):
            raise ValueError(f"Unexpected canonical channel signs: {self.channel_signs}")
        if self.source_regions <= 0 or self.source_regions > 12:
            raise ValueError("source_regions must be in [1,12]")
        if len(self.slot_mapping) != self.source_regions:
            raise ValueError("slot_mapping must contain one slot per source region")
        if len(set(self.slot_mapping)) != len(self.slot_mapping):
            raise ValueError("slot_mapping contains duplicate destination slots")
        if any(slot < 0 or slot >= 12 for slot in self.slot_mapping):
            raise ValueError("slot_mapping destination outside the 12-region protocol")
        if len(self.region_semantics) != self.source_regions:
            raise ValueError("region_semantics must contain one descriptor per source region")
        for hand, finger, segment in self.region_semantics:
            if hand not in HAND_SIDE_IDS or finger not in FINGER_IDS or segment not in SEGMENT_IDS:
                raise ValueError(f"Unknown region semantic descriptor: {(hand, finger, segment)}")
        if self.frame_interval_seconds is not None and self.frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be positive")
        if (
            self.inference_frame_interval_seconds is not None
            and self.inference_frame_interval_seconds <= 0
        ):
            raise ValueError("inference_frame_interval_seconds must be positive")


class UniversalLatentNormalizer:
    """Train-split statistics for the observed token slice and four future slices."""

    def __init__(
        self,
        *,
        observed_mean: torch.Tensor,
        observed_std: torch.Tensor,
        future_mean: torch.Tensor,
        future_std: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> None:
        self.observed_mean = torch.as_tensor(observed_mean, dtype=torch.float32).reshape(1, 1, 1, 512)
        self.observed_std = torch.as_tensor(observed_std, dtype=torch.float32).reshape(1, 1, 1, 512)
        self.future_mean = torch.as_tensor(future_mean, dtype=torch.float32).reshape(1, 4, 1, 512)
        self.future_std = torch.as_tensor(future_std, dtype=torch.float32).reshape(1, 4, 1, 512)
        self.metadata = dict(metadata)
        if not all(
            torch.isfinite(value).all()
            for value in (self.observed_mean, self.observed_std, self.future_mean, self.future_std)
        ):
            raise ValueError("Universal tactile latent statistics contain non-finite values")
        if (self.observed_std <= 0).any() or (self.future_std <= 0).any():
            raise ValueError("Universal tactile latent standard deviations must be positive")

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        codec_sha256: str,
        task_name: str,
        task_dataset_fingerprint: str | None = None,
    ) -> "UniversalLatentNormalizer":
        bundle = json.loads(Path(path).read_text())
        if int(bundle.get("version", 0)) != 1:
            raise ValueError("Unsupported universal tactile latent-statistics version")
        if bundle.get("codec_sha256") != codec_sha256:
            raise ValueError("Latent statistics were produced by a different tactile VAE")
        if bundle.get("task_name") != task_name:
            raise ValueError(
                f"Latent statistics task mismatch: {bundle.get('task_name')} != {task_name}"
            )
        if bundle.get("codec_data_fingerprint") != EXPECTED_DATA_FINGERPRINT:
            raise ValueError("Latent statistics use a different tactile codec data contract")
        if (
            task_dataset_fingerprint is not None
            and bundle.get("task_dataset_fingerprint") != task_dataset_fingerprint
        ):
            raise ValueError("Latent statistics use a different task dataset manifest")
        if not bool(bundle.get("complete_train_split", False)):
            raise ValueError("Latent statistics must cover the complete task train split")
        return cls(
            observed_mean=bundle["observed"]["mean"],
            observed_std=bundle["observed"]["std"],
            future_mean=bundle["future"]["mean"],
            future_std=bundle["future"]["std"],
            metadata=bundle,
        )

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if tuple(values.shape[1:]) != (5, 32, 512):
            raise ValueError(f"Expected latent [B,5,32,512], got {tuple(values.shape)}")
        observed = (values[:, :1].float() - self.observed_mean.to(values.device)) / self.observed_std.to(values.device)
        future = (values[:, 1:].float() - self.future_mean.to(values.device)) / self.future_std.to(values.device)
        return torch.cat((observed, future), dim=1)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        if tuple(values.shape[1:]) != (5, 32, 512):
            raise ValueError(f"Expected latent [B,5,32,512], got {tuple(values.shape)}")
        observed = values[:, :1].float() * self.observed_std.to(values.device) + self.observed_mean.to(values.device)
        future = values[:, 1:].float() * self.future_std.to(values.device) + self.future_mean.to(values.device)
        return torch.cat((observed, future), dim=1)

    def state_dict(self) -> Dict[str, Any]:
        return dict(self.metadata)


class FrozenUniversalTactileVAE:
    """Immutable support-aware 1D VAE plus exact physical/latent contracts.

    The module is deliberately stored with ``object.__setattr__`` by the WTAM
    wrapper rather than registered in its state dict. WTAM checkpoints record
    the immutable VAE SHA and statistics metadata instead of duplicating it.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        expected_sha256: str,
        latent_stats_path: str | Path | None,
        expected_latent_stats_sha256: str | None = None,
        expected_task_dataset_fingerprint: str | None = None,
        support_calibration_path: str | Path | None = None,
        expected_support_calibration_sha256: str | None = None,
        task: UniversalTactileTaskConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.latent_stats_path = (
            None if latent_stats_path is None else Path(latent_stats_path).resolve()
        )
        self.task = task
        actual_sha = file_sha256(self.checkpoint_path)
        if actual_sha != expected_sha256:
            raise RuntimeError(f"Tactile VAE SHA mismatch: expected={expected_sha256}, actual={actual_sha}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        schema = checkpoint.get("schema", checkpoint.get("codec_schema"))
        if not isinstance(schema, Mapping):
            raise RuntimeError("Tactile VAE checkpoint is missing codec_schema")
        if schema.get("name") != EXPECTED_CODEC_SCHEMA or int(schema.get("version", 0)) != EXPECTED_CODEC_VERSION:
            raise RuntimeError(f"Unsupported tactile VAE schema: {schema}")
        config = TactileAutoencoderConfig(**dict(schema["config"]))
        required_config = {
            "encoder": "one_d",
            "max_regions": 12,
            "domains": 3,
            "observed_frames": 2,
            "future_frames": 16,
            "latent_dim": 512,
            "latent_queries": 32,
            "future_slices": 4,
            "variational": True,
            "decoder_mode": "zero_inflated",
        }
        actual_config = {key: getattr(config, key) for key in required_config}
        if actual_config != required_config:
            raise RuntimeError(f"Unexpected 20k tactile VAE contract: {actual_config}")
        if float(config.support_threshold) != 0.5:
            raise RuntimeError("The frozen decoder fixed-threshold output must remain exactly 0.5")
        training_git_sha = str(checkpoint.get("git_sha", ""))
        if training_git_sha != EXPECTED_CODEC_GIT_SHA:
            raise RuntimeError(f"Unexpected tactile VAE training commit: {training_git_sha}")
        checkpoint_fingerprint = str(
            checkpoint.get("data_contract_sha256", "")
        )
        if checkpoint_fingerprint != EXPECTED_DATA_FINGERPRINT:
            raise RuntimeError(f"Unexpected tactile VAE data fingerprint: {checkpoint_fingerprint}")
        fork_devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=fork_devices):
            model = TactileAutoencoder(config)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device=device, dtype=dtype).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.physical_normalizer = DomainRobustNormalizer.from_state_dict(checkpoint["normalizer"])
        self.latent_stats_sha256 = (
            None if self.latent_stats_path is None else file_sha256(self.latent_stats_path)
        )
        if (
            expected_latent_stats_sha256 is not None
            and self.latent_stats_sha256 != expected_latent_stats_sha256
        ):
            raise RuntimeError(
                "Latent statistics SHA mismatch: "
                f"expected={expected_latent_stats_sha256}, actual={self.latent_stats_sha256}"
            )
        self.latent_normalizer = (
            None
            if self.latent_stats_path is None
            else UniversalLatentNormalizer.from_json(
                self.latent_stats_path,
                codec_sha256=actual_sha,
                task_name=task.name,
                task_dataset_fingerprint=expected_task_dataset_fingerprint,
            )
        )
        self.support_calibration_path = (
            None if support_calibration_path is None else Path(support_calibration_path).resolve()
        )
        self.support_calibration_sha256 = (
            None
            if self.support_calibration_path is None
            else file_sha256(self.support_calibration_path)
        )
        if (
            expected_support_calibration_sha256 is not None
            and self.support_calibration_sha256 != expected_support_calibration_sha256
        ):
            raise RuntimeError(
                "Support calibration SHA mismatch: "
                f"expected={expected_support_calibration_sha256}, "
                f"actual={self.support_calibration_sha256}"
            )
        self.calibrated_support_threshold = None
        if self.support_calibration_path is not None:
            calibration = json.loads(self.support_calibration_path.read_text(encoding="utf-8"))
            if calibration.get("schema") != "tactile_support_threshold_calibration_v1":
                raise RuntimeError("Unsupported tactile support calibration schema")
            if calibration.get("checkpoint_sha256") != actual_sha:
                raise RuntimeError("Support calibration was produced by a different tactile VAE")
            if calibration.get("data_contract_sha256") != EXPECTED_DATA_FINGERPRINT:
                raise RuntimeError("Support calibration uses a different codec data contract")
            if calibration.get("selection_split") != "validation":
                raise RuntimeError("Support calibration threshold was not selected on validation")
            if not calibration.get("dataset_manifest_combined_sha256"):
                raise RuntimeError("Support calibration is missing its frozen dataset manifest")
            domain = calibration.get("domains", {}).get(str(task.domain_id), {})
            if "threshold" not in domain:
                raise RuntimeError(f"Support calibration has no domain {task.domain_id}")
            threshold = float(domain["threshold"])
            if not 0.0 <= threshold <= 1.0:
                raise RuntimeError(f"Invalid calibrated support threshold: {threshold}")
            self.calibrated_support_threshold = threshold
        self.codec_sha256 = actual_sha
        self.schema = dict(schema)
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _default_uv(device: torch.device) -> torch.Tensor:
        rows = torch.linspace(0.0, 1.0, 10, device=device)
        cols = torch.linspace(0.0, 1.0, 14, device=device)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        return torch.stack((xx, yy), dim=0)

    def canonicalize(
        self,
        tactile_condition: torch.Tensor,
        tactile_target: torch.Tensor,
        *,
        frame_times: torch.Tensor | None = None,
        observed_frame_times: torch.Tensor | None = None,
        future_query_times: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        condition = tactile_condition.to(self.device, torch.float32)
        target = tactile_target.to(self.device, torch.float32)
        if condition.ndim == 5:
            condition = condition.unsqueeze(1)
            target = target.unsqueeze(1)
        expected_condition = (self.task.source_regions, 2, 3, 10, 14)
        expected_target = (self.task.source_regions, 16, 3, 10, 14)
        if tuple(condition.shape[1:]) != expected_condition:
            raise ValueError(f"Expected tactile condition [B,{expected_condition}], got {tuple(condition.shape)}")
        if tuple(target.shape[1:]) != expected_target:
            raise ValueError(f"Expected tactile target [B,{expected_target}], got {tuple(target.shape)}")
        if not torch.isfinite(condition).all() or not torch.isfinite(target).all():
            raise ValueError("Physical tactile batch contains non-finite values")
        batch = condition.shape[0]
        observed = torch.zeros((batch, 12, 2, 3, 10, 14), device=self.device, dtype=torch.float32)
        future = torch.zeros((batch, 12, 16, 3, 10, 14), device=self.device, dtype=torch.float32)
        region_mask = torch.zeros((batch, 12), device=self.device, dtype=torch.bool)
        grid_mask = torch.zeros((batch, 12, 10, 14), device=self.device, dtype=torch.bool)
        uv = torch.zeros((batch, 12, 2, 10, 14), device=self.device, dtype=torch.float32)
        hand = torch.zeros((batch, 12), device=self.device, dtype=torch.long)
        finger = torch.zeros((batch, 12), device=self.device, dtype=torch.long)
        segment = torch.zeros((batch, 12), device=self.device, dtype=torch.long)
        native_uv = self._default_uv(self.device)
        for source, slot in enumerate(self.task.slot_mapping):
            observed[:, slot] = condition[:, source]
            future[:, slot] = target[:, source]
            region_mask[:, slot] = True
            grid_mask[:, slot] = True
            uv[:, slot] = native_uv
            hand_name, finger_name, segment_name = self.task.region_semantics[source]
            hand[:, slot] = HAND_SIDE_IDS[hand_name]
            finger[:, slot] = FINGER_IDS[finger_name]
            segment[:, slot] = SEGMENT_IDS[segment_name]
        times = self.resolve_frame_times(
            batch=batch,
            frame_times=frame_times,
            observed_frame_times=observed_frame_times,
            future_query_times=future_query_times,
        )
        group = torch.full((batch,), self.task.normalization_group, device=self.device, dtype=torch.long)
        domain = torch.full((batch,), self.task.domain_id, device=self.device, dtype=torch.long)
        observed_valid = grid_mask[:, :, None, None]
        future_valid = grid_mask[:, :, None, None]
        return {
            "tactile_observed_physical": observed,
            "tactile_future_physical": future,
            "tactile_observed": self.physical_normalizer.normalize(observed, group, observed_valid),
            "tactile_future": self.physical_normalizer.normalize(future, group, future_valid),
            "region_mask": region_mask,
            "grid_mask": grid_mask,
            "uv_coordinates": uv,
            "domain_id": domain,
            "normalization_group": group,
            "frame_times": times,
            "hand_side_id": hand,
            "finger_id": finger,
            "segment_id": segment,
        }

    def resolve_frame_times(
        self,
        *,
        batch: int,
        frame_times: torch.Tensor | None = None,
        observed_frame_times: torch.Tensor | None = None,
        future_query_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame_times is not None and (
            observed_frame_times is not None or future_query_times is not None
        ):
            raise ValueError("Use full frame_times or observed/future query times, not both")
        if frame_times is not None:
            times = frame_times.to(self.device, torch.float32)
            if times.ndim == 1:
                times = times.unsqueeze(0).expand(batch, -1)
            if tuple(times.shape) != (batch, 18):
                raise ValueError(f"Expected tactile frame_times [B,18], got {tuple(times.shape)}")
            times = times - times[:, :1]
        else:
            interval = (
                self.task.inference_frame_interval_seconds
                if self.task.inference_frame_interval_seconds is not None
                else self.task.frame_interval_seconds
            )
            # Real observed/future timestamps are sufficient.  The frozen
            # interval is only a fallback for callers that omit one of them;
            # requiring it unconditionally made sparse-time replay impossible
            # even when the HDF5 physical clock was explicitly supplied.
            if interval is None and (
                observed_frame_times is None or future_query_times is None
            ):
                raise ValueError(
                    "Inference requires observed/future tactile times or a frozen control interval"
                )
            if observed_frame_times is None:
                observed_times = torch.tensor(
                    [0.0, interval], device=self.device, dtype=torch.float32
                ).unsqueeze(0).expand(batch, -1)
            else:
                observed_times = observed_frame_times.to(self.device, torch.float32)
                if observed_times.ndim == 1:
                    observed_times = observed_times.unsqueeze(0).expand(batch, -1)
                if tuple(observed_times.shape) != (batch, 2):
                    raise ValueError(
                        f"Expected observed tactile times [B,2], got {tuple(observed_times.shape)}"
                    )
            if future_query_times is None:
                offsets = torch.arange(
                    1, 17, device=self.device, dtype=torch.float32
                ).unsqueeze(0) * interval
                future_times = observed_times[:, -1:] + offsets
            else:
                future_times = future_query_times.to(self.device, torch.float32)
                if future_times.ndim == 1:
                    future_times = future_times.unsqueeze(0).expand(batch, -1)
                if tuple(future_times.shape) != (batch, 16):
                    raise ValueError(
                        f"Expected future tactile query times [B,16], got {tuple(future_times.shape)}"
                    )
            times = torch.cat((observed_times, future_times), dim=1)
            times = times - times[:, :1]
        if not torch.isfinite(times).all() or (times[:, 1:] < times[:, :-1]).any():
            raise ValueError("Tactile frame_times must be finite and monotonic")
        return times

    @staticmethod
    def hard_gate(
        conditional_force: torch.Tensor,
        support_scores: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        return torch.where(
            support_scores >= float(threshold),
            conditional_force,
            torch.zeros_like(conditional_force),
        )

    def _encoder_input_dtype(self) -> torch.dtype:
        """Return the actual frozen-VAE encoder input dtype.

        The downstream WTAM may intentionally use a different dtype from the
        independently trained frozen codec.  In particular, PPU model loading
        can retain BF16 codec weights even when the outer Video/Action model
        uses FP32.  Every tensor passed *into* the codec must therefore follow
        the codec's actual first fusion layer; tokens are converted back to
        ``self.dtype`` at the bridge boundary.  We deliberately do not inspect
        the first model parameter: the VAE retains a few scalar FP32
        parameters while its compute layers are BF16 on PPU.
        """
        parameter = next(self.model.observed_fusion.parameters(), None)
        if parameter is None:
            raise RuntimeError("Frozen tactile VAE observed fusion has no parameters")
        return parameter.dtype

    def _decoder_input_dtype(self) -> torch.dtype:
        parameter = next(self.model.decoder.parameters(), None)
        if parameter is None:
            raise RuntimeError("Frozen tactile VAE decoder has no parameters")
        return parameter.dtype

    @torch.no_grad()
    def encode_raw(
        self,
        tactile_condition: torch.Tensor,
        tactile_target: torch.Tensor,
        *,
        frame_times: torch.Tensor | None = None,
        observed_frame_times: torch.Tensor | None = None,
        future_query_times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.assert_frozen()
        canonical = self.canonicalize(
            tactile_condition,
            tactile_target,
            frame_times=frame_times,
            observed_frame_times=observed_frame_times,
            future_query_times=future_query_times,
        )
        model_dtype = self._encoder_input_dtype()
        encoded = self.model.encode(
            canonical["tactile_observed"].to(model_dtype),
            canonical["tactile_future"].to(model_dtype),
            canonical["region_mask"],
            canonical["grid_mask"],
            canonical["uv_coordinates"].to(model_dtype),
            canonical["domain_id"],
            canonical["frame_times"].to(model_dtype),
            canonical["hand_side_id"],
            canonical["finger_id"],
            canonical["segment_id"],
            deterministic=True,
        )
        tokens = torch.cat((encoded["observed_tokens"][:, None], encoded["future_mu"]), dim=1)
        return tokens.to(self.dtype), canonical

    @torch.no_grad()
    def encode(
        self,
        tactile_condition: torch.Tensor,
        tactile_target: torch.Tensor,
        *,
        frame_times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.latent_normalizer is None:
            raise RuntimeError("encode requires task train-split latent statistics")
        tokens, canonical = self.encode_raw(
            tactile_condition, tactile_target, frame_times=frame_times
        )
        return self.latent_normalizer.normalize(tokens).to(self.dtype), canonical

    @torch.no_grad()
    def encode_condition(
        self,
        tactile_condition: torch.Tensor,
        *,
        frame_times: torch.Tensor | None = None,
        observed_frame_times: torch.Tensor | None = None,
        future_query_times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tactile_condition.ndim == 5:
            batch = tactile_condition.shape[0]
            target = tactile_condition.new_zeros((batch, 16, 3, 10, 14))
        elif tactile_condition.ndim == 6:
            batch, regions = tactile_condition.shape[:2]
            target = tactile_condition.new_zeros((batch, regions, 16, 3, 10, 14))
        else:
            raise ValueError(f"Unexpected tactile condition shape: {tuple(tactile_condition.shape)}")
        if self.latent_normalizer is None:
            raise RuntimeError("encode_condition requires task train-split latent statistics")
        tokens, canonical = self.encode_raw(
            tactile_condition,
            target,
            frame_times=frame_times,
            observed_frame_times=observed_frame_times,
            future_query_times=future_query_times,
        )
        latent = self.latent_normalizer.normalize(tokens).to(self.dtype)
        return latent[:, :1], canonical

    @torch.no_grad()
    def decode(
        self,
        normalized_latent: torch.Tensor,
        canonical: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self.assert_frozen()
        if self.latent_normalizer is None:
            raise RuntimeError("decode requires task train-split latent statistics")
        model_dtype = self._decoder_input_dtype()
        latent = self.latent_normalizer.denormalize(normalized_latent).to(model_dtype)
        semantic = self.model.region_semantics(
            canonical["hand_side_id"], canonical["finger_id"], canonical["segment_id"]
        )
        decoded = self.model.decoder(
            latent[:, 0], latent[:, 1:],
            domain_id=canonical["domain_id"],
            frame_times=canonical["frame_times"].to(model_dtype),
            grid_mask=canonical["grid_mask"],
            uv_coordinates=canonical["uv_coordinates"].to(model_dtype),
            region_semantic_features=semantic,
        )
        valid = canonical["grid_mask"][:, :, None, None]
        group = canonical["normalization_group"]
        result = dict(decoded)
        if self.calibrated_support_threshold is None:
            raise RuntimeError("Decoding requires a frozen per-domain support calibration")
        result["reconstructed_future_hard_fixed_0_5"] = self.hard_gate(
            result["conditional_force_future"], result["support_scores_future"], 0.5
        )
        result["reconstructed_future_hard_calibrated"] = self.hard_gate(
            result["conditional_force_future"],
            result["support_scores_future"],
            self.calibrated_support_threshold,
        )
        for key in (
            "conditional_force_observed", "conditional_force_future",
            "reconstructed_observed_soft", "reconstructed_future_soft",
            "reconstructed_observed_hard", "reconstructed_future_hard",
            "reconstructed_observed", "reconstructed_future",
            "reconstructed_future_hard_fixed_0_5",
            "reconstructed_future_hard_calibrated",
        ):
            if key in result:
                result[f"{key}_physical"] = self.physical_normalizer.denormalize(
                    result[key].float(), group, valid
                )
        return result

    def assert_frozen(self) -> None:
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("The external 20k tactile VAE must remain frozen in eval mode")

    def identity_metadata(self) -> Dict[str, Any]:
        return {
            "checkpoint_sha256": self.codec_sha256,
            "codec_schema": self.schema,
            "codec_git_sha": EXPECTED_CODEC_GIT_SHA,
            "codec_data_fingerprint": EXPECTED_DATA_FINGERPRINT,
            "latent_stats_sha256": self.latent_stats_sha256,
            "support_calibration_sha256": self.support_calibration_sha256,
            "calibrated_support_threshold": self.calibrated_support_threshold,
            "task": asdict(self.task),
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            **self.identity_metadata(),
            "checkpoint_path": str(self.checkpoint_path),
            "latent_stats_path": (
                None if self.latent_stats_path is None else str(self.latent_stats_path)
            ),
            "support_calibration_path": (
                None
                if self.support_calibration_path is None
                else str(self.support_calibration_path)
            ),
            "latent_stats": (
                None if self.latent_normalizer is None else self.latent_normalizer.state_dict()
            ),
        }
