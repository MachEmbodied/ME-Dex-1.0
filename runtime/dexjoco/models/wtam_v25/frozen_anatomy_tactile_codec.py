from __future__ import annotations

import hashlib
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    """Resolve HEAD even in minimal inference images without the git binary."""
    git = shutil.which("git")
    if git is not None:
        return subprocess.run(
            (git, "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git_dir = root / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise RuntimeError(f"invalid gitdir marker: {git_dir}")
        git_dir = (root / marker.removeprefix("gitdir: ")).resolve()
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")) or " " not in line:
                continue
            revision, packed_ref = line.split(" ", 1)
            if packed_ref == ref:
                return revision
    raise RuntimeError(f"cannot resolve git HEAD without git executable: {root}")


def _register_numpy_pickle_compatibility() -> None:
    """Allow NumPy-2 checkpoints to load under Isaac's NumPy 1.26 runtime."""
    if int(np.__version__.split(".", 1)[0]) >= 2:
        return
    import numpy.core as core
    import numpy.core._multiarray_umath as multiarray_umath
    import numpy.core.multiarray as multiarray
    import numpy.core.numeric as numeric

    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core._multiarray_umath", multiarray_umath)
    sys.modules.setdefault("numpy._core.multiarray", multiarray)
    sys.modules.setdefault("numpy._core.numeric", numeric)


class FrozenAnatomyTactileCodecV3:
    """Immutable V3 TacAE encoder and the audited RoboTwin physical contract.

    The codec is intentionally not registered as a WTAM submodule.  WTAM
    checkpoints contain only trainable WTAM parameters and record the external
    codec SHA in the run configuration.
    """

    def __init__(
        self,
        *,
        source_root: str,
        expected_source_revision: str,
        checkpoint_path: str,
        expected_sha256: str,
        device: torch.device,
        dtype: torch.dtype,
        dexjoco_channel_scale: tuple[float, float, float] | None = None,
    ) -> None:
        root = Path(source_root).resolve()
        checkpoint = Path(checkpoint_path).resolve()
        if not (root / "manifeel" / "tactile_ae_v3" / "model.py").is_file():
            raise FileNotFoundError(f"V3 TacAE source tree is incomplete: {root}")
        source_revision = _git_revision(root)
        if source_revision != expected_source_revision:
            raise ValueError(
                "V3 TacAE source revision mismatch: "
                f"{source_revision} != {expected_source_revision}"
            )
        actual_sha = _sha256(checkpoint)
        if actual_sha != expected_sha256:
            raise ValueError(
                f"V3 TacAE checkpoint SHA mismatch: {actual_sha} != {expected_sha256}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        model_module = importlib.import_module("manifeel.tactile_ae_v3.model")
        config_module = importlib.import_module("manifeel.tactile_ae_v3.config")
        normalizer_module = importlib.import_module("manifeel.tactile_vae.normalizer")

        _register_numpy_pickle_compatibility()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        schema = state.get("schema", {})
        if schema.get("architecture") != "universal_anatomy_tactile_ae_v3":
            raise ValueError(f"Unexpected tactile codec schema: {schema}")
        if schema.get("latent_frame") != "[B,12,256]" or schema.get("time_compression") is not False:
            raise ValueError("V2.5 requires the non-temporal [B,12,256] V3 codec")
        config = config_module.AnatomyTactileAEV3Config(**schema["config"])
        model = model_module.UniversalAnatomyTactileAEV3(config)
        weights = state.get("ema", {}).get("shadow")
        if weights is None:
            raise ValueError("V3 checkpoint does not contain EMA weights")
        incompatible = model.load_state_dict(weights, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict V3 load failed: {incompatible}")
        model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.model = model
        self.normalizer = normalizer_module.DomainRobustNormalizer.from_state_dict(
            state["normalizer"]
        )
        if dexjoco_channel_scale is not None:
            scale = tuple(float(value) for value in dexjoco_channel_scale)
            if len(scale) != 3 or any(not np.isfinite(value) or value <= 0 for value in scale):
                raise ValueError(f"Invalid DexJoCo Fn/Fu/Fv scale: {scale}")
            previous = self.normalizer.groups[2]
            self.normalizer.groups[2] = type(previous)(
                bias=(0.0, 0.0, 0.0), scale=scale, quantile=previous.quantile
            )
        self.dexjoco_channel_scale = dexjoco_channel_scale
        self.device = device
        self.dtype = dtype
        self.checkpoint_sha256 = actual_sha
        self.source_revision = source_revision

    def assert_frozen(self) -> None:
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("V3 tactile encoder must remain eval-only and frozen")

    @staticmethod
    def _metadata(batch: int, time: int, device: torch.device) -> dict[str, torch.Tensor]:
        region_mask = torch.zeros(batch, time, 30, dtype=torch.bool, device=device)
        region_mask[:, :, :4] = True
        grid_mask = torch.zeros(batch, time, 30, 10, 14, dtype=torch.bool, device=device)
        grid_mask[:, :, :4] = True
        u = torch.linspace(-1.0, 1.0, 14, device=device)
        v = torch.linspace(-1.0, 1.0, 10, device=device)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        uv = torch.stack((uu, vv), dim=0)
        uv_coordinates = torch.zeros(batch, time, 30, 2, 10, 14, device=device)
        uv_coordinates[:, :, :4] = uv
        hand = torch.zeros(batch, time, 30, dtype=torch.long, device=device)
        finger = torch.zeros_like(hand)
        segment = torch.zeros_like(hand)
        hand[:, :, :4] = hand.new_tensor((1, 1, 2, 2))
        finger[:, :, :4] = finger.new_tensor((1, 2, 1, 2))
        segment[:, :, :4] = 1
        return {
            "region_mask": region_mask,
            "grid_mask": grid_mask,
            "uv_coordinates": uv_coordinates,
            "hand_side_id": hand,
            "finger_id": finger,
            "segment_id": segment,
        }

    @staticmethod
    def _dexjoco_metadata(
        grid_mask_source: torch.Tensor,
        *,
        time: int,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Map the audited 15/30-pad DexJoCo registry into 30 canonical slots."""
        if grid_mask_source.ndim != 4 or grid_mask_source.shape[-2:] != (10, 14):
            raise ValueError(
                "DexJoCo tactile_grid_mask_source must be [B,R,10,14], got "
                f"{tuple(grid_mask_source.shape)}"
            )
        batch, regions = grid_mask_source.shape[:2]
        if regions not in (15, 30):
            raise ValueError(f"DexJoCo requires 15 or 30 raw pads, got {regions}")
        canonical_source = torch.full((30,), -1, dtype=torch.long, device=device)
        if regions == 30:
            canonical_source[:15] = torch.arange(15, 30, device=device)
        canonical_source[15:] = torch.arange(0, 15, device=device)
        present = canonical_source.ge(0)
        canonical_grid = torch.zeros(batch, 30, 10, 14, dtype=torch.bool, device=device)
        canonical_grid[:, present] = grid_mask_source.to(device=device, dtype=torch.bool)[:, canonical_source[present]]
        region_mask_frame = canonical_grid.flatten(2).any(dim=-1)

        finger_raw = torch.tensor(
            (6, 1, 5, 6, 2, 2, 4, 2, 3, 3, 4, 1, 3, 4, 6),
            dtype=torch.long,
            device=device,
        )
        segment_raw = torch.tensor(
            (3, 1, 4, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 3, 2),
            dtype=torch.long,
            device=device,
        )
        hand_frame = torch.zeros(batch, 30, dtype=torch.long, device=device)
        finger_frame = torch.zeros_like(hand_frame)
        segment_frame = torch.zeros_like(hand_frame)
        if regions == 30:
            hand_frame[:, :15] = 1
            finger_frame[:, :15] = finger_raw
            segment_frame[:, :15] = segment_raw
        hand_frame[:, 15:] = 2
        finger_frame[:, 15:] = finger_raw
        segment_frame[:, 15:] = segment_raw
        hand_frame = torch.where(region_mask_frame, hand_frame, torch.zeros_like(hand_frame))
        finger_frame = torch.where(region_mask_frame, finger_frame, torch.zeros_like(finger_frame))
        segment_frame = torch.where(region_mask_frame, segment_frame, torch.zeros_like(segment_frame))

        uv_frame = torch.zeros(batch, 30, 2, 10, 14, dtype=torch.float32, device=device)
        for batch_index in range(batch):
            for region_index in range(30):
                rows, cols = torch.where(canonical_grid[batch_index, region_index])
                if rows.numel() == 0:
                    continue
                row_min, row_max = int(rows.min()), int(rows.max())
                col_min, col_max = int(cols.min()), int(cols.max())
                row_den = max(row_max - row_min, 1)
                col_den = max(col_max - col_min, 1)
                uv_frame[batch_index, region_index, 0, rows, cols] = (
                    cols.float() - col_min
                ) / col_den * 2.0 - 1.0
                uv_frame[batch_index, region_index, 1, rows, cols] = (
                    rows.float() - row_min
                ) / row_den * 2.0 - 1.0

        def expand(value: torch.Tensor) -> torch.Tensor:
            return value[:, None].expand(batch, time, *value.shape[1:])

        return (
            {
                "region_mask": expand(region_mask_frame),
                "grid_mask": expand(canonical_grid),
                "uv_coordinates": expand(uv_frame),
                "hand_side_id": expand(hand_frame),
                "finger_id": expand(finger_frame),
                "segment_id": expand(segment_frame),
            },
            canonical_source,
        )

    @torch.no_grad()
    def encode_raw(
        self,
        observed_source: torch.Tensor,
        future_source: torch.Tensor,
        grid_mask_source: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Dataset layout is [B,R,T,3,10,14]; V3 layout is [B,T,R,3,10,14].
        if observed_source.ndim != 6 or future_source.ndim != 6:
            raise ValueError("Tactile source tensors must be [B,R,T,3,10,14]")
        if observed_source.shape[:2] != future_source.shape[:2]:
            raise ValueError("Observed/future tactile batch and region axes differ")
        if observed_source.shape[2] != 2 or future_source.shape[2] < 1:
            raise ValueError(
                "V3 WTAM requires 2 observed and at least 1 future tactile slice"
            )
        force = torch.cat((observed_source, future_source), dim=2).permute(0, 2, 1, 3, 4, 5)
        batch, time, regions = force.shape[:3]
        canonical = torch.zeros(batch, time, 30, 3, 10, 14, device=force.device, dtype=force.dtype)
        if regions == 4:
            metadata = self._metadata(batch, time, force.device)
            canonical[:, :, :4] = force
            group_ids = torch.ones(batch, dtype=torch.long, device=force.device)
        elif regions in (15, 30):
            if grid_mask_source is None:
                raise ValueError("DexJoCo tactile encoding requires tactile_grid_mask_source")
            metadata, canonical_source = self._dexjoco_metadata(
                grid_mask_source, time=time, device=force.device
            )
            present = canonical_source.ge(0)
            canonical[:, :, present] = force[:, :, canonical_source[present]]
            group_ids = torch.full((batch,), 2, dtype=torch.long, device=force.device)
        else:
            raise ValueError(f"Unsupported tactile region count: {tuple(force.shape)}")
        valid = metadata["region_mask"][..., None, None, None] & metadata["grid_mask"][..., None, :, :]
        normalized = self.normalizer.normalize(canonical.float(), group_ids, valid)
        # Retain only geometry metadata for the analysis-only taxel projection;
        # the public encode_raw return contract is unchanged.
        self._last_encoder_grid_mask = metadata["grid_mask"].detach()
        tokens, token_valid = self.model.encode_sequence(
            normalized.to(dtype=self.dtype), **metadata
        )
        # Keep the fixed 12-slot anatomy interface. token_valid is returned for
        # auditing; missing anatomical slots are not mistaken for sensor zeros.
        return tokens, token_valid

    @torch.no_grad()
    def encode_condition(
        self,
        observed_source: torch.Tensor,
        grid_mask_source: torch.Tensor | None = None,
        **_: torch.Tensor,
    ):
        if observed_source.ndim != 6 or observed_source.shape[2] != 2:
            raise ValueError("Observed RoboTwin tactile must be [B,4,2,3,10,14]")
        future = observed_source.new_zeros(
            observed_source.shape[0], observed_source.shape[1], 16, 3, 10, 14
        )
        tokens, valid = self.encode_raw(
            observed_source, future, grid_mask_source=grid_mask_source
        )
        return tokens[:, :2], valid[:, :2]

    @torch.no_grad()
    def encode_raw_with_details(
        self,
        observed_source: torch.Tensor,
        future_source: torch.Tensor,
        grid_mask_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return tokens plus frozen encoder weights for taxel attribution."""
        details_holder: dict[str, dict[str, torch.Tensor]] = {}

        def hook(_module, _inputs, output):
            details_holder["value"] = output

        handle = self.model.encoder.register_forward_hook(hook)
        try:
            tokens, valid = self.encode_raw(
                observed_source,
                future_source,
                grid_mask_source=grid_mask_source,
            )
        finally:
            handle.remove()
        if "value" not in details_holder:
            raise RuntimeError("V3 encoder details hook did not run")
        details = details_holder["value"]
        batch, time = observed_source.shape[0], 2 + future_source.shape[2]
        reshaped: dict[str, torch.Tensor] = {}
        for key in ("region_pooling_weights", "region_anatomy_token_id", "anatomy_token_valid"):
            value = details[key]
            reshaped[key] = value.reshape(batch, time, *value.shape[1:])
        reshaped["surface_token_valid"] = details["surface_token_valid"].reshape(
            batch, time, *details["surface_token_valid"].shape[1:]
        )
        if hasattr(self, "_last_encoder_grid_mask"):
            reshaped["grid_mask"] = self._last_encoder_grid_mask
        return tokens, valid, reshaped

    @torch.no_grad()
    def encode_condition_with_details(
        self,
        observed_source: torch.Tensor,
        grid_mask_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Condition encoding plus frozen geometry for taxel heatmaps."""
        future = observed_source.new_zeros(
            observed_source.shape[0], observed_source.shape[1], 16, 3, 10, 14
        )
        tokens, valid, details = self.encode_raw_with_details(
            observed_source, future, grid_mask_source=grid_mask_source
        )
        return tokens[:, :2], valid[:, :2], {key: value[:, :2] for key, value in details.items()}
