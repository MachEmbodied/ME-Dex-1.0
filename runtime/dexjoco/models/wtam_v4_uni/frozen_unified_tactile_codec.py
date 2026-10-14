"""Strict, frozen adapter for the V4-Uni 50k tactile autoencoder.

The external AE remains outside the WTAM module tree, so a WTAM checkpoint
contains only trainable WTAM state.  This adapter pins both the trusted EMA
checkpoint and the external source files and exposes the same ``encode_raw`` /
``encode_condition`` interface used by the older codecs.

DexJoCo inputs are deliberately strict: 32 physical surfaces ordered as
``left[16], right[16]``; within each hand the order is
``PALM_1, TH_PROX, TH_PAD, TH_TIP, FF_*, MF_*, RF_*``.  A single-hand episode
must still have 32 surfaces, with the absent hand represented by
``support=False`` rather than by a different tensor shape.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch


V4_UNI_ARCHITECTURE = "tactile_ae_v4_uni"
V4_UNI_LATENT_SHAPE = (34, 48)
V4_UNI_DEXJOCO_SURFACES = 32
V4_UNI_DEXJOCO_NORMALIZATION = "dexjoco_shared_si"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    """Return a source revision when git metadata is available.

    The reviewed shared-tree snapshot currently has no ``.git`` directory.
    In that case callers must pin the imported source files by SHA256 instead.
    """

    git_dir = root / ".git"
    if not git_dir.exists():
        return None
    git = shutil.which("git")
    if git is None:
        return None
    return subprocess.run(
        (git, "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_source_files(
    root: Path, expected: Mapping[str, str] | None
) -> dict[str, str]:
    if not expected:
        return {}
    actual: dict[str, str] = {}
    for relative, expected_digest in sorted(expected.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe V4-Uni source hash path: {relative!r}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"V4-Uni source hash path escapes root: {relative!r}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned V4-Uni source file: {path}")
        digest = _sha256(path)
        if digest != str(expected_digest).lower():
            raise ValueError(
                f"V4-Uni source SHA mismatch for {relative}: "
                f"{digest} != {expected_digest}"
            )
        actual[str(relative_path)] = digest
    return actual


def _ensure_external_module(root: Path, module_name: str):
    """Import from the selected source tree without accepting module shadowing."""

    existing = sys.modules.get(module_name)
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"Cannot verify already-imported module {module_name}")
        try:
            Path(module_file).resolve().relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"{module_name} was already imported from {module_file}, not {root}; "
                "run one external tactile-codec source tree per process"
            ) from exc
        return existing
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    try:
        module_file.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Imported {module_name} from {module_file}, expected source under {root}"
        ) from exc
    return module


class FrozenUnifiedTactileCodecV4:
    """Immutable V4-Uni EMA encoder using the audited DexJoCo force contract."""

    def __init__(
        self,
        *,
        source_root: str,
        expected_source_revision: str | None,
        expected_source_files_sha256: Mapping[str, str] | None,
        checkpoint_path: str,
        expected_sha256: str,
        expected_step: int,
        normalization_group: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        root = Path(source_root).resolve()
        checkpoint = Path(checkpoint_path).resolve()
        required = (
            root / "manifeel" / "tactile_ae_uni" / "model.py",
            root / "manifeel" / "tactile_ae_uni" / "layout.py",
            root / "manifeel" / "tactile_ae_uni" / "data.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"V4-Uni source tree is incomplete: {missing}")

        source_revision = _git_revision(root)
        if source_revision is not None and expected_source_revision:
            if source_revision != expected_source_revision:
                raise ValueError(
                    "V4-Uni source revision mismatch: "
                    f"{source_revision} != {expected_source_revision}"
                )
        source_hashes = _validate_source_files(root, expected_source_files_sha256)
        if source_revision is None and not source_hashes:
            raise ValueError(
                "V4-Uni source has no git metadata; source_files_sha256 is required"
            )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing V4-Uni checkpoint: {checkpoint}")
        actual_sha = _sha256(checkpoint)
        if actual_sha != str(expected_sha256).lower():
            raise ValueError(
                f"V4-Uni checkpoint SHA mismatch: {actual_sha} != {expected_sha256}"
            )
        if normalization_group != V4_UNI_DEXJOCO_NORMALIZATION:
            raise ValueError(
                "DexJoCo DJ-TFA-2.0-palmar must use checkpoint normalization "
                f"{V4_UNI_DEXJOCO_NORMALIZATION!r}, got {normalization_group!r}"
            )

        model_module = _ensure_external_module(
            root, "manifeel.tactile_ae_uni.model"
        )
        layout_module = _ensure_external_module(
            root, "manifeel.tactile_ae_uni.layout"
        )
        normalizer_module = _ensure_external_module(
            root, "manifeel.tactile_ae_v41.normalizer"
        )
        normalizer_config = _ensure_external_module(
            root, "manifeel.tactile_ae_v41.config"
        )

        # weights_only=False is intentional for this explicitly SHA-pinned,
        # trusted checkpoint.  Never use it for an untrusted artifact.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("architecture") != V4_UNI_ARCHITECTURE:
            raise ValueError(
                f"Unexpected V4-Uni architecture: {state.get('architecture')!r}"
            )
        if int(state.get("step", -1)) != int(expected_step):
            raise ValueError(
                f"Unexpected V4-Uni step: {state.get('step')!r} != {expected_step}"
            )
        normalizations = state.get("config", {}).get("normalizations", {})
        if normalization_group not in normalizations:
            raise ValueError(
                f"Checkpoint has no normalization group {normalization_group!r}"
            )
        weights = state.get("ema", {}).get("shadow")
        if weights is None:
            raise ValueError("V4-Uni checkpoint does not contain EMA shadow weights")

        model = model_module.UnifiedTactileAE()
        incompatible = model.load_state_dict(weights, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict V4-Uni EMA load failed: {incompatible}")
        model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        spec = normalizations[normalization_group]
        if spec["type"] != "positive_normal_si":
            raise ValueError("DexJoCo requires positive_normal_si normalization")
        normalizer = normalizer_module.ForceNormalizerV41(
            normalizer_config.ForceNormalizationV41Config(**spec["parameters"])
        )
        if hasattr(normalizer, "eval"):
            normalizer.eval()
        if hasattr(normalizer, "requires_grad_"):
            normalizer.requires_grad_(False)
        if hasattr(normalizer, "to"):
            normalizer.to(device=device)

        slots = layout_module.surface_slots("dexjoco")
        if tuple(slots.shape) != (V4_UNI_DEXJOCO_SURFACES, 4):
            raise RuntimeError(
                f"V4-Uni DexJoCo layout must be [32,4], got {tuple(slots.shape)}"
            )
        declared = slots.ge(0)
        if int(declared.sum().item()) != V4_UNI_DEXJOCO_SURFACES:
            raise RuntimeError("Each DexJoCo physical pad must declare exactly one token")
        declared_slots = slots[declared]
        expected_missing = {3, 20}
        actual_missing = set(range(34)) - set(int(v) for v in declared_slots.tolist())
        if actual_missing != expected_missing:
            raise RuntimeError(
                f"Unexpected V4-Uni DexJoCo missing slots: {sorted(actual_missing)}"
            )

        self.model = model
        self.normalizer = normalizer
        self.surface_slots = slots.to(device=device, dtype=torch.long)
        self.device = torch.device(device)
        self.dtype = dtype
        self.normalization_group = normalization_group
        self.checkpoint_sha256 = actual_sha
        self.checkpoint_step = int(expected_step)
        self.source_revision = source_revision or expected_source_revision
        self.source_file_sha256 = source_hashes

    def assert_frozen(self) -> None:
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("V4-Uni tactile encoder must remain eval-only and frozen")
        if hasattr(self.normalizer, "parameters") and any(
            p.requires_grad for p in self.normalizer.parameters()
        ):
            raise RuntimeError("V4-Uni normalizer must remain frozen")

    def _autocast(self):
        enabled = self.device.type == "cuda" and self.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        return (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if enabled
            else nullcontext()
        )

    @staticmethod
    def _support_tensor(
        grid_mask_source: torch.Tensor | None,
        *,
        batch: int,
        surfaces: int,
        device: torch.device,
    ) -> torch.Tensor:
        if grid_mask_source is None:
            raise ValueError(
                "V4-Uni DexJoCo encoding requires tactile_grid_mask_source"
            )
        support = grid_mask_source.to(device=device, dtype=torch.bool)
        if support.ndim == 4:
            support = support.unsqueeze(2)
        expected = (batch, surfaces, 1, 10, 14)
        if tuple(support.shape) != expected:
            raise ValueError(
                f"V4-Uni support must be [B,32,10,14] or [B,32,1,10,14], "
                f"got {tuple(grid_mask_source.shape)}"
            )
        if not support.flatten(2).any(-1).any(-1).all().item():
            raise ValueError(
                "A wholly missing tactile sample must use an explicit missing-modality path"
            )
        return support

    @torch.no_grad()
    def _encode_sequence(
        self,
        source: torch.Tensor,
        *,
        grid_mask_source: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # WTAM dataset layout: [B,S,T,3,10,14].
        if source.ndim != 6 or tuple(source.shape[-3:]) != (3, 10, 14):
            raise ValueError(
                "V4-Uni tactile source must be [B,32,T,3,10,14], got "
                f"{tuple(source.shape)}"
            )
        batch, surfaces, time = source.shape[:3]
        if surfaces != V4_UNI_DEXJOCO_SURFACES:
            raise ValueError(
                "V4-Uni DexJoCo requires fixed left16+right16 surfaces; "
                f"got {surfaces}"
            )
        if time < 1:
            raise ValueError("V4-Uni tactile sequence must contain at least one frame")
        force = source.to(device=self.device)
        if not torch.isfinite(force).all().item():
            raise ValueError("V4-Uni physical force contains NaN or Inf")
        support = self._support_tensor(
            grid_mask_source,
            batch=batch,
            surfaces=surfaces,
            device=self.device,
        )
        # DJ-TFA-2.0-palmar guarantees non-negative compression Fn.  Check
        # supported taxels only: padding values have no physical semantics.
        fn = force[:, :, :, 0].float()
        supported_fn = fn.masked_select(support[:, :, None, 0].expand_as(fn))
        if supported_fn.amin() < -1.0e-6:
            raise ValueError("V4-Uni DexJoCo Fn violates the non-negative contract")
        observed = support.flatten(2).any(-1)
        packed_force = force[observed]
        packed_support = support[observed]
        slots = self.surface_slots[None].expand(batch, -1, -1)[observed]
        owner = torch.arange(batch, device=self.device)[:, None].expand(
            batch, surfaces
        )[observed]
        normalized, _, _ = self.normalizer.normalize(
            packed_force.float(), packed_support[:, None]
        )
        self.model.eval()
        with self._autocast():
            output = self.model(
                normalized.to(device=self.device),
                packed_support,
                owner,
                slots,
                batch,
            )
        tokens = output.get("frame_tokens")
        valid = output.get("frame_token_valid")
        if tuple(tokens.shape) != (batch, time, *V4_UNI_LATENT_SHAPE):
            raise RuntimeError(
                f"V4-Uni latent contract failed: {tuple(tokens.shape)}"
            )
        if tuple(valid.shape) != (batch, time, V4_UNI_LATENT_SHAPE[0]):
            raise RuntimeError(
                f"V4-Uni validity contract failed: {tuple(valid.shape)}"
            )
        valid = valid.to(dtype=torch.bool)
        # Missing slots are padding; supported zero-force pads remain valid.
        if bool(valid[..., 3].any().item()) or bool(valid[..., 20].any().item()):
            raise RuntimeError("V4-Uni declared an unavailable thumb slot valid")
        return tokens.float() * valid[..., None], valid

    @torch.no_grad()
    def encode_raw(
        self,
        observed_source: torch.Tensor,
        future_source: torch.Tensor,
        grid_mask_source: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observed_source.ndim != 6 or future_source.ndim != 6:
            raise ValueError("Observed/future tactile tensors must be rank six")
        if observed_source.shape[:2] != future_source.shape[:2]:
            raise ValueError("Observed/future tactile batch and surface axes differ")
        if observed_source.shape[2] != 2 or future_source.shape[2] < 1:
            raise ValueError(
                "WTAM V4-Uni requires 2 observed and at least 1 future slice"
            )
        source = torch.cat((observed_source, future_source), dim=2)
        return self._encode_sequence(source, grid_mask_source=grid_mask_source)

    @torch.no_grad()
    def encode_condition(
        self,
        observed_source: torch.Tensor,
        grid_mask_source: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observed_source.ndim != 6 or observed_source.shape[2] != 2:
            raise ValueError(
                "WTAM V4-Uni condition must be [B,32,2,3,10,14]"
            )
        return self._encode_sequence(
            observed_source, grid_mask_source=grid_mask_source
        )

    def schema(self) -> dict[str, object]:
        return {
            "architecture": V4_UNI_ARCHITECTURE,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_step": self.checkpoint_step,
            "ema": True,
            "normalization_group": self.normalization_group,
            "input": "[B,32,T,3,10,14] left16_then_right16",
            "latent": "[B,T,34,48]",
            "validity": "True means observed token; slots 3 and 20 unavailable",
            "source_revision": self.source_revision,
            "source_file_sha256": self.source_file_sha256,
        }
