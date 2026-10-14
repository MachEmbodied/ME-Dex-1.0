"""Strict model-only warm-start from a VA checkpoint into a VAT model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch


WAN_PREFIX = "video_model.wan_model."
ACTION_PREFIX = "action_expert."
TACTILE_PREFIXES = ("tactile_expert.", "tactile_codec.")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_va_model_checkpoint(path: str | Path) -> Tuple[Path, Path]:
    """Return ``(checkpoint_root, model_state_file)`` for supported layouts."""
    requested = Path(path).expanduser().resolve()
    if requested.is_dir():
        candidates = (
            requested / "pytorch_model" / "mp_rank_00_model_states.pt",
            requested / "mp_rank_00_model_states.pt",
        )
        model_file = next((candidate for candidate in candidates if candidate.is_file()), None)
        if model_file is None:
            raise FileNotFoundError(
                "VA model checkpoint was not found; tried:\n"
                + "\n".join(f"  - {candidate}" for candidate in candidates)
            )
        return requested, model_file

    if not requested.is_file():
        raise FileNotFoundError(f"VA model checkpoint does not exist: {requested}")
    if requested.parent.name == "pytorch_model":
        return requested.parent.parent, requested
    return requested.parent, requested


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        # Compatibility with older pinned PyTorch builds.
        return torch.load(path, map_location="cpu")


def _load_source_contract(checkpoint_root: Path) -> Dict[str, Any]:
    config_path = checkpoint_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Strict VA->VAT warm-start requires checkpoint config.json: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    architecture = config.get("architecture")
    attention_mode = config.get(
        "joint_attention_mode",
        config.get("model", {}).get("joint_attention_mode"),
    )
    tactile_enabled = bool(config.get("model", {}).get("tactile", {}).get("enabled", False))
    violations = []
    if architecture != "wam_va":
        violations.append(f"architecture={architecture!r}, expected 'wam_va'")
    if attention_mode != "full_bidirectional":
        violations.append(
            f"joint_attention_mode={attention_mode!r}, expected 'full_bidirectional'"
        )
    if tactile_enabled:
        violations.append("source tactile branch must be disabled")
    if violations:
        raise RuntimeError("Incompatible source VA checkpoint: " + "; ".join(violations))
    return {
        "config_path": str(config_path),
        "architecture": architecture,
        "joint_attention_mode": attention_mode,
        "tactile_enabled": tactile_enabled,
        "common": config.get("common", {}),
    }


def _prefix_counts(keys: list[str]) -> Dict[str, int]:
    return {
        "wan": sum(key.startswith(WAN_PREFIX) for key in keys),
        "action_expert": sum(key.startswith(ACTION_PREFIX) for key in keys),
        "tactile": sum(key.startswith(TACTILE_PREFIXES) for key in keys),
        "other": sum(
            not key.startswith((WAN_PREFIX, ACTION_PREFIX, *TACTILE_PREFIXES)) for key in keys
        ),
    }


def load_va_checkpoint_for_vat(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    report_path: Optional[str | Path] = None,
    expected_sha256: Optional[str] = None,
    compute_sha256: bool = False,
    expected_wan_key_count: Optional[int] = None,
    expected_action_key_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Load the complete compatible WAN/Action backbone and leave tactile fresh.

    All validation happens before model mutation. Optimizer, scheduler, RNG,
    dataloader and global-step state are intentionally ignored.
    """
    checkpoint_root, model_file = resolve_va_model_checkpoint(checkpoint_path)
    source_contract = _load_source_contract(checkpoint_root)
    target_state = model.state_dict()

    target_wan_keys = sorted(key for key in target_state if key.startswith(WAN_PREFIX))
    target_action_keys = sorted(key for key in target_state if key.startswith(ACTION_PREFIX))
    target_tactile_keys = sorted(key for key in target_state if key.startswith(TACTILE_PREFIXES))
    if not target_wan_keys or not target_action_keys:
        raise RuntimeError("Target VAT model has no canonical WAN or Action Expert parameters")
    if not target_tactile_keys:
        raise RuntimeError("Target model has no tactile parameters; refusing VA->VAT warm-start")
    if expected_wan_key_count is not None and len(target_wan_keys) != int(expected_wan_key_count):
        raise RuntimeError(
            f"Target WAN key count is {len(target_wan_keys)}, expected {expected_wan_key_count}"
        )
    if expected_action_key_count is not None and len(target_action_keys) != int(
        expected_action_key_count
    ):
        raise RuntimeError(
            "Target Action Expert key count is "
            f"{len(target_action_keys)}, expected {expected_action_key_count}"
        )

    actual_sha256 = None
    if expected_sha256 is not None or compute_sha256:
        actual_sha256 = sha256_file(model_file)
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise RuntimeError(
            f"VA checkpoint SHA-256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )

    checkpoint = _load_checkpoint(model_file)
    source_state = checkpoint.get("module", checkpoint)
    if not isinstance(source_state, Mapping):
        raise TypeError("Checkpoint 'module' entry is not a state-dict mapping")

    required_keys = target_wan_keys + target_action_keys
    missing = sorted(key for key in required_keys if key not in source_state)
    shape_mismatches = []
    dtype_mismatches = []
    for key in required_keys:
        if key not in source_state:
            continue
        source_tensor = source_state[key]
        target_tensor = target_state[key]
        if not torch.is_tensor(source_tensor):
            shape_mismatches.append({"key": key, "source_type": type(source_tensor).__name__})
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
        elif source_tensor.dtype != target_tensor.dtype:
            dtype_mismatches.append(
                {
                    "key": key,
                    "source_dtype": str(source_tensor.dtype),
                    "target_dtype": str(target_tensor.dtype),
                }
            )

    failure_report = {
        "status": "failed",
        "checkpoint_root": str(checkpoint_root),
        "model_state_file": str(model_file),
        "model_state_size_bytes": model_file.stat().st_size,
        "checkpoint_sha256": actual_sha256,
        "source_contract": source_contract,
        "source_key_count": len(source_state),
        "target_key_count": len(target_state),
        "required_wan_key_count": len(target_wan_keys),
        "required_action_key_count": len(target_action_keys),
        "fresh_tactile_key_count": len(target_tactile_keys),
        "missing_required_keys": missing,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "optimizer_loaded": False,
        "scheduler_loaded": False,
        "global_step_loaded": False,
    }
    if missing or shape_mismatches:
        if report_path is not None:
            report_file = Path(report_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(failure_report, indent=2), encoding="utf-8")
        problems = []
        if missing:
            problems.append(f"missing required keys={len(missing)}")
        if shape_mismatches:
            problems.append(f"shape mismatches={len(shape_mismatches)}")
        raise RuntimeError("VA->VAT warm-start validation failed: " + ", ".join(problems))

    loaded = {key: source_state[key] for key in required_keys}
    model.load_state_dict(loaded, strict=False)
    source_keys = list(source_state.keys())
    report = {
        **failure_report,
        "status": "PASS",
        "verdict": "PASS",
        "loaded_key_count": len(loaded),
        "loaded_wan_key_count": len(target_wan_keys),
        "loaded_action_key_count": len(target_action_keys),
        "wan_coverage": 1.0,
        "action_coverage": 1.0,
        "fresh_tactile_keys": target_tactile_keys,
        "source_prefix_counts": _prefix_counts(source_keys),
        "ignored_source_key_count": len(source_state) - len(loaded),
        "fresh_optimizer": True,
        "fresh_scheduler": True,
        "initial_global_step": 0,
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
