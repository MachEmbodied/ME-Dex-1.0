from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


WAN_PREFIX = "video_model.wan_model."
ACTION_PREFIXES = (
    "action_expert.time_embedding.",
    "action_expert.time_projection.",
    "action_expert.blocks.",
)
FP32_PROMOTION_PREFIXES = (
    "action_expert.time_embedding.",
    "action_expert.time_projection.",
)
PROTECTED_ACTION_PREFIXES = (
    "action_expert.input_encoder.",
    "action_expert.decoder.",
)
TASK_CONDITIONING_TOKENS = (
    "task_embedding",
    "task_text_features",
    "task_text_projection",
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_stage2_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        candidates = (
            checkpoint / "mp_rank_00_model_states.pt",
            checkpoint / "pytorch_model" / "mp_rank_00_model_states.pt",
        )
        checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError(
                "Stage 2 checkpoint not found. Tried:\n"
                + "\n".join(f"  - {candidate}" for candidate in candidates)
            )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage 2 checkpoint does not exist: {checkpoint}")
    return checkpoint.resolve()


def _normalize_source_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("module", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping):
        raise TypeError(f"Stage 2 checkpoint state must be a mapping, got {type(state).__name__}")
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            continue
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in normalized:
            raise RuntimeError(f"Duplicate Stage 2 key after module-prefix normalization: {key}")
        normalized[key] = value
    return normalized


def _category(key: str) -> str:
    lower = key.lower()
    if any(token in lower for token in TASK_CONDITIONING_TOKENS):
        return "task_conditioning"
    if key.startswith(WAN_PREFIX):
        return "wan"
    if key.startswith(ACTION_PREFIXES):
        return "action_dit"
    if key.startswith(PROTECTED_ACTION_PREFIXES) or key == "action_expert.registers":
        return "action_io_registers"
    if key.startswith(("vlm_model.", "und_expert.", "und_module.")):
        return "vlm_understanding"
    if key.startswith("video_model.vae."):
        return "vae"
    if key.startswith("video_module.video_model.wan_model."):
        return "wan_alias"
    if key.startswith("action_module.action_expert."):
        return "action_alias"
    return "other"


def _write_report(report: dict[str, Any], report_path: Optional[str | Path]) -> None:
    if report_path is None:
        return
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def load_stage2_warmstart_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    report_path: Optional[str | Path] = None,
    expected_sha256: Optional[str] = None,
    compute_sha256: bool = True,
    include_wan: bool = True,
    include_action: bool = True,
) -> dict[str, Any]:
    """Atomically load the Stage 2 Action-DiT and, optionally, its WAN."""
    source_path = resolve_stage2_checkpoint(checkpoint_path)
    source_sha256 = sha256_file(source_path) if compute_sha256 else None
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source = _normalize_source_state(checkpoint)
    target = model.state_dict()

    required_wan = (
        sorted(key for key in target if key.startswith(WAN_PREFIX))
        if include_wan
        else []
    )
    required_action = (
        sorted(key for key in target if key.startswith(ACTION_PREFIXES))
        if include_action
        else []
    )
    required = required_wan + required_action
    alias_mismatches = []
    for key in required:
        if key.startswith(WAN_PREFIX):
            alias = "video_module.video_model.wan_model." + key[len(WAN_PREFIX) :]
        else:
            alias = "action_module.action_expert." + key[len("action_expert.") :]
        if alias not in target:
            alias_mismatches.append({"key": key, "alias": alias, "reason": "missing"})
        elif target[key].untyped_storage().data_ptr() != target[alias].untyped_storage().data_ptr():
            alias_mismatches.append({"key": key, "alias": alias, "reason": "storage differs"})
    allowed_prefixes = tuple(
        ([WAN_PREFIX] if include_wan else [])
        + (list(ACTION_PREFIXES) if include_action else [])
    )
    source_allowed = sorted(key for key in source if key.startswith(allowed_prefixes))
    missing_required = sorted(set(required) - set(source))
    unexpected_allowed = sorted(set(source_allowed) - set(required))
    shape_mismatches: list[dict[str, Any]] = []
    dtype_mismatches: list[dict[str, Any]] = []
    dtype_promotions: list[dict[str, Any]] = []
    for key in sorted(set(required) & set(source)):
        source_tensor = source[key]
        target_tensor = target[key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
        if (
            source_tensor.dtype == torch.bfloat16
            and target_tensor.dtype == torch.float32
            and key.startswith(FP32_PROMOTION_PREFIXES)
        ):
            dtype_promotions.append(
                {
                    "key": key,
                    "source_dtype": str(source_tensor.dtype),
                    "target_dtype": str(target_tensor.dtype),
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

    category_counts = Counter(_category(key) for key in source)
    task_keys = sorted(key for key in source if _category(key) == "task_conditioning")
    checksum_mismatch = bool(
        expected_sha256
        and source_sha256 is not None
        and source_sha256.lower() != expected_sha256.lower()
    )
    errors = []
    if checksum_mismatch:
        errors.append("source SHA-256 differs from expected_sha256")
    if include_wan and not required_wan:
        errors.append("target model has no canonical WAN keys")
    if include_action and not required_action:
        errors.append("target model has no canonical Action-DiT keys")
    if missing_required:
        errors.append(f"missing required keys: {len(missing_required)}")
    if unexpected_allowed:
        errors.append(f"unexpected keys under allowed prefixes: {len(unexpected_allowed)}")
    if shape_mismatches:
        errors.append(f"shape mismatches: {len(shape_mismatches)}")
    if dtype_mismatches:
        errors.append(f"dtype mismatches: {len(dtype_mismatches)}")
    if task_keys:
        errors.append(f"task-conditioning keys are forbidden: {len(task_keys)}")
    if alias_mismatches:
        errors.append(f"canonical/alias storage mismatches: {len(alias_mismatches)}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "warmstart_mode": (
            "wan_and_action" if include_wan and include_action
            else "wan_only" if include_wan
            else "action_only"
        ),
        "status": "failed" if errors else "validated",
        "checkpoint_path": str(source_path),
        "checkpoint_size_bytes": source_path.stat().st_size,
        "checkpoint_sha256": source_sha256,
        "expected_sha256": expected_sha256,
        "sha256_computed": compute_sha256,
        "checksum_mismatch": checksum_mismatch,
        "source_tensor_count": len(source),
        "source_category_counts": dict(sorted(category_counts.items())),
        "required_wan_key_count": len(required_wan),
        "required_action_dit_key_count": len(required_action),
        "loaded_wan_key_count": 0,
        "loaded_action_dit_key_count": 0,
        "skipped_vlm_understanding_count": category_counts["vlm_understanding"],
        "skipped_task_conditioning_count": category_counts["task_conditioning"],
        "skipped_action_io_registers_count": category_counts["action_io_registers"],
        "skipped_vae_count": category_counts["vae"],
        "skipped_alias_count": category_counts["wan_alias"] + category_counts["action_alias"],
        "missing_required_key_count": len(missing_required),
        "unexpected_allowed_key_count": len(unexpected_allowed),
        "shape_mismatch_count": len(shape_mismatches),
        "dtype_mismatch_count": len(dtype_mismatches),
        "dtype_promotion_count": len(dtype_promotions),
        "missing_required_keys": missing_required,
        "unexpected_allowed_keys": unexpected_allowed,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "dtype_promotions": dtype_promotions,
        "task_conditioning_keys": task_keys,
        "alias_mismatch_count": len(alias_mismatches),
        "alias_mismatches": alias_mismatches,
        "errors": errors,
    }
    if errors:
        _write_report(report, report_path)
        raise RuntimeError("Stage 2 warm-start validation failed: " + "; ".join(errors))

    loadable = {
        key: source[key].to(dtype=target[key].dtype)
        if key.startswith(FP32_PROMOTION_PREFIXES)
        else source[key]
        for key in required
    }
    incompatible = model.load_state_dict(loadable, strict=False)
    loaded_state = model.state_dict()
    post_load_mismatches = []
    for key in required:
        actual = loaded_state[key].detach().reshape(-1)
        expected = source[key].to(dtype=target[key].dtype).reshape(-1)
        indices = (0,) if actual.numel() == 1 else (0, actual.numel() - 1)
        if any(actual[index].item() != expected[index].item() for index in indices):
            post_load_mismatches.append(key)
    if post_load_mismatches:
        report["status"] = "failed"
        report["post_load_mismatch_count"] = len(post_load_mismatches)
        report["post_load_mismatches"] = post_load_mismatches
        report["errors"] = [f"post-load equality failures: {len(post_load_mismatches)}"]
        _write_report(report, report_path)
        raise RuntimeError(report["errors"][0])

    report.update(
        {
            "status": "loaded",
            "loaded_wan_key_count": len(required_wan),
            "loaded_action_dit_key_count": len(required_action),
            "load_state_missing_count": len(incompatible.missing_keys),
            "load_state_unexpected_count": len(incompatible.unexpected_keys),
            "post_load_mismatch_count": 0,
            "post_load_verified_key_count": len(required),
        }
    )
    _write_report(report, report_path)
    return report


def load_stage2_action_only_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    report_path: Optional[str | Path] = None,
    expected_sha256: Optional[str] = None,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Load only the 276 Action-DiT tensors from an official Stage 2 checkpoint."""
    report = load_stage2_warmstart_weights(
        model,
        checkpoint_path,
        report_path=report_path,
        expected_sha256=expected_sha256,
        compute_sha256=compute_sha256,
        include_wan=False,
    )
    if report["loaded_wan_key_count"] != 0:
        raise RuntimeError("Action-only Stage 2 warm-start unexpectedly loaded WAN tensors")
    return report



def load_stage2_wan_only_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    report_path: Optional[str | Path] = None,
    expected_sha256: Optional[str] = None,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Load only WAN tensors and preserve the initialized Action Expert."""
    report = load_stage2_warmstart_weights(
        model,
        checkpoint_path,
        report_path=report_path,
        expected_sha256=expected_sha256,
        compute_sha256=compute_sha256,
        include_wan=True,
        include_action=False,
    )
    if report["loaded_action_dit_key_count"] != 0:
        raise RuntimeError("WAN-only Stage 2 warm-start unexpectedly loaded Action-DiT tensors")
    return report
