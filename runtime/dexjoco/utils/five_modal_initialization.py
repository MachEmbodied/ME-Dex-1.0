from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def build_five_modal_initialization_report(
    *,
    wan_report: Mapping[str, Any],
    task_conditioning_report: Mapping[str, Any],
    action_report: Mapping[str, Any],
    joint_attention_mode: str,
) -> dict[str, Any]:
    checks = {
        "joint_attention_full_bidirectional": joint_attention_mode == "full_bidirectional",
        "wan_loaded_825": wan_report.get("loaded_tensor_count") == 825,
        "wan_expected_825": wan_report.get("expected_tensor_count") == 825,
        "wan_missing_zero": wan_report.get("missing_count") == 0,
        "wan_unexpected_zero": wan_report.get("unexpected_count") == 0,
        "wan_shape_mismatch_zero": wan_report.get("shape_mismatch_count") == 0,
        "wan_dtype_mismatch_zero": wan_report.get("dtype_mismatch_count") == 0,
        "task_conditioning_loaded_2": task_conditioning_report.get("loaded_tensor_count") == 2,
        "task_features_shape": task_conditioning_report.get("features_shape") == [5, 4096],
        "task_projection_shape": task_conditioning_report.get("projection_shape") == [3072, 4096],
        "official_stage2_wan_loaded_zero": action_report.get("loaded_wan_key_count") == 0,
        "action_dit_loaded_276": action_report.get("loaded_action_dit_key_count") == 276,
        "action_missing_zero": action_report.get("missing_required_key_count") == 0,
        "action_unexpected_zero": action_report.get("unexpected_allowed_key_count") == 0,
        "action_shape_mismatch_zero": action_report.get("shape_mismatch_count") == 0,
        "action_dtype_mismatch_zero": action_report.get("dtype_mismatch_count") == 0,
        "action_post_load_mismatch_zero": action_report.get("post_load_mismatch_count") == 0,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": "robotwin_wam_va_five_modal_initialization_v1",
        "status": "validated" if not failed else "failed",
        "joint_attention_mode": joint_attention_mode,
        "checks": checks,
        "failed_checks": failed,
        "wan": dict(wan_report),
        "task_conditioning": dict(task_conditioning_report),
        "official_stage2_action_only": dict(action_report),
    }
    if failed:
        raise RuntimeError(f"Five-modal initialization contract failed: {failed}")
    return report


def write_initialization_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
