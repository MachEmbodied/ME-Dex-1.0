from __future__ import annotations

import json
from pathlib import Path


JOINT_ATTENTION_MODES = frozenset({"full_bidirectional", "wan_full_action_causal"})


def validate_checkpoint_joint_attention_mode(
    checkpoint_path: str | Path,
    runtime_mode: str,
) -> dict:
    """Reject resume across incompatible joint-attention contracts."""
    runtime_mode = str(runtime_mode)
    if runtime_mode not in JOINT_ATTENTION_MODES:
        raise ValueError(f"Unsupported runtime joint_attention_mode: {runtime_mode!r}")

    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.is_file():
        if runtime_mode != "full_bidirectional":
            raise RuntimeError(
                "Causal resume requires checkpoint config.json with explicit "
                "joint_attention_mode=wan_full_action_causal"
            )
        return {"runtime_mode": runtime_mode, "checkpoint_mode": None, "legacy": True}

    with config_path.open("r", encoding="utf-8") as handle:
        checkpoint_config = json.load(handle)
    checkpoint_mode = checkpoint_config.get("joint_attention_mode")
    if checkpoint_mode is None:
        if runtime_mode != "full_bidirectional":
            raise RuntimeError(
                "Checkpoint lacks joint_attention_mode and can only be resumed with "
                "explicit full_bidirectional runtime mode"
            )
        return {"runtime_mode": runtime_mode, "checkpoint_mode": None, "legacy": True}
    if checkpoint_mode not in JOINT_ATTENTION_MODES:
        raise RuntimeError(
            f"Checkpoint has unsupported joint_attention_mode: {checkpoint_mode!r}"
        )
    if checkpoint_mode != runtime_mode:
        raise RuntimeError(
            "Checkpoint/runtime joint_attention_mode mismatch: "
            f"checkpoint={checkpoint_mode!r}, runtime={runtime_mode!r}"
        )
    return {
        "runtime_mode": runtime_mode,
        "checkpoint_mode": checkpoint_mode,
        "legacy": False,
    }
