from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


def load_state_dict_strictly(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    source: str,
    expected_num_tensors: Optional[int] = None,
) -> dict[str, Any]:
    """Validate an entire state dict before mutating the target module."""
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint {source} must contain a state-dict mapping")

    target_state = module.state_dict()
    checkpoint_keys = set(state_dict)
    target_keys = set(target_state)
    missing = sorted(target_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - target_keys)
    shape_mismatches = sorted(
        key
        for key in checkpoint_keys & target_keys
        if tuple(state_dict[key].shape) != tuple(target_state[key].shape)
    )
    dtype_mismatches = sorted(
        key
        for key in checkpoint_keys & target_keys
        if state_dict[key].dtype != target_state[key].dtype
    )

    count_mismatch = expected_num_tensors is not None and len(state_dict) != expected_num_tensors
    if count_mismatch or missing or unexpected or shape_mismatches or dtype_mismatches:
        details = {
            "source": source,
            "checkpoint_tensor_count": len(state_dict),
            "expected_tensor_count": expected_num_tensors,
            "target_tensor_count": len(target_state),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "shape_mismatch_count": len(shape_mismatches),
            "dtype_mismatch_count": len(dtype_mismatches),
            "missing_sample": missing[:10],
            "unexpected_sample": unexpected[:10],
            "shape_mismatch_sample": shape_mismatches[:10],
            "dtype_mismatch_sample": dtype_mismatches[:10],
        }
        raise RuntimeError(f"Strict checkpoint validation failed: {details}")

    module.load_state_dict(dict(state_dict), strict=True)
    return {
        "source": source,
        "loaded_tensor_count": len(state_dict),
        "expected_tensor_count": expected_num_tensors,
        "target_tensor_count": len(target_state),
        "missing_count": 0,
        "unexpected_count": 0,
        "shape_mismatch_count": 0,
        "dtype_mismatch_count": 0,
        "strict": True,
    }
