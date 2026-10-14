"""Validated OpenPI-style quantile normalization for DexJoCo 44-D policy IO."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch


NEAR_CONSTANT_THRESHOLD = 1.0e-4
EPSILON = 1.0e-6
EXPECTED_CONTRACT = "dexjoco_official_rightfirst_v2"


def _tensor(value, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if result.shape != (44,) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [44] vector, got {tuple(result.shape)}")
    return result.contiguous()


def normalize_quantile(value: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    q01 = q01.to(device=value.device, dtype=value.dtype)
    q99 = q99.to(device=value.device, dtype=value.dtype)
    width = q99 - q01
    centered = value - (q99 + q01) / 2
    scaled = (value - q01) / (width + EPSILON) * 2 - 1
    return torch.where(width < NEAR_CONSTANT_THRESHOLD, centered, scaled)


def denormalize_quantile(value: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    q01 = q01.to(device=value.device, dtype=value.dtype)
    q99 = q99.to(device=value.device, dtype=value.dtype)
    width = q99 - q01
    unscaled = (value + 1) / 2 * (width + EPSILON) + q01
    uncentered = value + (q99 + q01) / 2
    return torch.where(width < NEAR_CONSTANT_THRESHOLD, uncentered, unscaled)


@dataclass(frozen=True)
class DexJoCoQuantileNormalizer44:
    state_q01: torch.Tensor
    state_q99: torch.Tensor
    action_q01: torch.Tensor
    action_q99: torch.Tensor

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return normalize_quantile(value, self.state_q01, self.state_q99)

    def normalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return normalize_quantile(value, self.action_q01, self.action_q99)

    def denormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return denormalize_quantile(value, self.action_q01, self.action_q99)


def load_quantile_normalizer44(path: str | Path) -> tuple[DexJoCoQuantileNormalizer44, dict]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("data_contract") != EXPECTED_CONTRACT:
        raise ValueError(f"Incompatible DexJoCo normalization contract: {path}")
    if payload.get("normalization_type") != "quantile":
        raise ValueError(f"Expected quantile normalization: {path}")
    state, action = payload.get("state", {}), payload.get("action", {})
    if state.get("dim") != 44 or action.get("dim") != 44:
        raise ValueError(f"Expected 44-D state/action statistics: {path}")
    result = DexJoCoQuantileNormalizer44(
        _tensor(state.get("q01"), "state.q01"),
        _tensor(state.get("q99"), "state.q99"),
        _tensor(action.get("q01"), "action.q01"),
        _tensor(action.get("q99"), "action.q99"),
    )
    if (result.state_q99 < result.state_q01).any() or (result.action_q99 < result.action_q01).any():
        raise ValueError(f"q99 is below q01: {path}")
    return result, payload
