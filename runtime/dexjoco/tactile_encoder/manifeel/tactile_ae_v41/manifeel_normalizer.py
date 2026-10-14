from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ManiFeelForceNormalizationV41Config:
    """Zero-preserving normalization for ManiFeel simulator-native force."""

    deadband_low_native: float = 1.0e-5
    deadband_high_native: float = 1.0e-4
    knee_native: tuple[float, float, float] = (1.0e-4, 2.0e-5, 1.0e-5)
    upper_native: tuple[float, float, float] = (3.0e-3, 3.0e-3, 3.5e-3)

    def __post_init__(self) -> None:
        if not 0 <= self.deadband_low_native < self.deadband_high_native:
            raise ValueError("deadband bounds must satisfy 0 <= low < high")
        if len(self.knee_native) != 3 or len(self.upper_native) != 3:
            raise ValueError("normalizer requires three channel parameters")
        if any(value <= 0 for value in (*self.knee_native, *self.upper_native)):
            raise ValueError("knee and upper parameters must be positive")


class ManiFeelForceNormalizerV41(nn.Module):
    """Canonicalize normal compression, smooth tiny values, then signed-asinh."""

    def __init__(self, config: ManiFeelForceNormalizationV41Config):
        super().__init__()
        self.config = config
        knee = torch.tensor(config.knee_native, dtype=torch.float32)
        upper = torch.tensor(config.upper_native, dtype=torch.float32)
        self.register_buffer("knee_native", knee, persistent=True)
        self.register_buffer("upper_native", upper, persistent=True)
        self.register_buffer("denominator", torch.asinh(upper / knee), persistent=True)

    @staticmethod
    def _channel_view(values: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        shape = [1] * values.ndim
        shape[-3] = 3
        return channels.reshape(shape)

    @staticmethod
    def canonicalize(physical_force: torch.Tensor) -> torch.Tensor:
        canonical = physical_force.float().clone()
        # ManiFeel stores signed spring response in channel 0. Compression is
        # non-negative, and |Fn| tracks indentation across all four tasks.
        canonical.select(-3, 0).abs_()
        return canonical

    def deadband_gate(self, canonical_force: torch.Tensor) -> torch.Tensor:
        magnitude = canonical_force.norm(dim=-3, keepdim=True)
        unit = (magnitude - self.config.deadband_low_native) / (
            self.config.deadband_high_native - self.config.deadband_low_native
        )
        unit = unit.clamp(0.0, 1.0)
        return unit.square() * (3.0 - 2.0 * unit)

    def normalize(
        self, physical_force: torch.Tensor, support_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if physical_force.shape[-3] != 3:
            raise ValueError("force channel dimension must be three")
        support = support_mask.to(dtype=physical_force.dtype)
        canonical = self.canonicalize(physical_force) * support
        gate = self.deadband_gate(canonical)
        cleaned = canonical * gate
        knee = self._channel_view(cleaned, self.knee_native)
        denominator = self._channel_view(cleaned, self.denominator)
        normalized = torch.asinh(cleaned / knee) / denominator
        normalized = normalized.clamp(-1.0, 1.0) * support
        return normalized, cleaned, gate * support

    def denormalize(
        self, normalized_force: torch.Tensor, support_mask: torch.Tensor
    ) -> torch.Tensor:
        support = support_mask.to(dtype=normalized_force.dtype)
        normalized = normalized_force.clamp(-1.0, 1.0) * support
        knee = self._channel_view(normalized, self.knee_native)
        denominator = self._channel_view(normalized, self.denominator)
        return knee * torch.sinh(normalized * denominator) * support

    def contract(self) -> dict:
        return {
            "type": "manifeel_compressive_normal_per_channel_signed_asinh",
            "physical_unit": "simulator_native",
            "source_channel_order": ["signed_normal", "shear_u", "shear_v"],
            "canonical_channel_order": ["compressive_normal", "shear_u", "shear_v"],
            "normal_canonicalization": "absolute_value",
            "deadband_low_native": self.config.deadband_low_native,
            "deadband_high_native": self.config.deadband_high_native,
            "knee_native": list(self.config.knee_native),
            "upper_native": list(self.config.upper_native),
            "normal_range": [0.0, 1.0],
            "tangential_range": [-1.0, 1.0],
        }
