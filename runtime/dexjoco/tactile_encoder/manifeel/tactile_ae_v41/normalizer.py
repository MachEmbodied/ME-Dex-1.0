from __future__ import annotations

import torch
import torch.nn as nn

from .config import ForceNormalizationV41Config


class ForceNormalizerV41(nn.Module):
    """Smooth vector deadband followed by per-channel signed-asinh."""

    def __init__(self, config: ForceNormalizationV41Config):
        super().__init__()
        self.config = config
        knee = torch.tensor(config.knee_n, dtype=torch.float32)
        upper = torch.tensor(config.upper_n, dtype=torch.float32)
        self.register_buffer("knee_n", knee, persistent=True)
        self.register_buffer("upper_n", upper, persistent=True)
        self.register_buffer("denominator", torch.asinh(upper / knee), persistent=True)

    @staticmethod
    def _channel_view(values: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        shape = [1] * values.ndim
        shape[-3] = 3
        return channels.reshape(shape)

    def deadband_gate(self, physical_force: torch.Tensor) -> torch.Tensor:
        magnitude = physical_force.float().norm(dim=-3, keepdim=True)
        unit = (magnitude - self.config.deadband_low_n) / (
            self.config.deadband_high_n - self.config.deadband_low_n
        )
        unit = unit.clamp(0.0, 1.0)
        return unit.square() * (3.0 - 2.0 * unit)

    def normalize(
        self, physical_force: torch.Tensor, support_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if physical_force.shape[-3] != 3:
            raise ValueError("physical force channel dimension must be three")
        support = support_mask.to(dtype=physical_force.dtype)
        physical = physical_force * support
        gate = self.deadband_gate(physical)
        cleaned = physical * gate
        knee = self._channel_view(cleaned, self.knee_n)
        denominator = self._channel_view(cleaned, self.denominator)
        normalized = torch.asinh(cleaned / knee) / denominator
        normalized = normalized.clamp(-1.0, 1.0) * support
        # TFA2 normal force uses positive compression.  Tiny negative numerical
        # residues are represented as zero; larger violations belong in audit.
        normalized_normal = normalized.select(-3, 0).clamp_min(0.0)
        normalized = normalized.clone()
        normalized.select(-3, 0).copy_(normalized_normal)
        return normalized, cleaned, gate * support

    def denormalize(
        self, normalized_force: torch.Tensor, support_mask: torch.Tensor
    ) -> torch.Tensor:
        if normalized_force.shape[-3] != 3:
            raise ValueError("normalized force channel dimension must be three")
        support = support_mask.to(dtype=normalized_force.dtype)
        normalized = normalized_force.clamp(-1.0, 1.0) * support
        knee = self._channel_view(normalized, self.knee_n)
        denominator = self._channel_view(normalized, self.denominator)
        physical = knee * torch.sinh(normalized * denominator)
        return physical * support

    def contract(self) -> dict:
        return {
            "type": "per_channel_signed_asinh",
            "channel_order": ["normal", "shear_u", "shear_v"],
            "deadband_low_n": self.config.deadband_low_n,
            "deadband_high_n": self.config.deadband_high_n,
            "knee_n": list(self.config.knee_n),
            "upper_n": list(self.config.upper_n),
            "normal_range": [0.0, 1.0],
            "tangential_range": [-1.0, 1.0],
        }
