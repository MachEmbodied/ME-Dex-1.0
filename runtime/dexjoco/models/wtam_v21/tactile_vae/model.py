from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import FINGER_IDS, HAND_SIDE_IDS, SEGMENT_IDS


EncoderKind = Literal["one_d", "two_d", "hybrid"]
DecoderMode = Literal["continuous_legacy", "zero_inflated"]


@dataclass(frozen=True)
class TactileAutoencoderConfig:
    encoder: EncoderKind = "hybrid"
    max_regions: int = 12
    domains: int = 3
    observed_frames: int = 2
    future_frames: int = 16
    local_codes: int = 16
    local_hidden: int = 256
    latent_dim: int = 512
    latent_queries: int = 32
    future_slices: int = 4
    attention_heads: int = 8
    resampler_layers: int = 2
    decoder_layers: int = 2
    ffn_dim: int = 2048
    decoder_chunk_size: int = 4096
    variational: bool = False
    hand_side_types: int = 3
    finger_types: int = 6
    segment_types: int = 5
    decoder_mode: DecoderMode = "continuous_legacy"
    support_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.encoder not in ("one_d", "two_d", "hybrid"):
            raise ValueError(f"Unknown tactile encoder {self.encoder}")
        if self.decoder_mode not in ("continuous_legacy", "zero_inflated"):
            raise ValueError(f"Unknown tactile decoder mode {self.decoder_mode}")
        if not 0.0 < self.support_threshold < 1.0:
            raise ValueError("support_threshold must be in (0,1)")
        if self.local_codes != 16 or self.latent_queries != 32:
            raise ValueError("First-round comparison fixes 16 local codes and 32 latent queries")
        if self.future_frames != 16 or self.future_slices != 4:
            raise ValueError("First-round protocol fixes 16 future frames and four slices")
        if self.latent_dim % self.attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if (self.hand_side_types, self.finger_types, self.segment_types) != (3, 6, 5):
            raise ValueError("Semantic vocabularies are fixed at hand=3, finger=6, segment=5")

    @property
    def total_tokens(self) -> int:
        return self.latent_queries * (1 + self.future_slices)


class Residual1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class Residual2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


def _masked_bin_pool(
    values: torch.Tensor, mask: torch.Tensor, bins: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the masked feature path in the encoder's compute dtype.  A float32
    # mask promotes BF16 PPU activations to float32, which later conflicts
    # with the BF16 observed-fusion projection of a frozen codec.
    mask = mask.to(values.dtype)
    pooled_values = F.adaptive_avg_pool1d(values * mask, bins)
    pooled_mask = F.adaptive_avg_pool1d(mask, bins)
    return pooled_values / pooled_mask.clamp_min(1e-6), pooled_mask.squeeze(1) > 0


class LocalEncoder1d(nn.Module):
    def __init__(self, hidden: int, local_codes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 128, 5, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            Residual1d(128),
            nn.Conv1d(128, hidden, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            Residual1d(hidden),
        )
        self.local_codes = local_codes

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, regions, frames, channels, height, width = values.shape
        flat = values.reshape(batch * regions * frames, channels, height * width)
        flat_mask = mask[:, :, None].expand(-1, -1, frames, -1, -1)
        flat_mask = flat_mask.reshape(batch * regions * frames, 1, height * width).to(flat.dtype)
        hidden = self.net(flat)
        hidden_mask = F.max_pool1d(flat_mask, kernel_size=3, stride=2, padding=1)
        pooled, code_valid = _masked_bin_pool(hidden, hidden_mask, self.local_codes)
        encoded = pooled.transpose(1, 2).reshape(
            batch, regions, frames, self.local_codes, -1
        )
        return encoded, code_valid.reshape(batch, regions, frames, self.local_codes)


class LocalEncoder2d(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            Residual2d(128),
            nn.Conv2d(128, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            Residual2d(hidden),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, regions, frames, channels, height, width = values.shape
        flat = values.reshape(batch * regions * frames, channels, height, width)
        flat_mask = mask[:, :, None].expand(-1, -1, frames, -1, -1)
        flat_mask = flat_mask.reshape(batch * regions * frames, 1, height, width).to(flat.dtype)
        hidden = self.net(flat)
        hidden_mask = F.max_pool2d(flat_mask, kernel_size=3, stride=2, padding=1)
        numerator = F.adaptive_avg_pool2d(hidden * hidden_mask, (4, 4))
        denominator = F.adaptive_avg_pool2d(hidden_mask, (4, 4)).clamp_min(1e-6)
        pooled = (numerator / denominator).flatten(2).transpose(1, 2)
        code_valid = F.adaptive_avg_pool2d(hidden_mask, (4, 4)).flatten(1) > 0
        encoded = pooled.reshape(batch, regions, frames, 16, -1)
        return encoded, code_valid.reshape(batch, regions, frames, 16)


class CausalTemporalEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, 3, stride=2)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, stride=2)
        # Normalize channels independently at each time step. GroupNorm over a
        # [N,C,T] tensor would aggregate future timesteps and break causality.
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    @staticmethod
    def _causal(values: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        # Two strided causal convolutions produce overlapping receptive fields
        # ending at frames 4, 8, 12, and 16. Later raw frames never affect an
        # earlier slice; decoder prefix attention accumulates prior slices.
        left = conv.kernel_size[0] - conv.stride[0]
        return conv(F.pad(values, (left, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # [B,R,16,L,C] -> causal 16 -> 8 -> 4
        batch, regions, frames, codes, channels = values.shape
        hidden = values.permute(0, 1, 3, 4, 2).reshape(batch * regions * codes, channels, frames)
        hidden = self._causal(hidden, self.conv1)
        hidden = F.silu(self.norm1(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self._causal(hidden, self.conv2)
        hidden = F.silu(self.norm2(hidden.transpose(1, 2)).transpose(1, 2))
        slices = hidden.shape[-1]
        return hidden.reshape(batch, regions, codes, channels, slices).permute(0, 1, 4, 2, 3)


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn), nn.GELU(), nn.Linear(ffn, hidden))

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        update, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class MaskedSliceResampler(nn.Module):
    def __init__(self, config: TactileAutoencoderConfig):
        super().__init__()
        cfg = config
        self.observed_queries = nn.Parameter(torch.randn(cfg.latent_queries, cfg.latent_dim) * 0.02)
        self.future_queries = nn.Parameter(torch.randn(cfg.latent_queries, cfg.latent_dim) * 0.02)
        self.layers = nn.ModuleList(
            CrossAttentionLayer(cfg.latent_dim, cfg.attention_heads, cfg.ffn_dim)
            for _ in range(cfg.resampler_layers)
        )
        self.future_time = nn.Embedding(cfg.future_slices, cfg.latent_dim)
        self.config = cfg

    def forward(
        self,
        candidates: torch.Tensor,
        valid_candidates: torch.Tensor,
        *,
        future_slice: int | None,
    ) -> torch.Tensor:
        batch, regions, codes, hidden = candidates.shape
        if valid_candidates.shape != (batch, regions, codes):
            raise ValueError(
                f"candidate mask {tuple(valid_candidates.shape)} does not match "
                f"{(batch, regions, codes)}"
            )
        if not valid_candidates.flatten(1).any(dim=1).all():
            raise ValueError("Every sample needs at least one valid tactile region")
        memory = candidates.reshape(batch, regions * codes, hidden)
        padding = ~valid_candidates.reshape(batch, -1)
        query_bank = self.observed_queries if future_slice is None else self.future_queries
        query = query_bank.unsqueeze(0).expand(batch, -1, -1)
        if future_slice is not None:
            index = torch.full((batch,), future_slice, device=query.device, dtype=torch.long)
            query = query + self.future_time(index).unsqueeze(1)
        for layer in self.layers:
            query = layer(query, memory, padding)
        return query


class RegionSemanticEmbedding(nn.Module):
    """Shared compositional identity for sensors across hands and embodiments."""

    def __init__(self, config: TactileAutoencoderConfig):
        super().__init__()
        hidden = config.latent_dim
        self.hand_side = nn.Embedding(config.hand_side_types, hidden, padding_idx=0)
        self.finger = nn.Embedding(config.finger_types, hidden, padding_idx=0)
        self.segment = nn.Embedding(config.segment_types, hidden, padding_idx=0)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        hand_side_id: torch.Tensor,
        finger_id: torch.Tensor,
        segment_id: torch.Tensor,
    ) -> torch.Tensor:
        if hand_side_id.shape != finger_id.shape or hand_side_id.shape != segment_id.shape:
            raise ValueError("Region semantic id tensors must have identical [B,R] shapes")
        features = (
            self.hand_side(hand_side_id)
            + self.finger(finger_id)
            + self.segment(segment_id)
        )
        return self.norm(features)


class CoordinateQueryDecoder(nn.Module):
    def __init__(self, config: TactileAutoencoderConfig):
        super().__init__()
        cfg = config
        self.domain = nn.Embedding(cfg.domains, cfg.latent_dim)
        self.time = nn.Sequential(nn.Linear(1, cfg.latent_dim), nn.SiLU(), nn.Linear(cfg.latent_dim, cfg.latent_dim))
        self.uv = nn.Sequential(nn.Linear(16, cfg.latent_dim), nn.SiLU(), nn.Linear(cfg.latent_dim, cfg.latent_dim))
        self.native_size = nn.Sequential(nn.Linear(2, cfg.latent_dim), nn.SiLU(), nn.Linear(cfg.latent_dim, cfg.latent_dim))
        self.layers = nn.ModuleList(
            CrossAttentionLayer(cfg.latent_dim, cfg.attention_heads, cfg.ffn_dim)
            for _ in range(cfg.decoder_layers)
        )
        if cfg.decoder_mode == "continuous_legacy":
            # Preserve the exact v3 parameter names for strict legacy loading.
            self.output = nn.Sequential(
                nn.LayerNorm(cfg.latent_dim),
                nn.Linear(cfg.latent_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 3),
            )
            self.support_head = None
            self.conditional_force_head = None
        else:
            self.output = None
            self.support_head = nn.Sequential(
                nn.LayerNorm(cfg.latent_dim),
                nn.Linear(cfg.latent_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            )
            self.conditional_force_head = nn.Sequential(
                nn.LayerNorm(cfg.latent_dim),
                nn.Linear(cfg.latent_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 3),
            )
            # Balanced BCE intentionally has no prevalence prior. Domain
            # identity already enters through the decoder query embedding.
            nn.init.zeros_(self.support_head[-1].weight)
            nn.init.zeros_(self.support_head[-1].bias)
        self.config = cfg

    @staticmethod
    def _fourier(uv: torch.Tensor) -> torch.Tensor:
        frequencies = uv.new_tensor([1.0, 2.0, 4.0, 8.0]) * math.pi
        angles = uv.unsqueeze(-1) * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)

    def _queries(
        self,
        domain_id: torch.Tensor,
        frame_times: torch.Tensor,
        grid_mask: torch.Tensor,
        uv: torch.Tensor,
        region_semantic_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, regions, height, width = grid_mask.shape
        if region_semantic_features.shape != (batch, regions, self.config.latent_dim):
            raise ValueError("Region semantic features do not match decoder regions")
        semantics = region_semantic_features[:, :, None, None]
        domain = self.domain(domain_id)[:, None, None, None]
        valid_rows = grid_mask.any(dim=-1).sum(dim=-1).float().clamp_min(1)
        valid_cols = grid_mask.any(dim=-2).sum(dim=-1).float().clamp_min(1)
        size = torch.stack((valid_rows / height, valid_cols / width), dim=-1)
        size = self.native_size(size)[:, :, None, None]
        spatial = self.uv(self._fourier(uv.permute(0, 1, 3, 4, 2)))
        base = semantics + domain + size + spatial
        time = self.time(frame_times[:, :, None])
        return base[:, None] + time[:, :, None, None, None]

    def _decode_group(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        force_outputs = []
        support_outputs = []
        for start in range(0, query.shape[1], self.config.decoder_chunk_size):
            hidden = query[:, start : start + self.config.decoder_chunk_size]
            for layer in self.layers:
                hidden = layer(hidden, memory)
            if self.config.decoder_mode == "continuous_legacy":
                assert self.output is not None
                force_outputs.append(self.output(hidden))
            else:
                assert self.support_head is not None
                assert self.conditional_force_head is not None
                support_outputs.append(self.support_head(hidden))
                force_outputs.append(self.conditional_force_head(hidden))
        force = torch.cat(force_outputs, dim=1)
        if self.config.decoder_mode == "continuous_legacy":
            return force
        return force, torch.cat(support_outputs, dim=1)

    @staticmethod
    def _reshape_force(
        values: torch.Tensor,
        *,
        batch: int,
        frames: int,
        regions: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return values.reshape(batch, frames, regions, height, width, 3).permute(
            0, 2, 1, 5, 3, 4
        )

    @staticmethod
    def _reshape_support(
        values: torch.Tensor,
        *,
        batch: int,
        frames: int,
        regions: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return values.reshape(batch, frames, regions, height, width, 1).permute(
            0, 2, 1, 5, 3, 4
        )

    def forward(
        self,
        observed_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        domain_id: torch.Tensor,
        frame_times: torch.Tensor,
        grid_mask: torch.Tensor,
        uv_coordinates: torch.Tensor,
        region_semantic_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, regions, height, width = grid_mask.shape
        queries = self._queries(
            domain_id, frame_times, grid_mask, uv_coordinates, region_semantic_features
        )
        observed_query = queries[:, :2].reshape(batch, -1, self.config.latent_dim)
        observed_decoded = self._decode_group(observed_query, observed_tokens)
        future_outputs = []
        future_support_outputs = []
        for slice_index in range(self.config.future_slices):
            frame_start = 2 + 4 * slice_index
            frame_stop = frame_start + 4
            query = queries[:, frame_start:frame_stop].reshape(batch, -1, self.config.latent_dim)
            memory = torch.cat((observed_tokens, future_tokens[:, : slice_index + 1].flatten(1, 2)), dim=1)
            decoded = self._decode_group(query, memory)
            if self.config.decoder_mode == "continuous_legacy":
                assert isinstance(decoded, torch.Tensor)
                future_outputs.append(decoded)
            else:
                assert isinstance(decoded, tuple)
                future_outputs.append(decoded[0])
                future_support_outputs.append(decoded[1])
        mask = grid_mask[:, :, None, None]
        if self.config.decoder_mode == "continuous_legacy":
            assert isinstance(observed_decoded, torch.Tensor)
            observed = self._reshape_force(
                observed_decoded,
                batch=batch,
                frames=2,
                regions=regions,
                height=height,
                width=width,
            ) * mask
            future = self._reshape_force(
                torch.cat(future_outputs, dim=1),
                batch=batch,
                frames=self.config.future_frames,
                regions=regions,
                height=height,
                width=width,
            ) * mask
            return {
                "reconstructed_observed": observed,
                "reconstructed_future": future,
            }

        assert isinstance(observed_decoded, tuple)
        conditional_observed = self._reshape_force(
            observed_decoded[0],
            batch=batch,
            frames=2,
            regions=regions,
            height=height,
            width=width,
        ) * mask
        conditional_future = self._reshape_force(
            torch.cat(future_outputs, dim=1),
            batch=batch,
            frames=self.config.future_frames,
            regions=regions,
            height=height,
            width=width,
        ) * mask
        support_mask = grid_mask[:, :, None, None]
        support_logits_observed = self._reshape_support(
            observed_decoded[1],
            batch=batch,
            frames=2,
            regions=regions,
            height=height,
            width=width,
        )
        support_logits_future = self._reshape_support(
            torch.cat(future_support_outputs, dim=1),
            batch=batch,
            frames=self.config.future_frames,
            regions=regions,
            height=height,
            width=width,
        )
        # Invalid/padded taxels are represented as exact zero in every public
        # tensor. Their logits are not consumed because the loss/metrics mask
        # them explicitly.
        support_logits_observed = support_logits_observed * support_mask
        support_logits_future = support_logits_future * support_mask
        scores_observed = support_logits_observed.sigmoid() * support_mask
        scores_future = support_logits_future.sigmoid() * support_mask
        soft_observed = scores_observed * conditional_observed
        soft_future = scores_future * conditional_future
        hard_observed = torch.where(
            (scores_observed >= self.config.support_threshold) & support_mask,
            conditional_observed,
            torch.zeros_like(conditional_observed),
        )
        hard_future = torch.where(
            (scores_future >= self.config.support_threshold) & support_mask,
            conditional_future,
            torch.zeros_like(conditional_future),
        )
        return {
            "conditional_force_observed": conditional_observed,
            "conditional_force_future": conditional_future,
            "support_logits_observed": support_logits_observed,
            "support_logits_future": support_logits_future,
            "support_scores_observed": scores_observed,
            "support_scores_future": scores_future,
            "reconstructed_observed_soft": soft_observed,
            "reconstructed_future_soft": soft_future,
            "reconstructed_observed_hard": hard_observed,
            "reconstructed_future_hard": hard_future,
            # Existing training utilities consume these canonical aliases.
            "reconstructed_observed": soft_observed,
            "reconstructed_future": soft_future,
        }


class TactileAutoencoder(nn.Module):
    def __init__(self, config: TactileAutoencoderConfig):
        super().__init__()
        cfg = config
        self.config = cfg
        self.local_encoder = (
            LocalEncoder1d(cfg.local_hidden, cfg.local_codes)
            if cfg.encoder == "one_d"
            else LocalEncoder2d(cfg.local_hidden)
        )
        self.observed_fusion = nn.Sequential(
            nn.Linear(3 * cfg.local_hidden, 512), nn.SiLU(), nn.Linear(512, cfg.local_hidden)
        )
        self.future_group = (
            nn.Sequential(
                nn.Linear(cfg.local_hidden, 512),
                nn.SiLU(),
                nn.Linear(512, cfg.local_hidden),
            )
            if cfg.encoder == "two_d"
            else None
        )
        self.future_temporal = (
            None if cfg.encoder == "two_d" else CausalTemporalEncoder(cfg.local_hidden)
        )
        self.candidate_projection = nn.Linear(cfg.local_hidden, cfg.latent_dim)
        self.domain_embedding = nn.Embedding(cfg.domains, cfg.latent_dim)
        self.local_embedding = nn.Embedding(cfg.local_codes, cfg.latent_dim)
        self.temporal_role_embedding = nn.Embedding(2, cfg.latent_dim)
        self.candidate_time = nn.Sequential(
            nn.Linear(1, cfg.latent_dim),
            nn.SiLU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.resampler = MaskedSliceResampler(cfg)
        self.future_mu = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.future_logvar = (
            nn.Linear(cfg.latent_dim, cfg.latent_dim) if cfg.variational else None
        )
        self.region_semantics = RegionSemanticEmbedding(cfg)
        self.decoder = CoordinateQueryDecoder(cfg)

    def _encoder_input(
        self, force: torch.Tensor, grid_mask: torch.Tensor, uv: torch.Tensor
    ) -> torch.Tensor:
        frames = force.shape[2]
        mask = grid_mask[:, :, None, None].expand(-1, -1, frames, -1, -1, -1)
        coordinates = uv[:, :, None].expand(-1, -1, frames, -1, -1, -1)
        return torch.cat((force, mask.to(force.dtype), coordinates.to(force.dtype)), dim=3)

    def _decorate(
        self,
        values: torch.Tensor,
        domain_id: torch.Tensor,
        slice_time: torch.Tensor,
        region_semantic_features: torch.Tensor,
        *,
        future: bool,
    ) -> torch.Tensor:
        batch, regions, codes, _ = values.shape
        if region_semantic_features.shape != (batch, regions, self.config.latent_dim):
            raise ValueError("Region semantic features do not match encoder regions")
        local = self.local_embedding(torch.arange(codes, device=values.device))[None, None]
        domain = self.domain_embedding(domain_id)[:, None, None]
        time = self.candidate_time(slice_time[:, None])[:, None, None]
        role = self.temporal_role_embedding.weight[int(future)][None, None, None]
        return (
            self.candidate_projection(values)
            + region_semantic_features[:, :, None]
            + local
            + domain
            + time
            + role
        )

    def encode(
        self,
        tactile_observed: torch.Tensor,
        tactile_future: torch.Tensor,
        region_mask: torch.Tensor,
        grid_mask: torch.Tensor,
        uv_coordinates: torch.Tensor,
        domain_id: torch.Tensor,
        frame_times: torch.Tensor,
        hand_side_id: torch.Tensor,
        finger_id: torch.Tensor,
        segment_id: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        region_semantic_features = self.region_semantics(
            hand_side_id, finger_id, segment_id
        )
        combined = torch.cat((tactile_observed, tactile_future), dim=2)
        local, local_valid = self.local_encoder(
            self._encoder_input(combined, grid_mask, uv_coordinates), grid_mask
        )
        local_valid = local_valid & region_mask[:, :, None, None]
        previous, current = local[:, :, 0], local[:, :, 1]
        observed_local = self.observed_fusion(torch.cat((previous, current, current - previous), dim=-1))
        future_local = local[:, :, 2:]
        if self.config.encoder == "two_d":
            batch, regions, _, codes, hidden = future_local.shape
            future_local = future_local.reshape(batch, regions, 4, 4, codes, hidden).mean(dim=3)
            assert self.future_group is not None
            future_local = self.future_group(future_local)
        else:
            assert self.future_temporal is not None
            future_local = self.future_temporal(future_local)
        observed_tokens = self.resampler(
            self._decorate(
                observed_local,
                domain_id,
                frame_times[:, 1],
                region_semantic_features,
                future=False,
            ),
            local_valid[:, :, 1],
            future_slice=None,
        )
        posterior = []
        for slice_index in range(self.config.future_slices):
            posterior.append(
                self.resampler(
                    self._decorate(
                        future_local[:, :, slice_index],
                        domain_id,
                        frame_times[:, 2 + 4 * slice_index : 6 + 4 * slice_index].mean(dim=1),
                        region_semantic_features,
                        future=True,
                    ),
                    local_valid[:, :, 2 + 4 * slice_index : 6 + 4 * slice_index].any(dim=2),
                    future_slice=slice_index,
                )
            )
        posterior = torch.stack(posterior, dim=1)
        mu = self.future_mu(posterior)
        logvar = (
            self.future_logvar(posterior).clamp(-12.0, 6.0)
            if self.future_logvar is not None
            else torch.zeros_like(mu)
        )
        future_tokens = mu
        if self.config.variational and not deterministic:
            future_tokens = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return {
            "observed_tokens": observed_tokens,
            "future_tokens": future_tokens,
            "future_mu": mu,
            "future_logvar": logvar,
            "tokens": torch.cat((observed_tokens, future_tokens.flatten(1, 2)), dim=1),
            "region_semantic_features": region_semantic_features,
        }

    def forward(self, batch: dict[str, torch.Tensor], *, deterministic: bool = True) -> dict[str, torch.Tensor]:
        encoded = self.encode(
            batch["tactile_observed"],
            batch["tactile_future"],
            batch["region_mask"],
            batch["grid_mask"],
            batch["uv_coordinates"],
            batch["domain_id"],
            batch["frame_times"],
            batch["hand_side_id"],
            batch["finger_id"],
            batch["segment_id"],
            deterministic=deterministic,
        )
        decoded = self.decoder(
            encoded["observed_tokens"],
            encoded["future_tokens"],
            domain_id=batch["domain_id"],
            frame_times=batch["frame_times"],
            grid_mask=batch["grid_mask"],
            uv_coordinates=batch["uv_coordinates"],
            region_semantic_features=encoded["region_semantic_features"],
        )
        return {**encoded, **decoded}

    def schema(self) -> dict[str, Any]:
        config = asdict(self.config)
        version = 2
        if self.config.decoder_mode == "continuous_legacy":
            # Match the frozen v3 schema byte-for-byte after JSON
            # serialization so strict legacy checkpoint loading remains valid.
            config.pop("decoder_mode")
            config.pop("support_threshold")
        else:
            version = 3
        return {
            "name": "universal_tactile_autoencoder",
            "version": version,
            "config": config,
            "token_shape": [self.config.total_tokens, self.config.latent_dim],
            "region_semantic_vocabulary": {
                "hand_side": HAND_SIDE_IDS,
                "finger": FINGER_IDS,
                "segment": SEGMENT_IDS,
            },
            "contact_modules": False,
        }
