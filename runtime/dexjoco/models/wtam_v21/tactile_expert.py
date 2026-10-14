"""Fresh tactile expert for V2.1 Full-Joint Video-Action-Tactile MoT."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.dexjoco.wan.modules.model import WanLayerNorm, WanRMSNorm


DEXJOCO_LEFT_SLOT_IDS = (16, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
DEXJOCO_RIGHT_SLOT_IDS = (
    33,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
)
DEXJOCO_HAND2_SLOT_IDS = DEXJOCO_LEFT_SLOT_IDS + DEXJOCO_RIGHT_SLOT_IDS


@contextmanager
def _fork_cpu_seed(seed: int):
    state = torch.get_rng_state()
    torch.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)


def sinusoidal_embedding_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("sin-cos embedding dimension must be even")
    positions = positions.float().reshape(-1, 1)
    omega = torch.arange(dim // 2, device=positions.device, dtype=torch.float32)
    omega = torch.pow(10000.0, -omega / (dim // 2))
    angles = positions * omega.reshape(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


@dataclass(frozen=True)
class UniversalTactileExpertConfig:
    latent_dim: int = 256
    latent_slices: int = 18
    queries_per_slice: int = 12
    condition_slices: int = 2
    hidden_size: int = 512
    ffn_multiplier: int = 4
    num_layers: int = 30
    wan_num_heads: int = 24
    wan_head_dim: int = 128
    norm_eps: float = 1.0e-6
    freq_dim: int = 256
    initialization_seed: int = 42042
    token_packing: str = "none"

    @classmethod
    def from_mapping(
        cls, values: Dict[str, Any] | None, *, num_layers: int
    ) -> "UniversalTactileExpertConfig":
        values = dict(values or {})
        values["num_layers"] = int(num_layers)
        return cls(**values)

    def __post_init__(self) -> None:
        contract = (
            self.latent_dim,
            self.latent_slices,
            self.queries_per_slice,
            self.condition_slices,
        )
        legacy_support_vae = contract == (512, 5, 32, 1)
        # V3 TacAE is frame-wise and does not compress time.  Its downstream
        # expert therefore supports any configured sequence length while
        # keeping the physical token contract fixed: 12 anatomy slots, 256-D,
        # and two observed condition slices.  T is fixed for each model/run so
        # learned slice embeddings still have a deterministic shape.
        anatomy_v3 = (
            self.latent_dim == 256
            and self.queries_per_slice == 12
            and self.condition_slices == 2
            and self.latent_slices > self.condition_slices
        )
        # V4-Uni is also frame-wise.  It keeps all 34 fixed hand slots and
        # predicts the native 48-D codec latent; padding is handled by the
        # explicit frame_token_valid mask, never by compact re-numbering.
        unified_v4 = (
            self.latent_dim == 48
            and self.queries_per_slice == 34
            and self.condition_slices == 2
            and self.latent_slices > self.condition_slices
        )
        if not (legacy_support_vae or anatomy_v3 or unified_v4):
            raise ValueError(
                f"Unsupported tactile latent contract: {contract}"
            )
        if self.token_packing not in {"none", "dexjoco_left_right"}:
            raise ValueError(
                f"Unsupported tactile token packing: {self.token_packing!r}"
            )
        if self.token_packing == "dexjoco_left_right":
            if not unified_v4:
                raise ValueError(
                    "DexJoCo left/right packing requires V4-Uni [B,T,34,48]"
                )
            if self.hidden_size != 1024:
                raise ValueError(
                    "DexJoCo Hand2 uses the reviewed Linear(768,1024) contract"
                )
        elif self.hidden_size != 512:
            raise ValueError("V2.1 tactile expert requires hidden_size=512")

    @property
    def future_slices(self) -> int:
        return self.latent_slices - self.condition_slices

    @property
    def sequence_length(self) -> int:
        return self.latent_slices * self.tokens_per_slice

    @property
    def tokens_per_slice(self) -> int:
        return 2 if self.token_packing == "dexjoco_left_right" else self.queries_per_slice

    @property
    def packed_latent_dim(self) -> int:
        if self.token_packing == "dexjoco_left_right":
            return 16 * self.latent_dim
        return self.latent_dim

    def tokens_per_slice_for(self, query_count: int) -> int:
        query_count = int(query_count)
        if self.token_packing == "dexjoco_left_right":
            if query_count != self.queries_per_slice:
                raise ValueError(
                    "DexJoCo Hand2 requires all 34 canonical V4-Uni slots before packing"
                )
            return 2
        return query_count

    def pack_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.token_packing == "none":
            return latent
        if latent.ndim != 4 or tuple(latent.shape[1:]) != (
            self.latent_slices,
            self.queries_per_slice,
            self.latent_dim,
        ):
            raise ValueError(
                "DexJoCo Hand2 expects latent [B,18,34,48], got "
                f"{tuple(latent.shape)}"
            )
        slot_ids = torch.tensor(
            DEXJOCO_HAND2_SLOT_IDS, device=latent.device, dtype=torch.long
        )
        compact = latent.index_select(2, slot_ids)
        return compact.reshape(latent.shape[0], self.latent_slices, 2, -1)

    def unpack_latent(self, packed: torch.Tensor) -> torch.Tensor:
        if self.token_packing == "none":
            return packed
        expected = (self.latent_slices, 2, self.packed_latent_dim)
        if packed.ndim != 4 or tuple(packed.shape[1:]) != expected:
            raise ValueError(
                f"DexJoCo Hand2 expects packed latent [B,{expected}], got "
                f"{tuple(packed.shape)}"
            )
        compact = packed.reshape(
            packed.shape[0], self.latent_slices, 32, self.latent_dim
        )
        full = packed.new_zeros(
            packed.shape[0],
            self.latent_slices,
            self.queries_per_slice,
            self.latent_dim,
        )
        slot_ids = torch.tensor(
            DEXJOCO_HAND2_SLOT_IDS, device=packed.device, dtype=torch.long
        )
        return full.index_copy(2, slot_ids, compact)

    def pack_validity(self, valid: torch.Tensor) -> torch.Tensor:
        if valid.ndim != 3 or valid.shape[1] != self.latent_slices:
            raise ValueError(
                "Tactile validity must be [B,latent_slices,queries], got "
                f"{tuple(valid.shape)}"
            )
        valid = valid.bool()
        if self.token_packing == "none":
            return valid.reshape(valid.shape[0], -1)
        if valid.shape[2] != self.queries_per_slice:
            raise ValueError(
                "DexJoCo Hand2 validity requires all 34 canonical slots"
            )
        slot_ids = torch.tensor(
            DEXJOCO_HAND2_SLOT_IDS, device=valid.device, dtype=torch.long
        )
        per_slot = valid.index_select(2, slot_ids).reshape(
            valid.shape[0], self.latent_slices, 2, 16
        )
        hand_any = per_slot.any(dim=-1)
        hand_all = per_slot.all(dim=-1)
        if bool((hand_any & ~hand_all).any().item()):
            raise ValueError(
                "DexJoCo Hand2 found a partially present hand; each hand must have "
                "either all 16 slots or no slots"
            )
        return hand_all.reshape(valid.shape[0], -1)

    @property
    def wan_dim(self) -> int:
        return self.wan_num_heads * self.wan_head_dim


class UniversalTactileTokenizer(nn.Module):
    def __init__(self, config: UniversalTactileExpertConfig):
        super().__init__()
        self.config = config
        with _fork_cpu_seed(config.initialization_seed):
            self.input_projection = nn.Linear(
                config.packed_latent_dim, config.hidden_size
            )
            self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
            self.type_embedding = nn.Parameter(
                torch.randn(2, config.hidden_size) * 0.02
            )
            self.slice_embedding = nn.Parameter(
                torch.randn(config.latent_slices, config.hidden_size) * 0.02
            )
            self.query_embedding = nn.Parameter(
                torch.randn(config.tokens_per_slice, config.hidden_size) * 0.02
            )

    def forward(
        self,
        latent: torch.Tensor,
        anatomy_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        if latent.ndim != 4 or latent.shape[1] != cfg.latent_slices or latent.shape[3] != cfg.latent_dim:
            raise ValueError(
                "Expected latent [B,latent_slices,queries,latent_dim], got "
                f"{tuple(latent.shape)}"
            )
        query_count = int(latent.shape[2])
        if cfg.token_packing != "none" and anatomy_indices is not None:
            raise ValueError("DexJoCo Hand2 does not accept compact anatomy indices")
        if anatomy_indices is None:
            if query_count != cfg.queries_per_slice:
                raise ValueError("Compact tactile input requires anatomy_indices")
            anatomy_indices = torch.arange(
                cfg.queries_per_slice, device=latent.device, dtype=torch.long
            )
        anatomy_indices = anatomy_indices.to(device=latent.device, dtype=torch.long)
        if tuple(anatomy_indices.shape) != (query_count,):
            raise ValueError(
                f"Expected {query_count} anatomy indices, got {tuple(anatomy_indices.shape)}"
            )
        packed = cfg.pack_latent(latent)
        projected = self.input_projection(packed)
        tokens = F.layer_norm(
            projected.float(),
            (cfg.hidden_size,),
            self.input_norm.weight.float(),
            self.input_norm.bias.float(),
            self.input_norm.eps,
        ).to(projected.dtype)
        type_ids = torch.cat(
            (
                torch.zeros(cfg.condition_slices, dtype=torch.long, device=latent.device),
                torch.ones(cfg.future_slices, dtype=torch.long, device=latent.device),
            )
        )
        tokens = (
            tokens
            + self.type_embedding[type_ids][None, :, None]
            + self.slice_embedding[None, :, None]
            + (
                self.query_embedding[None, None]
                if cfg.token_packing != "none"
                else self.query_embedding.index_select(0, anatomy_indices)[None, None]
            )
        )
        return tokens.reshape(
            latent.shape[0],
            cfg.latent_slices * cfg.tokens_per_slice_for(query_count),
            cfg.hidden_size,
        )


class TactileExpertBlock(nn.Module):
    def __init__(self, config: UniversalTactileExpertConfig):
        super().__init__()
        self.norm1 = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.norm2 = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.wan_tactile_qkv = nn.Parameter(
            torch.randn(
                3,
                config.wan_num_heads,
                config.hidden_size,
                config.wan_head_dim,
            )
            / (config.hidden_size * config.wan_head_dim) ** 0.5
        )
        self.wan_tactile_norm_q = WanRMSNorm(config.wan_dim, eps=config.norm_eps)
        self.wan_tactile_norm_k = WanRMSNorm(config.wan_dim, eps=config.norm_eps)
        self.wan_tactile_o = nn.Linear(
            config.wan_dim, config.hidden_size, bias=False
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.ffn_multiplier),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_size * config.ffn_multiplier, config.hidden_size),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, config.hidden_size) / config.hidden_size**0.5
        )


class UniversalTactileOutputHead(nn.Module):
    def __init__(self, config: UniversalTactileExpertConfig):
        super().__init__()
        self.config = config
        self.norm = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.modulation = nn.Parameter(
            torch.randn(1, 2, config.hidden_size) / config.hidden_size**0.5
        )
        self.projection = nn.Linear(
            config.hidden_size, config.packed_latent_dim
        )
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self, tokens: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        shift, scale = (
            self.modulation.unsqueeze(0) + time_embedding.unsqueeze(2)
        ).chunk(2, dim=2)
        values = self.projection(
            self.norm(tokens) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        )
        if tokens.shape[1] % cfg.latent_slices:
            raise ValueError(
                f"Tactile token length {tokens.shape[1]} is not divisible by "
                f"{cfg.latent_slices} slices"
            )
        token_count = tokens.shape[1] // cfg.latent_slices
        values = values.reshape(
            tokens.shape[0],
            cfg.latent_slices,
            token_count,
            cfg.packed_latent_dim,
        )
        values = cfg.unpack_latent(values)
        return torch.cat(
            (
                torch.zeros_like(values[:, : cfg.condition_slices]),
                values[:, cfg.condition_slices :],
            ),
            dim=1,
        )


class UniversalTactileExpert(nn.Module):
    def __init__(self, config: UniversalTactileExpertConfig):
        super().__init__()
        self.config = config
        self.tokenizer = UniversalTactileTokenizer(config)
        with _fork_cpu_seed(config.initialization_seed + 2):
            self.time_embedding = nn.Sequential(
                nn.Linear(config.freq_dim, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.time_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size * 6)
            )
            self.blocks = nn.ModuleList(
                [TactileExpertBlock(config) for _ in range(config.num_layers)]
            )
            self.output_head = UniversalTactileOutputHead(config)

    def get_time_embeddings(
        self, timestep: torch.Tensor, *, queries_per_slice: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        query_count = int(queries_per_slice or cfg.queries_per_slice)
        token_count = cfg.tokens_per_slice_for(query_count)
        slice_t = torch.cat(
            (
                torch.zeros(timestep.shape[0], cfg.condition_slices, device=timestep.device),
                timestep.float()[:, None].expand(-1, cfg.future_slices),
            ),
            dim=1,
        )
        token_t = slice_t.repeat_interleave(token_count, dim=1)
        sequence_length = cfg.latent_slices * token_count
        time_features = sinusoidal_embedding_1d(cfg.freq_dim, token_t).reshape(
            timestep.shape[0], sequence_length, cfg.freq_dim
        )
        time_features = time_features.to(self.time_embedding[0].weight.dtype)
        embedded = self.time_embedding(time_features)
        projected = self.time_projection(embedded).reshape(
            timestep.shape[0], sequence_length, 6, cfg.hidden_size
        )
        return embedded, projected

    @staticmethod
    def modulation(block: TactileExpertBlock, projected: torch.Tensor):
        return (block.modulation.unsqueeze(0) + projected).chunk(6, dim=2)

    @staticmethod
    def apply_ffn(tokens: torch.Tensor, block: TactileExpertBlock, modulation):
        values = block.norm2(tokens).float() * (
            1 + modulation[4].squeeze(2)
        ) + modulation[3].squeeze(2)
        return tokens + block.ffn(values).to(tokens.dtype) * modulation[5].squeeze(2)

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "support_aware_universal_tactile_expert",
            "version": 1,
            "config": asdict(self.config),
        }
