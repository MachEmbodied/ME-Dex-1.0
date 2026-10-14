# Motus - Modular Architecture
# Three-modal UniDiffuser: Video Model (WAN) + Action Expert + Understanding Expert
# Implements MoT (Mixture of Tokens) architecture with unified attention

import os
import sys
import json
import math
import torch
import logging
import torch.nn as nn
from safetensors import safe_open
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple


from runtime.dexjoco.utils.common import get_t_distribution, sample_scheduler_timestep_ids
from runtime.dexjoco.wan.modules.model import sinusoidal_embedding_1d
from runtime.dexjoco.wan.modules.attention import (
    JOINT_ATTENTION_MODES,
    build_wan_full_action_causal_mask,
)
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .und_expert import UndExpert, UndExpertConfig
from .wtam_v21 import (
    FrozenUniversalTactileVAE,
    UniversalTactileExpert,
    UniversalTactileExpertConfig,
    UniversalTactileTaskConfig,
)
from .wtam_v25.frozen_anatomy_tactile_codec import FrozenAnatomyTactileCodecV3
from .wtam_v4_uni import FrozenUnifiedTactileCodecV4
# Add Flow-Matching schedulers
from runtime.dexjoco.wan.utils.fm import FlowMatchScheduler
from runtime.dexjoco.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from runtime.dexjoco.utils.stage2_warmstart import (
    load_stage2_action_only_weights,
    load_stage2_warmstart_weights,
    load_stage2_wan_only_weights,
)
from runtime.dexjoco.utils.va_to_vat_warmstart import load_va_checkpoint_for_vat

logger = logging.getLogger(__name__)

WAN_TASK_TEXT_FEATURES_KEY = "task_text_features"
WAN_TASK_TEXT_PROJECTION_KEY = "task_text_projection.weight"


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    per_sample_mean: bool = False,
) -> torch.Tensor:
    """MSE with an optional sample-balanced validity mask.

    ``per_sample_mean`` first normalizes each sample by its own number of
    valid scalar targets and only then averages the batch.  This prevents a
    44-D bimanual sample from receiving twice the weight of a 22-D sample.
    """
    error = (prediction.float() - target.float()).square()
    if valid_mask is None:
        return error.flatten(1).mean(dim=1).mean() if per_sample_mean else error.mean()
    mask = torch.as_tensor(valid_mask, device=error.device, dtype=error.dtype)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(1)
    try:
        mask = mask.expand_as(error)
    except RuntimeError as exc:
        raise ValueError(
            f"valid_mask {tuple(mask.shape)} cannot expand to error {tuple(error.shape)}"
        ) from exc
    if per_sample_mean:
        numerator = (error * mask).flatten(1).sum(dim=1)
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        return (numerator / denominator).mean()
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def prepare_action_flow_batch(
    actions: torch.Tensor,
    state: Optional[torch.Tensor],
    action_valid_mask: Optional[torch.Tensor],
    sigma: torch.Tensor,
    *,
    dtype: torch.dtype,
    masked_zero_padding: bool,
    pi05_zero_padding: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Construct action FM input/target without changing the legacy default.

    Masked-zero mode removes invalid dimensions from the complete diffusion
    trajectory. Pi0.5-zero mode keeps clean padding at zero but applies normal
    diffusion noise and supervision to every output dimension.
    """
    batch, _, action_dim = actions.shape
    if action_valid_mask is None:
        valid = torch.ones((batch, action_dim), device=actions.device, dtype=torch.bool)
    else:
        if tuple(action_valid_mask.shape) != (batch, action_dim):
            raise ValueError(
                f"action_valid_mask must be [B, action_dim], got {tuple(action_valid_mask.shape)}"
            )
        valid = action_valid_mask.to(device=actions.device, dtype=torch.bool)
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("Each sample must have at least one valid action dimension")

    action_values = valid[:, None, :].to(dtype=actions.dtype)
    clean_actions = actions
    resolved_state = state
    if masked_zero_padding or pi05_zero_padding:
        clean_actions = actions * action_values
        if state is not None and state.shape[-1] == action_dim:
            resolved_state = state * valid.to(device=state.device, dtype=state.dtype)

    noise = torch.randn_like(clean_actions, dtype=dtype)
    if masked_zero_padding:
        noise = noise * action_values.to(dtype=noise.dtype)
    noisy = clean_actions * (1 - sigma) + noise * sigma
    target = noise - clean_actions
    if masked_zero_padding:
        noisy = noisy * action_values.to(dtype=noisy.dtype)
        target = target * action_values.to(dtype=target.dtype)
    loss_mask = None if pi05_zero_padding else valid[:, None, :]
    return noisy, target, resolved_state, valid, loss_mask


def build_flowmatch_sigma_schedule(
    num_inference_steps: int,
    shift: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the exact shifted FM sigma nodes, including the final zero."""
    if num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"Flow-matching shift must be finite and positive, got {shift}")
    base_sigmas = torch.linspace(
        1.0,
        0.0,
        num_inference_steps + 1,
        device=device,
        dtype=dtype,
    )
    return shift * base_sigmas / (1 + (shift - 1) * base_sigmas)


def build_joint_tactile_validity_mask(
    *,
    video_len: int,
    action_len: int,
    tactile_token_valid: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive mask that excludes missing anatomy slots as keys."""
    if tactile_token_valid.ndim != 2:
        raise ValueError(
            "tactile_token_valid must be [B,tactile_len], got "
            f"{tuple(tactile_token_valid.shape)}"
        )
    batch, tactile_len = tactile_token_valid.shape
    prefix_valid = torch.ones(
        batch,
        int(video_len) + int(action_len),
        dtype=torch.bool,
        device=tactile_token_valid.device,
    )
    key_valid = torch.cat((prefix_valid, tactile_token_valid.bool()), dim=1)
    total_len = int(video_len) + int(action_len) + int(tactile_len)
    mask = torch.zeros(
        batch, 1, total_len, total_len,
        dtype=dtype,
        device=tactile_token_valid.device,
    )
    return mask.masked_fill(~key_valid[:, None, None, :], float("-inf"))


def compact_homogeneous_v3_tactile_tokens(
    latent: torch.Tensor,
    token_valid: torch.Tensor,
    *,
    force_bimanual_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact V3 anatomy slots, supporting mixed single/bimanual batches."""
    if latent.ndim != 4 or token_valid.shape != latent.shape[:-1]:
        raise ValueError(
            "Expected latent [B,T,12,D] and validity [B,T,12], got "
            f"{tuple(latent.shape)} and {tuple(token_valid.shape)}"
        )
    if latent.shape[2] != 12:
        raise ValueError(f"V3 anatomy compaction requires 12 slots, got {latent.shape[2]}")
    allowed_single = (6, 7, 8, 9, 11)
    allowed_bimanual = (0, 1, 2, 3, 5, 6, 7, 8, 9, 11)
    valid = token_valid.bool()
    signatures = {tuple(int(v) for v in row.tolist()) for row in valid[:, 0]}
    single_sig = tuple(i in allowed_single for i in range(12))
    bimanual_sig = tuple(i in allowed_bimanual for i in range(12))
    if not signatures.issubset({single_sig, bimanual_sig}):
        raise ValueError(f"Unexpected DexJoCo anatomy validity signatures: {sorted(signatures)}")
    selected = (
        allowed_bimanual
        if force_bimanual_layout or signatures != {single_sig}
        else allowed_single
    )
    anatomy_indices = torch.tensor(selected, device=latent.device, dtype=torch.long)
    return latent.index_select(2, anatomy_indices), anatomy_indices


def load_wan_task_embedding(path: str, expected_dim: int) -> torch.Tensor:
    """Load the exact DiffSynth task embedding without accepting extra state."""
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        if keys != ["task_embedding.weight"]:
            raise RuntimeError(
                f"Task embedding checkpoint must contain only task_embedding.weight, got {keys}"
            )
        weight = checkpoint.get_tensor("task_embedding.weight")
    if weight.ndim != 2 or int(weight.shape[1]) != int(expected_dim):
        raise RuntimeError(
            f"Task embedding must have shape [num_tasks,{expected_dim}], got {tuple(weight.shape)}"
        )
    if weight.shape[0] < 1 or not torch.isfinite(weight).all().item():
        raise RuntimeError("Task embedding must contain at least one finite row")
    return weight


def install_wan_task_embedding(wan_model: nn.Module, weight: torch.Tensor) -> nn.Embedding:
    """Register the full task table on WAN so its parameters use the WAN LR group."""
    task_embedding = nn.Embedding(weight.shape[0], weight.shape[1])
    wan_parameter = next(wan_model.parameters())
    task_embedding.to(device=wan_parameter.device, dtype=weight.dtype)
    with torch.no_grad():
        task_embedding.weight.copy_(weight.to(task_embedding.weight.device))
    wan_model.task_embedding = task_embedding
    return task_embedding


def add_wan_task_embedding(
    time_embedding: torch.Tensor,
    task_embedding: nn.Embedding,
    task_id: int | torch.Tensor,
) -> torch.Tensor:
    """Match DiffSynth's task-id addition between time embedding and projection."""
    task_ids = normalize_wan_task_ids(
        task_id,
        batch_size=time_embedding.shape[0],
        num_tasks=task_embedding.num_embeddings,
        device=time_embedding.device,
    )
    task_features = task_embedding(task_ids).to(
        device=time_embedding.device,
        dtype=time_embedding.dtype,
    )
    if time_embedding.ndim == 3:
        task_features = task_features[:, None, :]
    return time_embedding + task_features


def normalize_wan_task_ids(
    task_id: int | torch.Tensor,
    *,
    batch_size: int,
    num_tasks: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize scalar or per-sample task ids without silently changing batch size."""
    if isinstance(task_id, torch.Tensor):
        task_ids = task_id.to(device=device, dtype=torch.long).flatten()
    else:
        task_ids = torch.tensor([int(task_id)], device=device, dtype=torch.long)
    if task_ids.numel() == 1 and batch_size != 1:
        task_ids = task_ids.expand(batch_size)
    if task_ids.numel() != batch_size:
        raise ValueError(
            f"WAN task-id batch must contain 1 or {batch_size} values, got {task_ids.numel()}"
        )
    if bool(torch.any((task_ids < 0) | (task_ids >= num_tasks)).item()):
        raise ValueError(
            f"task_id must be in [0,{num_tasks - 1}], got {task_ids.detach().cpu().tolist()}"
        )
    return task_ids


def load_wan_task_text_conditioning(
    path: str,
    *,
    expected_num_tasks: int = 5,
    expected_text_dim: int = 4096,
    expected_model_dim: int = 3072,
) -> Dict[str, torch.Tensor]:
    """Load the exact two-tensor conditioning state from a five-modal WAN."""
    expected_keys = [WAN_TASK_TEXT_FEATURES_KEY, WAN_TASK_TEXT_PROJECTION_KEY]
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        if keys != expected_keys:
            raise RuntimeError(
                f"Task text-conditioning checkpoint must contain exactly {expected_keys}, got {keys}"
            )
        features = checkpoint.get_tensor(WAN_TASK_TEXT_FEATURES_KEY)
        projection = checkpoint.get_tensor(WAN_TASK_TEXT_PROJECTION_KEY)
    if list(features.shape) != [expected_num_tasks, expected_text_dim]:
        raise RuntimeError(
            f"{WAN_TASK_TEXT_FEATURES_KEY} must have shape "
            f"[{expected_num_tasks},{expected_text_dim}], got {tuple(features.shape)}"
        )
    if features.dtype != torch.float32:
        raise RuntimeError(f"{WAN_TASK_TEXT_FEATURES_KEY} must be FP32, got {features.dtype}")
    if list(projection.shape) != [expected_model_dim, expected_text_dim]:
        raise RuntimeError(
            f"{WAN_TASK_TEXT_PROJECTION_KEY} must have shape "
            f"[{expected_model_dim},{expected_text_dim}], got {tuple(projection.shape)}"
        )
    if not torch.isfinite(features).all().item() or not torch.isfinite(projection).all().item():
        raise RuntimeError("Task text-conditioning tensors must be finite")
    return {
        WAN_TASK_TEXT_FEATURES_KEY: features,
        WAN_TASK_TEXT_PROJECTION_KEY: projection,
    }


def install_wan_task_text_conditioning(
    wan_model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> nn.Linear:
    """Install DiffSynth's frozen T5 feature table and learned projection on Motus WAN."""
    features = state[WAN_TASK_TEXT_FEATURES_KEY]
    projection_weight = state[WAN_TASK_TEXT_PROJECTION_KEY]
    reference_parameter = next(wan_model.parameters())
    wan_model.register_buffer(
        WAN_TASK_TEXT_FEATURES_KEY,
        features.detach().to(device=reference_parameter.device, dtype=torch.float32),
        persistent=True,
    )
    projection = nn.Linear(
        projection_weight.shape[1],
        projection_weight.shape[0],
        bias=False,
        device=reference_parameter.device,
        dtype=projection_weight.dtype,
    )
    with torch.no_grad():
        projection.weight.copy_(projection_weight.to(device=projection.weight.device))
    wan_model.task_text_projection = projection
    wan_model.num_task_text_features = int(features.shape[0])
    return projection


def add_wan_task_text_conditioning(
    time_embedding: torch.Tensor,
    task_text_features: torch.Tensor,
    task_text_projection: nn.Linear,
    task_id: int | torch.Tensor,
) -> torch.Tensor:
    """Match DiffSynth's pooled-T5 task conditioning exactly."""
    task_ids = normalize_wan_task_ids(
        task_id,
        batch_size=time_embedding.shape[0],
        num_tasks=task_text_features.shape[0],
        device=time_embedding.device,
    )
    features = task_text_features[task_ids].to(
        device=task_text_projection.weight.device,
        dtype=task_text_projection.weight.dtype,
    )
    task_condition = task_text_projection(features).to(
        device=time_embedding.device,
        dtype=time_embedding.dtype,
    )
    if time_embedding.ndim == 3:
        task_condition = task_condition[:, None, :]
    return time_embedding + task_condition

@dataclass
class MotusConfig:
    """Configuration for Motus."""
    # Video model settings
    wan_checkpoint_path: str = ""
    vae_path: str = ""
    wan_config_path: str = ""
    video_precision: str = "bfloat16"
    wan_expected_num_tensors: Optional[int] = None
    wan_task_embedding_path: Optional[str] = None
    wan_task_text_conditioning_path: Optional[str] = None
    wan_task_id: Optional[int] = None
    wan_task_probabilities: Optional[List[float]] = None
    wan_frozen: bool = False
    video_training_strategy: str = "flow_matching"
    video_rollout_steps: int = 10

    # VLM settings
    vlm_checkpoint_path: str = ""
    use_understanding: bool = True

    # Understanding Expert settings - configurable from yaml
    und_expert_hidden_size: int = 512        # Understanding expert hidden dimension
    und_expert_ffn_dim_multiplier: int = 4   # Understanding expert FFN dimension multiplier
    und_expert_norm_eps: float = 1e-5        # Understanding expert layer norm epsilon
    und_layers_to_extract: List[int] = None  # Which VLM layers to extract from

    # VLM adapter settings for understanding expert
    vlm_adapter_input_dim: int = 2048        # VLM feature dimension (input)
    vlm_adapter_projector_type: str = "mlp3x_silu"  # VLM adapter type

    # Action expert settings
    num_layers: int = 30
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024           # Configurable hidden dimension
    action_expert_ffn_dim_multiplier: int = 4  # FFN dimension multiplier
    action_expert_norm_eps: float = 1e-6    # Layer norm epsilon for Action Expert
    joint_attention_mode: str = 'full_bidirectional'

    # Sampling settings
    global_downsample_rate: int = 3     # Global downsampling rate
    video_action_freq_ratio: int = 4    # Video:Action frequency ratio
    num_video_frames: int = 4           # Number of video frames to predict
    action_chunk_size: Optional[int] = None  # Explicitly decouples action from video count

    # Training timestep distribution
    timestep_sample_method: str = 'logit_normal'
    sigmoid_scale: float = 1.0
    min_t: float = 0.0
    max_t: float = 1.0
    video_train_schedule_shift: float = 5.0
    action_train_schedule_shift: float = 5.0
    tactile_train_schedule_shift: float = 5.0

    # Video dimensions
    video_height: int = 512             # Input video height
    video_width: int = 512              # Input video width

    # Training settings
    batch_size: int = 8

    # Training mode
    training_mode: str = 'finetune'  # 'pretrain' or 'finetune'

    # Loss weights
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    tactile_loss_weight: float = 0.1
    # Opt-in contracts. Defaults preserve every legacy config/checkpoint.
    per_sample_valid_mean_loss: bool = False
    masked_zero_padding: bool = False
    pi05_zero_padding: bool = False

    # V2.1 frozen TacVAE + fresh tactile expert.
    tactile_enabled: bool = False
    tactile_vae_checkpoint_path: Optional[str] = None
    tactile_vae_expected_sha256: Optional[str] = None
    tactile_latent_stats_path: Optional[str] = None
    tactile_latent_stats_expected_sha256: Optional[str] = None
    tactile_support_calibration_path: Optional[str] = None
    tactile_support_calibration_expected_sha256: Optional[str] = None
    tactile_latent_stats_task_name: str = "robotwin_vat_supportvae_sparse_multitask_v1"
    tactile_expert_config: Optional[Dict[str, Any]] = None
    tactile_codec_type: str = "v21_support_vae"
    tactile_codec_source_root: Optional[str] = None
    tactile_codec_source_revision: Optional[str] = None
    tactile_codec_source_files_sha256: Optional[Dict[str, str]] = None
    tactile_codec_expected_step: Optional[int] = None
    tactile_codec_normalization_group: Optional[str] = None
    tactile_dexjoco_channel_scale: Optional[tuple[float, float, float]] = None

    # Control whether to load pretrained WAN/VLM backbones.
    # None = default behavior (load), False = skip loading (init from config only)
    load_pretrained_backbones: Optional[bool] = None

    def __post_init__(self):
        """Calculate derived parameters."""
        # Some entry points use PyYAML directly, where values such as 1e-5 may
        # arrive as strings. Normalize numeric fields before building modules.
        self.und_expert_norm_eps = float(self.und_expert_norm_eps)
        self.action_expert_norm_eps = float(self.action_expert_norm_eps)
        self.video_loss_weight = float(self.video_loss_weight)
        self.action_loss_weight = float(self.action_loss_weight)
        self.tactile_loss_weight = float(self.tactile_loss_weight)
        self.per_sample_valid_mean_loss = bool(self.per_sample_valid_mean_loss)
        self.masked_zero_padding = bool(self.masked_zero_padding)
        self.pi05_zero_padding = bool(self.pi05_zero_padding)
        if self.masked_zero_padding and self.pi05_zero_padding:
            raise ValueError(
                "masked_zero_padding and pi05_zero_padding are mutually exclusive"
            )
        self.sigmoid_scale = float(self.sigmoid_scale)
        self.min_t = float(self.min_t)
        self.max_t = float(self.max_t)
        self.video_train_schedule_shift = float(self.video_train_schedule_shift)
        self.action_train_schedule_shift = float(self.action_train_schedule_shift)
        self.tactile_train_schedule_shift = float(self.tactile_train_schedule_shift)
        self.tactile_enabled = bool(self.tactile_enabled)
        if self.tactile_codec_expected_step is not None:
            self.tactile_codec_expected_step = int(self.tactile_codec_expected_step)
        if self.tactile_codec_source_files_sha256 is not None:
            self.tactile_codec_source_files_sha256 = dict(
                self.tactile_codec_source_files_sha256
            )
        self.video_rollout_steps = int(self.video_rollout_steps)
        self.wan_frozen = bool(self.wan_frozen)
        self.joint_attention_mode = str(self.joint_attention_mode)
        if self.joint_attention_mode not in JOINT_ATTENTION_MODES:
            raise ValueError(
                f"joint_attention_mode must be one of {sorted(JOINT_ATTENTION_MODES)}, "
                f"got {self.joint_attention_mode!r}"
            )
        if self.joint_attention_mode == 'wan_full_action_causal':
            if self.use_understanding:
                raise ValueError('wan_full_action_causal requires use_understanding=false')
            if self.training_mode != 'finetune':
                raise ValueError('wan_full_action_causal requires training_mode=finetune')
        for name, shift in (
            ("video_train_schedule_shift", self.video_train_schedule_shift),
            ("action_train_schedule_shift", self.action_train_schedule_shift),
            ("tactile_train_schedule_shift", self.tactile_train_schedule_shift),
        ):
            if not math.isfinite(shift) or shift <= 0:
                raise ValueError(f"{name} must be finite and positive, got {shift}")
        if self.tactile_enabled:
            if self.use_understanding:
                raise ValueError("V2.1 tactile and Understanding branches are mutually exclusive")
            required = {
                "tactile_vae_checkpoint_path": self.tactile_vae_checkpoint_path,
                "tactile_vae_expected_sha256": self.tactile_vae_expected_sha256,
            }
            if self.tactile_codec_type == "v21_support_vae":
                required.update({
                    "tactile_latent_stats_path": self.tactile_latent_stats_path,
                    "tactile_latent_stats_expected_sha256": self.tactile_latent_stats_expected_sha256,
                    "tactile_support_calibration_path": self.tactile_support_calibration_path,
                    "tactile_support_calibration_expected_sha256": self.tactile_support_calibration_expected_sha256,
                })
            elif self.tactile_codec_type == "v3_anatomy":
                required["tactile_codec_source_root"] = self.tactile_codec_source_root
                required["tactile_codec_source_revision"] = self.tactile_codec_source_revision
            elif self.tactile_codec_type == "v4_uni_50k":
                required.update({
                    "tactile_codec_source_root": self.tactile_codec_source_root,
                    "tactile_codec_source_files_sha256": (
                        self.tactile_codec_source_files_sha256
                    ),
                    "tactile_codec_expected_step": self.tactile_codec_expected_step,
                    "tactile_codec_normalization_group": (
                        self.tactile_codec_normalization_group
                    ),
                })
            else:
                raise ValueError(f"Unknown tactile_codec_type={self.tactile_codec_type!r}")
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"V2.1 tactile configuration is missing: {missing}")
        if self.video_training_strategy not in {"flow_matching", "frozen_rollout"}:
            raise ValueError(
                "video_training_strategy must be 'flow_matching' or 'frozen_rollout', "
                f"got {self.video_training_strategy!r}"
            )
        if self.video_rollout_steps <= 0:
            raise ValueError("video_rollout_steps must be positive")
        if self.wan_task_embedding_path and self.wan_task_text_conditioning_path:
            raise ValueError("Legacy task embedding and T5 task conditioning are mutually exclusive")
        if self.wan_task_probabilities is not None:
            self.wan_task_probabilities = [float(value) for value in self.wan_task_probabilities]
            if not self.wan_task_probabilities or any(
                not math.isfinite(value) or value < 0 for value in self.wan_task_probabilities
            ):
                raise ValueError("wan_task_probabilities must contain finite non-negative values")
            if sum(self.wan_task_probabilities) <= 0:
                raise ValueError("wan_task_probabilities must have a positive sum")
            if self.wan_task_text_conditioning_path and len(self.wan_task_probabilities) != 5:
                raise ValueError("Five-modal T5 conditioning requires exactly 5 probabilities")
            if self.wan_task_id is not None:
                raise ValueError("Configure either wan_task_id or wan_task_probabilities, not both")
        if self.video_training_strategy == "frozen_rollout":
            if not self.wan_frozen:
                raise ValueError("frozen_rollout requires wan_frozen=true")
            if self.video_loss_weight != 0.0:
                raise ValueError("frozen_rollout requires video_loss_weight=0")
            if self.wan_task_text_conditioning_path is None:
                raise ValueError("frozen_rollout requires wan_task_text_conditioning_path")

        # Legacy experiments derive action length from the video ratio.  New
        # native-clock contracts (e.g. DexJoCo Dense-20) configure it explicitly
        # because Video, Action and Tactile may have different sample counts
        # while covering the same physical horizon.
        if self.action_chunk_size is None:
            self.action_chunk_size = (
                self.num_video_frames * self.video_action_freq_ratio
            )
        self.action_chunk_size = int(self.action_chunk_size)
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")

        # Default understanding layers to extract from (if not specified)
        if self.und_layers_to_extract is None:
            # Extract from all layers for comprehensive understanding
            self.und_layers_to_extract = list(range(self.num_layers))


class VideoModule(nn.Module):
    """Video processing module - handles WAN + T5 operations."""

    def __init__(
        self,
        video_model,
        dtype,
        device,
        grid_sizes,
        task_id: Optional[int] = None,
        joint_attention_mode: str = 'full_bidirectional',
        state_action_len: int = 0,
        register_len: int = 0,
    ):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes
        self.task_id = task_id
        self.joint_attention_mode = joint_attention_mode
        self.state_action_len = int(state_action_len)
        self.register_len = int(register_len)
        self._joint_attention_mask_cache = {}
        self.attention_capture = None
        self.attention_capture_step = 0

    def set_attention_capture(self, capture, step: int = 0) -> None:
        self.attention_capture = capture
        self.attention_capture_step = int(step)

    def _prepare_attention_capture(self, wan_layer, layer_idx: int) -> None:
        wan_layer.self_attn.attention_capture = self.attention_capture
        wan_layer.self_attn.attention_capture_layer = int(layer_idx)
        wan_layer.self_attn.attention_capture_step = int(self.attention_capture_step)

    def get_video_action_attention_mask(
        self,
        video_len: int,
        action_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.joint_attention_mode == 'full_bidirectional':
            return None
        if self.joint_attention_mode != 'wan_full_action_causal':
            raise RuntimeError(f'Unsupported joint attention mode: {self.joint_attention_mode!r}')
        if action_len != self.state_action_len + self.register_len:
            raise ValueError(
                'Action token layout differs from causal mask contract: '
                f'action_len={action_len}, state_action_len={self.state_action_len}, '
                f'register_len={self.register_len}'
            )
        key = (
            int(video_len),
            self.state_action_len,
            self.register_len,
            device.type,
            device.index,
            dtype,
        )
        mask = self._joint_attention_mask_cache.get(key)
        if mask is None:
            mask = build_wan_full_action_causal_mask(
                int(video_len),
                self.state_action_len,
                self.register_len,
                device=device,
                dtype=dtype,
            )
            self._joint_attention_mask_cache[key] = mask
            logger.info(
                'Joint attention mask initialized: mode=%s backend=sdpa '
                'video_tokens=%s state_action_tokens=%s register_tokens=%s shape=%s',
                self.joint_attention_mode,
                video_len,
                self.state_action_len,
                self.register_len,
                tuple(mask.shape),
            )
        return mask

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare video tokens from pre-processed noisy latent."""
        # Through patch_embedding: 48 -> 3072 channels
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)

        # Flatten and convert to tokens
        video_features = video_patched.flatten(2).transpose(1, 2)

        # Calculate sequence length and padding
        # seq_lens = torch.tensor([u.size(1) for u in video_tokens_list], dtype=torch.long, device=self.device)
        # seq_len = seq_lens.max().item()

        # Concatenate with padding
        # video_tokens = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        #     for u in video_tokens_list
        # ])

        # return video_tokens

        return video_features

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """Pre-process T5 embeddings once for all layers."""
        # Handle both old format (List[torch.Tensor]) and new format (torch.Tensor)
        if isinstance(language_embeddings, list):
            # Old format: List[torch.Tensor] - do padding
            text_len = self.video_model.wan_model.text_len  # 512
            padded_embeddings = []

            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)

            t5_context_raw = torch.stack(padded_embeddings, dim=0)
        else:
            # New format: torch.Tensor [B, seq_len, dim] - already padded by collate_fn
            t5_context_raw = language_embeddings

        # Convert via text_embedding layer (4096 -> 3072)
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)

        return t5_context

    def get_time_embedding(
        self,
        t_video: torch.Tensor,
        seq_len: int,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get WAN's time embedding using WAN's own weights."""
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()

            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )
            resolved_task_id = self.task_id if task_id is None else task_id
            if resolved_task_id is not None:
                task_text_features = getattr(self.video_model.wan_model, "task_text_features", None)
                task_text_projection = getattr(self.video_model.wan_model, "task_text_projection", None)
                task_embedding = getattr(self.video_model.wan_model, "task_embedding", None)
                if task_text_features is not None and task_text_projection is not None:
                    t_emb = add_wan_task_text_conditioning(
                        t_emb,
                        task_text_features,
                        task_text_projection,
                        resolved_task_id,
                    )
                elif task_embedding is not None:
                    t_emb = add_wan_task_embedding(t_emb, task_embedding, resolved_task_id)
                else:
                    raise RuntimeError("WAN task_id is configured but task conditioning is missing")
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))
            assert t_emb.dtype == torch.float32 and t_emb_proj.dtype == torch.float32

        return t_emb, t_emb_proj

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor,
                               layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        """Process WAN cross attention with pre-processed T5 context."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None  # WAN uses None for fixed-length context
        cross_out = wan_layer.cross_attn(wan_layer.norm3(video_tokens), processed_t5_context, context_lens)
        return video_tokens + cross_out

    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for WAN (6 components)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process WAN FFN with proper AdaLN modulation."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params
        v_mod = video_adaln_modulation

        # WAN FFN with AdaLN (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)
        ffn_out = wan_layer.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        """Apply WAN's head + unpatchify for final video output."""
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    def process_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        und_tokens: torch.Tensor,
        und_block: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trimodal joint self-attention: WAN + Action + Understanding via WAN self-attn (MoT)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        self._prepare_attention_capture(wan_layer, layer_idx)

        # AdaLN params (already computed)
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        # Pre-attn normalization with AdaLN
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)

        # Get dimensions
        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n

        # Action heads for WAN space (1024 -> 24*128)
        a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        # Understanding Expert processing
        norm_und = und_block.norm1(und_tokens)
        L_u = norm_und.shape[1]

        # Understanding Expert heads for WAN space (2048 -> 24*128)
        u_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, und_block.wan_und_qkv)
        u_q_h, u_k_h, u_v_h = u_qkv[0], u_qkv[1], u_qkv[2]
        u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(B, L_u, n, d)
        u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(B, L_u, n, d)
        u_v = u_v_h.view(B, L_u, n, d)

        # Meta info for WAN attention
        seq_lens = torch.full((B,), L_v + L_a + L_u, dtype=torch.long, device=self.device)
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)

        # Call WAN self-attn with trimodal MoT
        y, action_out_h, und_out_h = wan_layer.self_attn(
            norm_video, seq_lens, self.grid_sizes, freqs,
            action_q=a_q, action_k=a_k, action_v=a_v,
            und_q=u_q, und_k=u_k, und_v=u_v
        )

        # Project Understanding Expert output
        und_out = und_block.wan_und_o(und_out_h.flatten(2))

        # Project back and residual connections
        action_out = action_block.wan_action_o(action_out_h.flatten(2))
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        und_tokens = und_tokens + und_out  # Regular residual connection

        return video_tokens, action_tokens, und_tokens

    def process_video_action_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bimodal joint self-attention: WAN + Action via WAN self-attn (MoT)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        self._prepare_attention_capture(wan_layer, layer_idx)

        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)

        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n

        a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        seq_lens = torch.full((B,), L_v + L_a, dtype=torch.long, device=self.device)
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)

        attn_mask = self.get_video_action_attention_mask(
            L_v,
            L_a,
            device=a_q.device,
            dtype=a_q.dtype,
        )
        y, action_out_h, _ = wan_layer.self_attn(
            norm_video, seq_lens, self.grid_sizes, freqs,
            action_q=a_q, action_k=a_k, action_v=a_v,
            attn_mask=attn_mask,
        )

        action_out = action_block.wan_action_o(action_out_h.flatten(2))
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)

        return video_tokens, action_tokens

    def process_video_action_tactile_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        tactile_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        tactile_block: nn.Module,
        tactile_token_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V2.1 all-to-all Video-Action-Tactile attention in one WAN call."""
        if self.joint_attention_mode != "full_bidirectional":
            raise RuntimeError(
                "V2.1 is Full-Joint MoT and requires joint_attention_mode=full_bidirectional"
            )
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        self._prepare_attention_capture(wan_layer, layer_idx)
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation
        t_mod = tactile_adaln_modulation

        norm_video = wan_layer.norm1(video_tokens).float() * (
            1 + v_mod[1].squeeze(2)
        ) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens).float() * (
            1 + a_mod[1].squeeze(2)
        ) + a_mod[0].squeeze(2)
        norm_tactile = tactile_block.norm1(tactile_tokens).float() * (
            1 + t_mod[1].squeeze(2)
        ) + t_mod[0].squeeze(2)

        batch, video_len, channels = norm_video.shape
        action_len = norm_action.shape[1]
        tactile_len = norm_tactile.shape[1]
        tactile_valid_values = None
        if tactile_token_valid is not None:
            if tuple(tactile_token_valid.shape) != (batch, tactile_len):
                raise ValueError(
                    "Expected tactile validity "
                    f"{(batch, tactile_len)}, got {tuple(tactile_token_valid.shape)}"
                )
            tactile_token_valid = tactile_token_valid.to(
                device=tactile_tokens.device, dtype=torch.bool
            )
            tactile_valid_values = tactile_token_valid.to(
                tactile_tokens.dtype
            ).unsqueeze(-1)
            tactile_tokens = tactile_tokens * tactile_valid_values
            norm_tactile = norm_tactile * tactile_valid_values
        heads = self.video_model.wan_model.num_heads
        head_dim = channels // heads

        action_qkv = torch.einsum(
            "BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv
        )
        action_q = action_block.wan_action_norm_q(
            action_qkv[0].flatten(-2)
        ).view(batch, action_len, heads, head_dim)
        action_k = action_block.wan_action_norm_k(
            action_qkv[1].flatten(-2)
        ).view(batch, action_len, heads, head_dim)
        action_v = action_qkv[2].view(batch, action_len, heads, head_dim)

        tactile_qkv = torch.einsum(
            "BTD,KNDE->KBTNE", norm_tactile, tactile_block.wan_tactile_qkv
        )
        tactile_q = tactile_block.wan_tactile_norm_q(
            tactile_qkv[0].flatten(-2)
        ).view(batch, tactile_len, heads, head_dim)
        tactile_k = tactile_block.wan_tactile_norm_k(
            tactile_qkv[1].flatten(-2)
        ).view(batch, tactile_len, heads, head_dim)
        tactile_v = tactile_qkv[2].view(batch, tactile_len, heads, head_dim)

        seq_lens = torch.full(
            (batch,),
            video_len + action_len + tactile_len,
            dtype=torch.long,
            device=video_tokens.device,
        )
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)
        attn_mask = (
            build_joint_tactile_validity_mask(
                video_len=video_len,
                action_len=action_len,
                tactile_token_valid=tactile_token_valid,
                dtype=norm_video.dtype,
            )
            if tactile_token_valid is not None
            else None
        )
        video_out, action_out_heads, tactile_out_heads = wan_layer.self_attn(
            norm_video,
            seq_lens,
            self.grid_sizes,
            freqs,
            action_q=action_q,
            action_k=action_k,
            action_v=action_v,
            und_q=tactile_q,
            und_k=tactile_k,
            und_v=tactile_v,
            attn_mask=attn_mask,
        )
        action_out = action_block.wan_action_o(action_out_heads.flatten(2))
        tactile_out = tactile_block.wan_tactile_o(tactile_out_heads.flatten(2))
        tactile_result = tactile_tokens + tactile_out * t_mod[2].squeeze(2)
        if tactile_valid_values is not None:
            tactile_result = tactile_result * tactile_valid_values
        return (
            video_tokens + video_out * v_mod[2].squeeze(2),
            action_tokens + action_out * a_mod[2].squeeze(2),
            tactile_result,
        )


class UndModule(nn.Module):
    """Understanding module - handles VLM with understanding queries and Understanding Expert."""

    def __init__(self, vlm_model, und_expert, config, dtype, device):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

        # VLM model reference
        self.vlm_model = vlm_model

        # Understanding Expert reference
        self.und_expert = und_expert

    def extract_und_features(
        self,
        vlm_inputs
    ) -> torch.Tensor:
        """Extract understanding features from VLM last layer."""
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)

        # Forward through VLM with proper attention_mask and DeepStack features
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': True,
            'return_dict': True
        }

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        with torch.no_grad():
            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly
        last_layer_features = vlm_output.hidden_states[-1]  # [B, seq_len, vlm_dim]

        # [B, seq_len, vlm_dim] -> [B, seq_len, und_dim]
        adapted_features = self.und_expert.vlm_adapter(last_layer_features)

        return adapted_features

    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of (inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids)
        """
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            # New format: Dict[str, Tensor] - already batched and padded by collate_fn
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)

        # Get input embeddings
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)

        # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
        image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)

        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)

        # Insert image embeddings
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids

    def process_ffn(self, und_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Process Understanding Expert FFN with regular LayerNorm."""
        block = self.und_expert.blocks[layer_idx]

        # Pre-norm for FFN (regular LayerNorm)
        ffn_input = block.norm2(und_tokens)
        ffn_output = block.ffn(ffn_input)

        # FFN residual connection
        und_tokens = und_tokens + ffn_output

        return und_tokens


class ActionModule(nn.Module):
    """Action processing module - handles Action Expert + joint attentions + masks."""

    def __init__(self, action_expert: ActionExpert, config, video_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model  # For accessing WAN weights
        self.dtype = dtype
        self.device = device

    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action time embedding."""
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()

            # Create sinusoidal embedding (same pattern as VideoModule)
            a_e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )  # [B, seq_len, freq_dim]

            # Project to AdaLN parameters (6 params: 3 for WAN-Action joint attn + 3 for FFN)
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))  # [B, seq_len, 6, dim]

            assert a_e.dtype == torch.float32 and a_e0.dtype == torch.float32

        return a_e, a_e0  # (basic_emb, adaln_params)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for 6 components (3 for WAN-Action joint attn + 3 for FFN)."""
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0)
                + action_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process Action Expert FFN with AdaLN modulation."""
        action_block = self.action_expert.blocks[layer_idx]

        # AdaLN params
        a_mod = action_adaln_modulation

        # Apply FFN with AdaLN modulation (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)
        ffn_out = action_block.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens


class Motus(nn.Module):
    """
    Modular Three-modal UniDiffuser with VGM, VLM, and Action modules.
    """

    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config

        # Set unified data type for the model
        self.dtype = torch.bfloat16

        # Initialize video/VLM backbones
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)
        self.use_understanding = bool(config.use_understanding)
        logger.info('Joint attention mode: %s', config.joint_attention_mode)

        # Initialize video model (WAN)
        logger.info("Initializing WAN video model...")
        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision,
                expected_num_tensors=config.wan_expected_num_tensors,
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device="cuda",
                precision=config.video_precision
            )

        self.video_task_id = None
        self.wan_task_embedding_load_report = None
        self.wan_task_text_conditioning_load_report = None
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_text_dim = getattr(self.video_model.wan_model, 'text_dim', 4096)
        if config.wan_task_text_conditioning_path:
            task_state = load_wan_task_text_conditioning(
                config.wan_task_text_conditioning_path,
                expected_num_tasks=5,
                expected_text_dim=wan_text_dim,
                expected_model_dim=wan_dim,
            )
            projection = install_wan_task_text_conditioning(
                self.video_model.wan_model,
                task_state,
            )
            if config.wan_task_id is not None:
                task_id = int(config.wan_task_id)
                if task_id < 0 or task_id >= task_state[WAN_TASK_TEXT_FEATURES_KEY].shape[0]:
                    raise ValueError(f"WAN task_id must be in [0,4], got {task_id}")
                self.video_task_id = task_id
            probabilities = config.wan_task_probabilities
            if probabilities is not None and len(probabilities) != 5:
                raise ValueError(
                    f"Five-modal WAN requires exactly 5 task probabilities, got {len(probabilities)}"
                )
            self.wan_task_text_conditioning_load_report = {
                "path": str(config.wan_task_text_conditioning_path),
                "loaded_tensor_count": 2,
                "features_shape": list(task_state[WAN_TASK_TEXT_FEATURES_KEY].shape),
                "features_dtype": str(task_state[WAN_TASK_TEXT_FEATURES_KEY].dtype),
                "projection_shape": list(task_state[WAN_TASK_TEXT_PROJECTION_KEY].shape),
                "projection_dtype": str(task_state[WAN_TASK_TEXT_PROJECTION_KEY].dtype),
                "task_id": self.video_task_id,
                "task_probabilities": probabilities,
                "trainable": bool(projection.weight.requires_grad),
            }
            logger.info(
                "Loaded WAN five-modal T5 task conditioning: %s",
                self.wan_task_text_conditioning_load_report,
            )
        elif config.wan_task_embedding_path:
            task_weight = load_wan_task_embedding(
                config.wan_task_embedding_path,
                expected_dim=wan_dim,
            )
            task_id = 0 if config.wan_task_id is None else int(config.wan_task_id)
            if task_id < 0 or task_id >= task_weight.shape[0]:
                raise ValueError(
                    f"WAN task_id must be in [0,{task_weight.shape[0] - 1}], got {task_id}"
                )
            task_embedding = install_wan_task_embedding(
                self.video_model.wan_model,
                task_weight,
            )
            self.video_task_id = task_id
            self.wan_task_embedding_load_report = {
                "path": str(config.wan_task_embedding_path),
                "shape": list(task_weight.shape),
                "dtype": str(task_weight.dtype),
                "task_id": task_id,
                "trainable": bool(task_embedding.weight.requires_grad),
            }
            logger.info("Loaded WAN RGB task conditioning: %s", self.wan_task_embedding_load_report)
        elif config.wan_task_id is not None:
            raise ValueError("wan_task_id requires a WAN task-conditioning path")

        if config.wan_task_probabilities is not None:
            probabilities = torch.tensor(config.wan_task_probabilities, dtype=torch.float64)
            probabilities = probabilities / probabilities.sum()
            self.register_buffer("wan_task_probabilities", probabilities, persistent=True)
        else:
            self.wan_task_probabilities = None

        self.wan_frozen = bool(config.wan_frozen)
        if self.wan_frozen:
            for parameter in self.video_model.wan_model.parameters():
                parameter.requires_grad = False
            self.video_model.wan_model.eval()
            logger.info(
                "WAN frozen before optimizer construction: parameters=%s trainable=0",
                sum(parameter.numel() for parameter in self.video_model.wan_model.parameters()),
            )

        # Get WAN configuration directly
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads

        logger.info(f"Model configurations:")
        logger.info(f"  WAN: {wan_num_heads} heads × {wan_head_dim} head_dim = {wan_dim}D")

        # Create config dictionaries for ActionExpert
        wan_config = {
            'dim': wan_dim,
            'num_heads': wan_num_heads,
            'head_dim': wan_head_dim
        }

        # Initialize action expert with unified configs
        logger.info("Initializing Action Expert...")

        # Determine chunk_size based on training mode
        if config.training_mode == 'pretrain':
            action_chunk_size_for_expert = config.action_chunk_size
        else:
            action_chunk_size_for_expert = config.action_chunk_size + 1  # include state token

        # Configure registers by mode: no registers in pretrain, keep default (e.g., 4) in finetune
        num_registers = 0 if config.training_mode == 'pretrain' else 4

        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            video_feature_dim=wan_dim,
            causal=False,
            eps=config.action_expert_norm_eps,
            training_mode=config.training_mode,
        )

        self.action_expert = ActionExpert(action_config, wan_config)

        # Move models to device
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)

        self.tactile_expert = None
        self.tactile_codec = None
        if config.tactile_enabled:
            expert_config = UniversalTactileExpertConfig.from_mapping(
                config.tactile_expert_config,
                num_layers=config.num_layers,
            )
            self.tactile_expert = UniversalTactileExpert(expert_config)
            self.tactile_expert.to(device=self.device, dtype=self.dtype)
            self.tactile_expert.time_embedding.float()
            self.tactile_expert.time_projection.float()
            tactile_task = UniversalTactileTaskConfig(
                name=config.tactile_latent_stats_task_name,
                domain_id=1,
                normalization_group=1,
                source_regions=4,
                slot_mapping=(0, 1, 2, 3),
                region_semantics=(
                    ("left", "thumb", "tip"),
                    ("left", "index", "tip"),
                    ("right", "thumb", "tip"),
                    ("right", "index", "tip"),
                ),
            )
            if config.tactile_codec_type == "v3_anatomy":
                codec = FrozenAnatomyTactileCodecV3(
                    source_root=config.tactile_codec_source_root,
                    expected_source_revision=config.tactile_codec_source_revision,
                    checkpoint_path=config.tactile_vae_checkpoint_path,
                    expected_sha256=config.tactile_vae_expected_sha256,
                    device=self.device,
                    dtype=self.dtype,
                    dexjoco_channel_scale=config.tactile_dexjoco_channel_scale,
                )
            elif config.tactile_codec_type == "v4_uni_50k":
                codec = FrozenUnifiedTactileCodecV4(
                    source_root=config.tactile_codec_source_root,
                    expected_source_revision=config.tactile_codec_source_revision,
                    expected_source_files_sha256=(
                        config.tactile_codec_source_files_sha256
                    ),
                    checkpoint_path=config.tactile_vae_checkpoint_path,
                    expected_sha256=config.tactile_vae_expected_sha256,
                    expected_step=config.tactile_codec_expected_step,
                    normalization_group=config.tactile_codec_normalization_group,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                codec = FrozenUniversalTactileVAE(
                    checkpoint_path=config.tactile_vae_checkpoint_path,
                    expected_sha256=config.tactile_vae_expected_sha256,
                    latent_stats_path=config.tactile_latent_stats_path,
                    expected_latent_stats_sha256=(
                        config.tactile_latent_stats_expected_sha256
                    ),
                    support_calibration_path=(
                        config.tactile_support_calibration_path
                    ),
                    expected_support_calibration_sha256=(
                        config.tactile_support_calibration_expected_sha256
                    ),
                    task=tactile_task,
                    device=self.device,
                    dtype=self.dtype,
                )
            object.__setattr__(self, "tactile_codec", codec)
            self.tactile_codec.assert_frozen()
            logger.info(
                "Tactile branch initialized: codec=%s tokens=%d topology=full_bidirectional",
                config.tactile_codec_type,
                self.tactile_expert.config.sequence_length,
            )

        self.vlm_model = None
        self.und_expert = None
        self.und_module = None
        if self.use_understanding:
            from transformers import Qwen3VLForConditionalGeneration, AutoConfig

            # Initialize VLM (frozen)
            logger.info("Initializing VLM (frozen)...")
            if load_backbones:
                self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    config.vlm_checkpoint_path,
                    dtype=self.dtype,
                    device_map="cuda",
                    trust_remote_code=True
                )
                logger.info("Load pretrained VLM...")
            else:
                vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
                self.vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.dtype)
                self.vlm_model.to(device="cuda", dtype=self.dtype)
                logger.info("Initializing VLM from config...")

            for param in self.vlm_model.parameters():
                param.requires_grad = False
            logger.info("VLM parameters frozen")
            logger.info(f"VLM kept complete with {len(self.vlm_model.model.language_model.layers)} layers")

            vlm_dim = self.vlm_model.config.text_config.hidden_size
            vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
            vlm_num_kv_heads = getattr(self.vlm_model.config.text_config if hasattr(self.vlm_model.config, 'text_config') else self.vlm_model.config, 'num_key_value_heads', vlm_num_heads)
            vlm_num_hidden_layers  = self.vlm_model.config.text_config.num_hidden_layers
            vlm_head_dim = vlm_dim // vlm_num_heads
            logger.info(f"  VLM: {vlm_num_heads} Q heads, {vlm_num_kv_heads} KV heads × {vlm_head_dim} head_dim = {vlm_dim}D")
            vlm_config = {
                'hidden_size': vlm_dim,
                'num_attention_heads': vlm_num_heads,
                'num_key_value_heads': vlm_num_kv_heads,
                'head_dim': vlm_head_dim,
                'num_hidden_layers': vlm_num_hidden_layers,
            }

            logger.info("Initializing Understanding Expert...")
            und_config = UndExpertConfig(
                dim=config.und_expert_hidden_size,
                ffn_dim=config.und_expert_hidden_size * config.und_expert_ffn_dim_multiplier,
                num_layers=config.num_layers,
                vlm_input_dim=config.vlm_adapter_input_dim,
                vlm_projector_type=config.vlm_adapter_projector_type,
                eps=config.und_expert_norm_eps,
            )
            self.und_expert = UndExpert(und_config, wan_config, vlm_config)
            self.und_expert.to(device=self.device, dtype=self.dtype)
        else:
            logger.info("Understanding branch: disabled")

        # Set time embedding layers to float32 for numerical stability
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)

        # Pre-compute grid_sizes for training batch size
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        batch_size = config.batch_size
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)  # [batch_size, 3] - pre-expanded

        logger.info(f"Pre-computed grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}")

        # Initialize modular components
        self.video_module = VideoModule(
            self.video_model,
            self.dtype,
            self.device,
            self.grid_sizes,
            task_id=self.video_task_id,
            joint_attention_mode=config.joint_attention_mode,
            state_action_len=action_chunk_size_for_expert,
            register_len=num_registers,
        )
        if self.use_understanding:
            self.und_module = UndModule(self.vlm_model, self.und_expert, self.config, self.dtype, self.device)
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.dtype, self.device)

        # Quantile buckets are sampled independently for video and action.
        self.training_t_distribution = get_t_distribution(config)
        logger.info(
            "Training timestep distribution: method=%s scale=%s range=[%s, %s]",
            config.timestep_sample_method,
            config.sigmoid_scale,
            config.min_t,
            config.max_t,
        )

        # Flow-Matching scheduler for training (video branch only)
        try:
            self.fm_train_scheduler = FlowMatchScheduler(
                shift=config.video_train_schedule_shift,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode to build per-timestep weights (if used)
            self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            logger.info(
                "Initialized FlowMatchScheduler for training (video): shift=%s",
                config.video_train_schedule_shift,
            )
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler: {e}")

        # Flow-Matching scheduler for training (action branch)
        try:
            self.fm_train_scheduler_action = FlowMatchScheduler(
                shift=config.action_train_schedule_shift,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode for action as well
            self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
            logger.info(
                "Initialized FlowMatchScheduler for training (action): shift=%s",
                config.action_train_schedule_shift,
            )
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler for action: {e}")

        if config.tactile_enabled:
            self.fm_train_scheduler_tactile = FlowMatchScheduler(
                shift=config.tactile_train_schedule_shift,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000,
            )
            self.fm_train_scheduler_tactile.set_timesteps(
                num_inference_steps=1000, training=True
            )

        # Log parameter counts
        self.log_parameter_counts()

    def train(self, mode: bool = True):
        """Keep a frozen WAN in eval mode while training the Action branch."""
        super().train(mode)
        if getattr(self, "wan_frozen", False):
            self.video_model.wan_model.eval()
        tactile_codec = getattr(self, "tactile_codec", None)
        if tactile_codec is not None:
            tactile_codec.model.eval()
        return self

    def sample_video_task_ids(self, batch_size: int) -> torch.Tensor:
        """Sample one five-modal task id per sample for an entire V rollout."""
        if self.wan_task_probabilities is None:
            if self.video_task_id is None:
                raise RuntimeError("No WAN task probabilities or fixed task_id are configured")
            return torch.full(
                (batch_size,),
                int(self.video_task_id),
                device=self.device,
                dtype=torch.long,
            )
        return torch.multinomial(
            self.wan_task_probabilities.to(device=self.device),
            num_samples=batch_size,
            replacement=True,
        ).to(dtype=torch.long)

    def log_parameter_counts(self):
        """Log detailed parameter counts for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        video_params = sum(p.numel() for p in self.video_model.parameters())
        action_params = sum(p.numel() for p in self.action_expert.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters()) if self.vlm_model is not None else 0
        und_params = sum(p.numel() for p in self.und_expert.parameters()) if self.und_expert is not None else 0

        logger.info(f"Motus parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
        logger.info(f"  Video Model (WAN): {video_params / 1e9:.2f}B")
        logger.info(f"  Action Expert: {action_params / 1e6:.1f}M")
        if self.use_understanding:
            logger.info(f"  VLM (frozen): {vlm_params / 1e9:.2f}B")
            logger.info(f"  Und Expert: {und_params / 1e6:.1f}M")
        else:
            logger.info("  VLM (frozen): disabled")
            logger.info("  Und Expert: disabled")

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        # Handle directory path
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)

        # Load state dict
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['module']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

        # Return additional state
        additional_state = {k: v for k, v in checkpoint.items()
                          if k not in ['module', 'config']}
        return additional_state

    def load_warmup_weights(self, path: str, report_path: Optional[str] = None) -> Dict[str, Any]:
        """Load all shape-compatible weights from an existing Motus checkpoint.

        This is intended for WAM-VA warmup. It explicitly skips weights for
        modules that do not exist in the two-branch model, and records what was
        loaded or skipped instead of relying on silent strict=False behavior.
        """
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            possible_paths = [
                checkpoint_path / "mp_rank_00_model_states.pt",
                checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt",
            ]
            checkpoint_file = next((p for p in possible_paths if p.exists()), None)
            if checkpoint_file is None:
                raise FileNotFoundError(
                    "Warmup checkpoint not found. Tried:\n"
                    + "\n".join(f"  - {p}" for p in possible_paths)
                )
            path = str(checkpoint_file)
        elif not checkpoint_path.exists():
            raise FileNotFoundError(f"Warmup checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading warmup weights from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        source_state = checkpoint.get('module', checkpoint)
        target_state = self.state_dict()

        explicit_skip_prefixes = ("vlm_model.", "und_expert.", "und_module.", "video_model.vae.")
        loaded = {}
        skipped_explicit = []
        skipped_missing = []
        skipped_shape = []

        for key, tensor in source_state.items():
            normalized_key = key[7:] if key.startswith("module.") and key[7:] in target_state else key
            if normalized_key.startswith(explicit_skip_prefixes):
                skipped_explicit.append(key)
                continue
            if normalized_key not in target_state:
                skipped_missing.append(key)
                continue
            if tuple(target_state[normalized_key].shape) != tuple(tensor.shape):
                skipped_shape.append({
                    "key": key,
                    "source_shape": list(tensor.shape),
                    "target_shape": list(target_state[normalized_key].shape),
                })
                continue
            loaded[normalized_key] = tensor

        missing, unexpected = self.load_state_dict(loaded, strict=False)

        def count_prefixes(keys):
            counts = {}
            for item in keys:
                prefix = item.split(".", 1)[0]
                counts[prefix] = counts.get(prefix, 0) + 1
            return dict(sorted(counts.items()))

        report = {
            "checkpoint_path": str(path),
            "explicit_skip_prefixes": list(explicit_skip_prefixes),
            "loaded_count": len(loaded),
            "skipped_explicit_count": len(skipped_explicit),
            "skipped_missing_in_target_count": len(skipped_missing),
            "skipped_shape_mismatch_count": len(skipped_shape),
            "load_state_missing_count": len(missing),
            "load_state_unexpected_count": len(unexpected),
            "loaded_prefix_counts": count_prefixes(loaded.keys()),
            "skipped_explicit_prefix_counts": count_prefixes(skipped_explicit),
            "skipped_missing_in_target_prefix_counts": count_prefixes(skipped_missing),
            "loaded_keys_sample": list(loaded.keys())[:50],
            "skipped_explicit_sample": skipped_explicit[:50],
            "skipped_missing_in_target_sample": skipped_missing[:50],
            "skipped_shape_mismatch_sample": skipped_shape[:50],
        }
        logger.info(
            "Warmup load report: loaded=%d, skipped_explicit=%d, "
            "skipped_missing=%d, skipped_shape=%d",
            report["loaded_count"],
            report["skipped_explicit_count"],
            report["skipped_missing_in_target_count"],
            report["skipped_shape_mismatch_count"],
        )

        if report_path is not None:
            report_file = Path(report_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Wrote warmup load report to {report_file}")

        return report

    def load_va_checkpoint_for_vat(
        self,
        path: str,
        *,
        report_path: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        compute_sha256: bool = False,
        expected_wan_key_count: Optional[int] = None,
        expected_action_key_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Strictly initialize VAT WAN/Action weights from a trained VA checkpoint."""
        if self.config.training_mode != "finetune":
            raise ValueError("VA->VAT warm-start requires training_mode='finetune'")
        if self.use_understanding:
            raise ValueError("VA->VAT warm-start requires understanding disabled")
        if not self.config.tactile_enabled or self.tactile_expert is None:
            raise ValueError("VA->VAT warm-start requires the target tactile branch")
        return load_va_checkpoint_for_vat(
            self,
            path,
            report_path=report_path,
            expected_sha256=expected_sha256,
            compute_sha256=compute_sha256,
            expected_wan_key_count=expected_wan_key_count,
            expected_action_key_count=expected_action_key_count,
        )

    def load_pretrain_weights(self, path: str) -> None:
        """Load weights from a pretrain checkpoint when current mode is finetune.

        Skips layers that depend on state vs action-only differences:
          - action_expert.input_encoder.*
          - action_expert.decoder.*
        """
        if self.config.training_mode != 'finetune':
            raise ValueError("load_pretrain_weights should be called only in finetune mode")

        # Handle directory path - try two possible locations
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            # Try two possible paths
            possible_paths = [
                checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt",
                checkpoint_path / "mp_rank_00_model_states.pt",
            ]

            checkpoint_file = None
            for p in possible_paths:
                if p.exists():
                    checkpoint_file = p
                    logger.info(f"Found checkpoint: {checkpoint_file}")
                    break

            if checkpoint_file is None:
                raise FileNotFoundError(
                    f"Checkpoint not found. Tried:\n"
                    f"  - {possible_paths[0]}\n"
                    f"  - {possible_paths[1]}"
                )
            path = str(checkpoint_file)
        else:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading pretrain weights from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('module', checkpoint)

        filtered = {}
        for k, v in state_dict.items():
            if ('action_expert.input_encoder' in k or 'action_expert.decoder' in k):
                continue
            filtered[k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrain weights (filtered). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def load_stage2_warmstart_weights(
        self,
        path: str,
        report_path: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        compute_sha256: bool = True,
    ) -> Dict[str, Any]:
        """Strictly load the Stage 2 WAN and Action-DiT warm-start contract."""
        if self.config.training_mode != "finetune":
            raise ValueError("Stage 2 warm-start requires training_mode='finetune'")
        if self.use_understanding:
            raise ValueError("Stage 2 RGB multiview warm-start requires understanding disabled")
        if self.config.wan_task_embedding_path or self.config.wan_task_text_conditioning_path:
            raise ValueError("Stage 2 RGB multiview warm-start forbids WAN task conditioning")
        report = load_stage2_warmstart_weights(
            self,
            path,
            report_path=report_path,
            expected_sha256=expected_sha256,
            compute_sha256=compute_sha256,
        )
        logger.info(
            "Stage 2 warm-start loaded: WAN=%s Action-DiT=%s sha256=%s report=%s",
            report["loaded_wan_key_count"],
            report["loaded_action_dit_key_count"],
            report["checkpoint_sha256"],
            report_path,
        )
        return report

    def load_stage2_wan_only_weights(
        self,
        path: str,
        report_path: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        compute_sha256: bool = True,
    ) -> Dict[str, Any]:
        """Load official Stage 2 WAN while preserving the initialized Action Expert."""
        if self.config.training_mode != "finetune":
            raise ValueError("Stage 2 WAN-only warm-start requires training_mode=finetune")
        if self.use_understanding:
            raise ValueError("Stage 2 WAN-only warm-start requires understanding disabled")
        if self.config.wan_task_embedding_path or self.config.wan_task_text_conditioning_path:
            raise ValueError("Stage 2 WAN-only warm-start forbids WAN task conditioning")
        report = load_stage2_wan_only_weights(
            self,
            path,
            report_path=report_path,
            expected_sha256=expected_sha256,
            compute_sha256=compute_sha256,
        )
        logger.info(
            "Stage 2 WAN-only warm-start loaded: WAN=%s Action-DiT=%s report=%s",
            report["loaded_wan_key_count"],
            report["loaded_action_dit_key_count"],
            report_path,
        )
        return report

    def load_stage2_action_only_weights(
        self,
        path: str,
        report_path: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        compute_sha256: bool = True,
    ) -> Dict[str, Any]:
        """Load official Stage 2 Action-DiT while preserving the five-modal WAN."""
        if self.config.training_mode != "finetune":
            raise ValueError("Stage 2 Action-only warm-start requires training_mode='finetune'")
        if self.use_understanding:
            raise ValueError("Stage 2 Action-only warm-start requires understanding disabled")
        if not self.config.wan_task_text_conditioning_path:
            raise ValueError("Stage 2 Action-only warm-start requires five-modal T5 task conditioning")
        report = load_stage2_action_only_weights(
            self,
            path,
            report_path=report_path,
            expected_sha256=expected_sha256,
            compute_sha256=compute_sha256,
        )
        logger.info(
            "Stage 2 Action-only warm-start loaded: WAN=%s Action-DiT=%s sha256=%s report=%s",
            report["loaded_wan_key_count"],
            report["loaded_action_dit_key_count"],
            report["checkpoint_sha256"],
            report_path,
        )
        return report

    def forward(
        self,
        first_frame: torch.Tensor,
        video_frames: Optional[torch.Tensor] = None,
        state: torch.Tensor = None,
        actions: torch.Tensor = None,
        action_valid_mask: Optional[torch.Tensor] = None,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        return_dict: bool = True,
        video_latent: Optional[torch.Tensor] = None,
        condition_video_latent: Optional[torch.Tensor] = None,
        processed_t5_context: Optional[torch.Tensor] = None,
        video_timestep: Optional[torch.Tensor] = None,
        video_dt: Optional[torch.Tensor] = None,
        video_task_ids: Optional[torch.Tensor] = None,
        tactile_observed_source: Optional[torch.Tensor] = None,
        tactile_future_source: Optional[torch.Tensor] = None,
        tactile_grid_mask_source: Optional[torch.Tensor] = None,
        tactile_observed_frame_times: Optional[torch.Tensor] = None,
        tactile_future_query_times: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """DDP/DeepSpeed-compatible training forward."""
        if self.config.video_training_strategy == "frozen_rollout":
            return self.frozen_wan_rollout_substep(
                video_latent=video_latent,
                condition_video_latent=condition_video_latent,
                processed_t5_context=processed_t5_context,
                video_timestep=video_timestep,
                video_dt=video_dt,
                video_task_ids=video_task_ids,
                state=state,
                actions=actions,
                action_valid_mask=action_valid_mask,
                return_dict=return_dict,
            )
        if video_frames is None:
            raise ValueError("flow_matching training requires video_frames")
        return self.training_step(
            first_frame=first_frame,
            video_frames=video_frames,
            state=state,
            actions=actions,
            action_valid_mask=action_valid_mask,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
            video_task_ids=video_task_ids,
            tactile_observed_source=tactile_observed_source,
            tactile_future_source=tactile_future_source,
            tactile_grid_mask_source=tactile_grid_mask_source,
            tactile_observed_frame_times=tactile_observed_frame_times,
            tactile_future_query_times=tactile_future_query_times,
            return_dict=return_dict,
        )

    def initialize_frozen_video_rollout(
        self,
        first_frame: torch.Tensor,
        language_embeddings: torch.Tensor | List[torch.Tensor],
        video_task_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Create the pure-noise V state once; all returned tensors are graph-free."""
        if self.config.video_training_strategy != "frozen_rollout":
            raise RuntimeError("initialize_frozen_video_rollout requires frozen_rollout strategy")
        if self.use_understanding:
            raise RuntimeError("Frozen WAN rollout is defined only for the V-A architecture")
        first_frame = first_frame.to(device=self.device, dtype=self.dtype)
        batch_size = int(first_frame.shape[0])
        with torch.no_grad():
            first_frame_normalized = (first_frame * 2.0 - 1.0).unsqueeze(2)
            condition_video_latent = self.video_model.encode_video(first_frame_normalized)
            _, channels, _, height, width = condition_video_latent.shape
            latent_frames = 1 + self.config.num_video_frames // 4
            video_latent = torch.randn(
                (batch_size, channels, latent_frames, height, width),
                device=self.device,
                dtype=self.dtype,
            )
            video_latent[:, :, 0:1] = condition_video_latent
            processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)
            if video_task_ids is None:
                video_task_ids = self.sample_video_task_ids(batch_size)
            else:
                video_task_ids = normalize_wan_task_ids(
                    video_task_ids,
                    batch_size=batch_size,
                    num_tasks=5,
                    device=self.device,
                )
            video_timesteps = torch.linspace(
                1.0,
                0.0,
                self.config.video_rollout_steps + 1,
                device=self.device,
                dtype=self.dtype,
            )
        return {
            "video_latent": video_latent.detach(),
            "condition_video_latent": condition_video_latent.detach(),
            "processed_t5_context": processed_t5_context.detach(),
            "video_task_ids": video_task_ids.detach(),
            "video_timesteps": video_timesteps,
        }

    def _joint_video_action_velocity(
        self,
        *,
        video_latent: torch.Tensor,
        noisy_actions: torch.Tensor,
        state: Optional[torch.Tensor],
        processed_t5_context: torch.Tensor,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        video_task_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one complete V-A joint forward and return both flow velocities."""
        if self.use_understanding:
            raise RuntimeError("V-A rollout does not accept Understanding tokens")
        batch_size = int(video_latent.shape[0])
        video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(batch_size, -1, -1)
        else:
            registers = None
        if self.config.training_mode == "pretrain":
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            if state is None:
                raise ValueError("state is required for finetune V-A rollout")
            action_tokens = self.action_expert.input_encoder(
                state.unsqueeze(1).to(self.dtype),
                noisy_actions,
                registers,
            )

        video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
            video_timestep,
            video_tokens.shape[1],
            task_id=video_task_ids,
        )
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
            action_timestep,
            action_tokens.shape[1],
        )
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            for layer_idx in range(self.config.num_layers):
                video_adaln_modulation = self.video_module.compute_adaln_modulation(
                    video_adaln_params,
                    layer_idx,
                )
                action_adaln_modulation = self.action_module.compute_adaln_modulation(
                    action_adaln_params,
                    layer_idx,
                )
                video_tokens, action_tokens = self.video_module.process_video_action_joint_attention(
                    video_tokens,
                    action_tokens,
                    video_adaln_modulation,
                    action_adaln_modulation,
                    layer_idx,
                    self.action_expert.blocks[layer_idx],
                )
                video_tokens = self.video_module.process_cross_attention(
                    video_tokens,
                    video_adaln_params,
                    layer_idx,
                    processed_t5_context,
                )
                video_tokens = self.video_module.process_ffn(
                    video_tokens,
                    video_adaln_modulation,
                    layer_idx,
                )
                action_tokens = self.action_module.process_ffn(
                    action_tokens,
                    action_adaln_modulation,
                    layer_idx,
                )

            video_velocity = self.video_module.apply_output_head(
                video_tokens,
                video_head_time_emb,
            )
            action_velocity_full = self.action_expert.decoder(
                action_tokens,
                action_head_time_emb,
            )
            usable_length = action_velocity_full.shape[1] - self.action_expert.config.num_registers
            if self.config.training_mode == "pretrain":
                action_velocity = action_velocity_full[:, :usable_length, :]
            else:
                action_velocity = action_velocity_full[:, 1:usable_length, :]
        return video_velocity, action_velocity

    def _joint_video_action_tactile_velocity(
        self,
        *,
        video_latent: torch.Tensor,
        noisy_actions: torch.Tensor,
        tactile_latent: torch.Tensor,
        state: torch.Tensor,
        processed_t5_context: torch.Tensor,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        tactile_timestep: torch.Tensor,
        video_task_ids: Optional[torch.Tensor],
        tactile_anatomy_indices: Optional[torch.Tensor],
        tactile_token_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one V2.1 Full-Joint V-A-T flow-velocity evaluation."""
        if not self.config.tactile_enabled or self.tactile_expert is None:
            raise RuntimeError("V2.1 tactile velocity requested without tactile model")
        batch_size = int(video_latent.shape[0])
        video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
        registers = (
            self.action_expert.registers.expand(batch_size, -1, -1)
            if self.action_expert.registers is not None
            else None
        )
        action_tokens = self.action_expert.input_encoder(
            state.unsqueeze(1).to(self.dtype), noisy_actions, registers
        )
        tactile_tokens = self.tactile_expert.tokenizer(
            tactile_latent.to(self.dtype),
            anatomy_indices=tactile_anatomy_indices,
        )
        video_head_time, video_adaln = self.video_module.get_time_embedding(
            video_timestep, video_tokens.shape[1], task_id=video_task_ids
        )
        action_head_time, action_adaln = self.action_module.get_time_embedding(
            action_timestep, action_tokens.shape[1]
        )
        tactile_head_time, tactile_adaln = self.tactile_expert.get_time_embeddings(
            tactile_timestep.float(),
            queries_per_slice=int(tactile_latent.shape[2]),
        )
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            for layer_idx in range(self.config.num_layers):
                video_mod = self.video_module.compute_adaln_modulation(
                    video_adaln, layer_idx
                )
                action_mod = self.action_module.compute_adaln_modulation(
                    action_adaln, layer_idx
                )
                tactile_mod = self.tactile_expert.modulation(
                    self.tactile_expert.blocks[layer_idx], tactile_adaln
                )
                video_tokens, action_tokens, tactile_tokens = (
                    self.video_module.process_video_action_tactile_joint_attention(
                        video_tokens,
                        action_tokens,
                        tactile_tokens,
                        video_mod,
                        action_mod,
                        tactile_mod,
                        layer_idx,
                        self.action_expert.blocks[layer_idx],
                        self.tactile_expert.blocks[layer_idx],
                        tactile_token_valid=tactile_token_valid,
                    )
                )
                video_tokens = self.video_module.process_cross_attention(
                    video_tokens, video_adaln, layer_idx, processed_t5_context
                )
                video_tokens = self.video_module.process_ffn(
                    video_tokens, video_mod, layer_idx
                )
                action_tokens = self.action_module.process_ffn(
                    action_tokens, action_mod, layer_idx
                )
                tactile_tokens = self.tactile_expert.apply_ffn(
                    tactile_tokens,
                    self.tactile_expert.blocks[layer_idx],
                    tactile_mod,
                )
                if tactile_token_valid is not None:
                    tactile_tokens = tactile_tokens * tactile_token_valid.to(
                        tactile_tokens.dtype
                    ).unsqueeze(-1)
            video_velocity = self.video_module.apply_output_head(
                video_tokens, video_head_time
            )
            action_velocity_full = self.action_expert.decoder(
                action_tokens, action_head_time
            )
            usable_length = (
                action_velocity_full.shape[1]
                - self.action_expert.config.num_registers
            )
            action_velocity = action_velocity_full[:, 1:usable_length]
            tactile_velocity = self.tactile_expert.output_head(
                tactile_tokens, tactile_head_time
            )
        return video_velocity, action_velocity, tactile_velocity

    def frozen_wan_rollout_substep(
        self,
        *,
        video_latent: Optional[torch.Tensor],
        condition_video_latent: Optional[torch.Tensor],
        processed_t5_context: Optional[torch.Tensor],
        video_timestep: Optional[torch.Tensor],
        video_dt: Optional[torch.Tensor],
        video_task_ids: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        action_valid_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Train Action at an independent FM timestep while advancing V once."""
        required = {
            "video_latent": video_latent,
            "condition_video_latent": condition_video_latent,
            "processed_t5_context": processed_t5_context,
            "video_timestep": video_timestep,
            "video_dt": video_dt,
            "video_task_ids": video_task_ids,
            "actions": actions,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Frozen WAN rollout substep is missing: {missing}")
        batch_size = int(actions.shape[0])
        timestep_id_action = sample_scheduler_timestep_ids(
            self.fm_train_scheduler_action,
            self.training_t_distribution,
            batch_size,
        )
        action_timestep = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(
            dtype=self.dtype,
            device=self.device,
        )
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(
            dtype=self.dtype,
            device=self.device,
        ).view(batch_size, 1, 1)
        noisy_actions, action_target, state, action_valid_mask, action_loss_mask = (
            prepare_action_flow_batch(
                actions,
                state,
                action_valid_mask,
                sigma_action,
                dtype=self.dtype,
                masked_zero_padding=self.config.masked_zero_padding,
                pi05_zero_padding=self.config.pi05_zero_padding,
            )
        )

        scaled_video_timestep = video_timestep.expand(batch_size).to(
            device=self.device,
            dtype=self.dtype,
        ) * 1000

        video_velocity, action_velocity = self._joint_video_action_velocity(
            video_latent=video_latent,
            noisy_actions=noisy_actions,
            state=state,
            processed_t5_context=processed_t5_context,
            video_timestep=scaled_video_timestep,
            action_timestep=action_timestep,
            video_task_ids=video_task_ids,
        )
        action_loss = masked_mse(
            action_velocity,
            action_target,
            action_loss_mask,
            per_sample_mean=self.config.per_sample_valid_mean_loss,
        )
        total_loss = self.config.action_loss_weight * action_loss
        with torch.no_grad():
            next_video_latent = video_latent + video_velocity.detach() * video_dt
            next_video_latent[:, :, 0:1] = condition_video_latent
            next_video_latent = next_video_latent.detach()
        video_loss = action_loss.detach().new_zeros(())
        if not return_dict:
            return total_loss
        return {
            "total_loss": total_loss,
            "video_loss": video_loss,
            "action_loss": action_loss,
            "action_timestep_mean": sigma_action.detach().float().mean(),
            "next_video_latent": next_video_latent,
        }

    def training_step(
        self,
        first_frame: torch.Tensor,         # [B, C, H, W] - first frame
        video_frames: torch.Tensor,       # [B, num_frames, C, H, W] - target frames
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        action_valid_mask: Optional[torch.Tensor] = None,  # [B, action_dim]
        language_embeddings: Optional[List[torch.Tensor]] = None,  # Pre-encoded T5 embeddings for WAN
        vlm_inputs: Optional[List] = None,  # Complete VLM inputs from dataset
        video_task_ids: Optional[torch.Tensor] = None,
        tactile_observed_source: Optional[torch.Tensor] = None,
        tactile_future_source: Optional[torch.Tensor] = None,
        tactile_grid_mask_source: Optional[torch.Tensor] = None,
        tactile_observed_frame_times: Optional[torch.Tensor] = None,
        tactile_future_query_times: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        UniDiffuser training step with three modalities.

        Args:
            first_frame: First video frame for Teacher Forcing
            video_frames: Target video frames
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state
            actions: Target action sequence
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            return_dict: Whether to return detailed outputs

        Returns:
            Dictionary containing losses and metrics
        """
        B = video_frames.shape[0]
        task_text_features = getattr(self.video_model.wan_model, WAN_TASK_TEXT_FEATURES_KEY, None)
        task_embedding = getattr(self.video_model.wan_model, "task_embedding", None)
        num_tasks = (
            int(task_text_features.shape[0])
            if task_text_features is not None
            else int(task_embedding.num_embeddings) if task_embedding is not None else 0
        )
        if num_tasks:
            if video_task_ids is None:
                if self.video_task_id is None:
                    raise ValueError(
                        "Flow-matching with sampled task conditioning requires Dataset video_task_ids; "
                        "model-side sampling would decouple task_id from the supervised modality"
                    )
                video_task_ids = torch.full(
                    (B,), int(self.video_task_id), device=self.device, dtype=torch.long
                )
            else:
                video_task_ids = normalize_wan_task_ids(
                    video_task_ids,
                    batch_size=B,
                    num_tasks=num_tasks,
                    device=self.device,
                )
        elif video_task_ids is not None:
            raise ValueError("Dataset supplied video_task_ids but WAN task conditioning is disabled")

        # 1. Video pipeline
        # Normalize/format
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, num_frames, H, W]
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)  # [B, C, frames+1, H, W]

        # Encode video using VAE
        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))  # [B, 48, latent_frames, H', W']
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))  # [B, 48, 1, H', W']

        # Flow-Matching noise mixture
        timestep_id = sample_scheduler_timestep_ids(
            self.fm_train_scheduler,
            self.training_t_distribution,
            B,
        )
        # Scalar timesteps (0..num_train_timesteps) for time embedding
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for noise mixture
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        # Teacher Forcing on the first frame
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        # Flow-Matching target: noise - clean
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0

        # Latent to Tokens
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))

        # 2. Action pipeline
        timestep_id_action = sample_scheduler_timestep_ids(
            self.fm_train_scheduler_action,
            self.training_t_distribution,
            B,
        )
        # Discrete timesteps for time embedding (0..num_train_timesteps)
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for action noise mixture
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        noisy_actions, action_target, state, action_valid_mask, action_loss_mask = (
            prepare_action_flow_batch(
                actions,
                state,
                action_valid_mask,
                sigma_action,
                dtype=self.dtype,
                masked_zero_padding=self.config.masked_zero_padding,
                pi05_zero_padding=self.config.pi05_zero_padding,
            )
        )

        tactile_tokens = None
        tactile_target = None
        tactile_token_valid = None
        tactile_anatomy_indices = None
        tactile_valid_values = None
        sigma_tactile = None
        tactile_t_embed = None
        if self.config.tactile_enabled:
            tactile_inputs = {
                "tactile_observed_source": tactile_observed_source,
                "tactile_future_source": tactile_future_source,
                "tactile_observed_frame_times": tactile_observed_frame_times,
                "tactile_future_query_times": tactile_future_query_times,
            }
            missing = [name for name, value in tactile_inputs.items() if value is None]
            if missing:
                raise ValueError(f"V2.1 training batch is missing: {missing}")
            self.tactile_codec.assert_frozen()
            with torch.no_grad():
                clean_tactile, codec_aux = self.tactile_codec.encode_raw(
                    tactile_observed_source,
                    tactile_future_source,
                    grid_mask_source=tactile_grid_mask_source,
                    observed_frame_times=tactile_observed_frame_times,
                    future_query_times=tactile_future_query_times,
                )
                if self.config.tactile_codec_type == "v21_support_vae":
                    clean_tactile = self.tactile_codec.latent_normalizer.normalize(clean_tactile)
                    tactile_token_valid_3d = torch.ones(
                        clean_tactile.shape[:-1],
                        device=clean_tactile.device,
                        dtype=torch.bool,
                    )
                elif self.config.tactile_codec_type in (
                    "v3_anatomy",
                    "v4_uni_50k",
                ):
                    tactile_token_valid_3d = codec_aux
                else:
                    raise RuntimeError(
                        "Unsupported tactile codec validity contract: "
                        f"{self.config.tactile_codec_type}"
                    )
                clean_tactile = clean_tactile.to(device=self.device, dtype=self.dtype)
                tactile_token_valid_3d = tactile_token_valid_3d.to(
                    device=self.device, dtype=torch.bool
                )
                expected_slices = self.tactile_expert.config.latent_slices
                if clean_tactile.shape[1] != expected_slices:
                    raise ValueError(
                        "Frozen tactile codec/model slice mismatch: "
                        f"codec={clean_tactile.shape[1]}, expert={expected_slices}"
                    )
                if self.config.tactile_codec_type == "v3_anatomy":
                    clean_tactile, tactile_anatomy_indices = (
                        compact_homogeneous_v3_tactile_tokens(
                            clean_tactile,
                            tactile_token_valid_3d,
                            force_bimanual_layout=(
                                self.config.masked_zero_padding
                                or self.config.pi05_zero_padding
                            ),
                        )
                    )
                    tactile_token_valid_3d = tactile_token_valid_3d.index_select(2, tactile_anatomy_indices)
                    tactile_token_valid = (
                        self.tactile_expert.config.pack_validity(
                            tactile_token_valid_3d
                        )
                    )
                    tactile_valid_values = tactile_token_valid_3d.to(clean_tactile.dtype).unsqueeze(-1)
                    clean_tactile = clean_tactile * tactile_valid_values
                else:
                    tactile_token_valid = (
                        self.tactile_expert.config.pack_validity(
                            tactile_token_valid_3d
                        )
                    )
                    tactile_valid_values = tactile_token_valid_3d.to(
                        clean_tactile.dtype
                    ).unsqueeze(-1)
                    clean_tactile = clean_tactile * tactile_valid_values
            condition_slices = self.tactile_expert.config.condition_slices
            tactile_timestep_id = sample_scheduler_timestep_ids(
                self.fm_train_scheduler_tactile,
                self.training_t_distribution,
                B,
            )
            tactile_t_embed = self.fm_train_scheduler_tactile.timesteps[
                tactile_timestep_id
            ].to(dtype=self.dtype, device=self.device)
            sigma_tactile = self.fm_train_scheduler_tactile.sigmas[
                tactile_timestep_id
            ].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1)
            future_valid_values = tactile_valid_values[:, condition_slices:]
            tactile_noise = torch.randn_like(
                clean_tactile[:, condition_slices:], dtype=self.dtype
            )
            # V4-Uni's validity mask is part of the frozen codec contract.
            # It must never be disabled by the legacy pi0.5 zero-padding mode:
            # an absent hand is not an observed zero-force hand.
            preserve_codec_validity = (
                self.config.tactile_codec_type == "v4_uni_50k"
            )
            if not self.config.pi05_zero_padding or preserve_codec_validity:
                tactile_noise = tactile_noise * future_valid_values
            noisy_tactile_future = (
                clean_tactile[:, condition_slices:] * (1 - sigma_tactile)
                + tactile_noise * sigma_tactile
            )
            tactile_target = tactile_noise - clean_tactile[:, condition_slices:]
            tactile_latent = torch.cat(
                (clean_tactile[:, :condition_slices], noisy_tactile_future), dim=1
            )
            tactile_tokens = self.tactile_expert.tokenizer(
                tactile_latent, anatomy_indices=tactile_anatomy_indices
            )
            expected_tactile_tokens = (
                self.tactile_expert.config.latent_slices
                * self.tactile_expert.config.tokens_per_slice_for(
                    int(tactile_latent.shape[2])
                ),
                self.tactile_expert.config.hidden_size,
            )
            if tuple(tactile_tokens.shape[1:]) != expected_tactile_tokens:
                raise RuntimeError(
                    f"V2.1 tactile token contract failed: {tuple(tactile_tokens.shape)}"
                )

        # Encode Action Chunk with optional Registers
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)

        und_tokens = None
        if self.use_understanding:
            und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, seq_len, und_dim]

        # Time embeddings
        # Use scheduler-provided timesteps (0..num_train_timesteps) for WAN/action time embeddings
        video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
            video_t_embed,
            video_tokens.shape[1],
            task_id=video_task_ids,
        )
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_embed, action_tokens.shape[1])
        tactile_head_time_emb = None
        tactile_adaln_params = None
        if self.config.tactile_enabled:
            tactile_head_time_emb, tactile_adaln_params = (
                self.tactile_expert.get_time_embeddings(
                    tactile_t_embed.float(),
                    queries_per_slice=int(tactile_latent.shape[2]),
                )
            )

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. MoT forward
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            # Process through 30 layers - modality-grouped execution
            for layer_idx in range(self.config.num_layers):
                # Compute AdaLN modulation once per layer using pre-computed parameters
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)

                if self.config.tactile_enabled:
                    tactile_adaln_modulation = self.tactile_expert.modulation(
                        self.tactile_expert.blocks[layer_idx], tactile_adaln_params
                    )
                    video_tokens, action_tokens, tactile_tokens = (
                        self.video_module.process_video_action_tactile_joint_attention(
                            video_tokens,
                            action_tokens,
                            tactile_tokens,
                            video_adaln_modulation,
                            action_adaln_modulation,
                            tactile_adaln_modulation,
                            layer_idx,
                            self.action_expert.blocks[layer_idx],
                            self.tactile_expert.blocks[layer_idx],
                            tactile_token_valid=(
                                tactile_token_valid
                                if preserve_codec_validity
                                or not self.config.pi05_zero_padding
                                else None
                            ),
                        )
                    )
                elif self.use_understanding:
                    # Trimodal MoT: WAN + Action + Understanding Expert joint attention
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx,
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )
                else:
                    # Bimodal MoT: WAN + Action joint attention
                    video_tokens, action_tokens = self.video_module.process_video_action_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx,
                        self.action_expert.blocks[layer_idx]
                    )

                # WAN cross
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)

                # FFNs: WAN, Action, Understanding (each processes their own FFN)
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                if self.config.tactile_enabled:
                    tactile_tokens = self.tactile_expert.apply_ffn(
                        tactile_tokens,
                        self.tactile_expert.blocks[layer_idx],
                        tactile_adaln_modulation,
                    )
                    if tactile_token_valid is not None and (
                        preserve_codec_validity
                        or not self.config.pi05_zero_padding
                    ):
                        tactile_tokens = tactile_tokens * tactile_token_valid.to(
                            tactile_tokens.dtype
                        ).unsqueeze(-1)
                if self.use_understanding:
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)


            # 4. Heads + Losses
            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
            up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
            # Slice predicted actions depending on mode
            if self.config.training_mode == 'pretrain':
                action_pred = action_pred_full[:, :up_len, :]
            else:
                action_pred = action_pred_full[:, 1:up_len, :]

            # Video loss (mask the first frame)
            video_pred_masked = video_pred.clone()
            video_pred_masked[:, :, 0:1] = 0
            video_loss = torch.nn.functional.mse_loss(video_pred_masked, video_target, reduction='mean')

            # Action loss: ignore padded arm dimensions for single-arm samples.
            action_loss = masked_mse(
                action_pred,
                action_target,
                action_loss_mask,
                per_sample_mean=self.config.per_sample_valid_mean_loss,
            )
            with torch.no_grad():
                per_sample_error = (
                    action_pred.detach().float() - action_target.detach().float()
                ).square()
                per_sample_mask = (
                    torch.ones_like(per_sample_error)
                    if action_loss_mask is None
                    else action_loss_mask.to(dtype=per_sample_error.dtype).expand_as(per_sample_error)
                )
                action_loss_per_sample = (
                    (per_sample_error * per_sample_mask).sum(dim=(1, 2))
                    / per_sample_mask.sum(dim=(1, 2)).clamp_min(1)
                )
            tactile_loss = action_loss.new_zeros(())
            if self.config.tactile_enabled:
                tactile_pred = self.tactile_expert.output_head(
                    tactile_tokens, tactile_head_time_emb
                )
                tactile_loss = masked_mse(
                    tactile_pred[:, condition_slices:],
                    tactile_target,
                    (
                        future_valid_values
                        if preserve_codec_validity
                        or not self.config.pi05_zero_padding
                        else None
                    ),
                    per_sample_mean=self.config.per_sample_valid_mean_loss,
                )

        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss +
            self.config.tactile_loss_weight * tactile_loss
        )

        if return_dict:
            result = {
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
                'action_loss_per_sample': action_loss_per_sample,
                'tactile_loss': tactile_loss,
                'video_timestep_mean': sigma.float().mean().item(),
                'action_timestep_mean': sigma_action.float().mean().item(),
            }
            if sigma_tactile is not None:
                result['tactile_timestep_mean'] = sigma_tactile.float().mean().item()
            if video_task_ids is not None:
                task_counts = torch.bincount(video_task_ids, minlength=num_tasks)
                for task_id, count in enumerate(task_counts):
                    result[f'video_task_{task_id}_count'] = count.detach().float()
            return result

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        video_task_ids: Optional[torch.Tensor] = None,
        video_schedule_shift: float = 1.0,
        action_schedule_shift: float = 1.0,
        action_valid_mask: Optional[torch.Tensor] = None,
        tactile_observed_source: Optional[torch.Tensor] = None,
        tactile_grid_mask_source: Optional[torch.Tensor] = None,
        tactile_observed_frame_times: Optional[torch.Tensor] = None,
        tactile_future_query_times: Optional[torch.Tensor] = None,
        tactile_schedule_shift: float = 1.0,
        decode_video: bool = True,
        attention_capture=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.

        Args:
            first_frame: Initial frame [B, C, H, W]
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            video_schedule_shift: Flow-matching sigma shift for the Video branch
            action_schedule_shift: Flow-matching sigma shift for the Action branch

        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]
        self.video_module.set_attention_capture(attention_capture, step=0)

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        if self.config.training_mode != 'pretrain':
            if state is None:
                raise ValueError("state is required for finetune inference mode")
            state = state.to(self.device).to(self.dtype)
        inference_action_mask = None
        if self.config.masked_zero_padding:
            if action_valid_mask is None:
                action_valid_mask = torch.ones(
                    (B, self.config.action_dim), device=self.device, dtype=torch.bool
                )
                # Backward-compatible fallback for direct callers: all
                # DexJoCo single-arm entry points zero normalized state[22:].
                if state is not None and self.config.action_dim == 44:
                    single = state[..., 22:].eq(0).all(dim=-1)
                    action_valid_mask[single, 22:] = False
            else:
                action_valid_mask = torch.as_tensor(
                    action_valid_mask, device=self.device, dtype=torch.bool
                )
                if tuple(action_valid_mask.shape) != (B, self.config.action_dim):
                    raise ValueError(
                        "inference action_valid_mask must be [B, action_dim], got "
                        f"{tuple(action_valid_mask.shape)}"
                    )
            inference_action_mask = action_valid_mask[:, None, :].to(self.dtype)
            if state is not None and state.shape[-1] == self.config.action_dim:
                state = state * action_valid_mask.to(state.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)
        if video_task_ids is not None:
            num_tasks = int(
                getattr(
                    self.video_model.wan_model,
                    "num_task_text_features",
                    getattr(getattr(self.video_model.wan_model, "task_embedding", None), "num_embeddings", 0),
                )
            )
            video_task_ids = normalize_wan_task_ids(
                video_task_ids,
                batch_size=B,
                num_tasks=num_tasks,
                device=self.device,
            )
        elif hasattr(self.video_model.wan_model, WAN_TASK_TEXT_FEATURES_KEY):
            video_task_ids = self.sample_video_task_ids(B)

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)
        if inference_action_mask is not None:
            action_latent = action_latent * inference_action_mask
        tactile_observed = None
        tactile_future = None
        tactile_token_valid = None
        tactile_anatomy_indices = None
        tactile_future_valid_values = None
        if self.config.tactile_enabled:
            tactile_inputs = {
                "tactile_observed_source": tactile_observed_source,
                "tactile_observed_frame_times": tactile_observed_frame_times,
                "tactile_future_query_times": tactile_future_query_times,
            }
            missing = [name for name, value in tactile_inputs.items() if value is None]
            if missing:
                raise ValueError(f"V2.1 inference is missing: {missing}")
            tactile_observed, codec_aux = self.tactile_codec.encode_condition(
                tactile_observed_source,
                grid_mask_source=tactile_grid_mask_source,
                observed_frame_times=tactile_observed_frame_times,
                future_query_times=tactile_future_query_times,
            )
            cfg = self.tactile_expert.config
            if self.config.tactile_codec_type in (
                "v3_anatomy",
                "v4_uni_50k",
            ):
                tactile_observed_valid = codec_aux
            elif self.config.tactile_codec_type == "v21_support_vae":
                tactile_observed_valid = torch.ones(
                    tactile_observed.shape[:-1],
                    device=tactile_observed.device,
                    dtype=torch.bool,
                )
            else:
                raise RuntimeError(
                    "Unsupported tactile codec validity contract: "
                    f"{self.config.tactile_codec_type}"
                )
            tactile_observed = tactile_observed.to(device=self.device, dtype=self.dtype)
            tactile_observed_valid = tactile_observed_valid.to(
                device=self.device, dtype=torch.bool
            )
            if self.config.tactile_codec_type == "v3_anatomy":
                tactile_observed, tactile_anatomy_indices = (
                    compact_homogeneous_v3_tactile_tokens(
                        tactile_observed,
                        tactile_observed_valid,
                        force_bimanual_layout=(
                            self.config.masked_zero_padding
                            or self.config.pi05_zero_padding
                        ),
                    )
                )
                query_count = int(tactile_observed.shape[2])
                if not (
                    self.config.masked_zero_padding or self.config.pi05_zero_padding
                ):
                    tactile_future_valid_values = torch.ones(
                        (B, cfg.future_slices, query_count, 1),
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    tactile_observed_valid = tactile_observed_valid.index_select(
                        2, tactile_anatomy_indices
                    )
                    anatomy_valid = tactile_observed_valid[:, -1:]
                    tactile_future_valid = anatomy_valid.expand(
                        -1, cfg.future_slices, -1
                    )
                    tactile_future_valid_values = tactile_future_valid.to(
                        dtype=self.dtype
                    ).unsqueeze(-1)
                    tactile_observed = tactile_observed * tactile_observed_valid.to(
                        dtype=self.dtype
                    ).unsqueeze(-1)
                    if self.config.pi05_zero_padding:
                        tactile_future_valid_values = torch.ones_like(
                            tactile_future_valid_values
                        )
                    else:
                        tactile_token_valid = cfg.pack_validity(
                            torch.cat(
                                (tactile_observed_valid, tactile_future_valid),
                                dim=1,
                            )
                        )
            else:
                anatomy_valid = tactile_observed_valid[:, -1:]
                tactile_future_valid = anatomy_valid.expand(
                    -1, cfg.future_slices, -1
                )
                tactile_token_valid = cfg.pack_validity(
                    torch.cat(
                        (tactile_observed_valid, tactile_future_valid), dim=1
                    )
                )
                tactile_future_valid_values = tactile_future_valid.to(
                    dtype=self.dtype
                ).unsqueeze(-1)
                tactile_observed = tactile_observed * tactile_observed_valid.to(
                    dtype=self.dtype
                ).unsqueeze(-1)
            tactile_future = torch.randn(
                (
                    B,
                    cfg.future_slices,
                    int(tactile_observed.shape[2]),
                    cfg.latent_dim,
                ),
                device=self.device,
                dtype=self.dtype,
            ) * tactile_future_valid_values

        # 2. Optional Understanding Expert features and T5 context
        und_tokens = None
        if self.use_understanding:
            und_tokens = self.und_module.extract_und_features(vlm_inputs)

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop: independently shifted V/A sigmas from noise to clean.
        video_sigmas = build_flowmatch_sigma_schedule(
            num_inference_steps,
            video_schedule_shift,
            device=self.device,
            dtype=self.dtype,
        )
        action_sigmas = build_flowmatch_sigma_schedule(
            num_inference_steps,
            action_schedule_shift,
            device=self.device,
            dtype=self.dtype,
        )
        tactile_sigmas = (
            build_flowmatch_sigma_schedule(
                num_inference_steps,
                tactile_schedule_shift,
                device=self.device,
                dtype=self.dtype,
            )
            if self.config.tactile_enabled
            else None
        )
        for i in range(num_inference_steps):
            self.video_module.attention_capture_step = int(i)
            video_t = video_sigmas[i]
            action_t = action_sigmas[i]
            video_dt = video_sigmas[i + 1] - video_t
            action_dt = action_sigmas[i + 1] - action_t
            video_t_scaled = (video_t * 1000).expand(B).to(self.dtype)
            action_t_scaled = (action_t * 1000).expand(B).to(self.dtype)

            if self.config.tactile_enabled:
                tactile_t = tactile_sigmas[i]
                tactile_dt = tactile_sigmas[i + 1] - tactile_t
                tactile_t_scaled = (tactile_t * 1000).expand(B).to(self.dtype)
                video_velocity, action_velocity, tactile_velocity = (
                    self._joint_video_action_tactile_velocity(
                        video_latent=video_latent,
                        noisy_actions=action_latent,
                        tactile_latent=torch.cat(
                            (tactile_observed, tactile_future), dim=1
                        ),
                        state=state,
                        processed_t5_context=processed_t5_context,
                        video_timestep=video_t_scaled,
                        action_timestep=action_t_scaled,
                        tactile_timestep=tactile_t_scaled,
                        video_task_ids=video_task_ids,
                        tactile_anatomy_indices=tactile_anatomy_indices,
                        tactile_token_valid=tactile_token_valid,
                    )
                )
                video_latent = video_latent + video_velocity * video_dt
                action_latent = action_latent + action_velocity * action_dt
                if inference_action_mask is not None:
                    action_latent = action_latent * inference_action_mask
                tactile_future = (
                    tactile_future
                    + tactile_velocity[:, cfg.condition_slices:] * tactile_dt
                ) * tactile_future_valid_values
                video_latent[:, :, 0:1] = condition_frame_latent
                continue

            if not self.use_understanding:
                video_velocity, action_velocity = self._joint_video_action_velocity(
                    video_latent=video_latent,
                    noisy_actions=action_latent,
                    state=state,
                    processed_t5_context=processed_t5_context,
                    video_timestep=video_t_scaled,
                    action_timestep=action_t_scaled,
                    video_task_ids=video_task_ids,
                )
                video_latent = video_latent + video_velocity * video_dt
                action_latent = action_latent + action_velocity * action_dt
                if inference_action_mask is not None:
                    action_latent = action_latent * inference_action_mask
                video_latent[:, :, 0:1] = condition_frame_latent
                continue

            # Tokens with Registers
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            if self.config.training_mode == 'pretrain':
                registers = None
                action_tokens = self.action_expert.input_encoder(None, action_latent, registers)
            else:
                state_tokens = state.unsqueeze(1).to(self.dtype)
                registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            if self.use_understanding:
                # Re-extract understanding tokens per step to keep parity with the original path.
                und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, seq_len, und_dim]


            # Trimodal MoT forward - joint denoising for WAN, Action, Understanding
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                    video_t_scaled,
                    video_tokens.shape[1],
                    task_id=video_task_ids,
                )
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])

                # Process through all layers - trimodal denoising of WAN, Action, Understanding
                for layer_idx in range(self.config.num_layers):
                    # Compute AdaLN modulation using pre-computed parameters
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)

                    if self.use_understanding:
                        # Trimodal joint attention: WAN + Action + Understanding
                        video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                            video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx,
                            self.action_expert.blocks[layer_idx],
                            und_tokens, self.und_expert.blocks[layer_idx]
                        )
                    else:
                        # Bimodal joint attention: WAN + Action
                        video_tokens, action_tokens = self.video_module.process_video_action_joint_attention(
                            video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx,
                            self.action_expert.blocks[layer_idx]
                        )

                    # WAN cross-attention with T5 embeddings
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )

                    # FFNs: WAN, Action, Understanding
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    if self.use_understanding:
                        und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Heads (velocities)
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                if self.config.training_mode == 'pretrain':
                    action_velocity = action_pred_full
                else:
                    # Extract middle action chunk (skip first state token and last register tokens)
                    action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]

                # Euler integration
                video_latent = video_latent + video_velocity * video_dt
                action_latent = action_latent + action_velocity * action_dt
                if inference_action_mask is not None:
                    action_latent = action_latent * inference_action_mask

                # Teacher Forcing
                video_latent[:, :, 0:1] = condition_frame_latent

        # 4. Decode outputs only when the caller consumes future RGB. Policy
        # and action-diagnostic callers use the jointly denoised Action branch
        # but do not need a costly VAE decode of the final Video latent.
        predicted_frames = None
        if decode_video:
            with torch.no_grad():
                decoded_frames = self.video_model.decode_video(video_latent)
                predicted_frames = decoded_frames[:, :, 1:]  # Skip condition
                predicted_frames = (predicted_frames + 1.0) / 2.0
                predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [B, action_chunk_size, action_dim]
        if inference_action_mask is not None:
            predicted_actions = predicted_actions * inference_action_mask.float()
        self.video_module.set_attention_capture(None)

        return predicted_frames, predicted_actions

    # Alternative inference (DPM++ solver)
    '''
    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        solver: Optional[str] = None,
        shift: Optional[float] = None,
        seed: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction with dpm++ solver.

        Args:
            first_frame: Initial frame [1, C, H, W] - batch size must be 1 for inference
            state: Initial robot state [1, state_dim]
            num_inference_steps: Number of denoising steps (default: 50)
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            vlm_inputs: VLM inputs for understanding expert
            solver: Solver type ("dpm++"), defaults to config.inference_solver
            shift: Noise schedule shift, defaults to config.inference_shift
            seed: Random seed for reproducible generation (-1 for random)

        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        # Move inputs to device
        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # Use config defaults if not specified
        if solver is None:
            solver = self.config.inference_solver
        if shift is None:
            shift = self.config.inference_shift

        # Set random seed if specified
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 1. Encode condition frame and initialize latents
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [1, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [1, 48, 1, H', W']

        # Initialize video latent with noise - squeeze batch dimension for WAN format
        _, C_latent, _, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn(
            (C_latent, num_total_latent_frames, H_latent, W_latent),
            device=self.device,
            dtype=torch.float32,  # Use float32 for sampling numerical stability
            generator=generator
        )
        # Set first frame as condition (teacher forcing)
        video_latent[:, 0:1] = condition_frame_latent.squeeze(0).float()

        # Initialize action latent with noise
        action_latent = torch.randn(
            (1, self.config.action_chunk_size, self.config.action_dim),
            device=self.device,
            dtype=torch.float32,
            generator=generator
        )

        # 2. Prepare understanding features and T5 context (compute once, reuse for all steps)
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Setup flow-matching schedulers (separate for video and action due to different tensor shapes)
        if solver == "dpm++":
            # Video scheduler
            video_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1.0,  # Base shift is 1.0
                use_dynamic_shifting=False
            )
            # Action scheduler (independent instance)
            action_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1.0,
                use_dynamic_shifting=False
            )
            # Get custom sigmas with shift parameter
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                video_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
            # Set same timesteps for action scheduler
            _, _ = retrieve_timesteps(
                action_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(f"Solver '{solver}' not implemented. Currently only 'dpm++' is supported.")

        # 4. Denoising loop with flow-matching solver
        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                # Prepare model inputs (add batch dimension back)
                video_latent_input = video_latent.unsqueeze(0).to(self.dtype)  # [1, 48, T, H, W]
                action_latent_input = action_latent.to(self.dtype)  # [1, chunk_size, action_dim]

                # Prepare tokens
                video_tokens = self.video_module.prepare_input(video_latent_input)
                state_tokens = state.unsqueeze(1)
                registers = self.action_expert.registers.expand(1, -1, -1)
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent_input, registers)
                und_tokens = self.und_module.extract_und_features(vlm_inputs)

                # Model forward pass
                with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                    # Time embeddings (t is in [0, 1000] from scheduler)
                    video_t_scaled = t.expand(1).to(self.dtype)
                    action_t_scaled = t.expand(1).to(self.dtype)
                    video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                        video_t_scaled, video_tokens.shape[1]
                    )
                    action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
                        action_t_scaled, action_tokens.shape[1]
                    )

                    # Process through all layers - trimodal joint denoising
                    for layer_idx in range(self.config.num_layers):
                        video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                        action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)

                        # Trimodal joint attention
                        video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                            video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx,
                            self.action_expert.blocks[layer_idx],
                            und_tokens, self.und_expert.blocks[layer_idx]
                        )

                        # WAN cross-attention with T5
                        video_tokens = self.video_module.process_cross_attention(
                            video_tokens, video_adaln_params, layer_idx, processed_t5_context
                        )

                        # FFNs for each modality
                        video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                        action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                        und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                    # Prediction heads (predict velocity for flow-matching)
                    video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)  # [1, 48, T, H, W]
                    action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                    action_pred = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]  # [1, chunk_size, action_dim]

                # Update latents using separate schedulers (video and action have different tensor shapes)
                # Video: squeeze batch dim, call video_scheduler, squeeze back
                video_latent = video_scheduler.step(
                    video_pred.squeeze(0).unsqueeze(0),  # Add dummy batch dim for scheduler
                    t,
                    video_latent.unsqueeze(0),  # Add dummy batch dim for scheduler
                    return_dict=False,
                    generator=generator
                )[0].squeeze(0)  # Remove dummy batch dim

                # Action: directly use 3D tensor [1, chunk_size, action_dim] with action_scheduler
                # DPM-Solver doesn't require specific dimensions, just consistency
                action_latent = action_scheduler.step(
                    action_pred,  # [1, chunk_size, action_dim]
                    t,
                    action_latent,  # [1, chunk_size, action_dim]
                    return_dict=False,
                    generator=generator
                )[0]

                # Teacher forcing: keep first frame as condition
                video_latent[:, 0:1] = condition_frame_latent.squeeze(0).float()

        # 5. Decode final outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent.unsqueeze(0).to(self.dtype))  # Add batch dim back
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [1, chunk_size, action_dim]

        return predicted_frames, predicted_actions
    '''

    # Alternative inference (UniPC solver)
    '''
    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.

        Args:
            first_frame: Initial frame [B, C, H, W]
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model

        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        if self.config.training_mode != 'pretrain':
            state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. Understanding Expert features and T5 context
        # Extract understanding features from VLM
        und_tokens = self.und_module.extract_und_features(vlm_inputs)

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop: use FlowUniPCMultistepScheduler for video and action latents
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        scheduler.set_timesteps(num_inference_steps, device=self.device, shift=1.0)
        # Use a separate scheduler instance for the action branch to avoid shared internal state
        action_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        action_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=1.0)
        timesteps = scheduler.timesteps  # int64 on device

        for t in timesteps:
            # Tokens (with optional registers)
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
                registers = self.action_expert.registers.expand(B, -1, -1)
            else:
                registers = None
            if self.config.training_mode == 'pretrain':
                action_tokens = self.action_expert.input_encoder(None, action_latent, registers)
            else:
                state_tokens = state.unsqueeze(1).to(self.dtype)
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            # Re-extract understanding features per step (keeps alignment with current pipeline)
            und_tokens = self.und_module.extract_und_features(vlm_inputs)

            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings: use the current discrete t (0..num_train_timesteps)
                t_scalar = t.to(self.dtype).repeat(B)
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(t_scalar, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(t_scalar, action_tokens.shape[1])

                # Layer stack for joint denoising
                for layer_idx in range(self.config.num_layers):
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx,
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )
                    # WAN cross
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Predict velocities (video and action) and take scheduler steps
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_velocity_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                up_len = action_velocity_full.shape[1] - self.action_expert.config.num_registers
                if self.config.training_mode == 'pretrain':
                    action_velocity = action_velocity_full[:, :up_len, :]
                else:
                    action_velocity = action_velocity_full[:, 1:up_len, :]

                # Scheduler steps
                video_latent = scheduler.step(model_output=video_velocity, timestep=t, sample=video_latent, return_dict=False)[0]
                # Teacher Forcing on the first frame (video)
                video_latent[:, :, 0:1] = condition_frame_latent
                action_latent = action_scheduler.step(model_output=action_velocity, timestep=t, sample=action_latent, return_dict=False)[0]

        # 4. Decode outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [B, action_chunk_size, 14]

        return predicted_frames, predicted_actions
    '''


def test_motus():
    """Test the complete model."""
    print("Testing Motus...")

    config = MotusConfig()

    try:
        model = Motus(config)
        print("Model created successfully")

        # Test parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")

    except Exception as e:
        print(f"Model creation failed: {e}")
        print("This is expected without actual pretrained weights")

if __name__ == "__main__":
    test_motus()
