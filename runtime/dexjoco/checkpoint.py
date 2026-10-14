"""Single-device WTAM construction and strict checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import torch
from omegaconf import OmegaConf

from runtime.dexjoco.models.motus import Motus, MotusConfig
from runtime.dexjoco.models.wtam_v25.frozen_anatomy_tactile_codec import (
    _register_numpy_pickle_compatibility,
)


def build_motus_config(config: Any) -> MotusConfig:
    """Build the same model contract as ``train.create_model_and_optimizer``."""
    tactile = config.model.get("tactile", {})
    tactile_enabled = bool(tactile.get("enabled", False))
    return MotusConfig(
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        wan_expected_num_tensors=config.model.wan.get("expected_num_tensors"),
        wan_task_embedding_path=config.model.wan.get("task_embedding_path"),
        wan_task_text_conditioning_path=config.model.wan.get("task_text_conditioning_path"),
        wan_task_id=config.model.wan.get("task_id"),
        wan_task_probabilities=config.model.wan.get("task_probabilities"),
        wan_frozen=bool(config.model.wan.get("frozen", False)),
        video_training_strategy=config.model.wan.get("training_strategy", "flow_matching"),
        video_rollout_steps=config.model.wan.get("rollout_steps", 10),
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        action_expert_dim=config.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=config.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=config.model.action_expert.norm_eps,
        und_expert_hidden_size=config.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=config.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=config.model.und_expert.norm_eps,
        vlm_adapter_input_dim=config.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=config.model.und_expert.vlm.projector_type,
        global_downsample_rate=config.common.global_downsample_rate,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        action_chunk_size=config.common.get(
            "action_chunk_size",
            config.common.num_video_frames * config.common.video_action_freq_ratio,
        ),
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=1,
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        tactile_loss_weight=config.model.loss_weights.get("tactile_loss_weight", 0.1),
        per_sample_valid_mean_loss=bool(
            config.model.loss_weights.get("per_sample_valid_mean_loss", False)
        ),
        masked_zero_padding=bool(
            config.model.get("padding_contract", {}).get("masked_zero_padding", False)
        ),
        pi05_zero_padding=bool(
            config.model.get("padding_contract", {}).get("pi05_zero_padding", False)
        ),
        timestep_sample_method=config.model.time_distribution.get("timestep_sample_method", "uniform"),
        sigmoid_scale=config.model.time_distribution.get("sigmoid_scale", 1.0),
        min_t=config.model.time_distribution.get("min_t", 0.0),
        max_t=config.model.time_distribution.get("max_t", 1.0),
        video_train_schedule_shift=config.model.time_distribution.get("video_train_schedule_shift", 5.0),
        action_train_schedule_shift=config.model.time_distribution.get("action_train_schedule_shift", 5.0),
        tactile_train_schedule_shift=config.model.time_distribution.get("tactile_train_schedule_shift", 5.0),
        joint_attention_mode=config.model.get("joint_attention_mode", "full_bidirectional"),
        training_mode=getattr(config, "training_mode", "finetune"),
        load_pretrained_backbones=config.model.get("load_pretrained_backbones"),
        use_understanding=bool(config.model.get("use_understanding", True)),
        tactile_enabled=tactile_enabled,
        tactile_vae_checkpoint_path=tactile.get("checkpoint_path"),
        tactile_vae_expected_sha256=tactile.get("expected_sha256"),
        tactile_codec_type=tactile.get("codec_type", "v21_support_vae"),
        tactile_codec_source_root=tactile.get("source_root"),
        tactile_codec_source_revision=tactile.get("source_revision"),
        tactile_codec_source_files_sha256=(
            OmegaConf.to_container(tactile.get("source_files_sha256"), resolve=True)
            if tactile.get("source_files_sha256") is not None
            else None
        ),
        tactile_codec_expected_step=tactile.get("expected_step"),
        tactile_codec_normalization_group=tactile.get("normalization_group"),
        tactile_dexjoco_channel_scale=(
            tuple(tactile.get("dexjoco_channel_scale"))
            if tactile.get("dexjoco_channel_scale", None) is not None else None
        ),
        tactile_latent_stats_path=tactile.get("latent_stats_path"),
        tactile_latent_stats_expected_sha256=tactile.get("expected_latent_stats_sha256"),
        tactile_support_calibration_path=tactile.get("support_calibration_path"),
        tactile_support_calibration_expected_sha256=tactile.get("expected_support_calibration_sha256"),
        tactile_latent_stats_task_name=tactile.get(
            "latent_stats_task_name", "robotwin_vat_supportvae_sparse_multitask_v1"
        ),
        tactile_expert_config=(
            OmegaConf.to_container(config.model.tactile_expert, resolve=True)
            if tactile_enabled
            else None
        ),
    )


def resolve_model_state(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    candidates = (
        path,
        path / "mp_rank_00_model_states.pt",
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("checkpoint model state not found: " + str(path))


def validate_checkpoint_contract(config: Any, checkpoint_file: Path) -> None:
    """Reject shape-compatible policies trained with a different temporal/image recipe."""
    metadata = next((directory / "config.json" for directory in (
        checkpoint_file.parent, checkpoint_file.parent.parent
    ) if (directory / "config.json").is_file()), None)
    if metadata is None:
        return  # Older exports contain only the model state.
    saved = json.loads(metadata.read_text())
    for key in ("action_dim", "state_dim", "action_chunk_size", "num_video_frames",
                "video_height", "video_width", "global_downsample_rate", "video_action_freq_ratio"):
        value = saved.get("common", {}).get(key)
        if value is not None and int(value) != int(config.common[key]):
            raise ValueError(f"Checkpoint {key}={value} differs from config {config.common[key]}")
    mode = saved.get("joint_attention_mode", saved.get("model", {}).get("joint_attention_mode"))
    if mode is not None and mode != config.model.joint_attention_mode:
        raise ValueError(f"Checkpoint attention mode {mode} differs from config")


def load_single_device_model(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cuda:0",
) -> tuple[Motus, Any]:
    config = OmegaConf.load(config_path)
    checkpoint_file = resolve_model_state(checkpoint)
    validate_checkpoint_contract(config, checkpoint_file)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    config.model.load_pretrained_backbones = False
    model = Motus(build_motus_config(config))
    _register_numpy_pickle_compatibility()
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state = payload.get("module", payload)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, config
