#!/usr/bin/env python3
"""
Evaluation utilities for Motus.
Implements inference sampling and metrics computation for validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Suppress matplotlib font manager debug messages
matplotlib.set_loglevel("WARNING")
from PIL import Image
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def _frame_array(frames: torch.Tensor, batch_size: int) -> np.ndarray:
    values = frames[:batch_size].detach().cpu().float()
    if values.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W] video frames, got {tuple(values.shape)}")
    values = values.permute(0, 1, 3, 4, 2).numpy()
    if values.shape[-1] == 1:
        values = np.repeat(values, 3, axis=-1)
    if values.shape[-1] != 3:
        raise ValueError(f"Visualization requires 1 or 3 channels, got {values.shape[-1]}")
    return np.clip(values, 0, 1)


def create_video_sample_panel(
    predicted_frames: torch.Tensor,
    ground_truth_frames: torch.Tensor,
    *,
    sample_index: int = 0,
) -> Image.Image:
    """Render one sample as two adjacent rows: GT on top, Pred below."""
    if predicted_frames.ndim == 4:
        predicted_frames = predicted_frames.unsqueeze(0)
    if ground_truth_frames.ndim == 4:
        ground_truth_frames = ground_truth_frames.unsqueeze(0)
    if predicted_frames.shape[:2] != ground_truth_frames.shape[:2]:
        raise ValueError(
            "Predicted and GT frame counts must match: "
            f"pred={tuple(predicted_frames.shape)} gt={tuple(ground_truth_frames.shape)}"
        )
    pred_np = _frame_array(predicted_frames, 1)[0]
    gt_np = _frame_array(ground_truth_frames, 1)[0]
    num_frames = pred_np.shape[0]
    fig, axes = plt.subplots(
        2,
        num_frames,
        figsize=(max(3.0, 2.4 * num_frames), 5.0),
        squeeze=False,
    )
    for t in range(num_frames):
        axes[0, t].imshow(gt_np[t])
        axes[1, t].imshow(pred_np[t])
        axes[0, t].set_title(f"t={t + 1}")
        for row, label in ((0, "GT"), (1, "Pred")):
            axes[row, t].axis("off")
        axes[0, 0].set_ylabel("GT", rotation=0, labelpad=24, va="center", fontsize=11)
        axes[1, 0].set_ylabel("Pred", rotation=0, labelpad=24, va="center", fontsize=11)
    fig.suptitle(f"Sample {sample_index}: Ground Truth / Prediction", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(image)


def create_video_grid(predicted_frames: torch.Tensor, ground_truth_frames: torch.Tensor,
                     num_samples: int = 4) -> Image.Image:
    """
    Create a grid visualization comparing predicted and ground truth video frames.

    Args:
        predicted_frames: (B, T, C, H, W) predicted video frames
        ground_truth_frames: (B, T, C, H, W) ground truth video frames
        num_samples: number of samples to visualize

    Returns:
        PIL Image of the comparison grid
    """
    batch_size = min(predicted_frames.shape[0], num_samples)
    panels = [
        create_video_sample_panel(
            predicted_frames[index],
            ground_truth_frames[index],
            sample_index=index,
        )
        for index in range(batch_size)
    ]
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), color="white")
    top = 0
    for panel in panels:
        canvas.paste(panel, ((width - panel.width) // 2, top))
        top += panel.height
    return canvas


@torch.no_grad()
def inference_sample(model, batch: Dict, config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference to predict future video frames and actions using UniDiffuser's native inference method.

    Args:
        model: UniDiffuser model (LatentActionWorldModel)
        batch: Input batch containing observations, states, language embeddings, text instructions
        config: Configuration object containing inference parameters

    Returns:
        Tuple of (predicted_frames, predicted_actions)
        - predicted_frames: (B, num_pred_frames, C, H, W) in pixel space [0, 255]
        - predicted_actions: (B, action_chunk_size, action_dim)
    """
    model.eval()

    # Extract inference parameters from config
    num_inference_steps = config.model.inference.num_inference_timesteps

    # Move batch data to device
    device = next(model.parameters()).device
    first_frame = batch['first_frame'].to(device)  # [B, C, H, W] - conditioning frame

    state = batch['initial_state'].to(device) if 'initial_state' in batch and batch['initial_state'] is not None else None

    language_embeddings = batch['language_embedding']
    if language_embeddings is not None:
        language_embeddings = language_embeddings.to(device)

    vlm_inputs = batch['vlm_inputs']
    if vlm_inputs is not None:
        # Move all tensors in the VLM inputs dict to device
        vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in vlm_inputs.items()}

    with torch.no_grad():
        predicted_frames, predicted_actions = model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=num_inference_steps,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
            video_task_ids=(
                batch['video_task_ids'].to(device=device, dtype=torch.long)
                if batch.get('video_task_ids') is not None
                else None
            ),
            tactile_observed_source=(
                batch['tactile_observed_source'].to(device=device, dtype=torch.float32)
                if batch.get('tactile_observed_source') is not None
                else None
            ),
            tactile_grid_mask_source=(
                batch['tactile_grid_mask_source'].to(device=device, dtype=torch.bool)
                if batch.get('tactile_grid_mask_source') is not None
                else None
            ),
            tactile_observed_frame_times=(
                batch['tactile_observed_frame_times'].to(device=device, dtype=torch.float32)
                if batch.get('tactile_observed_frame_times') is not None
                else None
            ),
            tactile_future_query_times=(
                batch['tactile_future_query_times'].to(device=device, dtype=torch.float32)
                if batch.get('tactile_future_query_times') is not None
                else None
            ),
        )

    model.train()
    return predicted_frames, predicted_actions


def compute_action_metrics(predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> Dict[str, float]:
    """
    Compute action prediction metrics (MSE and L2 error).

    Args:
        predicted_actions: (B, T, action_dim) predicted actions
        ground_truth_actions: (B, T, action_dim) ground truth actions

    Returns:
        Dictionary containing MSE and L2 error metrics
    """
    # Compute MSE loss
    mse_loss = F.mse_loss(predicted_actions, ground_truth_actions, reduction='none').float()
    mse_loss_per_sample = mse_loss.reshape(predicted_actions.shape[0], -1).mean(1)

    # Compute L2 error (RMSE)
    l2_loss = mse_loss.sqrt() / (1 + 1e-3)
    l2_loss_per_sample = l2_loss.reshape(predicted_actions.shape[0], -1).mean(1)

    return {
        'mse_loss': mse_loss_per_sample.mean().item(),
        'l2_error': l2_loss_per_sample.mean().item(),
        'mse_std': mse_loss_per_sample.std().item(),
        'l2_std': l2_loss_per_sample.std().item()
    }


@torch.no_grad()
def evaluate_model(model, dataloader, accelerator, config, num_eval_batches: int = 2) -> Dict[str, float]:
    """
    Local-only evaluation: no distributed aggregation; safe for rank0-only evaluation.
    """
    logger.info(f"Running UniDiffuser evaluation for {num_eval_batches} batches...")
    model.eval()

    from collections import defaultdict
    metrics = defaultdict(list)
    visual_samples = []

    for step, batch in enumerate(dataloader):
        if step >= num_eval_batches:
            break
        if batch is None:
            continue

        # Inference
        predicted_frames, predicted_actions = inference_sample(model, batch, config)
        predicted_frames = predicted_frames.permute(0, 2, 1, 3, 4)     # [B, T, C, H, W]
        gt_frames = batch.get('video_frames')
        if gt_frames is not None:
            gt_frames = gt_frames.to(predicted_frames.device)
            video_mse = F.mse_loss(predicted_frames, gt_frames, reduction='mean').item()
            metrics['video_mse'].append(video_mse)

        # Action metrics (local)
        if 'action_sequence' in batch and predicted_actions is not None:
            gt_actions = batch['action_sequence'][:, :predicted_actions.shape[1]].to(predicted_actions.device)
            action_metrics = compute_action_metrics(predicted_actions, gt_actions)
            for key, value in action_metrics.items():
                metrics[f'action_{key}'].append(value)

        # Visualization sample
        if step == 0 and gt_frames is not None:
            provenance = {
                key: batch.get(key, [None] * predicted_frames.shape[0])[:4]
                for key in ("split", "task_name", "episode_name", "row_id", "condition_frame_idx", "modality")
                if key in batch
            }
            visual_samples.append({
                'predicted_frames': predicted_frames[:4],
                'ground_truth_frames': gt_frames[:4],
                'predicted_actions': predicted_actions[:4] if predicted_actions is not None else None,
                'ground_truth_actions': batch.get('action_sequence', None)[:4] if batch.get('action_sequence', None) is not None else None,
                'provenance': provenance,
            })

    # Aggregate metrics
    final_metrics = {}
    for key, values in metrics.items():
        if values:
            final_metrics[key] = float(np.mean(values))
            final_metrics[f'{key}_std'] = float(np.std(values))

    if visual_samples:
        sample = visual_samples[0]
        grid_visualization = create_video_grid(
            sample['predicted_frames'],
            sample['ground_truth_frames'],
            num_samples=4
        )
        final_metrics['visualization'] = grid_visualization
        final_metrics['visual_panels'] = [
            create_video_sample_panel(
                sample['predicted_frames'][index],
                sample['ground_truth_frames'][index],
                sample_index=index,
            )
            for index in range(sample['predicted_frames'].shape[0])
        ]
        final_metrics['visual_provenance'] = sample['provenance']

    model.train()
    return final_metrics


def log_evaluation_metrics(metrics: Dict, writer, accelerator, global_step: int):
    """
    Log evaluation metrics to tensorboard and wandb.

    Args:
        metrics: Dictionary containing evaluation metrics
        writer: TensorBoard writer (can be None)
        accelerator: HuggingFace accelerator
        global_step: Current training step
    """
    if accelerator.is_main_process:
        # Log scalar metrics
        log_dict = {}
        for key, value in metrics.items():
            if key not in ['visualization', 'visual_samples', 'visual_panels', 'visual_provenance'] and isinstance(value, (int, float)):
                log_dict[f'eval/{key}'] = value

        # Log to accelerator (wandb)
        if log_dict:
            accelerator.log(log_dict, step=global_step)

        # Log to TensorBoard
        if writer is not None:
            # Log scalar metrics to TensorBoard
            for key, value in log_dict.items():
                writer.add_scalar(key, value, global_step)

            # Log grid visualization
            if 'visualization' in metrics:
                img_array = np.array(metrics['visualization']).transpose(2, 0, 1)
                writer.add_image('eval/video_grid', img_array, global_step)
            panels = metrics.get('visual_panels', [])
            if panels:
                panel_root = Path(writer.log_dir) / 'eval_video_panels' / f'step_{global_step:08d}'
                panel_root.mkdir(parents=True, exist_ok=True)
                provenance = metrics.get('visual_provenance', {})
                (panel_root / 'provenance.json').write_text(
                    json.dumps(provenance, indent=2, ensure_ascii=False, default=str) + '\n',
                    encoding='utf-8',
                )
                for index, panel in enumerate(panels):
                    writer.add_image(
                        f'eval/video_sample_{index:02d}',
                        np.asarray(panel).transpose(2, 0, 1),
                        global_step,
                    )
                    panel.save(panel_root / f'sample_{index:02d}.png')

        # Print summary
        logger.info("=== UniDiffuser Evaluation Results ===")
        for key, value in metrics.items():
            if key != 'visualization' and isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
