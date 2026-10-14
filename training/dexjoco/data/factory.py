# Dataset Factory
# Simple factory to create different types of datasets

import gc
import os
from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf
import torch
import cv2


def _worker_init_fn(_worker_id: int) -> None:
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    if os.environ.get("MOTUS_DISABLE_PYTHON_GC") == "1":
        gc.disable()


def create_dataset(config: OmegaConf, val: bool = False):
    if config.dataset.type != "dexjoco_wtam_raw_aligned_dj_tfa2":
        raise ValueError("This entry point requires the DJ-TFA-2.0 DexJoCo dataset")
    from .dexjoco_raw_dj_tfa2_wtam_dataset import DexJoCoRawAlignedDJTFA2WTAMDataset
    params = OmegaConf.to_object(config.dataset.params)
    params.update(num_video_frames=config.common.num_video_frames,
                  video_size=(config.common.video_height, config.common.video_width),
                  image_aug=config.dataset.get("image_aug", False) and not val, val=val)
    return DexJoCoRawAlignedDJTFA2WTAMDataset(**params)


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]

    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            # Pad input_ids
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            # Pad attention_mask
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)

    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Process and batch language embeddings with padding."""
    padded_embeddings = []

    for emb in language_embeddings:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)

    # Stack to [B, seq_len, dim]
    return torch.stack(padded_embeddings, dim=0)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Universal collate function for all datasets.

    Args:
        batch: List of sample dictionaries (may contain None)

    Returns:
        Batched dictionary or None if all samples are None
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    # Stack tensors（支持无 initial_state 的样本）
    first_frames = torch.stack([sample['first_frame'] for sample in batch])
    video_values = [sample.get('video_frames') for sample in batch]
    if all(value is None for value in video_values):
        video_frames = None
    elif all(value is not None for value in video_values):
        video_frames = torch.stack(video_values)
    else:
        raise ValueError("A batch cannot mix condition-only samples with future-video samples")
    action_sequences = torch.stack([sample['action_sequence'] for sample in batch])
    has_initial_state = all(('initial_state' in sample and sample['initial_state'] is not None) for sample in batch)
    initial_states = torch.stack([sample['initial_state'] for sample in batch]) if has_initial_state else None

    # Process VLM inputs with padding in collate_fn
    vlm_inputs = [sample.get('vlm_inputs') for sample in batch]
    processed_vlm_inputs = None
    if vlm_inputs and all(vlm_input is not None for vlm_input in vlm_inputs):
        processed_vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)

    # Process language embeddings with padding in collate_fn
    language_embeddings = [sample.get('language_embedding') for sample in batch if 'language_embedding' in sample]
    processed_language_embeddings = None
    if language_embeddings and any(emb is not None for emb in language_embeddings):
        processed_language_embeddings = _process_language_embeddings_batch(language_embeddings)

    result = {
        'first_frame': first_frames,             # [B, C, H, W]
        'video_frames': video_frames,            # [B, F, C, H, W] or None
        'action_sequence': action_sequences,     # [B, F, D]
        'vlm_inputs': processed_vlm_inputs,
        'language_embedding': processed_language_embeddings,
    }

    # Optional per-dimension action validity. Raw DexJoCo single-arm samples
    # use the canonical 44-D action layout with the missing arm masked out.
    action_valid_masks = [sample.get('action_valid_mask') for sample in batch]
    if any(value is not None for value in action_valid_masks):
        if not all(value is not None for value in action_valid_masks):
            raise ValueError("A batch cannot mix samples with and without action_valid_mask")
        result['action_valid_mask'] = torch.stack(action_valid_masks).bool()

    tactile_keys = (
        'tactile_observed_source',
        'tactile_future_source',
        'tactile_observed_frame_times',
        'tactile_future_query_times',
    )
    tactile_present = [all(sample.get(key) is not None for key in tactile_keys) for sample in batch]
    if any(tactile_present):
        if not all(tactile_present):
            raise ValueError("A batch cannot mix samples with and without V2.1 tactile data")
        for key in tactile_keys:
            result[key] = torch.stack([sample[key] for sample in batch])
        grid_masks = [sample.get('tactile_grid_mask_source') for sample in batch]
        if any(value is not None for value in grid_masks):
            if not all(value is not None for value in grid_masks):
                raise ValueError("A batch cannot mix tactile samples with and without grid masks")
            result['tactile_grid_mask_source'] = torch.stack(grid_masks)

    task_ids = [sample.get('video_task_ids') for sample in batch]
    if any(value is not None for value in task_ids):
        if not all(value is not None for value in task_ids):
            raise ValueError("A batch cannot mix samples with and without video_task_ids")
        result['video_task_ids'] = torch.as_tensor(task_ids, dtype=torch.long).reshape(-1)

    for key in (
        'modality',
        'split',
        'task_name',
        'source_embodiment',
        'episode_name',
        'row_id',
        'condition_frame_idx',
        'video_indices',
        'action_indices',
        'target_indices',
        'video_path',
        'target_path',
    ):
        values = [sample.get(key) for sample in batch]
        if any(value is not None for value in values):
            result[key] = values

    if initial_states is not None:
        result['initial_state'] = initial_states
    component_times = [sample.get('_data_component_times') for sample in batch]
    if component_times and all(value is not None for value in component_times):
        result['_data_component_times'] = torch.stack(component_times)
    raw_component_times = [sample.get('_raw_data_component_times') for sample in batch]
    if raw_component_times and all(value is not None for value in raw_component_times):
        result['_raw_data_component_times'] = torch.stack(raw_component_times)
    raw_worker_resources = [sample.get('_raw_worker_resource_metrics') for sample in batch]
    if raw_worker_resources and all(value is not None for value in raw_worker_resources):
        result['_raw_worker_resource_metrics'] = torch.stack(raw_worker_resources)

    return result
