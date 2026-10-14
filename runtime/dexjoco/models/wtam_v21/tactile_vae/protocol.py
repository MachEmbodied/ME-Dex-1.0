from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


HAND_SIDE_IDS = {"unknown": 0, "left": 1, "right": 2}
FINGER_IDS = {
    "unknown": 0,
    "thumb": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "palm": 5,
}
SEGMENT_IDS = {"unknown": 0, "tip": 1, "pad": 2, "base": 3, "palm": 4}


@dataclass(frozen=True)
class TactileProtocol:
    max_regions: int = 12
    observed_frames: int = 2
    future_frames: int = 16
    channels: int = 3
    height: int = 10
    width: int = 14
    canonical_channel_order: tuple[str, str, str] = (
        "normal",
        "shear_u",
        "shear_v",
    )

    @property
    def total_frames(self) -> int:
        return self.observed_frames + self.future_frames


@dataclass(frozen=True)
class CanonicalTactileSample:
    tactile_observed: torch.Tensor
    tactile_future: torch.Tensor
    region_mask: torch.Tensor
    grid_mask: torch.Tensor
    uv_coordinates: torch.Tensor
    domain_id: torch.Tensor
    episode_id: torch.Tensor
    anchor_index: torch.Tensor
    frame_times: torch.Tensor
    normalization_group: torch.Tensor
    hand_side_id: torch.Tensor
    finger_id: torch.Tensor
    segment_id: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(vars(self))


def native_grid_metadata(
    valid: np.ndarray,
    *,
    height: int = 10,
    width: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical grid mask and physical normalized UV coordinates.

    Native taxels are zero-padded, never interpolated. UV uses the valid
    native bounding box so a 5x7 and a 10x14 sensor both cover [0, 1]^2.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 3 or valid.shape[1:] != (height, width):
        raise ValueError(f"Expected [R,{height},{width}] validity, got {valid.shape}")
    uv = np.zeros((valid.shape[0], 2, height, width), dtype=np.float32)
    for region, mask in enumerate(valid):
        rows, cols = np.nonzero(mask)
        if not len(rows):
            continue
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        row_scale = max(row_max - row_min, 1)
        col_scale = max(col_max - col_min, 1)
        uv[region, 0] = (np.arange(width, dtype=np.float32)[None] - col_min) / col_scale
        uv[region, 1] = (np.arange(height, dtype=np.float32)[:, None] - row_min) / row_scale
        uv[region] *= mask[None]
    return valid, uv


def canonicalize_force_clip(
    force: np.ndarray,
    *,
    slot_mapping: Mapping[int, int],
    region_semantics: Mapping[int, Mapping[str, str]],
    taxel_valid: np.ndarray | None,
    domain_id: int,
    normalization_group: int,
    episode_id: int,
    anchor_index: int,
    frame_times: np.ndarray,
    channel_indices: tuple[int, int, int] = (0, 1, 2),
    channel_signs: tuple[float, float, float] = (1.0, 1.0, 1.0),
    protocol: TactileProtocol = TactileProtocol(),
) -> CanonicalTactileSample:
    """Map a physical-force clip into a 12-region batch container.

    Destination slots are storage positions only. Physical identity is carried
    explicitly by hand, finger, and segment tensors.
    """
    force = np.asarray(force, dtype=np.float32)
    expected_prefix = (protocol.total_frames,)
    if force.ndim != 5 or force.shape[0] != expected_prefix[0] or force.shape[2] != 3:
        raise ValueError(f"Expected [18,R,3,H,W] force clip, got {force.shape}")
    native_h, native_w = force.shape[-2:]
    if native_h > protocol.height or native_w > protocol.width:
        raise ValueError(f"Native grid {native_h}x{native_w} exceeds canonical grid")
    if not np.isfinite(force).all():
        raise ValueError("Tactile clip contains non-finite forces")
    if sorted(channel_indices) != [0, 1, 2] or any(
        abs(sign) != 1.0 for sign in channel_signs
    ):
        raise ValueError("Invalid source-to-canonical channel transform")
    force = force[:, :, list(channel_indices)]
    force = force * np.asarray(channel_signs, dtype=np.float32)[None, None, :, None, None]
    frames = np.zeros(
        (protocol.total_frames, protocol.max_regions, 3, protocol.height, protocol.width),
        dtype=np.float32,
    )
    valid = np.zeros((protocol.max_regions, protocol.height, protocol.width), dtype=bool)
    region_mask = np.zeros(protocol.max_regions, dtype=bool)
    hand_side_id = np.zeros(protocol.max_regions, dtype=np.int64)
    finger_id = np.zeros(protocol.max_regions, dtype=np.int64)
    segment_id = np.zeros(protocol.max_regions, dtype=np.int64)
    source_valid = (
        np.ones((force.shape[1], native_h, native_w), dtype=bool)
        if taxel_valid is None
        else np.asarray(taxel_valid, dtype=bool)
    )
    if source_valid.shape != (force.shape[1], native_h, native_w):
        raise ValueError(f"taxel_valid mismatch: {source_valid.shape} vs {force.shape}")
    used_slots: set[int] = set()
    for source, slot in slot_mapping.items():
        if source < 0 or source >= force.shape[1]:
            raise IndexError(f"Source region {source} outside [0,{force.shape[1]})")
        if slot < 0 or slot >= protocol.max_regions or slot in used_slots:
            raise ValueError(f"Invalid or duplicate canonical slot {slot}")
        used_slots.add(slot)
        frames[:, slot, :, :native_h, :native_w] = force[:, source]
        valid[slot, :native_h, :native_w] = source_valid[source]
        frames[:, slot] *= valid[slot][None, None]
        region_mask[slot] = bool(source_valid[source].any())
        descriptor = region_semantics[source]
        hand_side_id[slot] = HAND_SIDE_IDS[descriptor["hand_side"]]
        finger_id[slot] = FINGER_IDS[descriptor["finger"]]
        segment_id[slot] = SEGMENT_IDS[descriptor["segment"]]
    valid, uv = native_grid_metadata(valid, height=protocol.height, width=protocol.width)
    times = np.asarray(frame_times, dtype=np.float32)
    if times.shape != (protocol.total_frames,) or not np.isfinite(times).all():
        raise ValueError(f"frame_times must be finite [18], got {times.shape}")
    if np.any(np.diff(times) < 0):
        raise ValueError("frame_times must be monotonic")
    tensor = torch.from_numpy(frames)
    return CanonicalTactileSample(
        tactile_observed=tensor[: protocol.observed_frames].permute(1, 0, 2, 3, 4),
        tactile_future=tensor[protocol.observed_frames :].permute(1, 0, 2, 3, 4),
        region_mask=torch.from_numpy(region_mask),
        grid_mask=torch.from_numpy(valid),
        uv_coordinates=torch.from_numpy(uv),
        domain_id=torch.tensor(domain_id, dtype=torch.long),
        episode_id=torch.tensor(episode_id, dtype=torch.long),
        anchor_index=torch.tensor(anchor_index, dtype=torch.long),
        frame_times=torch.from_numpy(times),
        normalization_group=torch.tensor(normalization_group, dtype=torch.long),
        hand_side_id=torch.from_numpy(hand_side_id),
        finger_id=torch.from_numpy(finger_id),
        segment_id=torch.from_numpy(segment_id),
    )
