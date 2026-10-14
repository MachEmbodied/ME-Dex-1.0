"""DexJoCo observations exactly matching the WTAM V2.5 training contract."""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

from runtime.dexjoco.contracts import official_state_to_rotvec_proprio


def _resize_with_padding(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize RGB without changing aspect ratio; ``size`` is (height, width)."""
    height, width = size
    source = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    scale = min(width / source.width, height / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return np.asarray(canvas)


def make_t_shape_rgb(
    *,
    front: np.ndarray,
    left_wrist: np.ndarray | None,
    right_wrist: np.ndarray | None,
    bimanual: bool,
    output_size: tuple[int, int] = (384, 320),
) -> np.ndarray:
    """Pack front/left/right views using the same layout as training.

    Single-arm DexJoCo is canonical-right: its only wrist view must be supplied
    as ``right_wrist`` and the lower-left panel is semantic black padding.
    """
    front = np.asarray(front, dtype=np.uint8)
    if front.ndim != 3 or front.shape[-1] != 3:
        raise ValueError(f"front must be HxWx3 RGB, got {front.shape}")
    if not bimanual and left_wrist is not None:
        raise ValueError("single-arm input must not populate canonical-left wrist")
    black = np.zeros_like(front)
    left = black if left_wrist is None else np.asarray(left_wrist, dtype=np.uint8)
    right = black if right_wrist is None else np.asarray(right_wrist, dtype=np.uint8)
    if bimanual and (left_wrist is None or right_wrist is None):
        raise ValueError("bimanual input requires both wrist cameras")
    half_size = (front.shape[1] // 2, front.shape[0] // 2)
    left_small = np.asarray(Image.fromarray(left).resize(half_size, Image.Resampling.BILINEAR))
    right_small = np.asarray(Image.fromarray(right).resize(half_size, Image.Resampling.BILINEAR))
    packed = np.concatenate((front, np.concatenate((left_small, right_small), axis=1)), axis=0)
    return _resize_with_padding(packed, output_size)


def make_horizontal_rgb(
    *,
    front: np.ndarray,
    left_wrist: np.ndarray | None,
    right_wrist: np.ndarray | None,
    bimanual: bool,
    output_size: tuple[int, int] = (256, 384),
) -> np.ndarray:
    """Pack front left, then left/right wrists vertically on the right."""
    front = np.asarray(front, dtype=np.uint8)
    if front.ndim != 3 or front.shape[-1] != 3:
        raise ValueError(f"front must be HxWx3 RGB, got {front.shape}")
    height, width = front.shape[:2]
    if height != width or height % 2:
        raise ValueError(f"horizontal layout requires an even square front, got {front.shape}")
    if not bimanual and left_wrist is not None:
        raise ValueError("single-arm input must not populate canonical-left wrist")
    if bimanual and (left_wrist is None or right_wrist is None):
        raise ValueError("bimanual input requires both wrist cameras")
    if output_size != (256, 384):
        raise ValueError("horizontal layout requires output_size=(256, 384)")
    left = np.zeros_like(front) if left_wrist is None else np.asarray(left_wrist, dtype=np.uint8)
    right = np.zeros_like(front) if right_wrist is None else np.asarray(right_wrist, dtype=np.uint8)
    half_size = (width // 2, height // 2)
    # Match the raw training loader's OpenCV interpolation exactly.
    left_small = cv2.resize(left, half_size)
    right_small = cv2.resize(right, half_size)
    packed = np.concatenate((front, np.concatenate((left_small, right_small), axis=0)), axis=1)
    return cv2.resize(packed, (output_size[1], output_size[0]))


def make_wtam_state(official_state: np.ndarray, *, bimanual: bool) -> np.ndarray:
    """Return the unified 44-D state used by the multitask checkpoint."""
    state = official_state_to_rotvec_proprio(official_state, bimanual=bimanual)
    if bimanual:
        if state.shape != (44,):
            raise RuntimeError(f"unexpected bimanual state shape: {state.shape}")
        return state
    if state.shape != (22,):
        raise RuntimeError(f"unexpected single-arm state shape: {state.shape}")
    return np.concatenate((state, np.zeros(22, dtype=np.float32)))
