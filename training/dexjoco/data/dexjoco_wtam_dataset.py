"""Read the user's native50/RGB30/TFA50 DexJoCo Stage-1 corpus.

This dataset deliberately does not use DexJoCo's public LeRobot converter.  The
local corpus contains richer replay products: native 50 Hz actions/states,
independently sampled 30 Hz RGB videos, and aligned 50 Hz tactile force fields.
All target indices are therefore defined in the native control clock and RGB is
looked up through ``rgb_frame_index.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from runtime.dexjoco.text_cache import DiskT5Cache


# Every distributed rank owns DataLoader workers.  Leaving OpenCV/FFmpeg at
# its host-wide default can create ~176 decoder threads per worker on the PPU
# node and oversubscribe the machine by two orders of magnitude.
cv2.setNumThreads(1)


SINGLE_ARM_TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
BIMANUAL_TASKS = (
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
)

DEFAULT_TASK_PROMPTS = {
    "bimanual_assembly": "Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole.",
    "bimanual_hanoi": "Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand.",
    "bimanual_microwave_cook": "Open the microwave door, place the food inside the microwave, close the door, and press the start button.",
    "bimanual_photograph": "Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand.",
    "bimanual_unlock_ipad": "Grasp the iPad and enter the password 123 to unlock the device.",
    "click_mouse": "Move the mouse to the purple mouse pad and click the left mouse button.",
    "fold_glasses": "Fold the glasses and place them into the case.",
    "hammer_nail": "Use the hammer to drive the nail into the wooden board.",
    "pick_bucket": "Place the boxed food into the bucket and then lift the bucket.",
    "pinch_tongs": "Grasp the tongs and perform three consecutive open-close motions.",
    "water_plant": "Grasp the watering can and apply water to the plant.",
}

RAW_PAD_NAMES = (
    "LF_BASE", "TH_TIP", "PALM_1", "LF_TIP", "FF_PAD",
    "FF_TIP", "RF_PAD", "FF_BASE", "MF_TIP", "MF_PAD",
    "RF_TIP", "TH_PAD", "MF_BASE", "RF_BASE", "LF_PAD",
)


@dataclass(frozen=True)
class DexJoCoEpisode:
    task_name: str
    episode_name: str
    root: Path
    length: int
    rgb_action_indices: tuple[int, ...]
    # third_person, left_wrist, right_wrist from the replay manifest.
    camera_validity: tuple[bool, bool, bool]
    prompt: str
    source_embodiment: str

    @property
    def trajectory_path(self) -> Path:
        return self.root / "trajectory.npz"

    @property
    def tactile_path(self) -> Path:
        return self.root / "tactile" / "tactile_force_field.tfa1.h5"

    @property
    def training_cache_path(self) -> Path:
        return self.root / "wtam_training_cache_v1.h5"


def _nearest_rgb_frame(rgb_action_indices: np.ndarray, action_index: int) -> int:
    """Return the RGB frame whose native-action timestamp is nearest."""
    pos = int(np.searchsorted(rgb_action_indices, action_index, side="left"))
    if pos <= 0:
        return 0
    if pos >= len(rgb_action_indices):
        return len(rgb_action_indices) - 1
    before = pos - 1
    return before if action_index - rgb_action_indices[before] <= rgb_action_indices[pos] - action_index else pos


def _exact_rgb_target_offsets(action_count: int) -> tuple[int, ...]:
    """Alternating 5/6-frame offsets with a 5.5-frame mean interval."""
    return tuple((11 * value) // 2 for value in range(1, int(action_count) + 1))


def _dense_native_target_schedule(
    rgb_action_indices: np.ndarray,
    anchor_rgb_index: int,
    *,
    action_count: int = 20,
    num_video_frames: int = 12,
) -> tuple[int, list[int], list[int], list[int]]:
    """Build the Dense-20 contract on the audited native 50 Hz clock.

    Action and tactile use every native control step after the condition.  RGB
    uses the next real camera frames from ``rgb_frame_index.json``; for the
    audited 30-on-50 clock, twelve RGB frames span exactly twenty native steps.
    Thus all three modalities cover one 400 ms interval without inventing RGB
    frames or sparsifying the control/tactile trajectory.
    """
    if int(action_count) != 20 or int(num_video_frames) != 12:
        raise ValueError(
            "native_dense20_video12_v1 requires action_count=20 and "
            "num_video_frames=12"
        )
    anchor_rgb_index = int(anchor_rgb_index)
    if anchor_rgb_index < 0 or anchor_rgb_index + num_video_frames >= len(rgb_action_indices):
        raise IndexError("Dense-20 RGB window exceeds the audited frame table")
    anchor = int(rgb_action_indices[anchor_rgb_index])
    action_indices = list(range(anchor + 1, anchor + action_count + 1))
    video_rgb_indices = list(
        range(anchor_rgb_index + 1, anchor_rgb_index + num_video_frames + 1)
    )
    video_action_indices = [
        int(rgb_action_indices[value]) for value in video_rgb_indices
    ]
    if video_action_indices[-1] != anchor + action_count:
        raise ValueError(
            "The next 12 RGB frames do not end at native +20: "
            f"anchor={anchor}, video_native={video_action_indices}"
        )
    action_index_set = set(action_indices)
    if any(value not in action_index_set for value in video_action_indices):
        raise ValueError("A Dense-20 RGB target falls outside the Action/Tactile clock")
    return anchor, action_indices, video_rgb_indices, video_action_indices


def _dense_anchor_rgb_bounds(
    rgb_action_indices: np.ndarray,
    episode_length: int,
    *,
    action_count: int = 20,
    num_video_frames: int = 12,
) -> tuple[int, int]:
    """Inclusive RGB-anchor bounds for a complete observed/future sample."""
    # t=0 is a valid condition.  Its unavailable t-1 tactile observation is
    # represented by duplicating t=0, preserving a two-slice codec interface
    # without inventing a zero/no-contact measurement.
    first = int(np.searchsorted(rgb_action_indices, 0, side="left"))
    last = len(rgb_action_indices) - 1 - int(num_video_frames)
    while (
        last >= first
        and int(rgb_action_indices[last]) + int(action_count) >= int(episode_length)
    ):
        last -= 1
    if last < first:
        raise ValueError("Episode is shorter than the requested tactile/video horizon")
    return first, last


def _dense_legal_anchor_rgb_indices(
    rgb_action_indices: np.ndarray,
    episode_length: int,
    *,
    action_count: int = 20,
    num_video_frames: int = 12,
) -> np.ndarray:
    """Return every condition RGB frame satisfying the strict shared endpoint."""
    first, last = _dense_anchor_rgb_bounds(
        rgb_action_indices,
        episode_length,
        action_count=action_count,
        num_video_frames=num_video_frames,
    )
    candidates = np.arange(first, last + 1, dtype=np.int64)
    endpoints = rgb_action_indices[candidates + int(num_video_frames)]
    anchors = rgb_action_indices[candidates]
    legal = candidates[endpoints == anchors + int(action_count)]
    if legal.size == 0:
        raise ValueError("Episode has no strict Dense-20 shared-endpoint anchor")
    return legal


def _sparse40_native_target_schedule(
    rgb_action_indices: np.ndarray,
    anchor_rgb_index: int,
    *,
    action_count: int = 40,
    num_video_frames: int = 16,
) -> tuple[int, list[int], list[int], list[int]]:
    """Build the 1.6-s native-clock contract.

    Action/Tactile targets are sampled at 25 Hz (native +2, +4, ..., +80).
    Video targets are sampled at 10 Hz (native +5, +10, ..., +80).  The
    audited 30-on-50 RGB clock contains every +5 target exactly, so no nearest
    frame approximation is permitted.  Eight timestamps (+10 multiples) and
    the shared +80 endpoint align all three modalities exactly.
    """
    if int(action_count) != 40 or int(num_video_frames) != 16:
        raise ValueError(
            "native_sparse40_video16_v1 requires action_count=40 and "
            "num_video_frames=16"
        )
    anchor_rgb_index = int(anchor_rgb_index)
    if anchor_rgb_index < 0 or anchor_rgb_index >= len(rgb_action_indices):
        raise IndexError("Sparse-40 RGB anchor exceeds the audited frame table")
    anchor = int(rgb_action_indices[anchor_rgb_index])
    action_indices = [anchor + 2 * value for value in range(1, 41)]
    video_action_indices = [anchor + 5 * value for value in range(1, 17)]
    positions = np.searchsorted(rgb_action_indices, video_action_indices)
    if (
        np.any(positions >= len(rgb_action_indices))
        or not np.array_equal(rgb_action_indices[positions], video_action_indices)
    ):
        raise ValueError(
            "The audited RGB table does not contain every native +5 target: "
            f"anchor={anchor}"
        )
    video_rgb_indices = [int(value) for value in positions.tolist()]
    return anchor, action_indices, video_rgb_indices, video_action_indices


def _sparse40_legal_anchor_rgb_indices(
    rgb_action_indices: np.ndarray,
    episode_length: int,
    *,
    action_count: int = 40,
    num_video_frames: int = 16,
) -> np.ndarray:
    """Return every RGB condition with a complete exact +80 contract."""
    if int(action_count) != 40 or int(num_video_frames) != 16:
        raise ValueError("Invalid Sparse-40 contract dimensions")
    anchors = np.asarray(rgb_action_indices, dtype=np.int64)
    frame_set = set(int(value) for value in anchors.tolist())
    legal = []
    for rgb_index, anchor in enumerate(anchors.tolist()):
        if anchor + 80 >= int(episode_length):
            continue
        if all(anchor + offset in frame_set for offset in range(5, 81, 5)):
            legal.append(rgb_index)
    if not legal:
        raise ValueError("Episode has no exact Sparse-40/Video-16 anchor")
    return np.asarray(legal, dtype=np.int64)


def _pad_right_arm_22_to_44(value: np.ndarray) -> np.ndarray:
    """Canonical multi-task layout: measured RIGHT22 followed by padded LEFT22."""
    if value.shape[-1] != 22:
        raise ValueError(f"Expected a 22-D right-arm value, got {value.shape}")
    widths = [(0, 0)] * value.ndim
    widths[-1] = (0, 22)
    return np.pad(value, widths)


def _pad_right_tactile_15_to_30(
    observed: np.ndarray,
    future: np.ndarray,
    grid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad an observed RIGHT15 registry with an invalid LEFT15 registry."""
    if observed.shape[1] != 15 or future.shape[1] != 15:
        raise ValueError("Single-arm tactile sequences must contain 15 right pads")
    if grid_mask.shape != (15, 10, 14):
        raise ValueError(f"Unexpected right-hand grid mask: {grid_mask.shape}")
    return (
        np.pad(observed, ((0, 0), (0, 15), (0, 0), (0, 0), (0, 0))),
        np.pad(future, ((0, 0), (0, 15), (0, 0), (0, 0), (0, 0))),
        np.pad(grid_mask, ((0, 15), (0, 0), (0, 0))),
    )


def _read_rgb_index(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Deliberately do not trust the descriptive fps/action_rate_hz fields.  The
    # preflight audit derives rates from per-frame and tactile timestamps.
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Empty RGB index: {path}")
    indices = tuple(int(item["source_action_index"]) for item in frames)
    if any(b < a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"Non-monotonic RGB/action index: {path}")
    gaps = tuple(b - a for a, b in zip(indices, indices[1:]))
    if indices[0] != 0 or any(gap not in (1, 2) for gap in gaps):
        raise ValueError(
            f"RGB index is not a valid 30-on-50 Hz clock at {path}: "
            f"first={indices[0]}, gaps={sorted(set(gaps))}"
        )
    return indices


def _h5_take_allow_duplicates(
    dataset: h5py.Dataset,
    indices: Sequence[int],
    *,
    dtype: np.dtype | type,
) -> np.ndarray:
    """Take ordered HDF5 rows while supporting duplicate indices such as [0,0].

    h5py requires fancy-index elements to be strictly increasing.  Reading the
    sorted unique rows once and applying the inverse map in NumPy preserves the
    requested order without loading the complete episode dataset.
    """
    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError(f"Expected a non-empty 1-D HDF5 index list: {requested}")
    unique, inverse = np.unique(requested, return_inverse=True)
    values = np.asarray(dataset[unique], dtype=dtype)
    return values[inverse]


class DexJoCoWTAMDataset(Dataset):
    """WTAM windows from either the single-arm or bimanual DexJoCo tasks.

    The two embodiments intentionally use separate training configurations.
    This keeps the original flow-matching action loss unchanged: single-arm
    targets are 22-D and bimanual targets are 44-D, with no fake padded action
    dimensions entering the objective.
    """

    def __init__(
        self,
        root: str,
        t5_cache_path: str,
        embodiment: str,
        samples_per_episode: int = 32,
        action_count: int = 16,
        video_action_freq_ratio: int = 2,
        num_video_frames: int = 8,
        alignment_mode: str = "rgb_index_exact_5p5frame_v1",
        video_size: tuple[int, int] = (384, 320),
        expected_episodes_per_task: int = 100,
        tasks: Sequence[str] | None = None,
        task_prompts: Mapping[str, str] | None = None,
        state_source: str = "policy_state",
        include_tactile: bool = True,
        val: bool = False,
        image_aug: bool = False,
    ) -> None:
        if image_aug:
            raise ValueError("DexJoCo V2.5 does not use image augmentation")
        if embodiment not in {"single", "bimanual", "multitask"}:
            raise ValueError("embodiment must be single, bimanual, or multitask")
        supported_alignment_modes = {
            "rgb_index_exact_5p5frame_v1",
            "native_dense20_video12_v1",
            "native_sparse40_video16_v1",
        }
        if alignment_mode not in supported_alignment_modes:
            raise ValueError(
                f"Unsupported DexJoCo alignment_mode={alignment_mode!r}"
            )
        if (
            alignment_mode == "rgb_index_exact_5p5frame_v1"
            and int(action_count) != int(num_video_frames) * int(video_action_freq_ratio)
        ):
            raise ValueError("Legacy DexJoCo action/video horizon contract is inconsistent")
        self.root = Path(root)
        self.embodiment = embodiment
        self.action_dim = 22 if embodiment == "single" else 44
        # A fixed, causally available command state is used for multi-task
        # training.  Official task observations append task-specific object
        # fields and therefore do not have one common dimension.
        self.state_dim = self.action_dim
        if state_source not in {"policy_state", "command_smoke_only"}:
            raise ValueError(
                "state_source must be policy_state or command_smoke_only"
            )
        self.state_source = state_source
        self.include_tactile = bool(include_tactile)
        self.region_count = 15 if embodiment == "single" else 30
        self.samples_per_episode = int(samples_per_episode)
        self.action_count = int(action_count)
        self.num_video_frames = int(num_video_frames)
        self.alignment_mode = str(alignment_mode)
        # Targets are selected in the *actual RGB frame table*, never by an
        # assumed constant FPS.  Alternating 5/6 RGB-frame increments gives a
        # mean action interval of 5.5/30 ~= 183.3 ms while every selected RGB,
        # action, and tactile sample shares one exact native source index.
        if self.alignment_mode == "rgb_index_exact_5p5frame_v1":
            self.action_rgb_frame_offsets = _exact_rgb_target_offsets(action_count)
            self.video_rgb_frame_offsets = tuple(
                self.action_rgb_frame_offsets[
                    int(video_action_freq_ratio) * value - 1
                ]
                for value in range(1, int(num_video_frames) + 1)
            )
            if self.video_rgb_frame_offsets[-1] != 88:
                raise ValueError("Unexpected exact-alignment horizon")
        elif self.alignment_mode == "native_dense20_video12_v1":
            if self.action_count != 20 or self.num_video_frames != 12:
                raise ValueError(
                    "native_dense20_video12_v1 requires 20 Action/Tactile "
                    "targets and 12 Video targets"
                )
            self.action_rgb_frame_offsets = ()
            self.video_rgb_frame_offsets = ()
        else:
            if self.action_count != 40 or self.num_video_frames != 16:
                raise ValueError(
                    "native_sparse40_video16_v1 requires 40 Action/Tactile "
                    "targets and 16 Video targets"
                )
            self.action_rgb_frame_offsets = ()
            self.video_rgb_frame_offsets = ()
        self.video_size = tuple(int(value) for value in video_size)
        self.val = bool(val)
        prompts = dict(DEFAULT_TASK_PROMPTS)
        if task_prompts:
            prompts.update({str(key): str(value) for key, value in task_prompts.items()})
        allowed_tasks: Sequence[str] = (
            SINGLE_ARM_TASKS
            if embodiment == "single"
            else BIMANUAL_TASKS
            if embodiment == "bimanual"
            else SINGLE_ARM_TASKS + BIMANUAL_TASKS
        )
        tasks = tuple(tasks) if tasks is not None else tuple(allowed_tasks)
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Duplicate tasks are not allowed: {tasks}")
        unknown = sorted(set(tasks) - set(allowed_tasks))
        if unknown:
            raise ValueError(f"Tasks do not belong to embodiment={embodiment}: {unknown}")
        # Trainer metrics resolve this immutable table through DataLoader and
        # Accelerate wrappers.  Returning only per-sample ``task_name`` is not
        # enough because every rank must allocate identically ordered metric
        # accumulators before it sees its first random batch.
        self.task_names = tasks
        records: list[DexJoCoEpisode] = []
        for task in tasks:
            source_embodiment = "single" if task in SINGLE_ARM_TASKS else "bimanual"
            source_regions = 15 if source_embodiment == "single" else 30
            episode_dirs = sorted((self.root / task / "episodes").glob("episode_*"))
            if len(episode_dirs) != int(expected_episodes_per_task):
                raise ValueError(
                    f"{task}: expected {expected_episodes_per_task} episodes, found {len(episode_dirs)}"
                )
            for episode_dir in episode_dirs:
                tactile_path = episode_dir / "tactile" / "tactile_force_field.tfa1.h5"
                with h5py.File(tactile_path, "r") as handle:
                    length = int(handle["tactile_force_field/force_canonical"].shape[0])
                    names = tuple(
                        value.decode() if isinstance(value, bytes) else str(value)
                        for value in handle[
                            "tactile_force_field_metadata/pad_name"
                        ][:].tolist()
                    )
                expected_names = (
                    RAW_PAD_NAMES
                    if source_embodiment == "single"
                    else tuple(f"RIGHT_{name}" for name in RAW_PAD_NAMES)
                    + tuple(f"LEFT_{name}" for name in RAW_PAD_NAMES)
                )
                if names != expected_names:
                    raise ValueError(f"Unexpected tactile pad registry: {tactile_path}")
                rgb_indices = _read_rgb_index(episode_dir / "rgb_frame_index.json")
                stage1_manifest = json.loads(
                    (episode_dir / "stage1_manifest.json").read_text(encoding="utf-8")
                )
                validity = stage1_manifest.get("camera_validity", {})
                camera_validity = tuple(
                    bool(validity.get(camera, False))
                    for camera in ("third_person", "left_wrist", "right_wrist")
                )
                if not camera_validity[0]:
                    raise ValueError(f"Invalid third-person RGB stream: {episode_dir}")
                records.append(
                    DexJoCoEpisode(
                        task_name=task,
                        episode_name=episode_dir.name,
                        root=episode_dir,
                        length=length,
                        rgb_action_indices=rgb_indices,
                        camera_validity=camera_validity,
                        prompt=prompts[task],
                        source_embodiment=source_embodiment,
                    )
                )
        self.records = tuple(records)
        self.dense_anchor_rgb_indices = (
            {
                record.root: tuple(
                    int(value)
                    for value in _dense_legal_anchor_rgb_indices(
                        np.asarray(record.rgb_action_indices, dtype=np.int64),
                        record.length,
                        action_count=self.action_count,
                        num_video_frames=self.num_video_frames,
                    ).tolist()
                )
                for record in self.records
            }
            if self.alignment_mode == "native_dense20_video12_v1"
            else {}
        )
        self.sparse40_anchor_rgb_indices = (
            {
                record.root: tuple(
                    int(value)
                    for value in _sparse40_legal_anchor_rgb_indices(
                        np.asarray(record.rgb_action_indices, dtype=np.int64),
                        record.length,
                        action_count=self.action_count,
                        num_video_frames=self.num_video_frames,
                    ).tolist()
                )
                for record in self.records
            }
            if self.alignment_mode == "native_sparse40_video16_v1"
            else {}
        )
        self.records_by_task = {
            task: tuple(record for record in self.records if record.task_name == task)
            for task in self.task_names
        }
        self.tasks_by_embodiment = {
            source: tuple(
                task
                for task in self.task_names
                if ("single" if task in SINGLE_ARM_TASKS else "bimanual") == source
            )
            for source in ("single", "bimanual")
        }
        self.records_by_embodiment_and_task = {
            source: {
                task: self.records_by_task[task]
                for task in self.tasks_by_embodiment[source]
            }
            for source in ("single", "bimanual")
        }
        self._global_homogeneous_batch_size: int | None = None
        self.t5_cache = DiskT5Cache(t5_cache_path)
        missing_prompts = sorted({record.prompt for record in records} - self.t5_cache.prompt_set)
        if missing_prompts:
            raise KeyError(f"DexJoCo prompts missing from T5 cache: {missing_prompts}")

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_episode

    def configure_global_embodiment_homogeneous_sampling(
        self, *, per_device_batch_size: int, world_size: int
    ) -> None:
        """Make every distributed micro-batch contain one embodiment.

        Accelerate assigns consecutive local batches to consecutive ranks.  We
        therefore keep single/bimanual fixed for
        ``per_device_batch_size * world_size`` consecutive dataset indices.
        Tasks are sampled uniformly *within* that embodiment, so a global batch
        may mix all six single-arm tasks or all five bimanual tasks.  Episodes
        and temporal anchors remain stochastic.
        This is required before physically compacting 12 TacAE anatomy slots to
        the 5 (single arm) or 10 (bimanual) sensors that DexJoCo actually owns.
        """
        global_batch = int(per_device_batch_size) * int(world_size)
        if global_batch <= 0:
            raise ValueError(f"Invalid global batch size: {global_batch}")
        self._global_homogeneous_batch_size = global_batch

    # Compatibility for old launchers; semantics intentionally changed from
    # task-homogeneous to the less restrictive embodiment-homogeneous contract.
    configure_global_task_homogeneous_sampling = (
        configure_global_embodiment_homogeneous_sampling
    )

    @staticmethod
    def _resize_with_padding(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        target_h, target_w = size
        h, w = frame.shape[:2]
        scale = min(target_h / h, target_w / w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y = (target_h - resized.shape[0]) // 2
        x = (target_w - resized.shape[1]) // 2
        output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return output

    @staticmethod
    def _decode_video_frames(
        path: Path,
        indices: Sequence[int],
        *,
        sequential: bool = False,
    ) -> dict[int, np.ndarray]:
        """Strictly decode requested frames from a camera declared valid.

        Missing-camera padding is handled by ``_t_shape_frames`` only when the
        replay manifest explicitly marks that camera invalid.  Turning an I/O
        failure from a *valid* stream into a black image would silently corrupt
        formal training and make it indistinguishable from intentional padding.
        """
        capture = cv2.VideoCapture(
            str(path),
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_N_THREADS, 1],
        )
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open valid RGB stream: {path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise RuntimeError(
                    f"Valid RGB stream reports no decodable frames: {path}"
                )
            requested = sorted(set(int(value) for value in indices))
            for index in requested:
                if index < 0 or index >= frame_count:
                    raise IndexError(
                        f"RGB frame {index} is outside [0,{frame_count}) for {path}"
                    )
            output: dict[int, np.ndarray] = {}
            if not requested:
                return output
            if sequential:
                # Raw-aligned clips request one short, monotonic horizon.  MP4
                # random seeking is much more expensive than decoding the
                # intervening frames, especially on CPFS.  Seek once and walk
                # forward so a 9-frame sample performs one seek, not nine.
                first, last = requested[0], requested[-1]
                requested_set = set(requested)
                capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                for index in range(first, last + 1):
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(
                            f"Failed to decode RGB frame {index} from valid stream: {path}"
                        )
                    if index in requested_set:
                        output[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                for index in requested:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(
                            f"Failed to decode RGB frame {index} from valid stream: {path}"
                        )
                    output[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return output
        finally:
            capture.release()

    @staticmethod
    def _decode_cached_rgb_frames(
        cache: h5py.File,
        camera: str,
        indices: Sequence[int],
    ) -> dict[int, np.ndarray]:
        dataset = cache[f"rgb/{camera}"]
        output = {}
        for index in sorted(set(int(value) for value in indices)):
            encoded = np.asarray(dataset[index], dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(
                    f"Failed to decode cached RGB frame {index}: {cache.filename}/{camera}"
                )
            output[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return output

    def _t_shape_frames(
        self,
        record: DexJoCoEpisode,
        rgb_indices: Sequence[int],
        cache: h5py.File | None = None,
    ) -> list[torch.Tensor]:
        video_root = record.root / "videos" / "rgb"
        decode = (
            lambda camera: self._decode_cached_rgb_frames(cache, camera, rgb_indices)
            if cache is not None
            else self._decode_video_frames(video_root / f"{camera}.mp4", rgb_indices)
        )
        third_frames = decode("third_person")
        left_frames = (
            decode("left_wrist")
            if record.camera_validity[1]
            else {}
        )
        right_frames = (
            decode("right_wrist")
            if record.camera_validity[2]
            else {}
        )
        output: list[torch.Tensor] = []
        for rgb_index in rgb_indices:
            third = third_frames[int(rgb_index)]
            h, w = third.shape[:2]
            if record.source_embodiment == "single":
                # DexJoCo/OpenPI stores the sole single-arm wrist stream in the
                # conventional `left_wrist` image key even though the Allegro
                # hand and the V3 TacAE registry are canonical RIGHT-hand data.
                # V2.5 deliberately makes all three modalities agree: the
                # single 22-D state/action, 15-pad tactile, and sole wrist RGB
                # are represented as the right arm.  The absent left view is
                # real padding, not a second copy of the wrist image.
                if not record.camera_validity[1] or record.camera_validity[2]:
                    raise ValueError(
                        "Single-arm canonical-right packing requires exactly "
                        f"the stored left_wrist stream: {record.root}"
                    )
                left_source = np.zeros_like(third)
                right_source = left_frames[int(rgb_index)]
            else:
                left_source = (
                    left_frames[int(rgb_index)]
                    if record.camera_validity[1]
                    else np.zeros_like(third)
                )
                right_source = (
                    right_frames[int(rgb_index)]
                    if record.camera_validity[2]
                    else np.zeros_like(third)
                )
            left = cv2.resize(left_source, (w // 2, h // 2))
            right = cv2.resize(right_source, (w // 2, h // 2))
            packed = np.concatenate(
                (third, np.concatenate((left, right), axis=1)), axis=0
            )
            packed = self._resize_with_padding(packed, self.video_size)
            output.append(
                torch.from_numpy(np.ascontiguousarray(packed))
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
            )
        return output

    def _record(self, index: int) -> DexJoCoEpisode:
        if self.val:
            return self.records[(index // self.samples_per_episode) % len(self.records)]
        if self._global_homogeneous_batch_size is not None:
            # Six single and five bimanual blocks per 11-block cycle preserve
            # equal marginal probability for every task: (6/11)/6 =
            # (5/11)/5 = 1/11. Interleaving avoids long same-embodiment runs.
            source_cycle = (
                "single", "bimanual", "single", "bimanual", "single",
                "bimanual", "single", "bimanual", "single", "bimanual",
                "single",
            )
            source = source_cycle[
                int(index) // self._global_homogeneous_batch_size
                % len(source_cycle)
            ]
            tasks = self.tasks_by_embodiment[source]
            if not tasks:
                raise RuntimeError(f"No configured DexJoCo {source} tasks")
            task = random.choice(tasks)
            return random.choice(self.records_by_embodiment_and_task[source][task])
        return random.choice(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._record(index)
        rgb_actions = np.asarray(record.rgb_action_indices, dtype=np.int64)
        cache_path = record.training_cache_path
        cache = h5py.File(cache_path, "r") if cache_path.is_file() else None
        try:
            if cache is not None:
                if cache.attrs.get("schema") != "dexjoco_wtam_training_cache_v1":
                    raise ValueError(f"Unsupported DexJoCo training cache: {cache_path}")
                cached_rgb_actions = np.asarray(
                    cache["rgb_source_action_index"][:], dtype=np.int64
                )
                if not np.array_equal(cached_rgb_actions, rgb_actions):
                    raise ValueError(
                        f"Cached RGB/native index table is stale: {cache_path}"
                    )
            if self.alignment_mode in {
                "native_dense20_video12_v1",
                "native_sparse40_video16_v1",
            }:
                # The v1 HDF5 cache contains every RGB frame, native action and
                # tactile frame.  Its precomputed *window table* encodes the old
                # sparse contract, so deliberately ignore only that tiny table
                # and derive Dense-20 indices here.  No RGB re-decode/cache
                # rebuild is needed.
                legal_anchor_rgb_indices = (
                    self.dense_anchor_rgb_indices[record.root]
                    if self.alignment_mode == "native_dense20_video12_v1"
                    else self.sparse40_anchor_rgb_indices[record.root]
                )
                anchor_rgb_index = (
                    int(legal_anchor_rgb_indices[index % len(legal_anchor_rgb_indices)])
                    if self.val
                    else int(random.choice(legal_anchor_rgb_indices))
                )
                (
                    anchor,
                    action_indices,
                    video_rgb_indices,
                    video_action_indices,
                ) = (
                    _dense_native_target_schedule(
                        rgb_actions,
                        anchor_rgb_index,
                        action_count=self.action_count,
                        num_video_frames=self.num_video_frames,
                    )
                    if self.alignment_mode == "native_dense20_video12_v1"
                    else _sparse40_native_target_schedule(
                        rgb_actions,
                        anchor_rgb_index,
                        action_count=self.action_count,
                        num_video_frames=self.num_video_frames,
                    )
                )
            elif cache is not None:
                window_count = int(cache["windows/anchor_rgb_index"].shape[0])
                window_index = index % window_count if self.val else random.randrange(window_count)
                anchor_rgb_index = int(cache["windows/anchor_rgb_index"][window_index])
                anchor = int(cache["windows/anchor_native_index"][window_index])
                video_rgb_all = np.asarray(
                    cache["windows/video_rgb_indices"][window_index], dtype=np.int64
                ).tolist()
                video_native_all = np.asarray(
                    cache["windows/video_native_indices"][window_index], dtype=np.int64
                ).tolist()
                video_rgb_indices = video_rgb_all[1:]
                video_action_indices = video_native_all[1:]
                action_indices = np.asarray(
                    cache["windows/action_native_indices"][window_index], dtype=np.int64
                ).tolist()
            else:
                first_anchor_rgb = int(np.searchsorted(rgb_actions, 1, side="left"))
                last_anchor_rgb = len(rgb_actions) - 1 - self.action_rgb_frame_offsets[-1]
                anchor_count = last_anchor_rgb - first_anchor_rgb + 1
                if anchor_count < 1:
                    raise ValueError(f"Episode is shorter than the WTAM horizon: {record.root}")
                anchor_rgb_index = (
                    first_anchor_rgb + index % anchor_count
                    if self.val
                    else random.randint(first_anchor_rgb, last_anchor_rgb)
                )
                anchor = int(rgb_actions[anchor_rgb_index])
                action_rgb_indices = [
                    anchor_rgb_index + offset for offset in self.action_rgb_frame_offsets
                ]
                action_indices = [int(rgb_actions[value]) for value in action_rgb_indices]
                video_rgb_indices = [
                    anchor_rgb_index + offset for offset in self.video_rgb_frame_offsets
                ]
                video_action_indices = [int(rgb_actions[value]) for value in video_rgb_indices]

            if self.alignment_mode == "rgb_index_exact_5p5frame_v1":
                if video_action_indices != action_indices[1::2]:
                    raise ValueError("Video targets are not the even Action/Tactile targets")
            elif self.alignment_mode == "native_dense20_video12_v1":
                if action_indices != list(range(anchor + 1, anchor + 21)):
                    raise ValueError("Dense-20 Action/Tactile targets are not contiguous")
                if video_action_indices[-1] != action_indices[-1]:
                    raise ValueError("Dense-20 modalities do not share one endpoint")
                if any(value not in set(action_indices) for value in video_action_indices):
                    raise ValueError("Dense-20 RGB target is outside the A/T horizon")
            else:
                expected_actions = [anchor + 2 * value for value in range(1, 41)]
                expected_video = [anchor + 5 * value for value in range(1, 17)]
                if action_indices != expected_actions:
                    raise ValueError("Sparse-40 Action/Tactile targets are not native stride-2")
                if video_action_indices != expected_video:
                    raise ValueError("Video-16 targets are not native stride-5")
                if video_action_indices[-1] != action_indices[-1]:
                    raise ValueError("Sparse-40 modalities do not share the +80 endpoint")
                common = set(video_action_indices).intersection(action_indices)
                if common != set(range(anchor + 10, anchor + 81, 10)):
                    raise ValueError("Sparse-40 cross-modal +10 alignment is incomplete")
            selected_rgb_actions = np.asarray([anchor, *video_action_indices], dtype=np.int64)
            requested_rgb_actions = np.asarray([anchor, *video_action_indices], dtype=np.int64)
            rgb_alignment_error_steps = selected_rgb_actions - requested_rgb_actions
            if np.abs(rgb_alignment_error_steps).max(initial=0) != 0:
                raise ValueError(f"RGB/action/tactile alignment is not exact: {record.root}")

            source_dim = 22 if record.source_embodiment == "single" else 44
            if cache is not None:
                trajectory = cache["trajectory"]
                source_actions = np.asarray(
                    trajectory[f"action_{source_dim}d"][action_indices], dtype=np.float32
                )
                state_key = f"policy_state_{source_dim}d"
                if self.state_source == "policy_state":
                    policy_state = cache[f"derived/{state_key}"]
                    source_state = np.asarray(policy_state[anchor], dtype=np.float32)
                else:
                    source_state = np.asarray(
                        trajectory[f"action_{source_dim}d"][anchor], dtype=np.float32
                    )
                observed_indices = [max(anchor - 1, 0), anchor]
                aligned_native_indices = observed_indices + action_indices
                control_times = _h5_take_allow_duplicates(
                    trajectory["sim_time_post_s"],
                    aligned_native_indices,
                    dtype=np.float64,
                )
                if self.include_tactile:
                    force = cache["tactile/force_canonical"]
                    observed = _h5_take_allow_duplicates(
                        force, observed_indices, dtype=np.float32
                    )
                    future = np.asarray(force[action_indices], dtype=np.float32)
                    grid_mask = np.asarray(cache["tactile/taxel_valid"][:], dtype=bool)
                    names = tuple(
                        value.decode() if isinstance(value, bytes) else str(value)
                        for value in cache["tactile/pad_name"][:].tolist()
                    )
                    times = _h5_take_allow_duplicates(
                        cache["tactile/sim_time_end_s"],
                        aligned_native_indices,
                        dtype=np.float64,
                    )
            else:
                with np.load(record.trajectory_path, mmap_mode="r") as trajectory:
                    source_actions = np.asarray(
                        trajectory[f"action_{source_dim}d"][action_indices], dtype=np.float32
                    )
                    state_key = f"policy_state_{source_dim}d"
                    if self.state_source == "policy_state":
                        state_path = record.root / "derived" / f"{state_key}.npy"
                        if not state_path.is_file():
                            raise KeyError(f"Missing derived policy state: {state_path}")
                        policy_state = np.load(state_path, mmap_mode="r")
                        source_state = np.asarray(policy_state[anchor], dtype=np.float32)
                    else:
                        source_state = np.asarray(
                            trajectory[f"action_{source_dim}d"][anchor], dtype=np.float32
                        )
                    observed_indices = [max(anchor - 1, 0), anchor]
                    aligned_native_indices = observed_indices + action_indices
                    control_times = np.asarray(
                        trajectory["sim_time_post_s"][aligned_native_indices], dtype=np.float64
                    )
                if self.include_tactile:
                    with h5py.File(record.tactile_path, "r") as handle:
                        force = handle["tactile_force_field/force_canonical"]
                        observed = _h5_take_allow_duplicates(
                            force, observed_indices, dtype=np.float32
                        )
                        future = np.asarray(force[action_indices], dtype=np.float32)
                        grid_mask = np.asarray(
                            handle["tactile_force_field_metadata/taxel_valid"][:], dtype=bool
                        )
                        names = tuple(
                            value.decode() if isinstance(value, bytes) else str(value)
                            for value in handle["tactile_force_field_metadata/pad_name"][:].tolist()
                        )
                        times = _h5_take_allow_duplicates(
                            handle["tactile_force_field/sim_time_end_s"],
                            aligned_native_indices,
                            dtype=np.float64,
                        )

            source_regions = 15 if record.source_embodiment == "single" else 30
            if self.state_source == "policy_state" and source_state.shape != (source_dim,):
                raise ValueError(f"Invalid policy-state shape at {record.root}: {source_state.shape}")
            if self.embodiment == "multitask" and source_dim == 22:
                actions = _pad_right_arm_22_to_44(source_actions)
                state = _pad_right_arm_22_to_44(source_state)
            else:
                actions, state = source_actions, source_state
            if self.include_tactile:
                if force.shape[0] != record.length or int(force.shape[1]) != source_regions:
                    raise ValueError(f"Tactile/action alignment mismatch: {record.root}")
                if not np.allclose(times, control_times, atol=1e-8, rtol=0):
                    raise ValueError(f"Action/tactile timestamp mismatch at {record.root}")
                if len(names) != source_regions or grid_mask.shape != (source_regions, 10, 14):
                    raise ValueError(f"Invalid DexJoCo tactile registry: {record.root}")
                if self.embodiment == "multitask" and source_regions == 15:
                    observed, future, grid_mask = _pad_right_tactile_15_to_30(
                        observed, future, grid_mask
                    )
                    names = tuple(f"RIGHT_{name}" for name in names) + tuple(
                        f"LEFT_PAD_{name}" for name in RAW_PAD_NAMES
                    )
            if not np.isfinite(actions).all() or not np.isfinite(state).all():
                raise ValueError(f"Non-finite state/action: {record.root}")
            t_frames = self._t_shape_frames(
                record, [anchor_rgb_index, *video_rgb_indices], cache=cache
            )
            sample = {
                "first_frame": t_frames[0],
                "video_frames": torch.stack(t_frames[1:]),
                "initial_state": torch.from_numpy(state.copy()),
                "action_sequence": torch.from_numpy(actions.copy()),
                "language_embedding": self.t5_cache.embedding(record.prompt),
                "vlm_inputs": None,
                "task_name": record.task_name,
                "source_embodiment": record.source_embodiment,
                "episode_name": record.episode_name,
                "condition_frame_idx": anchor,
                "condition_rgb_frame_idx": anchor_rgb_index,
                "video_indices": video_rgb_indices,
                "video_source_action_indices": torch.from_numpy(selected_rgb_actions.copy()),
                "video_alignment_error_steps_50hz": torch.from_numpy(
                    rgb_alignment_error_steps.astype(np.int64, copy=True)
                ),
                "action_indices": action_indices,
            }
            if self.include_tactile:
                times = (times - times[0]).astype(np.float32)
                sample.update({
                    "tactile_observed_source": torch.from_numpy(np.moveaxis(observed, 1, 0).copy()),
                    "tactile_future_source": torch.from_numpy(np.moveaxis(future, 1, 0).copy()),
                    "tactile_grid_mask_source": torch.from_numpy(grid_mask.copy()),
                    "tactile_observed_frame_times": torch.from_numpy(times[:2].copy()),
                    "tactile_future_query_times": torch.from_numpy(times[2:].copy()),
                    "tactile_pad_names": names,
                })
            return sample
        finally:
            if cache is not None:
                cache.close()
