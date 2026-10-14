"""VAT-v1 temporal contract with DJ-TFA-2.0-palmar physical tactile input.

This is an additive loader.  The TFA-1.0 loader and its 15/30-pad defaults are
unchanged.  New samples always expose 32 physical surfaces in canonical
left-then-right order; a missing hand has support=False rather than pretending
that padding is an observed zero-contact sensor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import cv2
import numpy as np
import torch

from training.dexjoco.data.dexjoco_raw_wtam_dataset import (
    DexJoCoRawAlignedWTAMDataset,
    _h5_take_allow_duplicates,
)
from training.dexjoco.data.dexjoco_wtam_dataset import DexJoCoWTAMDataset


DJ_TFA2_PADS = (
    "PALM_1", "TH_PROX", "TH_PAD", "TH_TIP", "FF_PROX", "FF_BASE",
    "FF_PAD", "FF_TIP", "MF_PROX", "MF_BASE", "MF_PAD", "MF_TIP",
    "RF_PROX", "RF_BASE", "RF_PAD", "RF_TIP",
)
DJ_TFA2_SCHEMA = "dexjoco_11task_tfa_alignment_v3_dj_tfa2_palmar"


def _names(values: Any) -> tuple[str, ...]:
    return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)


class _NumpyOrthogonalIndex:
    """Expose the tiny subset of the Zarr array API used by the base loader."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array
        self.oindex = self

    def __getitem__(self, index: Any) -> np.ndarray:
        return self._array[index]


class DexJoCoRawAlignedDJTFA2WTAMDataset(DexJoCoRawAlignedWTAMDataset):
    """Read VAT-v1 A, selectable aligned V, and audited 16-pad palmar T."""

    def __init__(
        self,
        *args: Any,
        manifest_filename: str = "accepted_manifest_dj_tfa2.json",
        include_tactile: bool = True,
        video_source_policy: str = "original",
        video_layout: str = "t_vertical",
        **kwargs: Any,
    ) -> None:
        if video_source_policy not in {"original", "prefer_replayed_fixed"}:
            raise ValueError(
                "video_source_policy must be 'original' or "
                "'prefer_replayed_fixed'"
            )
        self._dj_tfa2_include_tactile = bool(include_tactile)
        self.video_source_policy = str(video_source_policy)
        if video_layout not in {"t_vertical", "horizontal"}:
            raise ValueError("video_layout must be 't_vertical' or 'horizontal'")
        self.video_layout = video_layout
        super().__init__(
            *args,
            manifest_filename=manifest_filename,
            manifest_schema=DJ_TFA2_SCHEMA,
            include_tactile=False,
            **kwargs,
        )
        self.region_count = 32
        # Trainer/model capability checks inspect this public flag.
        self.include_tactile = self._dj_tfa2_include_tactile
        self.tactile_protocol = "DJ-TFA-2.0-palmar"
        self.tactile_surface_order = "left16_then_right16"
        self._corrected_video_directories: dict[tuple[str, int], Path] = {}
        self._corrected_action_arrays: dict[
            tuple[str, int], _NumpyOrthogonalIndex
        ] = {}
        if self.video_source_policy == "prefer_replayed_fixed":
            self._load_corrected_video_directories(manifest_filename)
        self.video_source_counts = {
            "replayed_fixed": len(self._corrected_video_directories),
            "original_demonstration": (
                len(self.records) - len(self._corrected_video_directories)
            ),
        }
        self.action_source_counts = {
            "state_tracking_v1": len(self._corrected_action_arrays),
            "original_replay_zarr": len(self.records) - len(self._corrected_action_arrays),
        }
        if self.video_layout == "horizontal" and self.video_size != (256, 384):
            raise ValueError("horizontal video_layout requires video_size=(256, 384)")

    def _frames(self, record, raw_indices):
        if self.video_layout == "t_vertical":
            return super()._frames(record, raw_indices)

        videos = self._video_directory(record)
        if record.source_embodiment == "single":
            third_name = "ego_right.mp4" if record.task_name == "click_mouse" else "front.mp4"
            front_path = videos / third_name
            right_path = videos / "wrist.mp4"
            left_path = None
        else:
            front_path = videos / "ego.mp4"
            right_path = videos / "wrist_right.mp4"
            left_path = videos / "wrist_left.mp4"
        front = DexJoCoWTAMDataset._decode_video_frames(
            front_path, raw_indices, sequential=self.sequential_video_decode
        )
        right = DexJoCoWTAMDataset._decode_video_frames(
            right_path, raw_indices, sequential=self.sequential_video_decode
        )
        left = {} if left_path is None else DexJoCoWTAMDataset._decode_video_frames(
            left_path, raw_indices, sequential=self.sequential_video_decode
        )

        result = []
        for index in raw_indices:
            third = front[int(index)]
            h, w = third.shape[:2]
            if h != w or h % 2:
                raise ValueError(
                    f"horizontal layout requires even square front frames, got {(h, w)}"
                )
            left_image = np.zeros_like(third) if left_path is None else left[int(index)]
            # Front on the left, left wrist above right wrist on the right.
            wrists = np.concatenate((
                cv2.resize(left_image, (w // 2, h // 2)),
                cv2.resize(right[int(index)], (w // 2, h // 2)),
            ), axis=0)
            packed = np.concatenate((third, wrists), axis=1)
            # The 3:2 mosaic exactly matches 256x384, so resize directly:
            # no aspect-ratio padding or one-pixel rounding bars.
            packed = cv2.resize(packed, (self.video_size[1], self.video_size[0]))
            result.append(
                torch.from_numpy(np.ascontiguousarray(packed))
                .permute(2, 0, 1).float().div_(255)
            )
        return result

    def _load_corrected_video_directories(self, manifest_filename: str) -> None:
        """Resolve and validate every corrected-video claim in the cohort.

        A missing manifest claim means the episode intentionally falls back to
        its original demonstration. A claim that points to incomplete files is
        treated as dataset corruption and fails fast instead of silently
        changing the requested experiment.
        """
        manifest_path = Path(manifest_filename)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = {(record.task_name, record.episode_index) for record in self.records}
        for task in self.task_names:
            for item in manifest["tasks"][task]:
                key = (task, int(item["episode_index"]))
                if key not in selected:
                    continue
                declared = item.get("corrected_replay_video_directory")
                if not declared:
                    continue
                video_dir = Path(declared)
                if not video_dir.is_absolute():
                    video_dir = self.root / video_dir
                report_name = item.get("corrected_replay_video_report")
                if not report_name:
                    raise ValueError(f"{task}/{key[1]}: corrected video has no report")
                report_path = Path(report_name)
                if not report_path.is_absolute():
                    report_path = self.root / report_path
                if not video_dir.is_dir() or not report_path.is_file():
                    raise FileNotFoundError(
                        f"{task}/{key[1]}: incomplete corrected replay video: "
                        f"{video_dir}, {report_path}"
                    )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (
                    report.get("schema") != "dexjoco_repaired_replay_video_v1"
                    or report.get("status") != "complete"
                    or report.get("task") != task
                    or int(report.get("episode_index", -1)) != key[1]
                ):
                    raise ValueError(f"{report_path}: invalid corrected-video report")
                record = next(
                    record for record in self.records
                    if (record.task_name, record.episode_index) == key
                )
                expected_names = (
                    ("ego_right.mp4", "wrist.mp4")
                    if record.source_embodiment == "single" and task == "click_mouse"
                    else ("front.mp4", "wrist.mp4")
                    if record.source_embodiment == "single"
                    else ("ego.mp4", "wrist_right.mp4", "wrist_left.mp4")
                )
                raw_stop = record.crop_start + record.length
                report_videos = report.get("videos", {})
                for name in expected_names:
                    path = video_dir / name
                    frames = int(report_videos.get(name, {}).get("frames", -1))
                    if not path.is_file() or frames < raw_stop:
                        raise ValueError(
                            f"{task}/{key[1]}: {name} does not cover raw index "
                            f"{raw_stop - 1}"
                        )
                self._corrected_video_directories[key] = video_dir
                corrected_action_name = item.get("corrected_replay_action")
                if corrected_action_name:
                    corrected_action_path = Path(corrected_action_name)
                    if not corrected_action_path.is_absolute():
                        corrected_action_path = self.root / corrected_action_path
                    if not corrected_action_path.is_file():
                        raise FileNotFoundError(corrected_action_path)
                    with np.load(corrected_action_path) as payload:
                        if "corrected_action_rotvec" not in payload:
                            raise ValueError(
                                f"{corrected_action_path}: missing corrected_action_rotvec"
                            )
                        corrected_action = np.asarray(
                            payload["corrected_action_rotvec"], dtype=np.float32
                        ).copy()
                    expected_dim = 44 if record.source_embodiment == "bimanual" else 22
                    if (
                        corrected_action.ndim != 2
                        or corrected_action.shape[1] != expected_dim
                        or corrected_action.shape[0] < raw_stop
                        or not np.isfinite(corrected_action).all()
                    ):
                        raise ValueError(
                            f"{corrected_action_path}: invalid corrected action shape/value "
                            f"{corrected_action.shape}, expected [T>={raw_stop},{expected_dim}]"
                        )
                    self._corrected_action_arrays[key] = _NumpyOrthogonalIndex(
                        corrected_action
                    )

    def _video_directory(self, record):
        return self._corrected_video_directories.get(
            (record.task_name, record.episode_index),
            super()._video_directory(record),
        )

    def _zarr_array(self, record, key: str):
        if key == "action_rotvec":
            corrected = self._corrected_action_arrays.get(
                (record.task_name, record.episode_index)
            )
            if corrected is not None:
                return corrected
        return super()._zarr_array(record, key)

    def _window_for_index(self, index: int):
        if not self.val and self.sampling_mode == "window_index":
            return self._resolve_window(index)
        record = self._record(index)
        if self.val:
            anchor = index % max(1, record.length - self.action_count * self.action_stride)
        else:
            anchor = self._rng_for_index(
                index, self._current_sampling_epoch()
            ).randrange(record.length - self.action_count * self.action_stride)
        return record, anchor

    @staticmethod
    def _read_hand(handle: h5py.File, hand: str, indices: np.ndarray):
        if hand not in handle:
            return None
        group = handle[hand]
        names = _names(group["metadata/pad_name"][:])
        if set(names) != set(DJ_TFA2_PADS) or len(names) != 16:
            raise ValueError(f"{handle.filename}/{hand}: invalid DJ-TFA-2.0 pad registry {names}")
        order = np.asarray([names.index(name) for name in DJ_TFA2_PADS], dtype=np.int64)
        force = _h5_take_allow_duplicates(
            group["force_canonical"], indices, dtype=np.float32
        )[:, order]
        taxel = np.asarray(group["metadata/taxel_valid"][:], dtype=bool)[order]
        region = np.asarray(group["metadata/region_valid"][:], dtype=bool)[order]
        channel = np.asarray(group["metadata/channel_valid"][:], dtype=bool)[order]
        if force.shape != (len(indices), 16, 3, 10, 14):
            raise ValueError(f"{handle.filename}/{hand}: force shape {force.shape}")
        if taxel.shape != (16, 10, 14) or region.shape != (16,) or channel.shape != (16, 3):
            raise ValueError(f"{handle.filename}/{hand}: invalid support metadata")
        if not channel.all():
            raise ValueError(f"{handle.filename}/{hand}: missing Fn/Fu/Fv channel")
        support = taxel & region[:, None, None]
        if not np.isfinite(force).all() or np.any(force[:, :, 0] < -1e-7):
            raise ValueError(f"{handle.filename}/{hand}: invalid force values")
        force *= support[None, :, None]
        return force, support

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Base path deliberately skips its old 15/30-pad tactile block.
        sample = super().__getitem__(index)
        if not self._dj_tfa2_include_tactile:
            return sample
        record, anchor = self._window_for_index(index)
        action_indices = np.arange(anchor + 3, anchor + 49, 3, dtype=np.int64)
        observed_indices = np.asarray([max(anchor - 1, 0), anchor], dtype=np.int64)
        all_indices = np.r_[observed_indices, action_indices]
        bimanual = record.source_embodiment == "bimanual"
        with h5py.File(record.tactile_path, "r") as handle:
            if str(handle.attrs.get("spec_version")) != "DJ-TFA-2.0-palmar":
                raise ValueError(f"{record.tactile_path}: protocol mismatch")
            if str(handle.attrs.get("force_units")) != "N":
                raise ValueError(f"{record.tactile_path}: expected Newtons")
            if json.loads(str(handle.attrs.get("force_channel_order"))) != ["Fn", "Fu", "Fv"]:
                raise ValueError(f"{record.tactile_path}: channel order mismatch")
            right = self._read_hand(handle, "right", all_indices)
            left = self._read_hand(handle, "left", all_indices)
            if right is None or (bimanual and left is None) or (not bimanual and left is not None):
                raise ValueError(f"{record.tactile_path}: hand-group/embodiment mismatch")
            times = _h5_take_allow_duplicates(
                handle["frame_time_seconds"], all_indices, dtype=np.float32
            )
        assert right is not None
        if left is None:
            left_force = np.zeros_like(right[0])
            left_support = np.zeros_like(right[1], dtype=bool)
        else:
            left_force, left_support = left
        force = np.concatenate((left_force, right[0]), axis=1)
        support = np.concatenate((left_support, right[1]), axis=0)
        qualified_names = tuple(
            f"{hand}/{name}" for hand in ("left", "right") for name in DJ_TFA2_PADS
        )
        sample.update({
            "tactile_observed_source": torch.from_numpy(
                np.moveaxis(force[:2], 1, 0).copy()
            ),
            "tactile_future_source": torch.from_numpy(
                np.moveaxis(force[2:], 1, 0).copy()
            ),
            "tactile_grid_mask_source": torch.from_numpy(support.copy()),
            "tactile_observed_frame_times": torch.from_numpy((times[:2] - times[0]).copy()),
            "tactile_future_query_times": torch.from_numpy((times[2:] - times[0]).copy()),
            "tactile_pad_names": qualified_names,
            "tactile_protocol": "DJ-TFA-2.0-palmar",
            "tactile_surface_order": "left16_then_right16",
        })
        return sample
