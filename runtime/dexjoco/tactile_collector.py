#!/usr/bin/env python3
"""DJ-TFA-2.0-palmar tactile collector for the DexJoCo Allegro asset.

This module is deliberately independent from the legacy TFA-1.0 collector.
The frozen 16-region geometry matches ``collision_surface_v4_16pad``:
12 palmar +X box faces and four capsule-following fingertip surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import h5py
except ModuleNotFoundError:  # Online evaluation does not read or write HDF5.
    h5py = None
import mujoco
import numpy as np


SPEC_VERSION = "DJ-TFA-2.0-palmar"
COLLECTOR_REVISION = "collision_surface_v4_16pad_replay_v1"
SURFACE_GEOMETRY = (
    "collision_surface_v4_16pad: palmar +x box faces, proximal coverage, "
    "capsule-following fingertip arcs, palm outer face"
)
CHANNELS = ("Fn", "Fu", "Fv")
ROWS = 10
COLS = 14
CAPSULE_THETA_MIN = -5.0 / 6.0
CAPSULE_THETA_MAX = 5.0 / 6.0


@dataclass(frozen=True)
class RegionDefinition:
    name: str
    body_stem: str
    kind: str
    surface_center_body_m: tuple[float, float, float]
    u_half_extent_m: float
    v_half_extent_m: float
    radius_m: float = 0.0


# Region order, centers and supports are frozen from the colleague data's
# tactile metadata.  PROX maps to the long proximal/medial collision link;
# BASE maps to medial; PAD maps to distal.
REGION_DEFINITIONS = (
    RegionDefinition("PALM_1", "allegro_palm", "plane", (0.0111, 0.0, -0.0475), 0.0505, 0.0425),
    RegionDefinition("TH_PROX", "th_medial", "plane", (0.0098, 0.0, 0.0257), 0.0120, 0.0240),
    RegionDefinition("TH_PAD", "th_distal", "plane", (0.0098, 0.0, 0.0157), 0.0120, 0.0140),
    RegionDefinition("TH_TIP", "th_tip", "capsule", (0.0120, 0.0, 0.0350), 0.0, 0.0080, 0.0120),
    RegionDefinition("FF_PROX", "ff_proximal", "plane", (0.0098, 0.0, 0.0270), 0.0120, 0.0250),
    RegionDefinition("FF_BASE", "ff_medial", "plane", (0.0098, 0.0, 0.0192), 0.0120, 0.0170),
    RegionDefinition("FF_PAD", "ff_distal", "plane", (0.0098, 0.0, 0.0080), 0.0120, 0.0070),
    RegionDefinition("FF_TIP", "ff_tip", "capsule", (0.0120, 0.0, 0.0190), 0.0, 0.0090, 0.0120),
    RegionDefinition("MF_PROX", "mf_proximal", "plane", (0.0098, 0.0, 0.0270), 0.0120, 0.0250),
    RegionDefinition("MF_BASE", "mf_medial", "plane", (0.0098, 0.0, 0.0192), 0.0120, 0.0170),
    RegionDefinition("MF_PAD", "mf_distal", "plane", (0.0098, 0.0, 0.0080), 0.0120, 0.0070),
    RegionDefinition("MF_TIP", "mf_tip", "capsule", (0.0120, 0.0, 0.0190), 0.0, 0.0090, 0.0120),
    RegionDefinition("RF_PROX", "rf_proximal", "plane", (0.0098, 0.0, 0.0270), 0.0120, 0.0250),
    RegionDefinition("RF_BASE", "rf_medial", "plane", (0.0098, 0.0, 0.0192), 0.0120, 0.0170),
    RegionDefinition("RF_PAD", "rf_distal", "plane", (0.0098, 0.0, 0.0080), 0.0120, 0.0070),
    RegionDefinition("RF_TIP", "rf_tip", "capsule", (0.0120, 0.0, 0.0190), 0.0, 0.0090, 0.0120),
)


def _geometry_arrays() -> dict[str, np.ndarray]:
    count = len(REGION_DEFINITIONS)
    centers = np.asarray([r.surface_center_body_m for r in REGION_DEFINITIONS], dtype=np.float64)
    kinds = np.asarray([r.kind for r in REGION_DEFINITIONS], dtype=object)
    valid = np.ones((count, ROWS, COLS), dtype=bool)
    positions = np.zeros((count, ROWS, COLS, 3), dtype=np.float64)
    axes = np.zeros((count, ROWS, COLS, 3, 3), dtype=np.float64)
    area = np.zeros((count, ROWS, COLS), dtype=np.float64)
    for index, region in enumerate(REGION_DEFINITIONS):
        z_values = np.linspace(-region.v_half_extent_m, region.v_half_extent_m, ROWS)
        if region.kind == "plane":
            y_values = np.linspace(-region.u_half_extent_m, region.u_half_extent_m, COLS)
            for row, z_value in enumerate(z_values):
                for col, y_value in enumerate(y_values):
                    positions[index, row, col] = (0.0, y_value, z_value)
                    axes[index, row, col] = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            area[index] = (
                4.0 * region.u_half_extent_m * region.v_half_extent_m / (ROWS * COLS)
            )
        else:
            theta_values = np.linspace(CAPSULE_THETA_MIN, CAPSULE_THETA_MAX, COLS)
            for row, z_value in enumerate(z_values):
                for col, theta in enumerate(theta_values):
                    c, s = float(np.cos(theta)), float(np.sin(theta))
                    positions[index, row, col] = (
                        region.radius_m * (c - 1.0), region.radius_m * s, z_value
                    )
                    axes[index, row, col] = ((-c, -s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0))
            area[index] = (
                (CAPSULE_THETA_MAX - CAPSULE_THETA_MIN)
                * region.radius_m
                * (2.0 * region.v_half_extent_m)
                / (ROWS * COLS)
            )
    return {
        "surface_center_body_m": centers,
        "surface_kind": kinds,
        "taxel_valid": valid,
        "taxel_position_from_surface_center_body_m": positions,
        "taxel_axis_frame_body": axes,
        "taxel_area_m2": area,
    }


GEOMETRY = _geometry_arrays()


def registry_sha256() -> str:
    payload: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "collector_revision": COLLECTOR_REVISION,
        "regions": [r.__dict__ for r in REGION_DEFINITIONS],
        "rows": ROWS,
        "cols": COLS,
        "capsule_theta": [CAPSULE_THETA_MIN, CAPSULE_THETA_MAX],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_registry_reference(path: Path) -> dict[str, Any]:
    """Verify our generated geometry against a colleague-produced HDF5 file."""
    if h5py is None:
        raise ModuleNotFoundError("h5py is required for registry-file validation")
    with h5py.File(path, "r") as handle:
        group = handle["tactile"]
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in group["pad_name"][:]]
        start = next(i for i, name in enumerate(names) if name.endswith("/PALM_1"))
        if len(names) >= 32 and names[start].startswith("left/"):
            right = [i for i, name in enumerate(names) if name.startswith("right/")]
            indices = np.asarray(right, dtype=np.int64)
        else:
            indices = np.arange(start, start + len(REGION_DEFINITIONS), dtype=np.int64)
        actual_names = [names[i].split("/", 1)[-1] for i in indices]
        expected_names = [r.name for r in REGION_DEFINITIONS]
        checks: dict[str, bool] = {"pad_name": actual_names == expected_names}
        for key in (
            "surface_center_body_m", "taxel_valid",
            "taxel_position_from_surface_center_body_m", "taxel_axis_frame_body",
            "taxel_area_m2",
        ):
            reference = np.asarray(group[key][indices])
            expected = np.asarray(GEOMETRY[key])
            checks[key] = bool(
                np.array_equal(reference, expected)
                if reference.dtype == bool
                else np.allclose(reference, expected, atol=2.0e-8, rtol=2.0e-6)
            )
        reference_kinds = [
            (v.decode() if isinstance(v, bytes) else str(v)) for v in group["surface_kind"][indices]
        ]
        checks["surface_kind"] = reference_kinds == list(GEOMETRY["surface_kind"])
        reference_attr = str(group.attrs.get("surface_geometry", ""))
        checks["surface_geometry"] = reference_attr == SURFACE_GEOMETRY
    report = {
        "reference_h5": str(path),
        "registry_sha256": registry_sha256(),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        raise RuntimeError(f"DJ-TFA-2.0 registry mismatch: {report}")
    return report


class DJTFA2PalmarCollector:
    """Accumulate one 16-region palmar force field for one Allegro hand."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        hand: str,
        suffix: str,
        surface_tolerance_m: float = 0.003,
        fn_epsilon_n: float = 1.0e-9,
    ) -> None:
        if hand not in {"left", "right"}:
            raise ValueError(f"invalid hand {hand!r}")
        self.model = model
        self.hand = hand
        self.suffix = suffix
        self.surface_tolerance_m = float(surface_tolerance_m)
        self.fn_epsilon_n = float(fn_epsilon_n)
        self.regions = REGION_DEFINITIONS
        self.body_ids: list[int] = []
        self.body_to_region: dict[int, int] = {}
        for index, region in enumerate(self.regions):
            body_name = region.body_stem + suffix
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise RuntimeError(f"{hand}/{region.name}: missing body {body_name}")
            if body_id in self.body_to_region:
                raise RuntimeError(f"duplicate tactile body mapping for {body_name}")
            self.body_ids.append(body_id)
            self.body_to_region[body_id] = index
        self.rows: list[dict[str, Any]] = []
        self._reset_interval()

    @property
    def pad_names(self) -> list[str]:
        return [region.name for region in self.regions]

    def _reset_interval(self) -> None:
        shape = (len(self.regions), 3, ROWS, COLS)
        count_shape = (len(self.regions),)
        self._grid_impulse = np.zeros(shape, dtype=np.float64)
        self._net_force_impulse = np.zeros((len(self.regions), 3), dtype=np.float64)
        self._raw_contact_count = 0
        self._registered_body_contact_count = np.zeros(count_shape, dtype=np.int32)
        self._accepted_contact_count = np.zeros(count_shape, dtype=np.int32)
        self._rejected_nonfinite_count = np.zeros(count_shape, dtype=np.int32)
        self._rejected_wrong_surface_count = np.zeros(count_shape, dtype=np.int32)
        self._rejected_surface_distance_count = np.zeros(count_shape, dtype=np.int32)
        self._rejected_uv_count = np.zeros(count_shape, dtype=np.int32)
        self._rejected_nonpositive_normal_count = np.zeros(count_shape, dtype=np.int32)
        self._physics_steps = 0
        self._elapsed_s = 0.0

    @staticmethod
    def _force_on_region_world(
        contact: mujoco.MjContact, wrench: np.ndarray, region_is_geom1: bool
    ) -> np.ndarray:
        contact_axes = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world_on_geom2 = contact_axes.T @ wrench[:3]
        return -force_world_on_geom2 if region_is_geom1 else force_world_on_geom2

    @staticmethod
    def _splat_bilinear(
        grid: np.ndarray, row_float: float, col_float: float, impulse: np.ndarray
    ) -> None:
        row0, col0 = int(np.floor(row_float)), int(np.floor(col_float))
        row1, col1 = min(row0 + 1, ROWS - 1), min(col0 + 1, COLS - 1)
        wr, wc = row_float - row0, col_float - col0
        for row, col, weight in (
            (row0, col0, (1.0 - wr) * (1.0 - wc)),
            (row0, col1, (1.0 - wr) * wc),
            (row1, col0, wr * (1.0 - wc)),
            (row1, col1, wr * wc),
        ):
            if weight:
                grid[:, row, col] += weight * impulse

    def capture_substep(self, data: mujoco.MjData) -> None:
        dt = float(self.model.opt.timestep)
        self._physics_steps += 1
        self._elapsed_s += dt
        self._raw_contact_count += int(data.ncon)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            for geom_id, is_geom1 in ((int(contact.geom1), True), (int(contact.geom2), False)):
                region_index = self.body_to_region.get(int(self.model.geom_bodyid[geom_id]))
                if region_index is None:
                    continue
                self._registered_body_contact_count[region_index] += 1
                wrench = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, data, contact_index, wrench)
                force_world = self._force_on_region_world(contact, wrench, is_geom1)
                body_id = self.body_ids[region_index]
                rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                position = np.asarray(data.xpos[body_id], dtype=np.float64)
                force_body = rotation.T @ force_world
                point_body = rotation.T @ (np.asarray(contact.pos, dtype=np.float64) - position)
                if not np.isfinite(force_body).all() or not np.isfinite(point_body).all():
                    self._rejected_nonfinite_count[region_index] += 1
                    continue

                region = self.regions[region_index]
                center = np.asarray(region.surface_center_body_m, dtype=np.float64)
                if region.kind == "plane":
                    delta = point_body - center
                    surface_distance = abs(float(delta[0]))
                    u_coord, v_coord = float(delta[1]), float(delta[2])
                    intended_surface = point_body[0] > 0.0
                    axes = np.asarray(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
                    col = (u_coord / (2.0 * region.u_half_extent_m) + 0.5) * (COLS - 1)
                    row = (v_coord / (2.0 * region.v_half_extent_m) + 0.5) * (ROWS - 1)
                else:
                    radial_norm = float(np.hypot(point_body[0], point_body[1]))
                    theta = float(np.arctan2(point_body[1], point_body[0]))
                    v_coord = float(point_body[2] - center[2])
                    surface_distance = abs(radial_norm - region.radius_m)
                    intended_surface = CAPSULE_THETA_MIN <= theta <= CAPSULE_THETA_MAX
                    c, s = float(np.cos(theta)), float(np.sin(theta))
                    axes = np.asarray(((-c, -s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))
                    col = (
                        (theta - CAPSULE_THETA_MIN)
                        / (CAPSULE_THETA_MAX - CAPSULE_THETA_MIN)
                        * (COLS - 1)
                    )
                    row = (v_coord / (2.0 * region.v_half_extent_m) + 0.5) * (ROWS - 1)

                if not intended_surface:
                    self._rejected_wrong_surface_count[region_index] += 1
                    continue
                if surface_distance > self.surface_tolerance_m:
                    self._rejected_surface_distance_count[region_index] += 1
                    continue
                if row < 0.0 or row > ROWS - 1 or col < 0.0 or col > COLS - 1:
                    self._rejected_uv_count[region_index] += 1
                    continue
                canonical = axes @ force_body
                if canonical[0] <= self.fn_epsilon_n:
                    self._rejected_nonpositive_normal_count[region_index] += 1
                    continue
                impulse = canonical * dt
                self._splat_bilinear(self._grid_impulse[region_index], row, col, impulse)
                self._net_force_impulse[region_index] += impulse
                self._accepted_contact_count[region_index] += 1

    def record_empty_interval(
        self, data: mujoco.MjData, *, observation_index: int, sim_time_end_s: float,
        interval_seconds: float,
    ) -> None:
        if self._physics_steps:
            raise RuntimeError("cannot replace a non-empty DJ-TFA interval")
        self._physics_steps = 1
        self._elapsed_s = float(interval_seconds)
        self.flush_interval(data, observation_index=observation_index, sim_time_end_s=sim_time_end_s)

    def flush_interval(
        self, data: mujoco.MjData, *, observation_index: int, sim_time_end_s: float
    ) -> None:
        if self._physics_steps < 1 or self._elapsed_s <= 0.0:
            raise RuntimeError("cannot flush an empty DJ-TFA interval")
        pose = np.zeros((len(self.regions), 7), dtype=np.float32)
        for index, body_id in enumerate(self.body_ids):
            pose[index, :3] = data.xpos[body_id]
            pose[index, 3:] = data.xquat[body_id]
        elapsed = self._elapsed_s
        self.rows.append({
            "force_canonical": (self._grid_impulse / elapsed).astype(np.float32),
            "net_force": (self._net_force_impulse / elapsed).astype(np.float32),
            "raw_contact_count": int(self._raw_contact_count),
            "registered_body_contact_count": self._registered_body_contact_count.copy(),
            "accepted_contact_count": self._accepted_contact_count.copy(),
            "rejected_nonfinite_count": self._rejected_nonfinite_count.copy(),
            "rejected_wrong_surface_count": self._rejected_wrong_surface_count.copy(),
            "rejected_surface_distance_count": self._rejected_surface_distance_count.copy(),
            "rejected_uv_count": self._rejected_uv_count.copy(),
            "rejected_nonpositive_normal_count": self._rejected_nonpositive_normal_count.copy(),
            "interval_seconds": float(elapsed),
            "physics_steps": int(self._physics_steps),
            "pad_pose_world": pose,
            "observation_index": int(observation_index),
            "sim_time_end_s": float(sim_time_end_s),
        })
        self._reset_interval()


def make_collectors(
    model: mujoco.MjModel, *, bimanual: bool, surface_tolerance_m: float,
    fn_epsilon_n: float,
) -> dict[str, DJTFA2PalmarCollector]:
    if bimanual:
        return {
            "right": DJTFA2PalmarCollector(model, hand="right", suffix="_right", surface_tolerance_m=surface_tolerance_m, fn_epsilon_n=fn_epsilon_n),
            "left": DJTFA2PalmarCollector(model, hand="left", suffix="_left", surface_tolerance_m=surface_tolerance_m, fn_epsilon_n=fn_epsilon_n),
        }
    try:
        right = DJTFA2PalmarCollector(model, hand="right", suffix="_right", surface_tolerance_m=surface_tolerance_m, fn_epsilon_n=fn_epsilon_n)
    except RuntimeError:
        right = DJTFA2PalmarCollector(model, hand="right", suffix="", surface_tolerance_m=surface_tolerance_m, fn_epsilon_n=fn_epsilon_n)
    return {"right": right}
