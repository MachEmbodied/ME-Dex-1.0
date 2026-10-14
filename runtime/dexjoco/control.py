from __future__ import annotations
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

def _hold_action_from_observation(
    observation: dict[str, Any], *, dual_arm: bool
) -> np.ndarray:
    """Build the official stay() action: command the current pose.

    Mirrors DexJoCoOpenPIEnv.stay(): hold the robot's current pose by sending
    the observed state as the action. Every task's flattened state starts with
    tcp_pose followed by gripper_pose, which matches the env action layouts
    exactly ([pos3, quat4, hand16] single-arm, [r_pose7, l_pose7, r_hand16,
    l_hand16] dual-arm).
    """
    state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    size = 46 if dual_arm else 23
    if state.size < size:
        raise ValueError(f"Expected at least {size} state values, got {state.size}")
    return state[:size].copy()

def _geodesic_quat(prev_quat: np.ndarray, target_quat: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Slerp along the SO(3) geodesic from prev to target (shortest arc).

    Port of the colleague's _geodesic_rotvec: the relative rotation is taken
    as the shortest-arc rotvec (norm <= pi), so waypoints that straddle the
    +/-pi rotvec wrap-around interpolate over the true angular distance
    instead of detouring a full 2*pi turn.
    """
    rot_prev = Rotation.from_quat(prev_quat, scalar_first=True)
    delta = (rot_prev.inv() * Rotation.from_quat(target_quat, scalar_first=True)).as_rotvec()
    steps = np.empty((len(alphas), 4), dtype=np.float64)
    for i, alpha in enumerate(alphas):
        rot = rot_prev * Rotation.from_rotvec(alpha * delta)
        steps[i] = rot.as_quat(scalar_first=True)
        ref = prev_quat if i == 0 else steps[i - 1]
        if float(np.dot(steps[i], ref)) < 0.0:
            steps[i] = -steps[i]
    return steps

def _densify_sparse_waypoints(
    current_env_action: np.ndarray,
    waypoint_env_actions: list[np.ndarray],
    control_offsets: tuple[int, ...],
) -> list[np.ndarray]:
    """Expand sparse absolute waypoints into one command per 50 Hz step.

    Port of the colleague's densify_sparse_plan (inference/dexjoco/control.py):
    translation and hand joints interpolate linearly between consecutive
    waypoints, rotations follow the SO(3) geodesic. The ramp starts from the
    anchor action (the current state or last executed command), exactly like
    the colleague's evaluator re-anchors every replan.
    """
    if len(waypoint_env_actions) != len(control_offsets):
        raise ValueError(
            f"got {len(waypoint_env_actions)} waypoints for {len(control_offsets)} offsets"
        )
    if any(b <= a for a, b in zip(control_offsets, control_offsets[1:])):
        raise ValueError("waypoint control offsets must be strictly increasing")
    size = len(current_env_action)
    if size not in (23, 46):
        raise ValueError(f"expected a 23-D or 46-D env action, got {size}")
    dense: list[np.ndarray] = []
    previous_offset = 0
    previous = np.asarray(current_env_action, dtype=np.float64)
    for offset, target in zip(control_offsets, waypoint_env_actions):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (size,):
            raise ValueError(f"waypoint shape {target.shape} does not match anchor {size}")
        duration = offset - previous_offset
        alphas = np.arange(1, duration + 1, dtype=np.float64) / duration
        steps = np.empty((duration, size), dtype=np.float64)
        steps[:, :3] = previous[:3] + alphas[:, None] * (target[:3] - previous[:3])
        steps[:, 3:7] = _geodesic_quat(previous[3:7], target[3:7], alphas)
        steps[:, 7:] = previous[7:] + alphas[:, None] * (target[7:] - previous[7:])
        if size == 46:
            # DualArmPolicyWrapper layout: [r_pose7, l_pose7, r_hand16, l_hand16];
            # the linear pass above already handled l_xyz [7:10] and l_hand
            # [14:30] and [30:46]. Only the second quaternion [10:14] needs the
            # geodesic treatment.
            steps[:, 10:14] = _geodesic_quat(previous[10:14], target[10:14], alphas)
        dense.extend(step.astype(np.float32) for step in steps)
        previous_offset, previous = offset, target
    return dense
