"""Auditable conversions at the WTAM/DexJoCo policy boundary."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _wxyz_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion)
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_rotvec()


def _rotvec_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return xyzw[[3, 0, 1, 2]]


def official_state_to_rotvec_proprio(state: np.ndarray, *, bimanual: bool) -> np.ndarray:
    """Convert official proprio prefix (quat) to the model's rotvec layout."""
    state = np.asarray(state, dtype=np.float32)
    if not bimanual:
        if state.shape[0] < 23:
            raise ValueError(f"Single-arm observation must contain 23 proprio values: {state.shape}")
        rotvec = _wxyz_to_rotvec(state[3:7])
        return np.concatenate((state[:3], rotvec, state[7:23])).astype(np.float32)
    if state.shape[0] < 46:
        raise ValueError(f"Bimanual observation must contain 46 proprio values: {state.shape}")
    right_rotvec = _wxyz_to_rotvec(state[3:7])
    left_rotvec = _wxyz_to_rotvec(state[10:14])
    # Official state prefix is [right_pose7, left_pose7, right_hand16, left_hand16].
    return np.concatenate(
        (state[:3], right_rotvec, state[14:30], state[7:10], left_rotvec, state[30:46])
    ).astype(np.float32)


def rotvec_action_to_environment(action: np.ndarray, *, bimanual: bool) -> np.ndarray:
    """Convert model 22/44-D output to official flat 23/46-D quaternion action."""
    action = np.asarray(action, dtype=np.float32)
    if not bimanual:
        if action.shape != (22,):
            raise ValueError(f"Expected single-arm action (22,), got {action.shape}")
        quat = _rotvec_to_wxyz(action[3:6])
        return np.concatenate((action[:3], quat, action[6:22])).astype(np.float32)
    if action.shape != (44,):
        raise ValueError(f"Expected bimanual action (44,), got {action.shape}")
    right_quat = _rotvec_to_wxyz(action[3:6])
    left_quat = _rotvec_to_wxyz(action[25:28])
    # Environment wrapper expects [right_pose7, left_pose7, right_hand16, left_hand16].
    return np.concatenate(
        (
            action[:3], right_quat,
            action[22:25], left_quat,
            action[6:22], action[28:44],
        )
    ).astype(np.float32)
