"""Official DexJoCo state/action dimension contracts.

Raw state and environment action use quaternion poses. Policy actions use
rotation vectors. Bimanual raw/environment tensors are pose-major across arms,
then hand-major; bimanual policy tensors are complete right-arm blocks followed
by complete left-arm blocks.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


SINGLE_RAW_STATE_DIM = 23
DUAL_RAW_STATE_DIM = 46
SINGLE_POLICY_ACTION_DIM = 22
DUAL_POLICY_ACTION_DIM = 44


def _pose7_to_policy_pose6(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float32).reshape(-1)
    if value.shape != (7,):
        raise ValueError(f"Expected pose7 [xyz, quat_wxyz], got {value.shape}")
    rotvec = Rotation.from_quat(value[3:7], scalar_first=True).as_rotvec()
    return np.concatenate((value[:3], rotvec)).astype(np.float32)


def raw_state_for_policy(raw_state: np.ndarray, *, dual_arm: bool) -> np.ndarray:
    """Convert official raw quaternion state to the unified 44-D policy state.

    Raw layouts:
      single: [pose7, hand16]
      dual:   [r_pose7, l_pose7, r_hand16, l_hand16]

    Policy layout is always [right22, left22]. Single-arm observations occupy
    the right block and use an all-zero left block, matching the training data.
    """
    value = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    expected = DUAL_RAW_STATE_DIM if dual_arm else SINGLE_RAW_STATE_DIM
    if value.size < expected:
        raise ValueError(f"Expected at least {expected} state values, got {value.shape}")
    value = value[:expected]

    if dual_arm:
        right = np.concatenate((_pose7_to_policy_pose6(value[:7]), value[14:30]))
        left = np.concatenate((_pose7_to_policy_pose6(value[7:14]), value[30:46]))
        return np.concatenate((right, left)).astype(np.float32)

    right = np.concatenate((_pose7_to_policy_pose6(value[:7]), value[7:23]))
    return np.concatenate((right, np.zeros(SINGLE_POLICY_ACTION_DIM, dtype=np.float32)))


def policy_action_to_env_action(
    action: np.ndarray,
    *,
    dual_arm: bool,
    padded_single_arm_slot: str = "right",
) -> np.ndarray:
    """Convert rotvec policy action to the official quaternion env action.

    Official policy layouts:
      single: [xyz3, rotvec3, hand16]
      dual:   [r_xyz3, r_rotvec3, r_hand16,
               l_xyz3, l_rotvec3, l_hand16]

    Official environment layouts:
      single: [xyz3, quat_wxyz4, hand16]
      dual:   [r_pose7, l_pose7, r_hand16, l_hand16]

    A unified 44-D model may emit a zero-padded single-arm action. Its physical
    slots follow the same right-first contract: right=0:22, left=22:44.
    """
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if dual_arm:
        if value.shape != (DUAL_POLICY_ACTION_DIM,):
            raise ValueError(f"Expected dual-arm policy action (44,), got {value.shape}")
        right = value[:SINGLE_POLICY_ACTION_DIM]
        left = value[SINGLE_POLICY_ACTION_DIM:]
        right_pose = np.concatenate(
            (right[:3], Rotation.from_rotvec(right[3:6]).as_quat(scalar_first=True))
        )
        left_pose = np.concatenate(
            (left[:3], Rotation.from_rotvec(left[3:6]).as_quat(scalar_first=True))
        )
        return np.concatenate((right_pose, left_pose, right[6:], left[6:])).astype(np.float32)

    if value.shape == (DUAL_POLICY_ACTION_DIM,):
        if padded_single_arm_slot not in ("right", "left"):
            raise ValueError(
                f"padded_single_arm_slot must be right or left, got {padded_single_arm_slot!r}"
            )
        value = (
            value[:SINGLE_POLICY_ACTION_DIM]
            if padded_single_arm_slot == "right"
            else value[SINGLE_POLICY_ACTION_DIM:]
        )
    if value.shape != (SINGLE_POLICY_ACTION_DIM,):
        raise ValueError(f"Expected single-arm policy action (22,) or padded (44,), got {value.shape}")
    pose = np.concatenate(
        (value[:3], Rotation.from_rotvec(value[3:6]).as_quat(scalar_first=True))
    )
    return np.concatenate((pose, value[6:])).astype(np.float32)
