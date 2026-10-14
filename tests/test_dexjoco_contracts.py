"""Regression checks for checkpoint-sensitive DexJoCo IO contracts."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from runtime.dexjoco.control import _densify_sparse_waypoints
from runtime.dexjoco.action_contract import policy_action_to_env_action, raw_state_for_policy
from runtime.dexjoco.normalization import normalize_quantile, denormalize_quantile
from runtime.dexjoco.models.motus import prepare_action_flow_batch, masked_mse
from runtime.dexjoco.checkpoint import validate_checkpoint_contract
from runtime.dexjoco.models.wtam_v21.tactile_expert import UniversalTactileExpertConfig, DEXJOCO_HAND2_SLOT_IDS

class Contracts(unittest.TestCase):
    def test_reject_shape_compatible_rgb256_checkpoint(self):
        config = OmegaConf.load('training/configs/dexjoco_uni.yaml')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps({'common': {'video_height': 256}}))
            with self.assertRaisesRegex(ValueError, 'video_height'):
                validate_checkpoint_contract(config, path / 'mp_rank_00_model_states.pt')
            (path / 'config.json').write_text(json.dumps({'common': {'video_height': 384}}))
            validate_checkpoint_contract(config, path / 'mp_rank_00_model_states.pt')

    def test_right_first_roundtrip(self):
        for dual in (False, True):
            value = np.arange(44, dtype=np.float32) / 100
            if not dual:
                value[22:] = 0
            restored = raw_state_for_policy(policy_action_to_env_action(value, dual_arm=dual), dual_arm=dual)
            np.testing.assert_allclose(restored, value, atol=1e-6)

    def test_geodesic_48_steps_and_waypoint_endpoints(self):
        for dual in (False, True):
            anchor = np.zeros(44, np.float32)
            anchor[5] = np.deg2rad(179)
            target = anchor.copy(); target[5] = np.deg2rad(-179); target[0] = 1
            start = policy_action_to_env_action(anchor, dual_arm=dual)
            finish = policy_action_to_env_action(target, dual_arm=dual)
            result = np.asarray(_densify_sparse_waypoints(start, [finish]*16, tuple(range(3,49,3))))
            self.assertEqual(result.shape, (48, 46 if dual else 23))
            endpoints = result[2::3].copy()
            # q and -q encode the same rotation; interpolation preserves sign continuity.
            for offset in ((3, 10) if dual else (3,)):
                dots = endpoints[:, offset:offset+4] @ finish[offset:offset+4]
                endpoints[dots < 0, offset:offset+4] *= -1
            np.testing.assert_allclose(endpoints, np.broadcast_to(finish, endpoints.shape), atol=1e-6)
            distance = (Rotation.from_quat(start[3:7], scalar_first=True).inv()*Rotation.from_quat(result[0,3:7], scalar_first=True)).magnitude()
            self.assertLess(distance, np.deg2rad(1))

    def test_missing_left_hand_is_absent_from_noise_and_loss(self):
        valid=torch.zeros(1,44,dtype=torch.bool);valid[:,:22]=True
        clean,noisy,state,_,target_mask=prepare_action_flow_batch(torch.ones(1,16,44),torch.ones(1,44),valid,torch.full((1,1,1),.5),dtype=torch.float32,masked_zero_padding=True,pi05_zero_padding=False)
        self.assertEqual(torch.count_nonzero(clean[...,22:]).item(),0)
        self.assertEqual(torch.count_nonzero(noisy[...,22:]).item(),0)
        self.assertEqual(torch.count_nonzero(state[...,22:]).item(),0)
        prediction=torch.ones(1,16,44);prediction[...,22:]=100
        self.assertEqual(masked_mse(prediction,torch.zeros_like(prediction),target_mask,per_sample_mean=True).item(),1)

    def test_tactile_hand2_slots_and_missing_slots(self):
        cfg=UniversalTactileExpertConfig(latent_dim=48,latent_slices=18,queries_per_slice=34,condition_slices=2,hidden_size=1024,token_packing='dexjoco_left_right')
        latent=torch.randn(1,18,34,48)
        packed=cfg.pack_latent(latent)
        self.assertEqual(tuple(packed.shape),(1,18,2,768))
        restored=cfg.unpack_latent(packed)
        torch.testing.assert_close(restored[:,:,list(DEXJOCO_HAND2_SLOT_IDS)],latent[:,:,list(DEXJOCO_HAND2_SLOT_IDS)])
        self.assertEqual(torch.count_nonzero(restored[:,:,[3,20]]).item(),0)
        valid=torch.zeros(1,18,34,dtype=torch.bool);valid[:,:,list(DEXJOCO_HAND2_SLOT_IDS[16:])]=True
        expected=torch.tensor([False,True]).repeat(18).unsqueeze(0)
        torch.testing.assert_close(cfg.pack_validity(valid),expected)

    def test_near_constant_quantile_roundtrip(self):
        lo=torch.zeros(44);hi=torch.ones(44);hi[0]=1e-8
        values=torch.randn(16,44)
        torch.testing.assert_close(denormalize_quantile(normalize_quantile(values,lo,hi),lo,hi),values)

if __name__=='__main__':unittest.main()
