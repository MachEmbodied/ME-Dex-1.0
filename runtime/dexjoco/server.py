#!/usr/bin/env python3
"""CUDA HTTP server for the aligned DJ-TFA-2.0 / V4-Uni WTAM checkpoint."""
from __future__ import annotations

import argparse
import base64
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from runtime.dexjoco.tasks import DEFAULT_TASK_PROMPTS
from runtime.dexjoco.text_cache import TaskT5Cache
from runtime.dexjoco.text_cache import DiskT5Cache
from runtime.dexjoco.checkpoint import load_single_device_model
from runtime.dexjoco.observations import make_horizontal_rgb, make_t_shape_rgb, make_wtam_state
from runtime.dexjoco.normalization import load_quantile_normalizer44


SINGLE_TASKS = {"click_mouse", "fold_glasses", "hammer_nail", "pick_bucket", "pinch_tongs", "water_plant"}
CONTRACT = "dexjoco_raw50_at16_stride3_v8_stride6_horizon960ms_v1"
TACTILE_PROTOCOL = "DJ-TFA-2.0-palmar"
TACTILE_SURFACE_ORDER = "left16_then_right16"
TACTILE_CODEC = "v4_uni_50k"
IMAGE_PREPROCESS = "original_native_views_then_t_mosaic_then_resize"


def decode_image(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid camera image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pad_language(embedding: torch.Tensor, length: int = 512) -> torch.Tensor:
    embedding = embedding[:length]
    if embedding.shape[0] < length:
        embedding = torch.cat((embedding, embedding.new_zeros(length - embedding.shape[0], embedding.shape[1])))
    return embedding.unsqueeze(0)


class RawAlignedPolicy:
    def __init__(self, config_path: Path, checkpoint: Path, device: str) -> None:
        self.model, self.config = load_single_device_model(
            config_path, checkpoint, device=device
        )
        self.video_layout = str(self.config.dataset.params.get("video_layout", "t_vertical"))
        if self.video_layout not in {"t_vertical", "horizontal"}:
            raise ValueError(f"unsupported video_layout={self.video_layout!r}")
        self.device = device
        self.normalizer, _ = load_quantile_normalizer44(
            self.config.dataset.params.normalization_stats_path
        )
        self.tactile_enabled = bool(self.config.model.tactile.enabled)
        self.tactile_codec = str(self.config.model.tactile.codec_type)
        if not self.tactile_enabled or self.tactile_codec != TACTILE_CODEC:
            raise ValueError(
                "DJ-TFA-2.0 evaluation requires tactile.enabled=true and "
                f"codec_type={TACTILE_CODEC!r}, got enabled={self.tactile_enabled} "
                f"codec={self.tactile_codec!r}"
            )
        self.action_count = int(self.config.common.action_chunk_size)
        if self.action_count != 16:
            raise ValueError(f"raw aligned checkpoint must predict 16 actions, got {self.action_count}")
        cache_path = Path(self.config.dataset.params.t5_cache_path)
        if cache_path.is_dir():
            cache = TaskT5Cache(cache_path)
            self.language = {
                task: pad_language(cache.embedding(task, prompt))
                for task, prompt in DEFAULT_TASK_PROMPTS.items()
            }
        else:
            cache = DiskT5Cache(cache_path)
            self.language = {
                task: pad_language(cache.embedding(prompt))
                for task, prompt in DEFAULT_TASK_PROMPTS.items()
            }
        try:
            self.checkpoint_step = int(checkpoint.name.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            self.checkpoint_step = -1
        self.lock = threading.Lock()
        self.visualization = False

    def metadata(self) -> dict:
        return {
            "step": self.checkpoint_step,
            "architecture": "wam_va",
            "tactile_enabled": self.tactile_enabled,
            "tactile_protocol": TACTILE_PROTOCOL,
            "tactile_surface_order": TACTILE_SURFACE_ORDER,
            "tactile_surface_count": 32,
            "tactile_codec": self.tactile_codec,
            "temporal_contract": CONTRACT,
            "action_count": self.action_count,
            "image_preprocess": (
                "original_native_views_then_horizontal_mosaic_then_resize"
                if self.video_layout == "horizontal" else IMAGE_PREPROCESS
            ),
            **({"visualization_enabled": True} if self.visualization else {}),
        }

    def _image(self, cameras: dict[str, str], task: str) -> np.ndarray:
        bimanual = task not in SINGLE_TASKS
        front_key = "random_camera" if "random_camera" in cameras else ("ego" if bimanual and "ego" in cameras else ("ego_right" if task == "click_mouse" else "front"))
        front = decode_image(cameras[front_key])
        if bimanual:
            left = decode_image(cameras["wrist_left"])
            right = decode_image(cameras["wrist_right"])
        else:
            left = None
            right = decode_image(cameras["wrist"])
        pack_rgb = make_horizontal_rgb if self.video_layout == "horizontal" else make_t_shape_rgb
        return pack_rgb(
            front=front, left_wrist=left, right_wrist=right, bimanual=bimanual,
            output_size=(int(self.config.common.video_height), int(self.config.common.video_width)),
        )

    def infer(self, request: dict) -> dict:
        task = str(request.get("task", ""))
        if task not in DEFAULT_TASK_PROMPTS:
            raise ValueError(f"unsupported task {task!r}")
        bimanual = task not in SINGLE_TASKS
        image = torch.from_numpy(self._image(request["cameras"], task).copy()).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        state = torch.from_numpy(make_wtam_state(np.asarray(request["state"], dtype=np.float32), bimanual=bimanual))
        state = self.normalizer.normalize_state(state)
        # Train-time single-arm samples zero the padded, invalid left-arm slot
        # after normalization.  Preserve that exact contract at inference.
        if not bimanual:
            state[..., 22:] = 0
        state = state.unsqueeze(0).to(self.device)
        kwargs = {
            "first_frame": image,
            "state": state,
            "action_valid_mask": torch.cat(
                (
                    torch.ones(22, dtype=torch.bool),
                    torch.full((22,), bimanual, dtype=torch.bool),
                )
            ).unsqueeze(0).to(self.device),
            "num_inference_steps": int(self.config.model.inference.num_inference_timesteps),
            "language_embeddings": self.language[task],
            "vlm_inputs": None,
            "decode_video": False,
        }
        if self.tactile_enabled:
            tactile = torch.as_tensor(request.get("tactile"), dtype=torch.float32)
            grid = torch.as_tensor(request.get("tactile_grid_mask"), dtype=torch.bool)
            observed_times = torch.as_tensor(request.get("tactile_observed_frame_times"), dtype=torch.float32)
            future_times = torch.as_tensor(request.get("tactile_future_query_times"), dtype=torch.float32)
            if tuple(tactile.shape) != (32, 2, 3, 10, 14) or tuple(grid.shape) != (32, 10, 14):
                raise ValueError(f"raw aligned tactile shape mismatch: {tuple(tactile.shape)} {tuple(grid.shape)}")
            if tuple(observed_times.shape) != (2,) or tuple(future_times.shape) != (16,):
                raise ValueError("raw aligned tactile times must be [2] and [16]")
            kwargs.update(
                tactile_observed_source=tactile.unsqueeze(0).to(self.device),
                tactile_grid_mask_source=grid.unsqueeze(0).to(self.device),
                tactile_observed_frame_times=observed_times.unsqueeze(0).to(self.device),
                tactile_future_query_times=future_times.unsqueeze(0).to(self.device),
            )
        diagnostic = {}
        with self.lock, torch.inference_mode():
            _, action = self.model.inference_step(**kwargs)
        action = self.normalizer.denormalize_action(action[0].float().cpu()).numpy()
        if tuple(action.shape) != (16, 44) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid raw aligned action output {tuple(action.shape)}")
        return {"actions": action.tolist(), **self.metadata(), **diagnostic}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_json({"status": "ok", **self.server.policy.metadata()})

    def do_POST(self) -> None:
        if self.path != "/infer":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            result = self.server.policy.infer(json.loads(self.rfile.read(size)))
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def send_json(self, value: dict, status: int = 200) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.policy = RawAlignedPolicy(args.config.resolve(), args.checkpoint.resolve(), args.device)
    print(json.dumps({"status": "ready", **server.policy.metadata()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
