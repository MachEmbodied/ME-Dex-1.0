"""DexJoCo client for the aligned raw 50 Hz WTAM CUDA HTTP policy server."""
from __future__ import annotations
import argparse, base64, json, random, urllib.error, urllib.request
from collections import deque
from pathlib import Path
import cv2, imageio.v2 as imageio, mujoco, numpy as np, yaml
from scipy.spatial.transform import Rotation
from dexjoco.tasks import CONFIG_MAPPING
from runtime.dexjoco.action_contract import policy_action_to_env_action
from runtime.dexjoco.physics import CaptureMjStep
from runtime.dexjoco.control import _hold_action_from_observation, _densify_sparse_waypoints
from runtime.dexjoco.tactile_collector import GEOMETRY, make_collectors
SINGLE_TASKS = {'click_mouse', 'fold_glasses', 'hammer_nail', 'pick_bucket', 'pinch_tongs', 'water_plant'}
OFFSETS = tuple(range(3, 49, 3))
TACTILE_PROTOCOL = 'DJ-TFA-2.0-palmar'
CLICK_MOUSE_WARMUP = np.asarray([-0.44294, 1.3729e-06, 1.517, -3.14156462, -6.91584035e-05, -0.00140317984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0], dtype=np.float32)

class OnlineDJTFA2Collector:
    """Expose the frozen 32-surface DJ-TFA-2.0 collector to the raw16 client."""

    def __init__(self, model, data, *, bimanual, mujoco_module):
        self.model = model
        self.data = data
        self.mujoco = mujoco_module
        self.collectors = make_collectors(model, bimanual=bimanual, surface_tolerance_m=0.003, fn_epsilon_n=1e-09)
        one_hand = np.asarray(GEOMETRY['taxel_valid'], dtype=bool)
        left = one_hand if bimanual else np.zeros_like(one_hand)
        self.taxel_valid = np.concatenate((left, one_hand), axis=0)
        self.observation_index = 0

    def capture_substep(self, data):
        for collector in self.collectors.values():
            collector.capture_substep(data)

    def flush_interval(self):
        values = {}
        for (hand, collector) in self.collectors.items():
            if collector._physics_steps < 1:
                raise RuntimeError(f'empty DJ-TFA-2.0 interval for {hand}')
            collector.flush_interval(self.data, observation_index=self.observation_index, sim_time_end_s=float(self.data.time))
            values[hand] = np.asarray(collector.rows.pop()['force_canonical'], dtype=np.float32)
        self.observation_index += 1
        right = values['right']
        left = values.get('left', np.zeros_like(right))
        force = np.concatenate((left, right), axis=0)
        if force.shape != (32, 3, 10, 14):
            raise RuntimeError(f'DJ-TFA-2.0 online force shape mismatch: {force.shape}')
        return force

def encode_png(image):
    (ok, encoded) = cv2.imencode('.png', cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('failed to encode camera image')
    return base64.b64encode(encoded.tobytes()).decode('ascii')

class Client:

    def __init__(self, url, timeout=600.0):
        (self.url, self.timeout) = (url.rstrip('/'), timeout)

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload, separators=(',', ':')).encode()
        req = urllib.request.Request(self.url + path, data=body, headers={'Content-Type': 'application/json'} if body else {}, method='POST' if body else 'GET')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                value = json.loads(exc.read())
            except Exception:
                raise
            if isinstance(value, dict) and 'error' in value:
                raise RuntimeError(f"server error {exc.code}: {value['error']}") from exc
            raise
        if not isinstance(value, dict):
            raise TypeError('server response must be an object')
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value

    def health(self):
        return self.request('/health')

    def infer(self, task, state, cameras, *, tactile=None, grid=None, observed_times=None, future_times=None):
        payload = {'task': task, 'state': np.asarray(state, dtype=np.float32).reshape(-1).tolist(), 'cameras': {k: encode_png(v) for (k, v) in cameras.items()}}
        if tactile is not None:
            payload.update({'tactile': np.asarray(tactile, dtype=np.float32).tolist(), 'tactile_grid_mask': np.asarray(grid, dtype=bool).tolist(), 'tactile_observed_frame_times': np.asarray(observed_times, dtype=np.float32).tolist(), 'tactile_future_query_times': np.asarray(future_times, dtype=np.float32).tolist()})
        result = self.request('/infer', payload)
        action = np.asarray(result.get('actions'), dtype=np.float32)
        self.last_result = result
        if action.shape != (16, 44) or not np.isfinite(action).all():
            raise RuntimeError(f'invalid raw aligned action shape {action.shape}')
        return action

def _camera_keys_for_observation(observation, dual_arm, task):
    available = set(observation)
    if dual_arm:
        keys = ('random_camera' if 'random_camera' in available else 'ego', 'wrist_left', 'wrist_right')
    else:
        fallback = 'ego_right' if task == 'click_mouse' else 'front'
        keys = ('random_camera' if 'random_camera' in available else fallback, 'wrist')
    missing = [key for key in keys if key not in available]
    if missing:
        raise KeyError(f'observation missing camera keys {missing}; available={sorted(available)}')
    return keys

def _ema_action(previous, target, alpha):
    """Causal low-pass command smoothing; one env.step per policy action."""
    previous = np.asarray(previous, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if previous.shape != target.shape:
        raise ValueError(f'smoothing shape mismatch: {previous.shape} vs {target.shape}')
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f'smooth alpha must be in (0,1], got {alpha}')
    value = previous + alpha * (target - previous)
    quaternion_starts = (3, 10) if value.size == 46 else (3,)
    for start in quaternion_starts:
        if start + 4 <= value.size:
            q_prev = previous[start:start + 4]
            q_target = target[start:start + 4]
            if float(np.dot(q_target, q_prev)) < 0.0:
                q_target = -q_target
                value[start:start + 4] = q_prev + alpha * (q_target - q_prev)
            norm = float(np.linalg.norm(value[start:start + 4]))
            if norm > 1e-08:
                value[start:start + 4] /= norm
    return value.astype(np.float32, copy=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--server-url', default='http://127.0.0.1:18080')
    p.add_argument('--task', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episode-seed-mode', choices=('per-episode-rebuild', 'legacy-stream'), default='per-episode-rebuild')
    p.add_argument('--max-replans', type=int, default=None)
    p.add_argument('--max-env-steps', type=int, default=1200)
    p.add_argument('--replan-env-steps', type=int, default=None, help='Execute at most this many native environment steps from each densified chunk before requesting a new chunk. By default the complete densified chunk is executed.')
    p.add_argument('--chunk-steps', '--actions-per-chunk', dest='chunk_steps', type=int, default=16, help='Sparse policy waypoints to consume per inference call before replanning.')
    p.add_argument('--smooth-alpha', type=float, default=None, help='Causal EMA target weight; 1 disables smoothing. Defaults to 0.35 in legacy mode and 1.0 with --sparse-densify.')
    p.set_defaults(rand_full=True, randomize_dynamics=True, record_video=True, sparse_densify=False)
    p.add_argument('--sparse-densify', dest='sparse_densify', action='store_true', help='Treat policy outputs as sparse waypoints at offsets 3,6,...,48 on the 50 Hz control clock and geodesically densify them. Default false preserves the historical one-env-step-per-waypoint behavior.')
    p.add_argument('--no-sparse-densify', dest='sparse_densify', action='store_false', help=argparse.SUPPRESS)
    p.add_argument('--rand-full', dest='rand_full', action='store_true')
    p.add_argument('--no-rand-full', dest='rand_full', action='store_false')
    p.add_argument('--randomize', dest='rand_full', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--randomize-dynamics', dest='randomize_dynamics', action='store_true')
    p.add_argument('--no-randomize-dynamics', dest='randomize_dynamics', action='store_false')
    p.add_argument('--record-video', dest='record_video', action='store_true')
    p.add_argument('--no-record-video', dest='record_video', action='store_false')
    args = p.parse_args()
    if args.smooth_alpha is None:
        args.smooth_alpha = 1.0 if args.sparse_densify else 0.35
    if not 1 <= args.chunk_steps <= 16:
        p.error('--chunk-steps must be in [1,16]')
    if not 0.0 < args.smooth_alpha <= 1.0:
        p.error('--smooth-alpha must be in (0,1]')
    if args.replan_env_steps is not None and args.replan_env_steps <= 0:
        p.error('--replan-env-steps must be positive')
    if args.replan_env_steps is not None and (not args.sparse_densify):
        p.error('--replan-env-steps requires --sparse-densify')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    task_cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    task = str(task_cfg['env_name'])
    if task != args.task:
        raise ValueError(f'task mismatch: config={task} arg={args.task}')
    client = Client(args.server_url)
    health = client.health()
    if health.get('temporal_contract') != 'dexjoco_raw50_at16_stride3_v8_stride6_horizon960ms_v1' or int(health.get('action_count', 0)) != 16:
        raise RuntimeError(f'server is not raw-aligned: {health}')
    if health.get('tactile_protocol') != TACTILE_PROTOCOL or int(health.get('tactile_surface_count', 0)) != 32 or health.get('tactile_surface_order') != 'left16_then_right16':
        raise RuntimeError(f'server is not DJ-TFA-2.0 aligned: {health}')
    tactile_enabled = bool(health.get('tactile_enabled', False))
    dual_arm = task not in SINGLE_TASKS
    camera_keys = tuple(dict.fromkeys((str(v) for v in task_cfg['camera_mapping'].values())))
    config = CONFIG_MAPPING[task]()
    episodes = []
    successes = 0
    stream_mode = args.episode_seed_mode == 'legacy-stream'
    env = None
    photograph_kwargs = {'camera_screen_effect': False} if task == 'bimanual_photograph' else {}
    if stream_mode:
        random.seed(args.seed)
        np.random.seed(args.seed)
        env = config.get_environment(policy_mode=True, render_mode='rgb_array', randomize=args.rand_full, seed=args.seed, randomize_dynamics=args.randomize_dynamics, **photograph_kwargs)
    for episode in range(args.episodes):
        seed = None if stream_mode else args.seed + episode
        if not stream_mode:
            random.seed(seed)
            np.random.seed(seed)
            env = config.get_environment(policy_mode=True, render_mode='rgb_array', randomize=args.rand_full, seed=seed, randomize_dynamics=args.randomize_dynamics, **photograph_kwargs)
        raw = env.unwrapped
        physics_model = raw._model if hasattr(raw, '_model') else raw.model
        physics_data = raw._data if hasattr(raw, '_data') else raw.data
        (observation, _) = env.reset() if stream_mode else env.reset(seed=seed)
        collector = OnlineDJTFA2Collector(physics_model, physics_data, bimanual=dual_arm, mujoco_module=mujoco) if tactile_enabled else None
        history = deque(maxlen=2)
        grid = None if collector is None else collector.taxel_valid.copy()
        camera_keys = _camera_keys_for_observation(observation, dual_arm, task)
        now = float(raw.data.time)
        history.extend(((now, np.zeros((32, 3, 10, 14), dtype=np.float32)), (now, np.zeros((32, 3, 10, 14), dtype=np.float32))))
        if grid is None:
            grid = np.zeros((32, 10, 14), dtype=bool)
        elif grid.shape != (32, 10, 14):
            raise RuntimeError(f'DJ-TFA-2.0 grid shape mismatch: {grid.shape}')
        record_pressed_digits = task == 'bimanual_unlock_ipad'
        unlock_sequence = [int(d) for d in getattr(raw, '_unlock_sequence', [])] if record_pressed_digits else None
        pressed_input = []
        env_steps = replans = 0
        success = False
        terminated = truncated = False
        temp_dir = args.output / f'episode_{episode:03d}_temp'
        writers = {}
        smooth_command = _hold_action_from_observation(observation, dual_arm=dual_arm)
        if args.record_video:
            temp_dir.mkdir()
            writers = {key: imageio.get_writer(temp_dir / f'{key}.mp4', fps=30) for key in camera_keys}
            for (key, writer) in writers.items():
                writer.append_data(np.asarray(observation[key], dtype=np.uint8))

        def step(action):
            nonlocal observation, env_steps, success, terminated, truncated
            if collector is None:
                (observation, _, terminated, truncated, info) = env.step(action)
            else:
                with CaptureMjStep(collector, physics_model, physics_data):
                    (observation, _, terminated, truncated, info) = env.step(action)
                force = collector.flush_interval()
                history.append((float(raw.data.time), force))
            env_steps += 1
            success = success or bool(info.get('succeed', False))
            for (key, writer) in writers.items():
                writer.append_data(np.asarray(observation[key], dtype=np.uint8))
            if record_pressed_digits and info.get('pressed_digits'):
                pressed_input.append([int(d) for d in info['pressed_digits']])
        if task == 'click_mouse':
            warmup = policy_action_to_env_action(CLICK_MOUSE_WARMUP, dual_arm=False)
            for _ in range(30):
                step(warmup)
        while not (terminated or truncated or success) and env_steps < args.max_env_steps and (args.max_replans is None or replans < args.max_replans):
            cameras = {key: np.asarray(observation[key], dtype=np.uint8) for key in camera_keys}
            state = np.asarray(observation['state'], dtype=np.float32)
            tactile = observed = future = None
            if tactile_enabled:
                observed = np.asarray([v[0] for v in history], dtype=np.float32)
                tactile = np.stack([v[1] for v in history], axis=1).astype(np.float32)
                future = observed[-1] + np.asarray(OFFSETS, dtype=np.float32) * 0.02
            actions = client.infer(task, state, cameras, tactile=tactile, grid=grid, observed_times=observed, future_times=future)
            if args.sparse_densify:
                waypoint_env_actions = [policy_action_to_env_action(action, dual_arm=dual_arm) for action in actions[:args.chunk_steps]]
                anchor = _hold_action_from_observation(observation, dual_arm=dual_arm)
                commands = _densify_sparse_waypoints(anchor, waypoint_env_actions, OFFSETS[:args.chunk_steps])
            else:
                commands = [policy_action_to_env_action(action, dual_arm=dual_arm) for action in actions[:args.chunk_steps]]
            commands_to_execute = commands
            if args.replan_env_steps is not None:
                commands_to_execute = commands[:args.replan_env_steps]
            for target in commands_to_execute:
                smooth_command = _ema_action(smooth_command, target, args.smooth_alpha)
                step(smooth_command)
                if terminated or truncated or success or (env_steps >= args.max_env_steps):
                    break
            replans += 1
        for writer in writers.values():
            writer.close()
        video_dir = None
        if args.record_video:
            video_path = args.output / f"episode_{episode:03d}_{('success' if success else 'failure')}"
            temp_dir.rename(video_path)
            video_dir = str(video_path)
        successes += int(success)
        record = {'episode': episode, 'episode_seed': seed, 'success': bool(success), 'env_steps': env_steps, 'replans': replans, 'tactile_enabled': tactile_enabled, 'video_dir': video_dir}
        if record_pressed_digits:
            record['pressed_input'] = pressed_input
            record['unlock_sequence'] = unlock_sequence
        episodes.append(record)
        if not stream_mode:
            env.close()
    if stream_mode and env is not None:
        env.close()
    dense_env_steps = args.chunk_steps * 3 if args.sparse_densify else args.chunk_steps
    executed_env_steps = min(dense_env_steps, args.replan_env_steps) if args.replan_env_steps is not None else dense_env_steps
    summary = {'task': task, 'server': health, 'seed': args.seed, 'episodes': args.episodes, 'successes': successes, 'success_rate': successes / max(args.episodes, 1), 'episode_seed_mode': args.episode_seed_mode, 'rand_full': args.rand_full, 'randomize_dynamics': args.randomize_dynamics, 'max_env_steps': args.max_env_steps, 'max_replans': args.max_replans, 'chunk_steps': args.chunk_steps, 'actions_per_chunk': args.chunk_steps, 'sparse_densify': args.sparse_densify, 'replan_env_steps': args.replan_env_steps, 'dense_env_steps_per_chunk': dense_env_steps, 'dense_env_steps_per_replan': executed_env_steps, 'smooth_alpha': args.smooth_alpha, 'record_video': args.record_video, 'video_fps': 30 if args.record_video else None, 'episodes_detail': episodes}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for (k, v) in summary.items() if k != 'episodes_detail'}, indent=2), flush=True)
if __name__ == '__main__':
    main()
