# DexJoCo inference and evaluation

The runtime is packaged under `runtime.dexjoco` and loads the original ME-Dex-1.0 V4-Uni Hand2 checkpoint with `strict=True`. It does not require a checkout of the original experiment repository. Its model and frozen tactile encoder are shared with `training.dexjoco`.

## Start a policy server

Install `runtime/dexjoco/requirements.txt` in the policy environment. Use the asset environment variables documented in [training](../../training/dexjoco/README.md). Inference requires Wan config/VAE, V4-Uni encoder weights, normalization and the eleven-task T5 cache; it does not require the Stage-1 initialization checkpoint or training dataset.

```bash
export CHECKPOINT=/path/to/checkpoint_step_60000
export CUDA_VISIBLE_DEVICES=0
export WAN_DISABLE_FLASH_ATTN=1
bash runtime/dexjoco/serve.sh
```

Optional settings: `PYTHON`, `CONFIG`, `HOST` (default `127.0.0.1`), `PORT` (21000), `DEVICE` (`cuda:0`). `/health` reports the model step and IO contract; `/infer` accepts JSON camera PNGs, raw environment state, tactile observations/masks and timestamps. The loader accepts a native checkpoint directory containing `pytorch_model/mp_rank_00_model_states.pt`, a directory with `mp_rank_00_model_states.pt`, or that file directly. A checkpoint must match this exact architecture. The later `rgb256` / ME-Dex1.1 model is a different recipe.

## Evaluate in DexJoCo

Install the DexJoCo simulator and its assets in a separate environment, then install `runtime/dexjoco/requirements-client.txt`. The simulator must expose `dexjoco.tasks.CONFIG_MAPPING`, the official single/bimanual environment wrappers, native camera observations, and the Allegro geometry used by `DJ-TFA-2.0-palmar`. This repository contains the policy adapter, action conversion, task configs, interpolation, and online tactile collector; it does not redistribute simulator assets.

```bash
export DEXJOCO_ROOT=/path/to/dexjoco-vla
export DEXJOCO_PY=/path/to/simulator-env/bin/python
bash runtime/dexjoco/eval.sh fold_glasses http://127.0.0.1:21000 outputs/fold_glasses
```

Defaults reproduce the specified evaluation: 50 episodes per task, seed 0, `legacy-stream`, 1,200 environment steps, no full/dynamics randomization, original camera T-mosaic, smoothing 1.0, and 16 sparse waypoints densified to 48 steps. Set `EPISODES`, `MAX_ENV_STEPS`, `SEED`, and `RECORD_VIDEO` to override run length/recording. Existing outputs are refused. EGL is probed before falling back to OSMesa; CPU-only simulators can explicitly set `MUJOCO_GL=osmesa`.

For the eleven-task shared-server evaluation:

```bash
export CHECKPOINT=/path/to/checkpoint_step_60000
export OUTPUT_ROOT=outputs/dexjoco_step60000
export PYTHON=/path/to/policy-env/bin/python
GPUS=0,1,2,3 bash runtime/dexjoco/launch_eval.sh
```

One model per GPU serves concurrent clients with serialized model calls. `TASKS` selects a space-separated subset, `GPUS` a comma-separated GPU list, and `BASE_PORT` the first server port. The launcher owns and cleans up its child processes. Simulator success rate is written to each task's `summary.json`.

## Closed-loop smoke test

This exercises single- and bimanual observations, force acquisition, model inference, quaternion interpolation and environment stepping. With 96 steps it checks execution, not task performance. Use a new output directory each time.

```bash
CHECKPOINT=/path/to/checkpoint_step_60000 \
OUTPUT_ROOT=outputs/dexjoco_smoke GPUS=0 \
TASKS='fold_glasses bimanual_assembly' EPISODES=1 MAX_ENV_STEPS=96 RECORD_VIDEO=0 \
  bash runtime/dexjoco/launch_eval.sh
```

The short-training checkpoint can also test train → save → strict reload → closed-loop inference. Its low task success must not be reported as the pretrained model's benchmark score.
