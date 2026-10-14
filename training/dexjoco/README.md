# DexJoCo training

This is the ME-Dex-1.0 **384 × 320, Stage-1 WAN initialization, V4-Uni Hand2** recipe. The entry point, dataset and trainer are extracted from the supplied experiment and use the same model as `runtime/dexjoco/`. RoboTwin entry points remain under `training/`.

## Assets and installation

Install the CUDA or PPU-specific PyTorch build for your accelerator, then:

```bash
pip install -r training/dexjoco/requirements.txt
```

Run commands from the repository root. Configure local assets through environment variables (or edit `training/configs/dexjoco_uni.yaml`):

```bash
export DEXJOCO_DATA_ROOT=/path/to/dexjoco_11task_dj_tfa2_palmar_vat_aligned_20260909
export DEXJOCO_T5_CACHE=/path/to/dexjoco_11task_T5_embedding
export ME_DEX_WAN_PATH=/path/to/Wan2.2-TI2V-5B
export ME_DEX_STAGE1_CHECKPOINT=/path/to/Motus_Wan2_2_5B_pretrain/model.safetensors
export ME_DEX_TACTILE_CHECKPOINT=/path/to/tactile_ae_v4_uni/step_50000.pt
```

Required data: `all_manifest_dj_tfa2.json`, the referenced original camera videos and raw Zarr actions, and the aligned DJ-TFA-2.0 HDF5 force files. The manifest contains paths to source episodes: those paths must exist on the training host. All 1,100 episodes are included, including failed replay/task-gate cases. The language cache has one `.safetensors` file and one matching `.json` instruction record per task (the original `embedding_000` format); the disk-backed prompt-table format is also supported.

The right-first 44-D quantile statistics are bundled at `assets/dexjoco/normalization.json`. Override `DEXJOCO_NORMALIZATION` only with statistics for the same checkpoint contract. The frozen V4-Uni encoder implementation is bundled in the runtime. Its original thirteen source SHA256 checks and its step-50,000 EMA checkpoint hash remain enforced. Policy checkpoints, pretrained weights, datasets and language embeddings are external assets, not Git contents.

## Reference training

On both nodes, set rank and rendezvous, then launch:

```bash
export NNODES=2 NPROC_PER_NODE=16
export NODE_RANK=0                 # 1 on the second node
export MASTER_ADDR=192.0.2.10      # address of rank 0
export MASTER_PORT=29719
bash training/dexjoco/train.sh
```

The default is 32 processes × microbatch 2 × accumulation 1 = global batch 64; 60,000 optimizer steps; ZeRO-1 with bf16; AdamW `(0.9, 0.95)`; linear schedule; 1,000 warmup steps. Learning rates are `5e-5` for action, `1e-5` for WAN and `1e-5` for tactile. Only the 825-tensor Stage-1 WAN checkpoint initializes pretrained policy weights. Action/tactile experts start fresh; the tactile encoder stays frozen. Video/action/tactile loss weights are all one.

Set `PYTHON=/path/to/python` to select an environment. PPU runs must use the vendor environment and SDK library paths already configured on the host. Set `WAN_DISABLE_FLASH_ATTN=1` to select PyTorch SDPA on machines without a compatible FlashAttention build. `CONFIG=/path/to/config.yaml` selects another config; extra launcher arguments go to the trainer. Full-state resume uses `--resume_from /path/to/checkpoint_step_N`.

## Smoke training

This executes real video/action/tactile forward, backward and AdamW updates and saves a reloadable checkpoint. It retains the full network and 384 × 320 inputs. A two-PPU run needs approximately 90 GiB per device; two 48-GiB GPUs are not sufficient for this ZeRO-1 recipe.

```bash
python -m training.dexjoco.prepare_smoke \
  --output outputs/smoke_train/config.yaml --world-size 2 --steps 2
NNODES=1 NPROC_PER_NODE=2 NODE_RANK=0 MASTER_ADDR=127.0.0.1 \
CONFIG=outputs/smoke_train/config.yaml \
  bash training/dexjoco/train.sh --report_to none
```

Only test duration, batch topology, workers, output/logging frequency and per-task reporting change. The tiny batch cannot cover all eleven tasks in a reporting window, so smoke mode disables that reporting assertion. The regular recipe still reports all tasks every 500 steps. A smoke checkpoint tests the software path; it is not a trained evaluation model.

See [runtime instructions](../../runtime/dexjoco/README.md) and [migration notes](../../docs/dexjoco_migration.md).
