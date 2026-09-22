# Training

This directory contains the Clean50 data loader, configuration, and distributed training entry point for ME-Dex-1.0.

The recipe uses the unified tactile encoder with 16 RoboTwin surface slots, 48-dimensional latent vectors, and 18 temporal slices: 2 observed slices followed by 16 future slices. Set `topology` to `full_joint` or `h_bridge` in the YAML configuration.

Start a run with:

```bash
torchrun --nproc_per_node=16 -m training.train \
  --config training/configs/clean50_uni.yaml
```

## Prepare the T5 cache

The generated T5 payload is intentionally not bundled with the dataset because the
all-instructions cache is very large. Build a Clean50-only cache once from the Wan2.2
T5 encoder before training:

```bash
python -m training.build_t5_cache \
  --clean-root datasets/ME-Dex-1.0-RoboTwin-Clean50-Tactile \
  --t5-checkpoint checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
  --tokenizer checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl \
  --output datasets/ME-Dex-1.0-RoboTwin-Clean50-Tactile/t5_prompt_table.pt
```

The command writes `t5_prompt_table.pt` and its adjacent `t5_prompt_table.bin`.
The training config reads the metadata file and memory-maps the payload during data loading.
