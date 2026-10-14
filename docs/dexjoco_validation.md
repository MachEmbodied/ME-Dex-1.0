# DexJoCo validation — 2026-10-14

The reorganized code was executed from the requested standalone release directory. No source checkout was added to its Python path. Test checkpoints/logs remain under `outputs/` and are excluded from Git.

| Check | Result |
| --- | --- |
| Python syntax, shell syntax, staged whitespace | Pass |
| Frozen encoder source SHA256 | All 13 original pins match |
| Original versus packaged dataset | All tensor shapes/byte hashes match for indices 0, 10,000 and 100,000 with seed 42 |
| Regression tests | 6/6 pass |
| Real-data training | 2 PPU devices, full model/resolution, 2 optimizer updates, native model + optimizer checkpoint saved |
| Strict reload on CUDA | All policy state keys load with `strict=True`; frozen V4-Uni source/checkpoint hashes verified |
| Single-arm closed loop | `fold_glasses`: 1 episode, 96 environment steps, 2 policy replans, exit 0 |
| Bimanual closed loop | `bimanual_assembly`: 1 episode, 96 environment steps, 2 policy replans, exit 0 |
| Final clean-package rerun | Both closed-loop smoke tasks pass after removing unused staging files and adding checkpoint metadata validation |

Training used the real aligned eleven-task dataset (184,470 sampled windows), Wan Stage-1 initialization and the pinned step-50,000 frozen tactile EMA. The smoke run used microbatch 1, world size 2, accumulation 1; no optimizer steps were skipped. At step 2, total loss was 3.3146 (video 0.2042, action 1.4666, tactile 1.6437). Action/tactile gradient norms were nonzero; the frozen tactile encoder had no gradients. Peak memory across ranks was 83.33 GiB allocated, 89.87 GiB reserved.

The first smoke attempt completed its first update but stopped at the original per-task reporting assertion because the two-sample batch did not contain every task. The smoke config generator now disables that reporting window; the formal 500-step reporting interval is unchanged. The rerun completed and saved `checkpoint_step_2`.

The six regression checks cover right-first action conversion, geodesic quaternion interpolation across the ±π boundary, invalid-hand masking in flow matching/loss, Hand2 slot packing/support, near-constant quantile normalization, and rejection of an otherwise shape-compatible rgb256 checkpoint via its saved metadata.

## Tested environments

- Training: two PPU-ZW810E devices with 97,920 MiB each; vendor PyTorch `2.10.0+v0.1.0.ppu2.1.0.oe`, DeepSpeed `0.19.0+v0.1.0.ppu2.1.0.ce`, Accelerate `1.11.0+ali`, Diffusers `0.35.2`, Transformers `4.53.3`, Zarr `2.18.7`.
- Policy inference: NVIDIA RTX5880-Ada-48Q; PyTorch `2.9.1`, Diffusers `0.40.0`, Transformers `5.0.0rc0`, NumPy `2.5.3`, SciPy `1.17.1`, OmegaConf `2.3.1`, Safetensors `0.8.0`.
- Evaluation: the supplied DexJoCo simulator environment with headless MuJoCo and the online DJ-TFA-2.0 collector. The server used ten denoising steps, 384 × 320 T-mosaics, and 48-step sparse densification.

## Scope and remaining asset limitation

The original launcher's **step-60,000 policy path was absent** in the available storage. The later rgb256 policy was not substituted. Inference tests therefore used the checkpoint produced by this smoke training, exercising train → save → strict reload → real simulation with the requested architecture and preprocessing. Both smoke episodes have zero task successes, as expected for a two-update policy; these are execution checks, **not a pretrained performance measurement**. Full 32-PPU training, 60,000-step convergence and the original model's 11 × 50-episode benchmark were not rerun.

Raw smoke summaries and dataset tensor fingerprints are in [validation/dexjoco/](validation/dexjoco/). To evaluate the trained model, set `CHECKPOINT` to the restored original step-60,000 checkpoint and follow the runtime README.
