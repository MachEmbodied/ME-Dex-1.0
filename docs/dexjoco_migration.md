# DexJoCo migration notes

Base: `MachEmbodied/ME-Dex-1.0` main commit `a2db63b979e748bcea692966b1c57ab2984155a9`.

The requested source training launcher is:
`run_dexjoco_wtam_raw_aligned_shift1_11task_dj_tfa2_v4_uni_hand2_stage1_wan_random_action_tactile_full1100_original_v_32ppu.sh`.

The source inference directory is:
`eval_raw_dj_tfa2_v4_uni_hand2_stage1_wan_random_action_tactile_full1100_original_v_step60000_sparse_densify_full48`.
Its effective shared-server launcher selects the Stage-1 WAN / random action+tactile step-60000 policy. The older README in that directory names another initialization variant; the launcher and requested training configuration define this release. Later rgb256 scripts and visualization experiments in the same directory are excluded.

## Organization

| Source component | Release path |
| --- | --- |
| `train/train.py`, `train/sample.py` | `training/dexjoco/` |
| raw aligned DJ-TFA-2 loader and inherited preprocessing | `training/dexjoco/data/` |
| training YAML, ZeRO-1 config | `training/configs/dexjoco_*` |
| `models`, required WAN and utilities | `runtime/dexjoco/models`, `wan`, `utils` |
| model loader and HTTP server | `runtime/dexjoco/checkpoint.py`, `server.py` |
| client, online collector, action conversion and interpolation | `runtime/dexjoco/` |
| V4-Uni frozen encoder | `runtime/dexjoco/tactile_encoder/` |
| original quantile values | `assets/dexjoco/normalization.json` |

Changes are packaging, explicit entry points and asset paths. The model's checkpoint parameter names are preserved. Modules have a `runtime.dexjoco` namespace to avoid colliding with RoboTwin's `models`/`wan`. Training uses the original trainer, parameter groups, sampler, losses and scheduler. Shared text cache helpers are extracted from the original loaders. The runtime builds the same `positive_normal_si` normalizer directly from the frozen checkpoint configuration without importing unrelated tactile-training datasets. The thirteen audited source files retain their SHA256 identity; package initializers contain no unrelated imports.

Full checkpoint inference skips loading Stage-1 weights before its strict policy-state load. This removes an unnecessary inference dependency. Local cluster/environment locations become environment variables. Normalization numeric values are unchanged; only the machine-specific provenance path is made relative. Source SHA256 provenance is recorded in `dexjoco_source_manifest.json`.

## Fixed model/data contracts

- Source clock 50 Hz. Actions/tactile: offsets 3, 6, …, 48. Video: 6, 12, …, 48. Two observed and sixteen future tactile frames.
- Native camera images → T-mosaic → aspect-preserving 384 × 320 resize/pad. The six single-arm and five bimanual task definitions retain the original prompts.
- Quantile-normalized `[right22, left22]` policy state/action. Single-arm left padding is zero throughout clean/noise/target/inference and excluded from per-sample action loss.
- Physical force is `[left16, right16] × [Fn, Fu, Fv] × 10 × 14`. Missing hands have no support. The frozen encoder outputs 34 × 48 slots, with slots 3 and 20 absent; Hand2 packing uses two 768-D hand vectors per frame.
- Full bidirectional joint attention; bf16; independent uniform flow-matching times with shift 1.0; ten inference denoising steps.
- No understanding expert. WAN initialized from the exact 825-tensor Stage-1 file. Action/tactile experts are freshly initialized; frozen tactile EMA is step 50,000.

## Checks

Run `WAN_DISABLE_FLASH_ATTN=1 python -m unittest discover -s tests -v` in the policy environment. Tests cover right-first action roundtrips, shortest-arc quaternion interpolation and exact 48-step waypoints, invalid-hand loss/noise masking, Hand2 slot packing, and near-constant quantile normalization, and rejection of mismatched rgb256 checkpoint metadata.

Actual smoke results and tested environment versions are recorded separately in `dexjoco_validation.md` after execution. A smoke pass verifies execution, not 60,000-step convergence or benchmark success.
