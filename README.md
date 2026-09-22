<p align="center">
  <img src="assets/li-auto.svg" alt="Li Auto" width="220">
</p>

<h1 align="center">ME-Dex 1.0</h1>

<p align="center">
  Bringing Heterogeneous Tactile Sensing into World Action Modeling
</p>

<p align="center">
  <a href="https://machembodied.com/ME-Dex/ME-Dex1.0.html">Technical Report</a>
  &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2609.21449">ArXiv</a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard">Model Weights</a>
  &nbsp;·&nbsp;
  <a href="#getting-started">Getting Started</a>
</p>

---

ME-Dex-1.0 is a video-action-tactile policy trained on RoboTwin Clean50. This repository provides its inference runtime and tactile encoder for standardized RoboTwin leaderboard evaluation using XPolicyLib.

The training entry point is included under [`training/`](training/). The Clean50 tactile dataset is available on
[Hugging Face](https://huggingface.co/datasets/liuxuetao/MachEmbodied-Dex1.0-RoboTwin-Clean50-Tactile).
The Clean50 tactile replay procedure is documented in [`training/robotwin_tactile/`](training/robotwin_tactile/).
The large generated T5 payload is not stored in the dataset release; follow the
[T5 cache preparation](training/README.md#prepare-the-t5-cache) step before training.

<p align="center">
  <a href="assets/pipeline.pdf">
    <img src="assets/pipeline.png" alt="ME-Dex 1.0 framework: joint video-action-tactile modeling and heterogeneous tactile encoding" width="100%">
  </a>
</p>

<p align="center">
  <em>ME-Dex 1.0 framework.</em>
</p>

## RoboTwin Performance

<div align="center">

| Method | Clean → Clean | Clean → Random | Average |
|:--|:--:|:--:|:--:|
| **ME-Dex-1.0** | **88.9%** | **71.9%** | **80.4%** |

</div>

## Model Weights

- **Policy & tactile AE:** [ME-Dex-1.0 on Hugging Face](https://huggingface.co/liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard).
- **Backbone assets:** VAE, T5 encoder, tokenizer and config from [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

## Evaluation

Clone XPolicyLab and install the adapter:

```bash
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab
bash policy/ME_Dex_1_0/install.sh
```

Download the released checkpoint and Wan2.2 assets:

```bash
CHECKPOINT_DIR=checkpoints/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard

hf download liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard \
  --local-dir "${CHECKPOINT_DIR}"

hf download Wan-AI/Wan2.2-TI2V-5B \
  config.json Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/special_tokens_map.json \
  google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json \
  google/umt5-xxl/tokenizer_config.json \
  --local-dir "${CHECKPOINT_DIR}/wan"
```

Run a RoboTwin evaluation through the standard interface:

```bash
cd policy/ME_Dex_1_0
ROBOTWIN_TASK_CONFIG=demo_randomized \
ROBOTWIN_TEST_NUM=100 \
bash eval.sh RoboTwin adjust_bottle \
  ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard \
  arx_x5 joint 42 0 0 <policy_env> <robotwin_env>
```

Use `demo_clean` for Clean evaluation. The released configuration uses BGR input and zero observed tactile force with the sensor support mask preserved.

## Getting Started

Install the runtime dependencies:

```bash
pip install -r runtime/requirements.txt
```

### Evaluation Notes

**Visual inputs.** The evaluation interface supplies RGB observations. Set `input_color_order: bgr` for a single RGB-to-BGR conversion, matching the checkpoint. No mean/std image normalization is applied.

**Tactile inputs.** Clean50 training included additionally collected three-axis tactile force data. Leaderboard observations contain no tactile measurements, so the observed tactile frames are set to zero while preserving the sensor support mask. Future tactile states are predicted by the model.

## Training

The released reference recipe is in [`training/`](training/). Prepare the Clean50 dataset,
the Wan2.2 assets, the initialization checkpoint, and the tactile AE checkpoint, generate
the local Clean50 T5 cache, and then set
their paths in [`training/configs/clean50_uni.yaml`](training/configs/clean50_uni.yaml).

```bash
pip install -r training/requirements.txt
torchrun --nnodes=2 --nproc_per_node=16 \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m training.train --config training/configs/clean50_uni.yaml
```

Set `model.topology` to `full_joint` for the default joint-attention recipe or to
`h_bridge` for the corresponding bridge configuration.

## Acknowledgements

Built on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Motus](https://github.com/motus-robotics/Motus).

## License

[Apache-2.0](LICENSE).
