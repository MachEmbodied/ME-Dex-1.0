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
  <a href="https://huggingface.co/liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard">Model Weights</a>
  &nbsp;·&nbsp;
  <a href="#robotwin-performance">Results</a>
  &nbsp;·&nbsp;
  <a href="#getting-started">Getting Started</a>
</p>

---

ME-Dex-1.0 is a video-action-tactile policy trained on RoboTwin Clean50. This repository provides its inference runtime for standardized RoboTwin leaderboard evaluation using XPolicyLib.

> **Coming soon:** Training code and data will be released publicly.

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

<table align="center">
  <thead>
    <tr>
      <th align="center">Method</th>
      <th align="center">Clean → Clean</th>
      <th align="center">Clean → Random</th>
      <th align="center">Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>ME-Dex-1.0</strong></td>
      <td align="center"><strong>89.6%</strong></td>
      <td align="center"><strong>68.1%</strong></td>
      <td align="center"><strong>78.9%</strong></td>
    </tr>
  </tbody>
</table>

</div>

## Model Weights

- **Policy & tactile AE:** [ME-Dex-1.0 on Hugging Face](https://huggingface.co/liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard).
- **Backbone assets:** VAE, T5 encoder, tokenizer and config from [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

## Getting Started

Install the runtime dependencies:

```bash
pip install -r runtime/requirements.txt
```

### Evaluation Notes

**Visual inputs.** The evaluation interface supplies RGB observations. Set `input_color_order: bgr` for a single RGB-to-BGR conversion, matching the checkpoint. No mean/std image normalization is applied.

**Tactile inputs.** Clean50 training included additionally collected three-axis tactile force data. Leaderboard observations contain no tactile measurements, so the current-frame tactile force is set to zero while preserving the sensor support mask. Future tactile states are predicted by the model.

## Acknowledgements

Built on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Motus](https://github.com/motus-robotics/Motus).

## License

[Apache-2.0](LICENSE).
