# Results

Updated as each phase produces a number. All accuracy figures are top-1, averaged across the 20 skill categories unless otherwise noted.

## Reference Point (from the published paper, not this project)

| Model | Test Acc (%) |
|---|---|
| Random baseline | 20.00 |
| Best published (VideoMamba, full dataset, 8x A6000) | 28.50 |
| Human experts | 66.00 (avg), 72.00 (top) |

## This Project

| Phase | Description | Overall Acc (%) | Notes |
|---|---|---|---|
| 1 (PILOT) | Swin3D-T (K400) on 150-video pilot, 16 frames, test n=30 | 18.5 | Chance level. NOT comparable to the paper's 28.5% (different backbone; 100 of the 19,578 official training videos, about 0.5%). See below |
| 2 | + Explicit player tracking | — | Pending |
| 3 | + Multimodal fusion (stats + pose) | — | Pending |
| 4 | + Clip-level aggregation | — | Pending |

### Backbone substitution (Phase 1)

The plan named VideoMamba (74M) as the Phase 1 backbone. Its pretrained weights are public on Hugging Face (`OpenGVLab/VideoMamba`), but the model depends on the compiled `mamba_ssm` CUDA kernels. On the local machine (Windows, Python 3.13, RTX 50-series) pip could not obtain a build of `mamba-ssm`, and neither `nvcc` nor the MSVC compiler is installed. Phase 1 therefore uses **torchvision Swin3D-T (27.9M parameters, Kinetics-400 pretrained)** as a stand-in video transformer. Phase 1 numbers are therefore **not a VideoMamba reproduction** and must not be compared to the paper's 28.5% as if they were. Revisit VideoMamba if a CUDA toolchain is set up (or on a Linux cloud/Savio machine).

### Phase 1 pilot run: what it does and does not show

- Setup: Swin3D-T (Kinetics-400 init), 100 train / 20 val / 30 test videos (NCAA D1 2018-19), 16 frames at 224x224, bf16, batch 4, no gradient checkpointing (not needed at this batch size), AdamW + one-cycle schedule, 15 epochs, seed 0. Hyperparameters were fixed before the run; none were tuned on val or test. Best epoch chosen on val, test evaluated once.
- Test accuracy 18.5% (val 24.0% at the selected epoch), against 20.0% for uniform random and 21.3% for always predicting each skill's train-majority level. Val accuracy over all 15 epochs ranged 16.5% to 24.0% with mean 19.2%, i.e. chance. The selected "best" epoch was epoch 1, picked by noise on a 20-video val set.
- Training loss fell from 1.63 to 0.53 while val loss rose from 1.61 to 2.22: the model memorized the 100 training videos and did not generalize. The earlier overfit smoke test (8 videos to 99.4%) shows the pipeline itself can learn.
- Per-category test accuracies (24.0 / 16.7 / 18.3 / 13.3 / 17.3) come from 30 videos per skill and are noise-dominated; they should not be read as a category pattern (for example, val Offense was 30.0% while test Offense was 17.3%).
- Interpretation limits: this run cannot separate "too little data" from "the signal is not learnable by this backbone/setup". The paper trains on the full dataset and reaches 28.5% even at best, so a pilot at chance is not surprising, but also not evidence either way.

## Inference optimization (Phase 1 pilot model)

Measured on the RTX 5070 Ti Laptop (12 GB), 16 frames at 224x224, the trained pilot checkpoint, 30 timed forwards after 5 warm-up (median), single session. Correctness is reported as agreement with the fp32 stock model over all 50 val+test pilot videos (1,000 per-skill predictions), not as accuracy: the pilot model is at chance (about 18.7% test), so accuracy deltas would be meaningless.

"SDPA" = the fused-attention patch for Swin3D (see PROGRESS.md); "weights" = the model converted to that dtype, versus "autocast".

| Variant | Latency b1 (ms) | Latency b4 (ms) | Throughput b4 (videos/s) | Peak VRAM b4 (GiB) | Top-1 agreement | Max abs logit diff |
|---|---|---|---|---|---|---|
| fp32 stock | 31.5 | 114.6 | 34.9 | 1.37 | 1.0000 (reference) | 0 (reference) |
| fp32 + SDPA | 24.0 | 90.3 | 44.3 | 1.05 | 1.0000 | < 0.0001 |
| bf16 autocast stock | 22.2 | 81.8 | 48.9 | 1.40 | 0.9970 | 0.0023 |
| bf16 autocast + SDPA | 11.8 | 40.6 | 98.5 | 0.78 | 0.9970 | 0.0024 |
| fp16 autocast stock | 22.9 | 77.4 | 51.7 | 1.47 | 1.0000 | 0.0003 |
| fp16 autocast + SDPA | 12.6 | 40.7 | 98.3 | 0.85 | 1.0000 | 0.0004 |
| bf16 weights + SDPA | 10.7 | 36.0 | 111.1 | 0.65 | 0.9960 | 0.0029 |
| **fp16 weights + SDPA** | **9.6** | **34.5** | **115.9** | **0.65** | **1.0000** | **0.0004** |

- Best variant (fp16 weights + SDPA) versus fp32 stock: 3.3x lower batch-1 latency (31.5 to 9.6 ms), 3.3x higher batch-4 throughput (34.9 to 115.9 videos/s), 53% less peak VRAM, identical top-1 predictions on all 1,000 checked, max logit difference 0.0004.
- bf16 flips 3-4 of 1,000 predictions relative to fp32 (agreement 0.996-0.997) with max logit difference about 0.002-0.003; fp16 flipped none, which is expected from fp16's extra mantissa bits at these logit magnitudes.
- Caveats: one session, one checkpoint, 50 evaluation videos; fp32 uses PyTorch defaults (no TF32 matmuls); fp16 weights were safe for this model on these inputs, which is not a general guarantee. This is PyTorch-side optimization; no TensorRT numbers exist (TensorRT is blocked on this machine, see PROGRESS.md).

## Per-Skill-Category Breakdown

Fill in once Phase 1 baseline is available, to track how each subsequent phase shifts category-level performance relative to the paper's own breakdown (Shooting 26.47%, Rebounding 26.62%, Defense 27.38%, Playmaking 25.39%, Offense 34.60%).

| Phase | Shooting | Rebounding | Defense | Playmaking | Offense |
|---|---|---|---|---|---|
| 1 (PILOT, test n=30) | 24.0 | 16.7 | 18.3 | 13.3 | 17.3 |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
