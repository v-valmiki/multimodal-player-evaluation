# Progress Log

Updated periodically as work moves forward. Most recent entry first.

---

## 2026-10-05 — Phase 1 pilot baseline run

- Swin3D-T (K400 init) trained on the 100-video pilot train split for 15 epochs (about 99 s/epoch, about 25 min total, 16 frames, batch 4, bf16, seed 0). Val evaluated each epoch; test evaluated once at the best-val checkpoint. Full details and caveats in RESULTS.md
- Result: test 18.5% (val 24.0% at the selected epoch, epoch 1). Val accuracy across all epochs: min 16.5%, max 24.0%, mean 19.2%. Baselines on the same test split: 20.0% uniform random, 21.3% train-majority
- Train loss 1.63 -> 0.53, val loss 1.61 -> 2.22: memorization without generalization at this data size
- This is a working end-to-end training pipeline with an honest pilot number at chance, not a result. Meaningful accuracy needs the full-season data (about 94 GB, over this loop's 10 GB download limit, so not started)
- Not run: multiple seeds, LR or augmentation sweeps (the 20/30-video val/test sets are too small to tell configurations apart)

---

## 2026-10-05 — Overfit smoke test

- Setup: first 8 pilot-train videos, 16 frames, batch 4 (2 steps/epoch), bf16 autocast, AdamW (lr 1e-4 backbone, 1e-3 head), seed 0, dropout and stochastic depth off. Accuracy = mean over the 20 skills, measured in eval mode on the same 8 videos
- Result: accuracy rose from 15.0% (untrained head) to 99.4% in 7 epochs (about 9.6 s/epoch), loss 1.63 -> 0.26

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Loss | - | 1.627 | 1.471 | 1.291 | 1.133 | 0.743 | 0.523 | 0.257 |
| Acc | 0.150 | 0.406 | 0.469 | 0.525 | 0.762 | 0.869 | 0.938 | 0.994 |

- This shows the pipeline learns (data -> model -> loss -> optimizer -> eval agree with each other). It shows memorization capacity only, not generalization; the label counts across the 8 videos were 42/34/31/29/24 for levels 0-4
- Single seed, single run

---

## 2026-10-05 — Model wrapper and GPU cost

- Added `SkillClassifier`: torchvision Swin3D-T (Kinetics-400 weights, 27.9M params) + dropout + linear head -> logits [B, 20, 5], mean cross-entropy over the 20 skills, optional per-stage gradient checkpointing
- Why not VideoMamba: weights exist on Hugging Face, but `mamba_ssm` (CUDA kernels) could not be obtained by pip here and there is no `nvcc`/MSVC; I did not attempt a from-source build. See RESULTS.md for the substitution note
- 4 model tests pass (shape, backward, checkpointed gradients equal plain gradients, bf16 CUDA forward with pretrained weights). One test initially failed because Swin3D applies stochastic depth in train mode; seeding the RNG identically fixed it, so the model code was not at fault
- Measured on the RTX 5070 Ti Laptop (12 GiB), 224x224, bf16 autocast, AdamW step included, mean of 3 steps after 1 warm-up:

| Frames | Batch | Grad ckpt | Peak GiB | Sec/step |
|---|---|---|---|---|
| 16 | 1 | no | 1.67 | 0.871 |
| 16 | 1 | yes | 1.20 | 1.166 |
| 16 | 2 | no | 2.91 | 1.742 |
| 16 | 2 | yes | 1.95 | 2.305 |
| 16 | 4 | no | 5.40 | 3.556 |
| 16 | 4 | yes | 3.43 | 4.774 |
| 16 | 8 | no | 10.37 | 54.905 |
| 16 | 8 | yes | 6.41 | 7.948 |
| 32 | 1 | no | 2.91 | 1.444 |
| 32 | 1 | yes | 1.95 | 1.942 |
| 32 | 2 | no | 5.39 | 3.562 |
| 32 | 2 | yes | 3.44 | 4.841 |
| 32 | 4 | no | 10.37 | 12.937 |
| 32 | 4 | yes | 6.41 | 9.844 |
| 32 | 8 | no | out of memory | - |
| 32 | 8 | yes | 12.37 (above the card's ~11.9 GiB) | 24.941 |

- Checkpointing saves about 35% peak memory (e.g. 5.40 -> 3.43 GiB at 16 frames, batch 4) at about 1.3x step time
- Throughput is roughly linear in batch (0.87 s for 1 video, 3.56 s for 4), i.e. no batching gain. That is slow for a 28M-parameter model and has not been explained; profiling is task 7. Other apps share the GPU (about 1.4 GiB in use before I started)
- Rows that approach or exceed ~10 GiB (16f/b8/no-ckpt at 54.9 s/step, 32f/b8/ckpt) are far slower per sample; consistent with memory pressure but I did not verify the cause
- Baseline-run plan (task 6): 16 frames, batch 4 fits comfortably without checkpointing (5.40 GiB, 3.56 s/step, about 25 steps per 100-video epoch). TASKS.md suggested checkpointing; it is not needed at this batch size, and I will say so in the run record

---

## 2026-10-05 — Pilot class balance

- Per-skill label distribution computed for pilot train (100), val (20), test (30); figure saved locally to `reports/class_balance.png` (gitignored)
- Baselines, averaged over the 20 skills (measured on the pilot labels):

| Split | Uniform random | Always predict train-majority level per skill |
|---|---|---|
| train | 20.0% | 24.5% (in-sample) |
| val | 20.0% | 18.8% |
| test | 20.0% | 21.3% |

- Train is close to balanced: the most common level of any skill holds at most 29% of videos (Offensive Consistency, Offensive Rebound), the least common level holds at least 10%
- Val and test are small, so their label shares are noisy (val: one video = 5% of a row; shares range 5% to 40%; test 7% to 43%)
- Consequence: on the pilot, val/test accuracy differences of a few points should not be read as real. The pilot run proves the pipeline works; it is not a result comparable to the paper's 28.5%

---

## 2026-10-05 — Loader throughput benchmark

- Added `make_loader` (pin_memory, persistent workers, prefetch 2) and 3 loader tests; full suite: 10 passed in 13.6 s
- Benchmark: pilot train split (100 videos), batch size 4, decord, one untimed warm-up of 3 batches then one timed full pass per config, AMD Ryzen AI 9 HX 370 (12 cores / 24 threads). Single run per config, so treat small differences as noise.

| Frames | Workers | Seconds (100 videos) | Videos/s |
|---|---|---|---|
| 16 | 0 | 34.7 | 2.88 |
| 16 | 2 | 20.7 | 4.83 |
| 16 | 4 | 15.6 | 6.41 |
| 16 | 8 | 25.3 | 3.95 |
| 16 | 12 | 15.9 | 6.27 |
| 32 | 0 | 53.7 | 1.86 |
| 32 | 2 | 34.6 | 2.89 |
| 32 | 4 | 24.7 | 4.05 |
| 32 | 8 | 24.7 | 4.06 |
| 32 | 12 | 24.5 | 4.09 |

- Scaling flattens after 4 workers (about 2.2x over 0 workers at 16 frames, 2.2x at 32). The 16-frame / 8-worker dip is unexplained: it was not re-run and the cause is not measured.
- Per-video cost is far above the earlier sequential-decode figure (~6,200 fps), so random-access seeking across a 15.8k-frame H.264 file, not raw decode speed, is the likely cost. This is a hypothesis; not yet profiled.
- Consequence for training: at 16 frames the loader alone needs about 16 s per pilot epoch at best (4 workers); use 4 workers as the default.

---

## 2026-10-05 — Dataset class

- Added a `BasketDataset` (decord decode with in-decoder resize to 224x224, deterministic uniform frame sampling over the full video, ImageNet normalization) returning `frames[T,3,224,224]` and 20 skill labels (0-4)
- 7 pytest tests pass in 14.6 s on the pilot CSVs: shape/dtype for T=8 and 16, label range, length, determinism, distinct videos differ, index coverage
- Visually checked a decoded 8-frame strip: colors and content correct; frames are aspect-squashed because 704x400 is resized directly to 224x224 (matches the plan)
- Not yet measured: loader throughput (task 2)

---

## 2026-10-05 — Data access and pilot subset

- Hugging Face login confirmed; gated BASKET access works
- Downloaded annotation CSVs (~2.8 MB) and scoped a 150-video NCAA D1 2018-19 pilot (~9 GB), all downloaded and decodable
- Pilot labels look reasonably balanced across the 5 levels
- Decode throughput measured at ~6,200 fps on CPU (704x400 H.264), so GPU decoding is not an obvious early win
- Next: dataset class and loader (uniform frame sampling, 224x224), per-skill class-balance check, then a first VideoMamba run on the pilot

---

## 2026-10-05 — Local GPU environment

- Local compute added: RTX 5070 Ti Laptop GPU (12 GB), driver CUDA 13.1
- Python 3.13 venv with PyTorch 2.11 (cu128 build, required for Blackwell); bf16 matmul on GPU verified
- Data/video stack installed and pinned in `requirements.txt` (huggingface_hub, pandas, pyarrow, av, decord, opencv)
- Not yet installed: CUDA Toolkit (`nvcc`, needed only for custom kernels later), TensorRT (optimization phase)
- Next: Hugging Face login, inspect BASKET file layout and size, download NCAA D1 subset only

---

## [DATE] — Project initialized

- Repo scaffolded, documentation written
- Reviewed BASKET paper (CVPR 2025) in full — see FINDINGS.md
- Compute access confirmed: Berkeley DataHub (CPU) available now; cloud GPU credits application in progress; CS 199 / Savio access targeted for next semester
- Data scope decided: NCAA Division I subset
- Next: begin Phase 0 — Hugging Face data agreement, download scripts, data loader

---

**Current phase:** 0 — Pipeline Scaffolding
**Blocked on:** nothing currently
