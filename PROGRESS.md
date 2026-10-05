# Progress Log

Updated periodically as work moves forward. Most recent entry first.

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
