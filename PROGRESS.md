# Progress Log

Updated periodically as work moves forward. Most recent entry first.

---

## 2026-10-05 — Fused attention (SDPA) for Swin3D: measured speedup

- Decision: the user answered the TensorRT question with "proceed with q1"; I took that as the PyTorch-side option and optimized against the profile instead of using TensorRT (still blocked, see below)
- Cause found in the torchvision source: the fp32 relative-position-bias parameter is added to the bf16 attention matrix, which promotes the whole [windows, heads, 392, 392] tensor to fp32; the mask add, softmax and dropout then run in fp32. The shift mask is also rebuilt with Python loops on every call
- Patch (`patch_swin3d_sdpa`): computes bias and mask in the compute dtype, calls `scaled_dot_product_attention`, and caches the mask. Tests (5 pass): fp32 outputs match the stock model within 1e-4 at two input sizes, gradients to the relative-position-bias table match, bf16 CUDA output stays within a rounding-scale tolerance of the fp32 stock output, all 12 attention layers patched
- Training step, 16 frames, batch 4, bf16 autocast, AdamW, no checkpointing; three alternating repetitions of 10 timed steps after 1 warm-up, current GPU state:

| Variant | Sec/step (reps) | Mean | Peak VRAM |
|---|---|---|---|
| Stock | 0.219, 0.219, 0.211 | 0.216 | 5.39 GiB |
| SDPA patch | 0.143, 0.131, 0.132 | 0.136 | 2.72 GiB |

- Result: 1.60x faster training step and 49.5% less peak VRAM. Single-batch and batch-2 runs (3 timed steps each) agree in direction: batch 1 0.059 -> 0.048 s, batch 2 0.111 -> 0.074 s
- Consequence: with the faster step, the input pipeline (about 6.4 videos/s at 4 workers) is now slower than the GPU (about 29 videos/s with the patch), so end-to-end training is input-bound until the loader is improved
- Not yet measured: inference-only latency and precision comparison, end-to-end epoch time, accuracy effect of the patch after training (the patched model is mathematically the same function; not retrained)

---

## 2026-10-05 — Correction: earlier GPU timings were taken in a slow GPU state

- While benchmarking the patch, the stock model ran 0.059 s/step at batch 1 where the identical benchmark earlier today gave 0.871 s. Re-running the original benchmark script unchanged now gives 0.062 s (batch 1) and 0.212 s (batch 4), against 0.871 s and 3.556 s earlier. Peak memory is identical (1.67 and 5.40 GiB), so the workload is the same and the GPU was running about 14-17x slower earlier
- Re-profile under the current state (same script, same 4,797 kernel launches): total kernel time 0.61 s over 3 steps vs 10.42 s before; mean kernel 128 us vs 2,173 us. Training step compute 0.236 s; data wait 0.235 s mean (49.9% of the step)
- Cause of the earlier slowness: not determined. Candidates are other GPU workloads or a power/clock state at the time (browser/wallpaper/game processes were present on the GPU in the first nvidia-smi listing; power plan is Performance on AC now); none was verified
- What is superseded: all seconds-per-step figures from the model-memory benchmark, the 3.55 s/step profile, "GPU-bound / 0.0% data wait", and the clock/power readings from the first profile. What is unaffected: accuracy results (the pilot baseline, the overfit test), the loader benchmark (CPU-side, 6.41 videos/s at 4 workers, 16 frames), peak-memory figures, label statistics
- Current kernel-time split (name-pattern heuristic): copy/cast/layout 27.3%, GEMM 26.7%, elementwise 21.8%, softmax 13.9%, layernorm 5.0%, reductions 4.2%. GPU busy 49.0% of the kernel span because the GPU waits for data
- Lesson: record GPU clocks and a quick reference benchmark with every timing; absolute timings from this laptop are only comparable within one session

---

## 2026-10-05 — TensorRT attempt: blocked by Windows Application Control

- Installed `onnx` 1.23.1, `onnxscript` 0.7.2 and `tensorrt` 11.3.0.99 (CUDA 13 build; the runtime-libs wheel is 1.95 GB) into the local venv
- ONNX export of the trained model succeeds (torch.onnx dynamo exporter, opset 18, 16x224x224 input; 1,354 graph nodes; passes the ONNX checker)
- `import tensorrt` fails: "An Application Control policy has blocked this file." Read-only checks: the Python binding `tensorrt.cp313-win_amd64.pyd` is not code-signed, NVIDIA's `nvinfer_11.dll` is signed, and the Smart App Control registry value is 1 (enforcing). No security settings were changed and no workaround was attempted
- Consequence: no TensorRT engine was built and no FP32 vs FP16 TensorRT numbers exist; nothing is claimed about TensorRT speedups. The `tensorrt` package remains installed but unusable until the policy question is settled
- Open question for the project owner is recorded in the (local) task file; options considered: PyTorch-side optimization against the profile above, ONNX Runtime GPU, turning Smart App Control off (not reversible without a Windows reset per Microsoft's documentation; not done), or TensorRT under WSL2/Linux

---

## 2026-10-05 — Profiling a training step (torch.profiler)

- **SUPERSEDED in part: see "Correction: earlier GPU timings" above.** The conclusions below ("GPU-bound", "0.0% data wait", step time 3.55 s) were measured while the GPU was in a much slower state and do not hold now. Kept for the record.
- Config: the baseline setup (Swin3D-T, 16 frames, batch 4, bf16 autocast, 4 loader workers), steady state after 2 warm-up steps, then `torch.profiler` over 3 active steps. Nsight Systems is not installed on this machine (no `nsys`), so this is `torch.profiler` only. Trace analysis uses GPU kernel events (not op wrappers, which double count)
- Step time: 3.55 s per step of 4 videos, measured without the profiler (12 steps; max 3.57 s). Kernel span in the profiled steps: 3.57 s/step, consistent with it
- Input pipeline: data wait was 0.000 s mean (0.0% of the step). Training is not input-bound
- GPU occupancy: GPU busy 97.4% of the kernel span (union of kernel intervals), 1,599 kernel launches per step, median kernel 179 us. So the step is GPU-bound, not CPU/launch-bound
- Where the GPU time goes (sum of kernel durations, family assigned by kernel-name pattern, so approximate):

| Kernel family | Share of GPU time | Launches (3 steps) |
|---|---|---|
| Copy / dtype cast / layout | 28.8% | 2,289 |
| Elementwise (add, mul, GELU, dropout...) | 27.1% | 1,005 |
| GEMM / matmul (tensor-core work) | 18.8% | 762 |
| Softmax | 16.2% | 78 |
| LayerNorm | 4.5% | 261 |
| Reductions | 4.0% | 261 |

- Biggest single kernels: a bf16 copy/cast kernel 12.8%, an fp32 elementwise binary kernel 10.8%, fp32 softmax backward 9.5% and forward 6.7% (softmax runs in fp32 even under bf16 autocast), cutlass bf16 GEMMs about 5.7% each
- Reading: only about a fifth of GPU time is the matmul work tensor cores are good at; most is memory-bound copies, elementwise ops and fp32 softmax in the attention path. That matches the earlier finding that cost scales linearly with batch (no batching gain)
- GPU clocks while busy (samples with util >= 80%, n=66): SM clock mean 1,741 MHz vs 3,090 MHz maximum, power mean 32.8 W, temp max 71 C. I did not determine why the clock is well below maximum (power/thermal limits and memory-bound load are both candidates); this is not established
- Not tested (hypotheses only): fused attention via `scaled_dot_product_attention`, `torch.compile`, avoiding fp32 softmax, fewer layout copies. These are the candidate optimizations for a future step; no speedup has been measured
- Not measured: a full-epoch profile (3 steps profiled, 12 timed), multiple runs, or a comparison with another backbone

---

## 2026-10-05 — Phase 1 pilot baseline run

- Swin3D-T (K400 init) trained on the 100-video pilot train split for 15 epochs (about 99 s/epoch, about 25 min total; these durations were measured in the slow GPU state described in the Correction entry, so they overstate current epoch time; accuracy results are unaffected, 16 frames, batch 4, bf16, seed 0). Val evaluated each epoch; test evaluated once at the best-val checkpoint. Full details and caveats in RESULTS.md
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

- **Timing columns SUPERSEDED: see "Correction: earlier GPU timings" above.** The peak-memory columns are unaffected (re-measured identical). The seconds-per-step values were taken in a slow GPU state; the same configs now run about 14-17x faster.
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
