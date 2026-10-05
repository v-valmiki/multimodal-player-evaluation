# Progress Log

Updated periodically as work moves forward. Most recent entry first.

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
