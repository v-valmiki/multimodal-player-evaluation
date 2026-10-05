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
| 1 | Baseline replication, NCAA D1 subset | — | Pending |
| 2 | + Explicit player tracking | — | Pending |
| 3 | + Multimodal fusion (stats + pose) | — | Pending |
| 4 | + Clip-level aggregation | — | Pending |

### Backbone substitution (Phase 1)

The plan named VideoMamba (74M) as the Phase 1 backbone. Its pretrained weights are public on Hugging Face (`OpenGVLab/VideoMamba`), but the model depends on the compiled `mamba_ssm` CUDA kernels. On the local machine (Windows, Python 3.13, RTX 50-series) pip could not obtain a build of `mamba-ssm`, and neither `nvcc` nor the MSVC compiler is installed. Phase 1 therefore uses **torchvision Swin3D-T (27.9M parameters, Kinetics-400 pretrained)** as a stand-in video transformer. Phase 1 numbers are therefore **not a VideoMamba reproduction** and must not be compared to the paper's 28.5% as if they were. Revisit VideoMamba if a CUDA toolchain is set up (or on a Linux cloud/Savio machine).

## Per-Skill-Category Breakdown

Fill in once Phase 1 baseline is available, to track how each subsequent phase shifts category-level performance relative to the paper's own breakdown (Shooting 26.47%, Rebounding 26.62%, Defense 27.38%, Playmaking 25.39%, Offense 34.60%).

| Phase | Shooting | Rebounding | Defense | Playmaking | Offense |
|---|---|---|---|---|---|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
