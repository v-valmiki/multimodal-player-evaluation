# Basketball Player Evaluation via Multimodal Deep Learning

**Status:** Active — Phase 0 (pipeline development)
**Started:** July 2026

## Problem

[BASKET](https://arxiv.org/abs/2503.20781) (Pan, Zhang & Bertasius, CVPR 2025) introduced a large-scale benchmark for fine-grained basketball skill estimation: 4,477 hours of video across 32,232 players, rated on 20 skills from "Poor" to "Excellent." Given a long highlight video, a model has to predict a player's level on each skill.

Their strongest video model (VideoMamba) reaches **28.5%** top-1 accuracy. Human experts reach **66%**, with top performers at 72%. That gap — and the fact that scaling web-pretrained backbones does not close it — is the starting point for this project.

## Hypothesis

The BASKET benchmark is video-only. No prior work fuses video with box-score statistics, explicit player tracking, or pose estimation. The paper's own results hint at why this matters: skill categories that correlate with statistical aggregates (their "Offense" category — Contribution, Efficiency, Impact) score substantially higher (~35%) than technique-heavy categories like Shooting (~26%). This project tests whether:

1. **Fusing video with box-score/play-by-play statistics** improves accuracy, with the largest gains concentrated in stats-correlated skill categories
2. **Explicit player tracking/re-identification**, replacing the original paper's implicit player-finding approach, improves accuracy on skills most affected by occlusion (Defense, Rebounding)
3. A **clip-level aggregation architecture** (scoring each of the ~50 discrete highlight clips independently, then pooling) outperforms encoding the full shuffled highlight as one sequence

See [FINDINGS.md](FINDINGS.md) for the full grounding in the published paper's results.

## Approach

| Phase | Goal | Compute Tier |
|---|---|---|
| 0 | Data pipeline, ingestion, preprocessing | Berkeley DataHub (CPU) |
| 1 | Baseline replication on NCAA D1 subset | Cloud GPU credits |
| 2 | Explicit player tracking ablation | Cloud GPU credits |
| 3 | Multimodal fusion (video + stats + pose) | Cloud GPU credits → Savio |
| 4 | Clip-level aggregation architecture | Savio (CS 199) |
| 5 | Downstream evaluation/scouting model | Savio or lightweight |

Full technical detail in [ARCHITECTURE.md](ARCHITECTURE.md).

## Data

NCAA Division I subset of BASKET. Chosen because it matches the exact subset used in the paper's own human-evaluation study (2018–19 season), giving a direct, citable human-performance comparison without re-deriving a baseline. See [DATA_SOURCES.md](DATA_SOURCES.md).

## Current Status

See [PROGRESS.md](PROGRESS.md) for the latest update.

## Results

Tracked in [RESULTS.md](RESULTS.md) as each phase completes.

## Code Status

This project is in active development with code maintained privately. The architecture, findings, and results are documented here for portfolio and discussion purposes. Happy to talk through design decisions, trade-offs, or the technical approach.

## Citation

This project builds directly on the BASKET benchmark:

```
@InProceedings{Pan_2025_CVPR,
  author    = {Pan, Yulu and Zhang, Ce and Bertasius, Gedas},
  title     = {BASKET: A Large-Scale Video Dataset for Fine-Grained Skill Estimation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025}
}
```
