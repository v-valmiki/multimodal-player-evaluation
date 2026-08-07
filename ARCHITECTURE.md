# Architecture & Phase Plan

See [FINDINGS.md](FINDINGS.md) for the published results this plan is grounded in.

## Compute Tiers

| Tier | Resource | Availability |
|---|---|---|
| 0 | Berkeley DataHub (CPU-only JupyterHub) | Now |
| 1 | Cloud student credits (GCP/AWS/Azure, T4/L4 GPU) | Now |
| 2 | Savio HPC (L40s/V100s/A5000s) via CS 199 faculty sponsorship | Pending enrollment |

## Phase 0 — Pipeline Scaffolding (Tier 0)

- Data access: Hugging Face agreement + scoped download of NCAA D1 subdirectories only
- Data loader: index video paths from the provided `train.csv` / `val.csv` / `test.csv` annotation structure, converting relative paths to absolute per BASKET's own README recommendation
- Preprocessing: frame extraction, resize to 224x224 (matches the paper's setup for a fair comparison), clip segmentation
- Stats-source scouting: identify a play-by-play / box-score data source for NCAA D1 covering the same seasons as the video subset — this is the second input stream for Phase 3 fusion
- Class-balance check across the 5 skill levels in the NCAA D1 subset

**Output:** working ingestion + preprocessing pipeline, validated on a handful of videos, no GPU training yet.

## Phase 1 — Baseline Replication (Tier 1)

- Backbone: VideoMamba (74M params) — smallest strong performer in the paper's table, most realistic to fit on a single T4/L4
- Frames: reduced from their optimal 64 to 16–32 given single-GPU memory limits
- Mixed precision (fp16/bf16) + gradient checkpointing to fit in 16–24GB VRAM
- Train on NCAA D1 subset only, holding out one season for testing (mirrors the paper's own held-out 2017–18 season design)
- Report this number as an honest reproduction baseline under a smaller data/compute budget, not a like-for-like comparison to their 28.5%

## Phase 2 — Explicit Player Tracking (Tier 1)

- Add a tracking/re-identification preprocessing step (e.g., ByteTrack + appearance-based re-ID) to identify and crop the subject player across clips before feeding the video backbone
- Compare against the Phase 1 baseline on the same held-out split
- Break down improvement by skill category — hypothesis predicts the largest gains in Defense/Rebounding, where occlusion and off-camera positioning are most common

## Phase 3 — Multimodal Fusion (Tier 1 → 2)

- **Stats branch:** encode box-score/play-by-play features (points, FG%, rebounds, assists, turnovers, usage rate) for the games each highlight clip is drawn from
- **Pose branch:** run a lightweight pose estimator over key event clips (not the full video — too compute-expensive) to extract joint trajectories relevant to shot mechanics and footwork
- **Fusion:** late fusion first — concatenate video backbone embedding + stats embedding + pose summary embedding before the 20 skill classification heads
- Compare against Phase 1/2 baselines, broken down by skill category — hypothesis predicts the largest gains in Offense-category skills, partial gains in Shooting

## Phase 4 — Clip-Level Aggregation (Tier 2)

- Alternative to encoding the full 9-minute shuffled highlight as one sequence: score each of the ~50 discrete event clips independently, then aggregate with attention-based pooling
- Mirrors how the highlight videos are actually constructed (50 discrete events) and how human scouts consume film
- Needs more compute (multiple forward passes per player) — realistic once Savio access lands

## Phase 5 — Downstream Evaluation Model

- Freeze the fine-tuned multimodal skill estimator from Phase 3/4
- Use its 20-skill probability output + stats summary as input features to a downstream model
- Downstream task options: composite prospect grade compared against public scouting rankings, or a structured scouting report generator (strengths/weaknesses by skill category)
- This is the actual end deliverable — the skill estimator is a feature extractor, not the final product

## Metrics Tracked at Every Phase

| Metric | Purpose |
|---|---|
| Overall top-1 accuracy (avg across 20 skills) | Direct comparison point to the paper's 28.5% |
| Per-skill-category accuracy (5 categories) | Tests the core fusion hypothesis |
| Cross-season generalization | Comparable to the paper's Table 5 |
| Cross-league generalization (if subset spans multiple leagues) | Tests whether architecture changes affect the paper's Finding 4 |
| Params / compute budget used | Honesty check against the paper's 8x A6000 setup |
