# Data Sources

## BASKET

- Paper: [arXiv:2503.20781](https://arxiv.org/abs/2503.20781)
- Project page: https://sites.google.com/cs.unc.edu/basket
- Data: hosted on Hugging Face — requires agreeing to their terms before download
- Scope used in this project: NCAA Division I subset only (not the full 21-league, 4,477-hour corpus)

**Why NCAA D1 only:**
1. Matches the project's stated scope
2. The paper's own human-evaluation study (Section 6.2) used exactly this subset (2018–19 season), giving a direct, citable human-baseline comparison
3. Fits available compute — the full dataset was trained on 8x NVIDIA RTX A6000 GPUs by the original authors; this project runs on cloud student credits (T4/L4) and Berkeley DataHub

### Measured dataset facts (verified 2026-10-05)

- Full repo: 32,246 files, 1.93 TB; annotation CSVs total ~2.8 MB
- Each row of `train/val/test.csv` is one player highlight video plus 20 skill levels (0 to 4)
- Videos: H.264, 704x400, ~31.3 fps, ~8.4 min (~15.8k frames), ~60 MB each
- NCAA D1 men's 2018-19: 1,566 videos (~94 GB); official split 1,046 train / 159 val / 361 test
- NBA, all five seasons: ~126 GB (alternative subset if NBA box-score fusion becomes the priority)
- Local CPU decode + RGB conversion: ~6,200 fps, so decoding is unlikely to be the first bottleneck at this resolution

### Local pilot subset

150 videos (100 train / 20 val / 30 test), randomly sampled with seed 0 from the NCAA D1 2018-19 slice of the official splits, ~9 GB. Pilot CSVs with absolute paths live under `data/BASKET/BASKET_annotation/pilot/` (git-ignored). Purpose: build and debug the loader, preprocessing and first training run before downloading the full season.

## Statistical Data (Phase 3 fusion input)

Source TBD — needs to cover box-score/play-by-play data for the same NCAA D1 seasons and players present in the BASKET video subset. Update this section once a source is selected.

## Licensing Note

BASKET's video content is drawn from a basketball game replay archive; usage is governed by the terms set on their Hugging Face dataset page. Any statistical data source added for fusion must be checked for its own licensing terms before use.
