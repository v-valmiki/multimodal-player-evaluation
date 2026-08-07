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

## Statistical Data (Phase 3 fusion input)

Source TBD — needs to cover box-score/play-by-play data for the same NCAA D1 seasons and players present in the BASKET video subset. Update this section once a source is selected.

## Licensing Note

BASKET's video content is drawn from a basketball game replay archive; usage is governed by the terms set on their Hugging Face dataset page. Any statistical data source added for fusion must be checked for its own licensing terms before use.
