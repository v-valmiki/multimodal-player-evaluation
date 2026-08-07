# Findings From the BASKET Benchmark Paper

This document summarizes the published results this project builds on. Source: Pan, Y., Zhang, C., & Bertasius, G. (2025). *BASKET: A Large-Scale Video Dataset for Fine-Grained Skill Estimation.* CVPR 2025. [arXiv:2503.20781](https://arxiv.org/abs/2503.20781)

## Main Results

| Method | Pretraining | Frames | Params (M) | Test Acc (%) |
|---|---|---|---|---|
| Random Baseline | — | — | — | 20.00 |
| SigLIP | WLI+K400 | 64 | 203 | 21.85 |
| MeMViT | K400 | 64 | 212 | 23.01 |
| LLaVA-OneVision | CC+BLIP+SD+UR+SG | 32 | 8200 | 23.84 |
| X-CLIP | WIT+K400 | 16 | 196 | 24.37 |
| VideoMAE2 | UH+K400 | 32 | 1012 | 24.43 |
| TimeSformer | IN21K+K400 | 96 | 121 | 25.21 |
| UnmaskedTeacher | K710+CC+VG+SBUC+CC15M+WV12M | 32 | 90 | 26.97 |
| InternVideo2 | LA+WV10M+WV2M+SC+K710 | 32 | 1024 | 27.52 |
| **VideoMamba (best)** | IN1K+K400 | 64 | 74 | **28.50** |

Human baseline (3-level version of the task, 33.3% random floor): novices 43%, experts 66% average, 72% top performer. VideoMamba on the same 3-level task: 41%.

## Finding 1 — Skill-category accuracy is not uniform

| Skill Category | Avg Test Acc (%) |
|---|---|
| Shooting | 26.47 |
| Rebounding | 26.62 |
| Defense | 27.38 |
| Playmaking | 25.39 |
| **Offense** (Contribution, Efficiency, Impact, Teamwork, Consistency) | **34.60** |

Offense sub-skills — Impact (37.24%) and Efficiency (36.14%) specifically — score well above every other category. These are the skills most correlated with aggregate box-score production. The paper does not test statistical fusion, but this pattern is the direct motivation for Hypothesis 1 in this project: skills closer to statistical aggregates are easier for a pure video model to approximate; skills requiring visual technique judgment (Contested-shots: 24.14%) are where video alone struggles most.

## Finding 2 — Scaling web pretraining does not transfer

InternVideo2 (1024M params, pretrained on LAION + WebVid + Kinetics-710) scores 27.52%, barely above VideoMamba (74M params, 28.50%). Larger, more heavily pretrained backbones are not the lever that closes the gap to human performance.

## Finding 3 — No explicit player identification

Per the paper's own task description (Section 4): player highlight videos are 50 randomly shuffled event clips with no bounding-box annotation. The model must implicitly learn to identify and track the subject player across camera cuts, occlusion, and visually similar teammates/opponents. This project's Phase 2 tests whether making this step explicit improves accuracy.

## Finding 4 — Cross-league generalization is the weakest point

| Season | League | Test Acc (%) |
|---|---|---|
| Seen | Seen | 28.50 |
| Unseen | Seen | 23.09 |
| Seen | Unseen | 26.83 |
| Unseen | Unseen | 23.53 |

The drop from unseen leagues (>4 points) is larger than the drop from unseen seasons, suggesting models partly learn league-specific visual context (uniforms, court branding, broadcast framing) rather than transferable skill cues.

## Finding 5 — Supervision signal is coarse

The paper's own ablation (Table 4c in the original paper) shows that re-sampling a different random set of 50 event clips per player changes accuracy by only about 1.3 points of variance. Combined with the 5-level bucketed label (ranked within league/season), this indicates the ground-truth signal itself is noisy — a factor to account for when interpreting any accuracy delta this project reports.

## Ablations Worth Inheriting

- 64 input frames outperformed 16, 32, and 128 in their sweep — but this project's compute budget (single cloud GPU vs. their 8x A6000) may require operating at 16–32 frames, which their own ablation shows still clears the random baseline meaningfully (32 frames: 25.20%)
- Uniform frame sampling outperformed all tested non-uniform sampling rates
- Training on the full dataset outperformed every reduced-fraction subset they tested — a caution for this project's NCAA D1-only scope, and a reason to treat this project's baseline number as a reproduction under a smaller data/compute budget, not a like-for-like comparison
