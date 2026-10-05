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
| 1 (PILOT) | Swin3D-T (K400) on 150-video pilot, 16 frames, test n=30 | 18.5 | Chance level. NOT comparable to the paper's 28.5% (different backbone; 100 of the 19,578 official training videos, about 0.5%). See below |
| 2 | + Explicit player tracking | — | Pending |
| 3 | + Multimodal fusion (stats + pose) | — | Pending |
| 4 | + Clip-level aggregation | — | Pending |

### Backbone substitution (Phase 1)

The plan named VideoMamba (74M) as the Phase 1 backbone. Its pretrained weights are public on Hugging Face (`OpenGVLab/VideoMamba`), but the model depends on the compiled `mamba_ssm` CUDA kernels. On the local machine (Windows, Python 3.13, RTX 50-series) pip could not obtain a build of `mamba-ssm`, and neither `nvcc` nor the MSVC compiler is installed. Phase 1 therefore uses **torchvision Swin3D-T (27.9M parameters, Kinetics-400 pretrained)** as a stand-in video transformer. Phase 1 numbers are therefore **not a VideoMamba reproduction** and must not be compared to the paper's 28.5% as if they were. Revisit VideoMamba if a CUDA toolchain is set up (or on a Linux cloud/Savio machine).

### Phase 1 pilot run: what it does and does not show

- Setup: Swin3D-T (Kinetics-400 init), 100 train / 20 val / 30 test videos (NCAA D1 2018-19), 16 frames at 224x224, bf16, batch 4, no gradient checkpointing (not needed at this batch size), AdamW + one-cycle schedule, 15 epochs, seed 0. Hyperparameters were fixed before the run; none were tuned on val or test. Best epoch chosen on val, test evaluated once.
- Test accuracy 18.5% (val 24.0% at the selected epoch), against 20.0% for uniform random and 21.3% for always predicting each skill's train-majority level. Val accuracy over all 15 epochs ranged 16.5% to 24.0% with mean 19.2%, i.e. chance. The selected "best" epoch was epoch 1, picked by noise on a 20-video val set.
- Training loss fell from 1.63 to 0.53 while val loss rose from 1.61 to 2.22: the model memorized the 100 training videos and did not generalize. The earlier overfit smoke test (8 videos to 99.4%) shows the pipeline itself can learn.
- Per-category test accuracies (24.0 / 16.7 / 18.3 / 13.3 / 17.3) come from 30 videos per skill and are noise-dominated; they should not be read as a category pattern (for example, val Offense was 30.0% while test Offense was 17.3%).
- Interpretation limits: this run cannot separate "too little data" from "the signal is not learnable by this backbone/setup". The paper trains on the full dataset and reaches 28.5% even at best, so a pilot at chance is not surprising, but also not evidence either way.

## Per-Skill-Category Breakdown

Fill in once Phase 1 baseline is available, to track how each subsequent phase shifts category-level performance relative to the paper's own breakdown (Shooting 26.47%, Rebounding 26.62%, Defense 27.38%, Playmaking 25.39%, Offense 34.60%).

| Phase | Shooting | Rebounding | Defense | Playmaking | Offense |
|---|---|---|---|---|---|
| 1 (PILOT, test n=30) | 24.0 | 16.7 | 18.3 | 13.3 | 17.3 |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
