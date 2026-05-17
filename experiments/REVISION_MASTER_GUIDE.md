# Revision Master Guide

## 0. This Guide Is Now a Checklist

This file is the working revision checklist for the current standalone project.

Only the following assets are allowed to count as existing code support:

- `clstm-residual.py`
- `goodmodel.py`
- `las.py`
- `LGBM.py`
- the paper draft `2026_JAG___Wendong__under_submission_.pdf`

Everything below is written under that constraint.

---

## 1. Scope Freeze

Before changing the paper or code, lock in the current scope:

- [ ] Treat the project as a **single-region proof-of-concept**, not a full benchmark paper.
- [ ] Treat `CNN-LSTM` as the main model family.
- [ ] Treat `LASSO` as the strongest directly comparable non-deep baseline.
- [ ] Treat `LightGBM` as either:
  - an auxiliary baseline plus SHAP explainer, or
  - a main baseline only after code alignment with the CNN-LSTM/LASSO task.
- [ ] Do **not** assume access to ConvLSTM, TCN, cross-region, multi-seed framework, masked loss, or unified benchmark code.

Working one-line paper positioning:

> This is a measurement-aware, single-region, dense displacement forecasting proof-of-concept built around CNN-LSTM, with LASSO comparison and LightGBM-based attribution analysis.

---

## 2. Current Code Reality

Use this section as the truth source when editing the paper.

### 2.1 CNN-LSTM Reality

- [ ] `goodmodel.py` and `clstm-residual.py` are not two separate benchmark models.
- [ ] They are two variants of the same `CNN-LSTM` setup.
- [ ] Both use:
  - `EGMS_L3_E32N34_100km_U_2018_2022_1.csv`
  - first `300` timesteps as input
  - column `313` target map
  - `256x256` interpolated dense grid
  - `griddata(..., method='linear')`
  - `np.nan_to_num(...)`
- [ ] Neither script implements a rigorous train/val/test protocol.
- [ ] Neither script implements masked loss.

Recommended rule:

- [ ] Use `clstm-residual.py` as the main CNN-LSTM script.
- [ ] Treat `goodmodel.py` as an older/simpler CNN-LSTM variant or backup.

### 2.2 LASSO Reality

- [ ] `las.py` is aligned with the same `300 -> 313`, `256x256`, `E32N34` task family.
- [ ] It is currently the cleanest directly comparable non-deep baseline.

### 2.3 LightGBM Reality

- [ ] `LGBM.py` is **not** on the same protocol as `clstm-residual.py` and `las.py`.
- [ ] It uses sliding windows with `T_IN = 29`, not the exact `300 -> 313` target setup.
- [ ] It does support:
  - LightGBM training
  - SHAP summary / local explanation
  - error diagnostics
  - full-frame prediction visualization

Decision rule:

- [ ] If the paper keeps LightGBM in the main quantitative comparison, the code must be aligned first.
- [ ] If the code is not aligned, LightGBM must be described as an auxiliary/interpretability baseline.

---

## 3. Paper Revision Checklist

This is the most important section. Work through it top to bottom.

### 3.1 Title and Overall Positioning

- [ ] Keep the paper centered on:
  - measurement-aware dense displacement estimation
  - interpolation bias discussion
  - CNN-LSTM as the main forecasting model
- [ ] Do not oversell the work as a completed unified benchmarking framework.

### 3.2 Abstract

Keep:

- [ ] measurement-aware framing
- [ ] CNN-LSTM improves over simpler baselines
- [ ] interpolation bias matters
- [ ] LightGBM attribution / SHAP can explain temporal dependence

Revise:

- [ ] Remove or soften any claim that all baselines are already under one strict unified protocol.
- [ ] Remove claims that imply masked-domain evaluation unless you really add it.
- [ ] If LightGBM remains in the abstract, make sure it is described consistently with what the code actually does.

Safer wording:

- [ ] Use phrases like:
  - `under the current experimental setup`
  - `in a single-region case study`
  - `as an auxiliary tree-based baseline and attribution model`

### 3.3 Introduction

Keep:

- [ ] problem motivation
- [ ] sparse-to-dense reconstruction framing
- [ ] interpolation and uncertainty/bias motivation
- [ ] CNN-LSTM as the main spatio-temporal learner

Replace the contributions list with something the current code can support:

- [ ] Contribution 1: measurement-aware sparse-to-dense reconstruction framing
- [ ] Contribution 2: CNN-LSTM versus simple tabular baselines in a single-region setup
- [ ] Contribution 3: comparative error diagnostics plus LightGBM SHAP interpretation

Remove from contributions:

- [ ] ConvLSTM
- [ ] TCN
- [ ] multi-seed significance testing as completed work
- [ ] masked loss as completed work
- [ ] cross-region generalization as completed work

### 3.4 Related Work

- [ ] Keep this section and strengthen it.
- [ ] Compare against:
  - InSAR displacement forecasting
  - sparse-to-dense reconstruction
  - CNN/LSTM spatio-temporal forecasting
  - interpretable tabular models such as LightGBM / LASSO
- [ ] Do not promise code-backed deep baselines that you do not have.

### 3.5 Methods

This section needs the largest rewrite.

#### 3.5.1 Data Preprocessing

- [ ] State clearly that the dense maps are built with `linear` interpolation.
- [ ] State clearly that missing interpolated values are currently filled using `np.nan_to_num`.
- [ ] Remove any wording that says missing values are propagated only by masks if that is not true.

#### 3.5.2 Task Definition

- [ ] Rewrite the task definition to match the scripts:
  - single region `E32N34`
  - first 300 timesteps as input
  - final target displacement map from column 313
  - output size `256x256`

- [ ] Do not describe the current experiment as a multi-region, multi-split benchmark.

#### 3.5.3 Models and Baselines

The current code only supports this list safely:

- [ ] `CNN-LSTM` as the main model
- [ ] `LASSO` as the linear baseline
- [ ] `LightGBM` as a tree-based baseline / attribution model

Delete or move to future work:

- [ ] ConvLSTM
- [ ] TCN
- [ ] graph models
- [ ] transformer family

#### 3.5.4 Training Protocol

Keep only what is true now:

- [ ] CNN-LSTM uses standard regression loss
- [ ] LASSO uses L1-regularized regression
- [ ] LightGBM uses its own training/validation routine

Delete or rewrite:

- [ ] `masked MSE`
- [ ] `no spatial leakage / no temporal leakage` unless you add it
- [ ] `mean ± std over 5 seeds` unless you run it
- [ ] `all models are trained under the same split` unless you align LightGBM

#### 3.5.5 Resolution Section

- [ ] Remove the full 128/256/512 resolution experiment section unless you really reproduce it from this project.
- [ ] If you want to keep a note on resolution, make it a short implementation choice statement, not a completed sensitivity study.

#### 3.5.6 Interpretability

- [ ] Keep SHAP for LightGBM.
- [ ] Remove CNN-LSTM saliency / occlusion unless you add code for it.

### 3.6 Results

Use this structure:

- [ ] 4.1 Main quantitative comparison: CNN-LSTM vs LASSO
- [ ] 4.2 Diagnostic comparison: CNN-LSTM, LASSO, and optionally LightGBM
- [ ] 4.3 Spatial map comparison
- [ ] 4.4 LightGBM SHAP attribution
- [ ] 4.5 Limitations of the current setup

Results claims that must be softened:

- [ ] any claim of strict fairness across all models
- [ ] any claim of masked-domain evaluation
- [ ] any claim of completed multi-seed statistics
- [ ] any claim of completed cross-region validation

### 3.7 Conclusions

Keep:

- [ ] CNN-LSTM outperforms simpler baselines under the current setup
- [ ] interpolation choices matter
- [ ] LightGBM SHAP reveals strong recent-lag dependence

Move to future work:

- [ ] stronger deep baselines
- [ ] multiple seeds
- [ ] cross-region validation
- [ ] uncertainty propagation
- [ ] interpolation sensitivity expansion

---

## 4. Hard "Do Not Say This" List

Before submitting anything, search the draft for these claims and fix them if they remain.

- [ ] `ConvLSTM baseline was included`
- [ ] `TCN baseline was included`
- [ ] `all models used the same split`
- [ ] `masked MSE` or `mask-aware loss` was used
- [ ] `missing values were not zero-imputed`
- [ ] `results are averaged over 5 seeds`
- [ ] `cross-region validation was completed`
- [ ] `CNN-LSTM saliency/occlusion analysis was performed`

---

## 5. Code Revision Checklist

This section tells us what to change in code first.

### 5.1 File Priorities

Work in this order:

1. `clstm-residual.py`
2. `las.py`
3. `LGBM.py`
4. `goodmodel.py`

### 5.2 Immediate Code Decisions

- [ ] Choose `clstm-residual.py` as the canonical CNN-LSTM script.
- [ ] Do not keep both CNN-LSTM files as if they are separate models in the paper.
- [ ] Use `goodmodel.py` only as backup, legacy reference, or for cross-checking outputs.

### 5.3 Minimum Code Cleanup to Support the Paper

#### A. CNN-LSTM main script

In `clstm-residual.py`:

- [ ] add a clearer header comment documenting the exact task
- [ ] confirm saved outputs:
  - scatter plot
  - residual plot
  - binned MAE
  - binned residual boxplot
  - predicted spatial map
- [ ] make the printed metrics easy to cite in the paper

#### B. LASSO baseline

In `las.py`:

- [ ] confirm the task definition matches the CNN-LSTM task exactly
- [ ] keep the same diagnostics and spatial output style as the CNN-LSTM script
- [ ] make sure reported metrics are in the same format/order as CNN-LSTM

#### C. LightGBM baseline

In `LGBM.py`, pick one of these two paths:

- [ ] Path A: keep it as an auxiliary baseline and SHAP script
- [ ] Path B: rewrite it so it matches the `300 -> 313` task

Until Path B is done:

- [ ] do not present LightGBM as a fully aligned main benchmark result

#### D. Optional utility script

If we want one extra code file with high value, add:

- [ ] `compare_main_models.py`

Suggested purpose:

- load the outputs or metrics from `clstm-residual.py`, `las.py`, and optionally aligned `LGBM.py`
- print a clean comparison table
- save a single paper-ready metrics summary

---

## 6. New Code We Are Actually Allowed to Add

Because the project is intentionally small, do not explode it into a large framework.

Only add code that directly supports the current paper.

Recommended additions:

- [ ] `compare_main_models.py`
- [ ] `lightgbm_aligned.py` only if LightGBM must become protocol-matched
- [ ] `interpolation_sensitivity.py` only if we decide to keep that result in the paper

Do not add yet:

- [ ] TCN training code
- [ ] ConvLSTM training code
- [ ] cross-region evaluation code
- [ ] multi-seed infrastructure package

unless the paper strategy changes again.

---

## 7. Run Order Checklist

Once the paper text is corrected, run experiments in this order.

### Option 1: Minimal safe revision

- [ ] Run `clstm-residual.py`
- [ ] Run `las.py`
- [ ] Run `LGBM.py`
- [ ] Collect:
  - quantitative metrics
  - diagnostic plots
  - spatial comparison figures
  - SHAP figure

### Option 2: Slightly stronger revision

- [ ] First align LightGBM with the CNN-LSTM/LASSO task
- [ ] Then rerun:
  - `clstm-residual.py`
  - `las.py`
  - aligned LightGBM script
- [ ] Produce one clean comparison table

### Optional extra experiment

- [ ] Run a small interpolation sensitivity test:
  - `linear`
  - `nearest`
  - `cubic` if stable

Only keep this in the paper if the outputs are clean and interpretable.

---

## 8. Submission-Ready End State

The revision is in a safe state only when all boxes below are true:

- [ ] The paper no longer claims unsupported baselines.
- [ ] The paper no longer claims masked-domain evaluation unless added.
- [ ] The paper no longer claims multi-seed statistics unless run.
- [ ] The paper clearly distinguishes between:
  - the main CNN-LSTM result
  - the LASSO comparison
  - the LightGBM explanatory role
- [ ] The main figures can be regenerated from the files in this project.
- [ ] Every number quoted in the Results section can be traced back to one of these scripts.

---

## 9. Final Working Strategy

The correct strategy for this project is:

- [ ] first shrink the paper to match the real code
- [ ] then minimally improve the code only where that strengthens the paper
- [ ] avoid rebuilding the paper into a large benchmark study unless more code is actually added

One-sentence reminder:

> We are revising this paper as a tightly scoped, code-supported CNN-LSTM proof-of-concept study, not as a completed multi-model benchmark suite.
