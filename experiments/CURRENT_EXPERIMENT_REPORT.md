# Current Experiment Report

Date: 2026-05-15

## Scope

This report summarizes the current experiment status for the standalone project in `found_training_project`.
All conclusions below are based on the revision-aligned, single-region, single-task setup that has been implemented and actually run in this project.

Primary result sources:

- `revision_outputs/cg_suite/E2_primary_multiseed/spatial_tile/grid_256/primary_multiseed_summary.csv`
- `revision_outputs/deep_model_repair/primary_multiseed/deep_repair_summary.csv`
- `revision_outputs/deep_model_round2_multiseed/round2_summary.csv`
- `revision_outputs/deep_model_round2_hybrid_v2_multiseed/round2_summary.csv`
- `revision_outputs/deep_model_round2_convlstm_multiseed/round2_summary.csv`
- `revision_outputs/nontransformer_round3_cnntcn_multiseed/round3_summary.csv`
- `revision_outputs/nontransformer_round3_cnnlstm_l1_5seed/combined_summary.csv`

## Main Result Snapshot

Current best-performing models under the aligned `grid_256 + spatial_tile` setting:

| Rank | Model | RMSE mean | RMSE std | Notes |
|---|---|---:|---:|---|
| 1 | cnn_lstm_hybrid (1-layer) | 1.3077 | 0.0927 | current best 5-seed non-Transformer result |
| 2 | temporal_linear_hybrid v2 | 1.3602 | 0.0917 | strongest previous hybrid benchmark |
| 3 | cnn_tcn_hybrid | 1.3782 | 0.0685 | best 5-seed TCN-style result |
| 4 | lasso | 1.3836 | 0.0603 | strongest classical baseline |
| 5 | patch_unet_residual | 1.4165 | 0.0825 | strongest non-hybrid deep model from round 2 |
| 6 | temporal_channel_cnn | 1.4503 | 0.0830 | stable but weaker than U-Net |
| 7 | conv_lstm_residual | 1.5067 | 0.0936 | much slower, not competitive |
| 8 | random_forest | 1.5229 | 0.1421 | useful baseline, slower than lasso |
| 9 | persistence | 1.5589 | 0.0954 | strong naive baseline |
| 10 | lightgbm | 2.0629 | 0.4739 | under current aligned setup, weaker than expected |

Exploratory but not fully matched in seed count:

- `cnn_lstm_hybrid` with 2 ConvLSTM layers reached RMSE `1.3049 +- 0.0714` on 3 seeds, but has not yet been extended to a full 5-seed run.

Practical training-efficiency note:

- The original 5-seed `cnn_lstm_hybrid (1-layer)` setting at `batch_size=16, lr=3e-4` reached RMSE `1.3077` with runtime about `529.9 s/seed`.
- A faster setting at `batch_size=32, lr=6e-4` reached RMSE `1.3140` with runtime about `309.9 s/seed`.
- This means we now have a fast configuration that is about `1.71x` faster with only a very small mean-RMSE tradeoff.

Important summary files:

- `revision_outputs/nontransformer_round3_cnnlstm_l1_5seed/combined_summary.csv`
- `revision_outputs/nontransformer_round3_cnntcn_multiseed/round3_summary.csv`
- `revision_outputs/deep_model_round2_hybrid_v2_multiseed/round2_summary.csv`
- `revision_outputs/deep_model_round2_multiseed/round2_summary.csv`
- `revision_outputs/deep_model_round2_convlstm_multiseed/round2_summary.csv`
- `revision_outputs/cg_suite/E2_primary_multiseed/spatial_tile/grid_256/primary_multiseed_summary.csv`

## Deep Model Evolution

### Stage 0: original aligned deep models were effectively broken

The earliest aligned `cnn_lstm_maskaware` and `cnn_tcn` runs were clearly bad:

- `cnn_lstm_maskaware`: RMSE `7.3435`
- `cnn_tcn`: RMSE `7.3370`

Interpretation:

- The old deep-learning pipeline was not a trustworthy representation of deep model capability.
- The main problem was not just hyperparameters. The training sample construction itself was wrong for this task.

Source:

- `revision_outputs/cg_suite/E2_primary_multiseed/spatial_tile/grid_256/primary_multiseed_summary.csv`

### Stage 1: patch-residual repair fixed the major implementation issue

After rewriting the deep pipeline as patch-based residual prediction:

- `cnn_lstm_maskaware`: RMSE `1.5591`
- `cnn_tcn`: RMSE `1.5593`

Interpretation:

- This confirmed that the earlier very poor deep-learning results were largely caused by implementation and data-construction issues.
- After repair, deep models recovered to roughly persistence-level performance.

Source:

- `revision_outputs/deep_model_repair/primary_multiseed/deep_repair_summary.csv`

### Stage 2: round-2 architecture search improved deep models further

Round-2 deep variants:

- `patch_unet_residual`: RMSE `1.4165`
- `temporal_channel_cnn`: RMSE `1.4503`
- `temporal_linear_hybrid` v1: RMSE `1.4361`

Interpretation:

- Structural changes beyond simple recurrent patch models were worthwhile.
- `patch_unet_residual` became the strongest non-hybrid deep model in this phase.

Source:

- `revision_outputs/deep_model_round2_multiseed/round2_summary.csv`

### Stage 3: hybrid v2 surpassed lasso

After upgrading `temporal_linear_hybrid` with:

- lasso-style warm-started linear head
- learnable residual refinement
- recent-lag gating with recency-biased initialization

the result improved to:

- `temporal_linear_hybrid v2`: RMSE `1.3602 +- 0.0917`

Interpretation:

- This was the first deep model in this project to outperform the current `lasso` baseline on 5-seed mean RMSE.
- It established that warm-started linear shortcuts plus recent-lag gating are a strong design direction.

Source:

- `revision_outputs/deep_model_round2_hybrid_v2_multiseed/round2_summary.csv`

### Stage 4: non-Transformer round-3 pushed CNNLSTM and TCN further

Targeted follow-up experiments were run specifically on the `CNNLSTM / TCN` route using:

- lasso-warm-started linear shortcut
- recent-lag gating
- spatially preserved temporal aggregation

Results:

- `cnn_lstm_hybrid` with 2 ConvLSTM layers: RMSE `1.3049 +- 0.0714` on 3 seeds
- `cnn_lstm_hybrid` with 1 ConvLSTM layer: RMSE `1.3077 +- 0.0927` on 5 seeds
- `cnn_tcn_hybrid`: RMSE `1.3782 +- 0.0685` on 5 seeds

Interpretation:

- The non-Transformer route remains highly competitive.
- The 5-seed `cnn_lstm_hybrid` result is now the strongest fully repeated result in the project.
- `cnn_tcn_hybrid` also edges past `lasso`, while remaining much cheaper than the recurrent hybrid.
- The 1-layer `cnn_lstm_hybrid` is a strong speed-quality compromise, while the 2-layer version is slightly stronger on the current 3-seed evidence but much slower.

Sources:

- `revision_outputs/nontransformer_round3_cnnlstm_l1_5seed/combined_summary.csv`
- `revision_outputs/nontransformer_round3_cnntcn_multiseed/round3_summary.csv`
- `revision_outputs/nontransformer_round3_cnnlstm_3seed/round3_summary.csv`
- `revision_outputs/nontransformer_round3_cnnlstm_l1_bs32_lr6e4_5seed/round3_summary.csv`

### ConvLSTM follow-up

`ConvLSTM` has now been added and tested fairly under the same patch-residual protocol:

- `conv_lstm_residual`: RMSE `1.5067 +- 0.0936`
- runtime: about `463.1 s/seed`
- peak GPU memory: about `5739 MB`

Interpretation:

- `ConvLSTM` is now covered experimentally.
- Under the current task definition, it is much slower than the better models and does not beat `lasso` or `temporal_linear_hybrid v2`.

Source:

- `revision_outputs/deep_model_round2_convlstm_multiseed/round2_summary.csv`

## Additional Experiment Summary

### E2: primary multi-seed comparison

Key conclusion:

- In the original pre-repair comparison, `lasso` was the strongest baseline and the old deep models were clearly underperforming.
- This table is still important historically, but it should not be used to represent the final deep-learning state after repair.

Source:

- `revision_outputs/cg_suite/E2_primary_multiseed/spatial_tile/grid_256/primary_multiseed_summary.csv`

### E3: mask ablation

Key findings:

- `no input mask` was almost identical to `mask-aware`
- `no loss mask` looked numerically better, but it is not a fair main result

Interpretation:

- The input mask channel contributes little under the old CNN-LSTM setup.
- Loss masking matters methodologically, even when turning it off can produce more optimistic numbers.

Source:

- `revision_outputs/cg_suite/E3_mask_ablation/spatial_tile/grid_256/mask_ablation_summary.csv`

### E4: interpolation sensitivity

Forecast-table takeaway:

- `idw` was the strongest interpolation choice overall
- `nearest` was the weakest
- `rbf` was often the second-best option

Point-holdout interpolation takeaway:

- `idw` had the lowest holdout interpolation RMSE
- `linear` was weaker than `idw`
- `nearest` was clearly worst

This is especially important because interpolation choice materially changes downstream forecasting difficulty.

Sources:

- `revision_outputs/cg_suite/E4_interpolation_sensitivity/spatial_tile/grid_256/forecast_metric_summary.csv`
- `revision_outputs/cg_suite/E4_interpolation_sensitivity/spatial_tile/grid_256/point_holdout_interpolation_summary.csv`

### E5: split comparison

Key findings:

- random-pixel split is optimistic compared with spatial-tile split
- `lightgbm` optimism inflation was about `19.6%`
- old `cnn_lstm_maskaware` and `cnn_tcn` also showed about `7.4%` optimism inflation

Interpretation:

- Spatial split is the correct main evaluation protocol for this project.

Source:

- `revision_outputs/cg_suite/E5_split_comparison/grid_256/split_comparison_summary.csv`

### E7: resolution scaling

Key findings:

- `lasso` remained strong and stable at `128`, `256`, and `512`
- `lightgbm` remained weaker than `lasso`
- the old `cnn_lstm_maskaware` did not provide a valid `512` result

Interpretation:

- Resolution scaling does not rescue the old deep model family.
- Classical linear temporal extrapolation remains very strong across scales.

Source:

- `revision_outputs/cg_suite/E7_resolution_scaling/resolution_scaling_summary.csv`

### E10: interpretability and persistence similarity

Key findings:

- `lightgbm` was very similar to persistence, correlation about `0.968`
- the old bad `cnn` models were essentially uncorrelated with persistence

Interpretation:

- The old poor CNN results were not just slightly noisy versions of persistence. They were structurally wrong.
- The stronger non-deep models largely behave like high-quality persistence refinements.

Source:

- `revision_outputs/cg_suite/E10_interpretability/spatial_tile/seed_42/persistence_similarity.csv`

## Statistical Note

The old paired comparison strongly showed that the original proposed deep model was worse than the strongest baseline:

- paired t-test p-value: `2.4028e-05`
- strongest baseline in that table: `lasso`

This remains useful as evidence that the original deep-learning implementation was not publishable in its initial form.

Source:

- `revision_outputs/cg_suite/E2_primary_multiseed/spatial_tile/grid_256/paired_model_stats.json`

## Current Project-Level Conclusions

1. The original aligned deep-learning results were not trustworthy because of implementation and data-construction issues.
2. The deep pipeline repair was successful and materially changed the scientific conclusion.
3. The strongest fully repeated result in the project is now `cnn_lstm_hybrid` with 1 ConvLSTM layer.
4. `temporal_linear_hybrid v2`, `cnn_tcn_hybrid`, and `lasso` now form a very tight top tier, with all three clearly stronger than the older repaired baselines.
5. `ConvLSTM` has now been fairly tested, but the plain `conv_lstm_residual` route is not competitive enough to be a lead model.
6. Interpolation choice matters a great deal, and `idw` currently looks like the strongest candidate for a next-round main setting.
7. For day-to-day experimentation, `cnn_lstm_hybrid (1-layer)` now has both a best-quality config and a faster near-lossless config.

## Recommended Next Priorities

1. Re-run the strongest current models under `idw`:
   - `cnn_lstm_hybrid` (1-layer)
   - `cnn_tcn_hybrid`
   - `temporal_linear_hybrid v2`
   - `lasso`
2. Produce a single unified main comparison table that includes:
   - `cnn_lstm_hybrid`
   - `cnn_tcn_hybrid`
   - `temporal_linear_hybrid v2`
   - `lasso`
   - `patch_unet_residual`
3. Decide whether the paper's main deep result should now center on:
   - `cnn_lstm_hybrid` as the strongest current deep result
   - `lasso` as the strongest classical baseline
4. If further deep improvement is needed, continue from the hybrid non-Transformer family rather than from the plain `ConvLSTM` route.

## Practical Reading Order

If someone wants the quickest possible overview, read these files in this order:

1. `revision_outputs/deep_model_round2_hybrid_v2_multiseed/round2_summary.csv`
2. `revision_outputs/cg_suite/E2_primary_multiseed/spatial_tile/grid_256/primary_multiseed_summary.csv`
3. `revision_outputs/deep_model_repair/primary_multiseed/deep_repair_summary.csv`
4. `revision_outputs/deep_model_round2_multiseed/round2_summary.csv`
5. `revision_outputs/deep_model_round2_convlstm_multiseed/round2_summary.csv`
6. `revision_outputs/cg_suite/E4_interpolation_sensitivity/spatial_tile/grid_256/forecast_metric_summary.csv`
