# Paper figures

All manuscript and supplementary figures can be regenerated from released
aggregate and prediction artifacts; no source EGMS CSV is required.

```powershell
python .\paper_figures\gen_fig01_support_workflow.py
python .\paper_figures\gen_fig02_model_architecture.py
python .\paper_figures\gen_fig03_primary_results.py
python .\paper_figures\gen_fig04_ablation_external.py
python .\paper_figures\gen_fig05_operator_support.py
python .\paper_figures\gen_fig05_temporal_analytic.py
python .\paper_figures\gen_fig06_boundaries.py
python .\paper_figures\gen_fig06_prediction_maps.py
python .\paper_figures\gen_fig07_support_quality.py
python .\paper_figures\gen_fig08_support_diagnostics.py
```

The spatial map uses the released seed-42 SPAR and no-refit LASSO direct-prediction
arrays. `results/spar_v2/manifests/release_evidence_v2_1/fig06_plot_manifest.json`
records the artifact hashes, quantile rule, and display limits.

The v2.3 primary, regional, temporal, sampler, anchor, and analytic figures read
the released v2.3 aggregate CSVs. The multi-resolution panel is explicitly
labelled as a development diagnostic using the superseded capped sampler.
