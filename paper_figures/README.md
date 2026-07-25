# Paper figures

All six figures can be regenerated from the released aggregate and prediction
artifacts; the source EGMS CSV is not required for Figure 6 in v2.1.0.

```powershell
python .\paper_figures\gen_fig01_support_workflow.py
python .\paper_figures\gen_fig02_model_architecture.py
python .\paper_figures\gen_fig03_primary_results.py
python .\paper_figures\gen_fig04_ablation_external.py
python .\paper_figures\gen_fig05_operator_support.py
python .\paper_figures\gen_fig06_prediction_maps.py
```

Figure 6 uses the released seed-42 SPAR and no-refit LASSO direct-prediction
arrays. `results/spar_v2/manifests/release_evidence_v2_1/fig06_plot_manifest.json`
records the artifact hashes, quantile rule, and display limits.
