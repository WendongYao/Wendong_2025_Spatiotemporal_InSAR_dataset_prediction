# Paper figures

Figures 1--5 can be regenerated directly from the released aggregate tables:

```powershell
python .\paper_figures\gen_fig01_support_workflow.py
python .\paper_figures\gen_fig02_model_architecture.py
python .\paper_figures\gen_fig03_primary_results.py
python .\paper_figures\gen_fig04_ablation_external.py
python .\paper_figures\gen_fig05_operator_support.py
```

Figure 6 additionally requires the E32N34 CSV available from Zenodo:

```powershell
python .\paper_figures\gen_fig06_prediction_maps.py `
  --csv-path C:\path\to\EGMS_L3_E32N34_100km_U_2018_2022_1.csv
```

The seed-42 SPAR prediction artifact and LASSO state are already included under
`results/spar_v2/`.
