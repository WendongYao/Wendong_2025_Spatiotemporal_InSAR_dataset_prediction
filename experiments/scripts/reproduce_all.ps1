param(
    [string]$CsvPath
)

$ErrorActionPreference = "Stop"
$csvArgs = @()
if ($CsvPath) {
    $csvArgs = @("--csv-path", $CsvPath)
}

python .\smoke_test_revision.py --strict-all
python .\preflight_revision.py @csvArgs
python .\run_cg_additional_suite.py --phase all @csvArgs
python .\run_deep_model_repair.py @csvArgs
python .\run_deep_model_round2.py @csvArgs --models temporal_channel_cnn patch_unet_residual conv_lstm_residual temporal_linear_hybrid --output-root revision_outputs/deep_model_round2
python .\run_nontransformer_round3.py @csvArgs --models cnn_lstm_hybrid cnn_tcn_hybrid --output-root revision_outputs/nontransformer_round3
python .\run_nontransformer_round3.py @csvArgs --models cnn_lstm_hybrid --patch-batch-size 32 --learning-rate 6e-4 --output-root revision_outputs/nontransformer_round3_fast
