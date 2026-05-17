#!/usr/bin/env bash
set -euo pipefail

CSV_ARGS=()
if [ "$#" -gt 0 ]; then
  CSV_ARGS=(--csv-path "$1")
fi

python ./smoke_test_revision.py --strict-all
python ./preflight_revision.py "${CSV_ARGS[@]}"
python ./run_cg_additional_suite.py --phase all "${CSV_ARGS[@]}"
python ./run_deep_model_repair.py "${CSV_ARGS[@]}"
python ./run_deep_model_round2.py "${CSV_ARGS[@]}" --models temporal_channel_cnn patch_unet_residual conv_lstm_residual temporal_linear_hybrid --output-root revision_outputs/deep_model_round2
python ./run_nontransformer_round3.py "${CSV_ARGS[@]}" --models cnn_lstm_hybrid cnn_tcn_hybrid --output-root revision_outputs/nontransformer_round3
python ./run_nontransformer_round3.py "${CSV_ARGS[@]}" --models cnn_lstm_hybrid --patch-batch-size 32 --learning-rate 6e-4 --output-root revision_outputs/nontransformer_round3_fast
