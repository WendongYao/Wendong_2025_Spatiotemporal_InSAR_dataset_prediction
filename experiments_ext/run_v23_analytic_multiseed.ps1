param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$resultRoot = Join-Path $ProjectRoot "results\R097_v23_analytic_multiseed"
New-Item -ItemType Directory -Path $resultRoot -Force | Out-Null
$queueLog = Join-Path $resultRoot "queue.log"
"R097 queue started $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog
foreach ($seed in 42..51) {
    $outputRoot = Join-Path $resultRoot "seed_$seed"
    "Launching seed $seed at $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog -Append
    & $PythonExe (Join-Path $ProjectRoot "experiments_ext\run_interpolated_target_confound.py") `
        --output-root $outputRoot `
        --scenario composite `
        --grid-size 64 `
        --support-points 1024 `
        --noise 0.35 `
        --seed $seed `
        --input-interpolation idw `
        --target-interpolations idw `
        --models cnn_lstm_hybrid `
        --epochs 40 `
        --patience 8 2>&1 |
        Tee-Object -FilePath $queueLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "R097 seed $seed failed with exit code $LASTEXITCODE"
    }
    "Completed seed $seed at $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog -Append
}
"R097 queue completed $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog -Append
