param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$resultsRoot = Join-Path $ProjectRoot "results\R089_native_baselines_confirmation"
$cacheDir = Join-Path $ProjectRoot "results\_raw_task_cache"

foreach ($seed in 43..46) {
    $outputRoot = Join-Path $resultsRoot "E32N34\seed_$seed"
    & $PythonExe (Join-Path $PSScriptRoot "run_v22_native_baselines.py") `
        --csv-path $CsvPath `
        --tile E32N34 `
        --output-root $outputRoot `
        --cache-dir $cacheDir `
        --seed $seed `
        --models persistence linear_trend dlinear `
        --epochs 60 `
        --patience 12 `
        --batch-size 4096 `
        --moving-average 25 `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Native baseline run failed for seed $seed with exit code $LASTEXITCODE."
    }
}
