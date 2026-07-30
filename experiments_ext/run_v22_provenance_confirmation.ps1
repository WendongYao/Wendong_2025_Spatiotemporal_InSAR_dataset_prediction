param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$resultsRoot = Join-Path $ProjectRoot "results\R090_provenance_confirmation"
$cacheDir = Join-Path $ProjectRoot "results\_raw_task_cache"

foreach ($seed in 42..46) {
    $outputRoot = Join-Path $resultsRoot "E32N34\seed_$seed"
    & $PythonExe (Join-Path $PSScriptRoot "run_raw_holdout_pilot.py") `
        --csv-path $CsvPath `
        --tile E32N34 `
        --output-root $outputRoot `
        --cache-dir $cacheDir `
        --seed $seed `
        --grid-size 256 `
        --block-side 8 `
        --epochs 60 `
        --patience 12 `
        --batch-size 16 `
        --models lasso_raw_supervised saqr_point_query `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Provenance confirmation failed for seed $seed with exit code $LASTEXITCODE."
    }
}
