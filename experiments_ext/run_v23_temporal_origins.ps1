param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$cacheDir = Join-Path $ProjectRoot "results\_raw_task_cache"
$resultRoot = Join-Path $ProjectRoot "results\R096_v23_temporal_origins\E32N34"
$lockPath = Join-Path $ProjectRoot "results\R094_v23_sampler_ablation\E32N34\seed_42\LOCKED_CONFIG.json"
$expectedLockHash = "b51b6441b77254d6a3359f446dbf58228b3c24deb6dae79e1a142d2004ce9c9a"
$actualLockHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $lockPath).Hash.ToLowerInvariant()
if ($actualLockHash -ne $expectedLockHash) {
    throw "Locked configuration hash changed: $actualLockHash"
}
$lock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $lock.code_sha256.PSObject.Properties) {
    $path = Join-Path $ProjectRoot ($entry.Name -replace '/', '\')
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($actual -ne [string]$entry.Value) {
        throw "Frozen code hash changed for $($entry.Name): $actual"
    }
}
New-Item -ItemType Directory -Path $resultRoot -Force | Out-Null
$queueLog = Join-Path $resultRoot "queue.log"
"R096 queue started $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog
foreach ($targetColumn in 252, 272, 292, 312) {
    $historyStartColumn = $targetColumn - 241
    $outputRoot = Join-Path $resultRoot "target_col_$targetColumn"
    "Launching target column $targetColumn (history start $historyStartColumn) at $(Get-Date -Format o)" |
        Tee-Object -FilePath $queueLog -Append
    & $PythonExe (Join-Path $ProjectRoot "experiments_ext\run_v23_native_suite.py") `
        --csv-path $CsvPath `
        --tile E32N34 `
        --output-root $outputRoot `
        --cache-dir $cacheDir `
        --seed 47 `
        --history-start-col $historyStartColumn `
        --history-length 240 `
        --target-col $targetColumn `
        --models persistence dlinear lasso tcn spar `
        --spar-sampler all_cells_uniform `
        --epochs 60 `
        --patience 12 `
        --batch-size 1024 `
        --dlinear-batch-size 4096 2>&1 |
        Tee-Object -FilePath $queueLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "R096 target column $targetColumn failed with exit code $LASTEXITCODE"
    }
    "Completed target column $targetColumn at $(Get-Date -Format o)" |
        Tee-Object -FilePath $queueLog -Append
}
"R096 queue completed $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog -Append
