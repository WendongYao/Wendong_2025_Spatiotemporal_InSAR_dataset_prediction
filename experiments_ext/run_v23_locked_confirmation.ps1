param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$cacheDir = Join-Path $ProjectRoot "results\_raw_task_cache"
$resultRoot = Join-Path $ProjectRoot "results\R095_v23_locked_confirmation\E32N34"
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
"R095 queue started $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog
foreach ($seed in 47, 48, 49, 50) {
    $outputRoot = Join-Path $resultRoot "seed_$seed"
    "Launching seed $seed at $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog -Append
    & $PythonExe (Join-Path $ProjectRoot "experiments_ext\run_v23_native_suite.py") `
        --csv-path $CsvPath `
        --tile E32N34 `
        --output-root $outputRoot `
        --cache-dir $cacheDir `
        --seed $seed `
        --models persistence dlinear lasso tcn spar `
        --spar-sampler all_cells_uniform `
        --epochs 60 `
        --patience 12 `
        --batch-size 1024 `
        --dlinear-batch-size 4096 2>&1 |
        Tee-Object -FilePath $queueLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "R095 seed $seed failed with exit code $LASTEXITCODE"
    }
    "Completed seed $seed at $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog -Append
}
"R095 queue completed $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog -Append
