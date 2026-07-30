param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [string]$E29N33Csv,
    [Parameter(Mandatory = $true)]
    [string]$E36N31Csv,
    [Parameter(Mandatory = $true)]
    [string]$E37N41Csv,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$cacheDir = Join-Path $ProjectRoot "results\_raw_task_cache"
$resultRoot = Join-Path $ProjectRoot "results\R099_v23_external_locked"
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

$tiles = @(
    [pscustomobject]@{
        Name = "E29N33"
        Csv = $E29N33Csv
    },
    [pscustomobject]@{
        Name = "E36N31"
        Csv = $E36N31Csv
    },
    [pscustomobject]@{
        Name = "E37N41"
        Csv = $E37N41Csv
    }
)

New-Item -ItemType Directory -Path $resultRoot -Force | Out-Null
$queueLog = Join-Path $resultRoot "queue.log"
"R099 queue started $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog
foreach ($tile in $tiles) {
    foreach ($seed in 47..50) {
        $outputRoot = Join-Path $resultRoot "$($tile.Name)\seed_$seed"
        "Launching $($tile.Name) seed $seed at $(Get-Date -Format o)" |
            Tee-Object -FilePath $queueLog -Append
        & $PythonExe (Join-Path $ProjectRoot "experiments_ext\run_v23_native_suite.py") `
            --csv-path $tile.Csv `
            --tile $tile.Name `
            --output-root $outputRoot `
            --cache-dir $cacheDir `
            --seed $seed `
            --models persistence dlinear lasso spar `
            --spar-sampler all_cells_uniform `
            --epochs 60 `
            --patience 12 `
            --batch-size 1024 `
            --dlinear-batch-size 4096 2>&1 |
            Tee-Object -FilePath $queueLog -Append
        if ($LASTEXITCODE -ne 0) {
            throw "R099 $($tile.Name) seed $seed failed with exit code $LASTEXITCODE"
        }
        "Completed $($tile.Name) seed $seed at $(Get-Date -Format o)" |
            Tee-Object -FilePath $queueLog -Append
    }
}
"R099 queue completed $(Get-Date -Format o)" | Tee-Object -FilePath $queueLog -Append
