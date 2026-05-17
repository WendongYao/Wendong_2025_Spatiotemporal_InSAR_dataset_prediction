# First-run environment bootstrap for the standalone revision project.
#
# Revision skeleton alignment:
# - Section 3.2 / reproducible data-processing environment
# - Section 3.3 / executable model baselines in a dedicated local environment
# - Section 3.11 / optional SHAP dependency kept separate from main first-round runs

param(
    [string]$PythonCmd = "python",
    [ValidateSet("auto", "cpu", "cu126", "cu128")]
    [string]$TorchComputePlatform = "auto",
    [switch]$InstallOptionalShap,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$tempRoot = Join-Path $projectRoot ".tmp"
$coreRequirements = Join-Path $projectRoot "requirements-revision.txt"
$optionalRequirements = Join-Path $projectRoot "requirements-revision-optional.txt"

function Resolve-TorchIndexUrl {
    param([string]$Platform)

    if ($Platform -eq "auto") {
        $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
        if ($null -ne $nvidiaSmi) {
            return "https://download.pytorch.org/whl/cu126"
        }
        return "https://download.pytorch.org/whl/cpu"
    }

    if ($Platform -eq "cpu") {
        return "https://download.pytorch.org/whl/cpu"
    }

    return "https://download.pytorch.org/whl/$Platform"
}

function Use-ProjectTemp {
    if (-not (Test-Path $tempRoot)) {
        New-Item -ItemType Directory -Path $tempRoot | Out-Null
    }
    $env:TEMP = $tempRoot
    $env:TMP = $tempRoot
}

function Assert-Success {
    param([string]$Message)
    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

Write-Host "Project root: $projectRoot"
Use-ProjectTemp

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating local virtual environment..."
    & $PythonCmd -m venv $venvPath
    Assert-Success "Failed to create the local virtual environment."
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found: $venvPython"
}

Write-Host "Checking pip inside the local virtual environment..."
& $venvPython -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip is missing; repairing the local virtual environment with ensurepip..."
    & $venvPython -m ensurepip --upgrade
    Assert-Success "Failed to repair pip inside the local virtual environment."
}

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip
Assert-Success "Failed to upgrade pip in the local virtual environment."

Write-Host "Installing first-round core requirements..."
& $venvPython -m pip install -r $coreRequirements
Assert-Success "Failed to install first-round core requirements."

$torchIndexUrl = Resolve-TorchIndexUrl -Platform $TorchComputePlatform
Write-Host "Installing PyTorch from $torchIndexUrl ..."
& $venvPython -m pip install --upgrade torch --index-url $torchIndexUrl
Assert-Success "Failed to install PyTorch for the selected compute platform."

if ($InstallOptionalShap) {
    Write-Host "Installing optional SHAP dependency..."
    & $venvPython -m pip install -r $optionalRequirements
    Assert-Success "Failed to install optional SHAP requirements."
}

if (-not $SkipSmokeTest) {
    Write-Host "Running strict core smoke test..."
    & $venvPython (Join-Path $projectRoot "smoke_test_revision.py") --strict-core
    Assert-Success "Core smoke test failed after environment bootstrap."
}

Write-Host "Environment bootstrap complete."
Write-Host "Interpreter: $venvPython"
