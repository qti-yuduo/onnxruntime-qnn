# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory=$true)]
    [string]$PythonVersion,
    [Parameter(Mandatory=$true)]
    [string]$WheelArch,
    [Parameter(Mandatory=$true)]
    [string]$WheelDirectory,
    [Parameter(Mandatory=$true)]
    [string]$ExpectedVersion,
    [Parameter(Mandatory=$true)]
    [string]$SamplePath
)

$ErrorActionPreference = 'Stop'

$pyNoDot = $PythonVersion.Replace(".", "")
$envName = "py${pyNoDot}_release_testing_env"

# Find the wheel that matches both the Python version and the platform
$wheel = Get-ChildItem -Path $WheelDirectory `
  -Filter "*cp${pyNoDot}-cp${pyNoDot}-${WheelArch}.whl" |
  Select-Object -First 1

if (-not $wheel) {
    Write-Host "No wheel found matching cp${pyNoDot}-${WheelArch} in $WheelDirectory" -ForegroundColor Red
    exit 1
}
Write-Host "Found wheel: $($wheel.FullName)" -ForegroundColor Cyan

# Create venv named py{XYZ}_release_testing_env using the py launcher
# to pick the requested Python version (no actions/setup-python needed).
$pyTag = if ($WheelArch -eq "win_arm64") { "$PythonVersion-arm64" } else { "$PythonVersion" }
py -$pyTag -m venv $envName

# Activate venv (dot-source so the PATH update persists in this script's scope)
. "$envName/Scripts/Activate.ps1"

# Upgrade pip and install the local wheel
python -m pip install --upgrade pip
python -m pip install $wheel.FullName

# Verify the package reports the expected version
$reported = (python -c "import onnxruntime_qnn as qnn_ep; print(qnn_ep.__version__)").Trim()
Write-Host "onnxruntime_qnn.__version__ = $reported"
if ($reported -ne $ExpectedVersion) {
    Write-Host "Version mismatch: expected '$ExpectedVersion', got '$reported'" -ForegroundColor Red
    exit 1
}
Write-Host "Version check PASS" -ForegroundColor Green

# Run the sample test
python $SamplePath
