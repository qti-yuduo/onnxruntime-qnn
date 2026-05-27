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
    [string]$OnnxruntimeVersion,
    [Parameter(Mandatory=$true)]
    [string]$SamplePath
)

$ErrorActionPreference = 'Stop'

$pyNoDot = $PythonVersion.Replace(".", "")
$envName = "py${pyNoDot}_release_backward_compatibility_env"

# Find the wheel that matches both the Python version and the platform
$wheel = Get-ChildItem -Path $WheelDirectory `
  -Filter "*cp${pyNoDot}-cp${pyNoDot}-${WheelArch}.whl" |
  Select-Object -First 1

if (-not $wheel) {
    Write-Host "No wheel found matching cp${pyNoDot}-${WheelArch} in $WheelDirectory" -ForegroundColor Red
    exit 1
}
Write-Host "Found wheel: $($wheel.FullName)" -ForegroundColor Cyan

# Create venv using the py launcher to pick the requested Python version
# (no actions/setup-python needed — runner has Python pre-installed)
py -$PythonVersion -m venv $envName

# Activate venv (dot-source so the PATH update persists in this script's scope)
. "$envName/Scripts/Activate.ps1"

# Upgrade pip
python -m pip install --upgrade pip

# Install onnxruntime-qnn from the local wheel (pulls onnxruntime as a dependency)
python -m pip install $wheel.FullName

# Replace the bundled onnxruntime with the older version to verify backward compatibility
python -m pip uninstall -y onnxruntime
python -m pip install "onnxruntime==$OnnxruntimeVersion"

# Run the sample test against the older onnxruntime
python $SamplePath
