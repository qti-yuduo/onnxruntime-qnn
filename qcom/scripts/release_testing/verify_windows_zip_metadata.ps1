# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory=$true)]
    [string]$ArchiveDirectory,
    [Parameter(Mandatory=$true)]
    [string]$ExpectedVersion
)

$ErrorActionPreference = 'Stop'

# Force UTF-8 so any native tool output renders correctly in the GHA log viewer.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# Verify the directory exists
if (-not (Test-Path $ArchiveDirectory -PathType Container)) {
    Write-Host ""
    Write-Error "Directory not found: $ArchiveDirectory"
    exit 1
}

# Expect exactly one win-arm64 zip
$zips = @(Get-ChildItem -Path $ArchiveDirectory -Filter "*.zip" | Where-Object { $_.Name -match "win-arm64\.zip$" })

if ($zips.Count -ne 1) {
    Write-Host ""
    Write-Host "Expected 1 win-arm64 zip in $ArchiveDirectory, found $($zips.Count)" -ForegroundColor Red
    $zips | ForEach-Object { Write-Host "  $($_.Name)" }
    exit 1
}

$zip = $zips[0]
Write-Host ""
Write-Host "Processing: $($zip.Name)" -ForegroundColor Yellow

$certPass    = $false
$versionPass = $false

$extractDir = Join-Path $zip.DirectoryName "$($zip.BaseName)_extracted"

try {
    Expand-Archive -Path $zip.FullName -DestinationPath $extractDir -Force

    $dll = Get-ChildItem -Path $extractDir -Recurse -Filter "onnxruntime_providers_qnn.dll" | Select-Object -First 1

    if ($null -eq $dll) {
        Write-Host "  CERTIFICATE FAIL: onnxruntime_providers_qnn.dll not found in archive" -ForegroundColor Red
        Write-Host "  VERSION FAIL: onnxruntime_providers_qnn.dll not found in archive" -ForegroundColor Red
    } else {
        # Certificate check
        $signature = Get-AuthenticodeSignature -FilePath $dll.FullName
        if ($signature.Status -ne 'Valid') {
            Write-Host "  CERTIFICATE FAIL: Invalid signature ($($signature.Status))" -ForegroundColor Red
        } elseif ($signature.SignerCertificate.Subject -like "*Qualcomm Inc*") {
            Write-Host "  CERTIFICATE PASS" -ForegroundColor Green
            $certPass = $true
        } else {
            Write-Host "  CERTIFICATE FAIL: Not signed by Qualcomm Inc (Subject: $($signature.SignerCertificate.Subject))" -ForegroundColor Red
        }

        # Version check
        $fileVersion = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($dll.FullName).FileVersion
        if ($fileVersion -eq $ExpectedVersion) {
            Write-Host "  VERSION PASS" -ForegroundColor Green
            $versionPass = $true
        } else {
            Write-Host "  VERSION FAIL: Expected $ExpectedVersion, got $fileVersion" -ForegroundColor Red
        }
    }
}
catch {
    Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not $certPass -or -not $versionPass) {
    exit 1
}
