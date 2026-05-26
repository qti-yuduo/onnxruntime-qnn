# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory=$true)]
    [string]$WheelDirectory,
    [Parameter(Mandatory=$true)]
    [string]$ExpectedVersion
)

function Test-QnnDll {
    param(
        [Parameter(Mandatory=$true)] [System.IO.FileInfo]$Dll,
        [Parameter(Mandatory=$true)] [string]$Label,
        [Parameter(Mandatory=$true)] [string]$ExpectedVersion
    )

    $result = [pscustomobject]@{ CertPass = $false; VersionPass = $false }

    $signature = Get-AuthenticodeSignature -FilePath $Dll.FullName

    if ($signature.Status -ne 'Valid') {
        Write-Host "  [$Label] CERTIFICATE FAIL: Invalid signature ($($signature.Status))" -ForegroundColor Red
    } elseif ($signature.SignerCertificate.Subject -like "*Qualcomm Inc*") {
        Write-Host "  [$Label] CERTIFICATE PASS" -ForegroundColor Green
        $result.CertPass = $true
    } else {
        Write-Host "  [$Label] CERTIFICATE FAIL: Not signed by Qualcomm Inc (Subject: $($signature.SignerCertificate.Subject))" -ForegroundColor Red
    }

    $fileVersion = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($Dll.FullName).FileVersion

    if ($fileVersion -eq $ExpectedVersion) {
        Write-Host "  [$Label] VERSION PASS" -ForegroundColor Green
        $result.VersionPass = $true
    } else {
        Write-Host "  [$Label] VERSION FAIL: Expected $ExpectedVersion, got $fileVersion" -ForegroundColor Red
    }

    return $result
}

# Verify the directory exists
if (-not (Test-Path $WheelDirectory -PathType Container)) {
    Write-Host ""
    Write-Error "Directory not found: $WheelDirectory"
    exit 1
}

$arm64Wheels = @(Get-ChildItem -Path $WheelDirectory -Recurse -Filter "*.whl" | Where-Object { $_.Name -match "win_arm64\.whl$" })
$amdWheels   = @(Get-ChildItem -Path $WheelDirectory -Recurse -Filter "*.whl" | Where-Object { $_.Name -match "win_amd64\.whl$" })

if (($arm64Wheels.Count + $amdWheels.Count) -eq 0) {
    Write-Host ""
    Write-Error "No wheels found matching win_amd64.whl or win_arm64.whl"
    exit 1
}

Write-Host ""
Write-Host "Found $($arm64Wheels.Count) ARM64 wheel(s) and $($amdWheels.Count) AMD wheel(s)" -ForegroundColor Cyan

$certPassCount    = 0
$certFailCount    = 0
$versionPassCount = 0
$versionFailCount = 0

foreach ($wheel in $arm64Wheels) {
    Write-Host ""
    Write-Host "Processing: $($wheel.Name)" -ForegroundColor Yellow

    $wheelDir   = $wheel.DirectoryName
    $extractDir = Join-Path $wheelDir "$($wheel.BaseName)_extracted"
    $zipPath    = Join-Path $wheelDir "$($wheel.BaseName).zip"

    try {
        Copy-Item -Path $wheel.FullName -Destination $zipPath -Force
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

        $dll = Get-ChildItem -Path $extractDir -Recurse -Filter "onnxruntime_providers_qnn.dll" | Select-Object -First 1

        if ($null -eq $dll) {
            Write-Host "  [arm64] CERTIFICATE FAIL: DLL not found" -ForegroundColor Red
            Write-Host "  [arm64] VERSION FAIL: DLL not found" -ForegroundColor Red
            $certFailCount++
            $versionFailCount++
            continue
        }

        $r = Test-QnnDll -Dll $dll -Label "arm64" -ExpectedVersion $ExpectedVersion
        if ($r.CertPass)    { $certPassCount++ }    else { $certFailCount++ }
        if ($r.VersionPass) { $versionPassCount++ } else { $versionFailCount++ }
    }
    catch {
        Write-Host "  [arm64] ERROR: $($_.Exception.Message)" -ForegroundColor Red
        $certFailCount++
        $versionFailCount++
    }
    finally {
        Remove-Item -Path $zipPath    -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

foreach ($wheel in $amdWheels) {
    Write-Host ""
    Write-Host "Processing: $($wheel.Name)" -ForegroundColor Yellow

    $wheelDir   = $wheel.DirectoryName
    $extractDir = Join-Path $wheelDir "$($wheel.BaseName)_extracted"
    $zipPath    = Join-Path $wheelDir "$($wheel.BaseName).zip"

    # Per-DLL outcomes; the wheel-level verdict is the AND of both
    $amd64CertPass      = $false
    $amd64VersionPass   = $false
    $arm64ecCertPass    = $false
    $arm64ecVersionPass = $false

    try {
        Copy-Item -Path $wheel.FullName -Destination $zipPath -Force
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

        $allDlls = Get-ChildItem -Path $extractDir -Recurse -Filter "onnxruntime_providers_qnn.dll"

        $amd64Dll   = $allDlls | Where-Object { $_.FullName -match "[\\/]libs[\\/]amd64[\\/]"   } | Select-Object -First 1
        $arm64ecDll = $allDlls | Where-Object { $_.FullName -match "[\\/]libs[\\/]arm64ec[\\/]" } | Select-Object -First 1

        # libs/amd64/onnxruntime_providers_qnn.dll
        if ($null -eq $amd64Dll) {
            Write-Host "  [amd64]   CERTIFICATE FAIL: DLL not found at libs/amd64/" -ForegroundColor Red
            Write-Host "  [amd64]   VERSION FAIL: DLL not found at libs/amd64/" -ForegroundColor Red
        } else {
            $r = Test-QnnDll -Dll $amd64Dll -Label "amd64  " -ExpectedVersion $ExpectedVersion
            $amd64CertPass    = $r.CertPass
            $amd64VersionPass = $r.VersionPass
        }

        # libs/arm64ec/onnxruntime_providers_qnn.dll
        if ($null -eq $arm64ecDll) {
            Write-Host "  [arm64ec] CERTIFICATE FAIL: DLL not found at libs/arm64ec/" -ForegroundColor Red
            Write-Host "  [arm64ec] VERSION FAIL: DLL not found at libs/arm64ec/" -ForegroundColor Red
        } else {
            $r = Test-QnnDll -Dll $arm64ecDll -Label "arm64ec" -ExpectedVersion $ExpectedVersion
            $arm64ecCertPass    = $r.CertPass
            $arm64ecVersionPass = $r.VersionPass
        }
    }
    catch {
        Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
        # On hard error, every per-DLL flag remains $false -> wheel-level FAIL
    }
    finally {
        Remove-Item -Path $zipPath    -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Wheel-level verdict: both DLLs must pass for the wheel to pass
    if ($amd64CertPass -and $arm64ecCertPass) {
        $certPassCount++
    } else {
        $certFailCount++
    }

    if ($amd64VersionPass -and $arm64ecVersionPass) {
        $versionPassCount++
    } else {
        $versionFailCount++
    }
}

Write-Host ""
Write-Host "=== Certificate Summary ===" -ForegroundColor Cyan
Write-Host "Total:  $($certPassCount + $certFailCount)"
Write-Host "Passed: $certPassCount" -ForegroundColor Green
Write-Host "Failed: $certFailCount" -ForegroundColor Red
Write-Host "=== Version Summary ===" -ForegroundColor Cyan
Write-Host "Total:  $($versionPassCount + $versionFailCount)"
Write-Host "Passed: $versionPassCount" -ForegroundColor Green
Write-Host "Failed: $versionFailCount" -ForegroundColor Red
Write-Host "=== End of Summary ===" -ForegroundColor Cyan

if ($certFailCount -gt 0 -or $versionFailCount -gt 0) {
    exit 1
}
