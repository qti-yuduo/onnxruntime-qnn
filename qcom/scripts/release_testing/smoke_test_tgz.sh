#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tgz-directory)    tgz_directory="$2";    shift 2 ;;
    --test-packages-zip) test_packages_zip="$2"; shift 2 ;;
    --model-path)       model_path="$2";       shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

: "${tgz_directory:?--tgz-directory is required}"
: "${test_packages_zip:?--test-packages-zip is required}"
: "${model_path:?--model-path is required}"

# --- Extract the linux-aarch64 release tgz ---
release_tgz=$(find "$tgz_directory" -maxdepth 1 -name "*linux-aarch64.tgz" | head -n 1)
if [ -z "$release_tgz" ]; then
    echo ""
    echo "ERROR: No linux-aarch64 tgz found in $tgz_directory"
    exit 1
fi
release_tgz_name=$(basename "$release_tgz")
release_extract_dir="${tgz_directory}/${release_tgz_name%.tgz}_extracted"
echo ""
echo "Extracting release tgz: $release_tgz_name"
mkdir -p "$release_extract_dir"
tar -xf "$release_tgz" -C "$release_extract_dir"

# --- Extract test_packages.zip ---
test_pkgs_dir="$(dirname "$test_packages_zip")/test_packages"
echo "Extracting test_packages.zip"
mkdir -p "$test_pkgs_dir"
python3 -c "import zipfile; zipfile.ZipFile('${test_packages_zip}').extractall('${test_pkgs_dir}')"

# --- Locate the linux-arm64 folder in test_packages ---
test_bin_dir="${test_pkgs_dir}/linux-arm64"
if [ ! -d "$test_bin_dir" ]; then
    echo "ERROR: linux-arm64 folder not found in $test_pkgs_dir"
    echo "Contents of $test_pkgs_dir:"
    ls -1 "$test_pkgs_dir"
    exit 1
fi

# --- Find onnxruntime_perf_test in the test binaries ---
perf_test=$(find "$test_bin_dir" -name "onnxruntime_perf_test" -not -name "*.exe" | head -n 1)
if [ -z "$perf_test" ]; then
    echo "ERROR: onnxruntime_perf_test not found in $test_bin_dir"
    exit 1
fi

# --- Copy onnxruntime_perf_test to the extracted release folder ---
cp "$perf_test" "$release_extract_dir/"
chmod +x "$release_extract_dir/onnxruntime_perf_test"
echo ""
echo "Copied onnxruntime_perf_test to release test directory"

# --- Run the perf test smoke test ---
echo "Running smoke test"
echo ""

cd "$release_extract_dir"
./onnxruntime_perf_test \
    -I \
    --plugin_ep_libs "QNNExecutionProvider|libonnxruntime_providers_qnn.so" \
    --plugin_eps QNNExecutionProvider \
    -m times \
    -r 1 \
    -p burst \
    -i "backend_path|libQnnHtp.so" \
    "$model_path"

echo ""
echo "Smoke test PASSED"
