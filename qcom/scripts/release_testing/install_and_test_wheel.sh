#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-version)    python_version="$2";    shift 2 ;;
    --wheel-arch)        wheel_arch="$2";        shift 2 ;;
    --wheel-directory)   wheel_directory="$2";   shift 2 ;;
    --expected-version)  expected_version="$2";  shift 2 ;;
    --sample-path)       sample_path="$2";       shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

: "${python_version:?--python-version is required}"
: "${wheel_arch:?--wheel-arch is required}"
: "${wheel_directory:?--wheel-directory is required}"
: "${expected_version:?--expected-version is required}"
: "${sample_path:?--sample-path is required}"

py_no_dot=$(echo "$python_version" | tr -d '.')
env_name="py${py_no_dot}_release_testing_env"

# The manylinux container ships every supported CPython under /opt/python/
python_bin="/opt/python/cp${py_no_dot}-cp${py_no_dot}/bin/python"
if [ ! -x "$python_bin" ]; then
  echo "ERROR: Python interpreter not found at $python_bin"
  exit 1
fi

# Find the wheel that matches both the Python version and the platform
wheel=$(find "$wheel_directory" -maxdepth 1 \
  -name "*cp${py_no_dot}-cp${py_no_dot}-${wheel_arch}.whl" | head -n 1)

if [ -z "$wheel" ]; then
  echo "ERROR: No wheel found matching cp${py_no_dot}-${wheel_arch} in $wheel_directory"
  exit 1
fi
echo "Found wheel: $wheel"

# Create venv named py{XYZ}_release_testing_env
"$python_bin" -m venv "$env_name"

# Activate venv
source "$env_name/bin/activate"

# Upgrade pip and install the local wheel
python -m pip install --upgrade pip
python -m pip install "$wheel"

# Verify the package reports the expected version
reported=$(python -c "import onnxruntime_qnn as qnn_ep; print(qnn_ep.__version__)")
echo "onnxruntime_qnn.__version__ = $reported"
if [ "$reported" != "$expected_version" ]; then
  printf '\033[0;31mVersion mismatch: expected %s, got %s\033[0m\n' "$expected_version" "$reported" >&2
  exit 1
fi
echo "Version check PASS"

# Run the sample test
python "$sample_path"
