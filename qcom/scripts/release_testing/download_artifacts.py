# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""
Download artifacts from Artifactory using JFROG API key.

Used by the release-testing workflow to pull artifacts from artifactory
"""

import argparse
import os
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ARTIFACTORY_BASE_URL = os.environ.get("ARTIFACTORY_BASE_URL")
JFROG_API_KEY = os.environ.get("JFROG_API_KEY")

# (artifact_type, index_server) -> Artifactory repo name
REPOSITORIES = {
    ("wheel", "test-users"): "aisw-pypi-test-users",
    ("wheel", "test-project"): "aisw-pypi-test-project-local",
    ("wheel", "project"): "aisw-pypi-project",
    ("wheel", "public"): "aisw-pypi-public",
    ("zip", "test-users"): "aisw-zip-test-users",
    ("zip", "test-project"): "aisw-zip-test-project",
    ("zip", "project"): "aisw-zip-project",
    ("zip", "public"): "aisw-zip-public",
    # tgz artifacts share the same Artifactory repos as zip
    ("tgz", "test-users"): "aisw-zip-test-users",
    ("tgz", "test-project"): "aisw-zip-test-project",
    ("tgz", "project"): "aisw-zip-project",
    ("tgz", "public"): "aisw-zip-public",
    ("nuget", "test-users"): "aisw-nuget-test-users",
    ("nuget", "test-project"): "aisw-nuget-test-project",
    ("nuget", "project"): "aisw-nuget-project",
    ("nuget", "public"): "aisw-nuget-public",
}

ARTIFACT_TYPES = sorted({k[0] for k in REPOSITORIES})
INDEX_SERVERS = sorted({k[1] for k in REPOSITORIES})

# The test-binaries package always lives in this repo, independent of index_server.
TEST_PACKAGES_REPO = "aisw-zip-test-project"
TEST_PACKAGES_FILENAME = "test_packages.zip"


def _check_env() -> None:
    if not ARTIFACTORY_BASE_URL:
        print("ERROR: ARTIFACTORY_BASE_URL not set in environment", file=sys.stderr)
        sys.exit(1)
    if not JFROG_API_KEY:
        print("ERROR: JFROG_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)


def list_artifacts(repo: str, path: str, verify_ssl: bool = False) -> list | None:
    """List file artifacts under <repo><path> via the storage API."""
    url = f"{ARTIFACTORY_BASE_URL}/api/storage/{repo}{path}"
    try:
        response = requests.get(
            url=url,
            auth=("", JFROG_API_KEY),
            verify=verify_ssl,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()
        return [item["uri"].lstrip("/") for item in results.get("children", []) if not item.get("folder", False)]
    except requests.exceptions.RequestException as e:
        print(f"Failed to list artifacts at {url}: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(f"  status={e.response.status_code}", file=sys.stderr)
            print(f"  body={e.response.text}", file=sys.stderr)
        return None


def download_artifact(repo: str, artifact_path: str, output_dir: str, verify_ssl: bool = False) -> bool:
    """Download a single artifact to output_dir."""
    url = f"{ARTIFACTORY_BASE_URL}/{repo}/{artifact_path}"
    output_path = Path(output_dir) / Path(artifact_path).name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {artifact_path}")
    try:
        response = requests.get(
            url=url,
            auth=("", JFROG_API_KEY),
            verify=verify_ssl,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"  Progress: {percent:.1f}%", end="\r")
        print("  Downloaded successfully")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Failed to download: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download artifacts from Artifactory")
    parser.add_argument(
        "--artifact_folder_name",
        required=True,
        help="Version folder under the package, e.g. 0.1.0.test0",
    )
    parser.add_argument(
        "--artifact_to_be_downloaded",
        choices=ARTIFACT_TYPES,
        help="Artifact type (not required with --download-test-packages, required otherwise)",
    )
    parser.add_argument(
        "--index_server",
        choices=INDEX_SERVERS,
        help="Which server hosts the artifact (not required with --download-test-packages, required otherwise)",
    )
    parser.add_argument(
        "--download-test-packages",
        dest="download_test_packages",
        action="store_true",
        help=(
            f"Download {TEST_PACKAGES_FILENAME} from "
            f"{TEST_PACKAGES_REPO}/onnxruntime-qnn/<artifact_folder_name>/ "
            "instead of a regular artifact folder."
        ),
    )
    parser.add_argument(
        "--output_directory",
        default="./output",
        help="Output directory for downloaded files",
    )
    args = parser.parse_args()

    _check_env()

    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Test-packages mode: fetch a single known file from a fixed repo ---
    if args.download_test_packages:
        artifact_path = f"onnxruntime-qnn/{args.artifact_folder_name}/{TEST_PACKAGES_FILENAME}"
        print(f"Downloading {TEST_PACKAGES_REPO}/{artifact_path} ...")
        ok = download_artifact(
            repo=TEST_PACKAGES_REPO,
            artifact_path=artifact_path,
            output_dir=str(output_dir),
        )
        return 0 if ok else 1

    # --- Regular mode: list and download a whole artifact folder ---
    if not args.artifact_to_be_downloaded or not args.index_server:
        print(
            "ERROR: --artifact_to_be_downloaded and --index_server are required unless --download-test-packages is set",
            file=sys.stderr,
        )
        return 1

    key = (args.artifact_to_be_downloaded, args.index_server)
    if key not in REPOSITORIES:
        print(
            f"ERROR: No repo configured for ({args.artifact_to_be_downloaded}, {args.index_server})",
            file=sys.stderr,
        )
        return 1
    repo = REPOSITORIES[key]

    base_path = f"onnxruntime-qnn/{args.artifact_folder_name}"

    artifacts = list_artifacts(repo=repo, path=f"/{base_path}")
    if not artifacts:
        print("No artifacts found", file=sys.stderr)
        return 1

    print(f"Found {len(artifacts)} artifact(s):")
    for name in artifacts:
        print(f"  - {name}")
    print()

    success = 0
    failure = 0
    for name in artifacts:
        if download_artifact(
            repo=repo,
            artifact_path=f"{base_path}/{name}",
            output_dir=str(output_dir),
        ):
            success += 1
        else:
            failure += 1
        print()

    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total files:             {len(artifacts)}")
    print(f"Successfully downloaded: {success}")
    print(f"Failed to download:      {failure}")
    print("=" * 60)
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
