"""Download benchmark datasets from provided URLs."""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile


def _download(url: str, target_path: str) -> None:
    with urllib.request.urlopen(url) as response, open(target_path, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract(archive_path: str, output_dir: str) -> None:
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(output_dir)
        return

    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(output_dir)
        return

    raise ValueError("Unsupported archive format")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--name", required=True, help="Dataset name")
    parser.add_argument("--url", required=True, help="Dataset URL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract archive if the URL points to zip/tar.gz",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    filename = os.path.basename(args.url)
    target_path = os.path.join(args.output, filename)

    _download(args.url, target_path)

    if args.extract:
        _extract(target_path, args.output)

    print(f"Downloaded {args.name} to {args.output}")


if __name__ == "__main__":
    main()
