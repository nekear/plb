from __future__ import annotations

import argparse
from pathlib import Path

from .extractors import POLICY, SUBMISSIONS
from .process import build_dataset
from plb.utils import get_dataset_as_path

DEFAULT_RELEASE_DIR = get_dataset_as_path("release")
DEFAULT_OUTPUT_DIR = get_dataset_as_path("interim")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m plb.normalize")
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="Directory holding the public release tables downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where normalized JSONL files will be written.",
    )
    args = parser.parse_args(argv)

    counts = build_dataset(SUBMISSIONS, args.release_dir, args.output_dir, policy=POLICY)

    failed = counts.pop("failed_submissions", 0)
    width = max(len(k) for k in counts) + 1
    print("\nDataset summary:")
    for key, value in counts.items():
        print(f"  {key + ':':<{width}} {value:>10,}")
    if failed:
        print(f"  {'failed:':<{width}} {failed:>10,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
