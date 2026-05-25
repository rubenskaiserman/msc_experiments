"""Download The Well splits to local storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thewell_videomae.data import download_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download The Well dataset splits")
    parser.add_argument("--data-base", type=str, required=True, help="Directory that will contain ./datasets")
    parser.add_argument("--dataset", type=str, default="shear_flow")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    download_dataset(args.data_base, args.dataset, tuple(args.splits))


if __name__ == "__main__":
    main()

