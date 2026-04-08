"""Download and cache the CUAD dataset (SQuAD format) from HuggingFace.

Usage:
    python scripts/download_cuad.py [--output data/raw/cuad]
"""

import argparse
import logging
from pathlib import Path

from src.common.utils import setup_logging

logger = logging.getLogger(__name__)


def download_cuad(output_dir: str = "data/raw/cuad") -> None:
    """Download kenlevine/CUAD dataset and save to disk.

    Args:
        output_dir: Directory to save the raw dataset files.
    """
    raise NotImplementedError


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Download CUAD dataset.")
    parser.add_argument("--output", default="data/raw/cuad", help="Output directory.")
    args = parser.parse_args()
    download_cuad(args.output)
