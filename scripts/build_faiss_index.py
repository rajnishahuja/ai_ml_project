"""Build FAISS index from labeled clause embeddings.

Usage:
    python scripts/build_faiss_index.py [--config configs/stage3_config.yaml]
"""

import argparse
import logging

from src.common.utils import setup_logging

logger = logging.getLogger(__name__)


def main(config_path: str = "configs/stage3_config.yaml") -> None:
    """Load synthetic-labeled clauses, embed them, build FAISS index.

    Steps:
        1. Load risk_labels.json from data/synthetic/.
        2. Encode clause texts with all-MiniLM-L6-v2.
        3. Build FAISS index and save to data/faiss_index/.

    Args:
        config_path: Path to stage3 config YAML.
    """
    raise NotImplementedError


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Build FAISS index.")
    parser.add_argument("--config", default="configs/stage3_config.yaml")
    args = parser.parse_args()
    main(args.config)
