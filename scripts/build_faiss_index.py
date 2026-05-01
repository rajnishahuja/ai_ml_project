"""Build FAISS index from labeled clause embeddings.

Usage:
    python scripts/build_faiss_index.py [--config configs/stage3_config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.utils import setup_logging, load_config
from src.stage3_risk_agent.embeddings import build_index

logger = logging.getLogger(__name__)


def main(config_path: str = "configs/stage3_config.yaml") -> None:
    cfg = load_config(config_path)
    training_data_path = cfg["risk_classifier"]["training_data_path"]
    index_path = cfg["faiss_index_path"]

    logger.info("Building FAISS index")
    logger.info("  Source : %s", training_data_path)
    logger.info("  Output : %s", index_path)

    build_index(training_data_path, index_path)
    logger.info("Done.")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Build FAISS index.")
    parser.add_argument("--config", default="configs/stage3_config.yaml")
    args = parser.parse_args()
    main(args.config)
