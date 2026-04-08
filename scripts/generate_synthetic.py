"""Generate synthetic risk labels using Qwen-32B (primary) or backup LLMs.

Usage:
    python scripts/generate_synthetic.py [--config configs/stage3_config.yaml]
"""

import argparse
import logging

from src.common.utils import setup_logging

logger = logging.getLogger(__name__)


def main(config_path: str = "configs/stage3_config.yaml") -> None:
    """Load CUAD clauses and generate synthetic risk labels.

    Steps:
        1. Load flattened QA examples from CUAD.
        2. Filter to non-empty answer spans (actual clauses).
        3. Generate risk labels via Qwen-32B prompt.
        4. Save to data/synthetic/risk_labels.json.

    Args:
        config_path: Path to stage3 config YAML.
    """
    raise NotImplementedError


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate synthetic risk labels.")
    parser.add_argument("--config", default="configs/stage3_config.yaml")
    args = parser.parse_args()
    main(args.config)
