"""
Fine-tuning script for DeBERTa-base on CUAD QA task.

Reads all hyperparameters from configs/stage1_config.yaml.
Uses HuggingFace Trainer with sliding window for long contracts.
"""

import logging

logger = logging.getLogger(__name__)


def fine_tune(config_path: str = "configs/stage1_config.yaml") -> None:
    """Fine-tune DeBERTa-base on CUAD dataset.

    Steps:
        1. Load config from YAML.
        2. Load and tokenize CUAD dataset via data_loader.
        3. Initialize DeBERTa model for QA.
        4. Train with HuggingFace Trainer.
        5. Save model + tokenizer to output_dir.

    Args:
        config_path: Path to stage1 config YAML.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Stage 1+2 DeBERTa model")
    parser.add_argument("--config", default="configs/stage1_config.yaml", help="Config path")
    args = parser.parse_args()
    fine_tune(args.config)
