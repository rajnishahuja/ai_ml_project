"""End-to-end pipeline: run all stages on a contract file.

Usage:
    python scripts/run_pipeline.py --input contract.pdf [--output report.json]
"""

import argparse
import logging

from src.common.utils import setup_logging

logger = logging.getLogger(__name__)


def run_pipeline(input_path: str, output_path: str = "output/report.json") -> None:
    """Execute full pipeline: extract → classify → risk assess → report.

    Stage 1+2: Extract and classify clauses from contract text.
    Stage 3:   Assess risk level for each clause via LangGraph agent.
    Stage 4:   Generate risk report with explanations and recommendations.

    Args:
        input_path: Path to input contract file (PDF, DOCX, or TXT).
        output_path: Path to save the final JSON report.
    """
    raise NotImplementedError


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Run full pipeline.")
    parser.add_argument("--input", required=True, help="Path to contract file.")
    parser.add_argument("--output", default="output/report.json", help="Output report path.")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
