import json
import argparse
import logging
from src.common.schema import ClauseObject
from src.stage3_risk_agent.agent import assess_clauses

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/output/stage1/stage1_output.json",
        help="Path to Stage 1 JSON",
    )
    parser.add_argument(
        "--output",
        default="data/output/stage3/stage3_output.json",
        help="Path to save Assessed JSON",
    )
    parser.add_argument("--config", default="configs/stage3_config.yaml")
    args = parser.parse_args()

    logger.info(f"Stage 3: Loading clauses from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)

    clauses = [ClauseObject(**c) for c in data]

    logger.info(f"Stage 3: Running Agent Assessment on all {len(clauses)} clauses...")
    assessed_clauses = assess_clauses(clauses=clauses, config_path=args.config)

    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save as JSON list
    output_data = [c.to_dict() for c in assessed_clauses]
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(
        f"Done! Risk assessment complete for {len(assessed_clauses)} clauses. saved to {args.output}"
    )


if __name__ == "__main__":
    main()
