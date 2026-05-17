import json
import argparse
import logging
from src.common.schema import RiskAssessedClause, AgentTraceEntry
from src.stage4_report_gen.report_builder import (
    build_report,
    save_report,
    save_markdown_report,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/output/stage3/stage3_output.json",
        help="Path to Stage 3 JSON",
    )
    parser.add_argument(
        "--output",
        default="data/output/stage4/final_report.json",
        help="Path to save Report JSON",
    )
    parser.add_argument("--config", default="configs/stage4_config.yaml")
    args = parser.parse_args()

    logger.info(f"Stage 4: Loading assessed clauses from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)

    # Re-construct objects including traces
    clauses = []
    for c in data:
        trace = [AgentTraceEntry(**t) for t in c.get("agent_trace", [])]
        c["agent_trace"] = trace
        clauses.append(RiskAssessedClause(**c))

    if not clauses:
        logger.error("No clauses found in input file.")
        return

    doc_id = clauses[0].document_id

    logger.info("Stage 4: Building Final Report...")
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = build_report(clauses=clauses, document_id=doc_id, config_path=args.config)

    # Save JSON Report
    save_report(report, args.output)

    # Commented out Markdown output - focusing on JSON-only outputs for now
    # md_path = args.output.replace(".json", ".md")
    # save_markdown_report(report, md_path)
    # logger.info(f"Done! Final reports saved to {args.output} and {md_path}")
    
    logger.info(f"Done! Final JSON report saved to {args.output}")


if __name__ == "__main__":
    main()
