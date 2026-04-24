"""
Report assembly: combines aggregation, explanations, and recommendations
into a final RiskReport.
"""

import logging

from src.common.schema import RiskAssessedClause, RiskReport

logger = logging.getLogger(__name__)


def build_report(
    clauses: list[RiskAssessedClause],
    document_id: str,
    config_path: str = "configs/stage4_config.yaml",
) -> RiskReport:
    """Build the final risk report for a contract.

    Steps:
        1. Aggregate clauses (group by risk, compute contract score).
        2. Generate FLAN-T5 explanations for HIGH/MEDIUM clauses.
        3. Look up recommendations from the curated table.
        4. Assemble into RiskReport dataclass.

    Args:
        clauses: Stage 3 output (list of RiskAssessedClause).
        document_id: Identifier for the source contract.
        config_path: Path to stage4 config YAML.

    Returns:
        Complete RiskReport ready for serialization.
    """
    raise NotImplementedError


def save_report(report: RiskReport, output_path: str) -> None:
    """Serialize and save a RiskReport to JSON.

    Args:
        report: The assembled report.
        output_path: File path to write the JSON output.
    """
    raise NotImplementedError
