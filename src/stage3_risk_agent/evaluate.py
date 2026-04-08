"""
Risk detection evaluation: metrics and ablation.

Measures: accuracy, precision/recall/F1 per risk level, confusion matrix.
Ablation: agent with both tools vs single tool vs no tools.
"""

import logging

from src.common.schema import RiskAssessedClause, SyntheticRiskLabel

logger = logging.getLogger(__name__)


def evaluate_risk_predictions(
    predictions: list[RiskAssessedClause],
    ground_truth: list[SyntheticRiskLabel],
) -> dict:
    """Evaluate risk classification against ground truth labels.

    Args:
        predictions: Agent output (RiskAssessedClause list).
        ground_truth: Validated synthetic labels.

    Returns:
        Dict with accuracy, per-class P/R/F1, and confusion matrix.
    """
    raise NotImplementedError


def run_ablation(
    clauses: list,
    ground_truth: list[SyntheticRiskLabel],
    config_path: str = "configs/stage3_config.yaml",
) -> dict:
    """Run ablation study: compare agent configurations.

    Configurations:
        - Full agent (both tools)
        - FAISS retrieval only
        - Contract search only
        - No tools (classifier + explanation only)

    Args:
        clauses: Stage 1+2 output clauses.
        ground_truth: Validated synthetic risk labels.
        config_path: Path to stage3 config YAML.

    Returns:
        Dict mapping config name → evaluation metrics.
    """
    raise NotImplementedError
