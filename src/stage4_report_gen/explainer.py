"""
FLAN-T5-base explanation generator for risk report.

Generates plain-language explanations of why a clause is risky,
using prompt templates.
"""

import logging

from src.common.schema import RiskAssessedClause

logger = logging.getLogger(__name__)


def load_explanation_model(
    model_name: str = "google/flan-t5-base",
    device: int = -1,
) -> object:
    """Load the FLAN-T5 model for explanation generation.

    Args:
        model_name: HuggingFace model name.
        device: -1 for CPU, 0+ for GPU.

    Returns:
        Loaded model/pipeline object.
    """
    raise NotImplementedError


def generate_explanation(
    clause: RiskAssessedClause,
    model: object,
    max_length: int = 200,
) -> str:
    """Generate a plain-language explanation for a risky clause.

    Uses a prompt template combining clause text, type, and risk level
    to produce a human-readable risk explanation.

    Args:
        clause: The risk-assessed clause to explain.
        model: Loaded FLAN-T5 model/pipeline.
        max_length: Maximum token length for the explanation.

    Returns:
        Explanation string.
    """
    raise NotImplementedError


def build_explanation_prompt(clause: RiskAssessedClause) -> str:
    """Build prompt template for FLAN-T5 explanation.

    Args:
        clause: The clause to explain.

    Returns:
        Formatted prompt string.
    """
    raise NotImplementedError
