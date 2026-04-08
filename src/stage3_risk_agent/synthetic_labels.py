"""
LLM-based synthetic risk label generation for CUAD clauses.

Primary: Qwen-32B-Instruct. Backup: Gemini / OpenAI API.
Output: data/synthetic/risk_labels.json
"""

import logging

from src.common.schema import SyntheticRiskLabel

logger = logging.getLogger(__name__)


def generate_risk_labels(
    clauses: list[dict],
    model: str = "Qwen/Qwen2.5-32B-Instruct",
    output_path: str = "data/synthetic/risk_labels.json",
) -> list[SyntheticRiskLabel]:
    """Generate synthetic risk labels for a batch of CUAD clauses.

    Prompts the LLM for each clause: "Given this clause, classify risk as
    LOW/MEDIUM/HIGH and explain why."

    Args:
        clauses: List of dicts with 'clause_text' and 'clause_type' keys.
        model: LLM model name or API endpoint.
        output_path: Path to save generated labels as JSON.

    Returns:
        List of SyntheticRiskLabel objects.
    """
    raise NotImplementedError


def build_prompt(clause_text: str, clause_type: str) -> str:
    """Build the LLM prompt for risk label generation.

    Args:
        clause_text: The clause text to label.
        clause_type: The CUAD clause type (e.g. "Indemnification").

    Returns:
        Formatted prompt string.
    """
    raise NotImplementedError


def parse_llm_response(response: str, clause_text: str, clause_type: str) -> SyntheticRiskLabel:
    """Parse the LLM text response into a SyntheticRiskLabel.

    Args:
        response: Raw LLM output text.
        clause_text: Original clause text.
        clause_type: CUAD clause type.

    Returns:
        Parsed SyntheticRiskLabel.
    """
    raise NotImplementedError
