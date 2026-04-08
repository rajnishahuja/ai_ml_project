"""
Inference module: contract text → List[ClauseObject].

Runs all 41 CUAD clause-type queries against a contract,
applies confidence threshold, and deduplicates overlapping spans.
"""

import logging

from src.common.schema import ClauseObject, ExtractionResult

logger = logging.getLogger(__name__)


# 41 CUAD clause types
CUAD_CLAUSE_TYPES: list[str] = [
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Most Favored Nation", "Non-Compete", "Exclusivity",
    "No-Solicit Of Customers", "No-Solicit Of Employees", "Non-Disparagement",
    "Termination For Convenience", "ROFR/ROFO/ROFN", "Change Of Control",
    "Anti-Assignment", "Revenue/Profit Sharing", "Price Restrictions",
    "Minimum Commitment", "Volume Restriction", "IP Ownership Assignment",
    "Joint IP Ownership", "License Grant", "Non-Transferable License",
    "Affiliate License-Licensor", "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License", "Irrevocable Or Perpetual License",
    "Source Code Escrow", "Post-Termination Services", "Audit Rights",
    "Uncapped Liability", "Cap On Liability", "Liquidated Damages",
    "Warranty Duration", "Insurance", "Covenant Not To Sue",
    "Third Party Beneficiary", "Indemnification",
]


def predict_clauses(
    contract_text: str,
    model_path: str,
    document_id: str = "unknown",
    confidence_threshold: float = 0.01,
) -> list[ClauseObject]:
    """Extract clauses from a contract by running all 41 QA queries.

    Args:
        contract_text: Full contract text string.
        model_path: Path to fine-tuned DeBERTa model.
        document_id: Identifier for the source document.
        confidence_threshold: Minimum score to include a clause.

    Returns:
        List of ClauseObject sorted by start_pos.
    """
    raise NotImplementedError


def predict_from_file(
    file_path: str,
    model_path: str,
    confidence_threshold: float = 0.01,
) -> ExtractionResult:
    """Convenience: extract text from file and predict clauses.

    Args:
        file_path: Path to contract file (PDF, DOCX, or TXT).
        model_path: Path to fine-tuned DeBERTa model.
        confidence_threshold: Minimum score to include a clause.

    Returns:
        ExtractionResult with document_id and clause list.
    """
    raise NotImplementedError


def deduplicate_spans(clauses: list[ClauseObject], iou_threshold: float = 0.5) -> list[ClauseObject]:
    """Remove overlapping clause predictions from sliding window.

    When the same clause spans multiple windows, keep the highest-confidence one.

    Args:
        clauses: Raw clause predictions (may contain overlaps).
        iou_threshold: IoU above which two predictions are considered duplicates.

    Returns:
        Deduplicated clause list.
    """
    raise NotImplementedError
