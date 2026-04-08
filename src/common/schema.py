"""
Shared data contracts for all pipeline stages.

All stages communicate through these typed dataclasses.
This file is the single source of truth for inter-stage data formats.
See ARCHITECTURE.md for the full JSON schema documentation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Stage 1+2 Output → Stage 3 Input
# ---------------------------------------------------------------------------

@dataclass
class ClauseObject:
    """A single extracted and classified clause from a contract."""
    clause_id: str
    document_id: str
    clause_text: str
    clause_type: str
    start_pos: int
    end_pos: int
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Full extraction result for one contract document."""
    document_id: str
    clauses: list[ClauseObject] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "clauses": [c.to_dict() for c in self.clauses],
        }


# ---------------------------------------------------------------------------
# Stage 3: FAISS retrieval result
# ---------------------------------------------------------------------------

@dataclass
class SimilarClause:
    """A clause retrieved from FAISS as similar to the query clause."""
    text: str
    risk_level: str
    similarity: float


# ---------------------------------------------------------------------------
# Stage 3 Output → Stage 4 Input
# ---------------------------------------------------------------------------

@dataclass
class AgentTraceEntry:
    """One tool invocation in the LangGraph agent's reasoning trace."""
    tool: str
    result_count: Optional[int] = None
    related_clauses: Optional[int] = None


@dataclass
class RiskAssessedClause:
    """A clause with risk assessment from the Stage 3 agent."""
    clause_id: str
    document_id: str
    clause_text: str
    clause_type: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_explanation: str
    similar_clauses: list[SimilarClause] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    confidence: float = 0.0
    agent_trace: list[AgentTraceEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Synthetic risk labels (training data for Stage 3)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticRiskLabel:
    """An LLM-generated risk label for a CUAD clause."""
    clause_text: str
    clause_type: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_reason: str
    labeled_by: str  # e.g. "qwen-32b", "gemini", "openai"


# ---------------------------------------------------------------------------
# Stage 4 Output (Final Report)
# ---------------------------------------------------------------------------

@dataclass
class ReportClause:
    """A single clause entry in the final risk report."""
    clause_id: str
    clause_type: str
    risk_level: str
    explanation: str
    recommendation: str


@dataclass
class ReportMetadata:
    """Metadata about the report generation run."""
    generated_at: str
    models_used: dict[str, str] = field(default_factory=dict)


@dataclass
class RiskReport:
    """The final structured risk report for a contract."""
    document_id: str
    summary: str
    high_risk: list[ReportClause] = field(default_factory=list)
    medium_risk: list[ReportClause] = field(default_factory=list)
    low_risk_summary: str = ""
    missing_protections: list[str] = field(default_factory=list)
    overall_risk_score: float = 0.0
    total_clauses: int = 0
    metadata: Optional[ReportMetadata] = None

    def to_dict(self) -> dict:
        return asdict(self)
