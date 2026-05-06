from typing import List
from pydantic import BaseModel, Field


class ClauseInput(BaseModel):
    """A single clause from Stage 1 extraction — input to Stage 3."""
    clause_id: str
    document_id: str
    clause_type: str
    clause_text: str
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.95


class Stage3Request(BaseModel):
    clauses: List[ClauseInput]
    use_contract_search: bool = Field(
        default=True,
        description="Enable contract_search tool for cross-clause context",
    )


class AssessedClauseInput(BaseModel):
    """A risk-assessed clause from Stage 3 — input to Stage 4."""
    clause_id: str
    document_id: str
    clause_text: str
    clause_type: str
    risk_level: str
    risk_explanation: str
    confidence: float = 0.0
    similar_clauses: List[dict] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)
    agent_trace: List[dict] = Field(default_factory=list)


class Stage4Request(BaseModel):
    document_id: str
    assessed_clauses: List[AssessedClauseInput]
