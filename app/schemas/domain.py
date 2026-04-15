from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# STAGE 1+2 SCHEMAS
# ==========================================

class ExtractedClause(BaseModel):
    """
    Output from Stage 1+2 (DeBERTa Extraction).
    Acts as the direct Input to Stage 3.
    """
    clause_id: str
    clause_text: str
    clause_type: str
    start_pos: int
    end_pos: int
    confidence: float
    confidence_logit: float


# ==========================================
# STAGE 3 SCHEMAS
# ==========================================

class SimilarClause(BaseModel):
    text: str
    risk: str
    similarity: float

class RiskAssessedClause(ExtractedClause):
    """
    Output from Stage 3 (LangGraph Risk Agent).
    Inherits base properties from ExtractedClause and appends risk metadata.
    Acts as the direct Input to Stage 4.
    """
    risk_level: str
    risk_reason: str
    similar_clauses: List[SimilarClause] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)
    
    # Override confidence if the risk agent adjusts it
    confidence: float


# ==========================================
# STAGE 4 SCHEMAS
# ==========================================

class RiskReportRecommendation(BaseModel):
    clause_id: str
    explanation: str
    recommendation: str

class FinalRiskReport(BaseModel):
    """
    Output from Stage 4 (Hybrid Generation).
    The Final API Payload returned to the client.
    """
    summary: str
    high_risk: List[RiskReportRecommendation] = Field(default_factory=list)
    medium_risk: List[RiskReportRecommendation] = Field(default_factory=list)
    low_risk_summary: str
    missing_protections: List[str] = Field(default_factory=list)
    overall_risk_score: float
    total_clauses: int
