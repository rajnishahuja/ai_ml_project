# System Architecture Context

This document serves as the primary AI context reference for the Legal Contract Risk Analyzer. 

## End-to-End Pipeline Topology

```text
       Contract PDF/Text
              │
              ▼
┌─────────────────────────────┐
│  Stage 1+2: Extract &       │  Single DeBERTa-base model
│  Classify (combined)        │  CUAD dataset, QA format
│  Output: clause objects     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 3: Risk Detection    │  LangGraph agent
│  Agent with RAG             │  DeBERTa-base (risk classifier)
│  Tools: FAISS retrieval,    │  Mistral-7B-Instruct (explanations)
│         contract search     │  all-MiniLM-L6-v2 (embeddings)
│  Output: risk-assessed      │  FAISS vector store
│          clause objects     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 4: Report Generation │  Python aggregation code
│  Hybrid: code + LLM         │  FLAN-T5-base (explanations)
│  Output: structured risk    │  Lookup table (recommendations)
│          report             │
└─────────────────────────────┘
```

## Component Breakdown

### Stage 1+2: Extract & Classify (Combined)
*   **Purpose:** Extract target raw clauses from the parsed contract text.
*   **Model:** `DeBERTa-base` model fine-tuned on the CUAD dataset (using QA format constraints/templates).
*   **Output DTO:** Clause Objects (containing character spans, confidence, and raw text).

### Stage 3: Risk Detection Agent
*   **Purpose:** Assess extracted clauses contextually using a multi-tool agentic architecture (RAG).
*   **Orchestrator:** `LangGraph`.
*   **Models Used:**
    *   **Risk Classifier:** `DeBERTa-base` (specialized for risk detection/flagging).
    *   **Generator:** `Mistral-7B-Instruct` (generates detailed textual explanations for flagged risks).
    *   **Embedder:** `all-MiniLM-L6-v2` (encodes text chunks for the vector store).
*   **Retrieval Tools:** Local FAISS vector store, Contract search function.
*   **Output DTO:** Risk-assessed Clause Objects.

### Stage 4: Report Generation
*   **Purpose:** Aggregate all risk-assessed clauses and format them into a client-ready comprehensive report.
*   **Approach:** Hybrid integration linking deterministic python grouping logic with lightweight LLM synthesis.
*   **Models/Tools Used:**
    *   **Generator:** `FLAN-T5-base` (for final explanation synthesis and formatting).
    *   **Static Data:** Python Dictionary/Database `Lookup Table` (mapped deterministic recommendations based on clause types).
*   **Final Output:** A structured, JSON-serializable, or Docx-renderable Final Risk Report.

## System Data Schemas (DTOs)

### 1. Stage 1+2 Output -> Stage 3 Input
```json
{
  "clause_id": "c_001",
  "clause_text": "Contractor shall indemnify Company...",
  "clause_type": "Indemnification",
  "start_pos": 4521,
  "end_pos": 4687,
  "confidence": 0.94,
  "confidence_logit": 10.7702
}
```

### 2. Stage 3 Output -> Stage 4 Input
```json
{
  "clause_id": "c_001",
  "clause_text": "Contractor shall indemnify Company...",
  "clause_type": "Indemnification",
  "risk_level": "HIGH",
  "risk_reason": "One-sided indemnification covering counterparty negligence",
  "similar_clauses": [
    {"text": "...", "risk": "HIGH", "similarity": 0.92}
  ],
  "cross_references": ["c_015"],
  "confidence": 0.88,
  "confidence_logit": 10.7702
}
```

### 3. Final Result Report (Stage 4 Output)
```json
{
  "summary": "This vendor agreement contains 23 clauses...",
  "high_risk": [
    {
      "clause_id": "c_001",
      "explanation": "The indemnification provision requires...",
      "recommendation": "Renegotiate to mutual indemnification..."
    }
  ],
  "medium_risk": [],
  "low_risk_summary": "15 clauses were assessed as standard/low risk...",
  "missing_protections": ["Data Protection", "Force Majeure"],
  "overall_risk_score": 6.8,
  "total_clauses": 23
}
```
