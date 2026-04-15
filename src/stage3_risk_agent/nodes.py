from app.schemas.domain import RiskAssessedClause
from src.workflow.state import RiskAnalysisState
from src.stage3_risk_agent.embeddings import embed_and_store


def node_faiss_embedding(state: RiskAnalysisState):
    """
    STAGE 3 (Part 1 — Memory/RAG Prep) — Node B
    Runs fully in parallel with Node A. Chunks the raw contract text
    and encodes it into the persistent FAISS vector index via local Ollama
    embeddings. Provides the semantic memory base for the Mistral RAG step.
    Output: faiss_status -> signals Node D that the index is ready
    """
    print("🔄 [NODE B | Stage 3 - FAISS] Chunking and encoding document into FAISS...")
    faiss_index_path = embed_and_store(
        contract_text=state["contract_text"],
        document_id=state["document_id"]
    )
    print(f"✅ [NODE B | Stage 3 - FAISS] Vectors synced at: {faiss_index_path}")
    return {"faiss_status": "Complete"}


def node_risk_classifier(state: RiskAnalysisState):
    """
    STAGE 3 (Part 2 — Risk Scoring) — Node C
    Waits for Node A (extracted clauses). Uses a secondary DeBERTa sequence
    classification model to assign raw risk severity flags (HIGH/MEDIUM/LOW)
    to each clause. This is deterministic — no LLM generation happens here.
    Output: flagged_clauses -> passed to Node D (Mistral explainer)
    """
    clauses = state.get("extracted_clauses", [])
    print(f"🔄 [NODE C | Stage 3 - Classifier] Scoring {len(clauses)} clauses via DeBERTa risk model...")

    # MOCK: Cycle through HIGH / MEDIUM / LOW so all state parameters are visible E2E
    risk_cycle = ["HIGH", "MEDIUM", "LOW"]
    flagged = [
        {"clause": c, "risk_level": risk_cycle[i % 3]}
        for i, c in enumerate(clauses)
    ]

    print(f"✅ [NODE C | Stage 3 - Classifier] Flags assigned to {len(flagged)} clauses.")
    return {"flagged_clauses": flagged}


def node_mistral_explainer(state: RiskAnalysisState):
    """
    STAGE 3 (Part 3 — RAG Reasoning) — Node D
    The convergence point. Waits for BOTH Node B (FAISS ready) AND Node C (risk flags).
    Queries the FAISS store via RAG to find similar historical clauses, then feeds
    Mistral-7B-Instruct a rich prompt to generate human-readable risk explanations
    and negotiation recommendations per flagged clause.
    Output: risk_assessed_clauses -> passed to Node E (report generator)
    """
    flagged = state.get("flagged_clauses", [])
    faiss_ready = state.get("faiss_status")
    print(f"🔄 [NODE D | Stage 3 - Mistral RAG] FAISS={faiss_ready}. Explaining {len(flagged)} flagged clauses...")

    # MOCK reasons per risk level — so the final report shows all three tiers
    mock_reasons = {
        "HIGH":   "[MOCK] One-sided indemnification covering counterparty negligence.",
        "MEDIUM": "[MOCK] Liability cap may be insufficient relative to contract value.",
        "LOW":    "[MOCK] Standard boilerplate with minor ambiguity in termination notice period.",
    }

    assessed = []
    for item in flagged:
        c = item["clause"]
        level = item["risk_level"]
        assessed.append(RiskAssessedClause(
            clause_id=c.clause_id,
            clause_text=c.clause_text,
            clause_type=c.clause_type,
            start_pos=c.start_pos,
            end_pos=c.end_pos,
            confidence=c.confidence,
            confidence_logit=c.confidence_logit,
            risk_level=level,
            risk_reason=mock_reasons.get(level, "[MOCK] Risk assessed."),
            similar_clauses=[],
            cross_references=[]
        ))

    print(f"✅ [NODE D | Stage 3 - Mistral RAG] Explained {len(assessed)} clauses.")
    return {"risk_assessed_clauses": assessed}
