"""
Stage 3: Risk Classification + Agentic Explanation Nodes.

Two-path architecture (per ARCHITECTURE.md):
  - Fast Path  (conf >= 0.6): precedent_search(k=3) + 1 Mistral call (direct, no loop)
  - Agent Path (conf <  0.6): ReAct tool loop (max 5 turns)
                               Tools: precedent_search + contract_search
                               Mistral may override DeBERTa's preliminary label.

Both paths are currently MOCKED with deterministic stubs.
Replace the _mock_* functions and LLM calls once:
  1. DeBERTa risk classifier training completes (Node C)
  2. Ollama + FAISS are connected (Node D)
"""

import logging
from typing import TypedDict, Any, Dict, List

from app.schemas.domain import RiskAssessedClause, SimilarClause
from src.workflow.state import RiskAnalysisState

logger = logging.getLogger(__name__)


# ==========================================
# STATE DEFINITIONS
# ==========================================

class RAGWorkerState(TypedDict):
    """State strictly for the isolated Map-Reduce worker node (Node D)."""
    clause_data: Dict[str, Any]


# ==========================================
# MOCK TOOL STUBS
# Replace with real FAISS + contract DB calls when models are ready.
# ==========================================

def _mock_precedent_search(clause_text: str, k: int = 5) -> List[Dict]:
    """
    Mock: FAISS vector similarity search over the labeled clause corpus.
    Production: all-MiniLM-L6-v2 embeddings → FAISS ANN lookup.
    Returns top-K similar clauses with {clause_text, clause_type, risk_level, risk_reason, similarity}.

    Edge cases:
      - Empty / too-short clause text  → returns [] (no match possible)
      - k=0                            → returns []
      - Fewer results than k available → returns what's available (graceful partial)
    """
    if not clause_text or len(clause_text.strip()) < 10:
        logger.warning("precedent_search: clause text too short for meaningful similarity search. Returning [].")
        return []

    if k <= 0:
        logger.warning("precedent_search: k=%d is invalid. Returning [].", k)
        return []

    # Mock: generate k synthetic precedents with decreasing similarity scores
    results = [
        SimilarClause(
            text=f"[MOCK precedent {i + 1}] Similar historical clause for context.",
            risk=["LOW", "MEDIUM", "HIGH"][i % 3],
            similarity=round(0.95 - (i * 0.07), 2),
        )
        for i in range(k)
    ]
    logger.debug("precedent_search: returning %d mock precedents (k=%d).", len(results), k)
    return results


def _mock_contract_search(document_id: str) -> List[Dict]:
    """
    Mock: Structured lookup of all clauses already extracted from the SAME contract.
    Production: direct dict/DB lookup keyed by document_id — no embeddings, no LLM.
    Purpose: resolve cross-references (e.g., "was IP already assigned in another clause?").

    Edge cases:
      - document_id == "unknown" or empty → returns [] (doc not found)
      - Normal id                         → returns list of sibling clause dicts
    """
    if not document_id or document_id in ("unknown", ""):
        logger.warning("contract_search: unknown document_id '%s'. Returning [].", document_id)
        return []

    # Mock: synthetic sibling clauses from the same hypothetical contract
    siblings = [
        {"clause_type": "Governing Law",     "clause_text": "This Agreement shall be governed by the laws of Delaware.", "risk_level": "LOW"},
        {"clause_type": "Termination",       "clause_text": "Either party may terminate upon 30 days written notice.",  "risk_level": "MEDIUM"},
        {"clause_type": "IP Ownership",      "clause_text": "All IP developed hereunder is assigned to the Company.",   "risk_level": "HIGH"},
        {"clause_type": "Indemnification",   "clause_text": "Each party shall indemnify the other for third-party claims.", "risk_level": "HIGH"},
    ]
    logger.debug("contract_search: returning %d sibling clauses for doc '%s...'.", len(siblings), document_id[:8])
    return siblings


# ==========================================
# FAST PATH  (conf >= 0.6)
# Deterministic, ~1 LLM call, no tool loop.
# DeBERTa's label is accepted unchanged.
# ==========================================

def _fast_path(c, level: str, confidence: float) -> Dict:
    """
    High-confidence explanation path.
    Step 1: Retrieve 3 precedents from FAISS for context enrichment.
    Step 2: Single Mistral forward pass → explanation grounded in clause + precedents.
    Returns agent_trace=[] (no tool loop ran).

    Edge case: FAISS returns 0 results — still proceeds, just without RAG context.
    """
    print(f"⚡ [FAST PATH] conf={confidence:.2f} >= 0.6 | {c.clause_type} | Fetching 3 precedents...")

    # Step 1: FAISS retrieval (mocked)
    precedents = _mock_precedent_search(c.clause_text, k=3)
    context_note = (
        f"({len(precedents)} precedent(s) retrieved)" if precedents
        else "(no precedents found — proceeding without RAG context)"
    )

    # Step 2: Single Mistral call (mocked)
    print(f"   ✅ [FAST PATH] Generating explanation {context_note}...")
    explanation = (
        f"[FAST PATH] DeBERTa classified '{c.clause_type}' as {level} risk "
        f"(conf={confidence:.2f}). {context_note}."
    )

    return {
        "risk_level": level,
        "risk_reason": explanation,
        "similar_clauses": precedents,
        "agent_trace": [],       # No tool calls in the fast path
        "overridden": False,
    }


# ==========================================
# AGENT PATH  (conf < 0.6)
# Non-deterministic, 2–5 LLM calls.
# ReAct tool loop — Mistral decides which tools to call and when to stop.
# May override DeBERTa's preliminary label.
# ==========================================

def _agent_path(c, deberta_label: str, confidence: float, doc_id: str) -> Dict:
    """
    Low-confidence agentic path.
    Mistral is invoked as a full ReAct tool-calling agent (max 5 turns).
    Tools: precedent_search (vector RAG) + contract_search (structured same-doc lookup).
    Mistral may produce a final_label that overrides DeBERTa's preliminary label.

    Edge cases:
      - precedent_search returns []    → agent logs warning, continues with contract_search
      - contract_search returns []     → agent uses only precedent evidence
      - Max 5 turns reached            → uses best available label with a warning note
      - Both tools return []           → agent falls back to DeBERTa label with low-confidence note
    """
    print(f"🤖 [AGENT PATH] conf={confidence:.2f} < 0.6 | {c.clause_type} | Starting ReAct loop...")

    tool_trace = []
    final_label = deberta_label
    overridden = False
    MAX_TURNS = 5
    has_precedents = False
    has_siblings = False

    for turn in range(1, MAX_TURNS + 1):
        print(f"   [Turn {turn}/{MAX_TURNS}] Mistral reasoning...")

        # Turn 1: Always call precedent_search first for corpus-wide evidence
        if turn == 1:
            print(f"   [Turn {turn}] → precedent_search(k=5)...")
            results = _mock_precedent_search(c.clause_text, k=5)
            tool_trace.append({"tool": "precedent_search", "turn": turn, "results_count": len(results)})
            has_precedents = len(results) > 0

            if not has_precedents:
                print(f"   [Turn {turn}] ⚠️  No FAISS precedents found. Will rely on contract context.")

        # Turn 2: Call contract_search for same-document cross-references
        elif turn == 2:
            print(f"   [Turn {turn}] → contract_search(doc_id={doc_id[:8]}...)...")
            siblings = _mock_contract_search(doc_id)
            tool_trace.append({"tool": "contract_search", "turn": turn, "results_count": len(siblings)})
            has_siblings = len(siblings) > 0

            # Edge case: both tools returned nothing → fall back with warning
            if not has_precedents and not has_siblings:
                print(f"   [Turn {turn}] ⚠️  No evidence from either tool. Keeping DeBERTa label.")
                break

            # Mock: agent may override the label based on sibling clause context.
            # Production: Mistral reads tool results and outputs structured JSON with final_label.
            if deberta_label == "HIGH" and has_siblings:
                final_label = "MEDIUM"
                overridden = True
                print(f"   [Turn {turn}] 🔄 Agent overrides DeBERTa: HIGH → MEDIUM (sibling clauses suggest lower risk scope).")

        # Turn 3: Agent has gathered enough evidence — finalises
        else:
            print(f"   [Turn {turn}] ✅ Evidence gathered. Producing final structured output...")
            break

    # Edge case: hit max turns without reaching a conclusion
    if len(tool_trace) >= MAX_TURNS:
        print(f"   ⚠️  [AGENT PATH] Max turns ({MAX_TURNS}) reached. Returning best-available label.")
        explanation = (
            f"[AGENT PATH - MAX TURNS REACHED] '{c.clause_type}' assessed after {len(tool_trace)} tool call(s). "
            f"Label: {final_label} (low confidence, manual review recommended)."
        )
    else:
        explanation = (
            f"[AGENT PATH - {len(tool_trace)} tool call(s)] "
            f"Mistral assessed '{c.clause_type}' via ReAct loop. "
            f"Final label: {final_label}"
            + (" [DeBERTa overridden — see agent_trace]" if overridden else " [DeBERTa confirmed].")
        )

    return {
        "risk_level": final_label,
        "risk_reason": explanation,
        "similar_clauses": [],       # Populated from tool results in production
        "agent_trace": tool_trace,
        "overridden": overridden,
    }


# ==========================================
# STAGE 3 MAIN GRAPH NODES
# ==========================================

def node_risk_classifier(state: RiskAnalysisState):
    """
    STAGE 3 (Part 1 — Risk Scoring) — Node C (Dispatcher)
    Assigns a preliminary risk level and classifier_confidence to each extracted clause.
    These are then fanned-out to parallel Node D workers via the LangGraph Send API.

    Mock: cycles HIGH/MEDIUM/LOW and alternates confidence 0.95 / 0.30
          to exercise both Fast Path and Agent Path on every API call.

    Production: Replace with fine-tuned DeBERTa sequence classifier output.
    """
    clauses = state.get("extracted_clauses", [])
    print(f"🔄 [NODE C | Stage 3 - Classifier] Scoring {len(clauses)} clauses...")

    risk_cycle = ["HIGH", "MEDIUM", "LOW"]
    conf_cycle = [0.95, 0.30]  # 0.95 → Fast Path, 0.30 → Agent Path

    flagged = []
    for i, c in enumerate(clauses):
        flagged.append({
            "clause": c,
            "risk_level": risk_cycle[i % 3],
            "classifier_confidence": conf_cycle[i % 2],
        })

    print(f"✅ [NODE C] Dispatching {len(flagged)} clause(s) to parallel workers.")
    return {"flagged_clauses": flagged}


def node_mistral_router(state: RAGWorkerState):
    """
    STAGE 3 (Part 2 — RAG Reasoning) — Node D (WORKER)
    Processes exactly ONE clause per invocation. Runs in parallel per clause via Map-Reduce.

    Fast Path  (conf >= 0.6): precedent_search(k=3) + 1 Mistral call (direct)
    Agent Path (conf <  0.6): ReAct tool loop (max 5 turns)
                               Tools: precedent_search + contract_search
                               May produce a different final_label than DeBERTa's.

    Output is appended to risk_assessed_clauses via the operator.add reducer.
    """
    item = state.get("clause_data", {})
    c = item["clause"]
    level = item["risk_level"]
    confidence = item["classifier_confidence"]

    # Extract document UUID prefix for contract_search
    doc_id = getattr(c, "clause_id", "unknown").split("_")[0] if "_" in getattr(c, "clause_id", "") else "unknown"

    THRESHOLD = 0.6
    if confidence >= THRESHOLD:
        result = _fast_path(c, level, confidence)
    else:
        result = _agent_path(c, level, confidence, doc_id)

    assessed = RiskAssessedClause(
        clause_id=c.clause_id,
        clause_text=c.clause_text,
        clause_type=c.clause_type,
        start_pos=c.start_pos,
        end_pos=c.end_pos,
        confidence=c.confidence,
        confidence_logit=c.confidence_logit,
        page_no=c.page_no,
        content_label=c.content_label,
        risk_level=result["risk_level"],
        risk_reason=result["risk_reason"],
        similar_clauses=result["similar_clauses"],
        cross_references=[t["tool"] for t in result["agent_trace"]],  # Surface which tools ran
        overridden=result["overridden"],
    )

    return {"risk_assessed_clauses": [assessed]}
