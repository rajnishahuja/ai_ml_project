"""
Stage 3: Risk Classification + Agentic Explanation Nodes.

Two-path architecture (per ARCHITECTURE.md):
  - Fast Path  (conf >= 0.6): precedent_search(k=3) + 1 Mistral call (direct, no loop)
  - Agent Path (conf <  0.6): ReAct tool loop (max 5 turns)
                               Tools: precedent_search + contract_search
                               Mistral may override DeBERTa's preliminary label.

`contract_search` design:
  - Bound to the contract's full clause set ONCE per worker via
    `make_contract_search(extracted_clauses)`. Mistral's tool call carries
    only `(document_id, clause_type)` — no clause text or list payloads
    cross the LLM boundary.
  - Metadata (Parties, Effective Date, Expiration Date) is injected
    directly into Mistral's prompt via `extract_metadata_block(...)` —
    not fetched through the tool. This is because Parties is the #1
    driver of label flips and must be available on turn 1.

Both paths are currently MOCKED with deterministic stubs.
Replace the _mock_* functions and LLM calls once:
  1. DeBERTa risk classifier training completes (Node C)
  2. Ollama + FAISS are connected (Node D)
"""

import logging
from typing import TypedDict, Any, Callable, Dict, List

from app.schemas.domain import RiskAssessedClause, SimilarClause
from src.stage3_risk_agent.tools import (
    extract_metadata_block,
    make_contract_search,
)
from src.workflow.state import RiskAnalysisState

logger = logging.getLogger(__name__)


# ==========================================
# STATE DEFINITIONS
# ==========================================

class RAGWorkerState(TypedDict):
    """State for the per-clause Map-Reduce worker (Node D).

    The dispatcher (`continue_to_mistral` in `src/workflow/graph.py`) packs
    every fan-out branch with these three fields:

      - clause_data:        the single flagged clause this worker assesses
      - extracted_clauses:  ALL clauses for the contract (used to bind
                            contract_search to the full contract context)
      - document_id:        the contract identifier (sanity arg for tools)
      - metadata_block:     {Parties, Effective Date, Expiration Date, ...}
                            for direct injection into Mistral's prompt
    """
    clause_data: Dict[str, Any]
    extracted_clauses: List[Any]
    document_id: str
    metadata_block: Dict[str, str]


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


# NOTE: `contract_search` is no longer mocked here — it's bound to the
# real contract data per worker via `make_contract_search(extracted_clauses)`.
# The bound tool (signature: `(document_id, clause_type) -> list[dict]`)
# drives the agent path's same-document lookups via the static
# clause_type_relations.json map.


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

def _agent_path(
    c,
    deberta_label: str,
    confidence: float,
    doc_id: str,
    bound_contract_search: Callable[[str, str], List[Dict]],
    metadata_block: Dict[str, str],
) -> Dict:
    """
    Low-confidence agentic path.
    Mistral is invoked as a full ReAct tool-calling agent (max 5 turns).
    Tools: precedent_search (vector RAG) + contract_search (structured same-doc lookup,
           bound to the contract's clauses via closure — Mistral passes only
           document_id + clause_type).
    Mistral may produce a final_label that overrides DeBERTa's preliminary label.

    Args:
        c:                     The single clause being assessed.
        deberta_label:         DeBERTa's preliminary risk label.
        confidence:            DeBERTa's confidence (already known < 0.6 here).
        doc_id:                Contract identifier — passed through to the
                               bound contract_search call as a sanity arg.
        bound_contract_search: Closure produced by `make_contract_search(...)`.
                               Has the contract's full clause list captured.
        metadata_block:        {Parties, Effective Date, ...} for prompt
                               injection. Provided to Mistral every turn so
                               signing-party direction is available on turn 1.

    Edge cases:
      - precedent_search returns []    → agent logs warning, continues with contract_search
      - contract_search returns []     → agent uses only precedent evidence
      - Max 5 turns reached            → uses best available label with a warning note
      - Both tools return []           → agent falls back to DeBERTa label with low-confidence note
    """
    print(f"🤖 [AGENT PATH] conf={confidence:.2f} < 0.6 | {c.clause_type} | Starting ReAct loop...")
    if metadata_block:
        print(f"   [Prompt] METADATA injected: {list(metadata_block.keys())}")

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

        # Turn 2: Call contract_search — bound closure, Mistral passes only IDs.
        elif turn == 2:
            print(f"   [Turn {turn}] → contract_search(doc_id={doc_id[:8]}..., clause_type={c.clause_type!r})...")
            siblings = bound_contract_search(doc_id, c.clause_type)
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

    Side effect: extracts the contract-level METADATA block (Parties, Dates, ...)
    once and writes it to state.metadata_block. The fan-out dispatcher
    (`continue_to_mistral` in graph.py) packs it into every worker's payload
    along with the full extracted_clauses list, so each worker can bind
    contract_search to the contract's clauses without re-extracting.

    Mock: cycles HIGH/MEDIUM/LOW and alternates confidence 0.95 / 0.30
          to exercise both Fast Path and Agent Path on every API call.

    Production: Replace with fine-tuned DeBERTa sequence classifier output.
    """
    clauses = state.get("extracted_clauses", [])
    print(f"🔄 [NODE C | Stage 3 - Classifier] Scoring {len(clauses)} clauses...")

    # Build the metadata block once per contract — reused across all workers.
    metadata_block = extract_metadata_block(clauses)
    if metadata_block:
        print(f"   [Metadata] Extracted {len(metadata_block)} field(s): {list(metadata_block.keys())}")

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
    return {
        "flagged_clauses": flagged,
        "metadata_block": metadata_block,
    }


def node_mistral_router(state: RAGWorkerState):
    """
    STAGE 3 (Part 2 — RAG Reasoning) — Node D (WORKER)
    Processes exactly ONE clause per invocation. Runs in parallel per clause via Map-Reduce.

    Fast Path  (conf >= 0.6): precedent_search(k=3) + 1 Mistral call (direct)
    Agent Path (conf <  0.6): ReAct tool loop (max 5 turns)
                               Tools: precedent_search + contract_search (bound)
                               May produce a different final_label than DeBERTa's.

    Tool binding:
      `contract_search` is built per-worker from the contract's full clause
      list (provided in `state["extracted_clauses"]`). Mistral only ever
      passes `(document_id, clause_type)` to it — the clause data lives in
      the closure, never on the LLM tape.

    Output is appended to risk_assessed_clauses via the operator.add reducer.
    """
    item = state.get("clause_data", {})
    c = item["clause"]
    level = item["risk_level"]
    confidence = item["classifier_confidence"]

    # Contract-level context provided by the dispatcher's Send payload.
    extracted_clauses = state.get("extracted_clauses", []) or []
    metadata_block = state.get("metadata_block", {}) or {}
    doc_id = state.get("document_id") or ""

    # Bind the tool ONCE per worker. The closure captures all clauses for
    # this contract; Mistral will call it via `(document_id, clause_type)`.
    bound_contract_search = make_contract_search(extracted_clauses)

    THRESHOLD = 0.6
    if confidence >= THRESHOLD:
        result = _fast_path(c, level, confidence)
    else:
        result = _agent_path(
            c, level, confidence, doc_id,
            bound_contract_search=bound_contract_search,
            metadata_block=metadata_block,
        )

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
