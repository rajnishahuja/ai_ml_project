"""
Stage 3 agent — main entry point for risk assessment.

assess_clauses() is called by run_pipeline.py with all ClauseObjects from
Stage 1+2 for a single contract. Each risk-relevant clause is routed through
DeBERTa, then either:

  Fast path  (conf >= threshold): single LLM call with precedent context.
  Agent path (conf <  threshold): LangGraph ReAct loop with tool access.
                                   May override DeBERTa's preliminary label.
"""

import json
import logging
from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage
from src.common.utils import make_llm
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.common.schema import AgentTraceEntry, ClauseObject, RiskAssessedClause
from src.common.utils import load_config
from src.stage3_risk_agent.embeddings import query_index
from src.stage3_risk_agent.risk_classifier import (
    RiskClassifier,
    extract_signing_party,
)
from src.stage3_risk_agent.tools import (
    make_contract_search_tool,
    make_deberta_classify_tool,
    make_precedent_search_tool,
)

logger = logging.getLogger(__name__)

# Clause types that carry contract metadata — not risk-assessed.
# They route to the Stage 4 report header instead.
METADATA_CLAUSE_TYPES = {
    "Document Name", "Parties", "Agreement Date",
    "Effective Date", "Expiration Date",
}

# Default HF Hub model IDs — overridable via assess_clauses() args for local dev.
CE_MODEL_ID   = "rajnishahuja/cuad-risk-deberta-ce-parties"
CORN_MODEL_ID = "rajnishahuja/cuad-risk-deberta-corn-parties"

# CUAD labeling convention + worked examples — injected into the synthesis prompt
# so the LLM's "HIGH/MEDIUM/LOW" scale matches what DeBERTa was trained on.
# "Signing party" = the party reviewing the contract for risk (counterparty to drafter).
_CUAD_CONVENTION = """\
CUAD risk scale (from the signing party's perspective):
  HIGH   — clause significantly disadvantages the signing party: one-sided IP transfer,
            unlimited/uncapped liability, no termination rights, severe penalties.
  MEDIUM — clause creates meaningful but bounded risk: capped liability, time-limited
            restrictions, mutual obligations with some asymmetry, conditional rights.
  LOW    — clause is standard, balanced, or net-positive for the signing party.

Examples:

[Ip Ownership Assignment | signing party = OntoChem (assignor)]
Text: "Anixa will own, and OntoChem hereby assigns to Anixa, all right, title and
interest in and to all Inventions other than OntoChem Inventions."
→ HIGH — signing party irrevocably transfers IP rights with no compensation carve-out.

[Anti-Assignment | mutual]
Text: "Neither party may assign this Agreement or subcontract its obligations under
this Agreement to another party without the other party's prior, written consent."
→ MEDIUM — mutual restriction limits both parties equally; risk is real but bounded
  and reciprocal, not one-sided.

[Governing Law | standard]
Text: "This Agreement will be governed and construed in accordance with the laws of
the State of California without giving effect to conflict of laws principles."
→ LOW — standard choice-of-law clause; no unusual jurisdiction burden on signing party.

"""


# ---------------------------------------------------------------------------
# Structured output schema — used by both paths
# ---------------------------------------------------------------------------

class RiskAssessment(BaseModel):
    final_label: str = Field(
        description="Risk level for the signing party: LOW, MEDIUM, or HIGH."
    )
    explanation: str = Field(
        description=(
            "1-3 sentence explanation of the risk level, grounded in the clause "
            "text and any evidence gathered from tools."
        )
    )
    override_reason: str = Field(
        default="",
        description=(
            "If your final_label differs from DeBERTa's preliminary label, "
            "explain why. Leave empty when confirming DeBERTa."
        ),
    )


# ---------------------------------------------------------------------------
# Fast path  (conf >= threshold)
# ---------------------------------------------------------------------------

def _fast_path(
    clause: ClauseObject,
    deberta_result: dict,
    signing_party: str,
    index_path: str,
    llm,
    k: int,
) -> RiskAssessedClause:
    # Fast path is dead code — confidence_threshold: 0 routes all clauses to agent.
    # Kept for reference; no FAISS call here (RAG lives in the agent's precedent_search tool).
    prompt = (
        f"You are a legal risk assessor.\n\n"
        f"Clause under review:\n"
        f"  Type:    {clause.clause_type}\n"
        f"  Parties: {signing_party or 'unknown'}\n"
        f"  Text:    {clause.clause_text}\n\n"
        f"DeBERTa's label: {deberta_result['label']} "
        f"({deberta_result['confidence']:.0%} confidence).\n\n"
        f"Determine the correct risk level (LOW / MEDIUM / HIGH) for the signing party."
    )

    result: RiskAssessment = llm.with_structured_output(
        RiskAssessment, method="function_calling"
    ).invoke(prompt)

    if result is None:
        logger.warning("fast_path: structured output returned None, falling back to DeBERTa label")
        result = RiskAssessment(
            final_label=deberta_result["label"],
            explanation="(LLM structured output failed — DeBERTa label used as fallback)",
        )

    return RiskAssessedClause(
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        clause_text=clause.clause_text,
        clause_type=clause.clause_type,
        risk_level=result.final_label,
        risk_explanation=result.explanation,
        similar_clauses=[],
        cross_references=[],
        confidence=deberta_result["confidence"],
        agent_trace=[],
    )


# ---------------------------------------------------------------------------
# Agent path  (conf < threshold)
# ---------------------------------------------------------------------------

def _agent_path(
    clause: ClauseObject,
    deberta_result: dict,
    signing_party: str,
    all_clauses: list[ClauseObject],
    index_path: str,
    llm,
    max_iterations: int,
    k: int,
    classifier=None,
    use_contract_search: bool = True,
) -> RiskAssessedClause:
    precedent_search = make_precedent_search_tool(index_path)
    tools = [precedent_search]
    if classifier is not None:
        tools.append(make_deberta_classify_tool(classifier, signing_party))
    if use_contract_search:
        tools.append(make_contract_search_tool(all_clauses))

    deberta_tool_guideline = (
        "- Call deberta_classify(clause_text, clause_type) if precedent similarity "
        "scores are low (below 0.75) or votes are split — it is a model trained on "
        "3,400 labeled CUAD clauses and is useful as a tiebreaker.\n"
        "  Interpreting deberta_classify results:\n"
        "    * confidence > 0.75 + agrees with precedent majority → strong signal, trust it\n"
        "    * confidence > 0.75 + disagrees with precedents → genuine conflict, call contract_search\n"
        "    * confidence < 0.60 → DeBERTa itself is uncertain, weight it lightly\n"
        if classifier is not None else ""
    )
    contract_search_guideline = (
        f"- Call contract_search(current_clause_id='{clause.clause_id}') when party "
        "roles or cross-clause context could change the assessment — especially for "
        "IP Ownership, License Grant, Affiliate License, and Assignment clause types "
        "where who signed determines the risk direction.\n"
        if use_contract_search else ""
    )
    system_prompt = (
        "You are a legal risk assessor using the CUAD risk scale:\n"
        "  HIGH   — one-sided IP transfer, uncapped liability, no termination rights.\n"
        "  MEDIUM — bounded risk: capped liability, mutual restrictions, conditional rights.\n"
        "  LOW    — standard, balanced, or net-positive for the signing party.\n\n"
        "Use the available tools to gather evidence and produce a well-grounded assessment.\n\n"
        "Evidence-gathering strategy:\n"
        "1. Always start with precedent_search (k=5).\n"
        "2. Evaluate what you got: precedent_search only returns clauses with "
        "similarity >= 0.75, so every result is a strong match. If it returns 0 "
        "results, no good precedents exist — skip to step 3 immediately.\n"
        "   If results are returned, check vote distribution: 4+ unanimous → conclude.\n"
        "3. If precedents are absent or votes are split, gather more evidence:\n"
        f"{deberta_tool_guideline}"
        f"{contract_search_guideline}"
        "4. Weigh all evidence. If multiple tools agree, that convergence is meaningful. "
        "If they conflict, explain which evidence you find more compelling and why.\n\n"
        f"Clause under review:\n"
        f"  clause_id:  {clause.clause_id}\n"
        f"  Type:       {clause.clause_type}\n"
        f"  Parties:    {signing_party or 'unknown'}\n"
        f"  Text:       {clause.clause_text}\n"
    )

    # Don't use response_format= here: langchain-openai 1.2.1 defaults to
    # method="json_schema" which calls openai's .parse() endpoint — llama.cpp
    # rejects that with 400. We do one explicit function_calling structured call
    # after the ReAct loop instead.
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
    )

    state = agent.invoke(
        {"messages": [{"role": "user", "content": "Assess the risk level for the clause described above."}]},
        config={"recursion_limit": max_iterations * 2 + 4},
    )

    # Build agent trace from ToolMessage entries in the message history
    trace = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                count = len(content) if isinstance(content, list) else None
            except (json.JSONDecodeError, TypeError):
                count = None
            trace.append(AgentTraceEntry(tool=msg.name, result_count=count))

    # Extract the agent's final reasoning text (last AI message with no tool calls).
    # Passing the full ReAct history to with_structured_output causes the model to
    # return plain text (it sees existing tool_call messages and doesn't re-call);
    # a clean single-turn prompt reliably triggers the function_calling tool fill.
    agent_conclusion = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", []):
            agent_conclusion = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    synthesis_prompt = (
        _CUAD_CONVENTION
        + f"Now assess the following clause:\n\n"
        f"Clause type: {clause.clause_type}\n"
        f"Parties: {signing_party or 'unknown'}\n"
        f"Text: {clause.clause_text}\n\n"
        f"Agent analysis:\n{agent_conclusion}\n\n"
        f"Produce the final structured risk assessment."
    )
    result: RiskAssessment = llm.with_structured_output(
        RiskAssessment, method="function_calling"
    ).invoke(synthesis_prompt)

    if result is None:
        logger.warning("agent_path: structured output returned None, falling back to DeBERTa label")
        result = RiskAssessment(
            final_label=deberta_result["label"],
            explanation="(LLM structured output failed — DeBERTa label used as fallback)",
        )

    return RiskAssessedClause(
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        clause_text=clause.clause_text,
        clause_type=clause.clause_type,
        risk_level=result.final_label,
        risk_explanation=result.explanation,
        similar_clauses=[],
        cross_references=[],
        confidence=deberta_result["confidence"],
        agent_trace=trace,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assess_clauses(
    clauses: list[ClauseObject],
    config_path: str = "configs/stage3_config.yaml",
    ce_model_path: Optional[str] = None,
    corn_model_path: Optional[str] = None,
    use_contract_search: bool = True,
    skip_ids: Optional[set] = None,
    checkpoint_file: Optional[str] = None,
) -> list[RiskAssessedClause]:
    """Assess risk for all risk-relevant clauses from a single contract.

    Args:
        clauses:              All ClauseObjects from Stage 1+2 for this contract.
        config_path:          Path to stage3_config.yaml.
        ce_model_path:        Local path to CE model (defaults to HF Hub).
        corn_model_path:      Local path to CORN model (defaults to HF Hub).
        use_contract_search:  If False, agent path uses precedent_search only.
        skip_ids:             Clause IDs to skip (already processed in a prior run).
        checkpoint_file:      Path to JSONL file; each result is appended immediately
                              so an interrupted run can be resumed via skip_ids.

    Returns:
        List of RiskAssessedClause — one per risk-relevant clause.
        Metadata clause types (Parties, dates, etc.) are excluded.
    """
    cfg = load_config(config_path)

    index_path  = cfg["faiss_index_path"]
    threshold   = cfg["confidence_threshold"]
    k_high      = cfg["similarity_top_k_high_conf"]
    k_low       = cfg["similarity_top_k_low_conf"]
    max_iter    = cfg["agent_max_iterations"]

    llm = make_llm(cfg)

    logger.info("Loading DeBERTa risk classifier (Ens-F) ...")
    classifier = RiskClassifier(
        ce_model_path=ce_model_path or CE_MODEL_ID,
        corn_model_path=corn_model_path or CORN_MODEL_ID,
    )

    # Resolve signing party once per document (O(n) not O(n²))
    doc_ids = {c.document_id for c in clauses}
    signing_parties = {
        doc_id: extract_signing_party(doc_id, clauses)
        for doc_id in doc_ids
    }

    _skip = skip_ids or set()
    results: list[RiskAssessedClause] = []
    risk_clauses = [c for c in clauses if c.clause_type not in METADATA_CLAUSE_TYPES]
    skipped_meta  = len(clauses) - len(risk_clauses)
    skipped_done  = sum(1 for c in risk_clauses if c.clause_id in _skip)
    pending       = [c for c in risk_clauses if c.clause_id not in _skip]
    logger.info(
        "Assessing %d clauses (%d metadata skipped, %d already done)",
        len(pending), skipped_meta, skipped_done,
    )

    ckpt_fh = open(checkpoint_file, "a") if checkpoint_file else None

    for i, clause in enumerate(pending, 1):
        signing_party  = signing_parties[clause.document_id]
        deberta_result = classifier.predict(
            clause_text=clause.clause_text,
            clause_type=clause.clause_type,
            signing_party=signing_party,
        )

        path = "fast" if deberta_result["confidence"] >= threshold else "agent"
        logger.info(
            "[%d/%d] %s | DeBERTa=%s (%.2f) → %s path",
            i, len(pending), clause.clause_type,
            deberta_result["label"], deberta_result["confidence"], path,
        )

        if path == "fast":
            assessed = _fast_path(
                clause, deberta_result, signing_party, index_path, llm, k_high
            )
        else:
            assessed = _agent_path(
                clause, deberta_result, signing_party, clauses,
                index_path, llm, max_iter, k_low,
                classifier=classifier,
                use_contract_search=use_contract_search,
            )

        results.append(assessed)
        if ckpt_fh:
            ckpt_fh.write(json.dumps({
                "clause_id":   assessed.clause_id,
                "clause_type": assessed.clause_type,
                "risk_level":  assessed.risk_level,
                "confidence":  assessed.confidence,
                "agent_trace": [{"tool": t.tool, "result_count": t.result_count}
                                for t in assessed.agent_trace],
                "explanation": assessed.risk_explanation,
            }) + "\n")
            ckpt_fh.flush()

    if ckpt_fh:
        ckpt_fh.close()

    return results
