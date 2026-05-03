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
from langchain_openai import ChatOpenAI
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
    llm: ChatOpenAI,
    k: int,
) -> RiskAssessedClause:
    similar = query_index(clause.clause_text, index_path, k)

    precedent_lines = "\n".join(
        f"  [{r.risk_level}] ({r.clause_type}, similarity={r.similarity:.2f}) "
        f"{r.text[:120]}"
        for r in similar
    ) or "  (none retrieved)"

    prompt = (
        f"You are a legal risk assessor.\n\n"
        f"DeBERTa classified this clause as {deberta_result['label']} risk "
        f"({deberta_result['confidence']:.0%} confidence).\n\n"
        f"Clause under review:\n"
        f"  Type:    {clause.clause_type}\n"
        f"  Parties: {signing_party or 'unknown'}\n"
        f"  Text:    {clause.clause_text}\n\n"
        f"Similar clauses from the labeled corpus:\n{precedent_lines}\n\n"
        f"Write a concise explanation (1-3 sentences) of why this clause is "
        f"{deberta_result['label']} risk, grounded in the clause text and precedents."
    )

    result: RiskAssessment = llm.with_structured_output(
        RiskAssessment, method="function_calling"
    ).invoke(prompt)

    return RiskAssessedClause(
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        clause_text=clause.clause_text,
        clause_type=clause.clause_type,
        risk_level=result.final_label,
        risk_explanation=result.explanation,
        similar_clauses=similar,
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
    llm: ChatOpenAI,
    max_iterations: int,
    k: int,
    use_contract_search: bool = True,
) -> RiskAssessedClause:
    precedent_search = make_precedent_search_tool(index_path)
    tools = [precedent_search]
    if use_contract_search:
        tools.append(make_contract_search_tool(all_clauses))

    contract_search_guideline = (
        "- Call contract_search only if precedent evidence is mixed or if knowing "
        "the full contract context (party roles, related clauses) would resolve "
        "the ambiguity.\n"
        if use_contract_search else ""
    )
    system_prompt = (
        "You are a legal risk assessor. DeBERTa classified the clause below with "
        "low confidence — use the available tools to gather evidence and produce "
        "a well-grounded final assessment.\n\n"
        "Tool usage guidelines:\n"
        "- Always call precedent_search first (k=5) to find similar labeled clauses.\n"
        f"{contract_search_guideline}"
        "- Base final_label on the weight of evidence, not just DeBERTa's label.\n\n"
        f"Clause under review:\n"
        f"  Type:               {clause.clause_type}\n"
        f"  Parties:            {signing_party or 'unknown'}\n"
        f"  Text:               {clause.clause_text}\n"
        f"  DeBERTa preliminary: {deberta_result['label']} "
        f"(confidence: {deberta_result['confidence']:.2f})"
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
        f"Clause under review:\n"
        f"  Type:    {clause.clause_type}\n"
        f"  Parties: {signing_party or 'unknown'}\n"
        f"  Text:    {clause.clause_text}\n\n"
        f"Agent analysis:\n{agent_conclusion}\n\n"
        f"Produce the final structured risk assessment."
    )
    result: RiskAssessment = llm.with_structured_output(
        RiskAssessment, method="function_calling"
    ).invoke(synthesis_prompt)

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

    llm = ChatOpenAI(
        model=cfg["agent_model"],
        base_url=cfg["agent_base_url"],
        api_key=cfg.get("agent_api_key", "none"),
        temperature=0,
    )

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
