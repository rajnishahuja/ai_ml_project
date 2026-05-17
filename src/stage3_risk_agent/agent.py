"""
Stage 3 agent — main entry point for risk assessment.

Architecture:
  Every clause goes through the LangGraph ReAct agent — no confidence-gating split.
  DeBERTa's label is passed into the agent's system prompt as the default signal.
  The agent calls precedent_search and contract_search to verify or challenge it,
  but is instructed to override DeBERTa only when tools provide clear consensus.

Decision guideline injected into the system prompt:
  - Tool evidence agrees with DeBERTa, or is weak/split  → confirm DeBERTa's label.
  - Strong consensus from BOTH tools contradicts DeBERTa → may override, must explain.
  - Never override based on solo legal reasoning — only on convergent tool evidence.
"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import os
import sqlite3
from tqdm import tqdm

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from src.common.schema import (
    AgentTraceEntry,
    ClauseObject,
    RiskAssessedClause,
    SimilarClause,
)
from src.common.utils import load_config, make_llm
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
METADATA_CLAUSE_TYPES = {
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
}

# Default HF Hub model IDs.
CE_MODEL_ID = "rajnishahuja/cuad-risk-deberta-ce-parties"
CORN_MODEL_ID = "rajnishahuja/cuad-risk-deberta-corn-parties"

# CUAD labeling convention + worked examples — injected into the synthesis prompt
# so the LLM's HIGH/MEDIUM/LOW scale matches what DeBERTa was trained on.
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
# Structured output schema
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
            "explain why the tool evidence outweighs the DeBERTa signal. "
            "Leave empty when confirming DeBERTa."
        ),
    )
    agent_confidence: float = Field(
        default=0.0,
        description=(
            "Your confidence score in the final risk assessment as a decimal fraction strictly between 0.0 and 1.0 (e.g. 0.95). "
            "Do NOT write it as a percentage out of 100 (e.g. do NOT write 95.0 or 95). "
            "If you override DeBERTa based on strong precedent search matches or sibling clauses, "
            "provide a high score (0.8 - 1.0) explaining the strong consensus. "
            "If evidence is split or inconclusive, provide a lower score (0.5 - 0.7)."
        ),
    )


# ---------------------------------------------------------------------------
# Agent path — runs for every clause
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
    use_contract_search: bool = True,
    free_override: bool = False,
    checkpointer: Optional[SqliteSaver] = None,
    embedding_model: str | None = None,
    sim_threshold: float = 0.75,
) -> RiskAssessedClause:
    precedent_search = make_precedent_search_tool(
        index_path, model_name=embedding_model, default_min_similarity=sim_threshold
    )
    tools = [precedent_search]
    if use_contract_search:
        tools.append(make_contract_search_tool(all_clauses))

    contract_search_guideline = (
        f"- Call contract_search(current_clause_id='{clause.clause_id}') when party "
        "roles or cross-clause context could change the assessment — especially for "
        "IP Ownership, License Grant, Affiliate License, and Assignment clause types "
        "where who signed determines the risk direction.\n"
        if use_contract_search
        else ""
    )

    if free_override:
        decision_rule = (
            "3. Decision rule:\n"
            "   Use all evidence — DeBERTa's label, precedent patterns, and contract context "
            "— together with your own legal reasoning to reach the best assessment. "
            "DeBERTa is a strong signal but you may override it whenever your analysis "
            "suggests a different label.\n\n"
        )
    else:
        decision_rule = (
            "3. Decision rule:\n"
            "   - Tool evidence agrees with DeBERTa, or is weak/mixed → keep DeBERTa's label.\n"
            "   - Strong consensus from tools contradicts DeBERTa → you may override, but "
            "you MUST explain exactly what evidence overrides the DeBERTa signal.\n"
            "   Do NOT override based on your own legal reasoning alone — only when tools "
            "provide clear, convergent evidence for a different label.\n\n"
        )

    system_prompt = (
        "You are a legal risk assessor using the CUAD risk scale:\n"
        "  HIGH   — one-sided IP transfer, uncapped liability, no termination rights.\n"
        "  MEDIUM — bounded risk: capped liability, mutual restrictions, conditional rights.\n"
        "  LOW    — standard, balanced, or net-positive for the signing party.\n\n"
        f"DeBERTa pre-classification: {deberta_result['label']} "
        f"({deberta_result['confidence']:.0%} confidence)\n"
        "DeBERTa was fine-tuned on 3,400 labeled CUAD clauses. "
        "Treat its label as the default — confirm it unless tool evidence clearly "
        "points elsewhere.\n\n"
        "Evidence-gathering strategy:\n"
        f"1. Always start with precedent_search (k=5). It only returns clauses with "
        f"similarity >= {sim_threshold:.2f}, so every result is a strong semantic match.\n"
        "2. Check vote distribution:\n"
        "   - 4+ results agreeing with DeBERTa → high confidence, confirm.\n"
        "   - 0 results → no precedent exists; keep DeBERTa's label.\n"
        "   - Votes split or pointing away from DeBERTa → gather more context.\n"
        f"{contract_search_guideline}"
        f"{decision_rule}"
        f"Clause under review:\n"
        f"  clause_id:  {clause.clause_id}\n"
        f"  Type:       {clause.clause_type}\n"
        f"  Parties:    {signing_party or 'unknown'}\n"
        f"  Text:       {clause.clause_text}\n"
    )

    agent = create_react_agent(
        llm, tools, prompt=system_prompt, checkpointer=checkpointer
    )

    # Use clause_id as the thread_id for granular persistence
    thread_id = f"thread_{clause.clause_id}"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_iterations * 2 + 4,
    }

    state = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Assess the risk level for the clause described above.",
                }
            ]
        },
        config=config,
    )

    # Build agent trace and extract evidence (from search tools)
    trace = []
    similar_clauses: list[SimilarClause] = []
    cross_references = []
    seen_texts = set()  # Global deduplication to prevent duplicate precedent clauses!

    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                if isinstance(msg.content, str):
                    import ast

                    try:
                        content = ast.literal_eval(msg.content)
                    except Exception:
                        content = json.loads(msg.content)
                else:
                    content = msg.content

                count = len(content) if isinstance(content, list) else 0
                trace.append(AgentTraceEntry(tool=msg.name, result_count=count))

                # Capture Precedents (from vector search) with global deduplication
                if msg.name == "precedent_search" and isinstance(content, list):
                    for item in content:
                        try:
                            if item["text"] not in seen_texts:
                                seen_texts.add(item["text"])
                                similar_clauses.append(
                                    SimilarClause(
                                        text=item["text"],
                                        clause_type=item["clause_type"],
                                        risk_level=item["risk_level"],
                                        similarity=item["similarity"],
                                    )
                                )
                        except (KeyError, TypeError):
                            pass

                # Capture Contract Lookups (sibling clauses)
                if msg.name == "contract_search" and isinstance(content, list):
                    cross_references.extend(content)

            except Exception as e:
                logger.error(f"Error parsing ToolMessage content: {e}")

    # Extract final reasoning text
    agent_conclusion = ""
    for msg in reversed(state["messages"]):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", [])
        ):
            agent_conclusion = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    synthesis_instruction = (
        "Make the best risk assessment using all available evidence: "
        "DeBERTa's classification, tool evidence gathered, and your own legal reasoning."
        if free_override
        else "Produce the final structured risk assessment. "
        "If the agent analysis does not provide strong tool-based evidence to "
        "contradict DeBERTa, use DeBERTa's label. "
        "Important: HIGH-risk clauses are under-represented in the precedent corpus, "
        "so if DeBERTa predicts HIGH and tools are split or inconclusive, ALWAYS confirm HIGH."
    )

    synthesis_prompt = (
        _CUAD_CONVENTION
        + f"You are a Senior Legal Analyst. Assess the following clause for the signing party ({signing_party or 'unknown'}):\n\n"
        f"Clause type: {clause.clause_type}\n"
        f"Full Text: {clause.clause_text}\n\n"
        f"Internal findings (Do not cite these directly):\n"
        f"- Research results: {agent_conclusion}\n\n"
        f"INSTRUCTION: {synthesis_instruction}"
    )

    try:
        result: RiskAssessment = llm.with_structured_output(RiskAssessment).invoke(
            synthesis_prompt
        )
    except Exception:
        # Attempt fallback if primary fails
        try:
            fallback_prompt = (
                synthesis_prompt
                + "\n\nCRITICAL: Respond ONLY with a valid JSON object matching the schema."
            )
            result = llm.with_structured_output(
                RiskAssessment, method="json_mode"
            ).invoke(fallback_prompt)
        except Exception:
            result = None

    if result is None:
        result = RiskAssessment(
            final_label=deberta_result["label"],
            explanation=f"Analysis: {agent_conclusion[:200]}...",
        )

    is_override = (result.final_label.upper() != deberta_result["label"].upper())

    return RiskAssessedClause(
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        clause_text=clause.clause_text,
        clause_type=clause.clause_type,
        risk_level=result.final_label,
        risk_explanation=result.explanation,
        similar_clauses=similar_clauses,
        cross_references=cross_references,
        extraction_confidence=clause.confidence,
        classifier_confidence=deberta_result["confidence"],
        agent_confidence=result.agent_confidence / 100.0 if result.agent_confidence > 1.0 else result.agent_confidence,
        is_override=is_override,
        extraction_confidence_logit=getattr(clause, "confidence_logit", None),
        content_label=getattr(clause, "content_label", None),
        agent_trace=trace,
        page_no=clause.page_no,
        start_pos=clause.start_pos,
        end_pos=clause.end_pos,
        metadata={"content_label": getattr(clause, "content_label", None)},
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
    free_override: bool = False,
    persist_db_path: str = "data/checkpoints/agent_state.db",
) -> list[RiskAssessedClause]:
    """Assess risk for all risk-relevant clauses from a single contract.

    Args:
        clauses:              All ClauseObjects from Stage 1+2 for this contract.
        config_path:          Path to stage3_config.yaml.
        ce_model_path:        Local path to CE model (defaults to HF Hub).
        corn_model_path:      Local path to CORN model (defaults to HF Hub).
        use_contract_search:  If False, agent uses precedent_search only.
        skip_ids:             Clause IDs to skip (already processed in a prior run).
        checkpoint_file:      Path to JSONL file; each result is appended immediately.
        persist_db_path:      Path to SQLite database for LangGraph checkpointing.

    Returns:
        List of RiskAssessedClause — one per risk-relevant clause.
    """
    cfg = load_config(config_path)

    index_path = cfg["faiss_index_path"]
    embedding_model = cfg.get("embedding_model")
    sim_threshold = cfg.get("similarity_threshold", 0.75)
    k = cfg["similarity_top_k_low_conf"]
    max_iter = cfg["agent_max_iterations"]
    num_workers = cfg.get("agent_num_workers", 1)

    llm = make_llm(cfg)

    logger.info("Loading DeBERTa risk classifier (Ens-F) ...")
    classifier = RiskClassifier(
        ce_model_path=ce_model_path or CE_MODEL_ID,
        corn_model_path=corn_model_path or CORN_MODEL_ID,
        ce_only=(corn_model_path is None and ce_model_path is not None),
    )

    doc_ids = {c.document_id for c in clauses}
    signing_parties = {
        doc_id: extract_signing_party(doc_id, clauses) for doc_id in doc_ids
    }

    _skip = skip_ids or set()
    risk_clauses = [c for c in clauses if c.clause_type not in METADATA_CLAUSE_TYPES]
    skipped_meta = len(clauses) - len(risk_clauses)
    skipped_done = sum(1 for c in risk_clauses if c.clause_id in _skip)
    pending = [c for c in risk_clauses if c.clause_id not in _skip]
    logger.info(
        "Assessing %d clauses (%d metadata skipped, %d already done)",
        len(pending),
        skipped_meta,
        skipped_done,
    )

    # ------------------------------------------------------------------
    # NEW: Initialize Persistent Checkpointer
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(persist_db_path), exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(persist_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # ------------------------------------------------------------------
    # Phase 1 — DeBERTa batch (sequential: GPU inference is not thread-safe)
    # ------------------------------------------------------------------
    deberta_cache: dict[str, tuple[dict, str]] = {}
    for clause in pending:
        sp = signing_parties[clause.document_id]
        deberta_cache[clause.clause_id] = (
            classifier.predict(
                clause_text=clause.clause_text,
                clause_type=clause.clause_type,
                signing_party=sp,
            ),
            sp,
        )

    # ------------------------------------------------------------------
    # Phase 2 — Agent loop (parallel when num_workers > 1)
    # Each worker makes independent HTTP calls to the LLM server.
    # With num_workers == llama.cpp -np slot count, calls run truly in parallel.
    # With num_workers == 1 (default), behaviour is identical to the old loop.
    # ------------------------------------------------------------------
    ckpt_fh = open(checkpoint_file, "a") if checkpoint_file else None
    ckpt_lock = threading.Lock() if (num_workers > 1 and ckpt_fh) else None

    def _assess_one(idx: int, clause: ClauseObject) -> tuple[int, RiskAssessedClause]:
        deberta_result, signing_party = deberta_cache[clause.clause_id]
        logger.info(
            "[%d/%d] %s | DeBERTa=%s (%.2f) → agent (worker %s)",
            idx + 1,
            len(pending),
            clause.clause_type,
            deberta_result["label"],
            deberta_result["confidence"],
            threading.current_thread().name,
        )
        assessed = _agent_path(
            clause=clause,
            deberta_result=deberta_result,
            signing_party=signing_party,
            all_clauses=clauses,
            index_path=index_path,
            llm=llm,
            max_iterations=max_iter,
            k=k,
            use_contract_search=use_contract_search,
            free_override=free_override,
            checkpointer=checkpointer,
            embedding_model=embedding_model,
            sim_threshold=sim_threshold,
        )
        if ckpt_fh:
            entry = (
                json.dumps(
                    {
                        "clause_id": assessed.clause_id,
                        "clause_type": assessed.clause_type,
                        "risk_level": assessed.risk_level,
                        "confidence": assessed.confidence,
                        "agent_trace": [
                            {"tool": t.tool, "result_count": t.result_count}
                            for t in assessed.agent_trace
                        ],
                        "explanation": assessed.risk_explanation,
                        "similar_clauses": assessed.similar_clauses,
                        "cross_references": assessed.cross_references,
                    }
                )
                + "\n"
            )
            if ckpt_lock:
                with ckpt_lock:
                    ckpt_fh.write(entry)
                    ckpt_fh.flush()
            else:
                ckpt_fh.write(entry)
                ckpt_fh.flush()
        return idx, assessed

    results: list[RiskAssessedClause | None] = [None] * len(pending)

    if num_workers == 1:
        for idx, clause in tqdm(list(enumerate(pending)), desc="Assessing Clauses", leave=True):
            _, assessed = _assess_one(idx, clause)
            results[idx] = assessed
    else:
        logger.info("Parallel agent mode: %d workers", num_workers)
        with ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="agent"
        ) as pool:
            futures = {
                pool.submit(_assess_one, idx, clause): idx
                for idx, clause in enumerate(pending)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Assessing Clauses", leave=True):
                idx, assessed = future.result()
                results[idx] = assessed

    if ckpt_fh:
        ckpt_fh.close()

    conn.close()

    # Proactively free Stage 3 DeBERTa classifier model from RAM/VRAM
    del classifier
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()

    return [r for r in results if r is not None]
