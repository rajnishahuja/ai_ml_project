"""
Stage 3 agent — main entry point for risk assessment.

Architecture:
  Label decision:  DeBERTa + FAISS ensemble (deterministic, no LLM).
  Explanation:     Single LLM call with clause text + evidence summary.

Ensemble rule:
  1. DeBERTa predicts label + confidence.
  2. FAISS retrieves top-k clauses with similarity >= 0.75.
  3. If FAISS has >=3 strong matches AND majority agrees with DeBERTa
       → use that label (88.5% historical accuracy on this case).
     Otherwise → use DeBERTa label (more reliable when FAISS signal is weak,
       especially for HIGH-risk clauses where FAISS precision is only 25.8%).
  4. LLM writes a 1-3 sentence explanation using the evidence.
     It does NOT make the label decision.

Contract context (targeted):
  For clause types where party roles matter, CLAUSE_CONTEXT_TYPES maps each
  type to the related clause types worth fetching from the same contract.
  Only Parties + related types are passed — not all clauses.
"""

import logging
from collections import Counter
from typing import Optional

from src.common.schema import AgentTraceEntry, ClauseObject, RiskAssessedClause
from src.common.utils import load_config, make_llm
from src.stage3_risk_agent.embeddings import query_index
from src.stage3_risk_agent.risk_classifier import (
    RiskClassifier,
    extract_signing_party,
)

logger = logging.getLogger(__name__)

# Clause types that carry contract metadata — not risk-assessed.
METADATA_CLAUSE_TYPES = {
    "Document Name", "Parties", "Agreement Date",
    "Effective Date", "Expiration Date",
}

# Default HF Hub model IDs.
CE_MODEL_ID   = "rajnishahuja/cuad-risk-deberta-ce-parties"
CORN_MODEL_ID = "rajnishahuja/cuad-risk-deberta-corn-parties"

# For each clause type, which related types provide useful context for explanation.
# The Parties clause is always included when present.
# Types not listed here get only the Parties clause.
CLAUSE_CONTEXT_TYPES: dict[str, list[str]] = {
    "Ip Ownership Assignment":        ["License Grant", "Affiliate License-Licensor", "Work For Hire"],
    "Affiliate License-Licensee":     ["License Grant", "Ip Ownership Assignment", "Sublicense"],
    "Affiliate License-Licensor":     ["License Grant", "Ip Ownership Assignment"],
    "License Grant":                  ["Ip Ownership Assignment", "Exclusivity", "Sublicense"],
    "Sublicense":                     ["License Grant", "Ip Ownership Assignment"],
    "Non-Compete":                    ["Non-Solicitation", "Exclusivity", "Termination For Convenience"],
    "Non-Solicitation":               ["Non-Compete"],
    "Cap On Liability":               ["Uncapped Liability"],
    "Uncapped Liability":             ["Cap On Liability"],
    "Anti-Assignment":                ["Change Of Control"],
    "Change Of Control":              ["Anti-Assignment", "Termination For Convenience"],
    "Exclusivity":                    ["Non-Compete", "License Grant", "Minimum Commitment"],
    "Termination For Convenience":    ["Notice Period To Terminate Renewal", "Renewal Term"],
    "Liquidated Damages":             ["Cap On Liability", "Minimum Commitment"],
    "Minimum Commitment":             ["Liquidated Damages", "Renewal Term"],
    "Covenant Not To Sue":            ["License Grant", "Ip Ownership Assignment"],
}

FAISS_MIN_SIMILARITY = 0.75
FAISS_ENSEMBLE_MIN_VOTES = 3   # need >= this many strong matches to trust FAISS majority


# ---------------------------------------------------------------------------
# Ensemble label decision
# ---------------------------------------------------------------------------

def _ensemble_label(
    deberta_result: dict,
    faiss_results: list,   # list[SimilarClause] already filtered to min_similarity
) -> tuple[str, str]:
    """Return (final_label, decision_reason).

    FAISS overrides DeBERTa only when it has >= FAISS_ENSEMBLE_MIN_VOTES results
    AND its majority agrees with DeBERTa. If FAISS majority disagrees, we trust
    DeBERTa — FAISS HIGH precision is only 25.8%, so disagreement is likely noise.
    """
    deberta_label = deberta_result["label"]

    if len(faiss_results) < FAISS_ENSEMBLE_MIN_VOTES:
        return deberta_label, f"deberta_only (faiss={len(faiss_results)} results < {FAISS_ENSEMBLE_MIN_VOTES})"

    # Weighted majority: each vote weighted by similarity score
    weighted = Counter()
    for r in faiss_results:
        weighted[r.risk_level] += r.similarity
    faiss_majority = weighted.most_common(1)[0][0]

    if faiss_majority == deberta_label:
        return deberta_label, f"deberta+faiss_agree (faiss={len(faiss_results)}, majority={faiss_majority})"
    else:
        # Disagreement → trust DeBERTa; FAISS is unreliable when labels conflict
        return deberta_label, f"deberta_wins_conflict (faiss_said={faiss_majority}, deberta={deberta_label})"


# ---------------------------------------------------------------------------
# Targeted contract context
# ---------------------------------------------------------------------------

def _targeted_context(clause: ClauseObject, all_clauses: list[ClauseObject]) -> list[dict]:
    """Return Parties clause + related clause types for this clause type.

    Returns a small, focused list so the LLM explanation call is grounded in
    relevant contract context rather than noise from unrelated clauses.
    """
    related_types = set(CLAUSE_CONTEXT_TYPES.get(clause.clause_type, []))
    related_types.add("Parties")

    return [
        {"clause_type": c.clause_type, "clause_text": c.clause_text}
        for c in all_clauses
        if c.clause_id != clause.clause_id and c.clause_type in related_types
    ]


# ---------------------------------------------------------------------------
# LLM explanation (single call, label already decided)
# ---------------------------------------------------------------------------

def _explain_label(
    clause: ClauseObject,
    signing_party: str,
    final_label: str,
    deberta_result: dict,
    faiss_results: list,
    context_clauses: list[dict],
    llm,
) -> str:
    """Ask the LLM to explain a pre-decided label. Does NOT choose the label."""

    # Build a compact evidence summary
    if faiss_results:
        label_counts = Counter(r.risk_level for r in faiss_results)
        avg_sim = sum(r.similarity for r in faiss_results) / len(faiss_results)
        faiss_summary = (
            f"{len(faiss_results)} precedents found (avg similarity {avg_sim:.2f}): "
            + ", ".join(f"{label}×{cnt}" for label, cnt in label_counts.most_common())
        )
    else:
        faiss_summary = "No strong precedents found in corpus (similarity < 0.75)"

    context_block = ""
    if context_clauses:
        lines = "\n".join(
            f"  [{c['clause_type']}] {c['clause_text'][:200]}"
            for c in context_clauses[:5]
        )
        context_block = f"\nRelated clauses from this contract:\n{lines}\n"

    prompt = (
        f"You are a legal contract analyst. The risk label for the clause below has "
        f"been determined to be {final_label}. Write a 1-3 sentence explanation "
        f"grounded in the clause text and evidence provided.\n\n"
        f"Clause type: {clause.clause_type}\n"
        f"Signing party: {signing_party or 'unknown'}\n"
        f"Text: {clause.clause_text}\n\n"
        f"Evidence:\n"
        f"  DeBERTa: {deberta_result['label']} ({deberta_result['confidence']:.0%} confidence)\n"
        f"  FAISS:   {faiss_summary}\n"
        f"{context_block}\n"
        f"Explanation (1-3 sentences, do not state the label — just explain why):"
    )

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()
    except Exception as e:
        logger.warning("LLM explanation failed: %s", e)
        return f"({deberta_result['label']} predicted by DeBERTa at {deberta_result['confidence']:.0%} confidence)"


# ---------------------------------------------------------------------------
# Per-clause assessment
# ---------------------------------------------------------------------------

def _assess_clause(
    clause: ClauseObject,
    deberta_result: dict,
    signing_party: str,
    all_clauses: list[ClauseObject],
    index_path: str,
    llm,
    k: int,
    use_contract_context: bool = True,
) -> RiskAssessedClause:
    # 1. FAISS retrieval
    faiss_raw     = query_index(clause.clause_text, index_path, k)
    faiss_results = [r for r in faiss_raw if r.similarity >= FAISS_MIN_SIMILARITY]

    # 2. Ensemble label decision (no LLM)
    final_label, decision_reason = _ensemble_label(deberta_result, faiss_results)

    # 3. Targeted contract context
    context_clauses = (
        _targeted_context(clause, all_clauses) if use_contract_context else []
    )

    logger.debug(
        "  %s | deberta=%s faiss=%d results | decision: %s",
        clause.clause_type, deberta_result["label"], len(faiss_results), decision_reason,
    )

    # 4. LLM explanation only
    explanation = _explain_label(
        clause, signing_party, final_label,
        deberta_result, faiss_results, context_clauses, llm,
    )

    # Trace: record what signals were used (for eval/audit, no tool calls)
    trace = []
    if faiss_results:
        trace.append(AgentTraceEntry(tool="precedent_search", result_count=len(faiss_results)))
    if context_clauses:
        trace.append(AgentTraceEntry(tool="contract_context", result_count=len(context_clauses)))

    return RiskAssessedClause(
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        clause_text=clause.clause_text,
        clause_type=clause.clause_type,
        risk_level=final_label,
        risk_explanation=explanation,
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
    use_contract_search: bool = True,   # kept for API compatibility
    skip_ids: Optional[set] = None,
    checkpoint_file: Optional[str] = None,
) -> list[RiskAssessedClause]:
    """Assess risk for all risk-relevant clauses from a single contract.

    Args:
        clauses:              All ClauseObjects from Stage 1+2 for this contract.
        config_path:          Path to stage3_config.yaml.
        ce_model_path:        Local path to CE model (defaults to HF Hub).
        corn_model_path:      Local path to CORN model (defaults to HF Hub).
        use_contract_search:  If False, skip targeted contract context lookup.
        skip_ids:             Clause IDs to skip (already processed in a prior run).
        checkpoint_file:      Path to JSONL file; each result is appended immediately.

    Returns:
        List of RiskAssessedClause — one per risk-relevant clause.
    """
    import json

    cfg = load_config(config_path)

    index_path = cfg["faiss_index_path"]
    k          = cfg["similarity_top_k_low_conf"]   # always use low-conf k (= max k)

    llm = make_llm(cfg)

    logger.info("Loading DeBERTa risk classifier (Ens-F) ...")
    classifier = RiskClassifier(
        ce_model_path=ce_model_path or CE_MODEL_ID,
        corn_model_path=corn_model_path or CORN_MODEL_ID,
    )

    doc_ids = {c.document_id for c in clauses}
    signing_parties = {
        doc_id: extract_signing_party(doc_id, clauses)
        for doc_id in doc_ids
    }

    _skip = skip_ids or set()
    results: list[RiskAssessedClause] = []
    risk_clauses   = [c for c in clauses if c.clause_type not in METADATA_CLAUSE_TYPES]
    skipped_meta   = len(clauses) - len(risk_clauses)
    skipped_done   = sum(1 for c in risk_clauses if c.clause_id in _skip)
    pending        = [c for c in risk_clauses if c.clause_id not in _skip]
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

        logger.info(
            "[%d/%d] %s | DeBERTa=%s (%.2f)",
            i, len(pending), clause.clause_type,
            deberta_result["label"], deberta_result["confidence"],
        )

        assessed = _assess_clause(
            clause=clause,
            deberta_result=deberta_result,
            signing_party=signing_party,
            all_clauses=clauses,
            index_path=index_path,
            llm=llm,
            k=k,
            use_contract_context=use_contract_search,
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
