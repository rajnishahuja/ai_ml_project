# Optional Enhancements

> Ideas to brainstorm and assign once the basic pipeline is complete.
> Add freely — no need to update ARCHITECTURE.md or TASK_LIST.md.

---

## Stage 3

**OE-1 — Agentic behavior deep verification** *(superseded — 2026-05-04)*
ReAct agent was removed. Stage 3 now uses a deterministic DeBERTa+FAISS ensemble for
label decisions; LLM writes explanation only. No tool-calling loop to verify.

**OE-2 — Fast-path LLM label override** *(superseded — 2026-05-04)*
Fast/slow path distinction removed. All clauses go through the same ensemble path.

**OE-5 — DeBERTa confidence calibration** *(superseded — 2026-05-04)*
Confidence gating removed. All clauses go through the ensemble regardless of DeBERTa
confidence. Calibration is no longer on the critical path.

**OE-6 — DeBERTa + FAISS ensemble** *(implemented — 2026-05-04)*
Done. `agent.py:_ensemble_label()` — per-label FAISS vote thresholds (LOW=2, MEDIUM=2,
HIGH=skip). FAISS confirms DeBERTa when they agree; DeBERTa wins on disagreement.
Result: pipeline F1 = DeBERTa F1 (0.607). FAISS coverage is too low (42%) for
FAISS to diverge from DeBERTa enough to change outcomes.

**OE-7 — Refactor to static LangGraph** *(superseded — 2026-05-04)*
Went further: removed LangGraph entirely. Label decision is now pure Python logic
(DeBERTa + FAISS ensemble). LLM makes a single non-tool-calling explanation call.

**OE-8 — Synthetic data generation for label-poor clause types** *(new)*
Root cause of the 0.607 ceiling: 8-10 clause types have label poverty (< 5 examples
of at least one label). The worst: Price Restrictions (0 HIGH), Post-Termination
Services (DeBERTa accuracy 0%), License Grant (20% accuracy).
Cannot mine more from CUAD — all positive spans already labeled.
Approach: prompt LLM to rewrite existing MEDIUM/LOW clauses into HIGH-risk variants
with specific adversarial modifications ("add one-sided liability cap", "remove
notice period"), then verify with human spot-check.
Expected gain: +0.02–0.04 F1 on the 5 worst types.
Target: ≥15 examples per (type, label) cell for every type.

**OE-9 — BM25 + FAISS RRF hybrid retrieval** *(new)*
FAISS coverage for HIGH clauses is only 27% at min_sim=0.75. FAISS finds semantic
matches but HIGH risk often comes from specific legal phrases ("without limitation",
"irrevocably assigns", "in no event") that BM25 keyword search would catch.
RRF (Reciprocal Rank Fusion): merge FAISS ranks + BM25 ranks with score = Σ 1/(60+rank_i).
Scale-free — no normalisation needed. Would improve HIGH coverage without lowering
the similarity threshold (which hurts precision).
Implement: add `rank_bm25` to requirements; build BM25 index at the same time as
FAISS; merge in `embeddings.py:query_index()`.

**OE-10 — Jina Embeddings v3** *(new)*
Replace `all-MiniLM-L6-v2` (22M params, 512 tokens) with `jinaai/jina-embeddings-v3`
(570M params, 8192 tokens). Better coverage expected because: (a) much larger model
captures nuanced legal semantics; (b) 8192-token context handles long clauses that
get truncated at 512 today; (c) task-specific LoRA adapters can be tuned for retrieval.
Cost: rebuild FAISS index (~15 min), add jina to requirements.
Measure: repeat coverage/precision analysis from 2026-05-04 session and compare.

**OE-11 — Allow FAISS to override DeBERTa for LOW** *(superseded — 2026-05-04)*
Was designed for the pure DeBERTa+FAISS ensemble (no LLM). With the LangGraph ReAct
agent restored, the LLM can already reason toward LOW when precedent evidence is strong —
the agent's consensus-based system prompt handles this case. A separate FAISS override
rule is redundant.

## Stage 4

**OE-3 — Missing protections detection**
Flag standard protective clauses absent from the contract.
Two approaches:
- Universal list: Governing Law, Termination for Convenience, Anti-Assignment (always expected)
- Companion map (Option B): if clause X is present, clause Y should also be present
  (e.g. License Grant → Non-Transferable License; Renewal Term → Notice Period)
Requires: either Stage 3 cross-clause awareness, or a post-Stage-3 contract-level check.
Note: our ground truth labels are per-clause — any risk impact of missing clauses
is not reflected in training data, so this should be informational only.

**OE-4 — Per-clause mitigation reasoning**
Replace the lookup-table recommendation with an LLM-generated mitigation per clause,
grounded in the clause text and precedents from FAISS.
Richer than a canned lookup but requires an extra Qwen call per HIGH/MEDIUM clause.
Consider cost vs. value trade-off before implementing.

*Add new entries below as they come up.*
