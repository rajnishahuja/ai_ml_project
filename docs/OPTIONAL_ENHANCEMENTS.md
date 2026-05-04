# Optional Enhancements

> Ideas to brainstorm and assign once the basic pipeline is complete.
> Add freely — no need to update ARCHITECTURE.md or TASK_LIST.md.

---

## Stage 3

**OE-1 — Agentic behavior deep verification**
Craft 2–3 clauses with deliberately mixed FAISS precedents to force `contract_search`
to fire. Verify `override_reason` is populated when agent disagrees with DeBERTa.
Verify multi-step loop (agent calls a tool, re-evaluates, calls another).
Currently: smoke test only verified single-tool agent path.

**OE-2 — Fast-path LLM label override** *(superseded)*
Fast path was removed — all clauses now go through the agent. DeBERTa confidence
was too miscalibrated (0.91 confidence with wrong label) to be a reliable gate.
See OE-5 for calibration notes.

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

---

**OE-5 — DeBERTa confidence calibration**
Fast path accuracy is only 70% despite conf ≥ 0.6 threshold — DeBERTa is miscalibrated
(overconfident). Two fixes worth comparing:

- **Temperature scaling** (proper fix): fit a single scalar `T` on the val split so reported
  confidence matches empirical accuracy. ~30 lines in `scripts/calibrate.py`; bake `T` into
  `infer.py:predict()`. After calibration, conf ≥ 0.6 should mean ~90%+ accuracy.
- **Lower threshold** (quick test): change 0.6 → 0.75 in `configs/stage3_config.yaml` and
  re-run `eval_stage3.py`. Pushes borderline overconfident cases to the agent path.

Agent-only (threshold = 1.0) is also an option but ~2.7× slower (452 vs 169 LLM calls).

**OE-6 — DeBERTa + FAISS probability ensemble (no LLM)**
DeBERTa already outputs a probability vector [p_LOW, p_MEDIUM, p_HIGH]. FAISS vote counts
can be normalized to the same space. A weighted average α×DeBERTa + (1-α)×FAISS_votes,
with α tuned on the validation split, would combine both signals without any LLM call.
This would add a clean ablation row between "DeBERTa only" and "Agent":
  DeBERTa only → DeBERTa+FAISS ensemble → Agent (FAISS+LLM)
showing whether FAISS improves over DeBERTa before the LLM is introduced.
Implementation: ~30 lines in eval_stage3.py as a new --ensemble ablation flag.

**OE-7 — Refactor agent to static LangGraph**
The current ReAct agent (`create_react_agent`) always executes the same sequence:
precedent_search → contract_search → synthesis. This is a fixed execution pattern,
not dynamic agentic reasoning. A static LangGraph with three explicit nodes would be
simpler, more predictable, faster (no ReAct overhead), and easier to test.
True dynamic behavior would require more tools with non-obvious selection criteria
(e.g. jurisdiction lookup, clause-type-specific tools) — worth revisiting if the
tool set expands.

*Add new entries below as they come up.*
