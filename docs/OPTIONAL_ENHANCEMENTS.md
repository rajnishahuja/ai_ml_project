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

**OE-2 — Fast-path LLM label override**
Fast path currently has 0% label change rate — LLM echoes DeBERTa's label.
Explore whether allowing the fast path to override (not just explain) would improve F1,
or whether the confidence gate already handles this correctly via the agent path.

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

*Add new entries below as they come up.*
