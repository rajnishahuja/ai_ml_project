# Project Handoff — Legal Contract Risk Analyzer

> Drop this file into your AI assistant's context window to resume work.
> Last updated: 2026-05-03

---

## What this project is

ML pipeline that reads legal contracts and flags risky clauses.
4-stage pipeline: **Extract clauses → Assess risk → Generate report**

- Dataset: CUAD (`theatticusproject/cuad-qa` on HuggingFace), 41 clause types
- Language: Python 3.10+, working directory `/home/ubuntu/rajnish/aiml`
- Python env: `/home/ubuntu/miniconda3/envs/rajnish-env/bin/python3` (has GPU libs)

---

## Current state (2026-05-03)

### What is DONE

**Stage 3 classifier (Ens-F, macro_F1 = 0.607)** — complete, models on HF Hub
- Ensemble of Run 22 (CE loss + parties metadata) and Run 23 (CORN loss + parties metadata)
- Local model paths:
  - `models/stage3_risk_deberta_v3_run22_parties/final/`
  - `models/stage3_risk_deberta_v3_run23_corn_parties/final/`
- HF Hub: `rajnishahuja/cuad-risk-deberta-ce-parties` + `rajnishahuja/cuad-risk-deberta-corn-parties`
- Inference API: `scripts/infer.py` — `RiskClassifier` class, `predict()` / `predict_batch()`

**Stage 3 Agent/RAG** — complete, smoke-tested 2026-05-01
- Architecture: **Hybrid Confidence-Gated**
  - `conf >= 0.6` → fast path: single LLM call + FAISS precedent context
  - `conf < 0.6`  → agent path: LangGraph ReAct loop with tools; may override DeBERTa label
- Key files:
  - `src/stage3_risk_agent/agent.py` — `assess_clauses()` main entry point
  - `src/stage3_risk_agent/embeddings.py` — FAISS index builder + `query_index()`
  - `src/stage3_risk_agent/risk_classifier.py` — wraps `RiskClassifier` + `extract_signing_party()`
  - `src/stage3_risk_agent/tools.py` — `make_precedent_search_tool()`, `make_contract_search_tool()`
  - `scripts/build_faiss_index.py` — builds `data/faiss_index/clauses.index` (4,276 vectors)
  - `scripts/smoke_test_stage3_agent.py` — end-to-end test (5 clauses, fast + agent paths)
- FAISS index: `data/faiss_index/clauses.index` + `clauses.json`, model=all-MiniLM-L6-v2, 384-dim
- LLM: Qwen3-30B Q4_K_XL via llama.cpp server on port 10006 (OpenAI-compatible API)

**Training data**
- `data/processed/training_dataset.json` — 4,384 rows, fields: `clause_text`, `clause_type`, `label` (LOW/MEDIUM/HIGH), `signing_party`
- `data/processed/splits.json` — pre-computed train/val/test split indices
- Test split: ~877 rows (stratified)

### What is PENDING (priority order)

**T6.1 — `scripts/eval_stage3.py`** ← START HERE
- Load test split rows from `training_dataset.json` + `splits.json`
- Sample ~150 clauses stratified by label (50 per class) OR run all ~877
- Convert to `ClauseObject` instances grouped by synthetic `document_id` (one doc per clause type, or one big doc — see note below)
- Call `assess_clauses()` with local model paths
- Compare `result.risk_level` vs ground-truth `label`
- Report:
  - Pipeline macro F1 vs DeBERTa-alone F1
  - Fast-path: how often LLM changes DeBERTa's label
  - Agent-path: label-change rate + accuracy vs ground truth
  - Per-class (LOW/MEDIUM/HIGH) precision/recall/F1
- Save results to `data/eval/eval_stage3_results.json`

**T6.2 — Agentic behavior verification**
- Craft 2–3 clauses with mixed precedents to force `contract_search` + multi-step loop
- Verify `override_reason` populated when agent disagrees with DeBERTa

**T6.3 — Stage 4 report generator**
- Input: `list[RiskAssessedClause]` + metadata clauses → `RiskReport`
- Sections: high/medium/low clause groups, per-clause remedy from lookup table, LLM executive summary

**T6.4 — `scripts/run_pipeline.py`** end-to-end wiring

---

## Key data contracts (schemas)

```python
# src/common/schema.py

class ClauseObject(BaseModel):
    clause_id: str
    document_id: str
    clause_type: str        # one of 41 CUAD types
    clause_text: str
    start_pos: int
    end_pos: int
    confidence: float       # Stage 1+2 extraction confidence (NOT risk confidence)

class RiskAssessedClause(BaseModel):
    clause_id: str
    document_id: str
    clause_text: str
    clause_type: str
    risk_level: str         # "LOW" | "MEDIUM" | "HIGH"
    risk_explanation: str
    similar_clauses: list   # list[SimilarClause] from FAISS
    cross_references: list
    confidence: float       # DeBERTa risk confidence
    agent_trace: list       # list[AgentTraceEntry], empty on fast path

class AgentTraceEntry(BaseModel):
    tool: str
    result_count: Optional[int]
```

---

## Critical implementation quirks

### 1. langchain-openai ≥ 1.2 + llama.cpp → HTTP 400
`with_structured_output()` defaults to `method="json_schema"` which calls OpenAI's
Structured Outputs endpoint (`.parse()`). llama.cpp rejects this with:
`"Failed to initialize samplers: std::exception"`

**Fix**: always pass `method="function_calling"` explicitly:
```python
llm.with_structured_output(MySchema, method="function_calling").invoke(prompt)
```
Both `_fast_path` and `_agent_path` in `agent.py` already do this.

### 2. Synthesis prompt pattern (agent path)
Passing the full ReAct message history (which contains `tool_call` messages) to
`with_structured_output(function_calling)` causes the model to return plain text —
it sees existing tool calls and skips calling the schema tool.

**Fix**: extract the agent's final plain-text AI message, build a clean single-turn
`synthesis_prompt`, pass that to `with_structured_output` instead. See `agent.py:_agent_path`.

### 3. Metadata clause types are NOT risk-assessed
`METADATA_CLAUSE_TYPES = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}`
These are skipped in `assess_clauses()` and route to the Stage 4 report header instead.
`extract_signing_party()` reads the "Parties" clause to set signing-party context.

### 4. Signing-party metadata is critical for accuracy
Segment A fed to DeBERTa is `"clause_type | signing party: <parties_span>"` not just `"clause_type"`.
This was +0.021 macro_F1. The `extract_signing_party()` function in `risk_classifier.py` handles this.

---

## How to run things

```bash
# Activate env
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate rajnish-env

# Run smoke test (verifies full pipeline end-to-end)
/home/ubuntu/miniconda3/envs/rajnish-env/bin/python3 scripts/smoke_test_stage3_agent.py

# LLM server must be running on port 10006 for agent path
# Start with: llama-server --model <path> --port 10006 --jinja --reasoning off
```

---

## File map (key files only)

```
configs/
  stage3_config.yaml          LLM URL, FAISS path, confidence threshold (0.6), k values

data/processed/
  training_dataset.json       4,384 labeled rows (clause_text, clause_type, label, signing_party)
  splits.json                 train/val/test indices

data/faiss_index/
  clauses.index               4,276-vector FAISS index (all-MiniLM-L6-v2, IndexFlatIP)
  clauses.json                Metadata for each vector (clause_text, clause_type, risk_level)

models/
  stage3_risk_deberta_v3_run22_parties/final/   CE model (352MB)
  stage3_risk_deberta_v3_run23_corn_parties/final/  CORN model (352MB, safetensors only)

scripts/
  infer.py                    RiskClassifier class (Ens-F ensemble predict/predict_batch)
  smoke_test_stage3_agent.py  End-to-end test — run this to verify the pipeline works
  build_faiss_index.py        Rebuilds data/faiss_index/ from training_dataset.json

src/stage3_risk_agent/
  agent.py                    assess_clauses() — main entry point
  embeddings.py               query_index(), build_index()
  risk_classifier.py          RiskClassifier wrapper + extract_signing_party()
  tools.py                    make_precedent_search_tool(), make_contract_search_tool()

src/common/
  schema.py                   ClauseObject, RiskAssessedClause, RiskReport, AgentTraceEntry
  utils.py                    load_config(), logging helpers

docs/
  STAGE3_EXPERIMENTS.md       Full training run log (24 runs, 14 ensembles) — canonical
  STAGE3_SYNTHETIC_LABELS_DISCUSSION.md  Metadata routing decision
  HANDOFF.md                  This file

ARCHITECTURE.md               Full data flow diagram, all schemas, implementation notes
CLAUDE.md                     Claude Code context (keep in sync when making changes)
```
