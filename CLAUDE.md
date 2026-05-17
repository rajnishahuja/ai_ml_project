# Legal Contract Risk Analyzer — Claude Context

> **Precedence rule:** `ARCHITECTURE.md` is the source of truth for this project. This file is a downstream pointer that summarizes and references it. On any conflict between this file and `ARCHITECTURE.md`, **`ARCHITECTURE.md` wins** and this file is updated to match. Do not retcon `ARCHITECTURE.md` from CLAUDE.md.

## Project Overview
ML pipeline that analyzes legal contracts and flags risky clauses using the CUAD dataset.
4-stage pipeline: Extract clauses (DeBERTa) → Classify clause type → Assess risk (DeBERTa Ens-F + LangGraph ReAct agent with RAG) → Generate report (Qwen3-30B executive summary + lookup-table recommendations).

## Current State (as of 2026-05-17)

### Branch: `main`

### What's Done
- **Phase 0 foundation**: configs, schema.py, utils.py — complete
- **data_loader.py**: `theatticusproject/cuad-qa` dataset, train (22,450) / test (4,182)
- **Stage 1/2 extraction**: `data/processed/all_positive_spans.json` — 6,702 positive clause spans
- **Stage 3 labeling pipeline**: Qwen/30B + Gemini 2.5 Flash + Claude Sonnet relabel pass merged → `data/processed/training_dataset.json` (**4,375 rows = 4,276 hard + 99 soft**, signing_party field included). See `docs/STAGE3_LABELING.md` for category breakdown.
- **Stage 3 DeBERTa classifier** (Ens-F, macro_F1=0.607):
  - Run 22 (CE + parties): `models/stage3_risk_deberta_v3_run22_parties/final/`
  - Run 23 (CORN + parties): `models/stage3_risk_deberta_v3_run23_corn_parties/final/`
  - HuggingFace Hub: `rajnishahuja/cuad-risk-deberta-ce-parties` + `cuad-risk-deberta-corn-parties`
  - Inference API: `scripts/infer.py` — `RiskClassifier` class (Ens-F predict/predict_batch)
- **Stage 3 Agent/RAG** (complete, smoke-tested 2026-05-01):
  - `src/stage3_risk_agent/embeddings.py` — FAISS index builder + query; **model-agnostic** (`model_name` param); supports MiniLM and LegalBERT
  - `src/stage3_risk_agent/risk_classifier.py` — wraps `RiskClassifier` + `extract_signing_party()`
  - `src/stage3_risk_agent/tools.py` — `make_precedent_search_tool(index_path, model_name, default_min_similarity)`, `make_contract_search_tool()`
  - `src/stage3_risk_agent/agent.py` — `assess_clauses()` entry point; reads `embedding_model` + `similarity_threshold` from config; every clause runs through LangGraph ReAct agent; DeBERTa default signal; override only on tool consensus. (Locked 2026-05-04)
  - `src/stage3_risk_agent/train.py` — **model-agnostic** (DeBERTa, LegalBERT, RoBERTa); `_get_backbone_attr()` helper; CORN + LLRD both work for any encoder
  - `scripts/build_faiss_index.py` — reads `embedding_model` + `faiss_index_path` from config
  - `scripts/eval_embeddings.py` — retrieval quality eval (precision@k, zero-result rate, per-class); side-by-side model comparison
  - `scripts/calibrate_threshold.py` — sweeps similarity thresholds, recommends optimal for a given embedding model
  - `scripts/monitor_qwen_latency.py` — polls port 10006 metrics every 30s, alerts on throughput drop
  - `scripts/smoke_test_stage3_agent.py` — end-to-end test; all 5 clauses assessed, Parties excluded, agent override verified
- **LegalBERT embedding experiment** (2026-05-17):
  - MiniLM precision@5 HIGH = 0.045, zero-result rate = 64.6%
  - **LegalBERT precision@5 HIGH = 0.424, zero-result rate = 0.2%** — major improvement
  - Calibrated threshold: 0.82 for LegalBERT (vs 0.75 for MiniLM)
  - FAISS indexes: `data/faiss_index/clauses_minilm.index` + `data/faiss_index/clauses_legalbert.index`
  - Config set to LegalBERT. MiniLM indexes retained at `data/faiss_index/clauses_minilm.*` for rollback.
  - **E2E eval COMPLETE (2026-05-17, Qwen3-30B, full 452 rows, no-CS constrained):**
    | Embedding model (FAISS) | DeBERTa-only | Agent macro F1 | Delta | HIGH F1 |
    |---|---|---|---|---|
    | `all-MiniLM-L6-v2` | 0.6118 | 0.6257 | +0.016 | 0.634 |
    | `nlpaueb/legal-bert-base-uncased` | 0.6097 | **0.6407** | **+0.031** | **0.650** |
  - Agent delta doubled; HIGH F1 is the headline win (MiniLM precision@5 HIGH was 0.045 → LegalBERT 0.424).

### LLM Setup (Stage 3)
- **Model**: Qwen3-30B Q4_K_XL, llama.cpp server on port 10006
- **Key quirk**: `langchain-openai ≥ 1.2` defaults `with_structured_output` to `method="json_schema"` (OpenAI Structured Outputs / `.parse()`), which llama.cpp rejects with HTTP 400. Always pass `method="function_calling"` explicitly. See ARCHITECTURE.md "Implementation Notes".

### Stage 4 Report Generator (complete 2026-05-03)
- `src/stage4_report_gen/aggregator.py` — groups by risk level, computes weighted score
- `src/stage4_report_gen/recommender.py` — lookup table: 108 entries (36 types × 3 levels)
- `src/stage4_report_gen/report_builder.py` — assembles `RiskReport`; one Qwen call for executive summary
- `configs/stage4_config.yaml` — Qwen via llama-server (shared with Stage 3, port 10006). **FLAN-T5 is dead** — see `src/stage4_report_gen/explainer.py` (NotImplementedError scaffolding, slated for removal).
- `missing_protections` left as `[]` — optional enhancement (see `docs/OPTIONAL_ENHANCEMENTS.md`)

### API Layer (FastAPI, added 2026-05-06)
- `app/main.py` + `app/routers/` + `app/services/` — full pipeline behind HTTP endpoints (`/api/v1/stage1/analyze`, `/api/v1/stage3/assess`, `/api/v1/stage4/report`). See ARCHITECTURE.md "API Layer (FastAPI)".

### Immediate Next Steps
1. ~~**LegalBERT E2E eval**~~ — DONE. LegalBERT is production embedding model. Results in `docs/STAGE3_EXPERIMENTS.md`.
2. **LegalBERT classifier training** — change `model_name` + `output_dir` in `configs/stage3_config.yaml` risk_classifier section; run `python3 -m src.stage3_risk_agent.train --loss ce`. ~30-35 min.
3. **Optional enhancements** — see `docs/OPTIONAL_ENHANCEMENTS.md`
4. **Code hygiene** — see `docs/CODE_REVIEW_NOTES_2026-05-13.md` (stage1 dataset regression, dead explainer.py, scaffolding tests, missing requirements.txt deps)

### Stage 1 — still pending (GPU needed)
1. **Tokenize full training set** — run `preprocess_for_qa()` on all 22,450 examples
2. **Implement train.py (T1.2)** — extract training logic from pipeline.py
3. **Train DeBERTa** — fine-tune on CUAD QA task
4. **Run baseline evaluation** — benchmark spaCy/regex baseline on CUAD test set

### Key Decisions Made
- **Dataset**: Using `theatticusproject/cuad-qa` (HuggingFace). Pre-flattened QA rows, no flatten/split needed.
- **Metadata routing (Option B)**: 5 metadata CUAD types (Document Name, Parties, Agreement Date, Effective Date, Expiration Date) are NOT risk-labeled — they route to the Stage 4 report header instead. See `docs/STAGE3_SYNTHETIC_LABELS_DISCUSSION.md`.
- **Dedup**: Label once per unique (whitespace-normalized) clause text, fan label to duplicate rows. Saves ~40% API calls combined with metadata routing (6,702 → 3,974).
- **Labeling perspective**: Always from the signing party (counterparty to the drafter).
- **Anushka's code**: Her Stage 1+2 work is in `src/stage1_extract_classify/` using `CUAD_v1.json` locally. Loaders use HuggingFace `cuad-qa` independently. Review suggestions in `docs/STAGE1_REVIEW_NOTES.md`.

## Working Style
- **Interactive and incremental**: Small pieces of code, explain each step, user runs and verifies before moving on.
- **Learning-focused**: The goal is to understand and implement, not just finish. Explain concepts before code.
- **Two machines**: Local laptop (no GPU) for development, separate server with GPU for training. Keep code portable.

## Architecture Reference
- `ARCHITECTURE.md` — full data flow, directory structure, model table
- `docs/TASK_LIST.md` — all tasks with status and dependencies
- `docs/STAGE1_REVIEW_NOTES.md` — alignment suggestions for Anushka's code
- `configs/stage1_config.yaml` — DeBERTa training hyperparameters
