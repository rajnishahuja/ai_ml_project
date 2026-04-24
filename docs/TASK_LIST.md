# Legal Contract Risk Analyzer — Implementation Task List

> **Purpose**: Detailed task breakdown for 4 team members to implement the project end-to-end.
> Organized into 6 phases with explicit dependencies and parallel work lanes.
> Review and update this file as tasks are agreed, started, and completed.

---

## Key Principles

- **T0.9 is the first priority** — shared data contracts are the interface between all stages; agree on them before any stage code starts.
- **Critical path**: T0.9 → T2.1 → T2.2 → T3.1 → T3.3 → T5.1 → T5.5
- **Phase 1 and Phase 2 run in parallel** — Members A & B do Stage 1+2; Members C & D do synthetic data.
- **Phase 3 and Phase 4 run in parallel** — Members C & D do Stage 3 agent; Members A & B do Stage 4 report.
- Phase 4 builds against **mock data** (using the T0.9 data contracts) until Stage 3 is ready.

---

## Phase 0 — Foundation & Shared Infrastructure

*All 4 members work in parallel. No stage code yet — just scaffolding everyone depends on.*

| ID | Task | Description | Assignable to | Depends on | Status |
|----|------|-------------|---------------|-----------|--------|
| T0.1 | Create `configs/stage1_config.yaml` | Model name, max_seq_length, stride, batch_size, learning_rate, epochs, output_dir | Member A | — | ✅ done |
| T0.2 | Create `configs/stage3_config.yaml` | Risk classifier path, embedding model, FAISS path, Mistral quantization config, agent max_iterations, similarity_threshold | Member B | — | ✅ done |
| T0.3 | Create `configs/stage4_config.yaml` | FLAN-T5 model path, prompt templates, risk thresholds, report output format | Member C | — | ✅ done |
| T0.4 | Implement `src/common/preprocessing.py` | PDF (PyMuPDF), DOCX (python-docx), plain text loader, text cleaning. Single `extract_text(file_path) → str` entry point. **Deferred** — not needed for CUAD training/eval; only for production inference with real contract files. | Member A | — | deferred |
| T0.5 | Implement `src/common/data_loader.py` | `load_cuad_dataset()` from `theatticusproject/cuad-qa` (already flat QA format with train/test split), `preprocess_for_qa()` for tokenization with sliding window. **Simplified** — `flatten_to_qa_examples()` and `split_dataset()` no longer needed since HF dataset is pre-flattened and pre-split. | Member B | — | in progress |
| T0.6 | Implement `src/common/utils.py` | Config loader (`load_config(yaml_path) → dict`), logging setup helper, JSON serialization for dataclasses, shared metric helpers (`normalize_answer`, `squad_em_f1`, `span_iou`). Removes duplication between `baseline.py` and `evaluate.py`. | Member C | — | ✅ done |
| T0.7 | Reconcile requirements files | Resolve version conflicts between `requirements.txt` (transformers 4.36, pymupdf) and `requirements_stage1_2.txt` (transformers 4.40, pdfplumber). Produce one consolidated file. | Member D | — | not started |
| T0.8 | Create `scripts/download_cuad.py` | Downloads CUAD from HuggingFace, saves to `data/raw/`, verifies integrity. **May not be needed** — `load_dataset()` handles download/caching automatically. | Member D | — | not started |
| T0.9 | Define shared data contracts | Dataclasses for `ClauseObject` (Stage 1+2 output), `RiskAssessedClause` (Stage 3 output), `RiskReport` (Stage 4 output). Must match JSON schemas in `ARCHITECTURE.md` exactly. Place in `src/common/schema.py`. | Member D | — | ✅ done |

> **T0.9 must be reviewed and agreed by all 4 members before any stage implementation begins.**

---

## Phase 1 — Stage 1+2: Clause Extraction & Classification

*Members A & B. Depends on T0.4, T0.5, T0.6, T0.9.*
*Runs in parallel with Phase 2.*

| ID | Task | Description | Assignable to | Depends on | Status |
|----|------|-------------|---------------|-----------|--------|
| T1.1 | Refactor `pipeline.py` → `model.py` | DeBERTa QA model wrapper class. Load from config, handle tokenization, return raw logits. No training or evaluation logic. | Member A | T0.1, T0.9 | not started |
| T1.2 | Refactor `pipeline.py` → `train.py` | Fine-tuning script using HF Trainer. All hyperparams from `stage1_config.yaml`. Handles sliding window for long contracts. Saves model + tokenizer. | Member A | T1.1, T0.5 | not started |
| T1.3 | Create `src/stage1_extract_classify/predict.py` | Inference: contract text → `List[ClauseObject]`. Runs 41 QA queries, applies confidence threshold, deduplicates overlapping spans. | Member B | T1.1, T0.4, T0.9 | not started |
| T1.4 | Refactor `baseline.py` | Update imports to use `src/common/` (preprocessing, metrics, data_loader). Remove duplicated functions. Output `ClauseObject` format from T0.9. | Member B | T0.4, T0.6, T0.9 | not started |
| T1.5 | Refactor `evaluate.py` | Use shared metrics from `utils.py`. Evaluate DeBERTa and baseline on same test split. Produce comparison report (JSON + text). Remove duplicated metric code. | Member A | T1.2, T1.3, T1.4, T0.6 | not started |
| T1.6 | Write `tests/test_stage1.py` | Unit tests: model loading, single-query inference, ClauseObject schema, baseline extractor. Mock actual model with small fixtures. | Member B | T1.3, T1.4 | not started |
| T1.7 | Create `notebooks/01_cuad_exploration.ipynb` | Dataset stats, clause type distribution, example contracts, train/test split analysis | Member B | T0.5, T0.8 | not started |

---

## Phase 2 — Synthetic Risk Labels & FAISS Index

*Members C & D. Gate phase — Stage 3 cannot start without this.*
*Can start as soon as T0.5 and T0.9 are done. Runs in parallel with Phase 1.*

| ID | Task | Description | Assignable to | Depends on | Status |
|----|------|-------------|---------------|-----------|--------|
| T2.1 | Implement `src/stage3_risk_agent/synthetic_labels.py` | Prompt template for LLM-based risk labeling (Low/Medium/High + reason). Batch process all CUAD clause annotations. Output to `data/synthetic/risk_labels.json`. | Member C | T0.5, T0.9 | not started |
| T2.2 | Create `notebooks/02_synthetic_labeling.ipynb` | Validate labels: distribution across risk levels, spot-check examples, filter low-quality labels | Member C | T2.1 | not started |
| T2.3 | Implement `src/stage3_risk_agent/embeddings.py` | `build_faiss_index()` using `all-MiniLM-L6-v2`, saves to `data/faiss_index/`. `query_similar(clause_text, top_k) → List[SimilarClause]`. | Member D | T2.1, T0.9 | not started |
| T2.4 | Create `scripts/generate_synthetic.py` | CLI wrapper around `synthetic_labels.py` for batch generation | Member C | T2.1 | not started |
| T2.5 | Create `scripts/build_faiss_index.py` | CLI wrapper around `embeddings.py` for index building | Member D | T2.3 | not started |

---

## Phase 3 — Stage 3: Risk Detection Agent

*Members C & D. Depends on Phase 2 completion.*
*Runs in parallel with Phase 4.*

| ID | Task | Description | Assignable to | Depends on | Status |
|----|------|-------------|---------------|-----------|--------|
| T3.1 | Implement `src/stage3_risk_agent/risk_classifier.py` | Fine-tune DeBERTa-base on synthetic risk labels. Input: clause text → Output: risk_level (Low/Medium/High) + confidence. Train/eval split. Hyperparams from `stage3_config.yaml`. | Member C | T2.2, T0.2 | not started |
| T3.2 | Implement `src/stage3_risk_agent/tools.py` | Two LangGraph-compatible tools: (1) `faiss_retrieval(clause_text) → List[SimilarClause]` wrapping `embeddings.py`; (2) `contract_search(clause_id, contract_clauses) → List[RelatedClause]` for related clauses in same document | Member D | T2.3, T0.9 | not started |
| T3.3 | Implement `src/stage3_risk_agent/agent.py` | LangGraph `StateGraph` with typed state dict. Loop: for each clause → call risk classifier → if ambiguous, invoke tools → Mistral-7B-Instruct (4-bit) generates explanation → output `List[RiskAssessedClause]` | Member D | T3.1, T3.2, T0.2 | not started |
| T3.4 | Implement `src/stage3_risk_agent/evaluate.py` | Risk detection metrics: accuracy, precision/recall/F1 per risk level, confusion matrix. Ablation: both tools vs one tool vs no tools. | Member C | T3.3 | not started |
| T3.5 | Write `tests/test_stage3.py` | Mock LLM calls, test tool outputs, test state transitions in agent graph, test risk classifier on small fixture | Member D | T3.3 | not started |
| T3.6 | Create `notebooks/04_stage3_agent_dev.ipynb` | Interactive agent testing, example runs on sample contracts, tool call traces | Member C | T3.3 | not started |

---

## Phase 4 — Stage 4: Report Generation

*Members A & B. Can start as soon as T0.9 is done.*
*Build against mock `RiskAssessedClause` data. Runs in parallel with Phase 3.*

| ID | Task | Description | Assignable to | Depends on | Status |
|----|------|-------------|---------------|-----------|--------|
| T4.1 | Implement `src/stage4_report_gen/aggregator.py` | Pure Python: group clauses by risk level, compute overall risk score, detect missing standard clause types (compare extracted vs expected list), generate summary stats. No model. | Member A | T0.9, T0.3 | not started |
| T4.2 | Implement `src/stage4_report_gen/recommender.py` | Curated lookup dict: `(clause_type, risk_pattern) → remediation_text`. Cover all 41 CUAD clause types × 3 risk levels. Optional LLM fallback for unmapped combinations. | Member B | T0.9 | not started |
| T4.3 | Implement `src/stage4_report_gen/explainer.py` | FLAN-T5-base with prompt templates per risk level. Batch inference for generating human-readable explanations. | Member A | T0.3, T0.9 | not started |
| T4.4 | Implement `src/stage4_report_gen/report_builder.py` | Assembles final `RiskReport` JSON from aggregator + explainer + recommender outputs. Must match Stage 4 output schema in `ARCHITECTURE.md`. | Member B | T4.1, T4.2, T4.3 | not started |
| T4.5 | Implement `src/stage4_report_gen/evaluate.py` | ROUGE scores for generated explanations vs reference, optional human eval framework | Member A | T4.4 | not started |
| T4.6 | Write `tests/test_stage4.py` | Test aggregator with mock data, test lookup table completeness, test report schema, mock FLAN-T5 calls | Member B | T4.4 | not started |

---

## Phase 5 — Integration & End-to-End Evaluation

*All 4 members. Depends on Phases 1–4 being individually working.*

| ID | Task | Description | Assignable to | Depends on | Status |
|----|------|-------------|---------------|-----------|--------|
| T5.1 | Create `scripts/run_pipeline.py` | End-to-end runner: PDF → Stage 1+2 → Stage 3 → Stage 4 → report output. CLI interface with config paths. | Member A | T1.3, T3.3, T4.4 | not started |
| T5.2 | Write `tests/test_preprocessing.py` | Test PDF/DOCX/TXT extraction, text cleaning edge cases | Member B | T0.4 | not started |
| T5.3 | Create `notebooks/05_evaluation.ipynb` | End-to-end eval on CUAD test set, per-stage metrics, ablation results, error analysis | Member C | T1.5, T3.4, T4.5 | not started |
| T5.4 | Create `notebooks/03_stage1_training.ipynb` | Training curves, hyperparameter experiments, comparison table (DeBERTa vs baseline) | Member D | T1.2, T1.5 | not started |
| T5.5 | Integration testing | Run pipeline on 5–10 sample contracts, verify all stage handoffs, validate output report schema | All 4 | T5.1 | not started |

---

## Dependency Flow

```
Phase 0 (all 4, fully parallel)
  T0.1  T0.2  T0.3  T0.4  T0.5  T0.6  T0.7  T0.8  T0.9
                                                      │
              ┌───────────────────────────────────────┤
              │                                       │
              ▼                                       ▼
  Phase 1 (Members A, B)                  Phase 2 (Members C, D)
    T1.1 → T1.2 → T1.5                     T2.1 → T2.2
    T1.1 → T1.3 → T1.6                     T2.1 → T2.3 → T2.5
    T1.4 → T1.6                             T2.1 → T2.4
              │                                       │
              │                                       ▼
              │                          Phase 3 (Members C, D)
              │                            T3.1 → T3.3 → T3.4
              │                            T3.2 → T3.3 → T3.5
              │                                       │
              ▼                                       │
  Phase 4 (Members A, B)                             │
    T4.1 ─┐                                          │
    T4.2 ─┼→ T4.4 → T4.5                            │
    T4.3 ─┘                                          │
              │                                       │
              └───────────────┬───────────────────────┘
                              ▼
                    Phase 5 (All 4)
                    T5.1 → T5.5
                    T5.2, T5.3, T5.4 (parallel)
```

---

## Member Load Summary

| Member | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Total |
|--------|---------|---------|---------|---------|---------|---------|-------|
| **A** | T0.1, T0.4 | T1.1, T1.2, T1.5 | — | — | T4.1, T4.3, T4.5 | T5.1 | 10 tasks |
| **B** | T0.2, T0.5 | T1.3, T1.4, T1.6, T1.7 | — | — | T4.2, T4.4, T4.6 | T5.2 | 11 tasks |
| **C** | T0.3, T0.6 | — | T2.1, T2.2, T2.4 | T3.1, T3.4, T3.6 | — | T5.3 | 9 tasks |
| **D** | T0.7, T0.8, T0.9 | — | T2.3, T2.5 | T3.2, T3.3, T3.5 | — | T5.4 | 9 tasks |

---

## Reference Files

| File | Purpose |
|------|---------|
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Data flow schemas, directory structure, model table — ground truth for all design decisions |
| [.github/copilot-instructions.md](../.github/copilot-instructions.md) | Code generation rules per stage |
| [src/stage1_extract_classify/pipeline.py](../src/stage1_extract_classify/pipeline.py) | Existing code — source for T0.4, T0.5, T1.1, T1.2, T1.3 refactoring |
| [src/stage1_extract_classify/baseline.py](../src/stage1_extract_classify/baseline.py) | Existing code — refactor in T1.4 |
| [src/stage1_extract_classify/evaluate.py](../src/stage1_extract_classify/evaluate.py) | Existing code — refactor in T1.5 |
