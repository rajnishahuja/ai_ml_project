# Legal Contract Risk Analyzer — Architecture Reference

> **Purpose**: This file is the single source of truth for AI assistants (Copilot, Claude, GPT) working on this codebase. Open this file or reference it (`#file:ARCHITECTURE.md`) when starting any coding session.

## Project Summary

A modular ML pipeline that analyzes legal contracts and flags risky clauses. Takes a contract PDF/text as input, extracts and classifies clauses, assesses risk using an agent with RAG, and generates a structured risk report.

## Architecture (3 Stages)

```
Contract PDF/Text
       │
       ▼
┌─────────────────────────────┐
│  Stage 1+2: Extract &       │  Single DeBERTa-base model
│  Classify (combined)        │  CUAD dataset, QA format
│  Output: clause objects     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 3: Risk Detection    │  LangGraph agent
│  Agent with RAG             │  DeBERTa-base (risk classifier)
│  Tools: FAISS retrieval,    │  Mistral-7B-Instruct (explanations)
│         contract search     │  all-MiniLM-L6-v2 (embeddings)
│  Output: risk-assessed      │  FAISS vector store
│          clause objects     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 4: Report Generation │  Python aggregation code
│  Hybrid: code + LLM        │  FLAN-T5-base (explanations)
│  Output: structured risk    │  Lookup table (recommendations)
│          report             │
└─────────────────────────────┘
```

## Stage 3 Design Options (To Be Decided — Post DeBERTa Training)

> **Immediate priority**: Train DeBERTa risk classifier first. Evaluate accuracy. Stage 3 agent design follows.
> **Project goal**: Learn agentic RAG — the chosen design should exercise agent tool-calling and reasoning, not just a static pipeline.

### Option A — Static LangGraph Pipeline (deterministic)
LangGraph orchestrates fixed sequential steps. Predictable, auditable, good for academic defense.
```
Clause + METADATA
    → RAG (FAISS top-5 similar clauses)
    → Contract Search (cross-referenced clause types)
    → DeBERTa (risk_level + confidence)
    → Mistral (explanation using clause + risk_level + RAG + METADATA)
    → output
```
- Pro: reproducible, easy to debug, every step auditable
- Con: no dynamic reasoning — DeBERTa's label is always final regardless of confidence
- METADATA injected into Mistral's explanation context (not DeBERTa's input)

### Option B — Reasoning Model as Brain (agentic) ← preferred direction
A reasoning model (Mistral/Qwen) controls all tools autonomously via LangGraph tool nodes.
DeBERTa is exposed as a tool the agent calls to get a calibrated classification signal.
```
Clause + METADATA
    → Reasoning Model decides tool calls:
        - RAG tool (retrieve similar clauses)
        - Contract Search tool (cross-references)
        - DeBERTa tool (specialist classifier — always called for final verdict)
    → Reasoning Model synthesizes: DeBERTa label + context + METADATA
    → final risk_level + explanation
```
- Pro: agent exercises multi-hop reasoning; METADATA naturally informs label interpretation;
  low-confidence DeBERTa scores can be escalated or cross-checked dynamically
- Con: non-deterministic, higher latency, more complex to debug
- DeBERTa remains the authoritative classification signal — reasoning model qualifies, not overrides

### Key Shared Decision (both options)
- **DeBERTa input**: clause_text + clause_type only (trained on these features)
- **METADATA** (parties, agreement type, dates): flows to reasoning/explanation model, not DeBERTa
- **Confidence threshold**: if DeBERTa confidence < 0.6, flag clause for human review
- **Training first**: evaluate DeBERTa accuracy before committing to agent complexity

## Open Design Question — METADATA in DeBERTa Training (Discuss After Labeling)

> **Context**: During manual label review, we found that many HIGH↔LOW flips between Qwen and Gemini
> are caused entirely by not knowing who the signing party is — both models reason correctly about the
> clause text but assume different parties. Without METADATA, DeBERTa faces the same ambiguity.

### The problem
The same clause text can be LOW or HIGH depending on who signed:
- "Vendor grants AT&T perpetual irrevocable license" → HIGH if Vendor signs, LOW if AT&T signs
- No amount of fine-tuning on clause text alone resolves this

### Three options to evaluate after DeBERTa baseline:

**Option 1 — clause_type only (current plan, baseline)**
`[CLS] clause_type [SEP] clause_text [SEP]`
- Partial signal — clause type hints at risk direction but doesn't resolve signing party
- Establishes baseline accuracy ceiling

**Option 2 — Add party role tag at training + inference time**
`[CLS] signing_party_role=licensor [SEP] clause_type [SEP] clause_text [SEP]`
- Stage 1+2 extracts METADATA (Parties field) and infers role (licensor/licensee, vendor/customer)
- DeBERTa trained with role tag → directly resolves signing party ambiguity
- Requires role inference logic from contract METADATA

**Option 3 — Reasoning model resolves METADATA ambiguity (Option B architecture)**
- DeBERTa classifies on text alone, will be uncertain on party-dependent cases
- Low-confidence predictions escalated to reasoning model
- Reasoning model uses METADATA to confirm or override DeBERTa's label
- No retraining needed — handles ambiguity at inference time

**Recommendation**: Run Option 1 first as baseline. If accuracy is limited by signing-party ambiguity,
evaluate Option 2 (cleaner training signal) vs Option 3 (no retraining, more flexible).

## Labeling Review Learnings (Discuss After DeBERTa Baseline)

These patterns emerged from manual review of 239 HIGH↔LOW flip cases and should inform
model design, RAG strategy, and agent reasoning.

### 1. Signing Party Ambiguity is the #1 Label Driver
81% of Rajnish's 60 rows were LOW — not because the clauses were genuinely low risk,
but because the signing party turned out to be the *recipient* of rights, not the grantor.
Both Qwen and Gemini reasoned correctly about clause text but assumed different parties.
**Implication**: A model trained on clause text alone will have a hard accuracy ceiling on
IP Ownership, License Grant, and Affiliate License clause types specifically.

### 2. Cross-Contract Clause Interaction Changes Labels
Several clauses were correctly assessed only after reading *other clauses* in the same contract:
- MR-095 (IP assistance clause): looked HIGH alone, but other clauses showed IP already
  assigned elsewhere — this was just administrative paperwork → LOW
- MR-234 (1-copy software restriction): looked restrictive alone, but full production license
  existed in another clause — test/backup restriction was supplemental → LOW
**Implication**: RAG retrieval should include *same-contract* clause context, not just
similar clauses from other contracts. The contract search tool in Stage 3 is critical.

### 3. Clause Type Labels Are Noisy at the Boundary
- Volume Restriction includes supply guarantees, image size limits, content duration caps —
  not just purchase minimums. Same clause type, very different risk patterns.
- Non-Transferable License ranges from LOW (standard distribution) to MEDIUM (M&A constraint)
  depending on deal context (affiliate carve-outs, term length, corporate structure).
**Implication**: clause_type alone is a weak feature. The RAG examples need to be diverse
within each type — DeBERTa must learn intra-type variation, not just type-level patterns.

### 4. Mutual vs One-Sided is a Strong Signal
For Uncapped Liability: mutual consequential damage exclusions are almost always LOW.
One-sided caps (only one party's liability limited) trend MEDIUM-HIGH.
**Implication**: "mutual" / "either party" / "both parties" keywords are strong LOW signals
for liability clause types. Worth checking if DeBERTa learns this or needs explicit feature.

### 5. Context-Dependent Clauses Need Confidence Flagging
Some clauses were genuinely ambiguous even with METADATA (MR-094 truncated GSK clause,
MR-093 spin-off restructuring). These warrant MEDIUM and low DeBERTa confidence scores.
**Implication**: confidence threshold + human review escalation path is not optional —
it's needed for a reliable production system.

## Directory Structure

```
AIML_project/
├── ARCHITECTURE.md          ← THIS FILE (AI context reference)
├── README.md                ← Project readme (create when ready)
├── requirements.txt         ← Python dependencies
├── .github/
│   └── copilot-instructions.md  ← Auto-loaded by GitHub Copilot
├── configs/
│   ├── stage1_config.yaml   ← Hyperparams for extraction/classification
│   ├── stage3_config.yaml   ← Agent config, RAG params, model paths
│   └── stage4_config.yaml   ← Report template config
├── docs/
│   ├── LegalAgents.docx                              ← Original project proposal
│   ├── Legal_Contract_Risk_Analyzer_Research_Analysis.docx  ← Research analysis v1
│   └── Legal_Contract_Risk_Analyzer_Project_Plan_v2.docx    ← Finalized plan v2
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── schema.py             ← Shared dataclasses (ClauseObject, RiskAssessedClause, RiskReport)
│   │   ├── preprocessing.py     ← PDF/DOCX text extraction, cleaning
│   │   ├── data_loader.py       ← CUAD dataset loading and formatting
│   │   └── utils.py             ← Shared utilities (config loader, metrics, logging)
│   ├── stage1_extract_classify/
│   │   ├── __init__.py
│   │   ├── model.py             ← DeBERTa QA model wrapper
│   │   ├── train.py             ← Fine-tuning script
│   │   ├── predict.py           ← Inference: contract → clause objects
│   │   ├── baseline.py          ← spaCy + regex baseline
│   │   └── evaluate.py          ← Extraction + classification metrics
│   ├── stage3_risk_agent/
│   │   ├── __init__.py
│   │   ├── agent.py             ← LangGraph agent definition
│   │   ├── tools.py             ← RAG retrieval tool + contract search tool
│   │   ├── risk_classifier.py   ← DeBERTa risk classifier
│   │   ├── embeddings.py        ← FAISS index builder + query
│   │   ├── synthetic_labels.py  ← LLM-based risk label generation
│   │   └── evaluate.py          ← Risk detection metrics + ablation
│   └── stage4_report_gen/
│       ├── __init__.py
│       ├── aggregator.py        ← Deterministic grouping, scoring, missing-clause detection
│       ├── explainer.py         ← FLAN-T5 / LLM explanation generation
│       ├── recommender.py       ← Lookup table + optional LLM recommendations
│       ├── report_builder.py    ← Assembles final report from 3 sub-tasks
│       └── evaluate.py          ← ROUGE + optional human eval
├── data/
│   ├── raw/                     ← Downloaded CUAD dataset files
│   ├── processed/               ← Preprocessed/formatted data
│   ├── synthetic/               ← LLM-generated risk labels
│   └── faiss_index/             ← Built FAISS vector index
├── notebooks/
│   ├── 01_cuad_exploration.ipynb      ← Dataset exploration and stats
│   ├── 02_synthetic_labeling.ipynb    ← Risk label generation + validation
│   ├── 03_stage1_training.ipynb       ← Extraction/classification training
│   ├── 04_stage3_agent_dev.ipynb      ← Agent development and testing
│   └── 05_evaluation.ipynb            ← End-to-end eval and ablation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_stage1.py
│   ├── test_stage3.py
│   └── test_stage4.py
└── scripts/
    ├── download_cuad.py         ← Download CUAD from HuggingFace
    ├── generate_synthetic.py    ← Batch synthetic label generation
    ├── build_faiss_index.py     ← Build FAISS index from labeled clauses
    └── run_pipeline.py          ← End-to-end pipeline runner
```

## Data Contracts (Shared Dataclasses)

All stages communicate through typed Python dataclasses defined in `src/common/schema.py`. These are the single source of truth for inter-stage data formats.

## Data Flow (Detailed)

### Stage 1+2 Input
```json
{
  "document_id": "contract_001",
  "contract_text": "Full contract text extracted from PDF...",
  "file_path": "data/raw/affiliate_agreement.pdf",
  "queries": [
    {
      "clause_type": "Indemnification",
      "question": "Highlight the parts (if any) of this contract related to \"Indemnification\" that should be reviewed by a lawyer. Details: Liability of one party to indemnify the other party (or parties)."
    },
    {
      "clause_type": "Termination For Convenience",
      "question": "Highlight the parts (if any) of this contract related to \"Termination For Convenience\" that should be reviewed by a lawyer. Details: Can a party terminate this contract without cause ('at will')?"
    }
  ]
}
```

> There are 41 queries total, one per CUAD clause type. The question templates come directly from the CUAD dataset. Each question asks the model to locate a specific clause type. An empty answer means that clause type is absent from this contract.

### Stage 1+2 Output → Stage 3 Input
```json
[
  {
    "clause_id": "contract_001_Indemnification_0030",
    "document_id": "contract_001",
    "clause_text": "Contractor shall indemnify Company against all claims...",
    "clause_type": "Indemnification",
    "start_pos": 4521,
    "end_pos": 4687,
    "confidence": 0.94
  }
]
```

> One object per detected clause. Clauses with confidence below threshold are excluded. Clause types with empty model answers (clause absent) are omitted.

### Stage 3 Output → Stage 4 Input
```json
[
  {
    "clause_id": "contract_001_Indemnification_0030",
    "document_id": "contract_001",
    "clause_text": "Contractor shall indemnify Company against all claims...",
    "clause_type": "Indemnification",
    "risk_level": "HIGH",
    "risk_explanation": "One-sided indemnification covering counterparty negligence",
    "similar_clauses": [
      {"text": "...", "risk_level": "HIGH", "similarity": 0.92},
      {"text": "...", "risk_level": "LOW", "similarity": 0.87}
    ],
    "cross_references": ["contract_001_Cap_On_Liability_0034"],
    "confidence": 0.88,
    "agent_trace": [
      {"tool": "faiss_retrieval", "result_count": 5},
      {"tool": "contract_search", "related_clauses": 2}
    ]
  }
]
```

> `risk_explanation` is generated by Mistral-7B-Instruct. `agent_trace` records which tools the LangGraph agent invoked for this clause (useful for debugging and ablation).

### Synthetic Risk Labels (Training Data for Stage 3)
```json
{
  "clause_text": "Contractor shall indemnify Company against all claims...",
  "clause_type": "Indemnification",
  "risk_level": "HIGH",
  "risk_reason": "One-sided indemnification covering counterparty negligence",
  "labeled_by": "qwen-32b"
}
```

> Generated by prompting Qwen-32B (primary) or Gemini/OpenAI (backup) on each CUAD clause. Stored in `data/synthetic/risk_labels.json`. Validated via notebooks before use.

### Stage 4 Output (Final Report)
```json
{
  "document_id": "contract_001",
  "summary": "This vendor agreement contains 23 clauses across 14 categories...",
  "high_risk": [
    {
      "clause_id": "contract_001_Indemnification_0030",
      "clause_type": "Indemnification",
      "risk_level": "HIGH",
      "explanation": "The indemnification provision requires...",
      "recommendation": "Renegotiate to mutual indemnification..."
    }
  ],
  "medium_risk": [
    {
      "clause_id": "contract_001_Non_Compete_0009",
      "clause_type": "Non-Compete",
      "risk_level": "MEDIUM",
      "explanation": "Geographic scope is broad but time-limited...",
      "recommendation": "Consider narrowing geographic restriction..."
    }
  ],
  "low_risk_summary": "15 clauses were assessed as standard/low risk...",
  "missing_protections": ["Data Protection", "Force Majeure"],
  "overall_risk_score": 6.8,
  "total_clauses": 23,
  "metadata": {
    "generated_at": "2026-04-08T12:00:00Z",
    "models_used": {
      "extraction": "microsoft/deberta-base",
      "risk_classification": "models/stage3_risk_deberta",
      "explanation": "mistralai/Mistral-7B-Instruct-v0.3",
      "report_explanation": "google/flan-t5-base"
    }
  }
}
```

## Key Models

| Model | Stage | HuggingFace ID | Purpose | VRAM |
|-------|-------|---------------|--------|------|
| DeBERTa-base | 1+2 | `microsoft/deberta-base` | QA extraction + classification | ~2 GB (train ~8 GB) |
| DeBERTa-base | 3 | `microsoft/deberta-base` | Risk classification (fine-tuned on synthetic labels) | ~2 GB (train ~8 GB) |
| Mistral-7B-Instruct | 3 | `mistralai/Mistral-7B-Instruct-v0.3` | Risk explanation generation (4-bit quantized) | ~5 GB |
| all-MiniLM-L6-v2 | 3 | `sentence-transformers/all-MiniLM-L6-v2` | Clause embeddings for FAISS | ~0.5 GB |
| FLAN-T5-base | 4 | `google/flan-t5-base` | Report explanation generation | ~1 GB |
| Qwen-32B-Instruct | 3 (data prep) | `Qwen/Qwen2.5-32B-Instruct` | Synthetic risk label generation (primary) | API or ~20 GB (4-bit) |
| spaCy en_core_web_sm | 1+2 | N/A (pip) | Baseline comparison | negligible |

## Key Datasets

| Dataset | Source | Format | Usage |
|---------|--------|--------|-------|
| CUAD (SQuAD format) | `huggingface: kenlevine/CUAD` | Nested JSON: `ds[0]['data'][i]['paragraphs'][0]['qas']` | Primary dataset for extraction, classification, and risk base |
| Synthetic risk labels | Generated via Qwen-32B (primary), Gemini/OpenAI (backup) | JSON: `data/synthetic/risk_labels.json` | Risk level labels (LOW/MEDIUM/HIGH + reason) for CUAD clauses |

**CUAD details:** 510 legal contracts, 41 clause types per contract, ~20,910 QA pairs total. Each QA pair has a question ("Highlight the parts related to X..."), the full contract text as context, and an answer span (or empty if clause absent). ~60-70% of QA pairs have empty answers. License: CC BY 4.0.

**Note:** `theatticusproject/cuad` on HuggingFace serves raw PDFs (requires pdfplumber). Use `kenlevine/CUAD` for the SQuAD-format JSON used for training.

## Config File Schemas

### `configs/stage1_config.yaml`
```yaml
model_name: microsoft/deberta-base
max_seq_length: 512
doc_stride: 128
batch_size: 8
learning_rate: 2.0e-5
epochs: 3
output_dir: models/stage1_2_deberta
confidence_threshold: 0.01
dataset: kenlevine/CUAD
fp16: true
```

### `configs/stage3_config.yaml`
```yaml
risk_classifier:
  model_name: microsoft/deberta-base
  output_dir: models/stage3_risk_deberta
  learning_rate: 2.0e-5
  epochs: 5
  batch_size: 16
embedding_model: sentence-transformers/all-MiniLM-L6-v2
faiss_index_path: data/faiss_index/clauses.index
explanation_model: mistralai/Mistral-7B-Instruct-v0.3
quantization: 4bit
agent_max_iterations: 3
similarity_threshold: 0.75
similarity_top_k: 5
synthetic_labels:
  model: Qwen/Qwen2.5-32B-Instruct
  backup: gemini/openai
  output_path: data/synthetic/risk_labels.json
```

### `configs/stage4_config.yaml`
```yaml
explanation_model: google/flan-t5-base
risk_thresholds:
  high: 0.75
  medium: 0.40
output_format: json
max_explanation_length: 200
```

## Known Issues & Limitations

1. **Classification evaluation is misleading in existing code.** The current `evaluate.py` infers `pred_type = true_type if pred_text else "NO_CLAUSE"`. This means any non-empty answer is counted as correctly classifying the clause type (since the type is derived from the question, not the model). This inflates classification accuracy. The proper evaluation should measure whether the model correctly identifies clause presence vs absence per type.

2. **CUAD class imbalance.** ~60-70% of QA pairs have empty answers (clause absent). Models may learn to predict empty spans by default. Training should handle this imbalance (e.g., balanced sampling or adjusted loss).

3. **Long contracts exceed 512 tokens.** CUAD contracts average 10K-50K characters. The sliding window approach (`doc_stride=128`) handles this but means one clause may span multiple windows. Post-processing must deduplicate overlapping predictions.

## Conventions

- Python 3.10+
- Type hints on all function signatures
- Configs in YAML, loaded via `configs/` directory
- All model paths configurable (local or HuggingFace hub)
- Use `logging` module, not print statements
- Shared data contracts in `src/common/schema.py` — all stages import from here
- One consolidated `requirements.txt` (no per-stage requirement files)
- Docstrings on all public functions
- Tests mirror src structure in `tests/`
