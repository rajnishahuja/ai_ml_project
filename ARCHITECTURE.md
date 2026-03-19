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
│   │   ├── preprocessing.py     ← PDF/DOCX text extraction, cleaning
│   │   ├── data_loader.py       ← CUAD dataset loading and formatting
│   │   └── utils.py             ← Shared utilities
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

## Data Flow (Detailed)

### Stage 1+2 Input
```json
{
  "contract_text": "Full contract text extracted from PDF...",
  "queries": [
    "Highlight the parts that discuss indemnification",
    "Highlight the parts that discuss termination",
    ... (41 queries, one per CUAD clause type)
  ]
}
```

### Stage 1+2 Output → Stage 3 Input
```json
[
  {
    "clause_id": "c_001",
    "clause_text": "Contractor shall indemnify Company against all claims...",
    "clause_type": "Indemnification",
    "start_pos": 4521,
    "end_pos": 4687,
    "confidence": 0.94
  },
  ...
]
```

### Stage 3 Output → Stage 4 Input
```json
[
  {
    "clause_id": "c_001",
    "clause_text": "Contractor shall indemnify Company against all claims...",
    "clause_type": "Indemnification",
    "risk_level": "HIGH",
    "risk_reason": "One-sided indemnification covering counterparty negligence",
    "similar_clauses": [
      {"text": "...", "risk": "HIGH", "similarity": 0.92},
      {"text": "...", "risk": "LOW", "similarity": 0.87}
    ],
    "cross_references": ["c_015"],
    "confidence": 0.88
  },
  ...
]
```

### Stage 4 Output (Final Report)
```json
{
  "summary": "This vendor agreement contains 23 clauses across 14 categories...",
  "high_risk": [
    {
      "clause_id": "c_001",
      "explanation": "The indemnification provision requires...",
      "recommendation": "Renegotiate to mutual indemnification..."
    }
  ],
  "medium_risk": [...],
  "low_risk_summary": "15 clauses were assessed as standard/low risk...",
  "missing_protections": ["Data Protection", "Force Majeure"],
  "overall_risk_score": 6.8,
  "total_clauses": 23
}
```

## Key Models

| Model | Stage | HuggingFace ID | Purpose |
|-------|-------|---------------|---------|
| DeBERTa-base | 1+2 | `microsoft/deberta-base` | QA extraction + classification |
| DeBERTa-base | 3 | `microsoft/deberta-base` | Risk classification (fine-tuned on synthetic labels) |
| Mistral-7B-Instruct | 3 | `mistralai/Mistral-7B-Instruct-v0.3` | Risk explanation generation |
| all-MiniLM-L6-v2 | 3 | `sentence-transformers/all-MiniLM-L6-v2` | Clause embeddings for FAISS |
| FLAN-T5-base | 4 | `google/flan-t5-base` | Report explanation generation |
| spaCy en_core_web_sm | 1+2 | N/A (pip) | Baseline comparison |

## Key Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| CUAD | `huggingface: theatticusproject/cuad` | Primary dataset for extraction, classification, and risk base |
| Synthetic risk labels | Self-generated via LLM API | Risk level labels (Low/Med/High) for all CUAD clauses |

## Conventions

- Python 3.10+
- Type hints on all function signatures
- Configs in YAML, loaded via `configs/` directory
- All model paths configurable (local or HuggingFace hub)
- Use `logging` module, not print statements
- Docstrings on all public functions
- Tests mirror src structure in `tests/`
