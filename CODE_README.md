# ⚖️ Legal Contract Risk Analyzer

An end-to-end AI pipeline for automated legal contract clause extraction, risk classification, and report generation. Built on a **LangGraph** orchestration engine, locally hosted **DeBERTa** models, **Ollama** embeddings, and a **FAISS** persistent vector store.

---

## 🏗️ Architecture Overview

The system is a **multi-stage parallel pipeline** invoked on every PDF upload. Each stage is a named node in a LangGraph DAG:

```
                         ┌──────────┐
                         │  START   │
                         └────┬─────┘
               ┌──────────────┴──────────────┐
               ▼                             ▼
   ┌─────────────────────┐       ┌─────────────────────┐
   │  Node A             │       │  Node B             │
   │  Stage 1+2          │       │  Stage 3 – FAISS    │
   │  DeBERTa Extraction │       │  Ollama Embeddings  │
   └──────────┬──────────┘       └──────────┬──────────┘
              ▼                             │
   ┌──────────────────────┐                │
   │  Node C              │                │
   │  Stage 3 – Classifier│                │
   │  DeBERTa Risk Flags  │                │
   └──────────┬───────────┘                │
              └──────────────┬─────────────┘
                             ▼
                ┌────────────────────────┐
                │  Node D               │
                │  Stage 3 – Mistral RAG│
                │  Risk Explanations    │
                └────────────┬──────────┘
                             ▼
                ┌────────────────────────┐
                │  Node E               │
                │  Stage 4 – Report Gen │
                │  Final JSON Report    │
                └────────────┬──────────┘
                             ▼
                          ┌──────┐
                          │ END  │
                          └──────┘
```

| Node | Stage | Model | Status |
|------|-------|-------|--------|
| **A** | 1+2 | DeBERTa (local) | ✅ Production |
| **B** | 3 – FAISS Sync | Ollama `all-minilm:l6-v2` (local) | ✅ Production |
| **C** | 3 – Risk Classifier | DeBERTa (local) | 🟡 Mock |
| **D** | 3 – Mistral RAG | Mistral-7B-Instruct (local via Ollama) | 🟡 Mock |
| **E** | 4 – Report Gen | Deterministic aggregator | ✅ Production logic |

---

## 📁 Project Structure

```text
AIML_project/
├── app/
│   ├── main.py                         # FastAPI app entry point
│   ├── routers/
│   │   ├── stage1_extract.py           # POST /api/v1/stage1/analyze (main pipeline trigger)
│   │   ├── documents.py                # GET /api/v1/documents (FAISS DB explorer)
│   │   ├── stage3_agent.py             # (Empty - Planned) Standalone endpoints to chat manually with the Mistral RAG
│   │   └── stage4_report.py            # (Empty - Planned) API to retrieve/regenerate past reports
│   ├── schemas/
│   │   ├── domain.py                   # Pydantic models: ExtractedClause, RiskAssessedClause, FinalRiskReport
│   │   ├── requests.py                 # (Empty - Planned) Custom request payload validations
│   │   └── responses.py                # (Empty - Planned) Custom API response formats
│   ├── services/
│   │   ├── stage1_extract_svc.py       # DeBERTa inference service wrapper
│   │   ├── stage3_agent_svc.py         # (Empty - Planned) Service to wrap isolated RAG prompts
│   │   └── stage4_report_svc.py        # (Empty - Planned) Service to format/export reports as PDF
│   └── dependencies/
│       ├── auth.py                     # (Empty - Planned) JWT / API Key authentication middleware
│       └── config.py                   # (Empty - Planned) Pydantic environment configuration class
│
├── src/
│   ├── workflow/                        # 🧠 Central LangGraph Orchestration
│   │   ├── state.py                    # RiskAnalysisState TypedDict (shared pipeline state)
│   │   └── graph.py                    # DAG builder: nodes + edges + compilation
│   │
│   ├── stage1_extract_classify/         # Stage 1+2: DeBERTa extraction
│   │   ├── model.py                    # Model loader and inference engine
│   │   ├── preprocessing.py            # PDF/DOCX/TXT text extraction
│   │   ├── nodes.py                    # LangGraph node: node_extract_clauses
│   │   ├── baseline.py / pipeline.py   # (Empty - Legacy) Removed monolithic ML research scripts
│   │   └── train.py / evaluate.py      # (Empty - Legacy) Removed ML training/evaluation files
│   │
│   ├── stage3_risk_agent/               # Stage 3: Risk classification + RAG
│   │   ├── embeddings.py               # FAISS vector store: embed_and_store, get_document_chunks
│   │   ├── nodes.py                    # LangGraph nodes: B (FAISS), C (Classifier), D (Mistral RAG)
│   │   ├── risk_classifier.py          # (Empty - Planned) Wrapper for DeBERTa Risk sequence classifier
│   │   ├── rag_retriever.py            # (Empty - Planned) LangChain logic for FAISS context retrieval
│   │   ├── tools.py                    # (Empty - Planned) Custom tools for the Mistral ReAct agent
│   │   ├── faiss_store.py              # (Empty - Planned) Eventual migration of FAISS logic from embeddings
│   │   └── synthetic_labels.py         # (Empty - Planned) Script for auto-labeling training datasets
│   │
│   ├── stage4_report_gen/               # Stage 4: Final report generation
│   │   ├── nodes.py                    # LangGraph node: node_report_generation
│   │   ├── aggregator.py               # (Empty - Planned) Logic to summarize clauses into metrics
│   │   ├── explainer.py                # (Empty - Planned) Formatter for Mistral's risk explanations
│   │   ├── recommender.py              # (Empty - Planned) Mapping of Risk Types to actionable legal advice
│   │   ├── report_builder.py           # (Empty - Planned) Logic to assemble the final unified JSON
│   │   └── generator.py                # (Empty - Planned) PDF/Word export generation engine
│   │
│   └── common/                          # Shared utilities
│       ├── constants.py                # Global pipeline variables (Thresholds, clause types)
│       └── data_loader.py / utils.py   # (Empty - Planned) Reusable cross-stage helper functions
│
├── data/
│   ├── raw/                            # Temp upload storage (auto-cleaned after processing)
│   ├── faiss_index/                    # Persistent FAISS vector database
│   └── output/                         # Sample API responses for reference
│
├── stage1_2_deberta/                    # ⚠️ Local DeBERTa model weights (not in git)
├── pyproject.toml                       # Dependencies managed with `uv`
├── .env                                 # Local environment config (see .env.example)
└── ARCHITECTURE.md                      # Detailed schema & data flow documentation
```

---

## ⚡ Prerequisites

Before running, ensure the following are installed and ready:

1. **Python 3.11+** with [`uv`](https://github.com/astral-sh/uv) package manager
2. **Ollama** running locally with the embedding model pulled:
   ```bash
   ollama serve
   ollama pull all-minilm:l6-v2
   ```
3. **DeBERTa model weights** placed at `./stage1_2_deberta/` in the project root (contact the team for the model files)

---

## 📦 Sample Data Included

This repository intentionally includes pre-generated artifacts so that code reviewers and developers can test the API components without needing to run a from-scratch extraction:
- **Pre-built FAISS Index** (`data/faiss_index/`): Contains fully vectorized document chunks. You can immediately call the `GET /api/v1/documents/` endpoints to explore the RAG indexing logic without needing to upload and embed a new contract!
- **Sample JSON Outputs** (`data/output/`): Contains the raw JSON responses generated from a full End-to-End pipeline run, useful for UI frontend integration and schema reference.

---

## 🚀 Getting Started

### 1. Clone and Install

```bash
git clone <repo-url>
cd AIML_project

# Install all dependencies using uv
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your local settings:

```env
LOCAL_MODEL_PATH="./stage1_2_deberta"
OLLAMA_HOST="http://localhost:11434"
OLLAMA_EMBED_MODEL="all-minilm:l6-v2"
```

### 3. Start the API Server

```bash
uv run uvicorn app.main:app --reload
```

The API is now live at **`http://127.0.0.1:8000`**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/stage1/analyze` | Upload a contract PDF/DOCX/TXT and trigger the full E2E pipeline |
| `GET` | `/api/v1/documents/` | List all document UUIDs stored in the FAISS vector database |
| `GET` | `/api/v1/documents/{document_id}` | Retrieve all chunked text for a specific stored document |
| `GET` | `/health` | Server health check |
| `GET` | `/docs` | Swagger UI for interactive testing |

---

## 📤 Sample API Response

Upload a PDF to `POST /api/v1/stage1/analyze` and receive:

```json
{
  "status": "success",
  "document_id": "f6e94e15-3e77-445c-80da-145772e00f6a",
  "faiss_sync_status": "Complete",
  "extracted_clauses": [
    {
      "clause_id": "f6e94e15..._Non-Compete_0009",
      "clause_text": "Beginning on the Launch Date...",
      "clause_type": "Non-Compete",
      "start_pos": 9834,
      "end_pos": 10087,
      "confidence": 0.9722,
      "confidence_logit": 7.1068
    }
  ],
  "risk_assessed_clauses": [
    {
      "clause_id": "f6e94e15..._Non-Compete_0009",
      "clause_type": "Non-Compete",
      "risk_level": "HIGH",
      "risk_reason": "One-sided non-compete with no reciprocal obligation.",
      "similar_clauses": [],
      "cross_references": []
    }
  ],
  "final_report": {
    "summary": "This contract contains 19 assessed clause(s). 7 HIGH, 6 MEDIUM, 6 LOW risk.",
    "high_risk": [
      {
        "clause_id": "f6e94e15..._Non-Compete_0009",
        "clause_type": "Non-Compete",
        "explanation": "One-sided non-compete with no reciprocal obligation.",
        "recommendation": "Renegotiate to achieve mutual and balanced protections."
      }
    ],
    "medium_risk": [...],
    "low_risk_summary": "6 clause(s) assessed as standard / low risk.",
    "missing_protections": [],
    "overall_risk_score": 23.5,
    "total_clauses": 19
  }
}
```

---

## 🗂️ Data Flow (State Schema)

Every node in the pipeline reads and writes to a shared `RiskAnalysisState` object:

```python
class RiskAnalysisState(TypedDict):
    contract_text: str           # Raw text extracted from uploaded file
    document_id: str             # UUID generated per upload (collision-safe)
    extracted_clauses: List      # [Node A output] DeBERTa extracted clauses
    faiss_status: str            # [Node B output] "Complete" when vectors synced
    flagged_clauses: List        # [Node C output] Clauses with risk severity flags
    risk_assessed_clauses: List  # [Node D output] Mistral explanations per clause
    final_report: Dict           # [Node E output] Compiled JSON report
```

---

## 🔍 Visualising the Pipeline Graph

Run this in a terminal to render the ASCII DAG in your console:

```bash
uv run python src/workflow/graph.py
```

---

## 🛠️ Development Notes

### FAISS Vector Database
- **Persistent** — all documents are stored in `data/faiss_index/legal_contracts_index/`
- **Cumulative** — new PDFs are **appended** to the existing global index; data from previous sessions is preserved
- **Safe** — every chunk is tagged with its UUID, so multi-tenant retrieval is always isolated
- **Reset** — delete `data/faiss_index/` to wipe all stored vectors; the system rebuilds on the next upload

### Inference Optimizations (Stage 1)
- **Hardware-Agnostic & Crash-Proof**: The `ClauseExtractorClassifier` dynamically detects `cuda` (NVIDIA), `mps` (Apple Silicon), or `cpu`, delegating inference to the native hardware accelerators automatically.
- **Automated Mixed Precision**: Leverages PyTorch's `autocast` using `bfloat16`. This avoids the infamous DeBERTa `float16` attention mask overflow crash while simultaneously halving GPU VRAM requirements and doubling tensor speeds.
- **Micro-Batched Tokenization**: Translates sequential inference loops into highly concurrent batches of 32 sequences, intelligently piecing predictions back together using Hugging Face's `overflow_to_sample_mapping`.

### Mock Stages (Pending Implementation)
- **Node C** (Risk Classifier): Currently cycles HIGH/MEDIUM/LOW for all clauses. Wire a trained DeBERTa sequence classifier here.
- **Node D** (Mistral RAG Explainer): Currently returns hardcoded strings. Wire `Mistral-7B-Instruct` via `langchain-ollama` + FAISS retriever here.

### UUID Safety
Every uploaded file is assigned a `uuid4()` at the routing layer. Both the `document_id` field and the temp file path are prefixed with this UUID, preventing concurrent uploads from colliding.

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `langgraph` | Multi-node DAG orchestration |
| `langchain-community` | FAISS + Ollama integrations |
| `langchain-text-splitters` | Contract chunking for vector indexing |
| `faiss-cpu` | Local vector similarity search |
| `torch` | DeBERTa model inference |
| `transformers` | HuggingFace model loading |
| `pymupdf` / `pdfplumber` | PDF text extraction |
| `grandalf` | Terminal ASCII graph rendering |
| `uv` | Fast Python package management |

---

## 👥 Team Contacts

Reach out to the core ML team for:
- Access to the `stage1_2_deberta/` model weights
- Ollama model configuration guidance
- Stage 3 Mistral prompt engineering tasks (pending)
- Stage 4 FLAN-T5 report synthesis tasks (pending)
