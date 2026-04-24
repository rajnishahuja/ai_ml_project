# ⚖️ Legal Contract Risk Analyzer — Developer Guide

An end-to-end AI pipeline for automated legal contract clause extraction, risk classification, and report generation. Built on a **LangGraph** orchestration engine with a **Hybrid Map-Reduce** agentic architecture, locally hosted **DeBERTa** models, **IBM Docling** layout-aware PDF parsing, and a **FAISS** persistent vector store.

---

## 🏗️ 1. Architecture & Flow

The system is a **multi-stage parallel pipeline** that analyzes uploaded PDF contracts. It is orchestrated by **LangGraph** using a Directed Acyclic Graph (DAG) architecture.

### The Pipeline Nodes:
```text
                         ┌──────────┐
                         │  START   │
                         └────┬─────┘
                              ▼
                   ┌─────────────────────┐
                   │  Node A             │
                   │  Stage 1+2          │
                   │  DeBERTa Extraction │
                   │  + Docling Metadata │
                   └──────────┬──────────┘
                              ▼
                   ┌──────────────────────┐
                   │  Node C              │
                   │  Stage 3 Dispatcher  │
                   │  Risk & Confidence   │
                   └──────────┬───────────┘
              (LangGraph Map-Reduce Fan-Out via Send API)
              ── × N clauses dispatched in parallel ──
                              ▼
             ┌────────────────────────────────┐
             │         Node D Worker          │
             │   (one instance per clause)    │
             │────────────────────────────────│
             │  confidence score check        │
             │                                │
             │  ┌─ conf >= 0.6 ───────────────┤
             │  │  FAST PATH                  │
             │  │  precedent_search(k=3)      │
             │  │  + 1 Mistral call (direct)  │
             │  │  label unchanged, trace=[]  │
             │  │                             │
             │  └─ conf < 0.6 ────────────────┤
             │     AGENT PATH                 │
             │     ReAct tool loop (max 5)    │
             │                                │
             │     Tools:                     │
             │     ┌──────────────────────┐   │
             │     │ precedent_search(k=5)│   │
             │     │   vector RAG / FAISS │   │
             │     ├──────────────────────┤   │
             │     │ contract_search(id)  │   │
             │     │   struct. same-doc   │   │
             │     │   clause lookup      │   │
             │     └──────────────────────┘   │
             │     → final_label              │
             │       (may override DeBERTa)   │
             └────────────────┬───────────────┘

                  (Map-Reduce Fan-In — operator.add reducer)
                              ▼
                 ┌────────────────────────┐
                 │  Node E                │
                 │  Stage 4 – Report Gen  │
                 │  Final JSON Report     │
                 └────────────┬───────────┘
                              ▼
                           ┌──────┐
                           │ END  │
                           └──────┘
```


| Node | Stage | Purpose | Tech Stack | Status |
|------|-------|---------|------------|--------|
| **A** | Stage 1+2 | Extracts legal clauses from PDFs | Local DeBERTa | ✅ Production |
| **C** | Stage 3 | Classifies risk severity (Dispatcher) | Local DeBERTa | 🟡 Mock (Pending) |
| **D** | Stage 3 | Generates risk explanations (Map-Reduce Workers) | Ollama (Mistral-7B) | 🟡 Mock (Pending) |
| **E** | Stage 4 | Aggregates clauses into JSON report (recommendations + score are placeholder stubs) | Deterministic | 🟡 Partial (Stub) |

*(Run `uv run python src/workflow/graph.py` to render this DAG dynamically in your console.)*

---

## 🗺️ 2. Code Navigation Map

The repository strictly separates the **Web Server (FastAPI)** from the **Machine Learning Pipeline (LangGraph)**. Below is the exact map of the codebase. 

*(Note: Files marked as **Empty - Planned** are architectural skeletons ready for your future code contributions).*

### 🌐 The Web API Layer (`app/`)
Handles all incoming HTTP requests, payloads, and authentication.
```text
app/
├── main.py                         # FastAPI app entry point
├── routers/
│   ├── stage1_extract.py           # POST /api/v1/stage1/analyze (Main trigger)
│   ├── documents.py                # GET /api/v1/documents (FAISS explorer)
│   ├── stage3_agent.py             # (Empty - Planned) Standalone Mistral RAG chat endpoints
│   └── stage4_report.py            # (Empty - Planned) API to retrieve/regenerate past reports
├── schemas/
│   ├── domain.py                   # Core Pydantic models (ExtractedClause, FinalRiskReport)
│   ├── requests.py                 # (Empty - Planned) Custom request payload validations
│   └── responses.py                # (Empty - Planned) Custom API response formats
├── services/
│   ├── stage1_extract_svc.py       # DeBERTa inference service wrapper
│   ├── stage3_agent_svc.py         # (Empty - Planned) Service to wrap isolated RAG prompts
│   └── stage4_report_svc.py        # (Empty - Planned) Service to format/export reports as PDF
└── dependencies/
    ├── auth.py                     # (Empty - Planned) JWT / API Key authentication middleware
    └── config.py                   # (Empty - Planned) Pydantic environment configuration class
```

### 🧠 The Intelligence Layer (`src/`)
Contains the LangGraph nodes, ML models, and core extraction logic.
```text
src/
├── workflow/                        # The Brain: Connects all stages together
│   ├── state.py                    # RiskAnalysisState TypedDict (Data flowing between nodes)
│   └── graph.py                    # DAG builder: Compiles all nodes and edges
│
├── stage1_extract_classify/         # Node A: Clause Extraction
│   ├── model.py                    # Hardware-agnostic DeBERTa inference engine
│   ├── preprocessing.py            # PDF/DOCX/TXT text extraction utilities
│   ├── nodes.py                    # LangGraph node execution logic
│   └── (baseline/train/evaluate).py# (Empty - Legacy) Removed monolithic ML research scripts
│
├── stage3_risk_agent/               # Nodes C, D: Risk Analysis & Vector DB
│   ├── embeddings.py               # FAISS vector store logic (Chunking, Syncing)
│   ├── nodes.py                    # LangGraph node execution logic (Fast Path + Agent Path)
│   ├── risk_classifier.py          # (Empty - Planned) Wrapper for fine-tuned DeBERTa sequence classifier
│   ├── rag_retriever.py            # (Empty - Planned) LangChain logic for FAISS context retrieval
│   ├── tools.py                    # (Empty - Planned) precedent_search + contract_search tool definitions
│   └── synthetic_labels.py         # (Empty - Planned) Script for auto-labeling training datasets
│
├── stage4_report_gen/               # Node E: Report Aggregation
│   ├── nodes.py                    # LangGraph node execution logic
│   ├── aggregator.py               # (Empty - Planned) Logic to summarize clauses into metrics
│   ├── explainer.py                # (Empty - Planned) Formatter for Mistral's risk explanations
│   ├── recommender.py              # (Empty - Planned) Risk-to-Advice mapping logic
│   ├── report_builder.py           # (Empty - Planned) Logic to assemble the final unified JSON
│   └── generator.py                # (Empty - Planned) PDF/Word export generation engine
│
└── common/                          # Shared cross-stage logic
    ├── constants.py                # Global pipeline variables (Thresholds, 41 Clause Types)
    └── data_loader.py / utils.py   # (Empty - Planned) Reusable cross-stage helper functions
```

### 💾 Data & Storage (`data/`)
This repository includes **pre-generated artifacts** so developers can test APIs without running heavy ML extractions:
- `data/faiss_index/`: Pre-built Vector Database. You can immediately call the `GET /api/v1/documents/` endpoints.
- `data/output/`: Sample JSON responses for UI frontend integration.
- `data/raw/`: Temporary upload storage (auto-cleaned after processing).

---

## ⚡ 3. Quick Start Guide

**1. Install Dependencies**
This project uses `uv` for lightning-fast package management.
```bash
git clone <repo-url>
cd AIML_project
uv sync
```

**2. Configure Environment**
```bash
cp .env.example .env
```
Edit `.env` (Requires local DeBERTa weights in `./stage1_2_deberta/`):
```env
LOCAL_MODEL_PATH="./stage1_2_deberta"
OLLAMA_HOST="http://localhost:11434"
OLLAMA_EMBED_MODEL="all-minilm:l6-v2"
```

**3. Run Local Ollama Models**
```bash
ollama serve
ollama pull all-minilm:l6-v2
```

**4. Start the API Server**
```bash
uv run uvicorn app.main:app --reload
```
*Note: The pipeline natively uses **IBM Docling** by default for highly-accurate, layout-aware PDF parsing. If you wish to fallback to the legacy, faster `pdfplumber` engine, start the server with `USE_PDFPLUMBER=true`.*

The API is now live at **`http://127.0.0.1:8000`**. View the Swagger UI at `/docs`.

---

## 🗂️ 4. Data Flow (State Schema)

Every node reads and writes to a shared `RiskAnalysisState` object defined in `src/workflow/state.py`. If you are building a new feature, you will interact with this state:

```python
class RiskAnalysisState(TypedDict):
    contract_text: str                          # Raw text extracted from uploaded file
    document_id: str                            # UUID generated per upload (collision-safe)
    extracted_clauses: List[ExtractedClause]    # [Node A] DeBERTa extracted + page metadata
    flagged_clauses: List[Dict]                 # [Node C] Clauses with risk level + confidence
    risk_assessed_clauses: Annotated[List, operator.add]  # [Node D] Parallel worker outputs (Map-Reduce reducer)
    final_report: Dict                          # [Node E] Compiled JSON report
```

> **Map-Reduce Design:** `risk_assessed_clauses` uses `Annotated[List, operator.add]` as its reducer. This allows LangGraph to safely merge results from parallel Node D workers without race conditions.

---

## 📡 5. API Endpoints & Signals

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
  "document_id": "3de21a56-b808-486e-a801-e04f113d5e01",
  "faiss_sync_status": null,
  "extracted_clauses": [
    {
      "clause_id": "3de21a56..._Non-Compete_0009",
      "clause_text": "Beginning on the Launch Date and continuing during the Term...",
      "clause_type": "Non-Compete",
      "start_pos": 9852,
      "end_pos": 10129,
      "confidence": 0.9856,
      "confidence_logit": 8.4548,
      "page_no": "2",
      "content_label": "list_item"
    }
  ],
  "risk_assessed_clauses": [
    {
      "clause_id": "3de21a56..._Non-Compete_0009",
      "clause_type": "Non-Compete",
      "page_no": "2",
      "content_label": "list_item",
      "risk_level": "MEDIUM",
      "risk_reason": "[AGENT PATH - 2 tool call(s)] Mistral assessed 'Non-Compete' via ReAct loop. Final label: MEDIUM [DeBERTa overridden — see agent_trace]",
      "similar_clauses": [],
      "cross_references": ["precedent_search", "contract_search"],
      "overridden": true
    },
    {
      "clause_id": "3de21a56..._License_Grant_0024",
      "clause_type": "License Grant",
      "page_no": "2",
      "content_label": "list_item",
      "risk_level": "MEDIUM",
      "risk_reason": "[FAST PATH] DeBERTa classified 'License Grant' as MEDIUM risk (conf=0.95). (3 precedent(s) retrieved).",
      "similar_clauses": [
        {"text": "[MOCK precedent 1]...", "risk": "LOW",  "similarity": 0.95},
        {"text": "[MOCK precedent 2]...", "risk": "MEDIUM", "similarity": 0.88},
        {"text": "[MOCK precedent 3]...", "risk": "HIGH", "similarity": 0.81}
      ],
      "cross_references": [],
      "overridden": false
    }
  ],
  "final_report": {
    "summary": "This contract contains 20 assessed clause(s). 4 HIGH, 10 MEDIUM, 6 LOW risk.",
    "high_risk": [
      {
        "clause_id": "3de21a56..._Anti-Assignment_0017",
        "clause_type": "Anti-Assignment",
        "page_no": "7",
        "content_label": "text",
        "explanation": "[FAST PATH] DeBERTa classified 'Anti-Assignment' as HIGH risk (conf=0.95). (3 precedent(s) retrieved).",
        "recommendation": "Renegotiate to achieve mutual and balanced protections."
      }
    ],
    "medium_risk": [...],
    "low_risk_summary": "6 clause(s) assessed as standard / low risk.",
    "missing_protections": [],
    "overall_risk_score": 20.0,
    "total_clauses": 20
  }
}
```

---

## ⚙️ 6. Advanced Developer Notes

### Inference Optimizations (Stage 1)
If you are working on ML inference, `src/stage1_extract_classify/model.py` has been highly optimized:
- **Hardware-Agnostic:** Dynamically detects `cuda` (NVIDIA), `mps` (Apple Silicon), or `cpu`.
- **Micro-Batched Tokenization:** Slices text into concurrent batches of 32 sequences for zero-latency throughput using `overflow_to_sample_mapping`.
- **Automated Mixed Precision:** Uses `bfloat16` to prevent `float16` overflow crashes while halving GPU memory usage.

### FAISS Vector Database Safety
- **Multi-Tenant Safe:** Every uploaded file is assigned a `uuid4()`. Every FAISS chunk is tagged with this UUID to prevent context collision across different contracts.
- **Persistent:** Deleting `data/faiss_index/` will safely wipe the database; it reconstructs dynamically.

### Docling Layout-Aware Metadata Propagation
Because DeBERTa runs on flat string text, spatial context (page, layout) is inherently lost during extraction. `src/stage1_extract_classify/model.py` implements a **heuristic forward-match reverse-lookup** against the raw Docling JSON saved to `data/processed/docling_outputs/{uuid}.json`:
- **`page_no`**: Exact PDF page number (or range like `"4-5"` for multi-page clauses). Uses 1-based page numbering matching Docling's native format.
- **`content_label`**: Layout element type — `text`, `list_item`, `table`, or `section_header`.
- **Match Algorithm:** Normalizes both the clause fragment and Docling block text to alphanumeric-only lowercase, then uses forward-only substring matching with an 8-character minimum guard to prevent false positives from short tokens.
- **Benefits:** Enables lawyers to navigate directly to the clause in the physical PDF and gives the Mistral Agent layout context (e.g., an indemnification clause buried inside a `table`).

### Map-Reduce Agentic Worker Pool
The pipeline uses LangGraph's `Send` API (in `src/workflow/graph.py`) to dynamically dispatch one parallel worker per clause:
- **Node C (Dispatcher):** Scores clauses and assigns `classifier_confidence`. Confidence `>= 0.6` → **Fast Path**. Confidence `< 0.6` → **Agent Path**.
- **Node D (Worker Pool):** Each worker runs independently in parallel, processing exactly one clause.
- **Reducer:** `risk_assessed_clauses` uses `Annotated[List, operator.add]` for thread-safe fan-in aggregation.

### Two-Path Processing (Node D)

**Fast Path (`conf >= 0.6`)** — Deterministic, ~1 LLM call:
1. `precedent_search(k=3)` — FAISS retrieves 3 similar labeled precedent clauses for context.
2. Single Mistral forward pass generates explanation grounded in clause + precedents.
3. Returns DeBERTa's label unchanged. `agent_trace: []`.

**Agent Path (`conf < 0.6`)** — Non-deterministic, 2–5 LLM calls:
1. Mistral is invoked as a full **ReAct tool-calling agent** (max 5 turns).
2. Two tools available:
   - **`precedent_search(clause_text, k=5)`** — FAISS vector similarity search. Returns top-K similar clauses `{clause_text, clause_type, risk_level, risk_reason, similarity}`.
   - **`contract_search(document_id)`** — Structured lookup of all typed clauses already extracted from the **same contract** by Stage 1+2. Purpose: resolve cross-references (e.g., "IP was already assigned in another clause"). No embeddings or LLM calls inside.
3. Mistral reasons over tool results and outputs structured JSON with `final_label`, `explanation`, and `override_reason`.
4. **Can override DeBERTa's preliminary label** if evidence contradicts it.

> **Current Status:** Both paths are **architecturally stubbed** with deterministic mocks. `precedent_search` returns `[]`, `contract_search` returns `[]`, and the label/explanation are hardcoded strings. These stubs will be replaced with real `ChatOllama` + FAISS + structured output calls once the DeBERTa risk classifier training completes.

### Pending Model Integration
Two components are currently mocked pending real model availability:
| Component | Node | Mock Behavior | Will Be Replaced With |
|-----------|------|---------------|-----------------------|
| DeBERTa Risk Classifier | Node C | Cycles `HIGH/MEDIUM/LOW` + `0.95/0.30` confidence | Fine-tuned DeBERTa sequence classifier (training in progress) |
| Mistral Fast Path | Node D | Hardcoded explanation string | `precedent_search(k=3)` + `ChatOllama` single call |
| Mistral Agent Path | Node D | Hardcoded explanation string | Full ReAct agent: `precedent_search` + `contract_search` + `ChatOllama` tool loop |

> [!NOTE]
> **Mock Confidence Cycle Artifact:** In the current mock, Node C assigns `conf=0.95` and `conf=0.30` alternately across clauses (odd index → Fast Path, even index → Agent Path). This means every second clause will show `overridden: true` and `cross_references: ["precedent_search", "contract_search"]` — this is expected mock behaviour, **not a bug**.
>
> Once the real DeBERTa risk classifier is connected, each clause will receive its actual predicted risk level and confidence score. Only genuinely low-confidence clauses (< 0.6) will enter the Agent Path and run tool calls.
