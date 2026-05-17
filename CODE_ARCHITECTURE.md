# Legal Contract Risk Analyzer — Technical Code Architecture Reference

This document serves as the primary technical specification for the **Legal Contract Risk Analyzer** codebase. It outlines the unified single-path agentic pipeline topography, data schemas, recent high-impact updates, and downstream visual layers.

---

## 1. End-to-End Pipeline Topology

The system uses a **Unified Single-Path Prior-Guided Agentic Architecture**. Unlike outdated split architectures, *every* risk-relevant clause flows through the LangGraph ReAct agent. The DeBERTa-v3 ensemble prediction serves as a prior guideline/signal injected directly into the agent's prompt context.

```text
       Contract PDF/Text
              │
              ▼
┌─────────────────────────────┐
│  Stage 1: Document Structure│  IBM Docling Layout Parsing
│  & Layout Parsing           │  Output: raw stage1_output.json and
│  Output: stage1_layout.txt  │          stitched layout text
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 2: Semantic Clause   │  Fine-tuned DeBERTa QA Extraction
│  Span Extraction            │  + Docling Layout-Aware Heuristics
│  Output: stage2_output.json │  Output: List[ClauseObject] with page_no
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 3: Unified Agent     │  LangGraph ReAct Agent Loop
│  Risk Assessment            │  - Prior Signal: DeBERTa-v3-base Ensemble (Ens-F)
│  Tools: precedent_search    │  - Research vectors: nlpaueb/legal-bert-base-uncased (768d)
│         contract_search     │  - LLM Synthesis: Qwen-3-30B (local/OpenAPI-compatible)
│  Output: stage3_output.json │  - Deduplication: Deduplicated precedent cache
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 4: Jinja2 Templating │  Progressive Report Rendering Pipeline
│  & Report Generation        │  - Deterministic lookup tables + LLM Executive Summary
│  Output: final_report.json  │  - Premium pastel styled HTML & PDF dashboards
└─────────────────────────────┘
```

---

## 2. Deep-Dive Stage Architectures & Recent Updates

### 💡 Stage 1: Document Structure & Layout Parsing
*   **Purpose:** Parse long-form contract PDFs and locate raw candidate clauses using document layout structural blocks.
*   **Engine:** Docling Layout Parser (defaults to stitched Markdown; cached locally in `data/output/final/{uuid}/stage1_layout.txt` to enable resume features).
*   **Output:** Layout-stitched raw flat text string `contract_text` passed downstream.

### 🧬 Stage 2: Token-Level QA Extraction & Metadata Propagation
*   **Purpose:** Refine span-level matches and propagate layout-aware bounding indicators.
*   **Engine:** Fine-tuned DeBERTa span-QA model.
*   **Strategy:** Sliding window QA inference mapping 41 category-specific questions.
*   **Docling Layout Heuristics:** Because DeBERTa runs on flat string text, spatial context (page, layout) is inherently lost during extraction. Stage 2 implements a **heuristic forward-match reverse-lookup** against Stage 1's raw Docling JSON (`stage1_output.json`) saved in the final output directory:
    *   `page_no`: Exact PDF page number (or range like `"4-5"` for multi-page clauses). Uses 1-based page numbering matching Docling's native format.
    *   `content_label`: Layout element type — `text`, `list_item`, `table`, or `section_header`.
    *   *Match Algorithm:* Normalizes both the clause fragment and Docling block text to alphanumeric-only lowercase, then uses forward-only substring matching with an 8-character minimum guard to prevent false positives from short tokens.
*   **Inference Optimizations:**
    *   *Hardware-Agnostic:* Dynamically detects `cuda` (NVIDIA), `mps` (Apple Silicon), or `cpu`.
    *   *Micro-Batched Tokenization:* Slices text into concurrent batches of 32 sequences for zero-latency throughput using `overflow_to_sample_mapping`.
    *   *Automated Mixed Precision:* Uses `bfloat16`/`fp16` to prevent `float16` overflow crashes while halving GPU memory usage.
*   **Output Fields:** `ClauseObject` records with `clause_id`, `clause_text`, `clause_type`, `start_pos`, `end_pos`, `extraction_confidence` (formerly `confidence`), `confidence_logit`, `page_no`, and `content_label`.
*   **Memory Optimization:** Instantly garbage collects and calls `torch.mps.empty_cache()` or `torch.cuda.empty_cache()` immediately after completion, freeing ~1.1GB RAM/VRAM.

### 🛡️ Stage 3: Unified Agentic Risk Assessment (LangGraph ReAct Loop)
*   **Purpose:** Assess risk levels ("LOW", "MEDIUM", "HIGH") and synthesize explanations for every extracted clause.
*   **Multi-Tenant Safety:** Assigns a unique `uuid4()` document identifier to every processed contract. All precedent chunks stored in FAISS are dynamically tagged and scoped using this UUID to prevent cross-contract data collision in multi-user environments.
*   **Pipeline Design:** **No split paths.** Every clause is evaluated by a LangGraph ReAct agent.
*   **DeBERTa Ensemble prior (Ens-F):** A fine-tuned `DeBERTa-v3-base` ensemble (`ce-parties` + `corn-parties`) provides an initial classification. This prior label is passed into the Agent prompt.
*   **Agent Tools:**
    1.  `precedent_search`: Queries a FAISS index of historical clauses.
    2.  `contract_search`: Queries sibling clauses in the active contract.
*   **LegalBERT Retrieval Upgrade:** Upgraded sentence representations from general MiniLM to domain-specific `nlpaueb/legal-bert-base-uncased` (768 dimensions), maximizing cosine similarity precision for specialized legal vocabulary.
*   **Semantic Deduplication Engine:** Integrated a runtime filter inside `query_index` that expands the search pool (`k * 2`) and filters out duplicate standard boilerplate clauses using a whitespace-normalized case-insensitive set, returning exactly `k` unique, high-quality precedents.
*   **Dynamic Confidence Normalization:** Standardizes LLM-generated confidences by dividing parsed values > 1.0 by 100.0, bulletproofing the pipeline against percentage-to-fraction formatting mismatches (e.g., preventing 95% from rendering as 9500%).
*   **VRAM Flush:** Dynamically unloads the risk classification model and flushes MPS/CUDA caches immediately after Stage 3 assessments complete.

### 🔌 True Step-by-Step Caching & Checkpointer
*   **Architecture:** The pipeline is fully checkpointed and cached at every stage (`stage1_output.json`, `stage2_output.json`, `stage3_output.json`, `final_report.json`).
*   **Zero-Copy Resuming:** Re-auditing or resuming suspended audits reloads intermediate parsed structures, completely bypassing Stage 1 and Stage 2 execution in **0.00 seconds** while starting Stage 3 immediately with the upgraded LegalBERT index!

### 🎨 Stage 4: Progressive Report Generation
*   **Purpose:** Aggregate risk assessments into a client-ready premium interactive report.
*   **Jinja2 Templating:** Custom Jinja2 HTML layout file loaded with elegant pastel typography, distinct border-left risk tags, and detailed markdown tables.
*   **Dynamic Decoupled Overrides:** Correctly pulls `agent_confidence` for overridden clauses and `risk_confidence` for default classifications, presenting a unified confidence metric without rendering visual override badges.
*   **Executive Summary:** A Qwen-3-30B OpenRouter API call generates a high-quality summary, while missing protections are mapped deterministically via Python dictionaries.

---

## 3. Core System Data Contracts

The pipelines are decoupled and communicate strictly through typed Python dataclasses declared in [schema.py](file:///Users/ts/Desktop/C/AI_ML_Project/ai_ml_project_pipeline/src/common/schema.py).

### 📥 Stage 1+2 Output (Stage 3 Input)
```json
{
  "clause_id": "contract_001_Indemnification_0030",
  "document_id": "contract_001",
  "clause_text": "Contractor shall indemnify Company against all claims...",
  "clause_type": "Indemnification",
  "start_pos": 4521,
  "end_pos": 4687,
  "confidence": 0.94,
  "extractor_confidence": 0.94,
  "extraction_confidence": 0.94
}
```

### 🧠 Stage 3 Output (Stage 4 Input)
```json
{
  "clause_id": "contract_001_Indemnification_0030",
  "document_id": "contract_001",
  "clause_text": "Contractor shall indemnify Company against all claims...",
  "clause_type": "Indemnification",
  "risk_level": "HIGH",
  "risk_explanation": "One-sided indemnification covering counterparty negligence",
  "similar_clauses": [
    {
      "text": "Vendor shall indemnify Client against all negligent acts...",
      "clause_type": "Indemnification",
      "risk_level": "HIGH",
      "similarity": 0.89
    }
  ],
  "cross_references": ["contract_001_Liability_0042"],
  "extraction_confidence": 0.94,
  "classifier_confidence": 0.88,
  "agent_confidence": 0.95,
  "is_override": true,
  "recommendation": "Renegotiate to mutual indemnification...",
  "extraction_confidence_logit": 4.52,
  "content_label": "text",
  "page_no": "12",
  "start_pos": 4521,
  "end_pos": 4687,
  "agent_trace": [
    {"tool": "precedent_search", "result_count": 5},
    {"tool": "contract_search", "result_count": 12}
  ]
}
```

### 🏆 Stage 4 Final Structured Output
```json
{
  "document_id": "contract_001",
  "summary": "This vendor agreement contains 23 clauses...",
  "high_risk": [
    {
      "clause_id": "contract_001_Indemnification_0030",
      "clause_type": "Indemnification",
      "risk_level": "HIGH",
      "explanation": "The indemnification provision requires Contractor to...",
      "recommendation": "Renegotiate to mutual indemnification...",
      "similar_clauses": [
        {
          "text": "Vendor shall indemnify Client against all negligent acts...",
          "clause_type": "Indemnification",
          "risk_level": "HIGH",
          "similarity": 0.89
        }
      ],
      "cross_references": ["contract_001_Liability_0042"],
      "page_no": "12",
      "risk_confidence": 0.95,
      "extraction_confidence": 0.94,
      "is_override": true,
      "agent_confidence": 0.95,
      "extraction_confidence_logit": 4.52,
      "content_label": "text",
      "clause_text": "Contractor shall indemnify Company against all claims...",
      "agent_trace": [
        {"tool": "precedent_search", "result_count": 5}
      ]
    }
  ],
  "medium_risk": [],
  "low_risk": [],
  "low_risk_summary": "15 clauses were assessed as standard/low risk...",
  "missing_protections": ["Data Protection", "Force Majeure"],
  "overall_risk_score": 6.8,
  "total_clauses": 23,
  "metadata": {
    "generated_at": "2026-05-17T13:50:00Z",
    "models_used": {
      "extraction": "rajnishahuja/cuad-stage1-deberta",
      "risk_classification": "models/stage3_risk_deberta_v3",
      "embeddings": "nlpaueb/legal-bert-base-uncased",
      "explanation": "Qwen3-30B-Q4_K_XL"
    }
  }
}
```

---

## 4. Downstream Visual Layer & Streamlit Web App

The **Legal Contract Risk Analyzer** includes a backend API service and a state-of-the-art visual auditing workbench frontend.

### ⚙️ 1. Start the Backend API Server (FastAPI)
To launch the FastAPI backend server with auto-reload, execute the following command in your terminal:
```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### ⚙️ 2. Start the Frontend Audit Web App (Streamlit)
In a separate terminal, launch the Streamlit frontend client that communicates with the pipeline:
```bash
uv run streamlit run app/streamlit_app.py --server.port 8501
```

### ⚡ Fast-Running Development Tips
1. **Interactive Audit Workbench**: The UI dynamically scans `data/output/final/` to populate completed and suspended/incomplete audits.
2. **Resume Audit Tracker**: Clicking "Resume Audit" on an incomplete contract allows you to run pending stages incrementally. Thanks to zero-copy caching, Stages 1 & 2 run instantly in **0.00 seconds** by loading caches from disk.
```
