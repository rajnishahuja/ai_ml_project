# Copilot Instructions — Legal Contract Risk Analyzer

## Project Context
This is an academic ML project building a modular pipeline that analyzes legal contracts and flags risky clauses. It has 3 processing stages (Stage 1+2 combined, Stage 3, Stage 4). See `ARCHITECTURE.md` in the project root for the full architecture, data flow, and directory structure.

## Key Rules for Code Generation

### General
- Python 3.10+, type hints on all functions
- Use `logging` module, never `print()` for status output
- All model paths and hyperparams come from YAML configs in `configs/`
- Docstrings on all public functions (Google style)
- Imports at top of file, grouped: stdlib → third-party → local

### Stage 1+2 (src/stage1_extract_classify/)
- Uses DeBERTa-base for QA-based clause extraction + classification
- Dataset: CUAD from HuggingFace (`theatticusproject/cuad`)
- Native SQuAD-style QA format: question + context → answer span
- One model produces both extraction and classification; evaluate on two dimensions separately

### Stage 3 (src/stage3_risk_agent/)
- This is a LangGraph agent with a reasoning loop and two tools
- Tool 1: FAISS RAG retrieval (embeddings via all-MiniLM-L6-v2)
- Tool 2: Contract search (find related clauses in same contract)
- Risk classifier: DeBERTa-base fine-tuned on synthetic risk labels
- Explanation generator: Mistral-7B-Instruct (4-bit quantized)
- When generating LangGraph code, use StateGraph with typed state dict

### Stage 4 (src/stage4_report_gen/)
- Hybrid: deterministic Python (aggregation) + LLM (explanations) + lookup table (recommendations)
- Aggregation is pure Python, no model needed
- Explanation uses FLAN-T5-base with prompt templates
- Recommendations use a curated dict mapping (clause_type, risk_pattern) → remediation text

### Data
- Raw CUAD data goes in `data/raw/`
- Processed/formatted data in `data/processed/`
- Synthetic risk labels in `data/synthetic/`
- FAISS index files in `data/faiss_index/`
- Never commit large model weights or data files to git

### Testing
- Tests in `tests/` mirror `src/` structure
- Use pytest
- Mock external API calls (LLM APIs) in tests
