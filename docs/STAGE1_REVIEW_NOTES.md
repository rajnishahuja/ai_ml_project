# Stage 1+2 Review Notes — Suggestions for Alignment

> **Context**: Anushka's PR #2 (merged 2026-04-14) implemented Stage 1+2 in `src/stage1_extract_classify/`. 
> These are suggestions to align her code with the shared architecture defined in Phase 0.
> None of these block current work — they're improvements to discuss.

---

## 1. Use shared `ClauseObject` from `schema.py`

**Current**: `baseline.py` defines `BaselineClause` (8 fields), `pipeline.py` has its own `ClauseObject` copy.  
**Suggested**: Import `ClauseObject` from `src/common/schema.py` (T0.9). Stage 3 expects this as input.  
**Files**: `baseline.py:254`, `pipeline.py:45`

## 2. Use shared metrics from `utils.py`

**Current**: `evaluate.py` re-implements `normalize_answer`, `squad_em_f1`, `span_iou` (lines 101–169).  
**Suggested**: Import from `src/common/utils.py` where these already exist.  
**Files**: `evaluate.py:101-169`

## 3. Single source for `CUAD_CLAUSE_TYPES`

**Current**: Defined in `constants.py` (good), but also duplicated in `predict.py:16-31`.  
**Suggested**: `predict.py` should import from `constants.py` when implemented.  
**Files**: `constants.py`, `predict.py:16-31`

## 4. Dataset approach — open discussion

**Current**: `preprocess_cuad.py` uses local `CUAD_v1.json` with custom flattening and 1:1 pos/neg balancing.  
**Alternative**: `src/common/data_loader.py` uses `theatticusproject/cuad-qa` (pre-flattened, pre-split from HuggingFace).  
**Questions**:
- How much did the 1:1 balancing improve accuracy vs unbalanced?
- Can the balancing logic be applied to `cuad-qa` as well if needed?
- Should we standardize on one dataset source?

## 5. Pipeline refactoring (T1.1–T1.3)

**Current**: `pipeline.py` remains monolithic (463 lines). `model.py`, `train.py`, `predict.py` are stubs.  
**Planned**: T1.1 (model.py), T1.2 (train.py), T1.3 (predict.py) — extract from pipeline.py.  
**Status**: Not started. Need to coordinate who picks these up.

---

*Created 2026-04-14 during architecture review.*
