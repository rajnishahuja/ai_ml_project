"""
eval_stage3.py
==============
Evaluates the Stage 3 hybrid pipeline against ground-truth labels from the test split.

Ablation modes (mutually exclusive, all report macro F1 vs DeBERTa baseline):
  (default)            Full pipeline: DeBERTa → FAISS + LLM + contract_search
  --rag-only           DeBERTa + FAISS majority vote, no LLM at all
  --no-contract-search DeBERTa + FAISS + LLM, contract_search tool disabled

Reports:
  1. DeBERTa-only macro F1  (no LLM — fast baseline, always shown)
  2. Ablation mode macro F1
  3. Fast-path label-change rate
  4. Agent-path: label-change rate, accuracy, contract_search usage
  5. Per-class (LOW / MEDIUM / HIGH) precision, recall, F1 for both

Usage:
    python scripts/eval_stage3.py [--sample N | --full] [--output PATH] [--seed S]
                                  [--rag-only | --no-contract-search]

    --sample N             Evaluate N clauses stratified by label (default: 150)
    --full                 Evaluate entire test split (~452 rows) — much slower
    --rag-only             FAISS majority vote only, no LLM
    --no-contract-search   Agent path uses precedent_search only
    --output               Results JSON path (default: data/eval/eval_stage3_results.json)
    --seed                 Random seed for sampling (default: 42)
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import classification_report, f1_score

from src.common.schema import ClauseObject
from src.stage3_risk_agent.agent import assess_clauses
from src.stage3_risk_agent.embeddings import query_index
from src.stage3_risk_agent.risk_classifier import RiskClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CE_MODEL_PATH   = "models/stage3_risk_deberta_v3_run22_parties/final"
CORN_MODEL_PATH = "models/stage3_risk_deberta_v3_run23_corn_parties/final"
LABELS          = ["LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_test_rows(data_path: str, splits_path: str) -> list[dict]:
    with open(data_path) as f:
        data = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)

    row_by_num = {r["row_num"]: r for r in data}
    test_rows  = [row_by_num[i] for i in splits["test"]]
    valid      = [r for r in test_rows if r["label"] in LABELS]
    logger.info(
        "Test split: %d rows total, %d with valid labels (%d None skipped)",
        len(test_rows), len(valid), len(test_rows) - len(valid),
    )
    return valid


def stratified_sample(rows: list[dict], n_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    sampled = []
    for label in LABELS:
        pool = by_label[label]
        k    = min(n_per_class, len(pool))
        sampled.extend(rng.sample(pool, k))
        logger.info("  Sampled %d / %d rows for label=%s", k, len(pool), label)

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Build ClauseObjects for assess_clauses()
# ---------------------------------------------------------------------------

def rows_to_clause_objects(rows: list[dict]) -> list[ClauseObject]:
    """
    Convert training_dataset rows to ClauseObjects, one per row.
    Also injects one synthetic Parties clause per unique contract so that
    extract_signing_party() can resolve the signing party.
    """
    # Collect one signing_party text per contract (use first non-empty)
    contract_party: dict[str, str] = {}
    for r in rows:
        contract = r["contract"]
        if contract not in contract_party and r.get("signing_party"):
            contract_party[contract] = r["signing_party"]

    clauses: list[ClauseObject] = []
    seen_parties_docs: set[str] = set()

    for r in rows:
        contract = r["contract"]
        # Inject synthetic Parties clause once per contract
        if contract not in seen_parties_docs:
            party_text = contract_party.get(contract, "")
            if party_text:
                clauses.append(ClauseObject(
                    clause_id=f"{contract}__Parties__synthetic",
                    document_id=contract,
                    clause_type="Parties",
                    clause_text=party_text,
                    start_pos=0,
                    end_pos=len(party_text),
                    confidence=1.0,
                ))
            seen_parties_docs.add(contract)

        clauses.append(ClauseObject(
            clause_id=r["id"],
            document_id=contract,
            clause_type=r["clause_type"],
            clause_text=r["clause_text"],
            start_pos=0,
            end_pos=len(r["clause_text"]),
            confidence=0.95,
        ))

    return clauses


# ---------------------------------------------------------------------------
# DeBERTa-only baseline
# ---------------------------------------------------------------------------

def run_deberta_only(rows: list[dict], classifier: RiskClassifier) -> list[dict]:
    logger.info("Running DeBERTa-only pass on %d rows ...", len(rows))
    results = []
    for r in rows:
        pred = classifier.predict(
            clause_text=r["clause_text"],
            clause_type=r["clause_type"],
            signing_party=r.get("signing_party", ""),
        )
        results.append({
            "id":              r["id"],
            "contract":        r["contract"],
            "clause_type":     r["clause_type"],
            "ground_truth":    r["label"],
            "deberta_label":   pred["label"],
            "deberta_conf":    round(pred["confidence"], 4),
        })
    return results


# ---------------------------------------------------------------------------
# RAG-only ablation (no LLM)
# ---------------------------------------------------------------------------

def run_rag_only(rows: list[dict], classifier: RiskClassifier, index_path: str, k: int = 5) -> list[dict]:
    """DeBERTa confidence-gate → FAISS majority vote, no LLM call."""
    from collections import Counter
    logger.info("Running RAG-only ablation on %d rows ...", len(rows))
    results = []
    for r in rows:
        pred    = classifier.predict(
            clause_text=r["clause_text"],
            clause_type=r["clause_type"],
            signing_party=r.get("signing_party", ""),
        )
        similar = query_index(r["clause_text"], index_path, k)
        if similar:
            votes         = Counter(s.risk_level for s in similar)
            majority      = votes.most_common(1)[0][0]
            majority_dist = dict(votes)
        else:
            majority      = pred["label"]
            majority_dist = {}

        results.append({
            "id":              r["id"],
            "clause_type":     r["clause_type"],
            "ground_truth":    r["label"],
            "deberta_label":   pred["label"],
            "deberta_conf":    round(pred["confidence"], 4),
            "pipeline_label":  majority,
            "path":            "rag",
            "override":        pred["label"] != majority,
            "rag_votes":       majority_dist,
        })
    return results


# ---------------------------------------------------------------------------
# Full pipeline (with optional ablation flags + checkpointing)
# ---------------------------------------------------------------------------

def run_pipeline(
    rows: list[dict],
    use_contract_search: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> list[dict]:
    mode = "full pipeline" if use_contract_search else "pipeline (no contract_search)"

    # Load checkpoint of already-processed clause IDs
    done_ids: set[str] = set()
    existing_results: list[dict] = []
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                ckpt_row = json.loads(line)
                done_ids.add(ckpt_row["clause_id"])
                existing_results.append(ckpt_row)
        logger.info("Resuming: %d clauses already processed", len(done_ids))

    logger.info("Running %s on %d rows (%d pending) ...", mode, len(rows), len(rows) - len(done_ids))

    if len(done_ids) == len(rows):
        logger.info("Nothing to process — all rows already checkpointed.")
        return _reformat_checkpoint_rows(existing_results, rows)

    clauses      = rows_to_clause_objects(rows)
    ground_truth = {r["id"]: r["label"] for r in rows}

    assessed = assess_clauses(
        clauses=clauses,
        config_path="configs/stage3_config.yaml",
        ce_model_path=CE_MODEL_PATH,
        corn_model_path=CORN_MODEL_PATH,
        use_contract_search=use_contract_search,
        skip_ids=done_ids,
        checkpoint_file=checkpoint_path,
    )

    new_results = []
    for a in assessed:
        gt = ground_truth.get(a.clause_id)
        if gt is None:
            continue
        tools_used = [t.tool for t in a.agent_trace] if a.agent_trace else []
        new_results.append({
            "id":                   a.clause_id,
            "clause_type":          a.clause_type,
            "ground_truth":         gt,
            "deberta_label":        None,
            "deberta_conf":         round(a.confidence, 4),
            "pipeline_label":       a.risk_level,
            "path":                 "agent" if a.agent_trace else "fast",
            "override":             False,
            "tools_used":           tools_used,
            "contract_search_used": "contract_search" in tools_used,
            "explanation":          a.risk_explanation,
            "override_reason":      getattr(a, "override_reason", ""),
        })

    # On resume, reconstruct previously-checkpointed rows into the same format
    resumed = _reformat_checkpoint_rows(existing_results, rows)
    return resumed + new_results


def _reformat_checkpoint_rows(ckpt_rows: list[dict], all_rows: list[dict]) -> list[dict]:
    """Convert raw checkpoint entries back into the pipeline row format."""
    gt_map = {r["id"]: r["label"] for r in all_rows}
    results = []
    for c in ckpt_rows:
        tools_used = [t["tool"] for t in c.get("agent_trace", [])]
        results.append({
            "id":                   c["clause_id"],
            "clause_type":          c["clause_type"],
            "ground_truth":         gt_map.get(c["clause_id"], ""),
            "deberta_label":        None,
            "deberta_conf":         c.get("confidence", 0.0),
            "pipeline_label":       c["risk_level"],
            "path":                 "agent" if c.get("agent_trace") else "fast",
            "override":             False,
            "tools_used":           tools_used,
            "contract_search_used": "contract_search" in tools_used,
            "explanation":          c.get("explanation", ""),
            "override_reason":      "",
        })
    return results


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def print_report(title: str, y_true: list[str], y_pred: list[str]) -> float:
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    return macro_f1


def path_stats(pipeline_rows: list[dict]) -> dict:
    fast_rows  = [r for r in pipeline_rows if r["path"] == "fast"]
    agent_rows = [r for r in pipeline_rows if r["path"] == "agent"]
    rag_rows   = [r for r in pipeline_rows if r["path"] == "rag"]

    def change_rate(rows):
        changed = sum(1 for r in rows if r.get("deberta_label") and r["deberta_label"] != r["pipeline_label"])
        return changed / len(rows) if rows else 0.0

    def accuracy(rows):
        correct = sum(1 for r in rows if r["pipeline_label"] == r["ground_truth"])
        return correct / len(rows) if rows else 0.0

    stats = {}
    if fast_rows:
        stats["fast"] = {"n": len(fast_rows), "label_change_rate": change_rate(fast_rows), "accuracy": accuracy(fast_rows)}
    if agent_rows:
        cs_used = sum(1 for r in agent_rows if r.get("contract_search_used"))
        stats["agent"] = {
            "n": len(agent_rows),
            "label_change_rate": change_rate(agent_rows),
            "accuracy": accuracy(agent_rows),
            "contract_search_used": cs_used,
            "contract_search_rate": round(cs_used / len(agent_rows), 3) if agent_rows else 0,
        }
    if rag_rows:
        stats["rag"] = {"n": len(rag_rows), "label_change_rate": change_rate(rag_rows), "accuracy": accuracy(rag_rows)}
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 pipeline vs DeBERTa baseline.")
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--sample", type=int, default=150,
                            help="Clauses to evaluate, stratified by label (default: 150)")
    size_group.add_argument("--full", action="store_true",
                            help="Evaluate entire test split")
    ablation_group = parser.add_mutually_exclusive_group()
    ablation_group.add_argument("--rag-only", action="store_true",
                                help="FAISS majority vote only — no LLM")
    ablation_group.add_argument("--no-contract-search", action="store_true",
                                help="Agent path uses precedent_search only")
    parser.add_argument("--output",  default="data/eval/eval_stage3_results.json")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--resume",  action="store_true",
                        help="Resume an interrupted run using the checkpoint file")
    args = parser.parse_args()

    # Load test rows
    test_rows = load_test_rows(
        "data/processed/training_dataset.json",
        "data/processed/splits.json",
    )

    if args.full:
        eval_rows = test_rows
        logger.info("Full eval mode: %d rows", len(eval_rows))
    else:
        n_per_class = args.sample // len(LABELS)
        logger.info("Sampling %d per class (%d total) ...", n_per_class, n_per_class * len(LABELS))
        eval_rows = stratified_sample(test_rows, n_per_class, args.seed)

    # --- DeBERTa-only baseline (always run — no LLM needed) ---
    logger.info("Loading DeBERTa classifier ...")
    classifier = RiskClassifier(
        ce_model_path=CE_MODEL_PATH,
        corn_model_path=CORN_MODEL_PATH,
    )
    deberta_rows = run_deberta_only(eval_rows, classifier)
    deberta_map  = {r["id"]: r for r in deberta_rows}

    deberta_macro = print_report(
        "DeBERTa-only (Ens-F baseline)",
        [r["ground_truth"]  for r in deberta_rows],
        [r["deberta_label"] for r in deberta_rows],
    )

    # --- Ablation / pipeline run ---
    checkpoint_path = args.output.replace(".json", ".checkpoint.jsonl")

    if args.rag_only:
        from src.common.utils import load_config
        cfg        = load_config("configs/stage3_config.yaml")
        index_path = cfg["faiss_index_path"]
        pipeline_rows = run_rag_only(eval_rows, classifier, index_path)
        mode_label = "RAG-only (FAISS majority vote)"
    else:
        use_cs = not args.no_contract_search
        pipeline_rows = run_pipeline(
            eval_rows,
            use_contract_search=use_cs,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
        )
        # Merge DeBERTa label + override flag into pipeline rows
        for r in pipeline_rows:
            db = deberta_map.get(r["id"])
            if db:
                r["deberta_label"] = db["deberta_label"]
                r["override"] = (r["deberta_label"] != r["pipeline_label"])
        mode_label = (
            "Pipeline: DeBERTa + FAISS + LLM (no contract_search)"
            if args.no_contract_search
            else "Full pipeline: DeBERTa + FAISS + LLM + contract_search"
        )

    pipeline_macro = print_report(
        mode_label,
        [r["ground_truth"]   for r in pipeline_rows],
        [r["pipeline_label"] for r in pipeline_rows],
    )

    # Path breakdown
    stats = path_stats(pipeline_rows)
    print("\n--- Path breakdown ---")
    for path, s in stats.items():
        cs_info = ""
        if path == "agent" and "contract_search_used" in s:
            cs_info = f"  contract_search={s['contract_search_used']}/{s['n']} ({s['contract_search_rate']:.0%})"
        print(
            f"  {path:5s}: n={s['n']:3d}  "
            f"label_change_rate={s['label_change_rate']:.1%}  "
            f"accuracy={s['accuracy']:.1%}{cs_info}"
        )

    delta = pipeline_macro - deberta_macro
    print(f"\n{mode_label} vs DeBERTa macro F1 delta: {delta:+.4f}")

    # --- Save results ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "summary": {
            "n_eval":          len(pipeline_rows),
            "deberta_macro_f1": round(deberta_macro, 4),
            "pipeline_macro_f1": round(pipeline_macro, 4),
            "mode":             mode_label,
            "delta":            round(delta, 4),
            "path_stats":       stats,
        },
        "rows": pipeline_rows,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
