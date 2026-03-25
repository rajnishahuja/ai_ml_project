"""
Stage 1+2: Combined Evaluation & Error Analysis
=================================================
Runs both the DeBERTa model and the rule-based baseline on the CUAD test
set and produces a side-by-side comparison report.

Two-dimensional evaluation (as specified in project plan):
  Dimension 1 — Extraction Quality:  Exact Match (EM), Token F1, Span IoU
  Dimension 2 — Classification:      Accuracy, Macro F1, Confusion matrix

Also runs error analysis: which clause types are hardest to extract/classify.

Usage:
  python stage1_2_evaluation.py \\
      --model_path ./models/stage1_2_deberta \\
      --output_dir ./results/stage1_2
"""

import json
import logging
import os
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    import string
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def squad_em_f1(prediction: str, ground_truths: list[str]) -> tuple[float, float]:
    if not ground_truths:
        return (1.0, 1.0) if not prediction else (0.0, 0.0)
    pred = normalize_answer(prediction)
    best_em, best_f1 = 0.0, 0.0
    for truth in ground_truths:
        truth_norm = normalize_answer(truth)
        em = float(pred == truth_norm)
        best_em = max(best_em, em)
        p_toks, t_toks = pred.split(), truth_norm.split()
        common = set(p_toks) & set(t_toks)
        if common:
            prec = len(common) / len(p_toks)
            rec = len(common) / len(t_toks)
            f1 = 2 * prec * rec / (prec + rec)
            best_f1 = max(best_f1, f1)
    return best_em, best_f1


def span_iou(pred_text: str, context: str, true_start: int, true_end: int) -> float:
    """
    Span overlap IoU: intersection / union of character spans.
    Requires locating predicted text in context.
    """
    pred_start = context.find(pred_text)
    if pred_start == -1 or not pred_text:
        return 0.0
    pred_end = pred_start + len(pred_text)

    intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(pred_end, true_end) - min(pred_start, true_start)
    return intersection / union if union > 0 else 0.0


def infer_clause_type(question: str, clause_types: list[str]) -> str:
    for ct in clause_types:
        if ct.lower() in question.lower():
            return ct
    return "Unknown"


# ---------------------------------------------------------------------------
# DeBERTa evaluation
# ---------------------------------------------------------------------------

def evaluate_deberta(
    model_path: str,
    test_examples: list[dict],
    clause_types: list[str],
) -> dict:
    """Run DeBERTa QA pipeline on test examples and compute metrics."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        handle_impossible_answer=True,
    )

    em_scores, f1_scores, iou_scores = [], [], []
    pred_types, true_types = [], []
    per_type_errors: dict[str, list] = {ct: [] for ct in clause_types}

    logger.info(f"Evaluating DeBERTa on {len(test_examples)} examples…")
    for example in test_examples:
        result = qa({"question": example["question"], "context": example["context"]})
        pred_text = result.get("answer", "").strip()
        true_answers = example.get("answers", {}).get("text", [])
        true_starts = example.get("answers", {}).get("answer_start", [])

        # Dimension 1
        em, f1 = squad_em_f1(pred_text, true_answers)
        em_scores.append(em)
        f1_scores.append(f1)

        iou = 0.0
        if true_starts:
            true_start = true_starts[0]
            true_end = true_start + len(true_answers[0]) if true_answers else true_start
            iou = span_iou(pred_text, example["context"], true_start, true_end)
        iou_scores.append(iou)

        # Dimension 2
        true_type = infer_clause_type(example["question"], clause_types)
        pred_type = true_type if pred_text else "NO_CLAUSE"
        pred_types.append(pred_type)
        true_types.append(true_type)

        # Track per-type errors for error analysis
        if true_type in per_type_errors and f1 < 0.5:
            per_type_errors[true_type].append({
                "id": example.get("id", ""),
                "true": true_answers[0] if true_answers else "",
                "pred": pred_text,
                "f1": round(f1, 3),
            })

    return _compile_results(
        "deberta_base",
        em_scores, f1_scores, iou_scores,
        true_types, pred_types,
        per_type_errors,
        clause_types,
    )


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline_model(
    test_examples: list[dict],
    clause_types: list[str],
    spacy_model: str = "en_core_web_sm",
) -> dict:
    """Run rule-based baseline on test examples and compute same metrics."""
    import sys; sys.path.insert(0, ".")
    from stage1_2_baseline import RuleBasedExtractor, _squad_em_f1, span_iou as _iou

    extractor = RuleBasedExtractor(spacy_model=spacy_model)

    em_scores, f1_scores, iou_scores = [], [], []
    pred_types, true_types = [], []
    per_type_errors: dict[str, list] = {ct: [] for ct in clause_types}

    logger.info(f"Evaluating baseline on {len(test_examples)} examples…")
    for example in test_examples:
        context = example["context"]
        true_type = infer_clause_type(example["question"], clause_types)
        true_answers = example.get("answers", {}).get("text", [])
        true_starts = example.get("answers", {}).get("answer_start", [])

        clauses = extractor.extract(context, doc_id=example.get("id", "eval"))
        matching = [c for c in clauses if c.clause_type == true_type]
        pred_text = matching[0].clause_text if matching else ""
        pred_type = true_type if pred_text else "NO_CLAUSE"

        em, f1 = _squad_em_f1(pred_text, true_answers)
        em_scores.append(em)
        f1_scores.append(f1)

        iou = 0.0
        if true_starts and true_answers:
            iou = span_iou(pred_text, context, true_starts[0],
                           true_starts[0] + len(true_answers[0]))
        iou_scores.append(iou)

        pred_types.append(pred_type)
        true_types.append(true_type)

        if true_type in per_type_errors and f1 < 0.5:
            per_type_errors[true_type].append({
                "id": example.get("id", ""),
                "true": true_answers[0] if true_answers else "",
                "pred": pred_text[:200],
                "f1": round(f1, 3),
            })

    return _compile_results(
        "rule_based_baseline",
        em_scores, f1_scores, iou_scores,
        true_types, pred_types,
        per_type_errors,
        clause_types,
    )


# ---------------------------------------------------------------------------
# Shared result compiler
# ---------------------------------------------------------------------------

def _compile_results(
    model_name: str,
    em_scores, f1_scores, iou_scores,
    true_types, pred_types,
    per_type_errors,
    clause_types,
) -> dict:
    class_report = classification_report(
        true_types, pred_types,
        labels=clause_types + ["NO_CLAUSE"],
        zero_division=0,
        output_dict=True,
    )

    # Hardest clause types: lowest per-class F1 (only actual CUAD types)
    per_class_f1 = {
        ct: round(class_report.get(ct, {}).get("f1-score", 0.0), 3)
        for ct in clause_types
    }
    hardest = sorted(per_class_f1.items(), key=lambda x: x[1])[:10]

    return {
        "model": model_name,
        "n_examples": len(em_scores),
        "extraction": {
            "exact_match_pct": round(np.mean(em_scores) * 100, 2),
            "token_f1_pct": round(np.mean(f1_scores) * 100, 2),
            "span_iou": round(float(np.mean(iou_scores)), 4),
        },
        "classification": {
            "accuracy": round(accuracy_score(true_types, pred_types), 4),
            "macro_f1": round(f1_score(true_types, pred_types, average="macro",
                                       zero_division=0), 4),
            "per_class_f1": per_class_f1,
        },
        "error_analysis": {
            "hardest_clause_types": hardest,
            "sample_errors_per_type": {
                ct: errors[:3]
                for ct, errors in per_type_errors.items() if errors
            },
        },
    }


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(
    deberta_results: dict,
    baseline_results: dict,
    output_path: str,
) -> None:
    """Write a side-by-side comparison of DeBERTa vs baseline."""
    d = deberta_results["extraction"]
    b = baseline_results["extraction"]
    dc = deberta_results["classification"]
    bc = baseline_results["classification"]

    report_lines = [
        "=" * 70,
        "STAGE 1+2 EVALUATION REPORT — DeBERTa vs Rule-Based Baseline",
        "=" * 70,
        "",
        "DIMENSION 1: EXTRACTION QUALITY",
        "-" * 40,
        f"{'Metric':<30} {'DeBERTa':>12} {'Baseline':>12} {'Delta':>10}",
        f"{'Exact Match (%)':<30} {d['exact_match_pct']:>12.2f} {b['exact_match_pct']:>12.2f} "
        f"{d['exact_match_pct'] - b['exact_match_pct']:>+10.2f}",
        f"{'Token F1 (%)':<30} {d['token_f1_pct']:>12.2f} {b['token_f1_pct']:>12.2f} "
        f"{d['token_f1_pct'] - b['token_f1_pct']:>+10.2f}",
        f"{'Span IoU':<30} {d['span_iou']:>12.4f} {b['span_iou']:>12.4f} "
        f"{d['span_iou'] - b['span_iou']:>+10.4f}",
        "",
        "DIMENSION 2: CLASSIFICATION QUALITY",
        "-" * 40,
        f"{'Metric':<30} {'DeBERTa':>12} {'Baseline':>12} {'Delta':>10}",
        f"{'Accuracy':<30} {dc['accuracy']:>12.4f} {bc['accuracy']:>12.4f} "
        f"{dc['accuracy'] - bc['accuracy']:>+10.4f}",
        f"{'Macro F1':<30} {dc['macro_f1']:>12.4f} {bc['macro_f1']:>12.4f} "
        f"{dc['macro_f1'] - bc['macro_f1']:>+10.4f}",
        "",
        "ERROR ANALYSIS — Hardest Clause Types (DeBERTa, by F1)",
        "-" * 40,
    ]
    for clause_type, f1 in deberta_results["error_analysis"]["hardest_clause_types"]:
        report_lines.append(f"  {clause_type:<45} F1: {f1:.3f}")

    report_lines += [
        "",
        "NOTE: These types warrant closer inspection for training data quality,",
        "annotation ambiguity (e.g., Termination vs. Cancellation), or model bias.",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_file = output_path.replace(".json", "_summary.txt")
    with open(report_file, "w") as f:
        f.write(report_text)

    combined = {
        "deberta": deberta_results,
        "baseline": baseline_results,
    }
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"Full results saved to {output_path}")
    logger.info(f"Summary saved to {report_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned DeBERTa")
    parser.add_argument("--output_dir", default="./results/stage1_2")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Limit test examples (for quick dev runs)")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    # Load CUAD test split
    logger.info("Loading CUAD test split…")
    dataset = load_dataset("theatticusproject/cuad", trust_remote_code=True)
    test_examples = list(dataset["test"])
    if args.n_examples:
        test_examples = test_examples[: args.n_examples]
        logger.info(f"Using {args.n_examples} examples for quick evaluation")

    import sys; sys.path.insert(0, ".")
    from stage1_2_pipeline import CUAD_CLAUSE_TYPES

    # DeBERTa evaluation
    deberta_results = evaluate_deberta(args.model_path, test_examples, CUAD_CLAUSE_TYPES)

    # Baseline evaluation
    if not args.skip_baseline:
        baseline_results = evaluate_baseline_model(test_examples, CUAD_CLAUSE_TYPES, args.spacy_model)
    else:
        baseline_results = {"model": "skipped", "extraction": {}, "classification": {}, "error_analysis": {}}

    # Combined report
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    generate_comparison_report(deberta_results, baseline_results, output_path)
