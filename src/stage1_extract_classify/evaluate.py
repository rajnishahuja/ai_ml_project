"""
Stage 1+2: Combined Evaluation & Error Analysis
=================================================
Runs both the DeBERTa model and the rule-based baseline on the CUAD test
set and produces a side-by-side comparison report.

Two-dimensional evaluation:
  Dimension 1 — Extraction Quality:  Exact Match (EM), Token F1, Span IoU
  Dimension 2 — Classification:      Accuracy, Macro F1, Confusion matrix

Also runs error analysis: which clause types are hardest to extract/classify.

Usage:
  python evaluate.py \
      --model_path ./models/stage1_2_deberta \
      --output_dir ./results/stage1_2
"""

import json
import logging
import os
import argparse
import string
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from constants import CUAD_CLAUSE_TYPES, QUESTION_TO_CLAUSE_TYPE, BASELINE_CONF_THRESHOLD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Load raw CUAD JSON
# ---------------------------------------------------------------------------

def load_cuad_examples(cuad_json_path: str = None, test_only: bool = False) -> list[dict]:
    """
    Load raw CUAD JSON and return flat list of QA examples.
    Each example has: id, question, context, answers.
    NOT the tokenized dataset — which has no question/context/answers fields.

    test_only=True: recreates the exact same 10% test split used in preprocessing
    (seed=42, test_size=0.10) so evaluation is on held-out data only.
    This avoids data leakage from evaluating on training examples.
    """
    from datasets import Dataset

    path = cuad_json_path or os.environ.get("CUAD_JSON", "./CUAD_v1.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CUAD JSON not found at: {path}\n"
            f"Set CUAD_JSON env var or pass --cuad_json argument."
        )

    logger.info(f"Loading CUAD examples from: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for doc in raw["data"]:
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "id": qa["id"],
                    "question": qa["question"].strip(),
                    "context": context,
                    "answers": {
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    } if qa["answers"] else {"text": [], "answer_start": []},
                })

    logger.info(f"Loaded {len(examples)} total QA examples")

    if test_only:
        # Recreate exact same split as preprocess_cuad.py:
        #   flat.train_test_split(test_size=0.10, seed=42)
        # Deterministic — same seed = same IDs every time
        flat = Dataset.from_dict({"id": [ex["id"] for ex in examples]})
        split = flat.train_test_split(test_size=0.10, seed=42)
        test_ids = set(split["test"]["id"])
        examples = [ex for ex in examples if ex["id"] in test_ids]
        logger.info(
            f"Filtered to {len(examples)} test examples "
            f"(10% split, seed=42 — matches preprocessing)"
        )

    return examples


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def squad_em_f1(prediction: str, ground_truths: list[str]) -> tuple[float, float]:
    """
    SQuAD-style EM and token F1.
    - Empty ground truth + empty prediction = (1.0, 1.0) correct NO_CLAUSE
    - Empty ground truth + non-empty prediction = (0.0, 0.0) false positive
    """
    if not ground_truths or not ground_truths[0].strip():
        return (1.0, 1.0) if not prediction.strip() else (0.0, 0.0)

    pred = normalize_answer(prediction)
    best_em, best_f1 = 0.0, 0.0

    for truth in ground_truths:
        truth_norm = normalize_answer(truth)
        em = float(pred == truth_norm)
        best_em = max(best_em, em)

        p_toks, t_toks = pred.split(), truth_norm.split()
        if not p_toks or not t_toks:
            continue
        common = sum((Counter(p_toks) & Counter(t_toks)).values())
        if common == 0:
            continue
        prec = common / len(p_toks)
        rec = common / len(t_toks)
        f1 = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)

    return best_em, best_f1


def span_iou(pred_text: str, context: str, true_start: int, true_end: int) -> float:
    if not pred_text:
        return 0.0

    # Search normalized pred in original context
    # This avoids position mismatch from whitespace normalization
    pred_norm = " ".join(pred_text.split())
    
    starts = []
    start = context.find(pred_norm)
    while start != -1:
        starts.append(start)
        start = context.find(pred_norm, start + 1)

    # Fallback — try original pred text
    if not starts:
        start = context.find(pred_text.strip())
        if start != -1:
            starts.append(start)

    if not starts:
        return 0.0

    best_iou = 0.0
    pred_len = len(pred_norm)
    for pred_start in starts:
        pred_end = pred_start + pred_len
        intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
        union = max(pred_end, true_end) - min(pred_start, true_start)
        if union > 0:
            best_iou = max(best_iou, intersection / union)

    return best_iou


def _infer_clause_type_from_question(question: str) -> str:
    if question in QUESTION_TO_CLAUSE_TYPE:
        return QUESTION_TO_CLAUSE_TYPE[question]
    question_lower = question.lower()
    for ct in CUAD_CLAUSE_TYPES:
        if ct.lower() in question_lower:
            return ct
    return "Unknown"


# ---------------------------------------------------------------------------
# DeBERTa evaluation
# ---------------------------------------------------------------------------
def evaluate_deberta(
    model_path: str,
    test_examples: list[dict],
    clause_types: list[str],
    output_path: str = "./results/deberta_eval.json",
    score_threshold: float = 0.15,   # 🔥 NEW
) -> dict:

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    qa = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        handle_impossible_answer=False,   # 🔥 FIX 1: disable strict no-answer
    )

    em_scores, f1_scores, iou_scores = [], [], []
    pred_types, true_types = [], []
    per_type_errors: dict[str, list] = {ct: [] for ct in clause_types}

    logger.info(f"Evaluating DeBERTa on {len(test_examples)} examples…")

    inputs = [
        {"question": ex["question"], "context": ex["context"]}
        for ex in test_examples
    ]

    results = []
    chunk_size = 200

    for i in range(0, len(inputs), chunk_size):
        chunk = inputs[i:i + chunk_size]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            chunk_results = qa(
                chunk,
                batch_size=32,
                max_seq_len=384,
                doc_stride=128,
                max_question_len=64,
            )
        results.extend(chunk_results)
        logger.info(f"Progress: {min(i + chunk_size, len(inputs))}/{len(inputs)}")

    # 🔥 DEBUG (optional: first few examples)
    # for i in range(3):
    #     print(results[i])

    for example, result in zip(test_examples, results):

        raw_answer = result.get("answer", "").strip()
        score = result.get("score", 0.0)

        # 🔥 FIX 2: Apply threshold
        if score < score_threshold:
            pred_text = ""
        else:
            pred_text = raw_answer

        true_answers = example.get("answers", {}).get("text", [])
        true_starts = example.get("answers", {}).get("answer_start", [])

        pred_has_answer = bool(pred_text)

        # -----------------------
        # Extraction metrics
        # -----------------------
        em, f1 = squad_em_f1(pred_text, true_answers)
        em_scores.append(em)
        f1_scores.append(f1)

        iou = 0.0
        if true_starts and true_answers and pred_text:
            true_start = true_starts[0]
            true_end = true_start + len(true_answers[0])
            iou = span_iou(pred_text, example["context"], true_start, true_end)
        iou_scores.append(iou)

        # -----------------------
        # Classification
        # -----------------------
        true_type = _infer_clause_type_from_question(example["question"])
        true_has_clause = bool(true_answers and true_answers[0].strip())

        true_type_label = true_type if true_has_clause else "NO_CLAUSE"
        pred_type = true_type if pred_has_answer else "NO_CLAUSE"

        true_types.append(true_type_label)
        pred_types.append(pred_type)

        # -----------------------
        # Error tracking
        # -----------------------
        if true_type in per_type_errors and f1 < 0.5:
            per_type_errors[true_type].append({
                "id": example.get("id", ""),
                "true": true_answers[0] if true_answers else "",
                "pred": pred_text,
                "score": round(score, 3),   # 🔥 useful for debugging
                "f1": round(f1, 3),
            })

    result = _compile_results(
        "deberta_base",
        em_scores, f1_scores, iou_scores,
        true_types, pred_types,
        per_type_errors,
        clause_types,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"DeBERTa results:\n{json.dumps(result['extraction'], indent=2)}")
    logger.info(f"Saved to {output_path}")

    return result


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline_model(
    test_examples: list[dict],
    clause_types: list[str],
    spacy_model: str = "en_core_web_sm",
    output_path: str = "./results/baseline_eval.json",
) -> dict:

    from baseline import RuleBasedExtractor

    extractor = RuleBasedExtractor(spacy_model=spacy_model)

    em_scores, f1_scores, iou_scores = [], [], []
    pred_types, true_types = [], []
    per_type_errors: dict[str, list] = {ct: [] for ct in clause_types}

    logger.info(f"Evaluating baseline on {len(test_examples)} examples…")

    for idx, example in enumerate(test_examples):
        if idx % 500 == 0:
            logger.info(f"Baseline progress: {idx}/{len(test_examples)}")

        context = example["context"]
        question = example["question"]

        true_type = _infer_clause_type_from_question(question)
        true_answers = example.get("answers", {}).get("text", [])
        true_starts = example.get("answers", {}).get("answer_start", [])

        clauses = extractor.extract(context, doc_id=example.get("id", "eval"))

        # 🔥 FIX 1: DO NOT use true_type filtering
        best_clause = max(clauses, key=lambda c: c.confidence) if clauses else None

        # 🔥 FIX 2: Apply threshold (same idea as DeBERTa)
        if best_clause and best_clause.confidence >= BASELINE_CONF_THRESHOLD:
            pred_text = best_clause.clause_text
            pred_type = best_clause.clause_type
        else:
            pred_text = ""
            pred_type = "NO_CLAUSE"

        # -----------------------
        # Extraction metrics
        # -----------------------
        em, f1 = squad_em_f1(pred_text, true_answers)
        em_scores.append(em)
        f1_scores.append(f1)

        iou = 0.0
        if true_starts and true_answers and pred_text:
            iou = span_iou(
                pred_text,
                context,
                true_starts[0],
                true_starts[0] + len(true_answers[0]),
            )
        iou_scores.append(iou)

        # -----------------------
        # Classification
        # -----------------------
        true_has_clause = bool(true_answers and true_answers[0].strip())

        true_type_label = true_type if true_has_clause else "NO_CLAUSE"
        pred_type_label = pred_type if pred_text else "NO_CLAUSE"

        true_types.append(true_type_label)
        pred_types.append(pred_type_label)

        # -----------------------
        # Error tracking
        # -----------------------
        if true_type in per_type_errors and f1 < 0.5:
            per_type_errors[true_type].append({
                "id": example.get("id", ""),
                "true": true_answers[0] if true_answers else "",
                "pred": pred_text[:200],
                "confidence": round(best_clause.confidence, 3) if best_clause else 0.0,
                "f1": round(f1, 3),
            })

    result = _compile_results(
        "rule_based_baseline",
        em_scores, f1_scores, iou_scores,
        true_types, pred_types,
        per_type_errors,
        clause_types,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Baseline results:\n{json.dumps(result['extraction'], indent=2)}")
    logger.info(f"Saved to {output_path}")

    return result

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
        f"{'Text F1 (%)':<30} {d['token_f1_pct']:>12.2f} {b['token_f1_pct']:>12.2f} "
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

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
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
    parser.add_argument("--cuad_json", type=str, default=None,
                        help="Path to CUAD_v1.json (default: ./CUAD_v1.json or CUAD_JSON env var)")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Limit test examples (for quick dev runs)")
    parser.add_argument("--test_only", action="store_true",
                        help="Evaluate on test split only (seed=42, 10%% — matches preprocessing)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline evaluation, run DeBERTa only")
    parser.add_argument("--skip_deberta", action="store_true",
                        help="Skip DeBERTa evaluation, run baseline only")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    logger.info("Loading CUAD examples from raw JSON…")
    test_examples = load_cuad_examples(args.cuad_json, test_only=args.test_only)

    if args.n_examples:
        test_examples = test_examples[:args.n_examples]
        logger.info(f"Using {args.n_examples} examples for quick evaluation")

    # DeBERTa evaluation
    if not args.skip_deberta:
        deberta_results = evaluate_deberta(
            args.model_path,
            test_examples,
            CUAD_CLAUSE_TYPES,
            output_path=os.path.join(args.output_dir, "deberta_eval.json"),
        )
    else:
        logger.info("Skipping DeBERTa evaluation (--skip_deberta)")
        deberta_results = {
            "model": "skipped",
            "extraction": {"exact_match_pct": 0.0, "token_f1_pct": 0.0, "span_iou": 0.0},
            "classification": {"accuracy": 0.0, "macro_f1": 0.0, "per_class_f1": {}},
            "error_analysis": {"hardest_clause_types": [], "sample_errors_per_type": {}},
        }

    # Baseline evaluation
    if not args.skip_baseline:
        baseline_results = evaluate_baseline_model(
            test_examples,
            CUAD_CLAUSE_TYPES,
            args.spacy_model,
            output_path=os.path.join(args.output_dir, "baseline_eval.json"),
        )
    else:
        baseline_results = {
            "model": "skipped",
            "extraction": {"exact_match_pct": 0.0, "token_f1_pct": 0.0, "span_iou": 0.0},
            "classification": {"accuracy": 0.0, "macro_f1": 0.0, "per_class_f1": {}},
            "error_analysis": {"hardest_clause_types": [], "sample_errors_per_type": {}},
        }

    # Combined report
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    generate_comparison_report(deberta_results, baseline_results, output_path)
