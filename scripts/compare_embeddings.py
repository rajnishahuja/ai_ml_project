"""
Embedding Model Side-by-Side Benchmarker.

This script compares sentence-transformers/all-MiniLM-L6-v2 against
BAAI/bge-small-en-v1.5 (and other options) on your validation and test splits,
showing exact Precision@K, MRR, and latency benchmarks.
"""

import json
import logging
import os
import sys
import time
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.utils import setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger("compare_embeddings")

# Enforce reproducibility
random.seed(42)
np.random.seed(42)


def evaluate_embedding_model(
    model_name: str, train_rows: list, eval_rows: list, k_levels: list[int] = [1, 5, 10]
) -> dict:
    """Helper that embeds train & eval rows and calculates retrieval metrics via matrix multiplication."""
    logger.info("Initializing embedding model: %s", model_name)

    # Load model with forced local offline loading if cached
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        logger.info("  (Failed offline load, downloading from HuggingFace Hub ...)")
        model = SentenceTransformer(model_name)

    # 1. Encode training split (the database)
    logger.info("  Encoding %d train clauses ...", len(train_rows))
    t0 = time.perf_counter()
    train_texts = [r["clause_text"] for r in train_rows]
    train_embeds = model.encode(
        train_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    train_embeds = torch.tensor(train_embeds, dtype=torch.float32)
    train_time = time.perf_counter() - t0

    # 2. Encode evaluation split (the queries)
    logger.info("  Encoding %d evaluation clauses ...", len(eval_rows))
    t0 = time.perf_counter()
    eval_texts = [r["clause_text"] for r in eval_rows]
    eval_embeds = model.encode(
        eval_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    eval_embeds = torch.tensor(eval_embeds, dtype=torch.float32)
    eval_time = time.perf_counter() - t0

    # 3. Compute exact Cosine Similarity Matrix (eval_count x train_count)
    # Since embeddings are normalized, dot product = cosine similarity
    similarities = torch.mm(eval_embeds, train_embeds.t())  # (eval_count, train_count)

    # 4. Compute metrics
    metrics = {
        "mrr": 0.0,
        "label_agreement_sum": 0.0,
        "label_agreement_count": 0,
    }
    for k in k_levels:
        metrics[f"precision_at_{k}"] = 0.0
        metrics[f"consistency_at_{k}"] = 0.0

    eval_count = len(eval_rows)

    for i, query in enumerate(eval_rows):
        q_type = query["clause_type"]
        q_label = query["label"]

        # Get top matching indices in descending order
        scores = similarities[i].tolist()
        sorted_indices = sorted(
            range(len(scores)), key=lambda idx: scores[idx], reverse=True
        )

        # 1. MRR
        first_match_rank = 0
        for rank, idx in enumerate(sorted_indices, 1):
            if train_rows[idx]["clause_type"] == q_type:
                first_match_rank = rank
                break
        if first_match_rank > 0:
            metrics["mrr"] += 1.0 / first_match_rank

        # 2. Precision@K and Consistency@K
        for k in k_levels:
            k_indices = sorted_indices[:k]
            # Precision@K: Matches category?
            matches = sum(
                1 for idx in k_indices if train_rows[idx]["clause_type"] == q_type
            )
            metrics[f"precision_at_{k}"] += matches / k

            # Top-K Consistency: Matches both Category AND Risk Label?
            label_matches = sum(
                1
                for idx in k_indices
                if train_rows[idx]["clause_type"] == q_type
                and train_rows[idx]["label"] == q_label
            )
            metrics[f"consistency_at_{k}"] += label_matches / k

        # 3. Global Cohort Label Agreement (Static Baseline)
        category_indices = [
            idx for idx in sorted_indices if train_rows[idx]["clause_type"] == q_type
        ]
        if category_indices:
            agreements = sum(
                1 for idx in category_indices if train_rows[idx]["label"] == q_label
            )
            metrics["label_agreement_sum"] += agreements / len(category_indices)
            metrics["label_agreement_count"] += 1

    # Normalize metrics
    metrics["mrr"] = round(metrics["mrr"] / eval_count, 4)
    for k in k_levels:
        metrics[f"precision_at_{k}"] = round(
            metrics[f"precision_at_{k}"] / eval_count, 4
        )
        metrics[f"consistency_at_{k}"] = round(
            metrics[f"consistency_at_{k}"] / eval_count, 4
        )

    if metrics["label_agreement_count"] > 0:
        metrics["label_agreement_rate"] = round(
            metrics["label_agreement_sum"] / metrics["label_agreement_count"], 4
        )
    else:
        metrics["label_agreement_rate"] = 0.0

    del metrics["label_agreement_sum"]
    del metrics["label_agreement_count"]

    metrics["avg_query_latency_ms"] = round((eval_time / eval_count) * 1000, 2)
    metrics["total_encode_time_seconds"] = round(train_time + eval_time, 2)

    return metrics


def main():
    splits_path = "data/processed/splits.json"
    dataset_path = "data/processed/training_dataset.json"

    if not os.path.exists(splits_path) or not os.path.exists(dataset_path):
        logger.error("Required dataset or splits files are missing.")
        sys.exit(1)

    with open(splits_path) as f:
        splits = json.load(f)
    with open(dataset_path) as f:
        dataset = json.load(f)

    row_map = {row["row_num"]: row for row in dataset if row.get("row_num") is not None}

    train_rows = [
        row_map[num]
        for num in splits["train"]
        if num in row_map and row_map[num].get("label") is not None
    ]
    val_rows = [
        row_map[num]
        for num in splits["test"]
        if num in row_map and row_map[num].get("label") is not None
    ]

    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "nlpaueb/legal-bert-base-uncased",
    ]

    results = {}
    for model_name in models_to_test:
        results[model_name] = evaluate_embedding_model(model_name, train_rows, val_rows)

    # ------------------------------------------------------------------
    # Breathtaking Symmetrical Unicode ANSI Table Drawer (2-Column)
    # ------------------------------------------------------------------
    import re

    BOLD = "\033[1m"
    CYAN = "\033[1;36m"
    GREEN = "\033[1;32m"
    RED = "\033[1;31m"
    RESET = "\033[0m"

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def format_diff(val, base, is_percent=True, is_higher_better=True):
        diff = val - base
        if abs(diff) < 1e-6:
            return ""
        sign = "+" if diff > 0 else ""
        color = GREEN if (diff > 0 if is_higher_better else diff < 0) else RED
        if is_percent:
            return f"{color}({sign}{diff:.2%}){RESET}"
        else:
            return f"{color}({sign}{diff:+.4f}){RESET}"

    def make_cell(val_str, diff_str="", width=24):
        visible_val = ansi_escape.sub("", val_str)
        visible_diff = ansi_escape.sub("", diff_str)
        visible_len = len(visible_val) + (len(visible_diff) + 1 if visible_diff else 0)
        padding = " " * max(0, width - visible_len)
        if diff_str:
            return f"{val_str} {diff_str}{padding}"
        else:
            return f"{val_str}{padding}"

    minilm_l6 = results["sentence-transformers/all-MiniLM-L6-v2"]
    legal_bert = results["nlpaueb/legal-bert-base-uncased"]

    # Identify winners
    p1_winner = max(models_to_test, key=lambda m: results[m]["precision_at_1"])
    p5_winner = max(models_to_test, key=lambda m: results[m]["precision_at_5"])
    p10_winner = max(models_to_test, key=lambda m: results[m]["precision_at_10"])

    c1_winner = max(models_to_test, key=lambda m: results[m]["consistency_at_1"])
    c5_winner = max(models_to_test, key=lambda m: results[m]["consistency_at_5"])
    c10_winner = max(models_to_test, key=lambda m: results[m]["consistency_at_10"])

    mrr_winner = max(models_to_test, key=lambda m: results[m]["mrr"])
    lat_winner = min(models_to_test, key=lambda m: results[m]["avg_query_latency_ms"])

    def val_format(model, key, winner, is_percent=True, decimal_places=2, suffix=""):
        val = results[model][key]
        val_str = (
            f"{val:.{decimal_places}%}" if is_percent else f"{val:.{decimal_places}f}"
        )
        val_str = f"{val_str}{suffix}"
        if model == winner:
            return f"{GREEN}{BOLD}👑 {val_str}{RESET}"
        return val_str

    print("\n" + CYAN + "╔" + "═" * 86 + "╗" + RESET)
    print(
        CYAN
        + "║"
        + BOLD
        + "             💡 DENSE EMBEDDING RETRIEVAL: MiniLM VS LegalBERT TOURNAMENT             "
        + RESET
        + CYAN
        + "║"
        + RESET
    )
    print(CYAN + "╠" + "═" * 86 + "╣" + RESET)
    print(
        CYAN
        + "║"
        + RESET
        + f"  {BOLD}Test Queries:{RESET} {len(val_rows)} clauses  │  {BOLD}Index Database Size:{RESET} {len(train_rows)} clauses            "
        + CYAN
        + "║"
        + RESET
    )

    # Table Header
    print(
        CYAN
        + "╠"
        + "═" * 32
        + "╦"
        + "═" * 26
        + "╦"
        + "═" * 26
        + "╣"
        + RESET
    )
    print(
        CYAN
        + "║"
        + BOLD
        + f" {CYAN}RETRIEVAL METRICS            {RESET}"
        + CYAN
        + "║"
        + BOLD
        + " all-MiniLM-L6-v2 (Base)  "
        + CYAN
        + "║"
        + BOLD
        + " LegalBERT (768d)         "
        + CYAN
        + "║"
        + RESET
    )
    print(
        CYAN
        + "╠"
        + "═" * 32
        + "╬"
        + "═" * 26
        + "╬"
        + "═" * 26
        + "╣"
        + RESET
    )

    # Row: Precision@1
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "precision_at_1", p1_winner))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "precision_at_1", p1_winner), format_diff(legal_bert["precision_at_1"], minilm_l6["precision_at_1"]))
    print(CYAN + "║" + RESET + " Precision@1 (Category Match)  " + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    # Row: Precision@5
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "precision_at_5", p5_winner))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "precision_at_5", p5_winner), format_diff(legal_bert["precision_at_5"], minilm_l6["precision_at_5"]))
    print(CYAN + "║" + RESET + " Precision@5                   " + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    # Row: Precision@10
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "precision_at_10", p10_winner))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "precision_at_10", p10_winner), format_diff(legal_bert["precision_at_10"], minilm_l6["precision_at_10"]))
    print(CYAN + "║" + RESET + " Precision@10                  " + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    # Row: MRR
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "mrr", mrr_winner, is_percent=False, decimal_places=4))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "mrr", mrr_winner, is_percent=False, decimal_places=4), format_diff(legal_bert["mrr"], minilm_l6["mrr"], is_percent=False))
    print(CYAN + "║" + RESET + " Mean Reciprocal Rank (MRR)    " + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    print(
        CYAN
        + "╠"
        + "═" * 32
        + "╬"
        + "═" * 26
        + "╬"
        + "═" * 26
        + "╣"
        + RESET
    )

    # Row: Consistency@1
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "consistency_at_1", c1_winner))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "consistency_at_1", c1_winner), format_diff(legal_bert["consistency_at_1"], minilm_l6["consistency_at_1"]))
    print(CYAN + "║" + RESET + " Top-1 Risk Consistency (Ragas) " + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    # Row: Consistency@5
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "consistency_at_5", c5_winner))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "consistency_at_5", c5_winner), format_diff(legal_bert["consistency_at_5"], minilm_l6["consistency_at_5"]))
    print(CYAN + "║" + RESET + " Top-5 Risk Consistency        " + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    # Row: Consistency@10
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "consistency_at_10", c10_winner))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "consistency_at_10", c10_winner), format_diff(legal_bert["consistency_at_10"], minilm_l6["consistency_at_10"]))
    print(CYAN + "║" + RESET + " Top-10 Risk Consistency       " + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    print(
        CYAN
        + "╠"
        + "═" * 32
        + "╬"
        + "═" * 26
        + "╬"
        + "═" * 26
        + "╣"
        + RESET
    )

    # Row: Encoding Time
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "total_encode_time_seconds", None, is_percent=False, suffix="s"))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "total_encode_time_seconds", None, is_percent=False, suffix="s"))
    print(CYAN + "║" + RESET + " Total Encoding Time (3.4k rows)" + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    # Row: Latency
    cell_l6 = make_cell(val_format("sentence-transformers/all-MiniLM-L6-v2", "avg_query_latency_ms", lat_winner, is_percent=False, suffix="ms"))
    cell_lb = make_cell(val_format("nlpaueb/legal-bert-base-uncased", "avg_query_latency_ms", lat_winner, is_percent=False, suffix="ms"))
    print(CYAN + "║" + RESET + " Avg Retrieval Latency / Query " + CYAN + "║ " + cell_l6 + " " + CYAN + "║ " + cell_lb + " " + CYAN + "║" + RESET)

    print(
        CYAN
        + "╚"
        + "═" * 32
        + "╩"
        + "═" * 26
        + "╩"
        + "═" * 26
        + "╝"
        + RESET
        + "\n"
    )

    # Save comparison report
    out_dir = "data/output/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embedding_comparison_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved embedding comparative metrics to %s", out_path)


if __name__ == "__main__":
    main()
