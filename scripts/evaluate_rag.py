"""
RAG Retrieval Quality Evaluator.

This script evaluates the FAISS index semantic retrieval performance on the 
validation and test splits (as defined in splits.json). It calculates core RAG 
metrics like Precision@K, Mean Reciprocal Rank (MRR), and Risk Label Consistency.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage3_risk_agent.embeddings import query_index
from src.common.utils import setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger("evaluate_rag")

def evaluate_split(split_name: str, rows: list, index_path: str, k_levels: list[int] = [1, 3, 5, 10], min_similarity: float = 0.75) -> dict:
    """Evaluate retrieval metrics for a single partition split."""
    logger.info("Evaluating split '%s' on %d clauses ...", split_name, len(rows))
    
    total_clauses = len(rows)
    if total_clauses == 0:
        return {}
        
    metrics = {
        "count": total_clauses,
        "mrr": 0.0,
        "label_agreement_sum": 0.0,
        "label_agreement_count": 0,
    }
    
    for k in k_levels:
        metrics[f"precision_at_{k}"] = 0.0
        
    start_time = time.perf_counter()
    
    for idx, query in enumerate(rows):
        if (idx + 1) % 50 == 0 or idx + 1 == total_clauses:
            logger.info("  Processed %d/%d clauses ...", idx + 1, total_clauses)
            
        q_text = query["clause_text"]
        q_type = query["clause_type"]
        q_label = query["label"]
        
        # Max query count K is max(k_levels)
        max_k = max(k_levels)
        results = query_index(q_text, index_path, k=max_k)
        
        # 1. Compute MRR (Mean Reciprocal Rank) for Category Match
        first_match_rank = 0
        for rank, res in enumerate(results, 1):
            if res.clause_type == q_type:
                first_match_rank = rank
                break
        if first_match_rank > 0:
            metrics["mrr"] += 1.0 / first_match_rank
            
        # 2. Compute Precision@K for each level
        for k in k_levels:
            k_results = results[:k]
            if not k_results:
                continue
            matches = sum(1 for r in k_results if r.clause_type == q_type)
            metrics[f"precision_at_{k}"] += matches / len(k_results)
            
        # 3. Compute Risk Label Consistency (Label Agreement) for matching categories
        # Out of the retrieved precedents with the same clause_type, do they agree on the risk level?
        category_matches = [r for r in results if r.clause_type == q_type]
        if category_matches:
            agreements = sum(1 for r in category_matches if r.risk_level == q_label)
            metrics["label_agreement_sum"] += agreements / len(category_matches)
            metrics["label_agreement_count"] += 1

    duration = time.perf_counter() - start_time
    
    # Normalize sums to get final metrics
    metrics["mrr"] = round(metrics["mrr"] / total_clauses, 4)
    for k in k_levels:
        metrics[f"precision_at_{k}"] = round(metrics[f"precision_at_{k}"] / total_clauses, 4)
        
    if metrics["label_agreement_count"] > 0:
        metrics["label_agreement_rate"] = round(metrics["label_agreement_sum"] / metrics["label_agreement_count"], 4)
    else:
        metrics["label_agreement_rate"] = 0.0
        
    del metrics["label_agreement_sum"]
    del metrics["label_agreement_count"]
    
    metrics["evaluation_time_seconds"] = round(duration, 2)
    metrics["average_latency_ms"] = round((duration / total_clauses) * 1000, 2)
    
    return metrics

def main():
    splits_path = "data/processed/splits.json"
    dataset_path = "data/processed/training_dataset.json"
    index_path = "data/faiss_index/clauses.index"
    
    if not os.path.exists(splits_path) or not os.path.exists(dataset_path) or not os.path.exists(index_path):
        logger.error("Required dataset or FAISS files are missing. Please verify data paths.")
        sys.exit(1)
        
    logger.info("Loading splits from %s", splits_path)
    with open(splits_path) as f:
        splits = json.load(f)
        
    logger.info("Loading training dataset from %s", dataset_path)
    with open(dataset_path) as f:
        dataset = json.load(f)
        
    # Map row_num → row data
    row_map = {row["row_num"]: row for row in dataset if row.get("row_num") is not None}
    
    val_rows = [row_map[num] for num in splits["val"] if num in row_map and row_map[num].get("label") is not None]
    test_rows = [row_map[num] for num in splits["test"] if num in row_map and row_map[num].get("label") is not None]
    
    # Subset val & test to 100 random rows each if too large, to keep it extremely fast
    import random
    random.seed(42)
    
    if len(val_rows) > 150:
        val_rows = random.sample(val_rows, 100)
        logger.info("Subsampled Validation Split to 100 representative clauses.")
    if len(test_rows) > 150:
        test_rows = random.sample(test_rows, 100)
        logger.info("Subsampled Test Split to 100 representative clauses.")
        
    val_results = evaluate_split("val", val_rows, index_path)
    test_results = evaluate_split("test", test_rows, index_path)
    
    # ------------------------------------------------------------------
    # Gorgeous Console Dashboard
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("             FAISS RAG VECTOR RETRIEVAL QUALITY AUDIT")
    print("=" * 65)
    print(f"  Index Vectors: {len(row_map):<5} | Index Dimensions: 384 (all-MiniLM-L6-v2)")
    print("-" * 65)
    print(f"  METRICS                     | VALIDATION SPLIT  | TEST SPLIT")
    print("-" * 65)
    print(f"  Evaluated Clauses           | {val_results['count']:<17} | {test_results['count']:<10}")
    print(f"  Precision@1 (Match@1)       | {val_results['precision_at_1']:<17.2%} | {test_results['precision_at_1']:<.2%}")
    print(f"  Precision@3                 | {val_results['precision_at_3']:<17.2%} | {test_results['precision_at_3']:<.2%}")
    print(f"  Precision@5                 | {val_results['precision_at_5']:<17.2%} | {test_results['precision_at_5']:<.2%}")
    print(f"  Precision@10                | {val_results['precision_at_10']:<17.2%} | {test_results['precision_at_10']:<.2%}")
    print(f"  Mean Reciprocal Rank (MRR)  | {val_results['mrr']:<17.4f} | {test_results['mrr']:<.4f}")
    print(f"  Label Risk Consistency      | {val_results['label_agreement_rate']:<17.2%} | {test_results['label_agreement_rate']:<.2%}")
    print("-" * 65)
    print(f"  Total Duration              | {val_results['evaluation_time_seconds']:>6.2f}s          | {test_results['evaluation_time_seconds']:>6.2f}s")
    print(f"  Average Latency / Query     | {val_results['average_latency_ms']:>6.2f}ms         | {test_results['average_latency_ms']:>6.2f}ms")
    print("=" * 65 + "\n")
    
    # Save results to output
    out_dir = "data/output/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rag_retrieval_metrics.json")
    with open(out_path, "w") as f:
        json.dump({
            "validation_split": val_results,
            "test_split": test_results
        }, f, indent=2)
    logger.info("Saved RAG evaluation metrics to %s", out_path)

if __name__ == "__main__":
    main()
