"""
Quick ablation study — run all 4 modes on a small stratified sample.

Runs:
  1. DeBERTa-only         (no LLM — always fast)
  2. RAG-only             (no LLM — always fast)
  3. Pipeline, no CS      (LLM, ~N * 7s)
  4. Pipeline, full       (LLM + contract_search fix, ~N * 8s)

Each mode writes its own output + checkpoint file so you can resume
any of them to the full 452-clause set afterward:

  python scripts/eval_stage3.py --full [--no-contract-search] \\
      --output <same output path> --resume

Usage:
  python scripts/quick_ablation.py              # 30 samples, seed 42
  python scripts/quick_ablation.py --sample 60  # 60 samples
  python scripts/quick_ablation.py --seed 7     # different random split
"""

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
EVAL   = "scripts/eval_stage3.py"

MODES = [
    {
        "label":  "RAG-only (FAISS majority vote)",
        "flags":  ["--rag-only"],
        "output": "data/eval/quick_rag_only.json",
        "llm":    False,
    },
    {
        "label":  "Pipeline — no contract_search",
        "flags":  ["--no-contract-search"],
        "output": "data/eval/quick_no_cs.json",
        "llm":    True,
    },
    {
        "label":  "Pipeline — full (contract_search fix)",
        "flags":  [],
        "output": "data/eval/quick_full.json",
        "llm":    True,
    },
]


def run_mode(mode: dict, sample: int, seed: int) -> float | None:
    """Run one eval mode; return macro F1 parsed from stdout."""
    cmd = [
        PYTHON, EVAL,
        "--sample", str(sample),
        "--seed",   str(seed),
        "--output", mode["output"],
    ] + mode["flags"]

    print(f"\n{'='*60}")
    print(f"  Running: {mode['label']}")
    print(f"  Sample : {sample} clauses  |  seed={seed}")
    print(f"  Output : {mode['output']}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, text=True, capture_output=False)

    if result.returncode != 0:
        print(f"  ERROR: mode failed (exit {result.returncode})")
        return None

    import json
    out = Path(mode["output"])
    if not out.exists():
        return None
    data = json.load(open(out))
    rows = data.get("rows", [])
    if not rows:
        return None
    from sklearn.metrics import f1_score
    y_true = [r["ground_truth"]   for r in rows]
    y_pred = [r["pipeline_label"] for r in rows]
    return f1_score(y_true, y_pred, labels=["LOW","MEDIUM","HIGH"], average="macro")


def main():
    parser = argparse.ArgumentParser(description="Quick ablation across all pipeline modes.")
    parser.add_argument("--sample", type=int, default=30,
                        help="Clauses to sample per run, stratified (default: 30 = 10 per class)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Only run the fast (no-LLM) modes — useful if server is busy")
    args = parser.parse_args()

    results = {}

    for mode in MODES:
        if args.skip_llm and mode["llm"]:
            print(f"\nSkipping (--skip-llm): {mode['label']}")
            continue
        f1 = run_mode(mode, args.sample, args.seed)
        results[mode["label"]] = f1

    # Summary table
    print(f"\n{'='*60}")
    print(f"  QUICK ABLATION SUMMARY  (n={args.sample}, seed={args.seed})")
    print(f"{'='*60}")
    print(f"  {'Mode':<40}  Macro F1")
    print(f"  {'-'*40}  --------")
    print(f"  {'DeBERTa-only baseline':<40}  ~0.607  (training val)")
    for label, f1 in results.items():
        val = f"{f1:.4f}" if f1 is not None else "  FAILED"
        print(f"  {label:<40}  {val}")
    print(f"{'='*60}")

    print("\nTo resume any mode to the full 452-clause eval:")
    for mode in MODES:
        flag = " ".join(mode["flags"]) if mode["flags"] else ""
        print(f"  python scripts/eval_stage3.py --full {flag} "
              f"--output {mode['output']} --resume")


if __name__ == "__main__":
    main()
