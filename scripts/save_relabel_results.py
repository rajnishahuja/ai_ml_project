"""
save_relabel_results.py
=======================
Parses Claude's JSON label output for a batch and appends to the master
results file. Must be run BEFORE /compact so the output is still in the
terminal scroll buffer — paste it in when prompted, or pipe it via stdin.

Usage (interactive — paste Claude's output when prompted):
    python scripts/save_relabel_results.py --batch 0

Usage (pipe from file if you saved Claude's output):
    python scripts/save_relabel_results.py --batch 0 --input-file batch_0_output.json

After all batches are done:
    python scripts/save_relabel_results.py --merge
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR   = Path("data/review/relabel_batches")
PROGRESS_FILE = RESULTS_DIR / "progress.json"
MERGED_FILE   = Path("data/review/claude_relabels.json")

VALID_LABELS = {"LOW", "MEDIUM", "HIGH"}


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_batches": [], "labeled_row_nums": []}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def parse_json_output(raw: str) -> list[dict]:
    """Extract and parse the JSON array from Claude's output."""
    raw = raw.strip()
    # Find the first '[' and last ']'
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in output — expected [...] structure")
    json_str = raw[start:end+1]
    results = json.loads(json_str)
    return results


def validate(results: list[dict], batch_file: Path) -> list[dict]:
    """Validate each result object has required fields."""
    # Load expected row nums from input file
    input_file = Path(str(batch_file).replace("_output.json", "_input.txt"))
    expected_rows = set()
    if input_file.exists():
        for line in input_file.read_text().splitlines():
            if line.startswith("## Row "):
                try:
                    rn = int(line.split("## Row ")[1].split()[0])
                    expected_rows.add(rn)
                except (IndexError, ValueError):
                    pass

    validated = []
    for item in results:
        if not isinstance(item, dict):
            print(f"  SKIP: non-dict item {item}", file=sys.stderr)
            continue
        if "row_num" not in item or "label" not in item:
            print(f"  SKIP: missing row_num or label: {item}", file=sys.stderr)
            continue
        if item["label"] not in VALID_LABELS:
            print(f"  SKIP row {item['row_num']}: invalid label {item['label']!r}", file=sys.stderr)
            continue
        if expected_rows and item["row_num"] not in expected_rows:
            print(f"  WARN row {item['row_num']}: not in expected batch rows", file=sys.stderr)
        validated.append({
            "row_num": int(item["row_num"]),
            "label":   item["label"],
            "reason":  item.get("reason", ""),
        })

    return validated


def save_batch(batch_idx: int, results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"batch_{batch_idx:03d}_output.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    progress = load_progress()
    if batch_idx not in progress["completed_batches"]:
        progress["completed_batches"].append(batch_idx)
    labeled_set = set(progress["labeled_row_nums"])
    labeled_set.update(r["row_num"] for r in results)
    progress["labeled_row_nums"] = sorted(labeled_set)
    save_progress(progress)

    print(f"Saved {len(results)} labels to {out_file}")
    print(f"Total labeled so far: {len(progress['labeled_row_nums'])}")


def merge_all():
    """Merge all batch output files into a single results file."""
    batch_files = sorted(RESULTS_DIR.glob("batch_*_output.json"))
    if not batch_files:
        print("No batch output files found.")
        return

    all_results = {}
    for bf in batch_files:
        batch = json.loads(bf.read_text())
        for item in batch:
            all_results[item["row_num"]] = item

    merged = sorted(all_results.values(), key=lambda x: x["row_num"])
    MERGED_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged {len(merged)} labels from {len(batch_files)} batches → {MERGED_FILE}")

    # Label distribution
    from collections import Counter
    dist = Counter(r["label"] for r in merged)
    print(f"Label distribution: {dict(dist)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",      type=int, help="Batch index to save")
    parser.add_argument("--input-file", type=str, help="Path to file with Claude's JSON output")
    parser.add_argument("--merge",      action="store_true", help="Merge all batches into final file")
    args = parser.parse_args()

    if args.merge:
        merge_all()
        return

    if args.batch is None:
        parser.error("--batch required unless --merge")

    if args.input_file:
        raw = Path(args.input_file).read_text(encoding="utf-8")
    else:
        print("Paste Claude's JSON output below, then press Ctrl+D (Linux) or Ctrl+Z (Windows):")
        raw = sys.stdin.read()

    try:
        results = parse_json_output(raw)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"ERROR parsing output: {e}", file=sys.stderr)
        sys.exit(1)

    batch_file = RESULTS_DIR / f"batch_{args.batch:03d}_output.json"
    validated  = validate(results, batch_file)

    if len(validated) != len(results):
        print(f"WARNING: {len(results) - len(validated)} items dropped during validation")

    save_batch(args.batch, validated)


if __name__ == "__main__":
    main()
