"""
Build a pre-computed contract → clauses index for the Stage 3 contract_search tool.

Reads the Stage 1+2 spans corpus (`data/processed/all_positive_spans.json`)
and writes a dict keyed by `document_id`, where each value is a small object
containing every clause Stage 1+2 extracted for that contract — both the 5
metadata clause types and the risk-relevant ones, with an `is_metadata` flag
so consumers can filter cheaply.

Output shape (`data/processed/contract_clause_index.json`):

    {
      "<document_id>": {
        "document_id": "...",
        "n_clauses": 14,
        "n_metadata": 5,
        "clauses": [
          {
            "clause_id":   "...",
            "clause_type": "Document Name",
            "clause_text": "...",
            "start_pos":   0,
            "is_metadata": true
          },
          ...
        ]
      },
      ...
    }

Why pre-compute:
  - At Stage 3 inference, the agent calls `contract_search(document_id)` from
    inside a Mistral tool loop. A direct dict lookup beats re-filtering the
    full 6,702-row corpus on every call.
  - The index file is a deliverable artifact: ships with the model, can be
    inspected by reviewers, and is independent of any pickle / runtime cache.
  - Keeping metadata in the file (with a flag) lets `contract_search`
    support both default (`include_metadata=False`) and metadata-aware
    callers without two passes.

Usage:
    python scripts/build_contract_clause_index.py
    python scripts/build_contract_clause_index.py --spans path/to/spans.json --out path/to/out.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

# Mirror the constant in src/stage3_risk_agent/tools.py — kept literal here so
# the script has no import-time dependency on the package.
METADATA_CLAUSE_TYPES = frozenset({
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
})

DEFAULT_SPANS_PATH = "data/processed/all_positive_spans.json"
DEFAULT_OUTPUT_PATH = "data/processed/contract_clause_index.json"

logger = logging.getLogger(__name__)


def _normalize_span(span: dict) -> dict:
    """Map a raw spans-file row to the index's per-clause shape.

    The on-disk spans file uses `id` / `contract` / `answer_start`. The
    index uses `clause_id` / `start_pos` and adds `is_metadata`.
    """
    clause_type = span.get("clause_type", "")
    return {
        "clause_id": span.get("clause_id") or span.get("id", ""),
        "clause_type": clause_type,
        "clause_text": span.get("clause_text", ""),
        "start_pos": span.get("start_pos") or span.get("answer_start", 0),
        "is_metadata": clause_type in METADATA_CLAUSE_TYPES,
    }


def build_index(spans_path: str) -> dict[str, dict]:
    """Group spans by `document_id` and order each group by `start_pos`.

    Args:
        spans_path: Path to `all_positive_spans.json`.

    Returns:
        Dict keyed by `document_id`. Each value is a small object with
        counts and the ordered list of clauses (metadata included).
    """
    path = Path(spans_path)
    if not path.exists():
        raise FileNotFoundError(f"Spans file not found: {spans_path}")

    with open(path) as f:
        spans = json.load(f)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        document_id = span.get("document_id") or span.get("contract", "")
        if not document_id:
            logger.warning("Span has no document_id / contract field; skipping.")
            continue
        grouped[document_id].append(_normalize_span(span))

    index: dict[str, dict] = {}
    for document_id, clauses in grouped.items():
        # Order by start_pos so the agent reads the contract top-to-bottom.
        clauses.sort(key=lambda c: c["start_pos"])
        n_metadata = sum(1 for c in clauses if c["is_metadata"])
        index[document_id] = {
            "document_id": document_id,
            "n_clauses": len(clauses),
            "n_metadata": n_metadata,
            "clauses": clauses,
        }

    return index


def write_index(index: dict[str, dict], output_path: str) -> None:
    """Write the index to disk, creating parent directories if needed."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    size_mb = out.stat().st_size / (1024 * 1024)
    logger.info("Wrote %d contracts to %s (%.2f MB)", len(index), out, size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a contract → clauses index for the contract_search tool."
        ),
    )
    parser.add_argument(
        "--spans",
        default=DEFAULT_SPANS_PATH,
        help="Input spans file (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_PATH,
        help="Output index JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    index = build_index(args.spans)
    write_index(index, args.out)

    # Quick stats so the operator sees what shipped.
    n_contracts = len(index)
    total_clauses = sum(v["n_clauses"] for v in index.values())
    total_metadata = sum(v["n_metadata"] for v in index.values())
    avg = total_clauses / n_contracts if n_contracts else 0
    print()
    print(f"Index summary:")
    print(f"  contracts        : {n_contracts}")
    print(f"  clauses (total)  : {total_clauses}")
    print(f"  metadata clauses : {total_metadata}")
    print(f"  risk-relevant    : {total_clauses - total_metadata}")
    print(f"  avg per contract : {avg:.1f}")


if __name__ == "__main__":
    main()
