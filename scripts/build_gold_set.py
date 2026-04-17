"""
build_gold_set.py
==================
Pick a 25-clause stratified gold set for Stage 3 prompt iteration.

Stratification is grounded in CUAD's own 41-category taxonomy, classified
by how strongly the *type alone* signals risk (based on Atticus's own
category descriptions — see data/reference/cuad_category_descriptions.csv):

  Group A — presence alone signals HIGH        (8 types  — all sampled)
  Group B — wording determines the level       (26 types — 10 sampled for reasoning-mode diversity)
  Group C — informational or neutral           (7 types  — excluded; auto-labeled LOW elsewhere)

Plus:
  Edge cases                                   (4 clauses: shortest, longest, fragmented, ambiguous)
  Random picks                                 (3 clauses, seed=42)

Total: 25 clauses. Deterministic — rerunning produces the same set.

Each output row is annotated with:
  - _bucket                  (high_risk | mixed | edge | random)
  - _pick_reason             (why this specific clause was picked)
  - _category_description    (Atticus's one-line definition of the clause type)
  - _atticus_group           (Atticus's own thematic group: 1-6 or "-")

Inputs:
    data/processed/all_positive_spans.json
    data/reference/cuad_category_descriptions.csv

Output:
    data/synthetic/gold_set.json

Usage:
    python scripts/build_gold_set.py
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

SPANS_PATH      = Path("data/processed/all_positive_spans.json")
CATEGORIES_PATH = Path("data/reference/cuad_category_descriptions.csv")
OUTPUT_PATH     = Path("data/synthetic/gold_set.json")

# -------- Risk-signal grouping (derived from Atticus descriptions) --------

# Group A: presence alone signals HIGH risk.
# Each of these is definitionally risky (unlimited, unrestricted, irrevocable,
# or binds you to minimum/restrictive terms regardless of wording nuance).
HIGH_RISK_TYPES = [
    "Uncapped Liability",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Most Favored Nation",
    "Minimum Commitment",
    "Non-Compete",
    "Exclusivity",
    "Covenant Not To Sue",
]

# Group B: wording-dependent. 10 picks chosen for reasoning-mode diversity —
# each type tests a different kind of risk reasoning the prompt must handle.
MIXED_TYPES = [
    "Cap On Liability",             # amount reasoning — how high is the cap?
    "Insurance",                    # amount reasoning — what coverage levels?
    "Liquidated Damages",           # amount reasoning — reasonable or punitive?
    "Warranty Duration",            # duration reasoning — how long is protection?
    "License Grant",                # scope reasoning — territory/exclusivity/field
    "Audit Rights",                 # scope reasoning — frequency/notice/breadth
    "Change Of Control",            # trigger reasoning — what events activate?
    "Termination For Convenience",  # trigger reasoning — notice period + cure
    "Governing Law",                # jurisdiction reasoning — favorable or hostile
    "Anti-Assignment",              # consent-standard reasoning — strict or loose
]

# Metadata clauses — purely descriptive, route away from risk classifier entirely
# (per Option B architectural decision). Excluded from the gold set's random and
# edge-case picks so the prompt is never validated on non-risk-bearing content.
METADATA_TYPES = frozenset({
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
})

AMBIGUOUS_STARTS = {"It", "Such", "These", "They", "He", "She", "This", "That"}
# "This Agreement is..." is a common non-ambiguous contract opener; exclude it.
NON_AMBIGUOUS_CONTINUATIONS = {"Agreement", "Contract", "Schedule", "Exhibit", "Section"}

MIN_EDGE_SHORT_LEN = 50  # Ignore degenerate 1-char / few-char spans as the "shortest" pick.


# ------------------------------- helpers -------------------------------

def load_category_metadata(path: Path) -> dict:
    """
    Parse the Atticus category_descriptions.csv.

    Each cell is prefixed with its column label (e.g. "Category: Foo",
    "Description: Bar") — we strip that. Also handles a leading BOM and
    normalizes whitespace.

    Returns a dict keyed by the *lowercased* category name so lookup
    tolerates casing differences between CSV ("Cap on Liability") and
    JSON ("Cap On Liability").
    """
    out: dict = {}
    with path.open(encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            name  = row[0].replace("Category: ", "").strip()
            desc  = row[1].replace("Description: ", "").strip()
            group = row[3].replace("Group: ", "").strip() if len(row) > 3 else ""
            out[name.lower()] = {"name": name, "description": desc, "group": group}
    return out


def describe(clause_type: str, cat_meta: dict) -> tuple[str, str]:
    """Return (description, atticus_group) for a clause type, case-insensitive."""
    m = cat_meta.get(clause_type.lower(), {})
    return m.get("description", ""), m.get("group", "")


def pick_first_of_type(spans: list, clause_type: str, used_texts: set) -> dict | None:
    """First span of the given type whose text hasn't already been picked."""
    for s in spans:
        if s["clause_type"] == clause_type and s["clause_text"] not in used_texts:
            return s
    return None


def pick_edge_cases(spans: list, used_ids: set, used_texts: set) -> list:
    """Return list of (span, reason) for 4 edge cases. Excludes metadata and duplicates."""
    picks = []

    def remaining():
        return [
            s for s in spans
            if s["id"] not in used_ids
            and s["clause_text"] not in used_texts
            and s["clause_type"] not in METADATA_TYPES
        ]

    shortest = min(
        (s for s in remaining() if len(s["clause_text"]) >= MIN_EDGE_SHORT_LEN),
        key=lambda s: len(s["clause_text"]),
    )
    used_ids.add(shortest["id"])
    picks.append((shortest, f"shortest (>= {MIN_EDGE_SHORT_LEN} chars, got {len(shortest['clause_text'])})"))

    longest = max(remaining(), key=lambda s: len(s["clause_text"]))
    used_ids.add(longest["id"])
    picks.append((longest, f"longest ({len(longest['clause_text'])} chars)"))

    fragmented = next(
        (s for s in remaining()
         if s["clause_text"].strip() and s["clause_text"].strip()[0].islower()),
        None,
    )
    if fragmented:
        used_ids.add(fragmented["id"])
        picks.append((fragmented, "fragmented (starts lowercase — likely mid-sentence)"))

    def is_ambiguous(text: str) -> bool:
        words = text.strip().split()[:2]
        if not words or words[0] not in AMBIGUOUS_STARTS:
            return False
        # "This Agreement" / "Such Agreement" etc. are well-defined, not ambiguous.
        if len(words) > 1 and words[1] in NON_AMBIGUOUS_CONTINUATIONS:
            return False
        return True

    ambiguous = next(
        (s for s in remaining() if is_ambiguous(s["clause_text"])),
        None,
    )
    if ambiguous:
        used_ids.add(ambiguous["id"])
        picks.append((ambiguous, "ambiguous (starts with pronoun — needs context)"))

    return picks


def annotate(span: dict, bucket: str, reason: str, cat_meta: dict) -> dict:
    desc, group = describe(span["clause_type"], cat_meta)
    return {
        **span,
        "_bucket": bucket,
        "_pick_reason": reason,
        "_category_description": desc,
        "_atticus_group": group,
    }


# --------------------------------- main ---------------------------------

def main() -> None:
    spans = json.loads(SPANS_PATH.read_text(encoding="utf-8"))
    cat_meta = load_category_metadata(CATEGORIES_PATH)
    print(f"Loaded {len(spans)} positive spans and {len(cat_meta)} category descriptions")

    # Sanity: every type we want to pick must exist in both the JSON and the CSV
    available_types = {s["clause_type"] for s in spans}
    for t in HIGH_RISK_TYPES + MIXED_TYPES:
        if t not in available_types:
            raise ValueError(f"Clause type not found in spans: {t!r}")
        if t.lower() not in cat_meta:
            raise ValueError(f"Clause type not found in category descriptions: {t!r}")

    used_ids: set = set()
    used_texts: set = set()  # dedup on clause_text — no point labeling identical text twice
    picks: list = []

    # Group A — high-risk types (all 8)
    for t in HIGH_RISK_TYPES:
        s = pick_first_of_type(spans, t, used_texts)
        picks.append(annotate(s, "high_risk", f"first '{t}' (Group A — presence signals HIGH)", cat_meta))
        used_ids.add(s["id"])
        used_texts.add(s["clause_text"])

    # Group B — mixed types (10 picks for reasoning-mode diversity)
    for t in MIXED_TYPES:
        s = pick_first_of_type(spans, t, used_texts)
        picks.append(annotate(s, "mixed", f"first '{t}' (Group B — wording determines level)", cat_meta))
        used_ids.add(s["id"])
        used_texts.add(s["clause_text"])

    # Edge cases (metadata excluded; text-dedup enforced)
    for s, reason in pick_edge_cases(spans, used_ids, used_texts):
        picks.append(annotate(s, "edge", reason, cat_meta))
        used_texts.add(s["clause_text"])

    # Random picks (metadata excluded; text-dedup enforced)
    random.seed(42)
    available = [
        s for s in spans
        if s["id"] not in used_ids
        and s["clause_text"] not in used_texts
        and s["clause_type"] not in METADATA_TYPES
    ]
    for s in random.sample(available, 3):
        picks.append(annotate(s, "random", "random seed=42 (risk-bearing only)", cat_meta))
        used_texts.add(s["clause_text"])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(picks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    buckets = Counter(p["_bucket"] for p in picks)
    print(f"\nWrote {len(picks)} clauses to {OUTPUT_PATH}")
    for b in ["high_risk", "mixed", "edge", "random"]:
        print(f"  {b:10s}: {buckets[b]}")


if __name__ == "__main__":
    main()
