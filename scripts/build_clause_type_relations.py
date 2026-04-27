"""
Build the static `clause_type → related_types` map used by Stage 3
contract_search.

The output JSON (`data/reference/clause_type_relations.json`) tells the
agent which OTHER CUAD clause types are likely to inform a given clause's
risk assessment. At runtime, `contract_search(document_id, clause_type)`
looks up `relations[clause_type]` and returns only the contract's clauses
whose type is in that list. This focuses Mistral's evidence window on
clauses that legally interact with the target — not the full ~13-clause
sibling dump.

Curation methodology:
  Hand-curated from legal interaction patterns and the label-review
  learnings in ARCHITECTURE.md §"Labeling Review Learnings". Six clusters:

    1. Liability cluster ......... Cap / Uncapped / Liquidated / Insurance /
                                   Warranty / Covenant Not To Sue
    2. Term & termination ........ Renewal / Notice / Termination For
                                   Convenience / Post-Termination /
                                   Effective / Expiration
    3. IP & licensing ............ License Grant + 8 license-related types,
                                   IP Ownership, Joint IP, Source Code Escrow
    4. Restrictive covenants ..... Non-Compete / Exclusivity / No-Solicit
                                   variants / Most Favored Nation
    5. Commercial terms .......... Minimum Commitment / Volume Restriction /
                                   Price / Revenue Sharing / Audit Rights
    6. Assignment & control ...... Anti-Assignment / Change of Control /
                                   ROFR / Affiliate License variants

  Every type lists 0-7 related types. Boilerplate metadata types
  (Document Name, Parties) carry empty lists — they have no risk-relevant
  cross-references and are filtered out of contract_search anyway.

Usage:
    python scripts/build_clause_type_relations.py
    python scripts/build_clause_type_relations.py --out path/to/out.json --validate
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "data/reference/clause_type_relations.json"


# Canonical clause-type names as they appear in the spans corpus
# (`data/processed/all_positive_spans.json`). Verified against
# `data/reference/cuad_category_descriptions.csv`.
ALL_CLAUSE_TYPES: tuple[str, ...] = (
    # Metadata (5)
    "Document Name", "Parties", "Agreement Date",
    "Effective Date", "Expiration Date",
    # Term & termination (4)
    "Renewal Term", "Notice Period To Terminate Renewal",
    "Termination For Convenience", "Post-Termination Services",
    # Assignment & control (3)
    "Change Of Control", "Anti-Assignment", "Rofr/Rofo/Rofn",
    # Liability (6)
    "Cap On Liability", "Uncapped Liability", "Liquidated Damages",
    "Insurance", "Warranty Duration", "Covenant Not To Sue",
    # IP & licensing (9)
    "License Grant", "Non-Transferable License",
    "Irrevocable Or Perpetual License",
    "Affiliate License-Licensor", "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Ip Ownership Assignment", "Joint Ip Ownership", "Source Code Escrow",
    # Restrictive covenants (7)
    "Non-Compete", "Exclusivity",
    "No-Solicit Of Customers", "No-Solicit Of Employees",
    "Competitive Restriction Exception", "Non-Disparagement",
    "Most Favored Nation",
    # Commercial terms (5)
    "Minimum Commitment", "Volume Restriction", "Price Restrictions",
    "Revenue/Profit Sharing", "Audit Rights",
    # Boilerplate (2)
    "Governing Law", "Third Party Beneficiary",
)


# =============================================================================
# RELATIONS — hand-curated. clause_type → list of related types.
# =============================================================================

RELATIONS: dict[str, list[str]] = {
    # ---------------------------------------------------------------------
    # Metadata — excluded from contract_search by default. Empty / minimal.
    # ---------------------------------------------------------------------
    "Document Name":     [],
    "Parties":           [],
    "Agreement Date":    ["Effective Date", "Expiration Date"],
    "Effective Date":    ["Agreement Date", "Expiration Date", "Renewal Term"],
    "Expiration Date":   [
        "Effective Date", "Renewal Term",
        "Notice Period To Terminate Renewal", "Termination For Convenience",
    ],

    # ---------------------------------------------------------------------
    # Term & termination cluster
    # ---------------------------------------------------------------------
    "Renewal Term": [
        "Notice Period To Terminate Renewal", "Expiration Date",
        "Termination For Convenience", "Post-Termination Services",
    ],
    "Notice Period To Terminate Renewal": [
        "Renewal Term", "Termination For Convenience",
        "Post-Termination Services",
    ],
    "Termination For Convenience": [
        "Notice Period To Terminate Renewal", "Post-Termination Services",
        "Renewal Term", "Change Of Control", "Liquidated Damages",
    ],
    "Post-Termination Services": [
        "Termination For Convenience", "Renewal Term",
        "Notice Period To Terminate Renewal", "Warranty Duration",
        "Source Code Escrow",
    ],

    # ---------------------------------------------------------------------
    # Assignment & control
    # ---------------------------------------------------------------------
    "Change Of Control": [
        "Anti-Assignment", "Termination For Convenience",
        "Affiliate License-Licensee", "Affiliate License-Licensor",
    ],
    "Anti-Assignment": [
        "Change Of Control", "Non-Transferable License",
        "Affiliate License-Licensee", "Affiliate License-Licensor",
        "Governing Law",
    ],
    "Rofr/Rofo/Rofn": [
        "Anti-Assignment", "Change Of Control", "Exclusivity",
    ],

    # ---------------------------------------------------------------------
    # Liability cluster — these clauses constantly cross-reference
    # ---------------------------------------------------------------------
    "Cap On Liability": [
        "Uncapped Liability", "Liquidated Damages",
        "Insurance", "Warranty Duration", "Covenant Not To Sue",
    ],
    "Uncapped Liability": [
        "Cap On Liability", "Liquidated Damages", "Insurance",
        "Covenant Not To Sue",
    ],
    "Liquidated Damages": [
        "Cap On Liability", "Uncapped Liability",
        "Termination For Convenience",
    ],
    "Insurance": [
        "Cap On Liability", "Uncapped Liability", "Warranty Duration",
    ],
    "Warranty Duration": [
        "Cap On Liability", "Insurance", "Post-Termination Services",
    ],
    "Covenant Not To Sue": [
        "Cap On Liability", "Uncapped Liability",
        "Anti-Assignment", "Third Party Beneficiary",
    ],

    # ---------------------------------------------------------------------
    # IP & licensing cluster — License Grant is the hub
    # ---------------------------------------------------------------------
    "License Grant": [
        "Non-Transferable License", "Irrevocable Or Perpetual License",
        "Affiliate License-Licensee", "Affiliate License-Licensor",
        "Unlimited/All-You-Can-Eat-License", "Volume Restriction",
        "Ip Ownership Assignment", "Exclusivity",
    ],
    "Non-Transferable License": [
        "License Grant", "Anti-Assignment",
        "Affiliate License-Licensee", "Change Of Control",
    ],
    "Irrevocable Or Perpetual License": [
        "License Grant", "Termination For Convenience",
        "Post-Termination Services", "Source Code Escrow",
    ],
    "Affiliate License-Licensor": [
        "Affiliate License-Licensee", "License Grant",
        "Anti-Assignment", "Change Of Control", "Ip Ownership Assignment",
    ],
    "Affiliate License-Licensee": [
        "Affiliate License-Licensor", "License Grant",
        "Anti-Assignment", "Change Of Control", "Non-Transferable License",
    ],
    "Unlimited/All-You-Can-Eat-License": [
        "License Grant", "Volume Restriction",
        "Minimum Commitment", "Price Restrictions",
    ],
    "Ip Ownership Assignment": [
        "Joint Ip Ownership", "License Grant",
        "Source Code Escrow", "Affiliate License-Licensor",
    ],
    "Joint Ip Ownership": [
        "Ip Ownership Assignment", "License Grant",
        "Affiliate License-Licensor",
    ],
    "Source Code Escrow": [
        "Ip Ownership Assignment", "Irrevocable Or Perpetual License",
        "Post-Termination Services",
    ],

    # ---------------------------------------------------------------------
    # Restrictive covenants
    # ---------------------------------------------------------------------
    "Non-Compete": [
        "Exclusivity", "No-Solicit Of Customers", "No-Solicit Of Employees",
        "Competitive Restriction Exception", "Non-Disparagement",
    ],
    "Exclusivity": [
        "Non-Compete", "Most Favored Nation",
        "Volume Restriction", "Minimum Commitment",
        "Competitive Restriction Exception", "License Grant",
        "Rofr/Rofo/Rofn",
    ],
    "No-Solicit Of Customers": [
        "Non-Compete", "No-Solicit Of Employees",
        "Competitive Restriction Exception",
    ],
    "No-Solicit Of Employees": [
        "Non-Compete", "No-Solicit Of Customers", "Non-Disparagement",
    ],
    "Competitive Restriction Exception": [
        "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
    ],
    "Non-Disparagement": [
        "Non-Compete", "No-Solicit Of Employees",
    ],
    "Most Favored Nation": [
        "Price Restrictions", "Exclusivity",
        "Volume Restriction", "Minimum Commitment",
    ],

    # ---------------------------------------------------------------------
    # Commercial terms
    # ---------------------------------------------------------------------
    "Minimum Commitment": [
        "Volume Restriction", "Exclusivity",
        "Price Restrictions", "Revenue/Profit Sharing",
        "Most Favored Nation", "Unlimited/All-You-Can-Eat-License",
    ],
    "Volume Restriction": [
        "Minimum Commitment", "Unlimited/All-You-Can-Eat-License",
        "Price Restrictions", "License Grant", "Most Favored Nation",
    ],
    "Price Restrictions": [
        "Most Favored Nation", "Volume Restriction",
        "Minimum Commitment", "Revenue/Profit Sharing",
        "Unlimited/All-You-Can-Eat-License",
    ],
    "Revenue/Profit Sharing": [
        "Price Restrictions", "Minimum Commitment", "Audit Rights",
    ],
    "Audit Rights": [
        "Revenue/Profit Sharing", "Minimum Commitment",
        "Volume Restriction", "Price Restrictions",
    ],

    # ---------------------------------------------------------------------
    # Boilerplate
    # ---------------------------------------------------------------------
    "Governing Law": [
        "Anti-Assignment", "Third Party Beneficiary",
    ],
    "Third Party Beneficiary": [
        "Anti-Assignment", "Governing Law", "Covenant Not To Sue",
    ],
}


# =============================================================================
# Validation
# =============================================================================

def validate(relations: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Sanity-check the relations dict.

    Returns:
        (errors, warnings) — lists of human-readable strings.

    Errors (block the build):
      - Missing types: every CUAD type must have an entry.
      - Unknown referenced types: a related-type points at a non-CUAD type.

    Warnings (logged but non-blocking):
      - Asymmetric relations: A says it's related to B but B doesn't list A.
        Sometimes intentional (a heavy hub like License Grant points at many
        leaves; not every leaf points back).
      - Empty relations on non-metadata types.
    """
    errors: list[str] = []
    warnings: list[str] = []

    cuad_set = set(ALL_CLAUSE_TYPES)
    metadata = {
        "Document Name", "Parties", "Agreement Date",
        "Effective Date", "Expiration Date",
    }

    # Coverage
    missing = cuad_set - relations.keys()
    extra = relations.keys() - cuad_set
    if missing:
        errors.append(f"Missing entries for: {sorted(missing)}")
    if extra:
        errors.append(f"Unknown clause types in relations: {sorted(extra)}")

    # Referenced types must exist
    for src, targets in relations.items():
        for tgt in targets:
            if tgt not in cuad_set:
                errors.append(f"{src!r} references unknown type {tgt!r}")
            if tgt == src:
                errors.append(f"{src!r} is listed as related to itself")

    # Asymmetry
    for src, targets in relations.items():
        for tgt in targets:
            if tgt in relations and src not in relations.get(tgt, []):
                warnings.append(f"asymmetric: {src!r} → {tgt!r}, but not back")

    # Empty non-metadata
    for src, targets in relations.items():
        if src not in metadata and not targets:
            warnings.append(f"empty relations on non-metadata type {src!r}")

    return errors, warnings


def fan_in(relations: dict[str, list[str]]) -> dict[str, int]:
    """How many other types reference each type. Useful sanity stat."""
    counts: dict[str, int] = defaultdict(int)
    for targets in relations.values():
        for t in targets:
            counts[t] += 1
    return dict(counts)


# =============================================================================
# Output
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the clause_type → related_types static map.",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path (default: %(default)s).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Run validation and exit; do not write the file.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    errors, warnings = validate(RELATIONS)

    for w in warnings:
        logger.info("WARN: %s", w)
    for e in errors:
        logger.error("ERR : %s", e)

    if errors or (args.strict and warnings):
        raise SystemExit(1)

    if args.validate_only:
        logger.info("Validation OK. %d types, %d warnings.",
                    len(RELATIONS), len(warnings))
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(RELATIONS, f, indent=2, ensure_ascii=False, sort_keys=True)
    size_kb = out.stat().st_size / 1024
    logger.info("Wrote %d entries to %s (%.1f KB).",
                len(RELATIONS), out, size_kb)

    # Stats
    fans = fan_in(RELATIONS)
    out_degrees = {k: len(v) for k, v in RELATIONS.items()}
    print()
    print("Top-5 most-referenced types (in-degree):")
    for t, n in sorted(fans.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {n:>2}  {t}")
    print("\nTop-5 hub types (out-degree):")
    for t, n in sorted(out_degrees.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {n:>2}  {t}")


if __name__ == "__main__":
    main()
