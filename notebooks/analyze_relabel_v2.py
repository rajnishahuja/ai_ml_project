"""
analyze_relabel_v2.py
=====================
Comprehensive data analysis of the v2 relabeling results to inform
which rows should be sent to Claude Sonnet for tiebreaking.

Produces:
  1. Overall resolution stats (2-tier: 3-way vote > Qwen+Gemini pairwise)
  2. Clause-type breakdown of unresolved rows
  3. Disagreement direction analysis for unresolved rows
  4. Parties field quality vs agreement rate
  5. Labeler calibration vs hard-label ground truth (235 MANUAL_REVIEW rows)
  6. Systematic bias per clause type (which labeler to trust)
  7. Saves full per-row analysis to data/synthetic/relabel_v2_analysis.json
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MASTER_CSV   = Path("data/review/master_label_review.csv")
SPANS_JSON   = Path("data/processed/all_positive_spans.json")
OUTPUT_DIR   = Path("data/synthetic")
GEMINI_JSON  = OUTPUT_DIR / "soft_label_relabel_v2_gemini.json"
QWEN_JSON    = OUTPUT_DIR / "soft_label_relabel_v2_qwen.json"
GROQ_JSON    = OUTPUT_DIR / "soft_label_relabel_v2_groq.json"
ANALYSIS_OUT = OUTPUT_DIR / "relabel_v2_analysis.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_master():
    with MASTER_CSV.open(encoding="utf-8") as f:
        return {int(r["row_num"]): r for r in csv.DictReader(f)}

def load_labeler(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["row_num"]: r for r in data if r["label"] != "ERROR"}

def load_parties_lookup():
    spans = json.loads(SPANS_JSON.read_text(encoding="utf-8"))
    lookup = {}
    for s in spans:
        if s["clause_type"] == "Parties":
            lookup[s["contract"]] = s["clause_text"]
    return lookup

def parties_length_bucket(text):
    if not text or text == "Not available":
        return "missing"
    words = len(text.split())
    chars = len(text)
    if chars <= 60:
        return "very_short"
    elif chars <= 150:
        return "short"
    else:
        return "long"

# ---------------------------------------------------------------------------
# 2-tier consensus builder
# ---------------------------------------------------------------------------

def build_consensus(master, gemini, qwen, groq):
    """
    For each SOFT_LABEL row:
    - If all 3 have valid label: use majority vote (2-of-3)
    - Else if Qwen+Gemini both available: use pairwise (agree = consensus, else unresolved)
    Returns dict: row_num -> {consensus_label, method, votes, resolved}
    """
    soft_rows = {rn: r for rn, r in master.items() if r["category"] == "SOFT_LABEL"}
    results = {}

    for rn in sorted(soft_rows.keys()):
        g = gemini.get(rn, {}).get("label")
        q = qwen.get(rn, {}).get("label")
        r = groq.get(rn, {}).get("label")

        votes = {k: v for k, v in [("gemini", g), ("qwen", q), ("groq", r)] if v}

        if len(votes) >= 3:
            # 3-way majority
            vote_counts = Counter(votes.values())
            top, cnt = vote_counts.most_common(1)[0]
            if cnt >= 2:
                who = "+".join(k for k, v in votes.items() if v == top)
                results[rn] = {"consensus": top, "method": "3way", "votes": votes,
                                "resolved": True, "source": who}
            else:
                results[rn] = {"consensus": None, "method": "3way_split", "votes": votes,
                                "resolved": False, "source": "none"}
        elif g and q:
            if g == q:
                results[rn] = {"consensus": g, "method": "qwen+gemini_agree", "votes": votes,
                                "resolved": True, "source": "qwen+gemini"}
            else:
                results[rn] = {"consensus": None, "method": "qwen+gemini_disagree", "votes": votes,
                                "resolved": False, "source": "none"}
        else:
            results[rn] = {"consensus": None, "method": "insufficient_data", "votes": votes,
                            "resolved": False, "source": "none"}

    return results, soft_rows

# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "N/A"

def main():
    print("Loading data...")
    master   = load_master()
    gemini   = load_labeler(GEMINI_JSON)
    qwen     = load_labeler(QWEN_JSON)
    groq     = load_labeler(GROQ_JSON)
    parties  = load_parties_lookup()

    consensus, soft_rows = build_consensus(master, gemini, qwen, groq)

    # -----------------------------------------------------------------------
    # 1. Overall resolution stats
    # -----------------------------------------------------------------------
    section("1. OVERALL RESOLUTION STATS (2-tier voting)")

    resolved   = {rn: d for rn, d in consensus.items() if d["resolved"]}
    unresolved = {rn: d for rn, d in consensus.items() if not d["resolved"]}

    total = len(consensus)
    print(f"Total SOFT_LABEL rows: {total}")
    print(f"Resolved:   {len(resolved)} ({pct(len(resolved), total)})")
    print(f"Unresolved: {len(unresolved)} ({pct(len(unresolved), total)})")

    method_counts = Counter(d["method"] for d in consensus.values())
    print(f"\nBy method:")
    for m, cnt in method_counts.most_common():
        print(f"  {m}: {cnt}")

    res_labels = Counter(d["consensus"] for d in resolved.values())
    print(f"\nResolved label distribution:")
    for lbl in ["LOW", "MEDIUM", "HIGH"]:
        print(f"  {lbl}: {res_labels[lbl]} ({pct(res_labels[lbl], len(resolved))})")

    # -----------------------------------------------------------------------
    # 2. Labeler coverage
    # -----------------------------------------------------------------------
    section("2. LABELER COVERAGE")
    print(f"Gemini: {len(gemini)}/1327 rows")
    print(f"Qwen:   {len(qwen)}/1327 rows")
    print(f"Groq:   {len(groq)}/1327 rows (partial — daily token limit)")

    # -----------------------------------------------------------------------
    # 3. Unresolved rows: disagreement directions
    # -----------------------------------------------------------------------
    section("3. UNRESOLVED ROWS — DISAGREEMENT DIRECTION")

    dir_counts = defaultdict(int)
    for rn, d in unresolved.items():
        g = d["votes"].get("gemini", "?")
        q = d["votes"].get("qwen", "?")
        dir_counts[f"Qwen={q} / Gemini={g}"] += 1

    print(f"Total unresolved: {len(unresolved)}")
    for direction, cnt in sorted(dir_counts.items(), key=lambda x: -x[1]):
        print(f"  {direction}: {cnt}")

    # -----------------------------------------------------------------------
    # 4. Clause-type breakdown of unresolved rows
    # -----------------------------------------------------------------------
    section("4. UNRESOLVED ROWS — BY CLAUSE TYPE")

    ct_total   = Counter(soft_rows[rn]["clause_type"] for rn in consensus)
    ct_unres   = Counter(soft_rows[rn]["clause_type"] for rn in unresolved)
    ct_res     = Counter(soft_rows[rn]["clause_type"] for rn in resolved)

    # Sort by unresolved count desc
    print(f"{'Clause Type':<45} {'Total':>6} {'Resolved':>9} {'Unresolved':>11} {'Unres%':>7}")
    print("-"*80)
    for ct, uc in ct_unres.most_common(20):
        tot = ct_total[ct]
        rc  = ct_res[ct]
        print(f"{ct:<45} {tot:>6} {rc:>9} {uc:>11} {pct(uc,tot):>7}")

    # -----------------------------------------------------------------------
    # 5. Disagreement direction BY clause type
    # -----------------------------------------------------------------------
    section("5. DISAGREEMENT DIRECTION BY CLAUSE TYPE (top 10 clause types)")

    ct_directions = defaultdict(Counter)
    for rn, d in unresolved.items():
        ct = soft_rows[rn]["clause_type"]
        g  = d["votes"].get("gemini", "?")
        q  = d["votes"].get("qwen", "?")
        ct_directions[ct][f"Q={q}/G={g}"] += 1

    for ct, dirs in sorted(ct_directions.items(), key=lambda x: -sum(x[1].values()))[:10]:
        total_ct = sum(dirs.values())
        print(f"\n{ct} (unresolved={total_ct}):")
        for direction, cnt in dirs.most_common():
            print(f"    {direction}: {cnt}")

    # -----------------------------------------------------------------------
    # 6. Parties field quality vs agreement rate
    # -----------------------------------------------------------------------
    section("6. PARTIES FIELD QUALITY vs AGREEMENT RATE")

    bucket_total   = Counter()
    bucket_resolved = Counter()

    for rn in consensus:
        contract = soft_rows[rn]["contract"]
        party_text = parties.get(contract, "")
        bucket = parties_length_bucket(party_text)
        bucket_total[bucket] += 1
        if consensus[rn]["resolved"]:
            bucket_resolved[bucket] += 1

    print(f"{'Bucket':<15} {'Total':>7} {'Resolved':>10} {'Resolve%':>10}")
    print("-"*45)
    for bucket in ["very_short", "short", "long", "missing"]:
        t = bucket_total[bucket]
        r = bucket_resolved[bucket]
        print(f"{bucket:<15} {t:>7} {r:>10} {pct(r,t):>10}")

    # -----------------------------------------------------------------------
    # 7. Calibration vs hard-label ground truth (MANUAL_REVIEW rows)
    # -----------------------------------------------------------------------
    section("7. LABELER CALIBRATION vs MANUAL_REVIEW GROUND TRUTH")

    manual_rows = {rn: r for rn, r in master.items()
                   if r["category"] in ("MANUAL_REVIEW", "GEMINI_PRO_REVIEW")
                   and r.get("final_label") in ("LOW", "MEDIUM", "HIGH")}

    # Check which manual rows have v2 labels (they shouldn't — v2 only ran on SOFT_LABEL)
    # Instead compare v1 labeler (qwen_label / gemini_label columns) vs final_label
    for labeler_col, name in [("qwen_label", "Qwen v1"), ("gemini_label", "Gemini v1")]:
        correct = 0
        total_w_label = 0
        errors_up = 0   # predicted too HIGH
        errors_down = 0 # predicted too LOW
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        for rn, r in manual_rows.items():
            pred = r.get(labeler_col, "").strip().upper()
            true = r["final_label"].strip().upper()
            if pred not in ("LOW", "MEDIUM", "HIGH"):
                continue
            total_w_label += 1
            if pred == true:
                correct += 1
            elif order[pred] > order[true]:
                errors_up += 1
            else:
                errors_down += 1

        print(f"\n{name} vs human labels ({total_w_label} rows with both):")
        print(f"  Correct:    {correct} ({pct(correct, total_w_label)})")
        print(f"  Too HIGH:   {errors_up} ({pct(errors_up, total_w_label)})")
        print(f"  Too LOW:    {errors_down} ({pct(errors_down, total_w_label)})")

    # Per-class precision/recall
    for labeler_col, name in [("qwen_label", "Qwen v1"), ("gemini_label", "Gemini v1")]:
        print(f"\n{name} — per-class breakdown:")
        tp = Counter(); fp = Counter(); fn = Counter()
        for rn, r in manual_rows.items():
            pred = r.get(labeler_col, "").strip().upper()
            true = r["final_label"].strip().upper()
            if pred not in ("LOW", "MEDIUM", "HIGH"):
                continue
            if pred == true:
                tp[true] += 1
            else:
                fp[pred] += 1
                fn[true] += 1
        print(f"  {'Class':<8} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Recall':>8}")
        for cls in ["LOW", "MEDIUM", "HIGH"]:
            p = tp[cls] / (tp[cls]+fp[cls]) if (tp[cls]+fp[cls]) else 0
            r_val = tp[cls] / (tp[cls]+fn[cls]) if (tp[cls]+fn[cls]) else 0
            print(f"  {cls:<8} {tp[cls]:>5} {fp[cls]:>5} {fn[cls]:>5} {100*p:>7.1f}% {100*r_val:>7.1f}%")

    # -----------------------------------------------------------------------
    # 8. Which clause types does each labeler systematically over/under-label?
    # -----------------------------------------------------------------------
    section("8. SYSTEMATIC LABELER BIAS BY CLAUSE TYPE (Qwen vs Gemini on unresolved)")

    # For unresolved rows, show clause types where Qwen is consistently higher vs lower
    ct_qwen_higher = Counter()
    ct_gemini_higher = Counter()
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    for rn, d in unresolved.items():
        ct = soft_rows[rn]["clause_type"]
        g  = d["votes"].get("gemini")
        q  = d["votes"].get("qwen")
        if g and q and g in order and q in order:
            if order[q] > order[g]:
                ct_qwen_higher[ct] += 1
            elif order[g] > order[q]:
                ct_gemini_higher[ct] += 1

    print("Clause types where QWEN consistently labels HIGHER (top 10):")
    for ct, cnt in ct_qwen_higher.most_common(10):
        tot = ct_unres[ct]
        print(f"  {ct}: {cnt}/{tot} unresolved rows ({pct(cnt,tot)})")

    print("\nClause types where GEMINI consistently labels HIGHER (top 10):")
    for ct, cnt in ct_gemini_higher.most_common(10):
        tot = ct_unres[ct]
        print(f"  {ct}: {cnt}/{tot} unresolved rows ({pct(cnt,tot)})")

    # -----------------------------------------------------------------------
    # 9. Sample unresolved rows for spot-check
    # -----------------------------------------------------------------------
    section("9. SAMPLE UNRESOLVED ROWS (5 per disagreement direction)")

    from itertools import islice

    shown_dirs = set()
    samples_by_dir = defaultdict(list)

    for rn, d in unresolved.items():
        g = d["votes"].get("gemini", "?")
        q = d["votes"].get("qwen", "?")
        direction = f"Q={q}/G={g}"
        samples_by_dir[direction].append(rn)

    for direction, rns in sorted(samples_by_dir.items(), key=lambda x: -len(x[1])):
        print(f"\n--- {direction} ({len(rns)} rows) — showing 3 ---")
        for rn in list(rns)[:3]:
            row = soft_rows[rn]
            g_data = gemini.get(rn, {})
            q_data = qwen.get(rn, {})
            contract = row["contract"]
            party_text = parties.get(contract, "N/A")[:80]
            clause_preview = row["clause_text"][:200].replace("\n", " ")
            print(f"\n  row_num={rn} | type={row['clause_type']}")
            print(f"  Parties: {party_text}")
            print(f"  Clause:  {clause_preview}...")
            if g_data.get("reason"):
                print(f"  Gemini ({g_data.get('label')}): {g_data['reason'][:150]}")
            if q_data.get("reason"):
                print(f"  Qwen   ({q_data.get('label')}): {q_data['reason'][:150]}")

    # -----------------------------------------------------------------------
    # 10. Full dataset category breakdown
    # -----------------------------------------------------------------------
    section("10. FULL MASTER CSV — CATEGORY BREAKDOWN")

    all_cats = Counter(r["category"] for r in master.values())
    for cat, cnt in all_cats.most_common():
        print(f"  {cat}: {cnt}")

    # What would training set look like after v2 relabel?
    hard_after = sum(1 for r in master.values()
                     if r["category"] in ("AGREED", "MANUAL_REVIEW", "GEMINI_PRO_REVIEW"))
    resolved_would_add = len(resolved)
    still_soft = len(unresolved)
    print(f"\nAfter applying v2 consensus ({len(resolved)} rows):")
    print(f"  Hard labels total: {hard_after + resolved_would_add}")
    print(f"  Still soft:        {still_soft}")
    print(f"  (If Sonnet resolves all {still_soft} → fully hard labeled)")

    # -----------------------------------------------------------------------
    # Save analysis JSON
    # -----------------------------------------------------------------------
    analysis = []
    for rn in sorted(consensus.keys()):
        row = soft_rows[rn]
        d   = consensus[rn]
        g_data = gemini.get(rn, {})
        q_data = qwen.get(rn, {})
        party_text = parties.get(row["contract"], "")
        analysis.append({
            "row_num":         rn,
            "clause_type":     row["clause_type"],
            "contract":        row["contract"],
            "parties":         party_text[:300],
            "parties_bucket":  parties_length_bucket(party_text),
            "clause_text":     row["clause_text"][:500],
            "resolved":        d["resolved"],
            "consensus_label": d["consensus"],
            "method":          d["method"],
            "source":          d["source"],
            "gemini_label":    g_data.get("label"),
            "gemini_party":    g_data.get("signing_party"),
            "gemini_reason":   g_data.get("reason"),
            "qwen_label":      q_data.get("label"),
            "qwen_party":      q_data.get("signing_party"),
            "qwen_reason":     q_data.get("reason"),
            "disagreement_direction": (
                f"Q={q_data.get('label','?')}/G={g_data.get('label','?')}"
                if not d["resolved"] else "agreed"
            ),
        })

    ANALYSIS_OUT.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull analysis saved to: {ANALYSIS_OUT}")
    print(f"({len(analysis)} rows total)")

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    main()
