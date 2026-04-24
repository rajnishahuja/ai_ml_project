# Stage 3 Label Comparison — Qwen vs Gemini vs Copilot

**Date**: 2026-04-17

## Models Compared

| Labeler | Model | Temp | File |
|---|---|---|---|
| Qwen/30B | mavenir-generic1-30b-q4_k_xl (A100 GPU, llama-server) | 0 | `data/synthetic/synthetic_risk_labels_qwen.json` |
| Gemini | gemini-2.5-flash (Google API, thinking off, JSON mode) | 0 | `data/synthetic/synthetic_risk_labels_gemini.json` |
| Copilot | Unknown — likely GPT-4 via chat UI, uncontrolled temp | — | `data/cuad_risk_labels_copilot.csv` |

---

## Overall Distribution

| Label | Qwen | Gemini | Copilot |
|---|---|---|---|
| HIGH | 24.2% (1,069) | 22.3% (984) | 14.2% (953) |
| MEDIUM | 38.8% (1,713) | 34.3% (1,513) | 41.8% (2,803) |
| LOW | 36.7% (1,620) | 43.1% (1,899) | 44.0% (2,946) |
| ERROR | 0.2% (8) | 0.3% (14) | — |
| **Total** | **4,410** | **4,410** | **6,702** |

- Gemini is most lenient (most LOW), closest to copilot distribution
- Qwen is most aggressive on HIGH (10 points above copilot)
- Both LLMs agree that copilot's 14.2% HIGH is an undercount — copilot baseline is unreliable (chat UI, no perspective anchor)

---

## Confidence

| | Avg | Min | Max | conf=0.0 |
|---|---|---|---|---|
| Qwen | 0.817 | 0.00 | 1.00 | 507 (11.5%) |
| Gemini | **0.901** | 0.00 | 1.00 | **148 (3.4%)** |

Gemini confidence scoring is more reliable — native JSON mode reduces conf=0.0 artifacts. Use Gemini confidence as the primary trust signal in merged dataset.

---

## Qwen vs Gemini Agreement

- Matched rows: 4,219
- **Agreement: 61.7%** (2,604 rows)
- Disagreement: 38.3% (1,615 rows)

### Disagreement Breakdown (Qwen → Gemini)

| Qwen | Gemini | Count | Pattern |
|---|---|---|---|
| MEDIUM | LOW | 546 | Qwen more conservative on mid-risk |
| LOW | MEDIUM | 334 | Gemini upgrades borderline clauses |
| MEDIUM | HIGH | 240 | Gemini more aggressive on upper end |
| HIGH | MEDIUM | 239 | Qwen more aggressive on upper end |
| HIGH | LOW | 165 | Extreme flip — needs review |
| LOW | HIGH | 70 | Extreme flip — needs review |

Main disagreement is at the **LOW/MEDIUM boundary** (880 cases). Direct HIGH↔LOW flips are rare (235 cases) — the extremes are mostly stable.

---

## Per-Type Agreement

| Clause Type | n | Agreement | Qwen HIGH | Gemini HIGH | Notes |
|---|---|---|---|---|---|
| Uncapped Liability | 108 | **30.6%** | 10 | 42 | Biggest gap — Qwen reads text (finds cap); Gemini anchors on type name |
| Source Code Escrow | 13 | 30.8% | 5 | 1 | Small sample |
| Price Restrictions | 15 | 33.3% | 3 | 1 | Small sample |
| Non-Disparagement | 37 | 43.2% | 16 | 9 | Qwen more aggressive |
| Irrevocable/Perpetual License | 68 | 50.0% | 32 | 14 | Qwen sees perpetual grants as high risk |
| Liquidated Damages | 61 | 59.0% | 46 | 23 | Qwen 75% HIGH vs Gemini 38% — Qwen more legally correct |
| Termination For Convenience | 179 | **75.4%** | 79 | 77 | Strong agreement — correct HIGH |
| Affiliate License-Licensor | 23 | 82.6% | 3 | 2 | Highest agreement |
| Anti-Assignment | 351 | 68.4% | 110 | 109 | Strong agreement |
| IP Ownership Assignment | 122 | 67.2% | 78 | 76 | Strong agreement — correct HIGH |
| Volume Restriction | 82 | 70.7% | 10 | 10 | Strong agreement — correct LOW |

---

## Key Findings

### 1. `Uncapped Liability` — Critical Discrepancy (30.6% agreement)
Qwen labels 71% LOW (reads text — often finds a liability cap described), Gemini labels 39% HIGH (possibly anchoring on type name "uncapped"). **Human review needed** — the text-reading behavior (Qwen) is likely correct.

### 2. `Liquidated Damages` — Qwen More Legally Correct
Qwen 75% HIGH vs Gemini 38% HIGH. Liquidated damages clauses are almost always punitive for the signing party. Qwen's aggressive HIGH labeling aligns with legal intuition.

### 3. `Irrevocable/Perpetual License` — Subjective Interpretation
32 HIGH (Qwen) vs 14 HIGH (Gemini) out of 68. Perpetual grants can be favorable (signing party gets permanent rights) or risky (locked into unfavorable terms forever). Needs human tiebreak.

### 4. Confidence Artifact in Qwen
507 rows (11.5%) with conf=0.0 despite clear reasoning — model limitation, not genuine uncertainty. Labels are valid. Use Gemini confidence as primary signal.

---

## Recommendation for Merged Labels

| Scenario | Action |
|---|---|
| Qwen == Gemini | Use agreed label — high confidence |
| Qwen ≠ Gemini, copilot available | Majority vote across all three |
| Qwen ≠ Gemini, no copilot | Flag for human review (priority: HIGH↔LOW flips = 235 rows) |
| conf=0.0 on both | Flag for human review regardless of label |

Priority audit list:
1. **235 HIGH↔LOW direct flips** between Qwen and Gemini
2. **All `Uncapped Liability` rows** (30.6% agreement, systematic interpretation gap)
3. **Liquidated Damages** disagreements (59% agreement)

---

## Label Reconciliation Strategy (decided 2026-04-18)

### Master Review File
`data/review/master_label_review.csv` — all 6,702 spans in one file with columns:
`row_num, review_id, id, contract, clause_type, clause_text, category,
disagreement_type, disagreement_direction, label_gap, clause_risk_profile,
typical_risk_for_type, qwen_label, gemini_label, copilot_label,
qwen_confidence, gemini_confidence, qwen_reason, gemini_reason,
final_label, reviewer, notes`

### Row Categories & How final_label Gets Filled

| Category | Rows | How Resolved | row_num range |
|---|---|---|---|
| METADATA | 2,292 | Pre-filled "METADATA" — routes to report header, not trained | 1–2292 |
| AGREED | 2,735 | Pre-filled — both Qwen & Gemini agree | 2293–5027 |
| MANUAL_REVIEW | 239 | Human fills `final_label` — HIGH↔LOW extreme flips | 5028–5266 |
| GEMINI_PRO_REVIEW | 87 | Gemini 2.5 Pro fills `final_label` — 3 focus type disagreements | 5267–5353 |
| SOFT_LABEL | 1,327 | Soft probability computed from Qwen+Gemini outputs, no hard label needed | 5354–6680 |
| ERROR | 22 | Dropped from training | 6681–6702 |

### SOFT_LABEL Approach
Adjacent disagreements (LOW↔MEDIUM or MEDIUM↔HIGH) are encoded as probability distributions
for DeBERTa training via KLDivLoss instead of CrossEntropyLoss:
```
Qwen=HIGH, Gemini=MEDIUM → [LOW=0.0, MEDIUM=0.5, HIGH=0.5]
Qwen=LOW,  Gemini=MEDIUM → [LOW=0.5, MEDIUM=0.5, HIGH=0.0]
```
Confidence-weighted variant available using `qwen_confidence` / `gemini_confidence` columns.

### Manual Review Instructions for Colleagues

**Repo**: https://github.com/rajnishahuja/ai_ml_project  
**File to edit**: `data/review/master_label_review.csv`  
**Your rows**: filter column `reviewer` by your name — all your rows are in the `MANUAL_REVIEW` category (row_num 5028–5266)

**What to do**:
1. Clone/pull the repo
2. Open `data/review/master_label_review.csv` in Excel or any CSV editor
3. Filter `reviewer` = your name
4. For each row, read: `clause_text`, `clause_type`, `clause_risk_profile`, `typical_risk_for_type`, `qwen_reason`, `gemini_reason`
5. Fill in `final_label` — must be exactly: `LOW`, `MEDIUM`, or `HIGH`
6. Optionally add a short note in `notes` if your reasoning differs from both models
7. Save, commit, and push directly to main

**Key principle**: assess risk **from the perspective of the party signing the contract** (the non-drafting party — typically the customer, licensee, or distributor). A clause that grants broad rights to the signing party is LOW risk; one that imposes uncapped obligations or strips their rights is HIGH.

**Helpful columns**:
- `clause_risk_profile` — domain knowledge about what makes this clause type risky
- `typical_risk_for_type` — the most common agreed label across all contracts for this type
- `qwen_reason` / `gemini_reason` — both models' reasoning; use as input, not as the answer
- `disagreement_direction` — which model said higher risk (Qwen↑ or Gemini↑)
- `copilot_label` — a third reference label (treat as additional signal, not ground truth)

**Assignment (239 rows, ~60 per person, whole clause types per reviewer):**
- **Anushka** (60 rows): Cap On Liability, Audit Rights, Covenant Not To Sue, Termination For Convenience, No-Solicit Of Customers, Warranty Duration, Non-Compete
- **Rajnish** (60 rows): Uncapped Liability, IP Ownership Assignment, License Grant, Volume Restriction, Non-Transferable License, Affiliate License-Licensee, Price Restrictions, ROFR
- **Sachin** (60 rows): Minimum Commitment, Exclusivity, Post-Termination Services, Competitive Restriction Exception, Revenue/Profit Sharing, Source Code Escrow, Insurance, Governing Law
- **Vishal** (59 rows): Irrevocable Or Perpetual License, Anti-Assignment, Renewal Term, Liquidated Damages, Non-Disparagement, Most Favored Nation, Change Of Control, Notice Period

**Git push** (no PR required, push directly to main):
```bash
git pull origin main
# edit the CSV
git add data/review/master_label_review.csv
git commit -m "Review: <your name> fills final_label for <clause types>"
git push origin main
```
If you hit a merge conflict on the CSV, ping Rajnish — he'll merge manually.

### Gemini 2.5 Pro Review (87 rows) — COMPLETE
Focus types with non-flip disagreements (MEDIUM↔HIGH or LOW↔MEDIUM):
- Uncapped Liability: 50 rows
- Liquidated Damages: 19 rows
- Irrevocable Or Perpetual License: 17 rows (non-flip subset, flips go to Vishal)
Script: `scripts/run_gemini_pro_review.py`

### Merging Back
Join `final_label` from review file into training dataset on `row_num`.
SOFT_LABEL rows: compute soft vectors programmatically from `qwen_label`, `gemini_label`,
`qwen_confidence`, `gemini_confidence` at training time.

## Next Steps

1. Colleagues complete MANUAL_REVIEW rows (filter by `reviewer` column in master CSV)
2. ~~Run Gemini 2.5 Pro on 87 GEMINI_PRO_REVIEW rows~~ — DONE (`scripts/run_gemini_pro_review.py`)
3. Merge final_label back → build training dataset with hard + soft labels
4. Train DeBERTa risk classifier on merged labels
