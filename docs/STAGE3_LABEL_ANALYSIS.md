# Stage 3 Synthetic Label Analysis

## Labeling Runs Compared

| Run | Model | File | Temperature | Notes |
|---|---|---|---|---|
| Copilot | Unknown (likely GPT-4 via chat UI) | `data/cuad_risk_labels_copilot.csv` | Uncontrolled | Chat interface — no reproducibility |
| Qwen/30B | mavenir-generic1-30b-q4_k_xl (GPU) | `data/synthetic/synthetic_risk_labels_qwen.json` | 0 (deterministic) | Local llama-server port 10012 |
| Gemini | TBD | TBD | TBD | Planned |

---

## Qwen/30B Run — Full Dataset Results (2026-04-17)

### Setup
- Script: `scripts/generate_synthetic_labels.py`
- Input: `data/processed/all_positive_spans.json` (6,702 spans)
- Metadata types skipped: 2,292 (Document Name, Parties, Agreement Date, Effective Date, Expiration Date)
- Risk-bearing spans: 4,410
- Unique texts after dedup: 3,974 API calls (436 duplicates fanned)
- Runtime: ~3 hrs (KV prefix caching reduced from estimated 6.7 hrs)

### Overall Distribution

| Label | Count | % | Copilot % |
|---|---|---|---|
| HIGH | 1,069 | 24.2% | 14.2% |
| MEDIUM | 1,713 | 38.8% | 41.8% |
| LOW | 1,620 | 36.7% | 44.0% |
| ERROR | 8 | 0.2% | — |

**Key observation**: Our model is systematically ~10 points more aggressive on HIGH vs copilot. The signing-party perspective anchor in the prompt drives this. Copilot labels were likely generated via chat UI with no system prompt or perspective anchor, at uncontrolled temperature — not a reliable baseline.

### Confidence Stats

- Average: 0.817 | Min: 0.00 | Max: 1.00
- `conf=0.0`: 515 rows (11.7%) — **model quirk, not real uncertainty**
  - Sample conf=0.0 rows have clear, well-argued reasoning
  - The 30B model defaults to 0.0 when it can't produce a meaningful confidence score
  - These labels are valid and should be kept
- `conf 0.01–0.69`: 27 rows — genuinely uncertain
- `conf >= 0.70`: 3,868 rows (87.7%)

### Per Clause Type Breakdown

| Clause Type | n | HIGH | MED | LOW | Avg Conf | Notes |
|---|---|---|---|---|---|---|
| Affiliate License-Licensee | 59 | 12 | 21 | 26 | 0.59 | Low conf — complex IP |
| Affiliate License-Licensor | 23 | 3 | 4 | 15 | 0.74 | |
| Anti-Assignment | 374 | 112 | 189 | 73 | 0.86 | |
| Audit Rights | 214 | 29 | 100 | 85 | 0.75 | |
| Cap On Liability | 275 | 132 | 93 | 50 | 0.89 | 48% HIGH — caps set too low |
| Change Of Control | 121 | 50 | 44 | 25 | 0.83 | |
| Competitive Restriction Exception | 76 | 15 | 19 | 42 | 0.84 | |
| Covenant Not To Sue | 100 | 44 | 32 | 24 | 0.92 | |
| Exclusivity | 180 | 42 | 85 | 52 | 0.76 | |
| Governing Law | 437 | 5 | 87 | 345 | 0.88 | 79% LOW — correct |
| Insurance | 166 | 8 | 88 | 69 | 0.75 | |
| Ip Ownership Assignment | 124 | 79 | 26 | 19 | 0.91 | 64% HIGH — correct |
| Irrevocable Or Perpetual License | 70 | 33 | 23 | 14 | 0.68 | Low conf |
| Joint Ip Ownership | 46 | 9 | 31 | 6 | 0.83 | |
| License Grant | 255 | 35 | 112 | 108 | 0.67 | 27% conf=0.0 |
| Liquidated Damages | 61 | 46 | 5 | 10 | 0.90 | 75% HIGH — correct |
| Minimum Commitment | 165 | 78 | 52 | 34 | 0.82 | |
| Most Favored Nation | 28 | 6 | 9 | 13 | 0.86 | |
| No-Solicit Of Customers | 34 | 8 | 17 | 9 | 0.77 | |
| No-Solicit Of Employees | 59 | 10 | 23 | 26 | 0.92 | |
| Non-Compete | 119 | 39 | 48 | 31 | 0.87 | |
| Non-Disparagement | 38 | 16 | 15 | 6 | 0.80 | |
| Non-Transferable License | 138 | 15 | 88 | 35 | 0.53 | 42% conf=0.0 — most uncertain |
| Notice Period To Terminate Renewal | 111 | 9 | 56 | 46 | 0.87 | |
| Post-Termination Services | 182 | 21 | 79 | 82 | 0.84 | |
| Price Restrictions | 15 | 3 | 11 | 1 | 0.89 | Small n |
| Renewal Term | 176 | 34 | 95 | 47 | 0.86 | |
| Revenue/Profit Sharing | 166 | 29 | 69 | 68 | 0.85 | |
| Rofr/Rofo/Rofn | 85 | 9 | 43 | 33 | 0.73 | |
| Source Code Escrow | 13 | 5 | 5 | 3 | 0.63 | Small n |
| Termination For Convenience | 183 | 79 | 44 | 60 | 0.90 | 43% HIGH — correct |
| Third Party Beneficiary | 32 | 6 | 13 | 13 | 0.90 | |
| Uncapped Liability | 111 | 10 | 22 | 79 | 0.87 | 71% LOW — reads text, not type name |
| Unlimited/All-You-Can-Eat-License | 17 | 4 | 5 | 8 | 0.66 | Small n |
| Volume Restriction | 82 | 10 | 23 | 49 | 0.75 | |
| Warranty Duration | 75 | 24 | 37 | 14 | 0.77 | |

### Strong Label Patterns (High Confidence, Intuitive)

| Clause Type | Dominant | % | Verdict |
|---|---|---|---|
| Governing Law | LOW 79% | conf 0.88 | Correct — standard jurisdictions |
| Liquidated Damages | HIGH 75% | conf 0.90 | Correct — always punitive |
| IP Ownership Assignment | HIGH 64% | conf 0.91 | Correct — giving away IP |
| Termination For Convenience | HIGH 43% | conf 0.90 | Correct — unilateral exit right |
| Uncapped Liability | LOW 71% | conf 0.87 | Correct — model reads text not type name |
| Covenant Not To Sue | HIGH 44% | conf 0.92 | Correct |

### Copilot Comparison

- Matched rows: 4,321 (join key: contract × clause_type, case-insensitive, .pdf stripped)
- Agreement: **1,902 (44.0%)**
- Disagreement: 2,419 (56.0%)

| Our Label | Copilot | Count | Note |
|---|---|---|---|
| Low | Medium | 777 | We more lenient on standard clauses |
| High | Medium | 518 | We more aggressive on risky clauses |
| Medium | Low | 332 | Copilot more lenient on mid-risk |
| **Low** | **High** | **298** | **Highest priority to audit** |
| Medium | High | 293 | |
| High | Low | 193 | |

**298 LOW→High cases** are the highest-stakes mismatches — clauses we called safe that copilot flagged HIGH. Audit these first.

**Important caveat on copilot labels**: Generated via chat UI (likely GPT-4) with no system prompt, no perspective anchor, uncontrolled temperature. Not a reproducible or calibrated baseline. Agreement percentage should be treated as a rough sanity check, not ground truth.

---

## Prompt Engineering Decisions (v1)

- Perspective anchor: signing party (counterparty to drafter)
- 3 calibration examples (LOW/MEDIUM/HIGH)
- Self-check instruction added after initial pilot showed contradictory reasoning
- `temperature=0` added after pilot showed 44% label instability across runs with default sampling
- Metadata types routed away (Option B) — not risk-labeled

---

## Known Issues

1. **conf=0.0 quirk**: 11.7% of rows. Model limitation, not data issue. Labels are valid.
2. **Redacted `[ * ]` tokens**: 17 spans (0.25%) have redacted values. Model sometimes misreads as "missing term." Negligible at scale.
3. **Our HIGH rate (24%) vs copilot (14%)**: Systematic difference — prompt perspective anchor + copilot baseline unreliability. Gemini comparison will clarify.

---

## Next Steps

1. Run Gemini labeling on same `all_positive_spans.json`
2. Three-way comparison: 30B vs Gemini vs Copilot
3. Audit 298 LOW→High disagreements (our vs copilot)
4. Decide label strategy: majority vote, trust one labeler, or human review of disagreements
