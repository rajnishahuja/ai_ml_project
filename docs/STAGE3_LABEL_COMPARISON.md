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

## Next Steps

1. Build three-way merged label file with majority vote logic
2. Audit 235 HIGH↔LOW flips manually
3. Review `Uncapped Liability` subset — determine which model is correct
4. Use merged labels to train DeBERTa risk classifier (Stage 3)
