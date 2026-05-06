# Stage 3 Labeling — Design, Process, and Final State

Consolidated reference covering all labeling decisions and history for the Stage 3 DeBERTa
risk classifier. Supersedes: `STAGE3_LABEL_ANALYSIS.md`, `STAGE3_LABEL_COMPARISON.md`,
`STAGE3_SYNTHETIC_LABELS_DISCUSSION.md`.

Current dataset: `data/processed/training_dataset.json` (4,375 rows).
Backup (pre-Sonnet swap): `data/processed/training_dataset_human_labels_backup.json`.

---

## 1. Design Decisions

### Input scope

- **Total positive spans**: 6,702 (from `data/processed/all_positive_spans.json`)
- **Metadata types excluded** (5 CUAD types — Document Name, Parties, Agreement Date,
  Effective Date, Expiration Date): 2,292 rows. These route to the Stage 4 report header,
  not the risk classifier. Auto-labeling them LOW would add training noise and report clutter.
- **Risk-bearing spans**: 4,410 → sent through labeling pipeline.

### Isolation-primary labeling

Each clause is labeled in isolation (not with sibling clauses). Rationale: the classifier
sees one clause at a time at inference, so training labels must come from the same
distribution. Same clause text in two contracts must get the same label — document-level
context introduces position bias and inconsistency.

### Deduplication

Label once per unique (whitespace-normalized) clause text; fan label to all rows sharing
that text. Saves ~16% API calls (6,702 → 5,603 unique texts; combined with metadata
exclusion: 4,410 → 3,974 unique risk-bearing texts).

### Labeling perspective

Always from the signing party (counterparty to the drafter). Without a perspective anchor,
models silently switch between buyer and seller views and labels become inconsistent.

### Risk scale

`HIGH` — one-sided IP transfer, uncapped liability, no termination rights.
`MEDIUM` — bounded risk: capped liability, mutual restrictions, conditional rights.
`LOW` — standard, balanced, or net-positive for the signing party.

---

## 2. Labeling Runs

### Initial run — Qwen 30B + Gemini 2.5 Flash (2026-04-17)

Both models ran on all 4,410 risk-bearing spans at temperature=0 with:
- Signing-party perspective anchor
- Clause-type definition injected
- Three calibration examples (LOW / MEDIUM / HIGH)

| Labeler | File | HIGH | MED | LOW | ERROR |
|---|---|---|---|---|---|
| Qwen 30B | `synthetic_risk_labels_qwen.json` | 24.2% | 38.8% | 36.7% | 0.2% |
| Gemini Flash | `synthetic_risk_labels_gemini.json` | 22.3% | 34.3% | 43.1% | 0.3% |
| Copilot (reference only) | `data/cuad_risk_labels_copilot.csv` | 14.2% | 41.8% | 44.0% | — |

Copilot labels were generated via chat UI (likely GPT-4) with no perspective anchor and
uncontrolled temperature — kept as reference only, not used in training.

**Qwen vs Gemini agreement: 61.7%** (2,604 / 4,219 matched rows).

Key per-type findings:
- `Uncapped Liability`: 30.6% agreement — Qwen reads text (often finds a cap), Gemini
  anchors on type name. Qwen behavior is more correct.
- `Liquidated Damages`: Qwen 75% HIGH vs Gemini 38% HIGH — Qwen more legally correct.
- `Irrevocable/Perpetual License`: 50% agreement — subjective interpretation of perpetual grants.

**Outcome**: 2,735 AGREED rows (both models matched) + 1,327 SOFT_LABEL rows (adjacent
disagreements: LOW↔MEDIUM or MEDIUM↔HIGH) + 22 ERROR rows dropped.

---

## 3. Label Reconciliation

### GEMINI_PRO_REVIEW — 87 rows

Three clause types with systematic non-flip disagreements sent to Gemini 2.5 Pro:
- Uncapped Liability: 51 rows
- Liquidated Damages: 19 rows
- Irrevocable Or Perpetual License: 17 rows (non-flip subset)

Script: `scripts/run_gemini_pro_review.py`. Result: 87 hard labels.

### MANUAL_REVIEW — 239 rows → 235 used

HIGH↔LOW extreme flips (235 rows after 4 became ERROR) assigned to 4 human reviewers.
Reviewers had access to both models' reasoning, the signing party from METADATA spans,
and `clause_risk_profile` / `typical_risk_for_type` reference columns.

| Reviewer | Rows | Agreement with Sonnet (measured later) |
|---|---|---|
| Anushka | 60 | 40% |
| Rajnish | 59 | 42% |
| Sachin | 58 | 17% |
| Vishal | 58 | 53% |

Reviewers applied different standards — Sachin in particular labeled many clauses LOW
that Sonnet and others rated MEDIUM (see §5).

---

## 4. v2 Soft Label Relabeling (2026-04-28)

The 1,327 SOFT_LABEL (adjacent-disagreement) rows were re-labeled with party metadata
injected into the prompt — models could identify the specific signing party before deciding.

**Groq (Llama 3.3 70B)**: rate-limited; 2,036 of 2,402 attempts returned ERROR. Effectively
unusable. Intermediate `v2_analysis.json` using Groq resolved only 64% (849/1,327).

**Final round — Qwen + Gemini Flash + Claude Sonnet (3-way majority vote)**:

| Source | Rows | How resolved |
|---|---|---|
| `qwen+gemini` agreed | 703 | Direct 2-of-3 majority |
| `qwen+claude` agreed | 282 | Sonnet as tiebreaker |
| `gemini+claude` agreed | 243 | Sonnet as tiebreaker |
| No consensus | 99 | Kept as soft probability vectors |

**Resolution rate: 92.5%** (1,228 / 1,327). The 99 unresolved rows remain as soft-label
probability vectors `[0.5, 0.5, 0.0]` or `[0.0, 0.5, 0.5]` — adjacent disagreements
only (label_gap = 1.0 for all 99).

Output: `data/synthetic/soft_label_relabel_v2_consensus.json`.

---

## 5. Sonnet Label Swap — MANUAL_REVIEW + GEMINI_PRO_REVIEW (2026-05-05)

### Motivation

After training, MEDIUM F1 consistently collapsed across 17 runs. Post-hoc analysis showed:

- Soft-label rows (resolved via v2): OK — Sonnet was already the tiebreaker.
- **MANUAL_REVIEW (235 rows)**: Human reviewers showed severe inconsistency. Sachin had
  only 17% agreement with Sonnet; the four reviewers applied materially different thresholds.
  Human labels: LOW=149 (63%), MEDIUM=48 (20%), HIGH=38 (16%).
  Sonnet labels: LOW=80 (34%), MEDIUM=112 (48%), HIGH=43 (18%).
  Humans systematically under-labeled MEDIUM, especially for borderline IP and liability clauses.

- **GEMINI_PRO_REVIEW (87 rows)**: Gemini Pro and Sonnet have near-identical distributions
  (both: MEDIUM=38, LOW≈31, HIGH≈17) but only 36% row-by-row agreement — disagreements
  on specific clauses in the three most ambiguous types (Uncapped Liability, Liquidated
  Damages, Irrevocable License).

### Action

Sonnet labels from `data/synthetic/relabel_claude.json` (946 rows, covering MANUAL_REVIEW
+ GEMINI_PRO_REVIEW + 624 SOFT_LABEL rows) replaced all 322 MANUAL_REVIEW and
GEMINI_PRO_REVIEW labels. `label_source` updated to `SONNET_REVIEW`.

Coverage: 100% for both categories (no gaps).

Label changes across the 322 rows:

| Change | Count |
|---|---|
| LOW → MEDIUM | 92 |
| MEDIUM → HIGH | 24 |
| HIGH → MEDIUM | 25 |
| MEDIUM → LOW | 29 |
| LOW → HIGH | 19 |
| HIGH → LOW | 12 |
| Unchanged | 121 |

172 rows in the train split changed labels; FAISS index needs rebuild.

---

## 6. Final Dataset State

**File**: `data/processed/training_dataset.json` (4,375 rows)

| label_source | Rows | label_type | How labels were generated |
|---|---|---|---|
| AGREED | 2,726 | hard | Qwen + Gemini Flash agreed on first pass |
| SOFT_LABEL_V2_AGREED | 1,228 | hard | 3-way vote (Qwen + Gemini + Sonnet) with party metadata |
| SONNET_REVIEW | 322 | hard | Sonnet labels replacing inconsistent human/Gemini Pro labels |
| SOFT_LABEL | 99 | soft | No consensus after v2 — probability vectors `[0.5, 0.5, 0]` or `[0, 0.5, 0.5]` |

**Overall label distribution**: LOW 1,805 / MEDIUM 1,587 / HIGH 884 (hard rows only).

**Train split class balance** (3,472 rows, 3,398 hard):
LOW 42.8% / MEDIUM 37.1% / HIGH 20.1% — imbalance ratio 2.13x, handled via class weights.

**Backup**: `data/processed/training_dataset_human_labels_backup.json` (pre-Sonnet swap,
human/Gemini Pro labels intact for MANUAL_REVIEW and GEMINI_PRO_REVIEW rows).

---

## 7. Audit Trail — Raw Label Files

| File | Contents |
|---|---|
| `data/synthetic/synthetic_risk_labels_qwen.json` | Qwen 30B first-pass labels (4,410 rows) |
| `data/synthetic/synthetic_risk_labels_gemini.json` | Gemini Flash first-pass labels (4,410 rows) |
| `data/cuad_risk_labels_copilot.csv` | Copilot reference labels (not used in training) |
| `data/synthetic/soft_label_relabel_v2_qwen.json` | Qwen v2 labels on 1,327 soft rows (party metadata) |
| `data/synthetic/soft_label_relabel_v2_gemini.json` | Gemini v2 labels on 1,327 soft rows |
| `data/synthetic/relabel_claude.json` | Sonnet labels on 946 rows (MANUAL_REVIEW + GEMINI_PRO_REVIEW + 624 soft) |
| `data/synthetic/soft_label_relabel_v2_consensus.json` | Final 3-way consensus for 1,327 soft rows |
| `data/review/master_label_review.csv` | Full 6,702-row review file with all columns and final_label |
