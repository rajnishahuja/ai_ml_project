# CUAD Dataset Insights
> Source: `master_clauses.csv` (ground truth)

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total contracts (documents) | 510 |
| Total clause categories | 41 |
| **Total QA pairs** | **20,910** |
| **Positive clauses (clause present)** | **6,702 (32.1%)** |
| &nbsp;&nbsp;&nbsp;↳ Metadata clauses (5 types) | 2,292 |
| &nbsp;&nbsp;&nbsp;↳ **Risk-relevant clauses (36 types)** | **4,410** |
| **Negative clauses (clause absent)** | **14,208 (67.9%)** |

> Each document is evaluated against all 41 clause types, producing one QA pair per (document, clause type) combination. The 41 types are **QA questions**, not clauses per document.
>
> **Metadata carve-out**: 5 clause types (`Document Name`, `Parties`, `Agreement Date`, `Effective Date`, `Expiration Date`) are structural metadata, not risk-bearing clauses — they route to the Stage 4 report header and are excluded from risk labeling. The remaining **36 clause types → 4,410 positive spans** form our actual risk-classification training pool.

---

## 2. Clauses Per Document

### All positive spans (metadata + risk-relevant)

| Metric | Value |
|---|---|
| Average | 13.14 |
| Maximum | 32 — `ENERGOUSCORP_03_16_2017 Strategic Alliance Agreement` |
| Minimum | 2 — `PelicanDeliversInc_20200211 Development Agreement` |

### Risk-relevant clauses only (excludes 5 metadata types)

| Metric | Value |
|---|---|
| Average | **9.28** |
| Maximum | 27 — `ENERGOUSCORP_03_16_2017 Strategic Alliance Agreement` |
| Minimum | 1 (several contracts) |
| Documents with ≥1 risk-relevant clause | 475 / 510 |

> **Why this matters for Stage 3 design**: A typical contract carries ~9 risk-relevant clauses (not 41). All clauses from a contract fit comfortably in any reasoning model's context window — which is why the `contract_search` tool can be a simple structured lookup rather than hierarchical/vectorless navigation.

---

## 3. Clauses Per Category

| # | Clause Type | Present | Absent | % Present |
|---|---|---|---|---|
| 1 | Document Name | 510 | 0 | 100.0% |
| 2 | Parties | 509 | 1 | 99.8% |
| 3 | Agreement Date | 470 | 40 | 92.2% |
| 4 | Governing Law | 437 | 73 | 85.7% |
| 5 | Expiration Date | 413 | 97 | 81.0% |
| 6 | Effective Date | 390 | 120 | 76.5% |
| 7 | Anti-Assignment | 374 | 136 | 73.3% |
| 8 | Cap On Liability | 275 | 235 | 53.9% |
| 9 | License Grant | 255 | 255 | 50.0% |
| 10 | Audit Rights | 214 | 296 | 42.0% |
| 11 | Termination For Convenience | 183 | 327 | 35.9% |
| 12 | Post-Termination Services | 182 | 328 | 35.7% |
| 13 | Exclusivity | 180 | 330 | 35.3% |
| 14 | Renewal Term | 176 | 334 | 34.5% |
| 15 | Revenue/Profit Sharing | 166 | 344 | 32.5% |
| 16 | Insurance | 166 | 344 | 32.5% |
| 17 | Minimum Commitment | 165 | 345 | 32.4% |
| 18 | Non-Transferable License | 138 | 372 | 27.1% |
| 19 | Ip Ownership Assignment | 124 | 386 | 24.3% |
| 20 | Change Of Control | 121 | 389 | 23.7% |
| 21 | Non-Compete | 119 | 391 | 23.3% |
| 22 | Notice Period To Terminate Renewal | 111 | 399 | 21.8% |
| 23 | Uncapped Liability | 111 | 399 | 21.8% |
| 24 | Covenant Not To Sue | 100 | 410 | 19.6% |
| 25 | Rofr/Rofo/Rofn | 85 | 425 | 16.7% |
| 26 | Volume Restriction | 82 | 428 | 16.1% |
| 27 | Competitive Restriction Exception | 76 | 434 | 14.9% |
| 28 | Warranty Duration | 75 | 435 | 14.7% |
| 29 | Irrevocable Or Perpetual License | 70 | 440 | 13.7% |
| 30 | Liquidated Damages | 61 | 449 | 12.0% |
| 31 | No-Solicit Of Employees | 59 | 451 | 11.6% |
| 32 | Affiliate License-Licensee | 59 | 451 | 11.6% |
| 33 | Joint Ip Ownership | 46 | 464 | 9.0% |
| 34 | Non-Disparagement | 38 | 472 | 7.5% |
| 35 | No-Solicit Of Customers | 34 | 476 | 6.7% |
| 36 | Third Party Beneficiary | 32 | 478 | 6.3% |
| 37 | Most Favored Nation | 28 | 482 | 5.5% |
| 38 | Affiliate License-Licensor | 23 | 487 | 4.5% |
| 39 | Unlimited/All-You-Can-Eat-License | 17 | 493 | 3.3% |
| 40 | Price Restrictions | 15 | 495 | 2.9% |
| 41 | Source Code Escrow | 13 | 497 | 2.5% |

---

## 4. Class Imbalance Analysis

The dataset is **significantly imbalanced** — only **32.1%** of all QA pairs are positive.

### Imbalance Groups

| Group | Clause Types | Positive Rate | Notes |
|---|---|---|---|
| Always present | Document Name, Parties | ~100% | Universal across all contracts |
| High presence | Agreement Date, Governing Law, Expiration Date, Effective Date, Anti-Assignment | 73–92% | Standard boilerplate clauses |
| Moderate presence | Cap On Liability, License Grant, Audit Rights | 42–54% | Common but not universal |
| Low presence | Termination For Convenience → Covenant Not To Sue | 20–36% | Deal-specific clauses |
| Rare | Rofr/Rofo/Rofn → Affiliate License-Licensee | 10–17% | Specialized provisions |
| Very rare | Joint Ip Ownership → Source Code Escrow | 2.5–9% | Niche/exceptional clauses |

### Key Imbalance Statistics

| Metric | Value |
|---|---|
| Most imbalanced clause | Source Code Escrow (2.5% positive) |
| Least imbalanced clause | License Grant (50.0% positive — perfectly balanced) |
| Clauses with < 10% positive rate | 8 out of 41 (20%) |
| Clauses with > 50% positive rate | 8 out of 41 (20%) |
| Overall positive rate | 32.1% |
| Positive : Negative ratio | 1 : 2.12 |

### Implications for Modeling

- **Stratified sampling** is essential — naive random splits will under-represent rare clause types.
- **Weighted loss functions** (e.g., `BCEWithLogitsLoss` with `pos_weight`) are recommended, especially for the bottom 10 clause types.
- **Evaluation metrics** should prioritize **F1 / precision-recall** over accuracy — a model predicting "absent" for everything would achieve 67.9% accuracy but be useless.
- Clauses like `Source Code Escrow` (13 positives) and `Price Restrictions` (15 positives) may need **data augmentation** or few-shot approaches.
