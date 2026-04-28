# Risk Report — synthetic_demo_contract_001

**Generated:** 2026-04-28T03:55:42.239707+00:00

**Overall risk score:** 4.73 / 10

**Total clauses assessed:** 11


## Summary

This contract contains 11 assessed clause(s) with an overall risk score of 4.7/10. 3 high-risk, 4 medium-risk, and 4 low-risk clause(s) were identified.


## 🔴 High-Risk Clauses

### 1. Indemnification  *(page 3)*

- **Clause ID:** `c1`
- **Section:** Section 9.1
- **Why it's risky:** One-sided indemnification with no liability cap and broad scope.
- **Recommendation:** Negotiate mutual indemnification. Cap liability at contract value. Add carve-outs for gross negligence and willful misconduct.

### 2. Uncapped Liability  *(page 7)*

- **Clause ID:** `c3`
- **Section:** Section 11.3
- **Why it's risky:** Uncapped exposure for both IP and confidentiality is asymmetric in vendor's favor.
- **Recommendation:** Convert to a capped liability clause with mutual exclusions for consequential damages. Reserve uncapped exposure only for IP infringement and gross negligence.

### 3. Irrevocable Or Perpetual License  *(page 5)*

- **Clause ID:** `c2`
- **Section:** Section 4.2
- **Why it's risky:** Perpetual license with no revocation rights blocks future flexibility.
- **Recommendation:** Replace with a fixed-term license with renewal terms. Add revocation rights on material breach or insolvency.


## 🟡 Medium-Risk Clauses

### 1. Warranty Duration  *(page 15)*

- **Clause ID:** `c11`
- **Section:** Section 8.1
- **Why it's risky:** 24-month warranty is acceptable but longer than typical 12-month industry norm — minor cost exposure.
- **Recommendation:** Have legal counsel review this clause. Confirm scope and obligations are commercially reasonable.

### 2. Termination For Convenience  *(page 8)*

- **Clause ID:** `c4`
- **Section:** Section 12.1
- **Why it's risky:** 30-day notice is shorter than typical 60-90 day industry standard for multi-year deals.
- **Recommendation:** Confirm the notice period is workable and that payment for in-progress work is preserved.

### 3. Non-Compete  *(page 9)*

- **Clause ID:** `c5`
- **Section:** Section 13.4
- **Why it's risky:** 5-year non-compete is at the upper boundary of enforceability; geographic scope reasonable.
- **Recommendation:** Tighten geographic / temporal scope. Add carve-outs for pre-existing business lines.

### 4. Exclusivity  *(page 10)*

- **Clause ID:** `c6`
- **Section:** Section 5.2
- **Why it's risky:** Exclusive dealing without minimum-volume off-ramps creates downside risk.
- **Recommendation:** Verify exclusivity is bounded by field of use and contains minimum-performance off-ramps.


## 🟢 Low-Risk Clauses

4 clauses were assessed as standard / low risk.


## Models Used

- **extraction:** microsoft/deberta-base
- **risk_classification:** models/stage3_risk_deberta_v3
- **explanation:** (stage3 risk_reason verbatim)
- **report_explanation:** google/flan-t5-base (not loaded in this run)