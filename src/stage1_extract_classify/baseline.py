"""
Stage 1+2: Rule-Based Baseline
================================
spaCy NER + regex patterns for section headers and numbering.
Used as non-ML comparison point against DeBERTa pipeline.

From the plan:
    "Rule-based baseline: spaCy NER + regex patterns for section headers
     and numbering."

Install: pip install spacy && python -m spacy download en_core_web_sm
"""

import re
import json
import logging
import os
import string
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from sklearn.metrics import accuracy_score, f1_score
from constants import CUAD_CLAUSE_TYPES, QUESTION_TO_CLAUSE_TYPE, BASELINE_CONF_THRESHOLD 

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for common clause signals
# ---------------------------------------------------------------------------

# Maps a clause type to a list of regex patterns that signal its presence.
# Patterns match section headers, numbered clauses, or keyword phrases.
CLAUSE_PATTERNS: dict[str, list[str]] = {
    "Indemnification": [
        r"(?i)\b(indemnif(?:ication|y|ies)|hold\s+harmless|defend\s+and\s+indemnify)\b",
        r"(?i)^[\d.]+\s*indemnif",
    ],
    "Termination For Convenience": [
        r"(?i)\b(termination\s+for\s+convenience|terminate\s+(?:this\s+agreement\s+)?(?:at\s+will|without\s+cause))\b",
        r"(?i)^[\d.]+\s*termination",
    ],
    "Governing Law": [
        r"(?i)\b(governing\s+law|choice\s+of\s+law|applicable\s+law|jurisdiction)\b",
        r"(?i)^[\d.]+\s*(governing|jurisdiction)",
    ],
    "Cap On Liability": [
        r"(?i)\b(limitation\s+of\s+liability|cap\s+on\s+(liability|damages)|aggregate\s+liability)\b",
        r"(?i)^[\d.]+\s*limitation",
    ],
    "Uncapped Liability": [
        r"(?i)\b(unlimited\s+liability|no\s+limit\s+on\s+liability|without\s+limitation\s+of\s+liability)\b",
    ],
    "Non-Compete": [
        r"(?i)\b(non[\s-]?compet(?:e|ition)|covenant\s+not\s+to\s+compet)\b",
        r"(?i)^[\d.]+\s*non[\s-]?compet",
    ],
    "Exclusivity": [
        r"(?i)\b(exclusiv(?:e|ity)|sole\s+(?:and\s+exclusive\s+)?(?:supplier|provider|vendor))\b",
    ],
    "License Grant": [
        r"(?i)\b(grant(?:s)?\s+(?:a\s+)?(?:non[\s-]?exclusive\s+)?licen(?:se|ce)|licen(?:se|ce)\s+grant)\b",
        r"(?i)^[\d.]+\s*licen",
    ],
    "Warranty Duration": [
        r"(?i)\b(warrant(?:y|ies|s)\s+(?:period|duration|term)|warrants?\s+for\s+a\s+period)\b",
        r"(?i)^[\d.]+\s*warrant",
    ],
    "Insurance": [
        r"(?i)\b(insurance|insur(?:e|ance)|general\s+liability\s+insurance|maintain\s+(?:insurance|coverage))\b",
        r"(?i)^[\d.]+\s*insurance",
    ],
    "Audit Rights": [
        r"(?i)\b(audit\s+rights?|right\s+to\s+audit|inspection\s+rights?)\b",
    ],
    "Change Of Control": [
        r"(?i)\b(change\s+of\s+control|acquisition|merger\s+or\s+acquisition|change\s+in\s+control)\b",
    ],
    "Anti-Assignment": [
        r"(?i)\b(anti[\s-]?assignment|(?:no|not)\s+assign(?:able)?|may\s+not\s+assign|prohibit\s+assignment)\b",
        r"(?i)^[\d.]+\s*assignment",
    ],
    "Confidentiality": [
        r"(?i)\b(confidential(?:ity)?|non[\s-]?disclosure|proprietary\s+information|trade\s+secret)\b",
        r"(?i)^[\d.]+\s*confidential",
    ],
    "Dispute Resolution": [
        r"(?i)\b(dispute\s+resolution|arbitration|mediation|resolution\s+of\s+disputes)\b",
    ],
    "Force Majeure": [
        r"(?i)\b(force\s+majeure|act\s+of\s+(?:god|nature)|beyond\s+(?:the\s+)?(?:party|parties)'?\s+control)\b",
    ],
    "Liquidated Damages": [
        r"(?i)\b(liquidated\s+damages|pre[\s-]?determined\s+damages|agreed[\s-]?upon\s+damages)\b",
    ],
    "Renewal Term": [
        r"(?i)\b(renewal\s+term|automatic(?:ally)?\s+renew(?:s)?|auto[\s-]?renew(?:al)?)\b",
    ],
    "Notice Period To Terminate Renewal": [
        r"(?i)\b(notice\s+(?:period\s+)?(?:to\s+)?(?:terminate|cancel)\s+renewal|written\s+notice\s+of\s+(?:non[\s-]?)?renewal)\b",
    ],
    "Parties": [
        r"(?i)^(this\s+agreement\s+is\s+(?:entered\s+into\s+)?between|between\s+and\s+among)",
        r"(?i)\b(hereinafter\s+(?:referred\s+to\s+as|called))\b",
    ],
    "Effective Date": [
        r"(?i)\b(effective\s+(?:date|as\s+of)|as\s+of\s+(?:the\s+date|[A-Z][a-z]+\s+\d{1,2}))\b",
    ],
    "Expiration Date": [
        r"(?i)\b(expir(?:ation|e[sd]?)\s+(?:date|on)|term(?:ination)?\s+date|ends?\s+on)\b",
    ],
    "Affiliate License-Licensor": [
        r"(?i)\b(affiliate\s+licen(?:se|ce)|licen(?:se|ce)\s+to\s+affiliates?)\b",
    ],
    "Affiliate License-Licensee": [
        r"(?i)\b(licensee\s+affiliates?|sublicen(?:se|ce)\s+to\s+affiliates?)\b",
    ],
    "Non-Disparagement": [
        r"(?i)\b(non[\s-]?disparagement|disparage|defame|derogatory\s+statements?)\b",
    ],
    "Covenant Not To Sue": [
        r"(?i)\b(covenant\s+not\s+to\s+sue|release\s+of\s+claims?|waiver\s+of\s+claims?)\b",
    ],
    "Third Party Beneficiary": [
        r"(?i)\b(third[\s-]?party\s+beneficiar(?:y|ies)|no\s+third[\s-]?party\s+rights?)\b",
    ],
    "Minimum Commitment": [
        r"(?i)\b(minimum\s+(?:purchase|order|commitment|volume)|purchase\s+obligation)\b",
    ],
    "Volume Restriction": [
        r"(?i)\b(volume\s+(?:restriction|limit|cap)|maximum\s+(?:volume|quantity|units?))\b",
    ],
    "IP Ownership Assignment": [
        r"(?i)\b(work[\s-]?for[\s-]?hire|assign(?:s|ment)\s+of\s+(?:all\s+)?(?:IP|intellectual\s+property))\b",
    ],
    "Joint IP Ownership": [
        r"(?i)\b(joint(?:ly)?\s+own(?:ed)?|co[\s-]?own(?:ership)?|jointly\s+developed)\b",
    ],
    "Price Restrictions": [
        r"(?i)\b(price\s+(?:restriction|floor|ceiling|control)|most\s+favou?red\s+(?:nation|customer))\b",
    ],
    "Rofr/Rofo/Rofn": [
        r"(?i)\b(right\s+of\s+first\s+(refusal|offer|negotiation)|ROFR|ROFO|ROFN)\b",
    ],
    "Source Code Escrow": [
        r"(?i)\b(source\s+code\s+escrow|escrow\s+agent|escrow\s+agreement)\b",
    ],
    "Post-Agreement Restrictions": [
        r"(?i)\b(post[\s-]?(?:termination|agreement|contract)\s+(?:restriction|obligation)|surviving\s+(?:clause|obligation|provision))\b",
    ],
    "Unlimited/All-You-Can-Eat-License": [
        r"(?i)\b(unlimited\s+(?:licen(?:se|ce)|use|access)|all[\s-]?you[\s-]?can[\s-]?(?:eat|use)|enterprise[\s-]?wide\s+licen(?:se|ce))\b",
    ],
    "Irrevocable Or Perpetual License": [
        r"(?i)\b(irrevocable\s+licen(?:se|ce)|perpetual\s+licen(?:se|ce)|non[\s-]?terminable\s+licen(?:se|ce))\b",
    ],
    "Revenue/Profit Sharing": [
        r"(?i)\b(revenue\s+shar(?:ing|e)|profit\s+shar(?:ing|e)|royalt(?:y|ies)|revenue\s+split)\b",
    ],
    "Sublicense": [
        r"(?i)\b(sublicen(?:se|ce|sing)|right\s+to\s+sublicen(?:se|ce))\b",
    ],
    "Non-Transferable License": [
        r"(?i)\b(non[\s-]?transferable|not\s+transferable|may\s+not\s+transfer\s+(?:this\s+)?licen(?:se|ce))\b",
    ],
}


# ---------------------------------------------------------------------------
# Section-header detection
# ---------------------------------------------------------------------------

SECTION_HEADER_RE = re.compile(
    r"""
    (?mx)                          # multiline + verbose
    ^                              # start of line
    (?:
      (?:\d+\.)+\d*\s+             # 1., 1.1, 1.1.1 numbering
    | [A-Z]{1,3}\.\s+              # A., B. lettered
    | (?:ARTICLE|SECTION)\s+[IVX\d]+[.:]\s*  # ARTICLE I: / SECTION 2.
    )
    .{3,80}                        # header text (3–80 chars)
    $
    """,
    re.VERBOSE | re.MULTILINE,
)


def detect_section_headers(text: str) -> list[dict]:
    """Find all section headers and their positions."""
    headers = []
    for match in SECTION_HEADER_RE.finditer(text):
        headers.append({
            "text": match.group().strip(),
            "start": match.start(),
            "end": match.end(),
        })
    return headers


# ---------------------------------------------------------------------------
# Paragraph / section splitter
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> list[dict]:
    """
    Split a contract into sections using detected headers as boundaries.
    Each section: { header, text, start, end }
    """
    headers = detect_section_headers(text)
    if not headers:
        return [{"header": "", "text": text, "start": 0, "end": len(text)}]

    sections = []
    for i, h in enumerate(headers):
        section_start = h["start"]
        section_end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)
        sections.append({
            "header": h["text"],
            "text": text[section_start:section_end].strip(),
            "start": section_start,
            "end": section_end,
        })
    return sections


# ---------------------------------------------------------------------------
# Baseline extractor
# ---------------------------------------------------------------------------

@dataclass
class BaselineClause:
    clause_id: str
    clause_text: str
    clause_type: str
    start_pos: int
    end_pos: int
    confidence: float  # fixed heuristic score (0.90 for header match, 0.70 for keyword)
    matched_pattern: str
    document_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class RuleBasedExtractor:
    """
    spaCy NER + regex rule-based clause extractor.
    Serves as the non-ML baseline for Stage 1+2.

    Strategy:
      1. Regex pattern matching on each detected section.
      2. spaCy NER to find named entities (parties, dates, orgs) that
         correlate with specific clause types.
      3. Each matched section gets assigned the highest-confidence
         clause type based on pattern hits.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            import spacy
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except (ImportError, OSError):
            logger.warning("spaCy not available — NER features disabled. "
                           "Install: pip install spacy && python -m spacy download en_core_web_sm")
            self.nlp = None

        self.spacy_available = self.nlp is not None
        self.patterns = CLAUSE_PATTERNS

    def extract(self, contract_text: str, doc_id: str = "baseline_doc") -> list[BaselineClause]:
        """
        Run rule-based extraction on a contract string.
        Returns list of BaselineClause objects (same interface as ClauseObject).
        """
        sections = split_into_sections(contract_text)
        clauses = []

        for section in sections:
            section_text = section["text"]
            section_start = section["start"]

            # Store: clause_type -> (confidence, pattern, match)
            type_scores: dict[str, tuple[float, str, object]] = {}

            for clause_type, pattern_list in self.patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, section_text)
                    if match:
                        confidence = 0.85 if re.search(pattern, section["header"]) else 0.60

                        if (
                            clause_type not in type_scores
                            or type_scores[clause_type][0] < confidence
                        ):
                            # ✅ store match object directly
                            type_scores[clause_type] = (confidence, pattern, match)

            # spaCy boost
            if self.spacy_available and section_text:
                doc = self.nlp(section_text[:5000])
                type_scores = self._apply_ner_boost(doc, type_scores)

            if not type_scores:
                continue

            # Pick best type
            best_type = max(type_scores, key=lambda ct: type_scores[ct][0])
            best_confidence, best_pattern, best_match = type_scores[best_type]

            # ✅ Use stored match instead of re-searching
            if best_match:
                span_start = section_text.rfind("\n", 0, best_match.start())
                span_start = 0 if span_start == -1 else span_start + 1

                span_end = section_text.find("\n", best_match.end())
                span_end = len(section_text) if span_end == -1 else span_end

                clause_text = section_text[span_start:span_end].strip()
                abs_start = section_start + span_start
                abs_end = section_start + span_end
            else:
                # spaCy-only case (no regex match)
                clause_text = section_text[:500]
                abs_start = section_start
                abs_end = section_start + min(500, len(section_text))

            clause_id = f"{doc_id}_{best_type.replace(' ', '_')}_{len(clauses):04d}"

            clauses.append(BaselineClause(
                clause_id=clause_id,
                clause_text=clause_text,
                clause_type=best_type,
                start_pos=abs_start,
                end_pos=abs_end,
                confidence=best_confidence,
                matched_pattern=best_pattern,
                document_id=doc_id,
            ))

        clauses.sort(key=lambda c: c.start_pos)
        logger.info(f"[Baseline] Extracted {len(clauses)} clauses from {doc_id}")
        return clauses

    def _apply_ner_boost(self, doc, type_scores: dict) -> dict:
        """
        Use spaCy NER to boost confidence only when corroborating
        keyword evidence already exists. NER confirms, not initiates.
        """
        entities = {ent.label_ for ent in doc.ents}
        text_lower = doc.text.lower()

        # ---------------------------
        # ORG → Parties
        # ---------------------------
        if "ORG" in entities and "Parties" in type_scores:
            if any(kw in text_lower for kw in ("between", "hereinafter", "party")):
                old_conf, pat, mat = type_scores["Parties"]
                type_scores["Parties"] = (
                    min(old_conf + 0.1, 1.0),
                    f"{pat}+spaCy:ORG",
                    mat,
                )

        # ---------------------------
        # DATE → multiple clause types
        # ---------------------------
        if "DATE" in entities:

            if "Effective Date" in type_scores:
                if any(kw in text_lower for kw in ("effective", "as of", "commencing")):
                    old_conf, pat, mat = type_scores["Effective Date"]
                    type_scores["Effective Date"] = (
                        min(old_conf + 0.1, 1.0),
                        f"{pat}+spaCy:DATE",
                        mat,
                    )

            if "Expiration Date" in type_scores:
                if any(kw in text_lower for kw in ("expir", "terminat", "ends on", "term ends")):
                    old_conf, pat, mat = type_scores["Expiration Date"]
                    type_scores["Expiration Date"] = (
                        min(old_conf + 0.1, 1.0),
                        f"{pat}+spaCy:DATE",
                        mat,
                    )

            if "Warranty Duration" in type_scores:
                if any(kw in text_lower for kw in ("warrant", "guarantee", "defect")):
                    old_conf, pat, mat = type_scores["Warranty Duration"]
                    type_scores["Warranty Duration"] = (
                        min(old_conf + 0.1, 1.0),
                        f"{pat}+spaCy:DATE",
                        mat,
                    )

        # ---------------------------
        # MONEY → financial clauses
        # ---------------------------
        if "MONEY" in entities:

            if "Cap On Liability" in type_scores:
                if any(kw in text_lower for kw in ("limitation", "cap", "aggregate", "not exceed")):
                    old_conf, pat, mat = type_scores["Cap On Liability"]
                    type_scores["Cap On Liability"] = (
                        min(old_conf + 0.1, 1.0),
                        f"{pat}+spaCy:MONEY",
                        mat,
                    )

            if "Liquidated Damages" in type_scores:
                if any(kw in text_lower for kw in ("liquidated", "predetermined", "agreed damages")):
                    old_conf, pat, mat = type_scores["Liquidated Damages"]
                    type_scores["Liquidated Damages"] = (
                        min(old_conf + 0.1, 1.0),
                        f"{pat}+spaCy:MONEY",
                        mat,
                    )

        return type_scores

def _infer_clause_type_from_question(question: str) -> str:
    """
    Infer clause type from a CUAD question string.
    Uses exact reverse lookup against known question templates first,
    falls back to substring matching only if no exact match found.
    """
    # Exact match — robust against substring ambiguity
    if question in QUESTION_TO_CLAUSE_TYPE:
        return QUESTION_TO_CLAUSE_TYPE[question]

    # Fallback — substring match for custom or slightly modified questions
    question_lower = question.lower()
    for ct in CUAD_CLAUSE_TYPES:
        if ct.lower() in question_lower:
            return ct

    return "Unknown"


def _normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _squad_em_f1(prediction: str, ground_truths: list[str]) -> tuple[float, float]:
    """Compute SQuAD-style EM and token-level F1 for a single example."""
    if not ground_truths:
        return (1.0, 1.0) if not prediction else (0.0, 0.0)

    pred_norm = _normalize_answer(prediction)
    best_em, best_f1 = 0.0, 0.0

    for truth in ground_truths:
        truth_norm = _normalize_answer(truth)

        # Exact match
        em = float(pred_norm == truth_norm)
        best_em = max(best_em, em)

        # Token F1 — Counter preserves duplicate tokens unlike set
        pred_tokens = pred_norm.split()
        truth_tokens = truth_norm.split()
        common = sum((Counter(pred_tokens) & Counter(truth_tokens)).values())
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_em, best_f1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1+2: Rule-Based Baseline")
    parser.add_argument("--contract_file", required=True, help="Path to .txt/.pdf/.docx contract")
    parser.add_argument("--output_file", default="baseline_clauses.json")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    import sys; sys.path.insert(0, ".")
    from pipeline import preprocess_contract

    contract_text = preprocess_contract(args.contract_file)
    extractor = RuleBasedExtractor(spacy_model=args.spacy_model)
    clauses = extractor.extract(contract_text, doc_id=Path(args.contract_file).stem)

    output = [c.to_dict() for c in clauses]
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(output)} clauses to {args.output_file}")
