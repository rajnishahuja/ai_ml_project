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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from constants import CUAD_CLAUSE_TYPES, _make_clause_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns — keyed by exact CUAD clause type names
# ---------------------------------------------------------------------------

CLAUSE_PATTERNS: dict[str, list[str]] = {
    "Document Name": [
        r"(?i)\b(agreement|contract|amendment|addendum|schedule|exhibit)\b",
        r"(?i)^[\d.]*\s*(this\s+(?:agreement|contract))",
    ],
    "Parties": [
        r"(?i)^(this\s+agreement\s+is\s+(?:entered\s+into\s+)?between|between\s+and\s+among)",
        r"(?i)\b(hereinafter\s+(?:referred\s+to\s+as|called))\b",
    ],
    "Agreement Date": [
        r"(?i)\b(dated?\s+(?:as\s+of\s+)?(?:this\s+)?[\d]+(?:st|nd|rd|th)?\s+day\s+of|entered\s+into\s+as\s+of)\b",
        r"(?i)\b(as\s+of\s+[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b",
    ],
    "Effective Date": [
        r"(?i)\b(effective\s+(?:date|as\s+of)|as\s+of\s+(?:the\s+date|[A-Z][a-z]+\s+\d{1,2}))\b",
    ],
    "Expiration Date": [
        r"(?i)\b(expir(?:ation|e[sd]?)\s+(?:date|on)|term(?:ination)?\s+date|ends?\s+on)\b",
    ],
    "Renewal Term": [
        r"(?i)\b(renewal\s+term|automatic(?:ally)?\s+renew(?:s)?|auto[\s-]?renew(?:al)?)\b",
    ],
    "Notice Period To Terminate Renewal": [
        r"(?i)\b(notice\s+(?:period\s+)?(?:to\s+)?(?:terminate|cancel)\s+renewal|written\s+notice\s+of\s+(?:non[\s-]?)?renewal)\b",
    ],
    "Governing Law": [
        r"(?i)\b(governing\s+law|choice\s+of\s+law|applicable\s+law|jurisdiction)\b",
        r"(?i)^[\d.]+\s*(governing|jurisdiction)",
    ],
    "Most Favored Nation": [
        r"(?i)\b(most\s+favou?red\s+nation|most[\s-]?favou?red\s+(?:customer|pricing)|MFN\b)\b",
    ],
    "Non-Compete": [
        r"(?i)\b(non[\s-]?compet(?:e|ition)|covenant\s+not\s+to\s+compet)\b",
        r"(?i)^[\d.]+\s*non[\s-]?compet",
    ],
    "Exclusivity": [
        r"(?i)\b(exclusiv(?:e|ity)|sole\s+(?:and\s+exclusive\s+)?(?:supplier|provider|vendor))\b",
    ],
    "No-Solicit Of Customers": [
        r"(?i)\b(solicit(?:ing|ation)?\s+(?:of\s+)?customers?|no[\s-]?solicit\s+(?:of\s+)?customers?)\b",
    ],
    "No-Solicit Of Employees": [
        r"(?i)\b(solicit(?:ing|ation)?\s+(?:of\s+)?employees?|no[\s-]?solicit\s+(?:of\s+)?employees?|hire\s+(?:away|any)\s+employees?)\b",
    ],
    "Non-Disparagement": [
        r"(?i)\b(non[\s-]?disparagement|disparage|defame|derogatory\s+statements?)\b",
    ],
    "Termination For Convenience": [
        r"(?i)\b(termination\s+for\s+convenience|terminate\s+(?:this\s+agreement\s+)?(?:at\s+will|without\s+cause))\b",
        r"(?i)^[\d.]+\s*termination",
    ],
    "ROFR/ROFO/ROFN": [
        r"(?i)\b(right\s+of\s+first\s+(refusal|offer|negotiation)|ROFR|ROFO|ROFN)\b",
    ],
    "Change Of Control": [
        r"(?i)\b(change\s+of\s+control|acquisition|merger\s+or\s+acquisition|change\s+in\s+control)\b",
    ],
    "Anti-Assignment": [
        r"(?i)\b(anti[\s-]?assignment|(?:no|not)\s+assign(?:able)?|may\s+not\s+assign|prohibit\s+assignment)\b",
        r"(?i)^[\d.]+\s*assignment",
    ],
    "Revenue/Profit Sharing": [
        r"(?i)\b(revenue\s+shar(?:ing|e)|profit\s+shar(?:ing|e)|royalt(?:y|ies)|revenue\s+split)\b",
    ],
    "Price Restrictions": [
        r"(?i)\b(price\s+(?:restriction|floor|ceiling|control)|most\s+favou?red\s+(?:nation|customer))\b",
    ],
    "Minimum Commitment": [
        r"(?i)\b(minimum\s+(?:purchase|order|commitment|volume)|purchase\s+obligation)\b",
    ],
    "Volume Restriction": [
        r"(?i)\b(volume\s+(?:restriction|limit|cap)|maximum\s+(?:volume|quantity|units?))\b",
    ],
    "IP Ownership Assignment": [
        r"(?i)\b(work[\s-]?for[\s-]?hire|assign(?:s|ment)\s+of\s+(?:all\s+)?(?:IP|intellectual\s+property))\b",
        r"(?i)\b(intellectual\s+property\s+(?:ownership|assignment|rights))\b",
    ],
    "Joint IP Ownership": [
        r"(?i)\b(joint(?:ly)?\s+own(?:ed)?|co[\s-]?own(?:ership)?|jointly\s+developed)\b",
    ],
    "License Grant": [
        r"(?i)\b(grant(?:s)?\s+(?:a\s+)?(?:non[\s-]?exclusive\s+)?licen(?:se|ce)|licen(?:se|ce)\s+grant)\b",
        r"(?i)^[\d.]+\s*licen",
    ],
    "Non-Transferable License": [
        r"(?i)\b(non[\s-]?transferable|not\s+transferable|may\s+not\s+transfer\s+(?:this\s+)?licen(?:se|ce))\b",
    ],
    "Affiliate License-Licensor": [
        r"(?i)\b(affiliate\s+licen(?:se|ce)|licen(?:se|ce)\s+to\s+affiliates?)\b",
    ],
    "Affiliate License-Licensee": [
        r"(?i)\b(licensee\s+affiliates?|sublicen(?:se|ce)\s+to\s+affiliates?)\b",
    ],
    "Unlimited/All-You-Can-Eat-License": [
        r"(?i)\b(unlimited\s+(?:licen(?:se|ce)|use|access)|all[\s-]?you[\s-]?can[\s-]?(?:eat|use)|enterprise[\s-]?wide\s+licen(?:se|ce))\b",
    ],
    "Irrevocable Or Perpetual License": [
        r"(?i)\b(irrevocable\s+licen(?:se|ce)|perpetual\s+licen(?:se|ce)|non[\s-]?terminable\s+licen(?:se|ce))\b",
    ],
    "Source Code Escrow": [
        r"(?i)\b(source\s+code\s+escrow|escrow\s+agent|escrow\s+agreement)\b",
    ],
    "Post-Termination Services": [
        r"(?i)\b(post[\s-]?termination\s+(?:services?|obligations?)|surviving\s+(?:clause|obligation|provision)|wind[\s-]?down)\b",
    ],
    "Audit Rights": [
        r"(?i)\b(audit\s+rights?|right\s+to\s+audit|inspection\s+rights?)\b",
    ],
    "Uncapped Liability": [
        r"(?i)\b(unlimited\s+liability|no\s+limit\s+on\s+liability|without\s+limitation\s+of\s+liability)\b",
    ],
    "Cap On Liability": [
        r"(?i)\b(limitation\s+of\s+liability|cap\s+on\s+(liability|damages)|aggregate\s+liability)\b",
        r"(?i)^[\d.]+\s*limitation",
    ],
    "Liquidated Damages": [
        r"(?i)\b(liquidated\s+damages|pre[\s-]?determined\s+damages|agreed[\s-]?upon\s+damages)\b",
    ],
    "Warranty Duration": [
        r"(?i)\b(warrant(?:y|ies|s)\s+(?:period|duration|term)|warrants?\s+for\s+a\s+period)\b",
        r"(?i)^[\d.]+\s*warrant",
    ],
    "Insurance": [
        r"(?i)\b(insurance|insur(?:e|ance)|general\s+liability\s+insurance|maintain\s+(?:insurance|coverage))\b",
        r"(?i)^[\d.]+\s*insurance",
    ],
    "Covenant Not To Sue": [
        r"(?i)\b(covenant\s+not\s+to\s+sue|release\s+of\s+claims?|waiver\s+of\s+claims?)\b",
    ],
    "Third Party Beneficiary": [
        r"(?i)\b(third[\s-]?party\s+beneficiar(?:y|ies)|no\s+third[\s-]?party\s+rights?)\b",
    ],
    "Indemnification": [
        r"(?i)\b(indemnif(?:ication|y|ies)|hold\s+harmless|defend\s+and\s+indemnify)\b",
        r"(?i)^[\d.]+\s*indemnif",
    ],
}

# Verify all CUAD clause types have patterns — log missing ones at import
_missing = [ct for ct in CUAD_CLAUSE_TYPES if ct not in CLAUSE_PATTERNS]
if _missing:
    logger.warning(f"No patterns for {len(_missing)} CUAD clause types: {_missing}")


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
    Falls back to paragraph splitting if no headers detected.
    """
    headers = detect_section_headers(text)
    if not headers:
        # Fallback — split by double newlines (paragraphs)
        # This handles CUAD contracts without clear section numbering
        paragraphs = []
        pos = 0
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) > 20:  # skip very short paragraphs
                paragraphs.append({
                    "header": "",
                    "text": para,
                    "start": text.find(para, pos),
                    "end": text.find(para, pos) + len(para),
                })
                pos = text.find(para, pos) + len(para)
        return paragraphs if paragraphs else [{"header": "", "text": text, "start": 0, "end": len(text)}]

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
    confidence: float
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
      2. spaCy NER to boost confidence when corroborating entity evidence exists.
      3. Each matched section emits ONE clause per matched type — not just the best.
         This ensures multi-type sections (e.g. "Indemnification and Insurance")
         contribute to both clause types during evaluation.
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
        Returns list of BaselineClause objects.

        ✅ FIX: Emits one clause per matched type per section — not just the
        single best. A section matching both Indemnification and Insurance
        produces two clauses, improving recall for multi-type sections.
        """
        sections = split_into_sections(contract_text)
        clauses = []
        clause_counter = 0

        for section in sections:
            section_text = section["text"]
            section_start = section["start"]

            # Store: clause_type -> (confidence, pattern, match)
            type_scores: dict[str, tuple[float, str, object]] = {}

            for clause_type, pattern_list in self.patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, section_text)
                    if match:
                        # Higher confidence if pattern matches section header
                        confidence = 0.85 if re.search(pattern, section["header"]) else 0.60
                        if (
                            clause_type not in type_scores
                            or type_scores[clause_type][0] < confidence
                        ):
                            type_scores[clause_type] = (confidence, pattern, match)

            # spaCy NER boost
            if self.spacy_available and section_text:
                doc = self.nlp(section_text[:5000])
                type_scores = self._apply_ner_boost(doc, type_scores)

            if not type_scores:
                continue

            # ✅ FIX: Emit one clause per matched type — not just the best type
            # Threshold to avoid very low confidence noise
            MIN_CONFIDENCE = 0.55
            for clause_type, (confidence, pattern, match) in type_scores.items():
                if confidence < MIN_CONFIDENCE:
                    continue

                if match:
                    span_start = section_text.rfind("\n", 0, best_match_start := match.start())
                    span_start = 0 if span_start == -1 else span_start + 1
                    span_end = section_text.find("\n", match.end())
                    span_end = len(section_text) if span_end == -1 else span_end
                    clause_text = section_text[span_start:span_end].strip()
                    abs_start = section_start + span_start
                    abs_end = section_start + span_end
                else:
                    clause_text = section_text[:500]
                    abs_start = section_start
                    abs_end = section_start + min(500, len(section_text))

                clause_id = _make_clause_id(doc_id, clause_type, clause_counter)
                clause_counter += 1

                clauses.append(BaselineClause(
                    clause_id=clause_id,
                    clause_text=clause_text,
                    clause_type=clause_type,
                    start_pos=abs_start,
                    end_pos=abs_end,
                    confidence=confidence,
                    matched_pattern=pattern,
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

        # ORG → Parties
        if "ORG" in entities and "Parties" in type_scores:
            if any(kw in text_lower for kw in ("between", "hereinafter", "party")):
                old_conf, pat, mat = type_scores["Parties"]
                type_scores["Parties"] = (min(old_conf + 0.1, 1.0), f"{pat}+spaCy:ORG", mat)

        # DATE → date-related clauses
        if "DATE" in entities:
            for ct, keywords in [
                ("Effective Date", ("effective", "as of", "commencing")),
                ("Expiration Date", ("expir", "terminat", "ends on", "term ends")),
                ("Warranty Duration", ("warrant", "guarantee", "defect")),
                ("Renewal Term", ("renew", "automatic", "extension")),
                ("Notice Period To Terminate Renewal", ("notice", "days prior", "written notice")),
                ("Agreement Date", ("dated", "as of", "entered into")),
            ]:
                if ct in type_scores and any(kw in text_lower for kw in keywords):
                    old_conf, pat, mat = type_scores[ct]
                    type_scores[ct] = (min(old_conf + 0.1, 1.0), f"{pat}+spaCy:DATE", mat)

        # MONEY → financial clauses
        if "MONEY" in entities:
            for ct, keywords in [
                ("Cap On Liability", ("limitation", "cap", "aggregate", "not exceed")),
                ("Liquidated Damages", ("liquidated", "predetermined", "agreed damages")),
                ("Minimum Commitment", ("minimum", "commit", "purchase obligation")),
                ("Revenue/Profit Sharing", ("revenue", "royalt", "profit")),
            ]:
                if ct in type_scores and any(kw in text_lower for kw in keywords):
                    old_conf, pat, mat = type_scores[ct]
                    type_scores[ct] = (min(old_conf + 0.1, 1.0), f"{pat}+spaCy:MONEY", mat)

        # GPE → Governing Law
        if "GPE" in entities and "Governing Law" in type_scores:
            if any(kw in text_lower for kw in ("govern", "jurisdiction", "applicable law")):
                old_conf, pat, mat = type_scores["Governing Law"]
                type_scores["Governing Law"] = (min(old_conf + 0.1, 1.0), f"{pat}+spaCy:GPE", mat)

        return type_scores


# ---------------------------------------------------------------------------
# CLI — single contract inference
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="Stage 1+2: Rule-Based Baseline")
    parser.add_argument("--contract_file", required=True, help="Path to .txt/.pdf/.docx contract")
    parser.add_argument("--output_file", default="baseline_clauses.json")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    from pipeline import preprocess_contract

    contract_text = preprocess_contract(args.contract_file)
    extractor = RuleBasedExtractor(spacy_model=args.spacy_model)
    clauses = extractor.extract(contract_text, doc_id=Path(args.contract_file).stem)

    output = [c.to_dict() for c in clauses]
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(output)} clauses to {args.output_file}")
