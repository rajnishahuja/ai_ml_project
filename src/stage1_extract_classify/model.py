"""
Inference-only Model wrapper for Stage 1+2.
Responsible for loading the local HuggingFace weights and running the extraction.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from ..common.constants import (
    CUAD_CLAUSE_TYPES,
    CUAD_QUESTION_TEMPLATES,
    _make_clause_id,
)

logger = logging.getLogger(__name__)


@dataclass
class ClauseObject:
    clause_id: str
    clause_text: str
    clause_type: str
    start_pos: int
    end_pos: int
    confidence: float
    confidence_logit: Optional[float] = None
    page_no: Optional[str] = None
    content_label: Optional[str] = None
    document_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ExtractionResult:
    document_id: str
    clauses: list = field(default_factory=list)

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "clauses": [
                c.to_dict() if isinstance(c, ClauseObject) else c for c in self.clauses
            ],
        }


class ClauseExtractorClassifier:
    MAX_ANSWER_LEN = 480

    def __init__(self, model_path: str):
        logger.info(f"Loading Stage 1+2 model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)

        # Architecture-agnostic device detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

        self.clause_types = CUAD_CLAUSE_TYPES
        self.question_templates = CUAD_QUESTION_TEMPLATES

    def _get_best_span(self, start_logits, end_logits, offsets, contract_text: str):
        start_log = start_logits.copy()
        end_log = end_logits.copy()

        for i, (s, e) in enumerate(offsets):
            if i != 0 and s == 0 and e == 0:
                start_log[i] = -1e9
                end_log[i] = -1e9

        cls_score = start_logits[0] + end_logits[0]
        best_score = -1e9
        best_start = 0
        best_end = 0

        for i in range(1, len(start_log)):
            if start_log[i] < -1e8:
                continue
            for j in range(i, min(i + self.MAX_ANSWER_LEN, len(end_log))):
                if end_log[j] < -1e8:
                    continue

                score = start_log[i] + end_log[j]

                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = j

        char_start = int(offsets[best_start][0])
        char_end = int(offsets[best_end][1])

        if char_start == 0 and char_end == 0:
            return None

        answer_text = contract_text[char_start:char_end].strip()

        if not answer_text or len(answer_text) < 5 or "@" in answer_text:
            return None

        return answer_text, char_start, char_end, best_score, cls_score

    def _resolve_metadata(self, doc_id: str, clause_text: str) -> dict:
        """
        Attempts to resolve the original PDF page number and element type
        (e.g., table, paragraph) by searching the Docling JSON metadata.
        Supports multi-page clauses returning formats like "2-3".
        """
        base_dir = Path(__file__).resolve().parent.parent.parent
        json_path = (
            base_dir / "data" / "processed" / "docling_outputs" / f"{doc_id}.json"
        )

        default_meta = {"page_no": None, "content_label": None}
        if not json_path.exists():
            return default_meta

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            def _norm(t: str) -> str:
                return "".join(c.lower() for c in t if c.isalnum())

            norm_start = _norm(clause_text[:40])
            norm_end = (
                _norm(clause_text[-40:]) if len(clause_text) >= 40 else norm_start
            )

            start_page = None
            end_page = None
            content_label = None

            for item in data.get("texts", []):
                it_text = _norm(item.get("text", ""))

                # Forward-only: our clause fragment must appear INSIDE a Docling block.
                # Bidirectional check causes false positives — e.g., a block containing just "3"
                # matches "suchotherpartythatremainsuncured30" via reverse substring.
                # 8-char minimum guard prevents single-word/digit tokens from matching.
                if not start_page and len(norm_start) >= 8 and norm_start in it_text:
                    prov = item.get("prov", [])
                    if prov:
                        start_page = prov[0].get("page_no")
                    content_label = item.get("label")

                # Check end mapping
                if not end_page and len(norm_end) >= 8 and norm_end in it_text:
                    prov = item.get("prov", [])
                    if prov:
                        end_page = prov[0].get("page_no")


            if start_page and end_page and start_page != end_page:
                page_no = f"{start_page}-{end_page}"
            elif start_page:
                page_no = str(start_page)
            else:
                page_no = None

            return {"page_no": page_no, "content_label": content_label}

        except Exception as e:
            logger.warning(f"Failed to resolve metadata for doc {doc_id}: {e}")
            return default_meta

    def extract(self, contract_text: str, doc_id: str = "unknown") -> list:
        clauses = []
        logger.info(
            f"Running Batched Inference for {len(self.clause_types)} queries on doc: {doc_id}"
        )

        questions = [self.question_templates[ct] for ct in self.clause_types]
        contracts = [contract_text] * len(self.clause_types)

        inputs = self.tokenizer(
            questions,
            contracts,
            truncation="only_second",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_mapping = inputs.pop("overflow_to_sample_mapping")

        batch_size = 32
        total_chunks = inputs["input_ids"].shape[0]
        logger.info(
            f"Generated {total_chunks} chunked inference blocks. Processing in batches of {batch_size}..."
        )

        start_logits_list = []
        end_logits_list = []

        with torch.no_grad():
            for i in range(0, total_chunks, batch_size):
                batch_inputs = {
                    k: v[i : i + batch_size].to(self.device) for k, v in inputs.items()
                }
                outputs = self.model(**batch_inputs)

                start_logits_list.append(outputs.start_logits.cpu().numpy())
                end_logits_list.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits_list, axis=0)
        end_logits = np.concatenate(end_logits_list, axis=0)

        # Track best span across all chunks per question
        global_best_answers = {
            idx: {"score": -1e9, "answer": None}
            for idx in range(len(self.clause_types))
        }

        for w in range(total_chunks):
            clause_idx = sample_mapping[w].item()
            offsets = offset_mapping[w].numpy()

            result = self._get_best_span(
                start_logits[w], end_logits[w], offsets, contract_text
            )

            if result and result[3] > global_best_answers[clause_idx]["score"]:
                global_best_answers[clause_idx]["score"] = result[3]
                global_best_answers[clause_idx]["answer"] = result

        for clause_idx, data in global_best_answers.items():
            if data["answer"] is None:
                continue

            clause_type = self.clause_types[clause_idx]
            answer_text, char_start, char_end, score, cls_score = data["answer"]

            CLAUSE_THRESHOLDS = {
                "Source Code Escrow": 3.0,
                "Unlimited/All-You-Can-Eat-License": 3.0,
                "Price Restrictions": 3.0,
                "Most Favored Nation": 2.5,
                "Third Party Beneficiary": 2.5,
                "No-Solicit Of Customers": 2.5,
                "Non-Disparagement": 2.5,
                "Affiliate License-Licensor": 2.0,
                "Joint IP Ownership": 2.0,
                "Irrevocable Or Perpetual License": 2.0,
                "Liquidated Damages": 2.0,
                "Affiliate License-Licensee": 2.0,
                "Volume Restriction": 2.0,
                "Audit Rights": 1.5,
            }
            if score < cls_score + CLAUSE_THRESHOLDS.get(clause_type, 1.0):
                continue

            CLAUSE_VALIDATORS = {
                "Volume Restriction": lambda t: any(
                    k in t.lower()
                    for k in (
                        "fee increase",
                        "consent",
                        "exceeds",
                        "overage",
                        "surcharge",
                        "additional charge",
                        "above",
                        "threshold",
                    )
                ),
                "Irrevocable Or Perpetual License": lambda t: any(
                    k in t.lower()
                    for k in ("license", "licen", "intellectual property", " ip ")
                ),
                "Audit Rights": lambda t: any(
                    k in t.lower()
                    for k in ("audit", "inspect", "examine", "records", "books")
                ),
                "Termination For Convenience": lambda t: any(
                    k in t.lower()
                    for k in (
                        "without cause",
                        "at will",
                        "convenience",
                        "upon notice",
                        "either party may terminate",
                    )
                ),
                "Joint IP Ownership": lambda t: any(
                    k in t.lower()
                    for k in (
                        "jointly own",
                        "co-own",
                        "joint ownership",
                        "jointly developed",
                        "shared ownership",
                    )
                ),
                "Indemnification": lambda t: any(
                    k in t.lower() for k in ("indemnif", "hold harmless", "defend")
                ),
            }
            validator = CLAUSE_VALIDATORS.get(clause_type)
            if validator and not validator(answer_text):
                logger.info(
                    f"[FILTERED] {clause_type} failed keyword val: {answer_text[:60]}"
                )
                continue

            conf_raw = float(score - cls_score)
            conf = 1 / (1 + np.exp(-(conf_raw / 2)))

            meta = self._resolve_metadata(doc_id, answer_text)

            new_clause = ClauseObject(
                clause_id=_make_clause_id(
                    doc_id, clause_type, clause_idx
                ),  # Now cleanly from constants!
                clause_text=answer_text,
                clause_type=clause_type,
                start_pos=char_start,
                end_pos=char_end,
                confidence=round(conf, 4),
                confidence_logit=round(conf_raw, 4),
                page_no=meta["page_no"],
                content_label=meta["content_label"],
                document_id=doc_id,
            )

            keep = True
            to_remove = []
            for i, existing in enumerate(clauses):
                if (
                    existing.start_pos == new_clause.start_pos
                    and existing.end_pos == new_clause.end_pos
                ):
                    if new_clause.confidence > existing.confidence:
                        to_remove.append(i)
                    else:
                        keep = False
                    continue

                overlap = min(existing.end_pos, new_clause.end_pos) - max(
                    existing.start_pos, new_clause.start_pos
                )
                if overlap > 0:
                    smaller_len = min(
                        existing.end_pos - existing.start_pos,
                        new_clause.end_pos - new_clause.start_pos,
                    )
                    if smaller_len > 0 and overlap / smaller_len > 0.5:
                        if new_clause.confidence > existing.confidence:
                            to_remove.append(i)
                        else:
                            keep = False

            for i in reversed(to_remove):
                clauses.pop(i)
            if keep:
                clauses.append(new_clause)

        clauses.sort(key=lambda c: c.start_pos)
        return clauses
