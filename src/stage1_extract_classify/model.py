import json
import logging
import os
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from ..common.constants import CUAD_CLAUSE_TYPES, CUAD_QUESTION_TEMPLATES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ClauseObject:
    """Output object for a single extracted + classified clause."""

    clause_id: str
    clause_text: str
    clause_type: str
    start_pos: int
    end_pos: int
    confidence: float
    confidence_logit: Optional[float] = None
    document_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class ClauseExtractorClassifier:
    """
    Stage 1+2 inference pipeline.
    Uses sliding windows to handle long contracts and resolves overlapping spans.
    """

    MAX_ANSWER_LEN = 340

    def __init__(self, model_path: str):
        logger.info(f"Loading Stage 1 model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.clause_types = CUAD_CLAUSE_TYPES
        self.question_templates = CUAD_QUESTION_TEMPLATES

    def _get_best_span(self, start_logits, end_logits, offsets, contract_text: str):
        """Finds the best valid answer span in a single tokenized window."""
        start_log = start_logits.copy()
        end_log = end_logits.copy()

        # Mask padding/special tokens (index 0 is CLS, keep it for no-answer)
        for i, (s, e) in enumerate(offsets):
            if i != 0 and s == 0 and e == 0:
                start_log[i] = -1e9
                end_log[i] = -1e9

        cls_score = start_logits[0] + end_logits[0]
        best_score, best_start, best_end = -1e9, 0, 0

        for i in range(1, len(start_log)):
            if start_log[i] < -1e8:
                continue
            for j in range(i, min(i + self.MAX_ANSWER_LEN, len(end_log))):
                if end_log[j] < -1e8:
                    continue
                score = start_log[i] + end_log[j]
                if score > best_score:
                    best_score, best_start, best_end = score, i, j

        char_start = int(offsets[best_start][0])
        char_end = int(offsets[best_end][1])

        if char_start == 0 and char_end == 0:
            return None

        answer_text = contract_text[char_start:char_end].strip()

        # Filter noise
        if not answer_text or len(answer_text) < 5 or "@" in answer_text:
            return None

        return answer_text, char_start, char_end, best_score, cls_score

    def extract(
        self, contract_text: str, doc_id: str = "unknown"
    ) -> List[ClauseObject]:
        """Runs all 41 CUAD queries against the text."""
        clauses = []
        logger.info(f"Running extraction for doc: {doc_id}")

        for clause_idx, clause_type in enumerate(self.clause_types):
            question = self.question_templates[clause_type]

            inputs = self.tokenizer(
                question,
                contract_text,
                truncation="only_second",
                max_length=384,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt",
            )

            offset_mapping = inputs.pop("offset_mapping")
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if k in ["input_ids", "attention_mask"]
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()

            global_best_answer = None
            global_best_score = -1e9

            for w in range(len(start_logits)):
                res = self._get_best_span(
                    start_logits[w],
                    end_logits[w],
                    offset_mapping[w].numpy(),
                    contract_text,
                )
                if res and res[3] > global_best_score:
                    global_best_score, global_best_answer = res[3], res

            if global_best_answer is None:
                continue

            answer_text, char_start, char_end, score, cls_score = global_best_answer

            # Confidence Threshold Filter
            if score < cls_score + 1.0:
                continue

            conf_raw = float(score - cls_score)
            conf = 1 / (1 + np.exp(-(conf_raw / 2)))

            new_clause = ClauseObject(
                clause_id=f"{doc_id}_{clause_type.replace(' ', '_')}_{clause_idx:04d}",
                clause_text=answer_text,
                clause_type=clause_type,
                start_pos=char_start,
                end_pos=char_end,
                confidence=round(conf, 4),
                confidence_logit=round(conf_raw, 4),
                document_id=doc_id,
            )

            # Overlap resolution
            keep = True
            for i in reversed(range(len(clauses))):
                existing = clauses[i]
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
                            clauses.pop(i)
                        else:
                            keep = False

            if keep:
                clauses.append(new_clause)

        return sorted(clauses, key=lambda c: c.start_pos)
