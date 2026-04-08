"""
Stage 1+2: Clause Extraction & Classification (Combined)
=========================================================
Uses DeBERTa-base fine-tuned on CUAD in native QA (span-extraction) format.
One model extracts clause boundaries AND assigns clause types simultaneously.

Architecture Decision (from plan):
  - Extraction + Classification merged into single model
  - CUAD native SQuAD-style QA format: question = "Where is the X clause?"
  - Evaluated on two dimensions: extraction quality + classification quality
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    pipeline,
    EarlyStoppingCallback,
)

from sklearn.metrics import accuracy_score, f1_score, classification_report

import evaluate
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from constants import CUAD_CLAUSE_TYPES, CUAD_QUESTION_TEMPLATES, QUESTION_TO_CLAUSE_TYPE

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClauseObject:
    """Output object for a single extracted + classified clause."""
    clause_id: str
    clause_text: str
    clause_type: str
    start_pos: int
    end_pos: int
    confidence: float
    document_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ExtractionResult:
    """Full result for one contract document."""
    document_id: str
    clauses: list = field(default_factory=list)

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "clauses": [c.to_dict() if isinstance(c, ClauseObject) else c for c in self.clauses],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fine_tune_deberta(
    model_name: str = "microsoft/deberta-base",
    output_dir: str = "./models/stage1_2_deberta",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    fp16: bool = True,
    data_path = None,
):
    """
    Fine-tune DeBERTa-base on CUAD QA task.

    Model choice (from plan):
      - DeBERTa-base (184M params): superior disentangled attention,
        5.4x better than BERT on CUAD benchmarks. VRAM: 8-16 GB.
      - Alternative: Legal-BERT (110M) — swap model_name if preferred.

    Training config targets Azure ML compute or A100 backup.
    """
    if data_path is None:
        data_path = "/content/drive/MyDrive/aiml_project/tokenized_cuad"

    logger.info(f"Loading preprocessed dataset from: {data_path}")
    tokenized = load_from_disk(data_path)

    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(data_path + "/tokenizer")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    
    logger.info(f"Tokenized sizes — train: {len(tokenized['train'])}, "
            f"val: {len(tokenized['validation'])}, test: {len(tokenized['test'])}")

    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs= num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,   
    per_device_eval_batch_size=per_device_eval_batch_size,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=learning_rate,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=fp16 and torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=2,
    gradient_accumulation_steps = 2,
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting fine-tuning…")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class ClauseExtractorClassifier:
    """
    Stage 1+2 inference pipeline.
    Runs all 41 CUAD query templates against a contract and returns
    a list of ClauseObject instances — one per detected clause.

    Usage:
        extractor = ClauseExtractorClassifier("./models/stage1_2_deberta")
        clauses = extractor.extract(contract_text, doc_id="contract_001")
        print(json.dumps([c.to_dict() for c in clauses], indent=2))
    """

    def __init__(self, model_path: str, device: int = -1):
        logger.info(f"Loading Stage 1+2 model from: {model_path}")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_path,
            tokenizer=model_path,
            device=device,
            handle_impossible_answer=True,
        )
        self.clause_types = CUAD_CLAUSE_TYPES
        self.question_templates = CUAD_QUESTION_TEMPLATES

    def extract(
        self,
        contract_text: str,
        doc_id: str = "unknown",
        confidence_threshold: float = 0.2,
    ) -> list[ClauseObject]:
        """
        Run all 41 clause-type queries against the contract.
        """
        clauses = []
        inputs = [
            {"question": self.question_templates[ct], "context": contract_text}
            for ct in self.clause_types
        ]

        logger.info(f"Running {len(inputs)} queries for doc: {doc_id}")
        results = self.qa_pipeline(inputs, batch_size=16)

        for idx, (clause_type, result) in enumerate(zip(self.clause_types, results)):
            answer_text = result.get("answer", "").strip()
            score = result.get("score", 0.0)

            if not answer_text or score < confidence_threshold:
                continue

            start = result.get("start", -1)
            end = result.get("end", -1)

            if start == -1:
                start = contract_text.find(answer_text)
                end = start + len(answer_text) if start != -1 else -1
                if start == -1:
                    logger.warning(
                        f"Could not locate answer text in doc '{doc_id}' "
                        f"for clause '{clause_type}': '{answer_text[:50]}'"
                    )
                    continue

            new_clause = ClauseObject(
                clause_id=f"{doc_id}_{clause_type.replace(' ', '_')}_{idx:04d}",
                clause_text=answer_text,
                clause_type=clause_type,
                start_pos=start,
                end_pos=end,
                confidence=round(score, 4),
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

                overlap = min(existing.end_pos, new_clause.end_pos) - max(existing.start_pos, new_clause.start_pos)

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

            for idx_to_remove in reversed(to_remove):
                clauses.pop(idx_to_remove)

            if keep:
                clauses.append(new_clause)

        clauses.sort(key=lambda c: c.start_pos)
        logger.info(f"Extracted {len(clauses)} clauses from {doc_id}")
        return clauses

    def extract_from_file(self, file_path: str, **kwargs) -> ExtractionResult:
        """
        Convenience method: read a .txt contract file and extract clauses.
        For PDF/DOCX, run preprocessing first (see preprocess_contract()).
        """
        text = Path(file_path).read_text(encoding="utf-8")
        doc_id = Path(file_path).stem
        clauses = self.extract(text, doc_id=doc_id, **kwargs)
        return ExtractionResult(document_id=doc_id, clauses=clauses)


# ---------------------------------------------------------------------------
# Contract preprocessing (PDF / DOCX → plain text)
# ---------------------------------------------------------------------------

def preprocess_contract(file_path: str) -> str:
    """
    Convert PDF or DOCX contract to clean plain text.
    Install: pip install pdfplumber python-docx
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)

    elif ext in (".docx", ".doc"):
        try:
            from docx import Document
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == ".txt":
        return Path(file_path).read_text(encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file format: {ext}. Use PDF, DOCX, or TXT.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_stage1_2(
    model_path: str,
    test_data_path: Optional[str] = None,
    output_path: str = "./results/stage1_2_eval.json",
    confidence_threshold: float = 0.2,
):
    squad_metric = evaluate.load("squad")

    if test_data_path is None:
        DATA_PATH = "/content/drive/MyDrive/aiml_project/tokenized_cuad"
        dataset = load_from_disk(DATA_PATH)
        test_examples = dataset["test"]
    else:
        with open(test_data_path) as f:
            test_examples = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer,
                  handle_impossible_answer=True)

    predictions, references = [], []
    pred_types, true_types = [], []

    for example in test_examples:
        result = qa({"question": example["question"], "context": example["context"]})
        pred_text = result.get("answer", "").strip()
        true_answers = example.get("answers", {}).get("text", [])

        predictions.append({
            "id": example["id"],
            "prediction_text": pred_text,
            "no_answer_probability": 1.0 - result["score"],
        })
        references.append({
            "id": example["id"],
            "answers": example["answers"],
        })

        true_type = _infer_clause_type_from_question(example["question"])
        true_has_clause = bool(true_answers and true_answers[0].strip())
        pred_has_clause = bool(pred_text and result["score"] >= confidence_threshold)
        true_type_label = true_type if true_has_clause else "NO_CLAUSE"
        pred_type = true_type if pred_has_clause else "NO_CLAUSE"
        true_types.append(true_type_label)
        pred_types.append(pred_type)

    squad_results = squad_metric.compute(predictions=predictions, references=references)
    class_accuracy = accuracy_score(true_types, pred_types)
    class_macro_f1 = f1_score(true_types, pred_types, average="macro", zero_division=0)
    class_report = classification_report(true_types, pred_types, zero_division=0)

    results = {
        "extraction": {
            "exact_match": squad_results["exact_match"],
            "f1": squad_results["f1"],
        },
        "classification": {
            "accuracy": class_accuracy,
            "macro_f1": class_macro_f1,
            "per_class_report": class_report,
        },
    }

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results:\n{json.dumps(results, indent=2)}")
    logger.info(f"Results saved to {output_path}")
    return results


def _infer_clause_type_from_question(question: str) -> str:
    if question in QUESTION_TO_CLAUSE_TYPE:
        return QUESTION_TO_CLAUSE_TYPE[question]
    question_lower = question.lower()
    for ct in CUAD_CLAUSE_TYPES:
        if ct.lower() in question_lower:
            return ct
    return "Unknown"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1+2: Clause Extraction & Classification")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Fine-tune DeBERTa on CUAD")
    train_parser.add_argument("--model", default="microsoft/deberta-base")
    train_parser.add_argument("--output_dir", default="./models/stage1_2_deberta")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--no_fp16", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="Run inference on a contract file")
    infer_parser.add_argument("--model_path", required=True)
    infer_parser.add_argument("--contract_file", required=True)
    infer_parser.add_argument("--output_file", default="clauses.json")
    infer_parser.add_argument("--threshold", type=float, default=0.2)
    infer_parser.add_argument("--device", type=int, default=-1)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on CUAD test set")
    eval_parser.add_argument("--model_path", required=True)
    eval_parser.add_argument("--output_path", default="./results/stage1_2_eval.json")

    args = parser.parse_args()

    if args.command == "train":
        fine_tune_deberta(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            fp16=not args.no_fp16,
        )

    elif args.command == "infer":
        contract_text = preprocess_contract(args.contract_file)
        extractor = ClauseExtractorClassifier(args.model_path, device=args.device)
        result = extractor.extract(contract_text, doc_id=Path(args.contract_file).stem,
                                   confidence_threshold=args.threshold)
        output = [c.to_dict() for c in result]
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved {len(output)} clauses to {args.output_file}")

    elif args.command == "evaluate":
        evaluate_stage1_2(
            model_path=args.model_path,
            output_path=args.output_path,
        )

    else:
        parser.print_help()