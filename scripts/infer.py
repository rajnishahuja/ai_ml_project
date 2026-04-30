#!/usr/bin/env python3
"""
infer.py — Stage 3 risk classifier inference (Ens-F: CE + CORN ensemble).

Loads both models once and exposes a predict() function for use in the agent
pipeline. Each call takes a clause_text, clause_type, and signing_party and
returns a risk label with confidence and per-class probabilities.

Models are loaded from HuggingFace Hub by default (or local paths for dev):
  CE model:   rajnishahuja/cuad-risk-deberta-ce-parties
  CORN model: rajnishahuja/cuad-risk-deberta-corn-parties

Usage — as a library:
    from scripts.infer import RiskClassifier
    clf = RiskClassifier()
    result = clf.predict(
        clause_text="Vendor hereby grants AT&T a perpetual irrevocable license...",
        clause_type="Affiliate License-Licensee",
        signing_party="Vendor",
    )
    # result = {"label": "HIGH", "confidence": 0.81, "probabilities": {"LOW": 0.07, "MEDIUM": 0.12, "HIGH": 0.81}}

Usage — as a CLI:
    python scripts/infer.py \\
        --clause_type "Affiliate License-Licensee" \\
        --signing_party "Vendor" \\
        --clause_text "Vendor hereby grants AT&T a perpetual irrevocable license..."
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.stage3_risk_agent.train import CORNWrapper  # noqa: E402

LABEL_NAMES = ["LOW", "MEDIUM", "HIGH"]
MAX_LENGTH = 512

CE_MODEL_ID   = "rajnishahuja/cuad-risk-deberta-ce-parties"
CORN_MODEL_ID = "rajnishahuja/cuad-risk-deberta-corn-parties"


def _load_ce_model(model_path: str, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=3,
    )
    return model.to(device).eval()


def _load_corn_model(model_path: str, device: torch.device):
    base = AutoModelForSequenceClassification.from_pretrained(
        CE_MODEL_ID, num_labels=1,
    )
    model = CORNWrapper(base)
    from safetensors.torch import load_file
    # model_path may be a local dir or HF repo — handle both
    p = Path(model_path)
    if p.exists() and (p / "model.safetensors").exists():
        state = load_file(str(p / "model.safetensors"), device="cpu")
    else:
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        state = load_file(weights_path, device="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def _seg_a(clause_type: str, signing_party: str) -> str:
    sp = signing_party.strip()
    return clause_type + (" | signing party: " + sp if sp else "")


class RiskClassifier:
    """Ens-F ensemble classifier. Loads both models once; reuse across calls."""

    def __init__(
        self,
        ce_model_path: str = CE_MODEL_ID,
        corn_model_path: str = CORN_MODEL_ID,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(f"Loading CE model from {ce_model_path} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(ce_model_path)
        self.ce_model = _load_ce_model(ce_model_path, self.device)

        print(f"Loading CORN model from {corn_model_path} ...", flush=True)
        self.corn_model = _load_corn_model(corn_model_path, self.device)
        print("Ready.")

    def _tokenize(self, clause_type: str, signing_party: str, clause_text: str):
        return self.tokenizer(
            _seg_a(clause_type, signing_party),
            clause_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    @torch.no_grad()
    def _ce_probs(self, inputs: dict) -> np.ndarray:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.ce_model(**inputs).logits
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]

    @torch.no_grad()
    def _corn_probs(self, inputs: dict) -> np.ndarray:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.corn_model(**inputs)
        logit1, logit2 = self.corn_model._corn_logits
        p_ge1 = torch.sigmoid(logit1).squeeze()
        p_ge2 = torch.sigmoid(logit2).squeeze() * p_ge1
        probs = torch.stack([1 - p_ge1, p_ge1 - p_ge2, p_ge2])
        return probs.float().cpu().numpy()

    def predict(
        self,
        clause_text: str,
        clause_type: str,
        signing_party: str = "",
    ) -> dict:
        """
        Classify a single clause.

        Args:
            clause_text:    Full text of the clause.
            clause_type:    CUAD clause type label (e.g. "Affiliate License-Licensee").
            signing_party:  Entity whose risk perspective to use (from Parties span).
                            Leave empty if unknown — model still runs, just less accurate.

        Returns:
            {
                "label":         "LOW" | "MEDIUM" | "HIGH",
                "confidence":    float  (max ensemble probability),
                "probabilities": {"LOW": float, "MEDIUM": float, "HIGH": float},
                "signing_party": str   (echoed back for traceability),
            }
        """
        inputs = self._tokenize(clause_type, signing_party, clause_text)
        ce_p   = self._ce_probs(inputs)
        corn_p = self._corn_probs(inputs)
        ens_p  = (ce_p + corn_p) / 2.0

        label_idx  = int(np.argmax(ens_p))
        return {
            "label":         LABEL_NAMES[label_idx],
            "confidence":    round(float(ens_p[label_idx]), 4),
            "probabilities": {
                lbl: round(float(ens_p[i]), 4)
                for i, lbl in enumerate(LABEL_NAMES)
            },
            "signing_party": signing_party,
        }

    def predict_batch(self, clauses: list[dict]) -> list[dict]:
        """
        Classify multiple clauses. Each dict must have keys:
          clause_text, clause_type, and optionally signing_party.
        """
        return [
            self.predict(
                clause_text=c["clause_text"],
                clause_type=c["clause_type"],
                signing_party=c.get("signing_party", ""),
            )
            for c in clauses
        ]


def main():
    ap = argparse.ArgumentParser(description="Stage 3 risk classifier — single clause")
    ap.add_argument("--clause_type",   required=True)
    ap.add_argument("--clause_text",   required=True)
    ap.add_argument("--signing_party", default="", help="Parties span entity name")
    ap.add_argument("--ce_model",   default=CE_MODEL_ID)
    ap.add_argument("--corn_model", default=CORN_MODEL_ID)
    ap.add_argument("--device",     default=None)
    args = ap.parse_args()

    clf = RiskClassifier(
        ce_model_path=args.ce_model,
        corn_model_path=args.corn_model,
        device=args.device,
    )
    result = clf.predict(
        clause_text=args.clause_text,
        clause_type=args.clause_type,
        signing_party=args.signing_party,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
