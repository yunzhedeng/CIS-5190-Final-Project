from __future__ import annotations

from typing import Any, Iterable, List, Dict
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =========================
# Configuration
# =========================
ID2LABEL = {0: "fox", 1: "nbc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

MODEL_NAME = "roberta-base"
MAX_SEQ_LENGTH = 128


class Model(nn.Module):
    """
    RoBERTa-based headline classifier for Project B.

    IMPORTANT:
    - The HuggingFace model is stored in `self.model`
      so that eval_project_b.py can correctly load model.pt.
    - We do NOT auto-load model.pt here.
      The evaluator will load weights via --weights.
    """

    def __init__(self, weights_path: str = "__no_weights__.pth") -> None:
        super().__init__()

        self.id2label: Dict[int, str] = dict(ID2LABEL)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # ---- IMPORTANT: name must be `self.model` ----
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(ID2LABEL)
        )

        # Do NOT auto-load weights unless explicitly asked
        if weights_path and weights_path != "__no_weights__.pth":
            sd = torch.load(weights_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            self.model.load_state_dict(sd, strict=False)

        self.eval()

    def eval(self) -> None:
        super().eval()
        self.model.eval()

    @torch.inference_mode()
    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = [str(x) for x in batch]
        if not texts:
            return []

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = self.model(**enc).logits
        pred_ids = torch.argmax(logits, dim=1).tolist()

        return [self.id2label[i] for i in pred_ids]


def get_model() -> Model:
    """
    Factory function required by the evaluation environment.
    """
    return Model()
