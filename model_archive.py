from __future__ import annotations

from typing import Any, Iterable, List, Dict
import os
import torch
from torch import nn

# --- External Dependencies ---
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    # Essential dependency check for the backend environment
    print("Error: The 'transformers' library is required for this model.")
    print("Please install it using: pip install torch transformers")
    # Define placeholders for graceful failure handling in non-standard environments
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


# =========================
# Model Configuration Settings
# =========================
ID2LABEL = {0: "fox", 1: "nbc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
MODEL_NAME = "roberta-base"
MAX_SEQ_LENGTH = 128  # Defined maximum sequence length for headline tokenization.


# =========================
# Model Class (PLM Fine-tuning for Inference)
# =========================
class Model(nn.Module):
    """
    Sequence Classification Model leveraging a Pre-trained Language Model (PLM)
    based on RoBERTa architecture for headline source discrimination.

    Implements the required inference interface: predict(batch) -> list of labels.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if AutoModelForSequenceClassification is None:
             raise RuntimeError("The 'transformers' library dependency is missing.")

        self.id2label: Dict[int, str] = dict(ID2LABEL)

        # 1. Initialize Tokenizer
        # Loads the vocabulary and encoding rules required for RoBERTa.
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 2. Initialize Pre-trained Model with Classification Head
        # AutoModelForSequenceClassification loads the RoBERTa encoder and adds 
        # a randomly initialized linear classification layer on top.
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(ID2LABEL)
        )
        
        
        # Load fine-tuned weights from the checkpoint file (model.pt)
        self._try_load_local_checkpoint("model.pt")

    def _try_load_local_checkpoint(self, path: str) -> None:
        """
        Attempts to load the fine-tuned model weights from a local checkpoint path.
        """
        if not os.path.exists(path):
            return
        try:
            # Load the state dictionary from the checkpoint
            state_dict = torch.load(path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                 sd = state_dict["state_dict"]
            elif isinstance(state_dict, dict):
                 sd = state_dict
            else:
                 return
            
            # Apply the loaded parameters to the classifier model
            self.classifier.load_state_dict(sd, strict=False)
            print(f"Loaded checkpoint from {path}")
        except Exception as e:
            print(f"Could not load local checkpoint: {e}")
            return

    def eval(self) -> None:
        """Sets the model to evaluation mode, disabling dropout, etc."""
        super().eval()
        self.classifier.eval()

    @torch.inference_mode()
    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Performs inference on a batch of raw text headlines.

        Args:
            batch: An iterable of headline strings.

        Returns:
            A list of predicted string labels (e.g., ["fox", "nbc"]).
        """
        items = list(batch)
        if not items:
            return []

        # Convert all inputs to strings
        texts: List[str] = [str(x) for x in items]
        
        # 1. Tokenization and Encoding
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,              # Pad sequences to the longest in the batch
            truncation=True,           # Truncate sequences exceeding MAX_SEQ_LENGTH
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"        # Return PyTorch tensor objects
        )
        
        # 2. Device Management
        # Infer the model's current device (CPU/GPU) and move input tensors there
        device = next(self.classifier.parameters()).device
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

        # 3. Model Forward Pass
        # Obtain model outputs, which include the classification logits
        outputs = self.classifier(**encoded_inputs)
        
        # 4. Extract and Determine Prediction
        logits = outputs.logits
        # Find the index with the maximum logit value (highest confidence class)
        pred_ids = torch.argmax(logits, dim=1).tolist()
        
        # 5. Map numerical IDs back to defined string labels
        return [self.id2label.get(int(i), "fox") for i in pred_ids]


def get_model() -> Model:
    """
    Standard factory function required by the evaluation environment.
    """
    return Model()