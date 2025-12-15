from __future__ import annotations

from typing import Any, Iterable, List, Dict
import os
import torch
from torch import nn

# ⚠️ External Dependency: The 'transformers' library is crucial.
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    # If the library is missing, the model cannot function.
    print("Error: The 'transformers' library is required for this model.")
    print("Please install it using: pip install torch transformers")
    # Define placeholders to allow the code structure to be parsed, though it will fail at runtime.
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


# =========================
# Settings
# =========================
ID2LABEL = {0: "fox", 1: "nbc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
MODEL_NAME = "roberta-base"
MAX_SEQ_LENGTH = 128  # Headlines are typically short, 128 is usually sufficient.


# =========================
# Model Class (PLM Fine-tuning)
# =========================
class Model(nn.Module):
    """
    RoBERTa/PLM based classifier for news headlines.
    Uses AutoModelForSequenceClassification for a streamlined implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if AutoModelForSequenceClassification is None:
             raise RuntimeError("The 'transformers' library is not installed or available.")

        self.id2label: Dict[int, str] = dict(ID2LABEL)

        # 1. Load Tokenizer
        # Used to convert text into the model's specific input ID sequences.
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 2. Load Pre-trained Model with Classification Head
        # AutoModelForSequenceClassification automatically adds a linear layer 
        # on top of the transformer for sequence classification.
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(ID2LABEL)
        )

        # Attempt to load a local checkpoint (weights saved after fine-tuning)
        self._try_load_local_checkpoint("model.pt")

    def _try_load_local_checkpoint(self, path: str) -> None:
        """
        Loads the model weights for the PLM.
        """
        if not os.path.exists(path):
            return
        try:
            # Load the state dictionary
            state_dict = torch.load(path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                 sd = state_dict["state_dict"]
            elif isinstance(state_dict, dict):
                 sd = state_dict
            else:
                 return
            
            # Load the parameters into the classifier model
            # Setting strict=False can help if the checkpoint was saved slightly differently
            self.classifier.load_state_dict(sd, strict=False)
            print(f"Loaded checkpoint from {path}")
        except Exception as e:
            print(f"Could not load local checkpoint: {e}")
            return

    def eval(self) -> None:
        # Set both the wrapper and the internal model to evaluation mode
        super().eval()
        self.classifier.eval()

    @torch.inference_mode()
    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Predicts the label for a batch of headlines.
        """
        items = list(batch)
        if not items:
            return []

        # Ensure inputs are strings
        texts: List[str] = [str(x) for x in items]
        
        # 1. Tokenization and Padding
        # Convert text to PyTorch tensors, applying padding and truncation
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,              # Pad to the longest sequence in the batch
            truncation=True,           # Truncate if longer than MAX_SEQ_LENGTH
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"        # Return PyTorch tensors
        )
        
        # 2. Move input tensors to the same device as the model (CPU or GPU)
        device = next(self.classifier.parameters()).device
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

        # 3. Model Forward Pass
        # The PLM returns an object containing the logits
        outputs = self.classifier(**encoded_inputs)
        
        # 4. Extract logits and compute predicted IDs
        logits = outputs.logits
        # Get the index with the highest probability
        pred_ids = torch.argmax(logits, dim=1).tolist()
        
        # 5. Map back to labels
        return [self.id2label.get(int(i), "fox") for i in pred_ids]


def get_model() -> Model:
    """
    Standard interface function to return the model instance.
    """
    return Model()