# train_model_pt.py (Transformer / RoBERTa fine-tuning script for project submission)
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
# Assuming preprocess.py contains a function named prepare_data
from preprocess import prepare_data 

# --- Label Mapping ---
# Labels must match the definition in model.py
ID2LABEL = {0: "fox", 1: "nbc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# --- Hyperparameters ---
MODEL_NAME = "roberta-base"
MAX_LEN = 128

# Data Configuration
TRAIN_CSV = "merged_news_urls.csv"
VAL_FRAC = 0.1 # Fraction of data used for validation

# Training Configuration
EPOCHS = 2                 # Recommended 1-3 epochs for CPU fine-tuning
BATCH_SIZE = 8             # Small batch size suitable for CPU memory
LR = 2e-5                  # Standard learning rate for RoBERTa fine-tuning (1e-5 to 3e-5)
WEIGHT_DECAY = 0.01        # L2 Regularization for AdamW
GRAD_ACCUM = 4             # Gradient Accumulation steps (Effective Batch Size = BATCH_SIZE * GRAD_ACCUM)
PATIENCE = 2               # Early stopping patience


class TextClsDataset(Dataset):
    """Custom Dataset for loading and tokenizing text classification data."""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Convert label string to ID
        y = LABEL2ID[self.labels[idx]]
        
        # Tokenize and encode the text
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        # Squeeze the tensor dimension added by return_tensors="pt"
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item


def accuracy(logits, labels):
    """Calculates batch accuracy."""
    pred = torch.argmax(logits, dim=-1)
    return (pred == labels).float().mean().item()


def main():
    # ---- Safety Check: Prevent accidental overwriting of the final model file ----
    if os.path.exists("model.pt"):
        print("WARNING: model.pt exists. Rename it before training to avoid accidental loading:", flush=True)
        print("  mv model.pt model_old.pt", flush=True)
        return

    # Use CPU since the environment is constrained
    device = torch.device("cpu")  
    print(f"[Info] Training device: {device}", flush=True)

    # ---- Load and Preprocess Data ----
    X, y = prepare_data(TRAIN_CSV)
    X = list(X)
    y = [str(v).strip().lower() for v in list(y)]

    # Filter data to ensure valid labels and non-empty strings
    pairs = [(xh, yh) for xh, yh in zip(X, y)
             if yh in ("fox", "nbc") and isinstance(xh, str) and xh.strip()]
    if not pairs:
        raise RuntimeError("No labeled examples after preprocess. Check input CSV and preprocess.py.")

    X = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    # Shuffle data
    idx = torch.randperm(len(y)).tolist()
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    # Split into training and validation sets
    n_val = max(1, int(len(y) * VAL_FRAC))
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    print(f"[Data] Training samples: {len(y_tr)} | Validation samples: {len(y_val)}", flush=True)

    # ---- Initialize Tokenizer and Model ----
    print(f"[Load] Initializing tokenizer and model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL)
    )
    model.to(device)

    # ---- Class Weights (optional but recommended for class imbalance) ----
    ytr_ids = torch.tensor([LABEL2ID[v] for v in y_tr], dtype=torch.long)
    counts = torch.bincount(ytr_ids, minlength=2).float()
    # Calculate inverse frequency weights
    w = counts.sum() / (counts + 1e-6)
    # Normalize weights
    w = w / w.sum() * 2.0 
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device))

    # ---- DataLoaders ----
    train_ds = TextClsDataset(X_tr, y_tr, tokenizer)
    val_ds = TextClsDataset(X_val, y_val, tokenizer)
    # num_workers=0 is necessary/recommended when debugging or using limited memory/CPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    # ---- Optimizer ----
    # AdamW is the standard optimizer for Transformer fine-tuning
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = 0.0
    bad = 0
    global_step = 0
    t0 = time.time()

    print("[Train] Starting fine-tuning...", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            # Move batch data to the target device
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits
            
            # Loss calculation and Gradient Accumulation
            loss = loss_fn(logits, labels) / GRAD_ACCUM
            loss.backward() # Accumulate gradients

            running_loss += loss.item() * GRAD_ACCUM
            running_acc += accuracy(logits.detach(), labels) * labels.size(0)
            seen += labels.size(0)

            if step % GRAD_ACCUM == 0:
                # Optimization step (after accumulating gradients)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            # Log training progress periodically
            if step % 50 == 0:
                avg_loss = running_loss / max(1, step)
                avg_acc = running_acc / max(1, seen)
                elapsed = time.time() - t0
                print(f"[Epoch {epoch}] Step {step}/{len(train_loader)} "
                      f"Loss={avg_loss:.4f} Acc={avg_acc:.4f} Elapsed={elapsed:.1f}s",
                      flush=True)

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                # Move validation batch data to the target device
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass for validation
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                
                pred = torch.argmax(logits, dim=-1)
                val_correct += int((pred == labels).sum().item())
                val_total += int(labels.numel())

        val_acc = val_correct / max(1, val_total)
        print(f"[Validation] Epoch {epoch} Validation Accuracy = {val_acc:.4f}", flush=True)

        # ---- Checkpoint and Early Stopping ----
        if val_acc > best_acc:
            best_acc = val_acc
            bad = 0
            # Save the state_dict of the fine-tuned RoBERTa model
            torch.save(model.state_dict(), "model.pt")
            print("[Save] model.pt (State dictionary of the RoBERTa classifier) saved.", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("[EarlyStop] Stopping training due to lack of improvement.", flush=True)
                break

    print(f"[Done] Training finished. Best Validation Accuracy: {best_acc:.4f}", flush=True)
    print("Final Upload Files: model.py + preprocess.py + model.pt", flush=True)


if __name__ == "__main__":
    main()