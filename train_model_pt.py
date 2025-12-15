# train_model_pt.py (Transformer / RoBERTa fine-tuning)
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from preprocess import prepare_data

# labels must match your model.py
ID2LABEL = {0: "fox", 1: "nbc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

MODEL_NAME = "roberta-base"
MAX_LEN = 128

TRAIN_CSV = "merged_news_urls.csv"
VAL_FRAC = 0.1

EPOCHS = 2                 # CPU上建议 1~3
BATCH_SIZE = 8             # CPU上建议 4/8/16
LR = 2e-5                  # RoBERTa 常用 1e-5 ~ 3e-5
WEIGHT_DECAY = 0.01
GRAD_ACCUM = 4             # 等效batch = BATCH_SIZE * GRAD_ACCUM
PATIENCE = 2               # early stop


class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        y = LABEL2ID[self.labels[idx]]
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item


def accuracy(logits, labels):
    pred = torch.argmax(logits, dim=-1)
    return (pred == labels).float().mean().item()


def main():
    # ---- safety: avoid accidentally loading/overwriting model.pt ----
    if os.path.exists("model.pt"):
        print("WARNING: model.pt exists. Rename it before training to avoid accidental loading:", flush=True)
        print("  mv model.pt model_old.pt", flush=True)
        return

    device = torch.device("cpu")  # your instance is CPU
    print(f"[Info] device={device}", flush=True)

    # ---- load data ----
    X, y = prepare_data(TRAIN_CSV)
    X = list(X)
    y = [str(v).strip().lower() for v in list(y)]

    pairs = [(xh, yh) for xh, yh in zip(X, y)
             if yh in ("fox", "nbc") and isinstance(xh, str) and xh.strip()]
    if not pairs:
        raise RuntimeError("No labeled examples after preprocess. Check merged_news_urls.csv and preprocess.py.")

    X = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    # shuffle
    idx = torch.randperm(len(y)).tolist()
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    # split
    n_val = max(1, int(len(y) * VAL_FRAC))
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    print(f"[Data] train={len(y_tr)}  val={len(y_val)}", flush=True)

    # ---- tokenizer & model ----
    print(f"[Load] tokenizer/model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL)
    )
    model.to(device)

    # ---- class weights (optional but helpful) ----
    ytr_ids = torch.tensor([LABEL2ID[v] for v in y_tr], dtype=torch.long)
    counts = torch.bincount(ytr_ids, minlength=2).float()
    w = counts.sum() / (counts + 1e-6)
    w = w / w.sum() * 2.0
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device))

    # ---- dataloaders ----
    train_ds = TextClsDataset(X_tr, y_tr, tokenizer)
    val_ds = TextClsDataset(X_val, y_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    # ---- optimizer ----
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = 0.0
    bad = 0
    global_step = 0
    t0 = time.time()

    print("[Train] start", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits
            loss = loss_fn(logits, labels) / GRAD_ACCUM
            loss.backward()

            running_loss += loss.item() * GRAD_ACCUM
            running_acc += accuracy(logits.detach(), labels) * labels.size(0)
            seen += labels.size(0)

            if step % GRAD_ACCUM == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            # log every ~50 mini-batches
            if step % 50 == 0:
                avg_loss = running_loss / max(1, step)
                avg_acc = running_acc / max(1, seen)
                elapsed = time.time() - t0
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} "
                      f"loss={avg_loss:.4f} acc={avg_acc:.4f} elapsed={elapsed:.1f}s",
                      flush=True)

        # ---- validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                pred = torch.argmax(logits, dim=-1)
                val_correct += int((pred == labels).sum().item())
                val_total += int(labels.numel())

        val_acc = val_correct / max(1, val_total)
        print(f"[Val] epoch {epoch} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            bad = 0
            # IMPORTANT: save ONLY the transformer classifier state_dict
            torch.save(model.state_dict(), "model.pt")
            print("[Save] model.pt (state_dict of RoBERTa classifier) ", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("[EarlyStop] stop training", flush=True)
                break

    print(f"[Done] best_val_acc={best_acc:.4f}", flush=True)
    print("Upload: model.py + preprocess.py + model.pt", flush=True)


if __name__ == "__main__":
    main()
