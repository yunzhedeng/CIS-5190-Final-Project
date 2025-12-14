# train_model_pt.py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import model
from preprocess import prepare_data


def compute_idf(texts, dim: int) -> torch.Tensor:
    """
    Compute IDF over hashed features using document frequency (DF).
    IDF = log((N+1)/(DF+1)) + 1
    """
    df = torch.zeros(dim, dtype=torch.float32)
    N = len(texts)

    for t in texts:
        idxs = model._active_feature_indices(t)
        if not idxs:
            continue
        df[idxs] += 1.0

    idf = torch.log((N + 1.0) / (df + 1.0)) + 1.0
    return idf


def main():
    train_csv = "merged_news_urls.csv"
    val_frac = 0.1

    # ========= 1) Load via preprocess (MUST match online, slug-only) =========
    X, y = prepare_data(train_csv)
    X = list(X)
    y = [str(v).strip().lower() for v in list(y)]

    pairs = [(xh, yh) for xh, yh in zip(X, y)
             if yh in ("fox", "nbc") and isinstance(xh, str) and xh.strip()]
    X = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    if len(y) == 0:
        raise RuntimeError("No labeled examples after preprocess. Check merged_news_urls.csv and preprocess.py.")

    # shuffle
    idx = torch.randperm(len(y)).tolist()
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    # split
    n_val = max(1, int(len(y) * val_frac))
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    # ========= 2) Build model =========
    # IMPORTANT: rename/remove old model.pt to avoid accidental local auto-load
    if os.path.exists("model.pt"):
        print("WARNING: model.pt exists. Rename it before training to avoid accidental loading:")
        print("  mv model.pt model_old.pt")

    m = model.Model()
    m.train()

    # ========= 3) Compute IDF on TRAIN split only =========
    idf = compute_idf(X_tr, model.FEATURE_DIM)
    m.idf.copy_(idf)

    # ========= 4) Featurize =========
    Xtr = torch.stack([model._featurize(t) for t in X_tr])
    Xva = torch.stack([model._featurize(t) for t in X_val])

    # Apply IDF + L2 normalize in training too (match predict)
    Xtr = Xtr * m.idf
    Xva = Xva * m.idf

    tr_norm = torch.linalg.norm(Xtr, dim=1, keepdim=True)
    va_norm = torch.linalg.norm(Xva, dim=1, keepdim=True)
    Xtr = torch.where(tr_norm > 0, Xtr / tr_norm, Xtr)
    Xva = torch.where(va_norm > 0, Xva / va_norm, Xva)

    ytr = torch.tensor([model.LABEL2ID[v] for v in y_tr], dtype=torch.long)
    yva = torch.tensor([model.LABEL2ID[v] for v in y_val], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=512, shuffle=False)

    # class weights
    counts = torch.bincount(ytr, minlength=2).float()
    w = counts.sum() / (counts + 1e-6)
    w = w / w.sum() * 2.0
    loss_fn = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)

    # ========= 5) Train =========
    best_acc = 0.0
    patience = 3
    bad = 0

    for epoch in range(1, 21):
        m.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            opt.zero_grad()
            logits = m.classifier(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        # val
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = m.classifier(xb).argmax(dim=1)
                correct += int((pred == yb).sum())
                total += yb.numel()

        val_acc = correct / max(1, total)
        print(f"epoch {epoch:02d} | train_loss={total_loss/max(1,len(train_loader)):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            bad = 0
            torch.save(m.state_dict(), "model.pt")
            print("  saved model.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Best val acc:", best_acc)
    print("Done. Upload: model.py + preprocess.py + model.pt")


if __name__ == "__main__":
    main()
