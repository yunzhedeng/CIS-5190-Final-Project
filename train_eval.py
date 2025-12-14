# train_eval.py
from __future__ import annotations

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from preprocess import prepare_data
from model import Model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 尽量可复现（GPU 上可能仍有少量不确定性，但会更稳）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    all_pred = []
    all_y = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        all_pred.append(pred)
        all_y.append(yb.numpy())
    all_pred = np.concatenate(all_pred)
    all_y = np.concatenate(all_y)
    return float(accuracy_score(all_y, all_pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="cleaned_headlines.csv")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} seed={args.seed} lr={args.lr}")

    # prepare_data: 期望返回 X(torch.FloatTensor [N,dim]) 和 y(0/1 list or np array)
    X, y = prepare_data(args.csv)
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    y = np.asarray(y, dtype=np.int64)

    # 固定用 seed 做可复现切分
    Xtr, Xva, ytr, yva = train_test_split(
        X.numpy(),
        y,
        test_size=0.15,
        random_state=args.seed,
        stratify=y
    )

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xva = torch.tensor(Xva, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    yva = torch.tensor(yva, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(Xva, yva),
        batch_size=256,
        shuffle=False,
        drop_last=False
    )

    # 关键：训练时不加载旧 ckpt
    model = Model(load_ckpt=False).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    best_sd = None
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        acc = eval_acc(model, val_loader, device)
        print(f"epoch {ep:02d}  val_acc={acc:.4f}")

        if acc > best + 1e-4:
            best = acc
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_sd is None:
        best_sd = model.state_dict()

    torch.save(best_sd, args.out)
    print("saved:", args.out, "best_val_acc=", best)

    # 输出 classification report
    model.load_state_dict(best_sd, strict=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            preds.extend(torch.argmax(model(xb), dim=-1).cpu().tolist())

    print(classification_report(yva.numpy(), np.array(preds), target_names=["fox(0)", "nbc(1)"]))


if __name__ == "__main__":
    main()
