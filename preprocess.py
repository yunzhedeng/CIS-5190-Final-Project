# preprocess.py
from __future__ import annotations

import re
import hashlib
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

FEATURE_DIM = 16384  # 先用 8192，快且效果不错；想冲更高再改 16384

_HEADLINE_COL_CANDIDATES = [
    "headline",
    "alternative_headline",
    "scraped_headline",
    "title",
    "news_title",
    "headline_text",
    "text",
]

_LABEL_COL_CANDIDATES = ["label", "source", "publisher", "outlet", "site"]
_URL_COL_CANDIDATES = ["url", "link", "href"]

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9']+|[^\s]", re.IGNORECASE)

def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u200b", "").replace("\ufeff", "")
    s = _WS_RE.sub(" ", s.strip())
    return s

def _stable_hash32(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:4], "little", signed=False)

def _add_hashed(vec: np.ndarray, key: str, dim: int) -> None:
    h = _stable_hash32(key)
    idx = h % dim
    sign = 1.0 if (h & 1) == 0 else -1.0
    vec[idx] += sign

def _featurize_one(text: str, dim: int = FEATURE_DIM) -> np.ndarray:
    raw = _clean_text(text)
    t = raw.lower()
    vec = np.zeros((dim,), dtype=np.float32)

    # ---------------- char ngrams 3-5 ----------------
    for n in (3, 4, 5):
        if len(t) >= n:
            for i in range(len(t) - n + 1):
                _add_hashed(vec, f"c{n}:{t[i:i+n]}", dim)

    # ---------------- word ngrams 1-3 ----------------
    toks = _TOKEN_RE.findall(t)
    for tok in toks:
        _add_hashed(vec, f"w1:{tok}", dim)
    for i in range(len(toks) - 1):
        _add_hashed(vec, f"w2:{toks[i]}_{toks[i+1]}", dim)
    for i in range(len(toks) - 2):
        _add_hashed(vec, f"w3:{toks[i]}_{toks[i+1]}_{toks[i+2]}", dim)

    # -------- style / punctuation features (新增, 不传权重) --------
    puncts = [":", "?", "!", "-", "—", "'", '"', "“", "”", "’", "…", ",", "."]
    for p in puncts:
        c = t.count(p)
        for _ in range(c):
            _add_hashed(vec, f"p:{p}", dim)

    if raw:
        digit_cnt = sum(ch.isdigit() for ch in raw)
        upper_cnt = sum(ch.isupper() for ch in raw)
        alpha_cnt = sum(ch.isalpha() for ch in raw)
        length = len(raw)

        for _ in range(digit_cnt):
            _add_hashed(vec, "meta:digits", dim)
        for _ in range(upper_cnt):
            _add_hashed(vec, "meta:upper", dim)
        for _ in range(alpha_cnt):
            _add_hashed(vec, "meta:alpha", dim)
        for _ in range(length):
            _add_hashed(vec, "meta:len", dim)

        # 比例用“分桶”的方式离散化（避免传 float 权重）
        if alpha_cnt > 0:
            upper_ratio = upper_cnt / alpha_cnt
            bucket = int(min(10, max(0, round(upper_ratio * 10))))
            _add_hashed(vec, f"meta:upper_ratio_b{bucket}", dim)

        if length > 0:
            digit_ratio = digit_cnt / length
            bucket = int(min(10, max(0, round(digit_ratio * 10))))
            _add_hashed(vec, f"meta:digit_ratio_b{bucket}", dim)

    # ---------------- post-process ----------------
    vec = np.sign(vec) * np.log1p(np.abs(vec))  # signed log
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec



def featurize_batch(texts: List[str], dim: int = FEATURE_DIM) -> torch.Tensor:
    feats = np.stack([_featurize_one(x, dim=dim) for x in texts], axis=0)
    return torch.from_numpy(feats)  # float32 [N, D]

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _map_label_to_int(v) -> int:
    # Fox=0, NBC=1
    s = str(v).strip().lower()
    if "fox" in s:
        return 0
    if "nbc" in s:
        return 1
    if s.isdigit():
        return int(s)
    return 0

def prepare_data(csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)

    # ====== 输入：只用 headline（禁止 url 泄漏）======
    if "headline" in df.columns:
        texts = df["headline"].fillna("").astype(str).tolist()
    elif "scraped_headline" in df.columns:
        texts = df["scraped_headline"].fillna("").astype(str).tolist()
    elif "alternative_headline" in df.columns:
        texts = df["alternative_headline"].fillna("").astype(str).tolist()
    else:
        raise ValueError("No headline column found. Expected one of: headline/scraped_headline/alternative_headline")

    # ====== 标签：label 或 source ======
    if "label" in df.columns:
        y = df["label"].astype(int).tolist()
    elif "source" in df.columns:
        def to_int(s: str) -> int:
            s = str(s).strip().lower()
            if s.startswith("fox"):
                return 0
            if s.startswith("nbc"):
                return 1
            raise ValueError(f"Unknown source label: {s}")
        y = [to_int(v) for v in df["source"].tolist()]
    else:
        raise ValueError("No label column found. Expected: label or source")

    # ====== 特征化 ======
    X = featurize_batch(texts, dim=FEATURE_DIM)
    return X, y
