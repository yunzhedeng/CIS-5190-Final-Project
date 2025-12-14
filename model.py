from __future__ import annotations

from typing import Any, Iterable, List, Dict
import os
import re
import hashlib
from urllib.parse import urlparse

import torch
from torch import nn


# =========================
# Settings (stable + fast)
# =========================
FEATURE_DIM = 2 ** 17   # 131072
CHAR_NGRAM_MIN = 3
CHAR_NGRAM_MAX = 5
WORD_NGRAM_MAX = 3      # word unigrams + bigrams
MAX_FEATURES_PER_SAMPLE = 768
MIN_LEN = 5

ID2LABEL = {0: "fox", 1: "nbc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# =========================
# Text utilities
# =========================
def _normalize_spaces(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def _normalize_text(s: str) -> str:
    return _normalize_spaces(s).lower()


def _is_url_like(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("http://") or s.startswith("https://")


def _url_to_headline(url: str) -> str:
    """
    Defensive fallback only. Per your rule, you should feed only the final slug
    segment (pseudo-headline) from preprocess. This is used ONLY if a URL slips in.
    """
    if not url:
        return ""
    u = urlparse(url)
    path = (u.path or "").strip()
    path = re.sub(r"\.print$", "", path)

    slug = path.strip("/").split("/")[-1] if path else ""
    slug = re.sub(r"-(rcna|ncna)\d+$", "", slug, flags=re.IGNORECASE)

    headline = slug.replace("-", " ").replace("_", " ")
    return _normalize_spaces(headline)


# =========================
# Stable hashing (signed)
# =========================
def _hash64(key: str) -> int:
    h = hashlib.md5(key.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


def _hash_index_and_sign(key: str, dim: int) -> tuple[int, float]:
    h = _hash64(key)
    idx = h % dim
    sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
    return idx, sign


def _tokenize_words(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    toks = [t for t in text.split() if t]
    return toks[:256]


# =========================
# Feature extraction
# =========================
def _active_feature_indices(text: str) -> List[int]:
    """
    Return unique feature indices activated by this sample.
    Used to compute document frequency (DF) for IDF.

    IMPORTANT: must be consistent with _featurize hashing scheme.
    """
    text = _normalize_text(text)
    if len(text) < MIN_LEN:
        return []

    text = text[:2048]
    used = 0
    idxs: List[int] = []

    # word n-grams
    toks = _tokenize_words(text)
    for n in range(1, WORD_NGRAM_MAX + 1):
        if len(toks) < n:
            continue
        for i in range(0, len(toks) - n + 1):
            ng = " ".join(toks[i : i + n])
            idx, _ = _hash_index_and_sign(f"w{n}:{ng}", FEATURE_DIM)
            idxs.append(idx)
            used += 1
            if used >= MAX_FEATURES_PER_SAMPLE:
                break
        if used >= MAX_FEATURES_PER_SAMPLE:
            break

    # char n-grams
    for n in range(CHAR_NGRAM_MIN, CHAR_NGRAM_MAX + 1):
        if len(text) < n:
            continue
        for i in range(0, len(text) - n + 1):
            ng = text[i : i + n]
            idx, _ = _hash_index_and_sign(f"c{n}:{ng}", FEATURE_DIM)
            idxs.append(idx)
            used += 1
            if used >= MAX_FEATURES_PER_SAMPLE:
                break
        if used >= MAX_FEATURES_PER_SAMPLE:
            break

    # unique indices only (for DF counting)
    return list(set(idxs))


def _featurize(text: str) -> torch.Tensor:
    """
    TF part:
    - signed hashing for word uni/bi-grams + char 3~5-grams
    - log1p sublinear TF
    - (IDF applied later in Model.predict using self.idf)
    - L2 normalize after TF-IDF
    """
    text = _normalize_text(text)
    x = torch.zeros(FEATURE_DIM, dtype=torch.float32)

    if len(text) < MIN_LEN:
        return x

    text = text[:2048]
    used = 0

    # word n-grams
    toks = _tokenize_words(text)
    for n in range(1, WORD_NGRAM_MAX + 1):
        if len(toks) < n:
            continue
        for i in range(0, len(toks) - n + 1):
            ng = " ".join(toks[i : i + n])
            idx, sign = _hash_index_and_sign(f"w{n}:{ng}", FEATURE_DIM)
            x[idx] += sign
            used += 1
            if used >= MAX_FEATURES_PER_SAMPLE:
                break
        if used >= MAX_FEATURES_PER_SAMPLE:
            break

    # char n-grams
    for n in range(CHAR_NGRAM_MIN, CHAR_NGRAM_MAX + 1):
        if len(text) < n:
            continue
        for i in range(0, len(text) - n + 1):
            ng = text[i : i + n]
            idx, sign = _hash_index_and_sign(f"c{n}:{ng}", FEATURE_DIM)
            x[idx] += sign
            used += 1
            if used >= MAX_FEATURES_PER_SAMPLE:
                break
        if used >= MAX_FEATURES_PER_SAMPLE:
            break

    # sublinear TF: sign * log1p(|tf|)
    nz = x != 0
    if torch.any(nz):
        x[nz] = torch.sign(x[nz]) * torch.log1p(torch.abs(x[nz]))

    return x


class Model(nn.Module):
    """
    Leaderboard model.

    Requirements:
    - Instantiable with no args.
    - Implements predict(batch) -> list of labels.
    - If model.pt provided, backend loads it via state_dict.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_classes = 2
        self.classifier = nn.Linear(FEATURE_DIM, self.num_classes)

        # IDF buffer: default ones means "no IDF" if not trained
        self.register_buffer("idf", torch.ones(FEATURE_DIM, dtype=torch.float32))

        self.id2label: Dict[int, str] = dict(ID2LABEL)

        # Optional local load (backend will also load weights if provided)
        self._try_load_local_checkpoint("model.pt")

    def _try_load_local_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            return

        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        elif isinstance(obj, dict):
            sd = obj
        else:
            return

        self.load_state_dict(sd, strict=False)

    def eval(self) -> None:
        super().eval()

    @torch.inference_mode()
    def predict(self, batch: Iterable[Any]) -> List[Any]:
        items = list(batch)
        if not items:
            return []

        texts: List[str] = []
        for x in items:
            if x is None:
                texts.append("")
            elif isinstance(x, str):
                # defensive only; normally preprocess already outputs slug pseudo-headlines
                texts.append(_url_to_headline(x) if _is_url_like(x) else x)
            else:
                texts.append(str(x))

        # TF features
        X = torch.stack([_featurize(t) for t in texts], dim=0)  # [N, D]

        # TF-IDF: apply IDF learned from training
        X = X * self.idf

        # L2 normalize (important for TF-IDF)
        norms = torch.linalg.norm(X, dim=1, keepdim=True)
        X = torch.where(norms > 0, X / norms, X)

        logits = self.classifier(X.float())
        pred_ids = torch.argmax(logits, dim=1).tolist()
        return [self.id2label.get(int(i), "fox") for i in pred_ids]


def get_model() -> Model:
    return Model()
