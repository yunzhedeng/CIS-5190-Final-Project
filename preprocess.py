from __future__ import annotations

from typing import List, Tuple
import re
from urllib.parse import urlparse

import pandas as pd


# =========================
# Helpers
# =========================

MIN_LEN = 5


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Pick the first existing column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_spaces(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def _normalize_label(v: str) -> str:
    v = str(v).strip().lower()
    if v in ["fox", "foxnews", "fox_news", "fox-news", "foxnews.com"]:
        return "fox"
    if v in ["nbc", "nbcnews", "nbc_news", "nbc-news", "nbcnews.com"]:
        return "nbc"
    return v


def _url_to_label(url: str) -> str:
    """Infer label from domain if label column does not exist."""
    if not url:
        return ""
    host = (urlparse(url).netloc or "").lower().replace("www.", "")
    if host.endswith("foxnews.com"):
        return "fox"
    if host.endswith("nbcnews.com"):
        return "nbc"
    return ""


def _url_to_headline(url: str) -> str:
    """
    Convert URL to headline slug.
    This REMOVES category paths and keeps only the last segment.
    """
    if not url:
        return ""

    u = urlparse(url)
    path = (u.path or "").strip()

    # remove .print (Fox often uses it)
    path = re.sub(r"\.print$", "", path)

    # take last segment only (drop categories)
    slug = path.strip("/").split("/")[-1] if path else ""

    # remove trailing article ids like -rcna12345
    slug = re.sub(r"-(rcna|ncna)\d+$", "", slug, flags=re.IGNORECASE)

    headline = slug.replace("-", " ").replace("_", " ")
    return _normalize_spaces(headline.lower())


# =========================
# Required API
# =========================

def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Required by Project B pipeline.

    Input:
        csv_path: path to CSV (url-only or url+label)

    Returns:
        X: List[str]  -> headline text (derived from URL slug)
        y: List[str]  -> labels ("fox" / "nbc")
    """
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    # --- pick url column ---
    url_col = _pick_column(df, ["url", "link", "expanded_url", "URL", "Url"])
    if url_col is None:
        raise RuntimeError(f"No URL column found in CSV columns: {df.columns.tolist()}")

    df[url_col] = df[url_col].astype(str).str.strip()

    # --- headline: ALWAYS derived from URL (match online test) ---
    df["headline"] = df[url_col].map(_url_to_headline)

    # --- label ---
    label_col = _pick_column(df, ["label", "source", "site", "domain"])
    if label_col is not None:
        df["label"] = df[label_col].map(_normalize_label)
    else:
        # infer from domain if not provided
        df["label"] = df[url_col].map(_url_to_label)

    # --- filter invalid rows ---
    df = df[df["headline"].str.len() >= MIN_LEN]
    df = df[df["label"].isin(["fox", "nbc"])]

    df = df.reset_index(drop=True)

    X: List[str] = df["headline"].tolist()
    y: List[str] = df["label"].tolist()

    return X, y
