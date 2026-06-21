import re
import numpy as np
import pandas as pd
import nltk


def _drop_sentences(text: str, alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return text
    sents = nltk.sent_tokenize(str(text))
    kept = [s for s in sents if np.random.random() > alpha]
    return ' '.join(kept) if kept else sents[0]


def _format_text(text: str, alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return text
    level = 1 if alpha < 0.20 else (2 if alpha < 0.40 else 3)
    sents = nltk.sent_tokenize(str(text))
    np.random.shuffle(sents)
    text = ' '.join(sents)
    if level >= 2:
        text = text.lower()
    if level >= 3:
        text = re.sub(r'[^\w\s]', '', text)
    return text


def _jitter_numbers(text: str, alpha: float, confidence: float) -> str:
    def _maybe_jitter(m):
        if np.random.random() < confidence:
            return m.group()
        return f"{float(m.group()) * (1 + np.random.uniform(-alpha, alpha)):.1f}"
    return re.sub(r'\b\d+\.?\d*\b', _maybe_jitter, str(text))


def perturb_text_sentence_missing_column(
    *,
    df: pd.DataFrame,
    column: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    """Drop alpha fraction of sentences. Row gate: confidence prob of keeping row unchanged."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    return df[column].apply(lambda v: v if pd.isna(v) else _drop_sentences(v, alpha, confidence))


def perturb_text_sentence_format_column(
    *,
    df: pd.DataFrame,
    column: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    """
    Structural format corruption. Row gate: confidence prob of keeping row unchanged.
    alpha < 0.20 → shuffle sentences only.
    alpha < 0.40 → shuffle + lowercase.
    alpha >= 0.40 → shuffle + lowercase + strip punctuation.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    return df[column].apply(lambda v: v if pd.isna(v) else _format_text(v, alpha, confidence))


def perturb_text_inline_number_column(
    *,
    df: pd.DataFrame,
    column: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    """Jitter numbers embedded in text. Per-number gate: confidence prob of keeping each number unchanged."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    return df[column].apply(lambda v: v if pd.isna(v) else _jitter_numbers(v, alpha, confidence))
