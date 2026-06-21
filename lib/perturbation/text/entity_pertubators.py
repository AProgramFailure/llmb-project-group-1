import numpy as np
import pandas as pd
from collections import defaultdict

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def build_entity_pools(series: pd.Series) -> dict[str, list[str]]:
    """
    Scan a text column with spaCy NER and collect all detected entity strings per type.
    Returns a dict mapping spaCy label (e.g. 'ORG', 'DATE', 'GPE') to a list of surface forms.
    Call this once before the perturbation loop to avoid re-scanning the corpus per level.
    """
    nlp = _get_nlp()
    pools = defaultdict(set)
    for doc in nlp.pipe(series.dropna().astype(str)):
        for ent in doc.ents:
            pools[ent.label_].add(ent.text)
    return {k: list(v) for k, v in pools.items()}


def _swap_entities_generic(
    text: str, doc, pools: dict, alpha: float, confidence: float
) -> str:
    if np.random.random() < confidence:
        return text
    result = list(text)
    for ent in reversed(doc.ents):  # reversed so char offsets stay valid after edits
        if np.random.random() < alpha:
            candidates = [
                e for e in pools.get(ent.label_, []) if e.lower() != ent.text.lower()
            ]
            if candidates:
                result[ent.start_char:ent.end_char] = list(np.random.choice(candidates))
    return "".join(result)


def perturb_text_entity_column(
    *,
    df: pd.DataFrame,
    column: str,
    alpha: float,
    confidence: float,
    pools: dict | None = None,
) -> pd.Series:
    """
    Swap named entities detected by spaCy with other entities of the same type.

    pools: pre-built entity pools from build_entity_pools(). Built automatically from
           df[column] if not provided, but pre-building is recommended when calling
           this function multiple times (e.g. once per perturbation level).
    Row gate: confidence prob of leaving a row completely unchanged.
    alpha: per-entity-span probability of being swapped when the row gate triggers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    nlp = _get_nlp()
    if pools is None:
        pools = build_entity_pools(df[column])
    valid_mask = df[column].notna()
    texts = df.loc[valid_mask, column].astype(str).tolist()
    docs = list(nlp.pipe(texts))
    results = df[column].copy()
    for idx, doc in zip(df.loc[valid_mask].index, docs):
        results[idx] = _swap_entities_generic(
            df.loc[idx, column], doc, pools, alpha, confidence
        )
    return results
