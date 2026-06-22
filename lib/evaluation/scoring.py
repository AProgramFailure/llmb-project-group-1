import numpy as np
import pandas as pd

# Efficacy weights per task type.
# Classification is highest due to binary right/wrong impact in CSRD regulatory context.
# Modify here to rebalance without touching any other code.
TASK_EFFICACY_WEIGHTS = {
    'classification': 0.40,
    'extraction':     0.35,
    'reasoning':      0.25,
}

# Map question_type values (as they appear in the datasets) to internal task categories
QUESTION_TYPE_MAP = {
    'PE':             'extraction',
    'NR':             'extraction',
    'LR':             'reasoning',
    'MCQ':            'classification',
    'CLASSIFICATION': 'classification',
}


def score_extraction(pred, truth) -> float:
    """Relative error mapped to [0, 1]. 1.0 = perfect, 0.0 = 100% or more off."""
    try:
        pred, truth = float(pred), float(truth)
        if truth == 0:
            return 1.0 if pred == 0 else 0.0
        return float(np.clip(1 - abs(pred - truth) / abs(truth), 0.0, 1.0))
    except (ValueError, TypeError):
        return 0.0


def score_classification(pred, truth) -> float:
    """Exact label match. 1.0 = correct, 0.0 = wrong."""
    return 1.0 if str(pred).strip().upper() == str(truth).strip().upper() else 0.0


def score_reasoning(pred: str, truth: str) -> float:
    """Placeholder until LLM-as-judge is implemented. Returns NaN so callers know it is unscored."""
    return float('nan')


def score_row(
    row: pd.Series,
    pred_col: str = 'prediction',
    truth_col: str = 'ground_truth',
    type_col: str = 'question_type',
) -> float:
    task = QUESTION_TYPE_MAP.get(str(row.get(type_col, '')).strip().upper())
    pred  = row.get(pred_col)
    truth = row.get(truth_col)
    if task == 'extraction':
        return score_extraction(pred, truth)
    if task == 'classification':
        return score_classification(pred, truth)
    if task == 'reasoning':
        return score_reasoning(pred, truth)
    return float('nan')


def score_dataframe(
    df: pd.DataFrame,
    pred_col: str = 'prediction',
    truth_col: str = 'ground_truth',
    type_col: str = 'question_type',
) -> pd.DataFrame:
    """Add a 'score' column to df. Returns a copy."""
    result = df.copy()
    result['score'] = result.apply(
        lambda row: score_row(row, pred_col=pred_col, truth_col=truth_col, type_col=type_col),
        axis=1,
    )
    return result


def confusion_matrix_report(
    df: pd.DataFrame,
    pred_col: str = 'prediction',
    truth_col: str = 'ground_truth',
    type_col: str = 'question_type',
) -> dict:
    """Confusion matrix and per-class metrics for classification tasks only."""
    clf = df[
        df[type_col].str.strip().str.upper().isin(['MCQ', 'CLASSIFICATION'])
    ].dropna(subset=[pred_col, truth_col])

    if clf.empty:
        return {}

    preds  = clf[pred_col].astype(str).str.strip().str.upper()
    truths = clf[truth_col].astype(str).str.strip().str.upper()
    labels = sorted(truths.unique())

    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    label_idx = {l: i for i, l in enumerate(labels)}
    for p, t in zip(preds, truths):
        if p in label_idx and t in label_idx:
            cm[label_idx[t]][label_idx[p]] += 1

    correct = int(np.diag(cm).sum())
    total   = int(cm.sum())

    per_class = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': int(cm[i].sum())}

    return {
        'accuracy':         correct / total if total > 0 else 0.0,
        'confusion_matrix': cm.tolist(),
        'labels':           labels,
        'per_class':        per_class,
    }


def assurance_score(
    df: pd.DataFrame,
    pred_col: str = 'prediction',
    type_col: str = 'question_type',
) -> float:
    """
    Fraction of predictions that are structured and parseable.
    MCQ: parseable as A/B/C/D. Extraction: parseable as a number. Reasoning: non-empty string.
    """
    def _is_parseable(row):
        task = QUESTION_TYPE_MAP.get(str(row.get(type_col, '')).strip().upper())
        pred = row.get(pred_col)
        if pd.isna(pred):
            return 0.0
        if task == 'classification':
            return 1.0 if str(pred).strip().upper() in ('A', 'B', 'C', 'D') else 0.0
        if task == 'extraction':
            try:
                float(pred)
                return 1.0
            except (ValueError, TypeError):
                return 0.0
        if task == 'reasoning':
            return 1.0 if str(pred).strip() else 0.0
        return 0.0

    scores = df.apply(_is_parseable, axis=1)
    return float(scores.mean()) if len(scores) > 0 else 0.0


def efficacy_score(
    df: pd.DataFrame,
    score_col: str = 'score',
    type_col: str = 'question_type',
    weights: dict | None = None,
) -> float:
    """
    Weighted efficacy across task types. Reasoning rows (NaN) are excluded
    until the LLM judge is implemented — weights are renormalized automatically.
    """
    w = {**TASK_EFFICACY_WEIGHTS, **(weights or {})}
    task_scores = {}
    for task, types in [
        ('extraction',     ['PE', 'NR']),
        ('classification', ['MCQ', 'CLASSIFICATION']),
        ('reasoning',      ['LR']),
    ]:
        subset = df[df[type_col].str.strip().str.upper().isin(types)][score_col].dropna()
        if not subset.empty:
            task_scores[task] = float(subset.mean())

    if not task_scores:
        return float('nan')

    total_weight = sum(w[t] for t in task_scores)
    return sum(w[t] * s for t, s in task_scores.items()) / total_weight
