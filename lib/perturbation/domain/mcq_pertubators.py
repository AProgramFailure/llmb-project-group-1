import numpy as np
import pandas as pd


def _swap_labels(row: pd.Series, option_columns: list[str], answer_column: str, level: int) -> pd.Series:
    correct = row[answer_column]
    wrong = [o for o in option_columns if o != correct]
    new = row.copy()
    if level == 1:
        new[correct], new[wrong[0]] = row[wrong[0]], row[correct]
    elif level == 2:
        new[correct], new[wrong[0]], new[wrong[1]] = row[wrong[0]], row[wrong[1]], row[correct]
    elif level == 3:
        shuffled = [row[o] for o in option_columns]
        np.random.shuffle(shuffled)
        for o, t in zip(option_columns, shuffled):
            new[o] = t
    return new


def perturb_mcq_label_column(
    *,
    df: pd.DataFrame,
    option_columns: list[str],
    answer_column: str,
    alpha: float,
    confidence: float,
) -> pd.DataFrame:
    """
    Swap MCQ option texts without changing the answer key column.

    Row gate: confidence prob of leaving a row completely unchanged.
    alpha < 0.20: swap correct <-> one wrong option.
    alpha < 0.40: rotate correct -> wrong[0] -> wrong[1] -> correct.
    alpha >= 0.40: shuffle all option texts.

    Returns the full DataFrame with option columns modified in place.
    The answer_column value is never changed.
    """
    if answer_column not in df.columns:
        raise ValueError(f"Column '{answer_column}' not found in DataFrame.")
    missing = [c for c in option_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Option columns not found in DataFrame: {missing}")
    level = 1 if alpha < 0.20 else (2 if alpha < 0.40 else 3)
    def _maybe_swap(row):
        if np.random.random() < confidence:
            return row
        return _swap_labels(row, option_columns, answer_column, level)
    return df.apply(_maybe_swap, axis=1)
