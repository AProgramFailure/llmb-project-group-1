import numpy as np
import pandas as pd
from scipy import stats


def compute_drift_table(
    df: pd.DataFrame,
    score_col: str = 'score',
    perturbation_type_col: str = 'perturbation_type',
    perturbation_level_col: str = 'perturbation_level',
) -> pd.DataFrame:
    """Mean score per (perturbation_type, perturbation_level). Starting point for drift plots."""
    return (
        df.groupby([perturbation_type_col, perturbation_level_col])[score_col]
        .agg(mean_score='mean', std_score='std', n='count')
        .reset_index()
    )


def bootstrap_drift_ci(
    clean_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap CI for drift delta (clean_mean - perturbed_mean).
    Positive delta means performance dropped under perturbation.
    """
    rng = np.random.default_rng(42)
    deltas = np.array([
        rng.choice(clean_scores, size=len(clean_scores), replace=True).mean()
        - rng.choice(perturbed_scores, size=len(perturbed_scores), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return {
        'delta':     float(clean_scores.mean() - perturbed_scores.mean()),
        'ci_lower':  float(np.percentile(deltas, 100 * alpha / 2)),
        'ci_upper':  float(np.percentile(deltas, 100 * (1 - alpha / 2))),
        'ci_alpha':  alpha,
        'n_bootstrap': n_bootstrap,
    }


def significance_test(clean_scores: np.ndarray, perturbed_scores: np.ndarray) -> dict:
    """
    Mann-Whitney U test (non-parametric, no normality assumption).
    Tests whether clean scores are significantly higher than perturbed scores.
    """
    stat, p_value = stats.mannwhitneyu(clean_scores, perturbed_scores, alternative='greater')
    return {
        'test':        'Mann-Whitney U',
        'statistic':   float(stat),
        'p_value':     float(p_value),
        'significant': bool(p_value < 0.05),
    }


def full_drift_report(
    df: pd.DataFrame,
    score_col: str = 'score',
    perturbation_type_col: str = 'perturbation_type',
    perturbation_level_col: str = 'perturbation_level',
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    For each (perturbation_type, level), compute:
      - drift delta vs. level 0 (clean)
      - bootstrap CI
      - Mann-Whitney significance

    Returns one row per (perturbation_type, level).
    A large positive delta + significant p-value means that perturbation type
    meaningfully degrades model performance.
    """
    clean = df[df[perturbation_level_col] == 0][score_col].dropna().values
    rows = []
    for p_type in df[perturbation_type_col].unique():
        if p_type == 'none':
            continue
        for level in [1, 2, 3]:
            perturbed = df[
                (df[perturbation_type_col] == p_type) &
                (df[perturbation_level_col] == level)
            ][score_col].dropna().values

            if len(perturbed) == 0 or len(clean) == 0:
                continue

            ci  = bootstrap_drift_ci(clean, perturbed, n_bootstrap=n_bootstrap)
            sig = significance_test(clean, perturbed)
            rows.append({
                'perturbation_type':  p_type,
                'perturbation_level': level,
                **ci,
                'mw_statistic':  sig['statistic'],
                'p_value':       sig['p_value'],
                'significant':   sig['significant'],
            })

    return pd.DataFrame(rows)
