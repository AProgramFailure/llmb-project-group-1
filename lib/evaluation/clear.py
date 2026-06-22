import numpy as np

# Default CLEAR dimension weights.
# Efficacy is highest — does it actually work?
# Reliability reflects consistency under perturbation.
# Assurance reflects structured/verifiable outputs.
# Latency and Cost are operational concerns, weighted lower for research context.
# All weights are renormalized automatically, so you can set any values here.
DEFAULT_WEIGHTS = {
    'efficacy':    0.40,
    'reliability': 0.25,
    'assurance':   0.20,
    'latency':     0.10,
    'cost':        0.05,
}


def _normalize_latency(latency_ms: float, target_ms: float) -> float:
    """1.0 at or below target, linear decay to 0 at 3× target."""
    return float(np.clip(1 - (latency_ms - target_ms) / (2 * target_ms), 0.0, 1.0))


def _normalize_cost(cost_usd: float, budget_usd: float) -> float:
    """1.0 at or below budget per query, linear decay to 0 at 10× budget."""
    return float(np.clip(1 - (cost_usd - budget_usd) / (9 * budget_usd), 0.0, 1.0))


def compute_clear_score(
    *,
    efficacy: float,
    reliability: float,
    assurance: float,
    latency_ms: float,
    cost_usd: float,
    latency_target_ms: float = 5000.0,
    cost_budget_usd: float = 0.01,
    weights: dict | None = None,
) -> dict:
    """
    Compute CLEAR score across five dimensions.

    Parameters
    ----------
    efficacy          : weighted task score from scoring.efficacy_score() — [0, 1]
    reliability       : from reliability_from_runs() or reliability_from_drift() — [0, 1]
    assurance         : from scoring.assurance_score() — fraction of parseable outputs — [0, 1]
    latency_ms        : mean inference latency per query in milliseconds
    cost_usd          : mean cost per query in USD
    latency_target_ms : latency at which the latency score = 1.0 (default 5 s)
    cost_budget_usd   : cost per query at which the cost score = 1.0 (default $0.01)
    weights           : override DEFAULT_WEIGHTS for any subset of dimensions

    Returns
    -------
    dict with 'dimensions' (per-dimension scores), 'weights', and 'clear_score' (composite).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}

    dimensions = {
        'efficacy':    efficacy,
        'reliability': reliability,
        'assurance':   assurance,
        'latency':     _normalize_latency(latency_ms, latency_target_ms),
        'cost':        _normalize_cost(cost_usd, cost_budget_usd),
    }

    scored = {k: v for k, v in dimensions.items() if not np.isnan(v)}
    if not scored:
        composite = float('nan')
    else:
        used_weight = sum(w[k] for k in scored)
        composite = sum(w[k] * v for k, v in scored.items()) / used_weight

    return {
        'dimensions':   dimensions,
        'weights':      w,
        'clear_score':  composite,
    }


def reliability_from_runs(score_lists: list[list[float]]) -> float:
    """
    Reliability when multiple runs are available.
    Computes 1 - mean(per-question std across runs).
    score_lists: one list per run, each list has one score per question (same order).
    """
    arr = np.array(score_lists, dtype=float)
    return float(1 - np.nanmean(arr.std(axis=0)))


def reliability_from_drift(
    clean_scores: np.ndarray,
    perturbed_scores_by_level: dict[int, np.ndarray],
) -> float:
    """
    Proxy reliability when only one run is available.
    Defined as the ratio of worst-level mean score to clean mean score.
    A value close to 1.0 means performance barely drops under perturbation.
    """
    if not perturbed_scores_by_level or np.mean(clean_scores) == 0:
        return 1.0
    worst = min(float(np.mean(v)) for v in perturbed_scores_by_level.values())
    return float(np.clip(worst / float(np.mean(clean_scores)), 0.0, 1.0))
