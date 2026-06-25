import numpy as np
import pandas as pd


def _normalize_tokens(total_tokens: int, token_budget: int) -> float:
    """1.0 at or below budget, linear decay to 0 at 5× budget."""
    return float(np.clip(1 - (total_tokens - token_budget) / (4 * token_budget), 0.0, 1.0))


def _normalize_latency(latency_ms: float, target_ms: float) -> float:
    """1.0 at or below target, linear decay to 0 at 3× target."""
    return float(np.clip(1 - (latency_ms - target_ms) / (2 * target_ms), 0.0, 1.0))


def compute_footprint_score(
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    token_budget: int = 2000,
    latency_target_ms: float = 5000.0,
    token_weight: float = 0.6,
    latency_weight: float = 0.4,
) -> dict:
    """
    Resource footprint for a single query. Higher = lighter footprint.

    token_budget      : total tokens at which token_score = 1.0
    latency_target_ms : latency at which latency_score = 1.0
    token_weight /
    latency_weight    : relative contribution to footprint_score (auto-normalized)
    """
    total_tokens  = prompt_tokens + completion_tokens
    token_score   = _normalize_tokens(total_tokens, token_budget)
    latency_score = _normalize_latency(latency_ms, latency_target_ms)

    w_total   = token_weight + latency_weight
    composite = (token_weight * token_score + latency_weight * latency_score) / w_total

    return {
        'prompt_tokens':     prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens':      total_tokens,
        'token_score':       token_score,
        'latency_score':     latency_score,
        'footprint_score':   composite,
    }


def footprint_report(
    df: pd.DataFrame,
    prompt_tokens_col: str = 'prompt_tokens',
    completion_tokens_col: str = 'completion_tokens',
    latency_col: str = 'latency_ms',
    token_budget: int = 2000,
    latency_target_ms: float = 5000.0,
    token_weight: float = 0.6,
    latency_weight: float = 0.4,
) -> pd.DataFrame:
    """Add total_tokens, token_score, latency_score, footprint_score columns. Returns a copy."""
    result = df.copy()

    _nan_row = {
        'prompt_tokens': float('nan'), 'completion_tokens': float('nan'),
        'total_tokens': float('nan'), 'token_score': float('nan'),
        'latency_score': float('nan'), 'footprint_score': float('nan'),
    }

    def _safe_compute(row):
        pt = row[prompt_tokens_col]
        ct = row[completion_tokens_col]
        lt = row[latency_col]
        if pd.isna(pt) or pd.isna(ct) or pd.isna(lt):
            return _nan_row
        return compute_footprint_score(
            prompt_tokens=int(pt),
            completion_tokens=int(ct),
            latency_ms=float(lt),
            token_budget=token_budget,
            latency_target_ms=latency_target_ms,
            token_weight=token_weight,
            latency_weight=latency_weight,
        )

    scores = result.apply(_safe_compute, axis=1, result_type='expand')
    for col in ('total_tokens', 'token_score', 'latency_score', 'footprint_score'):
        result[col] = scores[col]
    return result


def footprint_summary(
    df: pd.DataFrame,
    prompt_tokens_col: str = 'prompt_tokens',
    completion_tokens_col: str = 'completion_tokens',
    latency_col: str = 'latency_ms',
    **kwargs,
) -> dict:
    """Aggregate footprint stats (means + stds) across the DataFrame."""
    scored = footprint_report(df, prompt_tokens_col, completion_tokens_col, latency_col, **kwargs)
    return {
        'mean_prompt_tokens':     float(scored[prompt_tokens_col].mean()),
        'mean_completion_tokens': float(scored[completion_tokens_col].mean()),
        'mean_total_tokens':      float(scored['total_tokens'].mean()),
        'std_total_tokens':       float(scored['total_tokens'].std()),
        'mean_latency_ms':        float(scored[latency_col].mean()),
        'std_latency_ms':         float(scored[latency_col].std()),
        'mean_footprint_score':   float(scored['footprint_score'].mean()),
        'std_footprint_score':    float(scored['footprint_score'].std()),
    }
