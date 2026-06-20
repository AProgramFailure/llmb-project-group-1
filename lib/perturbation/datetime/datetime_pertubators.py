import numpy as np
import pandas as pd
from datetime import datetime


def _validate_datetime(value) -> None:
    if not isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        raise TypeError(f"Expected datetime, got {type(value).__name__}")


def _shift_datetime(value: pd.Timestamp, *, seconds: float) -> pd.Timestamp:
    return value + pd.Timedelta(seconds=seconds)


def _get_scale(alpha: float, confidence: float) -> float:
    return alpha * (1 - confidence) * 86400  # scale in seconds, 1 day base


def perturb_datetime_uniform(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    scale = _get_scale(alpha, confidence)
    shift = np.random.uniform(-scale, scale)
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_normal(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    scale = _get_scale(alpha, confidence)
    shift = np.random.normal(0, scale)
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_beta(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    a, b = confidence * 10, alpha * 10 + 1e-9
    rate = np.random.beta(a, b) - 0.5
    shift = rate * _get_scale(alpha, confidence) * 2
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_exponential(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    scale = _get_scale(alpha, confidence)
    shift = np.random.exponential(scale) * np.random.choice([-1, 1])
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_zipf(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    a = 1 + confidence / (alpha + 1e-9)
    shift = np.random.zipf(a) * _get_scale(alpha, confidence) * np.random.choice([-1, 1])
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_bernoulli(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    p = confidence * (1 - alpha)
    if np.random.binomial(1, p):
        return value
    scale = _get_scale(alpha, confidence)
    return _shift_datetime(pd.Timestamp(value), seconds=np.random.normal(0, scale))

def perturb_datetime_poisson(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    lam = max(_get_scale(alpha, confidence), 1e-9)
    shift = np.random.poisson(lam) * np.random.choice([-1, 1])
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_log_normal(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    sigma = alpha * (1 - confidence)
    shift = np.random.lognormal(0, sigma) * _get_scale(alpha, confidence) * np.random.choice([-1, 1])
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_gamma(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    shape = confidence * 10 + 1e-9
    scale = _get_scale(alpha, confidence) / shape
    shift = np.random.gamma(shape, scale) * np.random.choice([-1, 1])
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_binomial(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    n = 100
    p = alpha * (1 - confidence)
    shift = (np.random.binomial(n, p) - n * p) * _get_scale(alpha, confidence) / n
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_pareto(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    shape = 1 + confidence / (alpha + 1e-9)
    shift = np.random.pareto(shape) * _get_scale(alpha, confidence) * np.random.choice([-1, 1])
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_weibull(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    shape = 1 + confidence
    shift = np.random.weibull(shape) * _get_scale(alpha, confidence) * np.random.choice([-1, 1])
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_laplace(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    scale = _get_scale(alpha, confidence)
    shift = np.random.laplace(0, scale)
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_cauchy(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    scale = _get_scale(alpha, confidence)
    shift = np.random.standard_cauchy() * scale
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_chi_squared(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    df = max(1, int(confidence * 10))
    shift = (np.random.chisquare(df) - df) * _get_scale(alpha, confidence) * np.random.choice([-1, 1])
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_student_t(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    df = max(1, int(confidence * 10))
    shift = np.random.standard_t(df) * _get_scale(alpha, confidence)
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_dirichlet(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    concentration = [confidence * 10 + 1e-9, alpha * 10 + 1e-9]
    weights = np.random.dirichlet(concentration)
    shift = (weights[1] - weights[0]) * _get_scale(alpha, confidence) * 2
    return _shift_datetime(pd.Timestamp(value), seconds=shift)

def perturb_datetime_multinomial(value, *, alpha: float, confidence: float) -> pd.Timestamp:
    _validate_datetime(value)
    if np.random.random() < confidence:
        return value
    probs = np.array([confidence, alpha, max(1 - confidence - alpha, 1e-9)])
    probs /= probs.sum()
    counts = np.random.multinomial(100, probs)
    shift = (counts[0] - counts[1]) * _get_scale(alpha, confidence) / 100
    return _shift_datetime(pd.Timestamp(value), seconds=shift)


DATETIME_PDF_REGISTRY: dict[str, callable] = {
    "uniform":     perturb_datetime_uniform,
    "normal":      perturb_datetime_normal,
    "beta":        perturb_datetime_beta,
    "exponential": perturb_datetime_exponential,
    "zipf":        perturb_datetime_zipf,
    "bernoulli":   perturb_datetime_bernoulli,
    "poisson":     perturb_datetime_poisson,
    "log_normal":  perturb_datetime_log_normal,
    "gamma":       perturb_datetime_gamma,
    "binomial":    perturb_datetime_binomial,
    "pareto":      perturb_datetime_pareto,
    "weibull":     perturb_datetime_weibull,
    "laplace":     perturb_datetime_laplace,
    "cauchy":      perturb_datetime_cauchy,
    "chi_squared": perturb_datetime_chi_squared,
    "student_t":   perturb_datetime_student_t,
    "dirichlet":   perturb_datetime_dirichlet,
    "multinomial": perturb_datetime_multinomial,
}


def perturb_datetime_column(
    *,
    df: pd.DataFrame,
    column: str,
    pdf: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if pdf not in DATETIME_PDF_REGISTRY:
        raise ValueError(f"PDF '{pdf}' not supported. Choose from: {list(DATETIME_PDF_REGISTRY.keys())}")
    fn = DATETIME_PDF_REGISTRY[pdf]
    return df[column].apply(lambda v: fn(v, alpha=alpha, confidence=confidence))