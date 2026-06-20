import numpy as np
import pandas as pd


def _validate_timedelta(value) -> None:
    if not isinstance(value, (pd.Timedelta, np.timedelta64)):
        raise TypeError(f"Expected timedelta, got {type(value).__name__}")


def _shift_timedelta(value: pd.Timedelta, *, seconds: float) -> pd.Timedelta:
    return value + pd.Timedelta(seconds=seconds)


def _td_scale(value: pd.Timedelta, alpha: float, confidence: float) -> float:
    base = abs(value.total_seconds()) if value.total_seconds() != 0 else 86400
    return base * alpha * (1 - confidence)


def perturb_timedelta_uniform(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    return _shift_timedelta(pd.Timedelta(value), seconds=np.random.uniform(-scale, scale))

def perturb_timedelta_normal(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    return _shift_timedelta(pd.Timedelta(value), seconds=np.random.normal(0, scale))

def perturb_timedelta_beta(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    a, b = confidence * 10, alpha * 10 + 1e-9
    rate = np.random.beta(a, b) - 0.5
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    return _shift_timedelta(pd.Timedelta(value), seconds=rate * scale * 2)

def perturb_timedelta_exponential(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = np.random.exponential(scale) * np.random.choice([-1, 1])
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_zipf(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    a = 1 + confidence / (alpha + 1e-9)
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = np.random.zipf(a) * scale * np.random.choice([-1, 1])
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_bernoulli(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    p = confidence * (1 - alpha)
    if np.random.binomial(1, p):
        return value
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    return _shift_timedelta(pd.Timedelta(value), seconds=np.random.normal(0, scale))

def perturb_timedelta_poisson(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = np.random.poisson(max(scale, 1e-9)) * np.random.choice([-1, 1])
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_log_normal(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    sigma = alpha * (1 - confidence)
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = np.random.lognormal(0, sigma) * scale * np.random.choice([-1, 1])
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_gamma(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    shape = confidence * 10 + 1e-9
    scale = _td_scale(pd.Timedelta(value), alpha, confidence) / shape
    shift = np.random.gamma(shape, scale) * np.random.choice([-1, 1])
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_binomial(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    n, p = 100, alpha * (1 - confidence)
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = (np.random.binomial(n, p) - n * p) * scale / n
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_pareto(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    shape = 1 + confidence / (alpha + 1e-9)
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = np.random.pareto(shape) * scale * np.random.choice([-1, 1])
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_weibull(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    shape = 1 + confidence
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = np.random.weibull(shape) * scale * np.random.choice([-1, 1])
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_laplace(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    return _shift_timedelta(pd.Timedelta(value), seconds=np.random.laplace(0, scale))

def perturb_timedelta_cauchy(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    return _shift_timedelta(pd.Timedelta(value), seconds=np.random.standard_cauchy() * scale)

def perturb_timedelta_chi_squared(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    df = max(1, int(confidence * 10))
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = (np.random.chisquare(df) - df) * scale * np.random.choice([-1, 1])
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_student_t(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    df = max(1, int(confidence * 10))
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    return _shift_timedelta(pd.Timedelta(value), seconds=np.random.standard_t(df) * scale)

def perturb_timedelta_dirichlet(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    concentration = [confidence * 10 + 1e-9, alpha * 10 + 1e-9]
    weights = np.random.dirichlet(concentration)
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = (weights[1] - weights[0]) * scale * 2
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)

def perturb_timedelta_multinomial(value, *, alpha: float, confidence: float) -> pd.Timedelta:
    _validate_timedelta(value)
    if np.random.random() < confidence:
        return value
    probs = np.array([confidence, alpha, max(1 - confidence - alpha, 1e-9)])
    probs /= probs.sum()
    counts = np.random.multinomial(100, probs)
    scale = _td_scale(pd.Timedelta(value), alpha, confidence)
    shift = (counts[0] - counts[1]) * scale / 100
    return _shift_timedelta(pd.Timedelta(value), seconds=shift)


TIMEDELTA_PDF_REGISTRY: dict[str, callable] = {
    "uniform":     perturb_timedelta_uniform,
    "normal":      perturb_timedelta_normal,
    "beta":        perturb_timedelta_beta,
    "exponential": perturb_timedelta_exponential,
    "zipf":        perturb_timedelta_zipf,
    "bernoulli":   perturb_timedelta_bernoulli,
    "poisson":     perturb_timedelta_poisson,
    "log_normal":  perturb_timedelta_log_normal,
    "gamma":       perturb_timedelta_gamma,
    "binomial":    perturb_timedelta_binomial,
    "pareto":      perturb_timedelta_pareto,
    "weibull":     perturb_timedelta_weibull,
    "laplace":     perturb_timedelta_laplace,
    "cauchy":      perturb_timedelta_cauchy,
    "chi_squared": perturb_timedelta_chi_squared,
    "student_t":   perturb_timedelta_student_t,
    "dirichlet":   perturb_timedelta_dirichlet,
    "multinomial": perturb_timedelta_multinomial,
}


def perturb_timedelta_column(
    *,
    df: pd.DataFrame,
    column: str,
    pdf: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if pdf not in TIMEDELTA_PDF_REGISTRY:
        raise ValueError(f"PDF '{pdf}' not supported. Choose from: {list(TIMEDELTA_PDF_REGISTRY.keys())}")
    fn = TIMEDELTA_PDF_REGISTRY[pdf]
    return df[column].apply(lambda v: fn(v, alpha=alpha, confidence=confidence))