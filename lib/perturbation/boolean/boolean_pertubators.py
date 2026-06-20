import numpy as np
import pandas as pd


def _get_bool_distribution(series: pd.Series) -> tuple[float, float]:
    freq_true = series.mean()
    freq_false = 1 - freq_true
    return freq_true, freq_false


def _validate_bool(value) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Expected bool, got {type(value).__name__}")


def _flip(value: bool, *, p_keep: float) -> bool:
    return value if np.random.random() < p_keep else not value


def perturb_bool_uniform(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    return _flip(value, p_keep=confidence)

def perturb_bool_normal(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    p_keep = float(np.clip(np.random.normal(confidence, alpha * (1 - confidence)), 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_beta(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    a = confidence * 10
    b = alpha * 10 + 1e-9
    p_keep = float(np.random.beta(a, b))
    return _flip(value, p_keep=p_keep)

def perturb_bool_exponential(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    scale = alpha * (1 - confidence)
    p_keep = float(np.clip(1 - np.random.exponential(scale), 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_zipf(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    a = 1 + confidence / (alpha + 1e-9)
    draw = np.random.zipf(a)
    p_keep = float(np.clip(1 / draw, 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_bernoulli(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    p_keep = confidence * (1 - alpha)
    return _flip(value, p_keep=p_keep)

def perturb_bool_poisson(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    lam = max(alpha * (1 - confidence), 1e-9)
    flips = np.random.poisson(lam)
    return value if flips % 2 == 0 else not value

def perturb_bool_log_normal(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    sigma = alpha * (1 - confidence)
    p_keep = float(np.clip(np.random.lognormal(0, sigma), 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_gamma(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    shape = confidence * 10 + 1e-9
    scale = alpha * (1 - confidence) / shape
    p_keep = float(np.clip(np.random.gamma(shape, scale), 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_binomial(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    p = confidence * (1 - alpha)
    flips = np.random.binomial(1, p)
    return value if flips == 1 else not value

def perturb_bool_pareto(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    shape = 1 + confidence / (alpha + 1e-9)
    p_keep = float(np.clip(1 / (np.random.pareto(shape) + 1), 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_weibull(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    shape = 1 + confidence
    p_keep = float(np.clip(np.random.weibull(shape), 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_laplace(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    scale = alpha * (1 - confidence)
    p_keep = float(np.clip(np.random.laplace(confidence, scale), 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_cauchy(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    scale = alpha * (1 - confidence)
    p_keep = float(np.clip(confidence + np.random.standard_cauchy() * scale, 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_chi_squared(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    df = max(1, int(confidence * 10))
    p_keep = float(np.clip(np.random.chisquare(df) / (df * 2), 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_student_t(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    df = max(1, int(confidence * 10))
    scale = alpha * (1 - confidence)
    p_keep = float(np.clip(confidence + np.random.standard_t(df) * scale, 0, 1))
    return _flip(value, p_keep=p_keep)

def perturb_bool_dirichlet(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    concentration = [freq_true * confidence * 10 + 1e-9, (1 - freq_true) * alpha * 10 + 1e-9]
    weights = np.random.dirichlet(concentration)
    return bool(np.random.choice([True, False], p=weights))

def perturb_bool_multinomial(value, *, freq_true: float, alpha: float, confidence: float) -> bool:
    _validate_bool(value)
    probs = np.array([freq_true * confidence, (1 - freq_true) * alpha])
    probs = np.clip(probs, 1e-9, None)
    probs /= probs.sum()
    counts = np.random.multinomial(100, probs)
    weights = counts / counts.sum()
    return bool(np.random.choice([True, False], p=weights))


BOOL_PDF_REGISTRY: dict[str, callable] = {
    "uniform":     perturb_bool_uniform,
    "normal":      perturb_bool_normal,
    "beta":        perturb_bool_beta,
    "exponential": perturb_bool_exponential,
    "zipf":        perturb_bool_zipf,
    "bernoulli":   perturb_bool_bernoulli,
    "poisson":     perturb_bool_poisson,
    "log_normal":  perturb_bool_log_normal,
    "gamma":       perturb_bool_gamma,
    "binomial":    perturb_bool_binomial,
    "pareto":      perturb_bool_pareto,
    "weibull":     perturb_bool_weibull,
    "laplace":     perturb_bool_laplace,
    "cauchy":      perturb_bool_cauchy,
    "chi_squared": perturb_bool_chi_squared,
    "student_t":   perturb_bool_student_t,
    "dirichlet":   perturb_bool_dirichlet,
    "multinomial": perturb_bool_multinomial,
}


def perturb_bool_column(
    *,
    df: pd.DataFrame,
    column: str,
    pdf: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if pdf not in BOOL_PDF_REGISTRY:
        raise ValueError(f"PDF '{pdf}' not supported. Choose from: {list(BOOL_PDF_REGISTRY.keys())}")

    series = df[column]
    freq_true, _ = _get_bool_distribution(series)
    fn = BOOL_PDF_REGISTRY[pdf]

    return series.apply(
        lambda v: fn(v, freq_true=freq_true, alpha=alpha, confidence=confidence)
    )