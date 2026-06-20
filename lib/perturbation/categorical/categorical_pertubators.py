import numpy as np
import pandas as pd
from typing import Union

def _get_category_distribution(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = series.value_counts()
    categories = counts.index.to_numpy()
    frequencies = counts.values.astype(float)
    frequencies /= frequencies.sum()
    return categories, frequencies


def _validate_categorical(value) -> None:
    if not isinstance(value, (str, bool)) and not pd.api.types.is_categorical_dtype(type(value)):
        if isinstance(value, (int, float)):
            raise TypeError(f"Expected categorical value, got numeric: {value}")


def perturb_categorical_uniform(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    if np.random.random() < confidence:
        return value
    return np.random.choice(categories)


def perturb_categorical_normal(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    idx = np.where(categories == value)[0]
    current_idx = idx[0] if len(idx) > 0 else 0
    std = len(categories) * alpha * (1 - confidence)
    new_idx = int(np.round(np.random.normal(loc=current_idx, scale=std))) % len(categories)
    return categories[abs(new_idx)]


def perturb_categorical_beta(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    a = confidence * 10
    b = alpha * 10 + 1e-9
    weights = np.random.beta(a, b, size=len(categories))
    weights *= frequencies
    weights /= weights.sum()
    return np.random.choice(categories, p=weights)


def perturb_categorical_exponential(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    scale = alpha * (1 - confidence)
    weights = np.random.exponential(scale, size=len(categories))
    weights *= frequencies
    weights /= weights.sum()
    return np.random.choice(categories, p=weights)


def perturb_categorical_zipf(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    a = 1 + confidence / (alpha + 1e-9)
    weights = np.array([1 / (i + 1) ** a for i in range(len(categories))], dtype=float)
    weights /= weights.sum()
    return np.random.choice(categories, p=weights)


def perturb_categorical_bernoulli(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    p = confidence * (1 - alpha)
    flip = np.random.binomial(1, p)
    if flip:
        return value
    others = categories[categories != value]
    return np.random.choice(others) if len(others) > 0 else value


def perturb_categorical_poisson(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    idx = np.where(categories == value)[0]
    current_idx = idx[0] if len(idx) > 0 else 0
    lam = max(len(categories) * alpha * (1 - confidence), 1e-9)
    shift = np.random.poisson(lam)
    new_idx = (current_idx + shift) % len(categories)
    return categories[new_idx]


def perturb_categorical_log_normal(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    sigma = alpha * (1 - confidence)
    weights = np.random.lognormal(mean=0, sigma=sigma, size=len(categories))
    weights *= frequencies
    weights /= weights.sum()
    return np.random.choice(categories, p=weights)


def perturb_categorical_gamma(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    shape = confidence * 10 + 1e-9
    scale = alpha * (1 - confidence) / shape
    weights = np.random.gamma(shape, scale, size=len(categories))
    weights *= frequencies
    weights /= weights.sum()
    return np.random.choice(categories, p=weights)


def perturb_categorical_binomial(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    n = len(categories) - 1
    p = confidence * (1 - alpha)
    idx = np.random.binomial(n, p) % len(categories)
    return categories[idx]


def perturb_categorical_pareto(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    shape = 1 + confidence / (alpha + 1e-9)
    weights = np.random.pareto(shape, size=len(categories)) + 1
    weights *= frequencies
    weights /= weights.sum()
    return np.random.choice(categories, p=weights)


def perturb_categorical_weibull(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    shape = 1 + confidence
    weights = np.random.weibull(shape, size=len(categories))
    weights *= frequencies
    weights /= weights.sum()
    return np.random.choice(categories, p=weights)


def perturb_categorical_laplace(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    idx = np.where(categories == value)[0]
    current_idx = idx[0] if len(idx) > 0 else 0
    scale = len(categories) * alpha * (1 - confidence)
    new_idx = int(np.round(np.random.laplace(loc=current_idx, scale=scale))) % len(categories)
    return categories[abs(new_idx) % len(categories)]


def perturb_categorical_cauchy(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    idx = np.where(categories == value)[0]
    current_idx = idx[0] if len(idx) > 0 else 0
    scale = len(categories) * alpha * (1 - confidence)
    new_idx = int(np.round(current_idx + np.random.standard_cauchy() * scale)) % len(categories)
    return categories[abs(new_idx) % len(categories)]


def perturb_categorical_chi_squared(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    df = max(1, int(confidence * 10))
    weights = np.random.chisquare(df, size=len(categories))
    weights *= frequencies
    weights /= weights.sum()
    return np.random.choice(categories, p=weights)


def perturb_categorical_student_t(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    df = max(1, int(confidence * 10))
    idx = np.where(categories == value)[0]
    current_idx = idx[0] if len(idx) > 0 else 0
    scale = len(categories) * alpha * (1 - confidence)
    new_idx = int(np.round(current_idx + np.random.standard_t(df) * scale)) % len(categories)
    return categories[abs(new_idx) % len(categories)]


def perturb_categorical_dirichlet(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    concentration = frequencies * confidence * 10 + alpha + 1e-9
    weights = np.random.dirichlet(concentration)
    return np.random.choice(categories, p=weights)


def perturb_categorical_multinomial(value, *, categories: np.ndarray, frequencies: np.ndarray, alpha: float, confidence: float):
    n = 100
    probs = frequencies * confidence + alpha / len(categories)
    probs = np.clip(probs, 1e-9, None)
    probs /= probs.sum()
    counts = np.random.multinomial(n, probs)
    weights = counts / counts.sum()
    return np.random.choice(categories, p=weights)


CATEGORICAL_PDF_REGISTRY: dict[str, callable] = {
    "uniform":      perturb_categorical_uniform,
    "normal":       perturb_categorical_normal,
    "beta":         perturb_categorical_beta,
    "exponential":  perturb_categorical_exponential,
    "zipf":         perturb_categorical_zipf,
    "bernoulli":    perturb_categorical_bernoulli,
    "poisson":      perturb_categorical_poisson,
    "log_normal":   perturb_categorical_log_normal,
    "gamma":        perturb_categorical_gamma,
    "binomial":     perturb_categorical_binomial,
    "pareto":       perturb_categorical_pareto,
    "weibull":      perturb_categorical_weibull,
    "laplace":      perturb_categorical_laplace,
    "cauchy":       perturb_categorical_cauchy,
    "chi_squared":  perturb_categorical_chi_squared,
    "student_t":    perturb_categorical_student_t,
    "dirichlet":    perturb_categorical_dirichlet,
    "multinomial":  perturb_categorical_multinomial,
}


def perturb_categorical_column(
    *,
    df: pd.DataFrame,
    column: str,
    pdf: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if pdf not in CATEGORICAL_PDF_REGISTRY:
        raise ValueError(f"PDF '{pdf}' not supported. Choose from: {list(CATEGORICAL_PDF_REGISTRY.keys())}")

    series = df[column]
    categories, frequencies = _get_category_distribution(series)
    fn = CATEGORICAL_PDF_REGISTRY[pdf]

    return series.apply(
        lambda v: fn(
            v,
            categories=categories,
            frequencies=frequencies,
            alpha=alpha,
            confidence=confidence,
        )
    )