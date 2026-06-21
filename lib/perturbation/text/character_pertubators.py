import numpy as np
import pandas as pd
import random
import string


def _validate_text(value) -> None:
    if not isinstance(value, str):
        raise TypeError(f"Expected str, got {type(value).__name__}")


def _char_corrupt(text: str, n_corruptions: int) -> str:
    if not text:
        return text
    chars = list(text)
    ops = ["swap", "delete", "insert", "replace"]
    for _ in range(n_corruptions):
        op = random.choice(ops)
        idx = random.randint(0, len(chars) - 1)
        if op == "swap" and len(chars) > 1:
            j = random.randint(0, len(chars) - 1)
            chars[idx], chars[j] = chars[j], chars[idx]
        elif op == "delete" and len(chars) > 1:
            chars.pop(idx)
        elif op == "insert":
            chars.insert(idx, random.choice(string.ascii_letters))
        elif op == "replace":
            chars[idx] = random.choice(string.ascii_letters)
    return "".join(chars)


def _get_n_corruptions(text: str, *, alpha: float, confidence: float, distribution_fn) -> int:
    max_corruptions = max(1, int(len(text) * alpha))
    raw = distribution_fn(max_corruptions, confidence)
    return int(np.clip(raw, 0, max_corruptions)) if np.random.random() > confidence else 0


def perturb_text_char_uniform(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    n = max(1, int(len(value) * alpha))
    return _char_corrupt(value, np.random.randint(0, n + 1))

def perturb_text_char_normal(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    mean = len(value) * alpha
    std = mean * (1 - confidence)
    n = int(np.clip(np.random.normal(mean, std), 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_beta(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    a, b = confidence * 10, alpha * 10 + 1e-9
    rate = np.random.beta(a, b)
    n = int(len(value) * rate)
    return _char_corrupt(value, n)

def perturb_text_char_exponential(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    scale = len(value) * alpha * (1 - confidence)
    n = int(np.clip(np.random.exponential(scale), 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_zipf(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    a = 1 + confidence / (alpha + 1e-9)
    n = min(np.random.zipf(a), len(value))
    return _char_corrupt(value, n)

def perturb_text_char_bernoulli(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    chars = list(value)
    result = []
    for c in chars:
        if np.random.binomial(1, confidence * (1 - alpha)):
            result.append(c)
        else:
            result.append(random.choice(string.ascii_letters))
    return "".join(result)

def perturb_text_char_poisson(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    lam = max(len(value) * alpha * (1 - confidence), 1e-9)
    n = int(np.clip(np.random.poisson(lam), 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_log_normal(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    sigma = alpha * (1 - confidence)
    rate = np.random.lognormal(0, sigma)
    n = int(np.clip(len(value) * rate, 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_gamma(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    shape = confidence * 10 + 1e-9
    scale = alpha * (1 - confidence) * len(value) / shape
    n = int(np.clip(np.random.gamma(shape, scale), 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_binomial(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    chars = list(value)
    p = alpha * (1 - confidence)
    n_corrupt = np.random.binomial(len(chars), p)
    return _char_corrupt(value, n_corrupt)

def perturb_text_char_pareto(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    shape = 1 + confidence / (alpha + 1e-9)
    rate = min(np.random.pareto(shape) * alpha, 1.0)
    n = int(len(value) * rate)
    return _char_corrupt(value, n)

def perturb_text_char_weibull(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    shape = 1 + confidence
    rate = np.random.weibull(shape) * alpha * (1 - confidence)
    n = int(np.clip(len(value) * rate, 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_laplace(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    scale = len(value) * alpha * (1 - confidence)
    n = int(np.clip(abs(np.random.laplace(0, scale)), 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_cauchy(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    scale = len(value) * alpha * (1 - confidence)
    n = int(np.clip(abs(np.random.standard_cauchy() * scale), 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_chi_squared(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    df = max(1, int(confidence * 10))
    n = int(np.clip(np.random.chisquare(df) * alpha, 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_student_t(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    df = max(1, int(confidence * 10))
    scale = len(value) * alpha * (1 - confidence)
    n = int(np.clip(abs(np.random.standard_t(df)) * scale, 0, len(value)))
    return _char_corrupt(value, n)

def perturb_text_char_dirichlet(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    concentration = [confidence * 10 + 1e-9, alpha * 10 + 1e-9]
    weights = np.random.dirichlet(concentration)
    n = int(len(value) * weights[1])
    return _char_corrupt(value, n)

def perturb_text_char_multinomial(value, *, alpha: float, confidence: float) -> str:
    _validate_text(value)
    if np.random.random() < confidence:
        return value
    probs = np.array([confidence, alpha, max(1 - confidence - alpha, 1e-9)])
    probs /= probs.sum()
    counts = np.random.multinomial(len(value), probs)
    n = counts[1]
    return _char_corrupt(value, n)


CHAR_TEXT_PDF_REGISTRY: dict[str, callable] = {
    "uniform":     perturb_text_char_uniform,
    "normal":      perturb_text_char_normal,
    "beta":        perturb_text_char_beta,
    "exponential": perturb_text_char_exponential,
    "zipf":        perturb_text_char_zipf,
    "bernoulli":   perturb_text_char_bernoulli,
    "poisson":     perturb_text_char_poisson,
    "log_normal":  perturb_text_char_log_normal,
    "gamma":       perturb_text_char_gamma,
    "binomial":    perturb_text_char_binomial,
    "pareto":      perturb_text_char_pareto,
    "weibull":     perturb_text_char_weibull,
    "laplace":     perturb_text_char_laplace,
    "cauchy":      perturb_text_char_cauchy,
    "chi_squared": perturb_text_char_chi_squared,
    "student_t":   perturb_text_char_student_t,
    "dirichlet":   perturb_text_char_dirichlet,
    "multinomial": perturb_text_char_multinomial,
}


def perturb_text_char_column(
    *,
    df: pd.DataFrame,
    column: str,
    pdf: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if pdf not in CHAR_TEXT_PDF_REGISTRY:
        raise ValueError(f"PDF '{pdf}' not supported. Choose from: {list(CHAR_TEXT_PDF_REGISTRY.keys())}")
    fn = CHAR_TEXT_PDF_REGISTRY[pdf]
    return df[column].apply(lambda v: v if pd.isna(v) else fn(v, alpha=alpha, confidence=confidence))