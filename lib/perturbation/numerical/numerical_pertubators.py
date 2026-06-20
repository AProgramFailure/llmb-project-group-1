import numpy as np
from typing import Union

Number = Union[int, float]


def _validate(value: Number) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected int or float, got {type(value).__name__}")


def perturb_numerical_uniform(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    spread = abs(value) * alpha * (1 - confidence)
    return float(np.random.uniform(value - spread, value + spread))


def perturb_numerical_normal(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    std = abs(value) * alpha * (1 - confidence)
    return float(np.random.normal(loc=value, scale=std))


def perturb_numerical_beta(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    a = confidence * 10
    b = (1 - confidence) * 10 + alpha
    noise = np.random.beta(a, b) - 0.5
    return float(value + noise * abs(value) * alpha)


def perturb_numerical_exponential(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    scale = abs(value) * alpha * (1 - confidence)
    noise = np.random.exponential(scale)
    return float(value + noise * np.sign(value))


def perturb_numerical_zipf(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    a = 1 + confidence / (alpha + 1e-9)
    noise = (np.random.zipf(a) - 1) * alpha * (1 - confidence)
    return float(value + noise)


def perturb_numerical_bernoulli(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    p = 1 - (alpha * (1 - confidence))
    flip = np.random.binomial(1, p)
    return float(value if flip else value * (1 + alpha * np.random.randn()))


def perturb_numerical_poisson(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    lam = max(abs(value) * alpha * (1 - confidence), 1e-9)
    noise = np.random.poisson(lam)
    return float(value + noise * np.sign(value))


def perturb_numerical_log_normal(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    sigma = alpha * (1 - confidence)
    factor = np.random.lognormal(mean=0, sigma=sigma)
    return float(value * factor)


def perturb_numerical_gamma(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    shape = confidence * 10 + 1e-9
    scale = abs(value) * alpha * (1 - confidence) / shape
    noise = np.random.gamma(shape, scale)
    return float(value + noise * np.sign(value))


def perturb_numerical_binomial(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    n = max(1, int(abs(value)))
    p = confidence * (1 - alpha)
    noise = np.random.binomial(n, p) - n * p
    return float(value + noise * alpha)


def perturb_numerical_pareto(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    shape = 1 + confidence / (alpha + 1e-9)
    noise = (np.random.pareto(shape)) * alpha * (1 - confidence)
    return float(value + noise * np.sign(value))


def perturb_numerical_weibull(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    shape = 1 + confidence
    noise = np.random.weibull(shape) * abs(value) * alpha * (1 - confidence)
    return float(value + noise * np.sign(value))


def perturb_numerical_laplace(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    scale = abs(value) * alpha * (1 - confidence)
    return float(np.random.laplace(loc=value, scale=scale))


def perturb_numerical_cauchy(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    scale = abs(value) * alpha * (1 - confidence)
    return float(value + np.random.standard_cauchy() * scale)


def perturb_numerical_chi_squared(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    df = max(1, int(confidence * 10))
    noise = (np.random.chisquare(df) - df) * alpha * (1 - confidence)
    return float(value + noise)


def perturb_numerical_student_t(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    df = max(1, int(confidence * 10))
    noise = np.random.standard_t(df) * abs(value) * alpha * (1 - confidence)
    return float(value + noise)


def perturb_numerical_dirichlet(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    concentration = [confidence * 10, alpha * 10 + 1e-9]
    sample = np.random.dirichlet(concentration)
    return float(value * (sample[0] / (sample[0] + sample[1])) * 2)


def perturb_numerical_multinomial(value: Number, *, alpha: float, confidence: float) -> float:
    _validate(value)
    n = max(1, int(abs(value)))
    probs = [confidence, alpha, 1 - confidence - alpha]
    probs = np.array(probs)
    probs = np.clip(probs, 1e-9, None)
    probs /= probs.sum()
    noise = np.random.multinomial(n, probs)
    return float(value + (noise[0] - noise[1]) * alpha)


PDF_REGISTRY: dict[str, callable] = {
    "uniform":      perturb_numerical_uniform,
    "normal":       perturb_numerical_normal,
    "beta":         perturb_numerical_beta,
    "exponential":  perturb_numerical_exponential,
    "zipf":         perturb_numerical_zipf,
    "bernoulli":    perturb_numerical_bernoulli,
    "poisson":      perturb_numerical_poisson,
    "log_normal":   perturb_numerical_log_normal,
    "gamma":        perturb_numerical_gamma,
    "binomial":     perturb_numerical_binomial,
    "pareto":       perturb_numerical_pareto,
    "weibull":      perturb_numerical_weibull,
    "laplace":      perturb_numerical_laplace,
    "cauchy":       perturb_numerical_cauchy,
    "chi_squared":  perturb_numerical_chi_squared,
    "student_t":    perturb_numerical_student_t,
    "dirichlet":    perturb_numerical_dirichlet,
    "multinomial":  perturb_numerical_multinomial,
}