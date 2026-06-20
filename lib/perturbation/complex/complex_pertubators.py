import numpy as np
import pandas as pd
from lib.perturbation.numerical.numerical_pertubators import (
    perturb_uniform, perturb_normal, perturb_beta, perturb_exponential,
    perturb_zipf, perturb_bernoulli, perturb_poisson, perturb_log_normal,
    perturb_gamma, perturb_binomial, perturb_pareto, perturb_weibull,
    perturb_laplace, perturb_cauchy, perturb_chi_squared, perturb_student_t,
    perturb_dirichlet, perturb_multinomial,
)



def _validate_complex(value) -> None:
    if not isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"Expected complex, got {type(value).__name__}")


def _perturb_component(x: float, *, alpha: float, confidence: float, fn) -> float:
    return fn(x, alpha=alpha, confidence=confidence)


def _make_complex_perturber(real_fn, imag_fn):
    def perturb(value, *, alpha: float, confidence: float) -> complex:
        _validate_complex(value)
        new_real = real_fn(value.real, alpha=alpha, confidence=confidence)
        new_imag = imag_fn(value.imag, alpha=alpha, confidence=confidence)
        return complex(new_real, new_imag)
    return perturb


# reuse numerical perturbation functions for real and imaginary parts independently

COMPLEX_PDF_REGISTRY: dict[str, callable] = {
    name: _make_complex_perturber(fn, fn)
    for name, fn in {
        "uniform":     perturb_uniform,
        "normal":      perturb_normal,
        "beta":        perturb_beta,
        "exponential": perturb_exponential,
        "zipf":        perturb_zipf,
        "bernoulli":   perturb_bernoulli,
        "poisson":     perturb_poisson,
        "log_normal":  perturb_log_normal,
        "gamma":       perturb_gamma,
        "binomial":    perturb_binomial,
        "pareto":      perturb_pareto,
        "weibull":     perturb_weibull,
        "laplace":     perturb_laplace,
        "cauchy":      perturb_cauchy,
        "chi_squared": perturb_chi_squared,
        "student_t":   perturb_student_t,
        "dirichlet":   perturb_dirichlet,
        "multinomial": perturb_multinomial,
    }.items()
}


def perturb_complex_column(
    *,
    df: pd.DataFrame,
    column: str,
    pdf: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if pdf not in COMPLEX_PDF_REGISTRY:
        raise ValueError(f"PDF '{pdf}' not supported. Choose from: {list(COMPLEX_PDF_REGISTRY.keys())}")
    fn = COMPLEX_PDF_REGISTRY[pdf]
    return df[column].apply(lambda v: fn(v, alpha=alpha, confidence=confidence))