import numpy as np
import pandas as pd
from lib.llm.llm_client import client, MODEL


def _get_paraphrase_instruction(alpha: float, confidence: float) -> str:
    if alpha < 0.3 and confidence > 0.7:
        level = "very slightly. Change only 1-2 words at most. Preserve meaning completely."
    elif alpha < 0.6:
        level = "moderately. Rephrase some sentences but keep the core meaning."
    else:
        level = "aggressively. Significantly reword while keeping the general topic."
    return f"Paraphrase the following text {level} Return only the paraphrased text, nothing else."


def _semantic_perturb(value: str, *, alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    instruction = _get_paraphrase_instruction(alpha, confidence)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": value},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content.strip()


def _semantic_perturb_with_pdf(value: str, *, alpha: float, confidence: float, pdf: str) -> str:
    pdf_alpha_map = {
        "uniform":     lambda: np.random.uniform(0, alpha),
        "normal":      lambda: float(np.clip(np.random.normal(alpha, alpha * (1 - confidence)), 0, 1)),
        "beta":        lambda: float(np.random.beta(confidence * 10, alpha * 10 + 1e-9)),
        "exponential": lambda: float(np.clip(np.random.exponential(alpha * (1 - confidence)), 0, 1)),
        "zipf":        lambda: float(np.clip(1 / np.random.zipf(1 + confidence / (alpha + 1e-9)), 0, 1)),
        "bernoulli":   lambda: alpha if np.random.binomial(1, 1 - confidence) else 0.0,
        "poisson":     lambda: float(np.clip(np.random.poisson(max(alpha * (1 - confidence), 1e-9)), 0, 1)),
        "log_normal":  lambda: float(np.clip(np.random.lognormal(0, alpha * (1 - confidence)), 0, 1)),
        "gamma":       lambda: float(np.clip(np.random.gamma(confidence * 10 + 1e-9, alpha / (confidence * 10 + 1e-9)), 0, 1)),
        "binomial":    lambda: float(np.random.binomial(10, alpha * (1 - confidence)) / 10),
        "pareto":      lambda: float(np.clip(np.random.pareto(1 + confidence / (alpha + 1e-9)) * alpha, 0, 1)),
        "weibull":     lambda: float(np.clip(np.random.weibull(1 + confidence) * alpha, 0, 1)),
        "laplace":     lambda: float(np.clip(abs(np.random.laplace(0, alpha * (1 - confidence))), 0, 1)),
        "cauchy":      lambda: float(np.clip(abs(np.random.standard_cauchy()) * alpha * (1 - confidence), 0, 1)),
        "chi_squared": lambda: float(np.clip(np.random.chisquare(max(1, int(confidence * 10))) * alpha / 10, 0, 1)),
        "student_t":   lambda: float(np.clip(abs(np.random.standard_t(max(1, int(confidence * 10)))) * alpha * (1 - confidence), 0, 1)),
        "dirichlet":   lambda: float(np.random.dirichlet([confidence * 10 + 1e-9, alpha * 10 + 1e-9])[1]),
        "multinomial": lambda: float(np.random.multinomial(10, np.array([confidence, alpha, max(1 - confidence - alpha, 1e-9)]) / (confidence + alpha + max(1 - confidence - alpha, 1e-9)))[1] / 10),
    }
    effective_alpha = pdf_alpha_map[pdf]()
    return _semantic_perturb(value, alpha=effective_alpha, confidence=confidence)


SEMANTIC_TEXT_PDF_REGISTRY = {k: k for k in [
    "uniform", "normal", "beta", "exponential", "zipf", "bernoulli",
    "poisson", "log_normal", "gamma", "binomial", "pareto", "weibull",
    "laplace", "cauchy", "chi_squared", "student_t", "dirichlet", "multinomial",
]}


def perturb_text_semantic_column(
    *,
    df: pd.DataFrame,
    column: str,
    pdf: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if pdf not in SEMANTIC_TEXT_PDF_REGISTRY:
        raise ValueError(f"PDF '{pdf}' not supported. Choose from: {list(SEMANTIC_TEXT_PDF_REGISTRY.keys())}")
    return df[column].apply(
        lambda v: _semantic_perturb_with_pdf(v, alpha=alpha, confidence=confidence, pdf=pdf)
    )