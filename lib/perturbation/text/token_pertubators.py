import numpy as np
import pandas as pd
import random


def _tokenize(text: str) -> list[str]:
    return text.split()


def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def _token_corrupt(tokens: list[str], n_corruptions: int, vocab: list[str]) -> list[str]:
    if not tokens:
        return tokens
    result = tokens[:]
    ops = ["swap", "delete", "insert", "replace", "repeat"]
    for _ in range(n_corruptions):
        if not result:
            break
        op = random.choice(ops)
        idx = random.randint(0, len(result) - 1)
        if op == "swap" and len(result) > 1:
            j = random.randint(0, len(result) - 1)
            result[idx], result[j] = result[j], result[idx]
        elif op == "delete" and len(result) > 1:
            result.pop(idx)
        elif op == "insert":
            result.insert(idx, random.choice(vocab))
        elif op == "replace":
            result[idx] = random.choice(vocab)
        elif op == "repeat":
            result.insert(idx, result[idx])
    return result


def _build_vocab(series: pd.Series) -> list[str]:
    vocab = set()
    for text in series.dropna():
        vocab.update(text.split())
    return list(vocab) if vocab else ["[UNK]"]


def perturb_text_token_uniform(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    n = np.random.randint(0, max(1, int(len(tokens) * alpha)) + 1)
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_normal(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    mean = len(tokens) * alpha
    n = int(np.clip(np.random.normal(mean, mean * (1 - confidence)), 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_beta(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    a, b = confidence * 10, alpha * 10 + 1e-9
    rate = np.random.beta(a, b)
    n = int(len(tokens) * rate)
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_exponential(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    scale = len(tokens) * alpha * (1 - confidence)
    n = int(np.clip(np.random.exponential(scale), 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_zipf(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    a = 1 + confidence / (alpha + 1e-9)
    n = min(np.random.zipf(a), len(tokens))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_bernoulli(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    tokens = _tokenize(value)
    p_keep = confidence * (1 - alpha)
    result = []
    for t in tokens:
        if np.random.binomial(1, p_keep):
            result.append(t)
        else:
            result.append(random.choice(vocab))
    return _detokenize(result)

def perturb_text_token_poisson(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    lam = max(len(tokens) * alpha * (1 - confidence), 1e-9)
    n = int(np.clip(np.random.poisson(lam), 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_log_normal(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    sigma = alpha * (1 - confidence)
    rate = np.random.lognormal(0, sigma)
    n = int(np.clip(len(tokens) * rate, 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_gamma(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    shape = confidence * 10 + 1e-9
    scale = alpha * (1 - confidence) * len(tokens) / shape
    n = int(np.clip(np.random.gamma(shape, scale), 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_binomial(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    tokens = _tokenize(value)
    p = alpha * (1 - confidence)
    n = np.random.binomial(len(tokens), p)
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_pareto(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    shape = 1 + confidence / (alpha + 1e-9)
    rate = min(np.random.pareto(shape) * alpha, 1.0)
    n = int(len(tokens) * rate)
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_weibull(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    shape = 1 + confidence
    rate = np.random.weibull(shape) * alpha * (1 - confidence)
    n = int(np.clip(len(tokens) * rate, 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_laplace(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    scale = len(tokens) * alpha * (1 - confidence)
    n = int(np.clip(abs(np.random.laplace(0, scale)), 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_cauchy(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    scale = len(tokens) * alpha * (1 - confidence)
    n = int(np.clip(abs(np.random.standard_cauchy() * scale), 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_chi_squared(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    df = max(1, int(confidence * 10))
    n = int(np.clip(np.random.chisquare(df) * alpha, 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_student_t(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    df = max(1, int(confidence * 10))
    scale = len(tokens) * alpha * (1 - confidence)
    n = int(np.clip(abs(np.random.standard_t(df)) * scale, 0, len(tokens)))
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_dirichlet(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    concentration = [confidence * 10 + 1e-9, alpha * 10 + 1e-9]
    weights = np.random.dirichlet(concentration)
    n = int(len(tokens) * weights[1])
    return _detokenize(_token_corrupt(tokens, n, vocab))

def perturb_text_token_multinomial(value, *, vocab: list[str], alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return value
    tokens = _tokenize(value)
    probs = np.array([confidence, alpha, max(1 - confidence - alpha, 1e-9)])
    probs /= probs.sum()
    counts = np.random.multinomial(len(tokens), probs)
    n = counts[1]
    return _detokenize(_token_corrupt(tokens, n, vocab))


TOKEN_TEXT_PDF_REGISTRY: dict[str, callable] = {
    "uniform":     perturb_text_token_uniform,
    "normal":      perturb_text_token_normal,
    "beta":        perturb_text_token_beta,
    "exponential": perturb_text_token_exponential,
    "zipf":        perturb_text_token_zipf,
    "bernoulli":   perturb_text_token_bernoulli,
    "poisson":     perturb_text_token_poisson,
    "log_normal":  perturb_text_token_log_normal,
    "gamma":       perturb_text_token_gamma,
    "binomial":    perturb_text_token_binomial,
    "pareto":      perturb_text_token_pareto,
    "weibull":     perturb_text_token_weibull,
    "laplace":     perturb_text_token_laplace,
    "cauchy":      perturb_text_token_cauchy,
    "chi_squared": perturb_text_token_chi_squared,
    "student_t":   perturb_text_token_student_t,
    "dirichlet":   perturb_text_token_dirichlet,
    "multinomial": perturb_text_token_multinomial,
}


def perturb_text_token_column(
    *,
    df: pd.DataFrame,
    column: str,
    pdf: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if pdf not in TOKEN_TEXT_PDF_REGISTRY:
        raise ValueError(f"PDF '{pdf}' not supported. Choose from: {list(TOKEN_TEXT_PDF_REGISTRY.keys())}")
    vocab = _build_vocab(df[column])
    fn = TOKEN_TEXT_PDF_REGISTRY[pdf]
    return df[column].apply(lambda v: fn(v, vocab=vocab, alpha=alpha, confidence=confidence))