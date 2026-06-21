import re
import numpy as np
import pandas as pd


_CONFIDENCE_LEVELS = [
    'virtually certain',
    'very high confidence',
    'high confidence',
    'medium confidence',
    'low confidence',
]
_CONF_PATTERN = '|'.join(re.escape(c) for c in sorted(_CONFIDENCE_LEVELS, key=len, reverse=True))

_ENTITY_TAXONOMY = {
    'scope_label': {
        'pattern': re.compile(
            r'\bScopes?\s+1(?:\s*[,&+]\s*2(?:\s*[,&+]\s*3)?)?\b'
            r'|\bScopes?\s+2\b'
            r'|\bScopes?\s+3\b',
            re.IGNORECASE,
        ),
        'pool': ['Scope 1', 'Scope 2', 'Scope 3', 'Scope 1 & 2', 'Scope 1, 2 & 3'],
        'rule': 'random_other',
    },
    'fiscal_year': {
        'pattern': re.compile(r'\bFY\s*(\d{4})\b', re.IGNORECASE),
        'pool': None,
        'rule': 'shift_year',
    },
    'emission_unit': {
        'pattern': re.compile(r'\b(?:G|M|k)?tCO2e?\b'),
        'pool': ['tCO2e', 'ktCO2e', 'MtCO2e', 'GtCO2e'],
        'rule': 'random_other',
    },
    'currency': {
        'pattern': re.compile(r'\b(?:RMB|CNY|USD|EUR|GBP|JPY|AUD|CAD|SGD)\b'),
        'pool': ['RMB', 'USD', 'EUR', 'GBP', 'JPY'],
        'rule': 'random_other',
    },
    'emission_qualifier': {
        'pattern': re.compile(r'\b(?:net|gross)\s+emissions?\b', re.IGNORECASE),
        'pool': ['net emissions', 'gross emissions'],
        'rule': 'random_other',
    },
    'magnitude_unit': {
        'pattern': re.compile(
            r'\b(?:billion|million|thousand)\s+(?:metric\s+)?tons?\b', re.IGNORECASE
        ),
        'pool': ['billion tons', 'million tons', 'thousand tons'],
        'rule': 'random_other',
    },
    'scope2_method': {
        'pattern': re.compile(r'\b(?:market[- ]based|location[- ]based)\b', re.IGNORECASE),
        'pool': ['market-based', 'location-based'],
        'rule': 'random_other',
    },
}

_ENTITY_LEVELS = {
    1: ['scope_label', 'fiscal_year'],
    2: ['scope_label', 'fiscal_year', 'emission_unit', 'currency'],
    3: ['scope_label', 'fiscal_year', 'emission_unit', 'currency',
        'emission_qualifier', 'magnitude_unit', 'scope2_method'],
}


def _swap_confidence(text: str, alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return text
    adjacent_only = alpha < 0.20
    def _replace(m):
        original = m.group().lower()
        if original not in _CONFIDENCE_LEVELS:
            return m.group()
        idx = _CONFIDENCE_LEVELS.index(original)
        if adjacent_only:
            neighbours = []
            if idx > 0:
                neighbours.append(_CONFIDENCE_LEVELS[idx - 1])
            if idx < len(_CONFIDENCE_LEVELS) - 1:
                neighbours.append(_CONFIDENCE_LEVELS[idx + 1])
            return np.random.choice(neighbours) if neighbours else m.group()
        candidates = [c for c in _CONFIDENCE_LEVELS if c != original]
        return np.random.choice(candidates)
    return re.sub(_CONF_PATTERN, _replace, str(text), flags=re.IGNORECASE)


def _swap_entities(text: str, alpha: float, confidence: float) -> str:
    if np.random.random() < confidence:
        return text
    level = 1 if alpha < 0.20 else (2 if alpha < 0.40 else 3)
    result = str(text)
    for entity_type in _ENTITY_LEVELS[level]:
        defn = _ENTITY_TAXONOMY[entity_type]
        if defn['rule'] == 'random_other':
            pool = defn['pool']
            def replace_entity(m, pool=pool):
                original = m.group().lower()
                candidates = [p for p in pool if p.lower() != original]
                return np.random.choice(candidates) if candidates else m.group()
            result = defn['pattern'].sub(replace_entity, result)
        elif defn['rule'] == 'shift_year':
            shift = level
            def replace_year(m, shift=shift):
                year = int(m.group(1))
                direction = np.random.choice([-1, 1])
                return f"FY{year + direction * shift}"
            result = defn['pattern'].sub(replace_year, result)
    return result


def perturb_esg_confidence_column(
    *,
    df: pd.DataFrame,
    column: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    """
    Replace IPCC confidence phrases with a different level from the same scale.

    Row gate: confidence prob of leaving a row unchanged.
    alpha < 0.20: only adjacent steps (e.g. 'high confidence' -> 'very high' or 'medium').
    alpha >= 0.20: any other phrase from the scale.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    return df[column].apply(lambda v: _swap_confidence(v, alpha, confidence))


def perturb_esg_entity_column(
    *,
    df: pd.DataFrame,
    column: str,
    alpha: float,
    confidence: float,
) -> pd.Series:
    """
    Replace ESG domain entities using a fixed taxonomy.

    Row gate: confidence prob of leaving a row unchanged.
    alpha < 0.20: scope labels + fiscal years.
    alpha < 0.40: + emission units + currencies.
    alpha >= 0.40: + emission qualifiers + magnitude units + Scope 2 accounting method.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    return df[column].apply(lambda v: _swap_entities(v, alpha, confidence))
