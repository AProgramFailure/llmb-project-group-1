from typing import Literal, Optional
from dataclasses import dataclass, field
import pandas as pd

Operation = Literal["classification", "reasoning", "extraction"]
PerturbationStrategy = Literal["typo", "paraphrase", "noise", "reorder", "truncate"]
PDF = Literal[
    "uniform", "normal", "beta", "exponential", "zipf", "bernoulli",
    "poisson", "log_normal", "gamma", "binomial", "pareto", "weibull",
    "laplace", "cauchy", "chi_squared", "student_t", "dirichlet", "multinomial",
]


@dataclass
class PerturbationConfig:
    confidence: float                        
    alpha: float                             
    perturb_metadata: bool = False           
    columns: Optional[list[str]] = None


def _validate_config(config: PerturbationConfig) -> None:
    if not (0.0 <= config.confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {config.confidence}")
    if not (0.0 <= config.alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {config.alpha}")


def _resolve_columns(
    df_ground_truth: pd.DataFrame,
    df_perturbed: Optional[pd.DataFrame],
    gt_columns: Optional[list[str]],
    perturbed_columns: Optional[list[str]],
    config: Optional[PerturbationConfig],
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (gt_columns, perturbed_columns, metadata_columns)
    metadata_columns = columns in perturbed not in gt
    """

    # single df path
    if df_perturbed is None:
        cols = config.columns if (config and config.columns) else list(df_ground_truth.columns)
        missing = [c for c in cols if c not in df_ground_truth.columns]
        if missing:
            raise ValueError(f"columns not found in ground truth df: {missing}")
        return cols, cols, []

    gt_cols   = list(df_ground_truth.columns)
    pert_cols = list(df_perturbed.columns)

    if gt_columns is None and perturbed_columns is None:
        gt_set   = set(gt_cols)
        pert_set = set(pert_cols)

        shared        = [c for c in gt_cols if c in pert_set]   # preserve gt order
        metadata_cols = [c for c in pert_cols if c not in gt_set]

        if not shared:
            raise ValueError(
                "No overlapping columns between ground truth and perturbed DataFrames. "
                "Provide gt_columns and perturbed_columns to map them manually.\n"
                f"  Ground truth columns : {gt_cols}\n"
                f"  Perturbed columns    : {pert_cols}"
            )

        # perturbed has fewer columns than gt — ok, use shared only
        return shared, shared, metadata_cols

    # manual mapping
    resolved_gt   = gt_columns   if gt_columns   is not None else list(df_ground_truth.columns)
    resolved_pert = perturbed_columns if perturbed_columns is not None else resolved_gt

    if len(resolved_gt) != len(resolved_pert):
        raise ValueError(
            f"gt_columns and perturbed_columns must have the same length. "
            f"Got {len(resolved_gt)} and {len(resolved_pert)}."
        )

    metadata_cols = [c for c in pert_cols if c not in set(resolved_pert)]
    return resolved_gt, resolved_pert, metadata_cols


def evaluate(
    *,
    df_ground_truth: pd.DataFrame,
    df_perturbed: Optional[pd.DataFrame] = None,
    operation: Operation = "classification",
    gt_columns: Optional[list[str]] = None,
    perturbed_columns: Optional[list[str]] = None,
    perturbation_strategy: Optional[PerturbationStrategy] = None,
    pdf: Optional[PDF] = None,
    config: Optional[PerturbationConfig] = None,
) -> pd.DataFrame:

    # --- config validation ---
    if config is not None:
        _validate_config(config)

    # --- single df guards ---
    if df_perturbed is None:
        if perturbation_strategy is None and pdf is None:
            raise ValueError(
                "When only df_ground_truth is provided, either perturbation_strategy or pdf must be set.\n"
                f"  perturbation_strategy options : {PerturbationStrategy.__args__}\n"
                f"  pdf options                   : {PDF.__args__}"
            )
        if pdf is not None and config is None:
            raise ValueError("config (PerturbationConfig) is required when pdf is set.")

    # --- empty df guards ---
    if df_ground_truth.empty:
        raise ValueError("df_ground_truth is empty.")
    if df_perturbed is not None and df_perturbed.empty:
        raise ValueError("df_perturbed is empty.")

    # --- resolve columns ---
    resolved_gt, resolved_pert, metadata_cols = _resolve_columns(
        df_ground_truth, df_perturbed, gt_columns, perturbed_columns, config
    )

    # --- validate resolved columns exist ---
    missing_gt = [c for c in resolved_gt if c not in df_ground_truth.columns]
    if missing_gt:
        raise ValueError(f"Columns not found in ground truth df: {missing_gt}")

    if df_perturbed is not None:
        missing_p = [c for c in resolved_pert if c not in df_perturbed.columns]
        if missing_p:
            raise ValueError(f"Columns not found in perturbed df: {missing_p}")

    # --- slice ---
    gt = df_ground_truth[resolved_gt].copy()

    if df_perturbed is not None:
        perturbed_core = df_perturbed[resolved_pert].copy()
        metadata       = df_perturbed[metadata_cols].copy() if metadata_cols else pd.DataFrame()
    else:
        perturbed_core = gt.copy()   # will be perturbed in place below
        metadata       = pd.DataFrame()

    # --- length alignment ---
    min_len = min(len(gt), len(perturbed_core))
    gt             = gt.iloc[:min_len].reset_index(drop=True)
    perturbed_core = perturbed_core.iloc[:min_len].reset_index(drop=True)
    if not metadata.empty:
        metadata = metadata.iloc[:min_len].reset_index(drop=True)

    # --- generate perturbation if only gt provided ---
    if df_perturbed is None and pdf is not None:
        # TODO: route each column to the correct perturber based on dtype
        pass

    # --- optionally perturb metadata ---
    if not metadata.empty and config is not None and config.perturb_metadata and pdf is not None:
        # TODO: route metadata columns to correct perturbers based on dtype
        pass

    # --- operation-specific evaluation ---
    if operation == "classification":
        pass  # TODO: classification evaluator
    elif operation == "reasoning":
        pass  # TODO: reasoning evaluator
    elif operation == "extraction":
        pass  # TODO: extraction evaluator

    # --- assemble results ---
    results = pd.DataFrame()

    if not metadata.empty:
        results = pd.concat([results, metadata], axis=1)

    return results