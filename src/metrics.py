"""Metrics computation and aggregation across LOBO folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def clip_predictions(y_pred: np.ndarray, min_val: float = 0.0) -> np.ndarray:
    """Clip predictions to ensure they are >= min_val (default 0)."""
    return np.clip(y_pred, min_val, None)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MAE, RMSE, RPD, and R².

    Args:
        y_true: ground-truth values.
        y_pred: predicted values.
        mask: optional boolean array (same length as ``y_true``). When provided,
            metrics are computed only on the samples where ``mask`` is True — e.g.
            only samples whose target has a reference measurement (paper protocol).
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).flatten()
        if mask.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"mask length ({mask.shape[0]}) does not match y_true length ({y_true.shape[0]})"
            )
        if not mask.any():
            raise ValueError("mask selects no samples; cannot compute metrics")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "rpd": float(np.std(y_true) / np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def aggregate_metrics(
    fold_results: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Aggregate metrics across folds: mean ± std per model per metric.

    Handles both flat structure (single target) and nested structure (multi-target)
    where fold_results[0][model] may contain "target_metrics" with per-target metrics.
    """
    models = sorted(fold_results[0].keys())
    agg: dict[str, dict[str, tuple[float, float]]] = {}

    for model in models:
        agg[model] = {}
        for metric in ("mae", "rmse", "r2", "rpd"):
            values = np.array([f[model][metric] for f in fold_results])
            agg[model][metric] = (float(np.mean(values)), float(np.std(values)))

        # Also aggregate per-target metrics if present
        if "target_metrics" in fold_results[0][model]:
            target_names = fold_results[0][model]["target_metrics"].keys()
            agg[model]["target_metrics"] = {}
            for target_name in target_names:
                agg[model]["target_metrics"][target_name] = {}
                for metric in ("mae", "rmse", "r2", "rpd"):
                    values = np.array([f[model]["target_metrics"][target_name][metric] for f in fold_results])
                    agg[model]["target_metrics"][target_name][metric] = (float(np.mean(values)), float(np.std(values)))

    return agg


def print_results(agg: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Print formatted results table to stdout."""
    header = f"{'Model':<12} | {'MAE':>12} | {'RMSE':>12} | {'R²':>12} | {'RPD':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for model in sorted(agg.keys()):
        mae_mean, mae_std = agg[model]["mae"]
        rmse_mean, rmse_std = agg[model]["rmse"]
        r2_mean, r2_std = agg[model]["r2"]
        rpd_mean, rpd_std = agg[model]["rpd"]
        print(f"{model:<12} | {mae_mean:>10.4f} ± {mae_std:<6.4f} | "
              f"{rmse_mean:>10.4f} ± {rmse_std:<6.4f} | "
              f"{r2_mean:>10.4f} ± {r2_std:<6.4f} | "
              f"{rpd_mean:>10.4f} ± {rpd_std:<6.4f}")
        # Print per-target metrics if available
        if "target_metrics" in agg[model]:
            print(f"  Per-Target:")
            for target_name, m in agg[model]["target_metrics"].items():
                rmse_m, rmse_s = m["rmse"]
                r2_m, r2_s = m["r2"]
                print(f"    {target_name}: RMSE={rmse_m:.4f} ± {rmse_s:.4f}, R2={r2_m:.4f} ± {r2_s:.4f}")
    print(sep)


def save_results(
    fold_results: list[dict[str, dict]],
    agg: dict[str, dict],
    output_dir: str | Path = "outputs",
) -> None:
    """Save fold-level and aggregate results to JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "fold_results.json", "w") as f:
        json.dump(fold_results, f, indent=2, default=str)

    # Convert tuples to lists for JSON serialization
    agg_json = {}
    for model, metrics in agg.items():
        model_json = {}
        for m, v in metrics.items():
            if isinstance(v, dict) and "mae" in v:  # nested per-target metrics
                model_json[m] = {
                    target: list(values) if isinstance(values, tuple) else values
                    for target, values in v.items()
                }
            else:
                model_json[m] = list(v) if isinstance(v, tuple) else v
        agg_json[model] = model_json
    with open(out / "aggregate_results.json", "w") as f:
        json.dump(agg_json, f, indent=2)
