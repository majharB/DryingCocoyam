"""Visualization utilities for model comparison across LOBO folds.

Plots:
    - per-fold scatter plots of predictions vs ground truth
    - aggregated bar chart comparing metrics across models

All figures are saved as PNGs in the given output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ---------- constants -------------------------------------------------------

# Color palette borrowed from the existing codebase convention (Set1-like)
_MODEL_COLORS: list[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#ff9896",  # coral
    "#9edae5",  # sky
    "#98df8a",  # light green
    "#ffbb78",  # peach
    "#c49c6f",  # tan
    "#e7c6e8",  # lavender
]

_METRIC_ORDER = ["rmse", "mae", "r2", "rpd"]
_METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "rpd": "RPD",
}
_METRIC_XLIM: dict[str, tuple[float, float]] = {
    "rmse": (0, None),
    "mae": (0, None),
    "r2": (0, 1.05),
    "rpd": (0, None),
}


# ---------- helpers -----------------------------------------------------------

def _get_color(model: str) -> str:
    """Deterministic color assignment for a model name."""
    idx = hash(model) % len(_MODEL_COLORS)
    return _MODEL_COLORS[idx]


def _save_figure(fig: matplotlib.figure.Figure, name: str, output_dir: Path) -> Path:
    """Save figure to output_dir and return the path."""
    fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_dir / f"{name}.png"


# ---------- per-fold predictions plot ----------------------------------------


def plot_predictions_per_fold(
    fold_results: list[dict[str, dict[str, Any]]],
    y_true_global: np.ndarray,
    val_masks: list[np.ndarray],
    output_dir: Path,
    **kwargs: Any,
) -> list[Path]:
    """Plot predicted vs ground-truth moisture for each fold.

    For each fold a scatter plot is produced with one series per model.
    A reference y=x line is overlaid.

    Args:
        fold_results: list of dicts, each keyed by model name with
                      metrics including 'y_pred' and 'y_true' arrays.
        y_true_global: full ground-truth array (used for the y=x reference).
        val_masks: validation masks per fold (used to index y_true_global).
        output_dir: directory to save PNGs.
        **kwargs: ignored (passed through from callers that include them).

    Returns:
        List of saved figure paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    models = sorted(fold_results[0].keys())

    for fold_idx, fold_dict in enumerate(fold_results):
        fig, ax = plt.subplots(figsize=(6, 5))

        # Ground truth for this fold
        val_mask = val_masks[fold_idx]
        y_fold = y_true_global[val_mask]

        for i, model in enumerate(models):
            y_pred = fold_dict[model]["y_pred"]
            y_tru = fold_dict[model]["y_true"]
            color = _get_color(model)
            ax.scatter(y_tru, y_pred, c=color, alpha=0.6, s=8, label=model)

        # y=x reference line
        lims = max(y_fold.max(), y_fold.max()) if len(y_fold) > 0 else 100
        lims = (0, max(lims, 80))
        ax.plot(lims, lims, "--", c="gray", lw=1, label="y=x")

        ax.set_xlabel("Ground Truth (Moisture)")
        ax.set_ylabel("Predicted (Moisture)")
        ax.set_title(f"Fold {fold_idx + 1} — Predicted vs Ground Truth")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        p = output_dir / f"predictions_fold_{fold_idx + 1}.png"
        _save_figure(fig, f"predictions_fold_{fold_idx + 1}", output_dir)
        paths.append(p)

    return paths


# ---------- aggregated bar comparison ----------------------------------------


def plot_bar_comparison(
    agg: dict[str, dict[str, tuple[float, float]]],
    output_dir: Path,
) -> Path:
    """Grouped bar chart comparing aggregate metrics across models.

    Four subplots (RMSE, MAE, R2, RPD) are saved as a single figure.

    Args:
        agg: aggregated metrics dict e.g.
             {"HGP": {"rmse": (mean, std), "mae": (mean, std), ...}, ...}
        output_dir: directory to save PNG.

    Returns:
        Path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(agg.keys())
    metrics = [m for m in _METRIC_ORDER if m in next(iter(agg.values()))]
    n_models = len(models)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]

    bar_width = 0.7 / max(n_models, 1)

    for j, metric in enumerate(metrics):
        ax = axes[j]
        x = np.arange(n_models)

        for i, model in enumerate(models):
            mean_val, std_val = agg[model][metric]
            color = _get_color(model)
            ax.bar(
                x[i] - bar_width * (n_models - 1) / 2 + i * bar_width,
                mean_val,
                yerr=std_val,
                capsize=3,
                width=bar_width,
                color=color,
                alpha=0.85,
                label=model,
                edgecolor="white",
                linewidth=0.5,
            )

        label = _METRIC_LABELS.get(metric, metric.upper())
        xlim = _METRIC_XLIM.get(metric, (0, None))
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        if xlim[1] is not None:
            ax.set_xlim(*xlim)
        ax.grid(True, axis="y", alpha=0.3)

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(n_models, 1),
               fontsize=9, frameon=True, fancybox=True)

    fig.suptitle("Aggregated Model Comparison (mean +/- std)", fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    p = output_dir / "bar_comparison.png"
    _save_figure(fig, "bar_comparison", output_dir)
    return p
