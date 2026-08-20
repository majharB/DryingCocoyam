"""Retrain HSI 2-Step CNNLSTM model for all LOBO folds using saved hyperparameters.

CNNLSTMHSI2S is a CUSTOM deep learning architecture trained from scratch (NOT a foundation model).
This script retrains the 2-step sequential HSI-based model for each fold using hyperparameters
tuned via Optuna.

The 2-step model takes HSI and metadata from steps t-1 and t, then predicts moisture content
at t+1.

Usage:
    source coco-env.sh
    python src/train/train_cnn_lstm_hsi_2s.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    load_dataset_hsi_2s,
    load_ground_truth,
    get_lobo_folds_with_inner_val,
    reference_mask,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_fold_config(project_root: Path) -> dict:
    """Load fold configuration from JSON file.

    Args:
        project_root: Project root directory

    Returns:
        Dict mapping fold_name -> fold info with inner_val_batches
    """
    config_path = project_root / "outputs" / "hsi_2s" / "fold_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


from src.data_loader import (
    load_dataset_hsi_2s,
    load_ground_truth,
    get_lobo_folds_with_inner_val,
    reference_mask,
)
from src.metrics import compute_metrics, aggregate_metrics, print_results, save_results
from src.visualize import plot_predictions_per_fold, plot_bar_comparison
from src.cnn_lstm_model import CNNLSTMHSI2SModel, load_fold_hyperparams

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "hsi_2s"
HYPERPARAMS_DIR = OUTPUT_DIR / "hyperparams"

TARGET_NAMES = ["moisture"]


def train_fold(
    fold_idx: int,
    X_hsi_inner_tr: np.ndarray,
    y_inner_tr: np.ndarray,
    X_meta_inner_tr: np.ndarray,
    X_hsi_inner_val: np.ndarray,
    y_inner_val: np.ndarray,
    X_meta_inner_val: np.ndarray,
    X_hsi_test: np.ndarray,
    y_test: np.ndarray,
    X_meta_test: np.ndarray,
    project_root: Path,
    ref_test_mask: np.ndarray | None = None,
    ref_inner_val_mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Train 2-step HSI CNNLSTM for one fold using saved hyperparameters.

    The model is trained on inner-train data with early stopping on inner-val.
    Final evaluation is done on the test fold.

    Args:
        fold_idx: 0-based fold index
        X_hsi_inner_tr: Inner training HSI images (used for training)
        y_inner_tr: Inner training targets
        X_meta_inner_tr: Inner training metadata
        X_hsi_inner_val: Inner validation HSI images (used for early stopping)
        y_inner_val: Inner validation targets
        X_meta_inner_val: Inner validation metadata
        X_hsi_test: Test fold HSI images (for final evaluation)
        y_test: Test fold targets
        X_meta_test: Test fold metadata
        project_root: Project root directory

    Returns:
        Dictionary with metrics and predictions
    """
    # Load best hyperparameters
    params = load_fold_hyperparams(fold_idx + 1, project_root=project_root)

    # Build model with best params
    model = CNNLSTMHSI2SModel(
        n_bands=112,
        hsi_h=64,
        hsi_w=52,
        meta_dim_per_step=3,  # 3 metadata features per timestep
        seq_length=2,
        n_conv_layers=params.get("n_conv_layers", 3),
        conv_filters=params.get("conv_filters", 64),
        n_lstm_layers=params.get("n_lstm_layers", 2),
        lstm_units=params.get("lstm_units", 128),
        dropout_rate=params.get("dropout_rate", 0.3),
        learning_rate=params.get("learning_rate", 0.001),
        batch_size=params.get("batch_size", 16),
        epochs=params.get("epochs", 100),
        early_stopping_patience=params.get("early_stopping_patience", 15),
        random_state=50,
    )

    # Train with early stopping on inner validation data
    model.fit(
        X_hsi_inner_tr, y_inner_tr, X_meta_inner_tr,
        validation_data=(X_hsi_inner_val, y_inner_val, X_meta_inner_val),
        verbose=1,
    )

    # Evaluate on TEST fold (the held-out batch)
    y_pred = model.predict(X_hsi_test, X_meta_test)

    # Compute metrics for moisture content on test fold
    metrics = {}
    y_true_target = y_test.flatten()
    y_pred_target = y_pred.flatten()
    target_metrics = compute_metrics(y_true_target, y_pred_target, mask=ref_test_mask)
    metrics["moisture_mae"] = target_metrics["mae"]
    metrics["moisture_rmse"] = target_metrics["rmse"]
    metrics["moisture_r2"] = target_metrics["r2"]
    metrics["moisture_rpd"] = target_metrics["rpd"]
    metrics["n_test"] = int(len(y_true_target))
    metrics["n_test_reference"] = int(ref_test_mask.sum()) if ref_test_mask is not None else None

    # Store predictions for visualization
    metrics["y_pred"] = y_pred
    metrics["y_true"] = y_test

    # Save model weights and architecture
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"fold_{fold_idx + 1}.h5"
    arch_path = model_dir / f"fold_{fold_idx + 1}_arch.json"
    if model._model is not None:
        model._model.save(model_path)
        with open(arch_path, "w") as f:
            f.write(model._model.to_json())
        logger.info("  Saved model to %s", model_path)
        logger.info("  Saved arch to %s", arch_path)

    # Log inner val performance (for reference)
    inner_val_pred = model.predict(X_hsi_inner_val, X_meta_inner_val)
    inner_val_metrics = compute_metrics(y_inner_val.flatten(), inner_val_pred.flatten(), mask=ref_inner_val_mask)
    logger.info("  Inner-val RMSE: %.4f (used for early stopping)", inner_val_metrics["rmse"])
    logger.info("  Test fold RMSE: %.4f (final evaluation)", metrics["moisture_rmse"])

    # Save metrics to JSON file
    # metrics_path = OUTPUT_DIR / f"fold_{fold_idx + 1}_metrics.json"
    # with open(metrics_path, "w") as f:
    #     json.dump(metrics, f, indent=2)
    # Save metrics to JSON file (exclude non-serializable prediction arrays)
    metrics_to_save = {
        k: v for k, v in metrics.items() if k not in ("y_pred", "y_true")
    }
    metrics_path = OUTPUT_DIR / f"fold_{fold_idx + 1}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    logger.info("  Saved metrics to %s", metrics_path)

    return metrics



def run_training(
    folds: list[int] = None,              # List of fold indices to train (1-16)
    load_existing: bool = True,           # Skip folds if model already exists
    aggregate: bool = True,               # Aggregate metrics when training all folds
) -> dict[str, dict[str, tuple[float, float]]]:
    """Run full training pipeline: LOBO CV with 2-step HSI CNNLSTM.

    Args:
        folds: List of fold numbers to train (1-16). If None, trains all folds.
        load_existing: If True, skip training for folds where model already exists.
        aggregate: If True, aggregate metrics and save aggregate results.

    Returns:
        Dictionary with aggregated metrics per target.
    """
    logger.info("Loading 2-step HSI dataset...")
    X_hsi, X_meta, _, y, batch_ids, sample_ids, slice_info = load_dataset_hsi_2s()
    logger.info("Dataset: %d sequences, X_hsi=%s, X_meta=%s, y=%s",
                len(y), X_hsi.shape, X_meta.shape, y.shape)

    # Reference mask: True where a sequence's target (t+1) sample has a reference
    # measurement. Reported accuracy metrics use only these samples (paper protocol).
    ref_mask = reference_mask(sample_ids)
    logger.info("Sequences with a reference target: %d / %d", int(ref_mask.sum()), len(ref_mask))

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)
    (OUTPUT_DIR / "hyperparams").mkdir(parents=True, exist_ok=True)

    # Load fold configuration with intelligent inner validation batch selection
    project_root = Path(__file__).resolve().parents[1]
    fold_config = load_fold_config(project_root)

    # Create output directories and model directory reference
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)
    (OUTPUT_DIR / "hyperparams").mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / "models"

    all_fold_results: list[dict[str, dict[str, float]]] = []
    val_masks: list[np.ndarray] = []

    # Determine which folds to train
    if folds is None:
        # Default: train all folds (backward compatible)
        folds = list(range(1, 17))
        logger.info("No folds specified - training all 16 folds")
    else:
        # Validate fold numbers
        valid_folds = [f for f in folds if 1 <= f <= 16]
        if valid_folds != folds:
            logger.warning(f"Invalid fold numbers {folds}. Using valid folds: {valid_folds}")
            folds = valid_folds
        logger.info(f"Training folds: {folds}")

    for fold_num in folds:
        fold_idx = fold_num - 1  # Convert to 0-based index
        fold_name = f"fold_{fold_num}"
        fold_info = fold_config[fold_name]

        val_batch = fold_info["val_batch"]
        inner_val_batches = fold_info["inner_val_batches"]
        inner_train_batches = fold_info["inner_train_batches"]

        logger.info("=" * 60)
        logger.info("Fold %d (val_batch=%d, inner_val_batches=%s):",
                    fold_idx + 1,
                    val_batch,
                    inner_val_batches)

        # Create masks based on batch IDs
        val_mask = batch_ids == val_batch
        inner_train_mask = np.isin(batch_ids, inner_train_batches)
        inner_val_mask = np.isin(batch_ids, inner_val_batches)

        # Restrict reported accuracy to samples whose target has a reference.
        ref_test = ref_mask[val_mask]
        ref_inner_val = ref_mask[inner_val_mask]

        # Verify masks are mutually exclusive and cover training data
        assert not np.any(inner_train_mask & inner_val_mask), "Inner train/val overlap!"
        # Note: inner_train + inner_val should cover all NON-test sequences
        training_sequences = ~val_mask
        assert np.all((inner_train_mask | inner_val_mask) == training_sequences), \
            "Inner masks don't cover all training (excluding test fold)!"

        # Get data for inner validation (used in training)
        X_hsi_inner_tr = X_hsi[inner_train_mask]
        y_inner_tr = y[inner_train_mask]
        X_meta_inner_tr = X_meta[inner_train_mask]

        X_hsi_inner_val = X_hsi[inner_val_mask]
        y_inner_val = y[inner_val_mask]
        X_meta_inner_val = X_meta[inner_val_mask]

        # Get data for test fold evaluation
        X_hsi_test = X_hsi[val_mask]
        y_test = y[val_mask]
        X_meta_test = X_meta[val_mask]

        # Check if model already exists
        model_path = model_dir / f"fold_{fold_num}.h5"
        if load_existing and model_path.exists():
            logger.info(f"Skipping fold {fold_num} - model already exists at {model_path}")
            # Load existing metrics
            metrics_path = OUTPUT_DIR / f"fold_{fold_num}_metrics.json"
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            # Remove predictions from loaded metrics (not needed)
            if "y_pred" in metrics:
                del metrics["y_pred"]
            if "y_true" in metrics:
                del metrics["y_true"]
        else:
            # Train model
            metrics = train_fold(
                fold_idx,
                X_hsi_inner_tr, y_inner_tr, X_meta_inner_tr,
                X_hsi_inner_val, y_inner_val, X_meta_inner_val,
                X_hsi_test, y_test, X_meta_test,
                project_root=project_root,
                ref_test_mask=ref_test,
                ref_inner_val_mask=ref_inner_val,
            )

        all_fold_results.append({"CNNLSTMHSI_2S": metrics})
        val_masks.append(val_mask)

        # Print per-target metrics
        logger.info("Fold %d results:", fold_idx + 1)
        for name in TARGET_NAMES:
            rmse = metrics[f"{name}_rmse"]
            r2 = metrics[f"{name}_r2"]
            logger.info("  %s: RMSE=%.4f, R²=%.4f", name, rmse, r2)

    # Aggregate metrics per target (only if aggregating)
    agg = None
    if aggregate and len(all_fold_results) > 1:
        agg = aggregate_metrics_per_target(all_fold_results)
        print()
        print_results_per_target(agg)

        # Save results
        save_results(all_fold_results, agg, OUTPUT_DIR)
        plot_predictions_per_fold(all_fold_results, y, val_masks, OUTPUT_DIR, multioutput=False)
        plot_bar_comparison(agg, OUTPUT_DIR)
        logger.info("Aggregate results saved to %s/", OUTPUT_DIR)

    logger.info("Individual fold results saved for each trained fold")

    # Save hyperparameters used
    hyperparams_used = {}
    for fold_idx in range(1, 17):
        try:
            hyperparams_used[f"fold_{fold_idx}"] = load_fold_hyperparams(fold_idx)
        except FileNotFoundError:
            hyperparams_used[f"fold_{fold_idx}"] = "Not found"

    hyperparams_path = OUTPUT_DIR / "hyperparams_used.json"
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams_used, f, indent=2)

    logger.info("Hyperparameters saved to %s", hyperparams_path)

    return agg


def aggregate_metrics_per_target(
    fold_results: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Aggregate metrics per target: mean ± std across folds."""
    agg: dict[str, dict[str, tuple[float, float]]] = {}

    for target in TARGET_NAMES:
        agg[target] = {}
        for metric in ("mae", "rmse", "r2", "rpd"):
            values = np.array([f["CNNLSTMHSI_2S"][f"{target}_{metric}"] for f in fold_results])
            agg[target][metric] = (float(np.mean(values)), float(np.std(values)))

    return agg


def print_results_per_target(agg: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Print formatted results table per target."""
    header = f"{'Target':<12} | {'MAE':>12} | {'RMSE':>12} | {'R²':>12} | {'RPD':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for target in TARGET_NAMES:
        mae_mean, mae_std = agg[target]["mae"]
        rmse_mean, rmse_std = agg[target]["rmse"]
        r2_mean, r2_std = agg[target]["r2"]
        rpd_mean, rpd_std = agg[target]["rpd"]
        print(f"{target:<12} | {mae_mean:>10.4f} ± {mae_std:<6.4f} | "
              f"{rmse_mean:>10.4f} ± {rmse_std:<6.4f} | "
              f"{r2_mean:>10.4f} ± {r2_std:<6.4f} | "
              f"{rpd_mean:>10.4f} ± {rpd_std:<6.4f}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 2-step HSI CNNLSTM for LOBO cross-validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all folds (default)
  python train_cnn_lstm_hsi_2s.py

  # Train only fold 5
  python train_cnn_lstm_hsi_2s.py --folds 5

  # Train folds 1, 5, 10
  python train_cnn_lstm_hsi_2s.py --folds 1,5,10

  # Train folds 5 and 10 only
  python train_cnn_lstm_hsi_2s.py --folds 5,10
        """
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated list of fold numbers to train (1-16). "
             "If None, trains all folds. Example: --folds 5 or --folds 1,5,10"
    )
    parser.add_argument(
        "--no-load-existing",
        action="store_true",
        help="Force retraining even if model already exists "
             "(default: skip training if model exists)"
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip aggregation and plotting (useful for single-fold training)"
    )

    args = parser.parse_args()

    # Parse folds argument
    if args.folds:
        folds = [int(f.strip()) for f in args.folds.split(",")]
    else:
        folds = None

    # Set load_existing flag
    load_existing = not args.no_load_existing

    # Set aggregate flag
    aggregate = not args.no_aggregate

    # Run training
    run_training(
        folds=folds,
        load_existing=load_existing,
        aggregate=aggregate,
    )
