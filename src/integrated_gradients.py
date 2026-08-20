"""SWING (Shifted Window Integrated Gradients) for CNN-LSTM HSI 2-step model.

Replaces LRP and standard IG with SWING, which uses the previous time-step
observation as the integration baseline rather than a fixed zero or mean vector.
This produces temporally coherent attributions that explain *changes* in moisture
prediction rather than absolute prediction levels — physically meaningful for a
drying process.

SWING formula (per sample):
    SWING_i(x_t) = (x_i^(t) - x_i^(t-1)) * integral_0^1 [df/dx_i evaluated at
                   x^(t-1) + alpha*(x^(t) - x^(t-1))] d_alpha

where the integration path shifts with the time step: the baseline is always
the previous observation, not a fixed reference.

Key differences from existing run_xai_pipeline.py:
    - Baseline per sample = previous two-step window (x[t-2], x[t-1]) instead of zeros
    - Attribution explains delta-prediction, not absolute prediction
    - Metadata channels (tau, T, V) are attributed end-to-end alongside HSI
    - Methods are fully separated: SWING never mixes with gradient saliency
    - No leakage: attributions computed on inner-val split only, never test fold

Usage:
    source coco-env.sh
    TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 \\
        python src/xai/swing_attribution.py [--folds 1-16] [--n_steps 50] [--batch_size 4]

Outputs (per fold):
    outputs/xai/swing_hsi_2s/fold_{f}_wavelength_importance.csv
    outputs/xai/swing_hsi_2s/fold_{f}_pixel_importance.csv
    outputs/xai/swing_hsi_2s/fold_{f}_metadata_importance.csv
    outputs/xai/swing_hsi_2s/fold_{f}_timestep_split.csv

Outputs (global):
    outputs/xai/swing_hsi_2s/global_wavelength_importance.csv
    outputs/xai/swing_hsi_2s/global_pixel_importance.csv
    outputs/xai/swing_hsi_2s/global_metadata_importance.csv
    outputs/xai/swing_hsi_2s/global_timestep_split.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    load_dataset_hsi_2s,
    lobos_folds_for,
    get_lobo_folds_with_inner_val,
    reference_mask,
)
from src.cnn_lstm_model import CNNLSTMHSI2SModel, load_fold_hyperparams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "xai" / "swing_hsi_2s"
MODEL_DIR    = PROJECT_ROOT / "outputs" / "hsi_2s" / "models"
CONFIG_DIR   = PROJECT_ROOT / "configs" / "hsi_2s"

IMG_H    = 64
IMG_W    = 52
N_BANDS  = 112
T_STEPS  = 2
META_DIM = 6                         # [tau_{t-1}, T_{t-1}, V_{t-1}, tau_t, T_t, V_t]
WAVELENGTHS = [937.33,944.25,951.16,958.08,965,971.92,978.85,985.77,992.7,999.63,1006.57,1013.5,1020.44,1027.38,1034.32,1041.27,1048.21,1055.16,1062.12,1069.07,1076.03,1082.98,1089.94,1096.91,1103.87,1110.84,1117.81,1124.78,1131.75,1138.73,1145.71,1152.69,1159.67,1166.66,1173.64,1180.63,1187.63,1194.62,1201.62,1208.62,1215.62,1222.62,1229.63,1236.63,1243.64,1250.66,1257.67,1264.69,1271.71,1278.73,1285.75,1292.78,1299.8,1306.83,1313.87,1320.9,1327.94,1334.98,1342.02,1349.06,1356.11,1363.16,1370.21,1377.26,1384.31,1391.37,1398.43,1405.49,1412.56,1419.62,1426.69,1433.76,1440.84,1447.91,1454.99,1462.07,1469.15,1476.23,1483.32,1490.41,1497.5,1504.59,1511.69,1518.79,1525.89,1532.99,1540.09,1547.2,1554.31,1561.42,1568.53,1575.65,1582.77,1589.89,1597.01,1604.13,1611.26,1618.39,1625.52,1632.66,1639.79,1646.93,1654.07,1661.21,1668.36,1675.51,1682.65,1689.81,1696.96,1704.12,1711.28,1718.44]
WAVELENGTHS_INT = [int(wl) for wl in WAVELENGTHS]
# Grid for refined model retraining
TOP_K_VALUES      = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PIXEL_FRACTIONS   = [1]
INNER_VAL_BATCHES = 2     # number of training batches held out for inner validation
INNER_VAL_SEED    = 42


# ── Model loading ────────────────────────────────────────────────────────────

def load_fold_model(fold_idx: int) -> CNNLSTMHSI2SModel:
    """Load the trained LOBO model for fold *fold_idx* (1-based)."""
    model_path = MODEL_DIR / f"fold_{fold_idx}.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    params = load_fold_hyperparams(fold_idx)

    wrapper = CNNLSTMHSI2SModel(
        n_bands=params.get("n_bands", N_BANDS),
        hsi_h=IMG_H,
        hsi_w=IMG_W,
        meta_dim=META_DIM,
        seq_length=T_STEPS,
        n_conv_layers=params.get("n_conv_layers", 3),
        conv_filters=params.get("conv_filters", 64),
        n_lstm_layers=params.get("n_lstm_layers", 2),
        lstm_units=params.get("lstm_units", 128),
        dropout_rate=params.get("dropout_rate", 0.3),
        learning_rate=params.get("learning_rate", 1e-3),
        batch_size=params.get("batch_size", 16),
        epochs=params.get("epochs", 100),
        early_stopping_patience=params.get("early_stopping_patience", 15),
        random_state=50,
    )
    wrapper._model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "mse": tf.keras.losses.MeanSquaredError,
            "mae": tf.keras.losses.MeanAbsoluteError,
        },
    )
    logger.info("Loaded fold %d model from %s", fold_idx, model_path)
    return wrapper


# ── Inner-validation split ───────────────────────────────────────────────────

def make_inner_val_mask(
    batch_ids: np.ndarray,
    train_mask: np.ndarray,
    n_val_batches: int = INNER_VAL_BATCHES,
    seed: int = INNER_VAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the training fold into inner-train and inner-val at batch level.

    Args:
        batch_ids:    Array of batch IDs for all sequences.
        train_mask:   Boolean mask selecting the outer training fold.
        n_val_batches: How many training batches to hold out as inner-val.
        seed:         Random seed for reproducible inner split.

    Returns:
        (inner_train_mask, inner_val_mask) — both index into the full dataset.
    """
    rng = np.random.default_rng(seed)
    train_batches = np.unique(batch_ids[train_mask])
    val_batches   = rng.choice(train_batches, size=n_val_batches, replace=False)
    val_batches   = set(val_batches.tolist())

    inner_val_mask   = train_mask & np.isin(batch_ids, list(val_batches))
    inner_train_mask = train_mask & ~np.isin(batch_ids, list(val_batches))
    return inner_train_mask, inner_val_mask


# ── SWING core ───────────────────────────────────────────────────────────────

def _riemann_gradients(
    keras_model: tf.keras.Model,
    X_hsi_curr: np.ndarray,
    X_hsi_base: np.ndarray,
    X_meta_curr: np.ndarray,
    X_meta_base: np.ndarray,
    n_steps: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the SWING path integral via Riemann sum.

    Interpolates between baseline (previous window) and current input, evaluates
    the gradient of the model output w.r.t. the HSI input and the metadata input
    at each interpolation point, and averages.

    Args:
        keras_model:  Raw tf.keras.Model (wrapper._model).
        X_hsi_curr:   Current HSI input   (n, 2, H, W, B).
        X_hsi_base:   Baseline HSI input   (n, 2, H, W, B).
        X_meta_curr:  Current metadata     (n, meta_dim).
        X_meta_base:  Baseline metadata    (n, meta_dim).
        n_steps:      Number of integration steps.
        batch_size:   Samples processed per GPU call.

    Returns:
        (hsi_path_grads, meta_path_grads) — averaged gradients,
        shapes (n, 2, H, W, B) and (n, meta_dim).
    """
    n = X_hsi_curr.shape[0]
    hsi_delta  = X_hsi_curr  - X_hsi_base
    meta_delta = X_meta_curr - X_meta_base

    sum_hsi_grads  = np.zeros_like(X_hsi_curr,  dtype=np.float64)
    sum_meta_grads = np.zeros_like(X_meta_curr, dtype=np.float64)

    alphas = (np.arange(1, n_steps + 1) / n_steps).tolist()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        hsi_interp_all  = []
        meta_interp_all = []

        for alpha in alphas:
            hsi_interp_all.append(
                X_hsi_base[start:end]  + alpha * hsi_delta[start:end]
            )
            meta_interp_all.append(
                X_meta_base[start:end] + alpha * meta_delta[start:end]
            )

        # Stack along a leading "steps" axis, process all steps in one tape call
        hsi_stack  = np.stack(hsi_interp_all,  axis=0)   # (steps, bs, 2, H, W, B)
        meta_stack = np.stack(meta_interp_all, axis=0)   # (steps, bs, meta_dim)
        bs = end - start

        step_hsi_grads  = np.zeros((n_steps, bs) + X_hsi_curr.shape[1:],  dtype=np.float32)
        step_meta_grads = np.zeros((n_steps, bs, META_DIM),                dtype=np.float32)

        for s_idx in range(n_steps):
            x_hsi_tf  = tf.constant(hsi_stack[s_idx],  dtype=tf.float32)
            x_meta_tf = tf.constant(meta_stack[s_idx], dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch([x_hsi_tf, x_meta_tf])
                preds = keras_model(
                    {"image_input": x_hsi_tf, "meta_input": x_meta_tf},
                    training=False,
                )
                loss = tf.reduce_sum(preds)

            g_hsi  = tape.gradient(loss, x_hsi_tf)
            g_meta = tape.gradient(loss, x_meta_tf)
            del tape

            if g_hsi  is not None: step_hsi_grads[s_idx]  = g_hsi.numpy()
            if g_meta is not None: step_meta_grads[s_idx] = g_meta.numpy()

        # Average over integration steps
        sum_hsi_grads[start:end]  += step_hsi_grads.mean(axis=0)
        sum_meta_grads[start:end] += step_meta_grads.mean(axis=0)

    # Divide by n_steps (already averaged per batch above, sum over batches not needed)
    # — actually we accumulate the mean directly, so just return
    return sum_hsi_grads.astype(np.float32), sum_meta_grads.astype(np.float32)


def compute_swing_attribution(
    keras_model: tf.keras.Model,
    X_hsi: np.ndarray,
    X_meta: np.ndarray,
    sequence_index: np.ndarray,
    n_steps: int = 50,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Compute SWING attributions for all samples.

    For each sample i, the SWING baseline is sample (i-1) in the temporal ordering
    of the same slice. Samples at the first time step of a slice use the mean of
    the inner-val HSI and metadata as their baseline (no previous observation
    available).

    Args:
        keras_model:    Raw tf.keras.Model.
        X_hsi:          HSI sequences  (n, 2, H, W, B).
        X_meta:         Metadata       (n, meta_dim).
        sequence_index: Array mapping sample i to its position within its slice.
                        sequence_index[i] = 0 means sample i is the first step
                        of its slice and will use the mean-spectrum baseline.
        n_steps:        Riemann integration steps (50 recommended for publication).
        batch_size:     GPU batch size.

    Returns:
        dict with keys:
            'hsi'  : SWING attributions, shape (n, 2, H, W, B)
            'meta' : SWING attributions, shape (n, meta_dim)
    """
    n = X_hsi.shape[0]

    # Mean-spectrum baseline for slice-initial samples
    hsi_mean_baseline  = X_hsi.mean(axis=0, keepdims=True).repeat(n, axis=0)
    meta_mean_baseline = X_meta.mean(axis=0, keepdims=True).repeat(n, axis=0)

    # Build per-sample baselines: use previous sample if sequence_index > 0
    X_hsi_base  = hsi_mean_baseline.copy()
    X_meta_base = meta_mean_baseline.copy()
    for i in range(1, n):
        if sequence_index[i] > 0:
            X_hsi_base[i]  = X_hsi[i - 1]
            X_meta_base[i] = X_meta[i - 1]

    logger.info("  Running SWING path integral (%d steps, %d samples)...", n_steps, n)
    path_hsi, path_meta = _riemann_gradients(
        keras_model, X_hsi, X_hsi_base, X_meta, X_meta_base, n_steps, batch_size
    )

    # SWING = gradient_path_integral * (input - baseline)
    swing_hsi  = path_hsi  * (X_hsi  - X_hsi_base)
    swing_meta = path_meta * (X_meta - X_meta_base)

    return {"hsi": swing_hsi, "meta": swing_meta}


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_attributions(attributions: dict[str, np.ndarray]) -> dict:
    """Aggregate SWING attributions across samples.

    Args:
        attributions: Output of compute_swing_attribution.

    Returns:
        dict with:
            'wavelength'  : shape (N_BANDS,)   — mean |SWING| per band
            'pixel'       : shape (H, W)        — mean |SWING| per spatial pixel
            'metadata'    : shape (META_DIM,)   — mean |SWING| per metadata channel
            'timestep'    : shape (2,)           — mean |SWING| per time step
            'wl_per_step' : shape (2, N_BANDS)  — per-timestep spectral attribution
    """
    hsi  = np.abs(attributions["hsi"])   # (n, 2, H, W, B)
    meta = np.abs(attributions["meta"])  # (n, META_DIM)

    wavelength  = hsi.mean(axis=(0, 1, 2, 3))         # (B,)
    pixel       = hsi.mean(axis=(0, 1, 4))             # (H, W)
    timestep    = hsi.mean(axis=(0, 2, 3, 4))          # (2,)
    wl_per_step = hsi.mean(axis=(0, 2, 3))             # (2, B)
    metadata    = meta.mean(axis=0)                    # (META_DIM,)

    return {
        "wavelength":  wavelength,
        "pixel":       pixel,
        "metadata":    metadata,
        "timestep":    timestep,
        "wl_per_step": wl_per_step,
    }


def build_sequence_index(batch_ids: np.ndarray, slice_info: np.ndarray) -> np.ndarray:
    """Build a per-sample index indicating position within its slice's time series.

    Args:
        batch_ids:  Array of batch IDs per sample.
        slice_info: Array of slice IDs per sample (assumed sorted by time within slice).

    Returns:
        sequence_index: shape (n,), value = position of sample within its slice.
    """
    seq_idx = np.zeros(len(batch_ids), dtype=np.int32)
    slice_counter: dict = {}
    for i in range(len(batch_ids)):
        # key = (int(batch_ids[i]), int(slice_info[i]))
        si = slice_info[i]
        key = (int(batch_ids[i]), tuple(si) if hasattr(si, '__iter__') else int(si))
        pos = slice_counter.get(key, 0)
        seq_idx[i] = pos
        slice_counter[key] = pos + 1
    return seq_idx


# ── Per-fold SWING pipeline ───────────────────────────────────────────────────

def run_swing_fold(
    fold_idx: int,
    X_hsi: np.ndarray,
    X_meta: np.ndarray,
    batch_ids: np.ndarray,
    slice_info: np.ndarray,
    sequence_index: np.ndarray,
    n_steps: int,
    batch_size: int,
    inner_val_batches: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """Full SWING attribution pipeline for a single outer LOBO fold.

    Steps:
        1. Build outer train/test masks (test = batch fold_idx).
        2. Build inner train/val masks from outer training batches.
        3. Load trained fold model.
        4. Run SWING on inner-val samples only.
        5. Aggregate and save per-fold importance files.

    Args:
        fold_idx:       1-based fold index.
        X_hsi:          Full HSI array.
        X_meta:         Full metadata array.
        batch_ids:      Batch IDs per sample.
        slice_info:     Slice IDs per sample.
        sequence_index: Position of each sample within its slice.
        n_steps:        Riemann integration steps.
        batch_size:     GPU batch size.
        inner_val_batches: Optional list of batch IDs to use for inner validation.
                          If None, uses make_inner_val_mask with seed.
        seed:           Random seed for reproducible inner split.

    Returns:
        Aggregated importance dict for this fold.
    """
    fold_dir = OUTPUT_DIR / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Cache: skip if already computed
    cache_file = fold_dir / "wavelength_importance.csv"
    if cache_file.exists():
        logger.info("Fold %d: cached results found, loading.", fold_idx)
        wl_df  = pd.read_csv(fold_dir / "wavelength_importance.csv")
        px_df  = pd.read_csv(fold_dir / "pixel_importance.csv")
        md_df  = pd.read_csv(fold_dir / "metadata_importance.csv")
        ts_df  = pd.read_csv(fold_dir / "timestep_split.csv")
        return {
            "wavelength": wl_df["swing_importance"].values,
            "pixel":      px_df["swing_importance"].values.reshape(IMG_H, IMG_W),
            "metadata":   md_df["swing_importance"].values,
            "timestep":   ts_df["swing_importance"].values,
        }

    # ── Fold masks ────────────────────────────────────────────────────────────
    all_batches   = np.unique(batch_ids)
    test_batch    = all_batches[fold_idx - 1]        # 1-based → 0-based index
    test_mask     = batch_ids == test_batch
    train_mask    = ~test_mask

    # Use provided inner_val_batches or generate them
    if inner_val_batches is not None:
        inner_val_mask = train_mask & np.isin(batch_ids, inner_val_batches)
        inner_train_mask = train_mask & ~np.isin(batch_ids, inner_val_batches)
        logger.info(
            "Fold %d | test batch=%d | inner_val_batches=%s | inner-val samples=%d",
            fold_idx, test_batch, inner_val_batches, inner_val_mask.sum(),
        )
    else:
        inner_train_mask, inner_val_mask = make_inner_val_mask(batch_ids, train_mask)
        logger.info(
            "Fold %d | test batch=%d | inner-val samples=%d",
            fold_idx, test_batch, inner_val_mask.sum(),
        )
    if inner_val_mask.sum() == 0:
        raise RuntimeError(f"Fold {fold_idx}: inner-val mask is empty.")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_fold_model(fold_idx)
    keras_model = model._model

    # ── SWING on inner-val only ───────────────────────────────────────────────
    val_indices   = np.where(inner_val_mask)[0]
    X_hsi_val     = X_hsi[val_indices]
    X_meta_val    = X_meta[val_indices]
    seq_idx_val   = sequence_index[val_indices]

    attributions = compute_swing_attribution(
        keras_model, X_hsi_val, X_meta_val, seq_idx_val, n_steps, batch_size
    )
    agg = aggregate_attributions(attributions)

    # ── Save ─────────────────────────────────────────────────────────────────
    # Wavelength importance
    wl_df = pd.DataFrame({
        "band_index":      np.arange(N_BANDS),
        "wavelength_nm":   WAVELENGTHS,
        "swing_importance": agg["wavelength"],
        "swing_t0":        agg["wl_per_step"][0],
        "swing_t1":        agg["wl_per_step"][1],
        "rank":            N_BANDS - np.argsort(np.argsort(agg["wavelength"])),
    }).sort_values("rank")
    wl_df.to_csv(fold_dir / "wavelength_importance.csv", index=False)

    # Pixel importance
    px_flat = agg["pixel"].flatten()
    px_df = pd.DataFrame({
        "row":             np.repeat(np.arange(IMG_H), IMG_W),
        "col":             np.tile(np.arange(IMG_W), IMG_H),
        "swing_importance": px_flat,
        "rank":            IMG_H * IMG_W - np.argsort(np.argsort(px_flat)),
    }).sort_values("rank")
    px_df.to_csv(fold_dir / "pixel_importance.csv", index=False)

    # Metadata importance
    meta_names = ["tau_t-1", "T_t-1", "V_t-1", "tau_t", "T_t", "V_t"]
    md_df = pd.DataFrame({
        "channel":         meta_names,
        "swing_importance": agg["metadata"],
        "rank":            META_DIM - np.argsort(np.argsort(agg["metadata"])),
    }).sort_values("rank")
    md_df.to_csv(fold_dir / "metadata_importance.csv", index=False)

    # Time-step split
    ts_df = pd.DataFrame({
        "time_step":       ["t-1", "t"],
        "swing_importance": agg["timestep"],
        "fraction":        agg["timestep"] / (agg["timestep"].sum() + 1e-12),
    })
    ts_df.to_csv(fold_dir / "timestep_split.csv", index=False)

    logger.info(
        "Fold %d | top-5 bands: %s",
        fold_idx,
        wl_df["wavelength_nm"].iloc[:5].round(1).tolist(),
    )
    return agg


# ── Global aggregation ────────────────────────────────────────────────────────

def aggregate_global(fold_results: dict[int, dict]) -> None:
    """Aggregate SWING importance across all folds and save global CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_indices = sorted(fold_results.keys())

    # Wavelengths
    all_wl = np.stack([fold_results[f]["wavelength"] for f in fold_indices], axis=0)
    wl_mean = all_wl.mean(axis=0)
    wl_std  = all_wl.std(axis=0)
    ranked  = np.argsort(wl_mean)[::-1]

    global_wl = pd.DataFrame({
        "rank":           np.arange(1, N_BANDS + 1),
        "band_index":     ranked,
        "wavelength_nm":  WAVELENGTHS[ranked],
        "swing_mean":     wl_mean[ranked],
        "swing_std":      wl_std[ranked],
    })
    global_wl.to_csv(OUTPUT_DIR / "global_wavelength_importance.csv", index=False)

    # Pixels
    all_px = np.stack([fold_results[f]["pixel"].flatten() for f in fold_indices], axis=0)
    px_mean = all_px.mean(axis=0)
    px_ranked = np.argsort(px_mean)[::-1]
    global_px = pd.DataFrame({
        "pixel_rank":     np.arange(1, IMG_H * IMG_W + 1),
        "row":            px_ranked // IMG_W,
        "col":            px_ranked %  IMG_W,
        "swing_mean":     px_mean[px_ranked],
        "swing_std":      all_px.std(axis=0)[px_ranked],
    })
    global_px.to_csv(OUTPUT_DIR / "global_pixel_importance.csv", index=False)

    # Metadata
    meta_names = ["tau_t-1", "T_t-1", "V_t-1", "tau_t", "T_t", "V_t"]
    all_md = np.stack([fold_results[f]["metadata"] for f in fold_indices], axis=0)
    global_md = pd.DataFrame({
        "channel":    meta_names,
        "swing_mean": all_md.mean(axis=0),
        "swing_std":  all_md.std(axis=0),
    }).sort_values("swing_mean", ascending=False)
    global_md.to_csv(OUTPUT_DIR / "global_metadata_importance.csv", index=False)

    # Time-step split
    all_ts = np.stack([fold_results[f]["timestep"] for f in fold_indices], axis=0)
    ts_mean = all_ts.mean(axis=0)
    global_ts = pd.DataFrame({
        "time_step":  ["t-1", "t"],
        "swing_mean": ts_mean,
        "swing_std":  all_ts.std(axis=0),
        "fraction":   ts_mean / (ts_mean.sum() + 1e-12),
    })
    global_ts.to_csv(OUTPUT_DIR / "global_timestep_split.csv", index=False)

    logger.info("Global aggregation complete.")
    logger.info("Top-20 wavelengths by SWING importance:")
    for _, row in global_wl.head(20).iterrows():
        logger.info(
            "  Rank %2d | Band %3d | %.1f nm | mean=%.6f ± %.6f",
            int(row["rank"]), int(row["band_index"]), row["wavelength_nm"],
            row["swing_mean"], row["swing_std"],
        )


# ── Wavelength selection ──────────────────────────────────────────────────────

def select_top_k_wavelengths(k_values: list[int] = TOP_K_VALUES) -> dict[int, list[int]]:
    """Load global importance and select top-k band indices for each k.

    Importance is derived exclusively from SWING — no mixing with other methods.

    Returns:
        dict mapping k -> list of band indices (sorted by importance, descending).
    """
    global_wl = pd.read_csv(OUTPUT_DIR / "global_wavelength_importance.csv")
    selected = {}
    for k in k_values:
        top_k_bands = global_wl["band_index"].iloc[:k].astype(int).tolist()
        selected[k] = top_k_bands
        logger.info(
            "Top-%d bands (nm): %s",
            k,
            [f"{WAVELENGTHS[b]:.0f}" for b in top_k_bands[:5]],
        )
    with open(OUTPUT_DIR / "selected_wavelengths.json", "w") as fh:
        json.dump({str(k): v for k, v in selected.items()}, fh, indent=2)
    logger.info("Saved selected_wavelengths.json")
    return selected


def select_top_p_pixels(p_values: list[float] = PIXEL_FRACTIONS) -> dict[float, list[int]]:
    """Select top-p fraction of pixel indices by global SWING importance.

    Returns:
        dict mapping p -> list of flat pixel indices (row * IMG_W + col).
    """
    global_px = pd.read_csv(OUTPUT_DIR / "global_pixel_importance.csv")
    selected = {}
    total_pixels = IMG_H * IMG_W
    for p in p_values:
        n_sel = max(1, int(p * total_pixels))
        top_rows = global_px["row"].iloc[:n_sel].astype(int).values
        top_cols = global_px["col"].iloc[:n_sel].astype(int).values
        flat_indices = (top_rows * IMG_W + top_cols).tolist()
        selected[p] = flat_indices
    with open(OUTPUT_DIR / "selected_pixels.json", "w") as fh:
        json.dump({str(p): v for p, v in selected.items()}, fh, indent=2)
    logger.info("Saved selected_pixels.json")
    return selected


# ── Refined model retraining ──────────────────────────────────────────────────


# ── Adaptive hyperparameter scaling ───────────────────────────────────────────

def get_adaptive_hyperparams(base_params: dict, n_bands: int) -> dict:
    """Get adaptive hyperparameters based on number of spectral bands.

    When reducing the number of bands (k < 112), we adjust:
    - Learning rate: higher for smaller feature spaces (faster convergence)
    - Dropout rate: lower for smaller feature spaces (less regularization needed)
    - Epochs: fewer for smaller feature spaces (risk of overfitting is lower)

    Args:
        base_params: Hyperparameters tuned for FULL spectrum (112 bands)
        n_bands: Actual number of spectral bands used

    Returns:
        Adjusted hyperparameter dictionary
    """
    # Scaling factors for reduced feature space
    if n_bands >= 60:
        scale = 1.0
    elif n_bands >= 40:
        scale = 0.6
    elif n_bands >= 30:
        scale = 0.5
    elif n_bands >= 20:
        scale = 0.4
    else:
        scale = 0.3

    # Adjust hyperparameters based on feature space size
    adjusted = {
        "n_conv_layers": base_params.get("n_conv_layers", 3),
        "conv_filters": base_params.get("conv_filters", 64),
        "n_lstm_layers": base_params.get("n_lstm_layers", 2),
        "lstm_units": base_params.get("lstm_units", 128),
        "dropout_rate": base_params.get("dropout_rate", 0.3) * (1.0 - scale * 0.2),
        "learning_rate": base_params.get("learning_rate", 1e-3) * (1.5 / scale),
        "batch_size": min(64, max(4, int(base_params.get("batch_size", 16) * (n_bands / 112 + 0.2)))),
        "epochs": int(base_params.get("epochs", 100) * scale),
        "early_stopping_patience": int(base_params.get("early_stopping_patience", 15) * scale),
        "random_state": 50,
    }

    return adjusted

def retrain_refined_fold(
    fold_idx: int,
    X_hsi: np.ndarray,
    X_meta: np.ndarray,
    y: np.ndarray,
    batch_ids: np.ndarray,
    selected_band_indices: list[int],
    selected_pixel_flat: list[int],
    k: int,
    p: float,
    ref_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Retrain CNN-LSTM with reduced feature space for one outer fold.

    The fold-specific SWING ranking is used for feature selection (not the global
    average) to ensure no information from the test fold contaminates the
    selected feature set.

    Args:
        fold_idx:              1-based fold index.
        X_hsi:                 Full HSI array (n, 2, H, W, B).
        X_meta:                Full metadata (n, meta_dim). Always kept complete.
        y:                     Moisture targets (n,).
        batch_ids:             Batch IDs per sample.
        selected_band_indices: Top-k band indices from fold-specific SWING ranking.
        selected_pixel_flat:   Top-p pixel flat indices from fold-specific ranking.
        k:                     Number of selected wavelengths.
        p:                     Pixel fraction selected.

    Returns:
        dict with keys 'rmse', 'r2', 'rpd', 'n_test'.
    """
    all_batches = np.unique(batch_ids)
    test_batch  = all_batches[fold_idx - 1]
    test_mask   = batch_ids == test_batch
    train_mask  = ~test_mask

    # Build reduced HSI input: subset pixels then subset bands
    # pixel subset: shape (n, 2, n_pixels, B) -> select bands -> (n, 2, n_pixels, k)
    n_sel_pixels = len(selected_pixel_flat)
    selected_rows = [idx // IMG_W for idx in selected_pixel_flat]
    selected_cols = [idx %  IMG_W for idx in selected_pixel_flat]

    # Build spatial mask: keep only selected pixels, zero the rest
    pixel_mask = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    pixel_mask[selected_rows, selected_cols] = 1.0           # (64, 52)

    # Build band mask: keep only selected bands, zero the rest
    # Select pixels and truncate bands to actual feature space (no zero-padding!)
    # Step 1: Apply pixel mask (keep only selected pixels)
    X_hsi_pixel_selected = X_hsi * pixel_mask[np.newaxis, np.newaxis, :, :, np.newaxis]

    selected_band_indices_sorted = sorted(selected_band_indices)
    X_hsi_tr   = X_hsi_pixel_selected[train_mask][:, :, :, :, selected_band_indices_sorted]
    X_hsi_test = X_hsi_pixel_selected[test_mask][:, :, :, :, selected_band_indices_sorted]

    y_tr        = y[train_mask]
    y_test      = y[test_mask]
    X_meta_tr   = X_meta[train_mask]
    X_meta_test = X_meta[test_mask]

    logger.info("  Truncated to %d pixels x %d bands (no zero-padding)",
                n_sel_pixels, len(selected_band_indices_sorted))

    # Model constructor with adaptive hyperparameters
    params = load_fold_hyperparams(fold_idx)
    n_actual_bands = len(selected_band_indices)
    adjusted_params = get_adaptive_hyperparams(params, n_actual_bands)
    logger.info("Fold %d | k=%d bands, p=%.1f pixels | Adapted: lr=%.6f, dropout=%.3f, epochs=%d",
                fold_idx, k, p,
                adjusted_params.get("learning_rate", 1e-3),
                adjusted_params.get("dropout_rate", 0.3),
                adjusted_params.get("epochs", 100))
    wrapper = CNNLSTMHSI2SModel(
        n_bands=n_actual_bands,                        # actual reduced dimension
        hsi_h=IMG_H,                                   # always 64
        hsi_w=IMG_W,                                   # always 52
        meta_dim=META_DIM,
        seq_length=T_STEPS,
        n_conv_layers=adjusted_params.get("n_conv_layers", 3),
        conv_filters=adjusted_params.get("conv_filters", 64),
        n_lstm_layers=adjusted_params.get("n_lstm_layers", 2),
        lstm_units=adjusted_params.get("lstm_units", 128),
        dropout_rate=adjusted_params.get("dropout_rate", 0.3),
        learning_rate=adjusted_params.get("learning_rate", 1e-3),
        batch_size=adjusted_params.get("batch_size", 16),
        epochs=adjusted_params.get("epochs", 100),
        early_stopping_patience=adjusted_params.get("early_stopping_patience", 15),
        random_state=adjusted_params.get("random_state", 50),
    )

    # Inner val for early stopping during retraining
    _, inner_val_mask = make_inner_val_mask(batch_ids, train_mask)
    # Remap inner_val_mask to the training-fold index space
    train_indices = np.where(train_mask)[0]
    inner_val_in_train = inner_val_mask[train_mask]

    wrapper.fit(
        X_hsi_tr, y_tr, X_meta_tr,
        validation_data=(
            X_hsi_tr[inner_val_in_train],
            y_tr[inner_val_in_train],
            X_meta_tr[inner_val_in_train],
        ),
        verbose=0,
    )

    # Evaluate on test fold — first and only access
    y_pred = wrapper.predict(X_hsi_test, X_meta_test).flatten()
    y_true = y_test.flatten()

    # Restrict accuracy to samples whose target has a reference (paper protocol).
    if ref_mask is not None:
        ref_test = np.asarray(ref_mask[test_mask], dtype=bool).flatten()
        if ref_test.any():
            y_true = y_true[ref_test]
            y_pred = y_pred[ref_test]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-12))
    rpd  = float(np.std(y_true) / (rmse + 1e-12))

    n_ref = int(np.asarray(ref_mask[test_mask], dtype=bool).sum()) if ref_mask is not None else None
    return {"rmse": rmse, "r2": r2, "rpd": rpd, "n_test": int(y_test.shape[0]), "n_test_reference": n_ref}


def run_refined_pipeline(
    X_hsi: np.ndarray,
    X_meta: np.ndarray,
    y: np.ndarray,
    batch_ids: np.ndarray,
    k_values: list[int] = TOP_K_VALUES,
    p_values: list[float] = PIXEL_FRACTIONS,
    ref_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Retrain refined models for all (k, p) combinations across all 16 folds.

    Uses fold-specific SWING rankings to prevent leakage.

    Returns:
        DataFrame with columns [k, p, rmse_mean, rmse_std, r2_mean, r2_std,
                                 rpd_mean, rpd_std].
    """
    all_batches = np.unique(batch_ids)
    rows = []

    for k in k_values:
        for p in p_values:
            n_sel_pixels = max(1, int(p * IMG_H * IMG_W))
            combo_dir    = OUTPUT_DIR / f"k{k}_p{str(p).replace('.','')}"
            combo_dir.mkdir(parents=True, exist_ok=True)

            # Cache check
            agg_path = combo_dir / "aggregated_metrics.csv"
            if agg_path.exists():
                logger.info("k=%d p=%.1f: cached, skipping.", k, p)
                agg_df = pd.read_csv(agg_path)
                row = {"k": k, "p": p}
                for _, r in agg_df.iterrows():
                    row[f"{r['metric']}_mean"] = r["mean"]
                    row[f"{r['metric']}_std"]  = r["std"]
                rows.append(row)
                continue

            fold_metrics = []

            for fold_idx in range(1, len(all_batches) + 1):
                # Load fold-specific ranking (not global) to avoid leakage
                fold_wl_path = OUTPUT_DIR / f"fold_{fold_idx}" / "wavelength_importance.csv"
                fold_px_path = OUTPUT_DIR / f"fold_{fold_idx}" / "pixel_importance.csv"

                if not fold_wl_path.exists() or not fold_px_path.exists():
                    raise FileNotFoundError(
                        f"Fold {fold_idx} SWING results missing. "
                        "Run attribution step first."
                    )

                fold_wl = pd.read_csv(fold_wl_path).sort_values("rank")
                fold_px = pd.read_csv(fold_px_path).sort_values("rank")

                sel_bands  = fold_wl["band_index"].iloc[:k].astype(int).tolist()
                sel_rows   = fold_px["row"].iloc[:n_sel_pixels].astype(int).values
                sel_cols   = fold_px["col"].iloc[:n_sel_pixels].astype(int).values
                sel_pixels = (sel_rows * IMG_W + sel_cols).tolist()

                metrics = retrain_refined_fold(
                    fold_idx, X_hsi, X_meta, y, batch_ids,
                    sel_bands, sel_pixels, k, p,
                    ref_mask=ref_mask,
                )
                fold_metrics.append(metrics)
                logger.info(
                    "  k=%d p=%.1f fold=%d | RMSE=%.4f R²=%.4f RPD=%.4f",
                    k, p, fold_idx, metrics["rmse"], metrics["r2"], metrics["rpd"],
                )

            # Aggregate across folds
            rmse_arr = np.array([m["rmse"] for m in fold_metrics])
            r2_arr   = np.array([m["r2"]   for m in fold_metrics])
            rpd_arr  = np.array([m["rpd"]  for m in fold_metrics])

            agg_rows_csv = [
                {"metric": "rmse", "mean": rmse_arr.mean(), "std": rmse_arr.std()},
                {"metric": "r2",   "mean": r2_arr.mean(),   "std": r2_arr.std()},
                {"metric": "rpd",  "mean": rpd_arr.mean(),  "std": rpd_arr.std()},
            ]
            pd.DataFrame(agg_rows_csv).to_csv(agg_path, index=False)

            row = {
                "k": k, "p": p,
                "rmse_mean": rmse_arr.mean(), "rmse_std": rmse_arr.std(),
                "r2_mean":   r2_arr.mean(),   "r2_std":   r2_arr.std(),
                "rpd_mean":  rpd_arr.mean(),  "rpd_std":  rpd_arr.std(),
            }
            rows.append(row)
            logger.info(
                "k=%d p=%.1f | RMSE=%.4f±%.4f  R²=%.4f±%.4f  RPD=%.4f±%.4f",
                k, p,
                rmse_arr.mean(), rmse_arr.std(),
                r2_arr.mean(),   r2_arr.std(),
                rpd_arr.mean(),  rpd_arr.std(),
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "refined_model_results.csv", index=False)
    logger.info("Saved refined_model_results.csv")
    return results_df


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SWING attribution and model refinement for CNN-LSTM HSI 2-step model."
    )
    parser.add_argument(
        "--folds", default="1-16",
        help="Fold range to process, e.g. '1-16' or '1,3,5'.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=50,
        help="Riemann integration steps for SWING (default: 50).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="GPU batch size for attribution (default: 8).",
    )
    parser.add_argument(
        "--skip_retrain", action="store_true",
        help="Skip refined model retraining; only run attribution.",
    )
    return parser.parse_args()


def parse_fold_range(s: str) -> list[int]:
    """Parse '1-16' or '1,3,5' into a list of ints."""
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    args = parse_args()
    fold_indices = parse_fold_range(args.folds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    N_INNER_VAL_BATCHES = 2  # 2 batches from training for inner validation
    SEED = 42

    logger.info("Loading dataset...")
    X_hsi, X_meta, is_padded, y, batch_ids, sample_ids, slice_info = load_dataset_hsi_2s()
    # Reference mask for accuracy metrics (paper protocol: reference samples only).
    ref_mask_all = reference_mask(sample_ids)
    sequence_index = build_sequence_index(batch_ids, slice_info)
    logger.info("Dataset: %d sequences | %d batches", len(y), len(np.unique(batch_ids)))

    # Generate fold data with consistent inner validation batches
    folds_data = get_lobo_folds_with_inner_val(batch_ids, n_val_batches=N_INNER_VAL_BATCHES, seed=SEED)

    # ── Step 1: SWING attribution (per fold) ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: SWING attribution")
    fold_results: dict[int, dict] = {}
    for fold_idx in fold_indices:
        logger.info("-" * 40)
        logger.info("Processing fold %d / %d ...", fold_idx, len(fold_indices))
        fold_results[fold_idx] = run_swing_fold(
            fold_idx, X_hsi, X_meta, batch_ids, slice_info,
            sequence_index, args.n_steps, args.batch_size,
            inner_val_batches=folds_data[fold_idx - 1]['inner_val_batches'],
        )

    # ── Step 2: Global aggregation ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Global aggregation across %d folds", len(fold_indices))
    aggregate_global(fold_results)

    # ── Step 3: Feature selection ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Wavelength and pixel selection")
    select_top_k_wavelengths(TOP_K_VALUES)
    select_top_p_pixels(PIXEL_FRACTIONS)

    # ── Step 4: Refined model retraining ─────────────────────────────────────
    if not args.skip_retrain:
        logger.info("=" * 60)
        logger.info("STEP 4: Refined model retraining (%d combinations)",
                    len(TOP_K_VALUES) * len(PIXEL_FRACTIONS))
        results_df = run_refined_pipeline(
            X_hsi, X_meta, y, batch_ids, TOP_K_VALUES, PIXEL_FRACTIONS,
            ref_mask=ref_mask_all,
        )
        logger.info("\nBest configuration by RMSE:")
        best_row = results_df.loc[results_df["rmse_mean"].idxmin()]
        logger.info(
            "  k=%d  p=%.1f  |  RMSE=%.4f±%.4f  R²=%.4f±%.4f  RPD=%.4f±%.4f",
            int(best_row["k"]), best_row["p"],
            best_row["rmse_mean"], best_row["rmse_std"],
            best_row["r2_mean"],   best_row["r2_std"],
            best_row["rpd_mean"],  best_row["rpd_std"],
        )

    logger.info("=" * 60)
    logger.info("SWING pipeline complete. Outputs: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
