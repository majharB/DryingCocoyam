#!/usr/bin/env python3
"""
Post-training INT8 dynamic-range quantization for the CNNLSTMHSI_2S global-wavelength
model, with a SIZE + LATENCY SANITY-CHECK DIAGNOSTIC MODE that exits early
(sys.exit()) before running the full pipeline, so you can verify that:

  1. The "true" FP32 weight size (n_params x 4 bytes) vs. the actual INT8
     .tflite file size on disk gives a plausible compression ratio (should
     be <= 4x for dynamic-range quantization, since only weights are
     converted to int8; typically 2-4x in practice).
  2. FP32 latency measured via a warmed-up tf.function call (not the
     Python-dispatch-heavy .predict()) vs. INT8 TFLite interpreter latency
     gives a plausible, defensible speedup ratio.

Usage:
    source coco-env.sh
    python src/quantize/quantize_and_evaluate.py \
        --config-dir outputs/xai/hsi_2s_global_wl_1_1398 \
        --wavelengths-nm 1398 \
        --output-dir outputs/quantization/hsi_2s_global_wl_1_1398 \
        --diagnostic-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from src.data_loader import load_dataset_hsi_2s, lobos_folds_for_sequences, reference_mask
from src.cnn_lstm_model import CNNLSTMHSI2SModel

# Hide GPUs entirely for this script: cuDNN-fused LSTM kernels (CudnnRNNV3)
# are not representable in TFLite, and Keras only avoids them when no GPU
# is visible during model loading / graph tracing.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

N_FOLDS = 16
HSI_HEIGHT = 64
HSI_WIDTH = 52
META_DIM = 6
SEQ_LENGTH = 2
N_CALIB_SAMPLES = 100
N_BOOT = 10000
N_LATENCY_CHECK_SAMPLES = 50  # small sample sufficient for the diagnostic sanity check

ALL_WAVELENGTHS_NM = [
    937.33, 944.25, 951.16, 958.08, 965, 971.92, 978.85, 985.77, 992.7, 999.63,
    1006.57, 1013.5, 1020.44, 1027.38, 1034.32, 1041.27, 1048.21, 1055.16, 1062.12, 1069.07,
    1076.03, 1082.98, 1089.94, 1096.91, 1103.87, 1110.84, 1117.81, 1124.78, 1131.75, 1138.73,
    1145.71, 1152.69, 1159.67, 1166.66, 1173.64, 1180.63, 1187.63, 1194.62, 1201.62, 1208.62,
    1215.62, 1222.62, 1229.63, 1236.63, 1243.64, 1250.66, 1257.67, 1264.69, 1271.71, 1278.73,
    1285.75, 1292.78, 1299.8, 1306.83, 1313.87, 1320.9, 1327.94, 1334.98, 1342.02, 1349.06,
    1356.11, 1363.16, 1370.21, 1377.26, 1384.31, 1391.37, 1398, 1405, 1413, 1420,
    1426.69, 1433.76, 1440.84, 1447.91, 1454.99, 1462.07, 1469.15, 1476.23, 1483.32, 1490.41,
    1497.5, 1504.59, 1511.69, 1518.79, 1525.89, 1532.99, 1540.09, 1547.2, 1554.31, 1561.42,
    1568.53, 1575.65, 1582.77, 1589.89, 1597.01, 1604.13, 1611.26, 1618.39, 1625.52, 1632.66,
    1639.79, 1646.93, 1654.07, 1661.21, 1668.36, 1675.51, 1682.65, 1689.81, 1696.96, 1704.12,
    1711.28, 1718.44,
]
ALL_WAVELENGTHS_INT = [round(w) for w in ALL_WAVELENGTHS_NM]


def resolve_band_indices(target_wavelengths_nm: list[int], all_wavelengths_int: list[int]) -> list[int]:
    indices = []
    for target in target_wavelengths_nm:
        idx = min(range(len(all_wavelengths_int)), key=lambda i: abs(all_wavelengths_int[i] - target))
        indices.append(idx)
    return sorted(indices)


def compute_metrics_full(y_true, y_pred, mask=None) -> dict:
    """RMSE/MAE/R2/RPD; if ``mask`` is given, metrics use only those samples
    (e.g. samples whose target has a reference — paper protocol)."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).flatten()
        if mask.shape[0] != y_true.shape[0]:
            raise ValueError(f"mask length ({mask.shape[0]}) != y_true length ({y_true.shape[0]})")
        if not mask.any():
            raise ValueError("mask selects no samples; cannot compute metrics")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    sd = float(np.std(y_true, ddof=1))
    rpd = float(sd / rmse) if rmse > 0 else float("inf")
    return {"mae": mae, "rmse": rmse, "r2": r2, "rpd": rpd}


# ---------------------------------------------------------------------------
# Model reconstruction with unrolled LSTM (TFLite-compatible, no Flex ops)
# ---------------------------------------------------------------------------

def rebuild_model_with_unrolled_lstm(keras_model_path: Path) -> tf.keras.Model:
    """Reconstruct the CNNLSTMHSI_2S architecture with LSTM(unroll=True) and
    load the trained weights from the saved .h5 file.

    Rationale: with the default unroll=False, Keras builds LSTM layers using
    a dynamic tf.while_loop / TensorList, which TFLite exports as unsupported
    "Flex" custom ops. Since seq_length is fixed at 2, unrolling costs nothing
    computationally but produces a fully static graph representable purely in
    TFLITE_BUILTINS -- no Flex delegate required at inference time.
    """
    orig_model = tf.keras.models.load_model(keras_model_path, compile=False)

    image_layer = orig_model.get_layer("image_input")
    meta_layer = orig_model.get_layer("meta_input")
    image_input_shape = getattr(image_layer, "shape", None) or image_layer.output.shape
    meta_input_shape = getattr(meta_layer, "shape", None) or meta_layer.output.shape
    seq_length, hsi_h, hsi_w, n_bands = image_input_shape[1:]
    meta_dim = meta_input_shape[1]

    conv_layers = [l for l in orig_model.layers if isinstance(l, tf.keras.layers.TimeDistributed)
                   and isinstance(l.layer, tf.keras.layers.Conv2D)]
    n_conv_layers = len(conv_layers)
    conv_filters = conv_layers[0].layer.filters if conv_layers else 16

    lstm_layers = [l for l in orig_model.layers if isinstance(l, tf.keras.layers.LSTM)]
    n_lstm_layers = len(lstm_layers)
    lstm_units = lstm_layers[0].units if lstm_layers else 32

    dropout_layers = [l for l in orig_model.layers if isinstance(l, tf.keras.layers.Dropout)]
    dropout_rate = dropout_layers[0].rate if dropout_layers else 0.1

    wrapper = CNNLSTMHSI2SModel(
        n_bands=n_bands, hsi_h=hsi_h, hsi_w=hsi_w, meta_dim=meta_dim,
        seq_length=seq_length, n_conv_layers=n_conv_layers, conv_filters=conv_filters,
        n_lstm_layers=n_lstm_layers, lstm_units=lstm_units, dropout_rate=dropout_rate,
    )

    _orig_lstm_init = tf.keras.layers.LSTM.__init__

    def _patched_lstm_init(self, *args, **kwargs):
        kwargs["unroll"] = True
        _orig_lstm_init(self, *args, **kwargs)

    tf.keras.layers.LSTM.__init__ = _patched_lstm_init
    try:
        new_model = wrapper._build_model()
    finally:
        tf.keras.layers.LSTM.__init__ = _orig_lstm_init

    new_model.set_weights(orig_model.get_weights())

    del orig_model
    return new_model


# ---------------------------------------------------------------------------
# DIAGNOSTIC: true size + fair latency sanity check
# ---------------------------------------------------------------------------

def run_size_latency_diagnostic(model: tf.keras.Model, keras_model_path: Path,
                                 tflite_path: Path, X_hsi_val: np.ndarray,
                                 X_meta_val: np.ndarray,
                                 n_check: int = N_LATENCY_CHECK_SAMPLES) -> None:
    """Prints a corrected, apples-to-apples comparison of model size and
    inference latency, then exits the process. Use this to sanity-check
    numbers before trusting compression_ratio / speedup_ratio computed
    from raw .h5 file size and .predict()-based latency loops, which are
    both known to be misleading (see explanation in printed output)."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: TRUE SIZE COMPARISON (raw weight bytes)")
    print("=" * 60)

    n_params = model.count_params()
    true_fp32_bytes = n_params * 4
    true_int8_bytes = os.path.getsize(tflite_path)
    h5_bytes = os.path.getsize(keras_model_path)

    print(f"n_params:                     {n_params}")
    print(f"True FP32 weight size:        {true_fp32_bytes / 1024 / 1024:.4f} MB (params x 4 bytes)")
    print(f".h5 file size (misleading):   {h5_bytes / 1024 / 1024:.4f} MB "
          f"(includes optimizer state / metadata -- do NOT use for compression ratio)")
    print(f"INT8 .tflite file size:       {true_int8_bytes / 1024 / 1024:.4f} MB")
    true_compression_ratio = true_fp32_bytes / true_int8_bytes
    h5_based_ratio = h5_bytes / true_int8_bytes
    print(f"TRUE compression ratio:       {true_compression_ratio:.2f}x   <-- use this")
    print(f".h5-based ratio (WRONG):      {h5_based_ratio:.2f}x   <-- do not report this")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: FAIR LATENCY COMPARISON (apples-to-apples)")
    print("=" * 60)

    @tf.function
    def _infer_fp32(img, meta):
        return model({"image_input": img, "meta_input": meta}, training=False)

    n_check = min(n_check, X_hsi_val.shape[0])
    dummy_img = tf.constant(X_hsi_val[0:1], dtype=tf.float32)
    dummy_meta = tf.constant(X_meta_val[0:1], dtype=tf.float32)
    _ = _infer_fp32(dummy_img, dummy_meta)  # warm-up / trace, not timed

    fp32_lat = np.zeros(n_check)
    for i in range(n_check):
        img = tf.constant(X_hsi_val[i:i + 1], dtype=tf.float32)
        meta = tf.constant(X_meta_val[i:i + 1], dtype=tf.float32)
        t0 = time.perf_counter()
        _ = _infer_fp32(img, meta)
        t1 = time.perf_counter()
        fp32_lat[i] = (t1 - t0) * 1000.0

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image_detail = next(d for d in input_details if "image_input" in d["name"])
    meta_detail = next(d for d in input_details if "meta_input" in d["name"])

    interpreter.resize_tensor_input(image_detail["index"], (1,) + X_hsi_val.shape[1:])
    interpreter.resize_tensor_input(meta_detail["index"], (1,) + X_meta_val.shape[1:])
    interpreter.allocate_tensors()

    # Warm-up
    interpreter.set_tensor(image_detail["index"], X_hsi_val[0:1].astype(np.float32))
    interpreter.set_tensor(meta_detail["index"], X_meta_val[0:1].astype(np.float32))
    interpreter.invoke()

    int8_lat = np.zeros(n_check)
    for i in range(n_check):
        interpreter.set_tensor(image_detail["index"], X_hsi_val[i:i + 1].astype(np.float32))
        interpreter.set_tensor(meta_detail["index"], X_meta_val[i:i + 1].astype(np.float32))
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        int8_lat[i] = (t1 - t0) * 1000.0

    print(f"FP32 latency (tf.function, warmed up): {fp32_lat.mean():.4f} ms  (std {fp32_lat.std():.4f})")
    print(f"INT8 latency (TFLite interpreter):     {int8_lat.mean():.4f} ms  (std {int8_lat.std():.4f})")
    print(f"TRUE speedup ratio:                    {fp32_lat.mean() / int8_lat.mean():.2f}x   <-- use this")

    print("\n" + "=" * 60)
    print("Diagnostic complete for this fold. Exiting before continuing pipeline.")
    print("(Remove --diagnostic-only flag to run the full pipeline once verified.)")
    print("=" * 60)
    sys.exit(0)


# ---------------------------------------------------------------------------
# STEP 1: quantization
# ---------------------------------------------------------------------------

def make_representative_dataset(X_hsi_calib: np.ndarray, X_meta_calib: np.ndarray,
                                 n_samples: int = N_CALIB_SAMPLES):
    n = min(n_samples, len(X_hsi_calib))
    idx = np.random.RandomState(42).choice(len(X_hsi_calib), size=n, replace=False)

    def gen():
        for i in idx:
            img = X_hsi_calib[i:i + 1].astype(np.float32)
            meta = X_meta_calib[i:i + 1].astype(np.float32)
            yield [img, meta]

    return gen


def quantize_model_int8(keras_model_path: Path, output_path: Path,
                         X_hsi_val: np.ndarray = None, X_meta_val: np.ndarray = None,
                         diagnostic_only: bool = False) -> dict:
    """Convert a saved Keras .h5 model to a dynamic-range-quantized TFLite
    model (weights -> int8, activations stay float32). Float32 I/O retained.

    IMPORTANT: the model is loaded and traced under a forced CPU device
    context. Keras/TF automatically substitutes the fused cuDNN LSTM
    kernel (tf.CudnnRNNV3) whenever a GPU is visible during tracing; that
    op has no TFLite/Flex representation and makes conversion fail
    unconditionally. Loading under /CPU:0 forces the portable, standard
    LSTM implementation, which TFLite (with SELECT_TF_OPS) can represent.
    """
    with tf.device("/CPU:0"):
        model = rebuild_model_with_unrolled_lstm(keras_model_path)
    n_params = int(model.count_params())

    with tf.device("/CPU:0"):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    quantization_mode = "dynamic_range_int8"
    with tf.device("/CPU:0"):
        tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    if diagnostic_only:
        if X_hsi_val is None or X_meta_val is None:
            raise ValueError("--diagnostic-only requires validation data to measure latency")
        run_size_latency_diagnostic(model, keras_model_path, output_path, X_hsi_val, X_meta_val)
        # run_size_latency_diagnostic calls sys.exit(0); code below never runs

    orig_size_mb = keras_model_path.stat().st_size / (1024 ** 2)
    quant_size_mb = output_path.stat().st_size / (1024 ** 2)

    del model
    tf.keras.backend.clear_session()

    return {
        "n_params": n_params,
        "fp32_size_mb": round(orig_size_mb, 4),
        "int8_size_mb": round(quant_size_mb, 4),
        "compression_ratio": round(orig_size_mb / quant_size_mb, 3) if quant_size_mb > 0 else None,
        "quantization_mode": quantization_mode,
    }


# ---------------------------------------------------------------------------
# STEP 3 & 6: Evaluate quantized (TFLite) model + latency
# ---------------------------------------------------------------------------

def evaluate_tflite_model(tflite_path: Path, X_hsi_val: np.ndarray,
                           X_meta_val: np.ndarray, y_val: np.ndarray,
                           ref_mask: np.ndarray | None = None) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_detail = next(d for d in input_details if "image_input" in d["name"])
    meta_detail = next(d for d in input_details if "meta_input" in d["name"])

    n = X_hsi_val.shape[0]
    preds = np.zeros((n, 1), dtype=np.float32)
    latencies = np.zeros(n, dtype=np.float64)

    for i in range(n):
        img = X_hsi_val[i:i + 1].astype(np.float32)
        meta = X_meta_val[i:i + 1].astype(np.float32)

        interpreter.resize_tensor_input(image_detail["index"], img.shape)
        interpreter.resize_tensor_input(meta_detail["index"], meta.shape)
        interpreter.allocate_tensors()

        interpreter.set_tensor(image_detail["index"], img)
        interpreter.set_tensor(meta_detail["index"], meta)

        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()

        out = interpreter.get_tensor(output_details[0]["index"])
        preds[i] = out.flatten()[0]
        latencies[i] = (t1 - t0) * 1000.0

    preds = np.clip(preds, 0, None)
    metrics = compute_metrics_full(y_val, preds, mask=ref_mask)
    metrics["y_pred"] = preds.flatten().tolist()
    metrics["y_true"] = np.asarray(y_val).flatten().tolist()
    metrics["latency_ms_mean"] = float(latencies.mean())
    metrics["latency_ms_std"] = float(latencies.std())
    metrics["latency_ms_per_sample"] = latencies.tolist()
    return metrics


def evaluate_keras_model_latency(keras_model_path: Path, X_hsi_val: np.ndarray,
                                  X_meta_val: np.ndarray, y_val: np.ndarray,
                                  ref_mask: np.ndarray | None = None) -> dict:
    """Re-run the original full-precision Keras model sample-by-sample using
    a warmed-up tf.function call (NOT .predict(), which has heavy per-call
    Python dispatch overhead unrelated to actual compute) to obtain a fair,
    apples-to-apples per-sample latency directly comparable to the TFLite
    interpreter loop. Accuracy metrics computed here are for cross-check
    only; training-time fold_results.json values remain the official
    reference for accuracy."""
    model = tf.keras.models.load_model(keras_model_path, compile=False)

    @tf.function
    def infer(img, meta):
        return model({"image_input": img, "meta_input": meta}, training=False)

    dummy_img = tf.constant(X_hsi_val[0:1], dtype=tf.float32)
    dummy_meta = tf.constant(X_meta_val[0:1], dtype=tf.float32)
    _ = infer(dummy_img, dummy_meta)  # warm-up / trace, not timed

    n = X_hsi_val.shape[0]
    preds = np.zeros((n, 1), dtype=np.float32)
    latencies = np.zeros(n, dtype=np.float64)

    for i in range(n):
        img = tf.constant(X_hsi_val[i:i + 1], dtype=tf.float32)
        meta = tf.constant(X_meta_val[i:i + 1], dtype=tf.float32)
        t0 = time.perf_counter()
        out = infer(img, meta)
        t1 = time.perf_counter()
        preds[i] = out.numpy().flatten()[0]
        latencies[i] = (t1 - t0) * 1000.0

    preds = np.clip(preds, 0, None)
    metrics = compute_metrics_full(y_val, preds, mask=ref_mask)
    metrics["y_pred"] = preds.flatten().tolist()
    metrics["y_true"] = np.asarray(y_val).flatten().tolist()
    metrics["latency_ms_mean"] = float(latencies.mean())
    metrics["latency_ms_std"] = float(latencies.std())
    metrics["latency_ms_per_sample"] = latencies.tolist()

    del model
    tf.keras.backend.clear_session()
    return metrics


# ---------------------------------------------------------------------------
# STEP 4: Load pre-computed full-precision results
# ---------------------------------------------------------------------------

def load_original_results(config_dir: Path) -> tuple[list, dict]:
    with open(config_dir / "fold_results.json") as f:
        fold_results = json.load(f)
    with open(config_dir / "aggregate_results.json") as f:
        agg_results = json.load(f)
    return fold_results, agg_results


def extract_fold_metric_array(fold_results: list, model_key: str, metric: str) -> np.ndarray:
    return np.array([fold[model_key][metric] for fold in fold_results])


# ---------------------------------------------------------------------------
# STEP 5: Paired fold-level bootstrap (fp32 vs int8)
# ---------------------------------------------------------------------------

def paired_fold_bootstrap(metric_a: np.ndarray, metric_b: np.ndarray,
                           n_boot: int = N_BOOT, seed: int = 42) -> dict:
    assert len(metric_a) == len(metric_b)
    n_folds = len(metric_a)
    rng = np.random.default_rng(seed)

    observed_delta = metric_a.mean() - metric_b.mean()
    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_folds, size=n_folds)
        boot_deltas[i] = metric_a[idx].mean() - metric_b[idx].mean()

    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    centered = boot_deltas - boot_deltas.mean()
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_delta)))
    p_value = max(p_value, 1.0 / n_boot)

    return {
        "observed_delta": float(observed_delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }


def holm_bonferroni(p_values: list) -> list:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    prev_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (n - rank)
        adj = max(adj, prev_max)
        adj = min(adj, 1.0)
        adjusted[idx] = adj
        prev_max = adj
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="INT8 dynamic-range quantization + evaluation pipeline")
    parser.add_argument("--config-dir", required=True,
                         help="Source folder with trained models + results, "
                              "e.g. outputs/xai/hsi_2s_global_wl_1_1398")
    parser.add_argument("--wavelengths-nm", type=int, nargs="+", required=True,
                         help="Wavelengths (nm) used by this model, e.g. --wavelengths-nm 1398")
    parser.add_argument("--model-key", default="CNNLSTMHSI_2S_GlobalWL",
                         help="Key used in fold_results.json / aggregate_results.json")
    parser.add_argument("--output-dir", default=None,
                         help="Output dir (default: outputs/quantization/<config_dir_name>)")
    parser.add_argument("--n-calib-samples", type=int, default=N_CALIB_SAMPLES)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--diagnostic-only", action="store_true",
                         help="Run only the size/latency sanity check on fold 1, print "
                              "corrected numbers, then sys.exit() before running the "
                              "full pipeline. Use this to verify compression_ratio / "
                              "speedup_ratio are plausible before trusting the full run.")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/quantization") / config_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tflite_models").mkdir(parents=True, exist_ok=True)

    selected_indices = resolve_band_indices(args.wavelengths_nm, ALL_WAVELENGTHS_INT)
    logger.info("Wavelengths: %s -> band indices %s", args.wavelengths_nm, selected_indices)

    logger.info("Loading dataset...")
    X_hsi, X_meta, _, y, batch_ids, sample_ids, slice_info = load_dataset_hsi_2s()
    X_hsi_selected = X_hsi[..., selected_indices]
    logger.info("Dataset: %d sequences, X_hsi=%s, X_meta=%s", len(y), X_hsi_selected.shape, X_meta.shape)

    # Reference mask for accuracy metrics (paper protocol: reference samples only).
    ref_mask_all = reference_mask(sample_ids)

    fold_results_orig, agg_results_orig = load_original_results(config_dir)
    fp32_rmse_arr = extract_fold_metric_array(fold_results_orig, args.model_key, "rmse")
    fp32_r2_arr = extract_fold_metric_array(fold_results_orig, args.model_key, "r2")
    fp32_rpd_arr = extract_fold_metric_array(fold_results_orig, args.model_key, "rpd")

    quant_metrics_per_fold = []
    fp32_latency_per_fold = []
    quant_latency_per_fold = []
    model_info_per_fold = []

    for fold_idx, (train_mask, val_mask, val_batch) in enumerate(lobos_folds_for_sequences(batch_ids)):
        fold_num = fold_idx + 1
        logger.info("=" * 60)
        logger.info("Fold %d (Batch %d held out)", fold_num, val_batch)

        X_hsi_val = X_hsi_selected[val_mask]
        X_meta_val = X_meta[val_mask]
        y_val = y[val_mask]
        ref_val = ref_mask_all[val_mask]

        keras_model_path = config_dir / "models" / f"fold_{fold_num}.h5"
        tflite_path = output_dir / "tflite_models" / f"fold_{fold_num}_int8.tflite"

        # STEP 1: quantize. If --diagnostic-only, this call prints the
        # corrected size/latency comparison for fold 1 and exits.
        info = quantize_model_int8(
            keras_model_path, tflite_path,
            X_hsi_val=X_hsi_val, X_meta_val=X_meta_val,
            diagnostic_only=args.diagnostic_only,
        )
        info["fold"] = fold_num
        model_info_per_fold.append(info)

        # STEP 3 & 6: evaluate quantized model + latency
        quant_metrics = evaluate_tflite_model(tflite_path, X_hsi_val, X_meta_val, y_val, ref_mask=ref_val)
        quant_metrics["fold"] = fold_num
        quant_metrics_per_fold.append(quant_metrics)
        quant_latency_per_fold.append({
            "fold": fold_num,
            "latency_ms_mean": quant_metrics["latency_ms_mean"],
            "latency_ms_std": quant_metrics["latency_ms_std"],
        })

        # STEP 6: full-precision latency (fair, tf.function-based, sample-by-sample)
        fp32_eval = evaluate_keras_model_latency(keras_model_path, X_hsi_val, X_meta_val, y_val, ref_mask=ref_val)
        fp32_latency_per_fold.append({
            "fold": fold_num,
            "latency_ms_mean": fp32_eval["latency_ms_mean"],
            "latency_ms_std": fp32_eval["latency_ms_std"],
        })

        logger.info("  Fold %d | FP32 RMSE=%.4f  INT8 RMSE=%.4f  |  FP32 latency=%.3fms  INT8 latency=%.3fms",
                    fold_num, fp32_rmse_arr[fold_idx], quant_metrics["rmse"],
                    fp32_eval["latency_ms_mean"], quant_metrics["latency_ms_mean"])

    quant_rmse_arr = np.array([m["rmse"] for m in quant_metrics_per_fold])
    quant_r2_arr = np.array([m["r2"] for m in quant_metrics_per_fold])
    quant_rpd_arr = np.array([m["rpd"] for m in quant_metrics_per_fold])
    quant_mae_arr = np.array([m["mae"] for m in quant_metrics_per_fold])

    quant_agg = {
        "rmse_mean": float(quant_rmse_arr.mean()), "rmse_std": float(quant_rmse_arr.std()),
        "mae_mean": float(quant_mae_arr.mean()), "mae_std": float(quant_mae_arr.std()),
        "r2_mean": float(quant_r2_arr.mean()), "r2_std": float(quant_r2_arr.std()),
        "rpd_mean": float(quant_rpd_arr.mean()), "rpd_std": float(quant_rpd_arr.std()),
    }

    bootstrap_rows = []
    for metric_name, fp32_arr, quant_arr in [
        ("rmse", fp32_rmse_arr, quant_rmse_arr),
        ("r2", fp32_r2_arr, quant_r2_arr),
        ("rpd", fp32_rpd_arr, quant_rpd_arr),
    ]:
        result = paired_fold_bootstrap(fp32_arr, quant_arr, n_boot=args.n_boot)
        bootstrap_rows.append({
            "metric": metric_name,
            "fp32_mean": float(fp32_arr.mean()),
            "int8_mean": float(quant_arr.mean()),
            "delta_fp32_minus_int8": result["observed_delta"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "p_value": result["p_value"],
        })
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df["p_value_holm"] = holm_bonferroni(bootstrap_df["p_value"].tolist())
    bootstrap_df["significant_holm_0.05"] = bootstrap_df["p_value_holm"] < 0.05

    model_info_df = pd.DataFrame(model_info_per_fold)
    fp32_latency_df = pd.DataFrame(fp32_latency_per_fold).rename(
        columns={"latency_ms_mean": "fp32_latency_ms_mean", "latency_ms_std": "fp32_latency_ms_std"})
    quant_latency_df = pd.DataFrame(quant_latency_per_fold).rename(
        columns={"latency_ms_mean": "int8_latency_ms_mean", "latency_ms_std": "int8_latency_ms_std"})
    latency_df = fp32_latency_df.merge(quant_latency_df, on="fold")

    # Corrected size metrics: raw weight bytes (n_params x 4) instead of .h5 file size
    model_info_df["true_fp32_size_mb"] = model_info_df["n_params"] * 4 / (1024 ** 2)
    model_info_df["true_compression_ratio"] = model_info_df["true_fp32_size_mb"] / model_info_df["int8_size_mb"]

    deployment_summary = {
        "n_params": int(model_info_df["n_params"].iloc[0]),
        "fp32_size_mb_mean_h5_INFLATED": float(model_info_df["fp32_size_mb"].mean()),
        "true_fp32_size_mb_mean": float(model_info_df["true_fp32_size_mb"].mean()),
        "int8_size_mb_mean": float(model_info_df["int8_size_mb"].mean()),
        "compression_ratio_mean_h5_INFLATED": float(model_info_df["compression_ratio"].mean()),
        "true_compression_ratio_mean": float(model_info_df["true_compression_ratio"].mean()),
        "fp32_latency_ms_mean": float(latency_df["fp32_latency_ms_mean"].mean()),
        "fp32_latency_ms_std": float(latency_df["fp32_latency_ms_mean"].std()),
        "int8_latency_ms_mean": float(latency_df["int8_latency_ms_mean"].mean()),
        "int8_latency_ms_std": float(latency_df["int8_latency_ms_mean"].std()),
        "speedup_ratio_mean": float(
            (latency_df["fp32_latency_ms_mean"] / latency_df["int8_latency_ms_mean"]).mean()
        ),
        "fp32_metrics": agg_results_orig.get(args.model_key, {}),
        "int8_metrics": quant_agg,
    }

    with open(output_dir / "quantized_fold_results.json", "w") as f:
        json.dump(quant_metrics_per_fold, f, indent=2, default=str)

    with open(output_dir / "quantized_aggregate_results.json", "w") as f:
        json.dump(quant_agg, f, indent=2)

    bootstrap_df.to_csv(output_dir / "bootstrap_fp32_vs_int8.csv", index=False)
    model_info_df.to_csv(output_dir / "model_info_per_fold.csv", index=False)
    latency_df.to_csv(output_dir / "latency_per_fold.csv", index=False)

    with open(output_dir / "deployment_summary.json", "w") as f:
        json.dump(deployment_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("QUANTIZATION SUMMARY (corrected numbers)")
    logger.info("=" * 60)
    logger.info("Params: %d", deployment_summary["n_params"])
    logger.info("Size: TRUE FP32=%.4f MB -> INT8=%.4f MB (%.2fx smaller)",
                deployment_summary["true_fp32_size_mb_mean"], deployment_summary["int8_size_mb_mean"],
                deployment_summary["true_compression_ratio_mean"])
    logger.info("Latency: FP32=%.3f ms -> INT8=%.3f ms (%.2fx speedup)",
                deployment_summary["fp32_latency_ms_mean"], deployment_summary["int8_latency_ms_mean"],
                deployment_summary["speedup_ratio_mean"])
    logger.info("RMSE: FP32=%.4f -> INT8=%.4f", fp32_rmse_arr.mean(), quant_rmse_arr.mean())
    logger.info("R2:   FP32=%.4f -> INT8=%.4f", fp32_r2_arr.mean(), quant_r2_arr.mean())
    logger.info("RPD:  FP32=%.4f -> INT8=%.4f", fp32_rpd_arr.mean(), quant_rpd_arr.mean())
    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
