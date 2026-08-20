"""Hyperspectral-image dataloader for 2-step sequential CNN-LSTM.

Reads ground truth (y.csv), metadata (meta_data.csv), and hyperspectral images
(data/hsi/*.mat) and builds temporal sequences for 2-step prediction.

This dataloader is designed for the 2-step sequential CNN-LSTM model that:
- Input: HSI and metadata from steps t-1 and t
- Output: Moisture content at step t+1

This module is independent of other dataloaders (no circular imports).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import h5py
from scipy.io import loadmat
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _PROJECT_ROOT / "data"
Y_PATH = DATA_DIR / "y.csv"
META_PATH = DATA_DIR / "meta_data.csv"
HSI_DIR = DATA_DIR / "hsi"

# Targets for 2-step prediction - moisture only
Y_TARGETS_2S: List[str] = [
    "Moisture Content",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_hsi_mat(mat_path: Path) -> np.ndarray | None:
    """Read one HSI .mat file (HDF5 or MATLAB v5) and return a ``(H, W, D)`` float32 array."""
    if not mat_path.exists():
        logger.warning("HSI file not found: %s", mat_path)
        return None

    data = None

    # Try HDF5 first
    try:
        with h5py.File(mat_path, "r") as f:
            if "slice_data" in f:
                data = np.array(f["slice_data"])
            else:
                keys = list(f.keys())
                if len(keys) == 1:
                    data = np.array(f[keys[0]])
    except Exception:
        pass

    # Fallback to scipy for MATLAB v5 files
    if data is None:
        try:
            mat = loadmat(str(mat_path))
            if "slice_data" in mat:
                data = mat["slice_data"]
        except Exception:
            pass

    if data is None:
        logger.warning("Could not read HSI data from %s", mat_path)
        return None

    data = np.transpose(data, (1, 2, 0)) if data.ndim == 3 else data
    return data.astype(np.float32)


def _pad_to_target(image: np.ndarray, target_shape: tuple[int, int, int] = (64, 52, 112)) -> np.ndarray:
    """Zero-pad ``image`` to ``target_shape`` along each axis."""
    pad = [(0, max(0, target_shape[i] - image.shape[i])) for i in range(image.ndim)]
    return np.pad(image, pad, mode="constant")


def load_ground_truth() -> pd.DataFrame:
    """Join y.csv, meta_data.csv on sample_id (skip spectra)."""
    y_df = pd.read_csv(Y_PATH)
    y_df.columns = [c.strip() for c in y_df.columns]
    meta_df = pd.read_csv(META_PATH)
    meta_df.columns = [c.strip() for c in meta_df.columns]
    joined = y_df.merge(meta_df, on="sample_id", how="inner")
    joined = joined.sort_values(["Batch", "Time step", "Slice ID"]).reset_index(drop=True)
    if len(joined) != 1056:
        raise ValueError(f"Joined ground truth has {len(joined)} rows; expected 1056")
    logger.info("Joined ground truth (no spectra): %d samples", len(joined))
    return joined


def load_reference_flags() -> dict[str, bool]:
    """Map sample_id -> has_reference (bool) from ``meta_data.csv``.

    The ``Reference`` column marks samples that carry a reference (ground-truth)
    moisture measurement. Accuracy metrics are computed only on samples whose
    target has a reference, matching the evaluation protocol in the paper.
    """
    meta_df = pd.read_csv(META_PATH)
    meta_df.columns = [c.strip() for c in meta_df.columns]
    ref = meta_df.set_index("sample_id")["Reference"]
    return {str(sid): bool(int(v)) for sid, v in ref.items()}


_REFERENCE_FLAGS: dict[str, bool] | None = None


def reference_mask(sample_ids) -> np.ndarray:
    """Boolean mask aligned to ``sample_ids``.

    True where a sequence's target sample (the t+1 observation of each 2-step
    sequence) has a reference measurement. Use this to restrict accuracy metrics
    to samples with a reference, as done in the paper.

    Args:
        sample_ids: list of target sample IDs, as returned by ``load_dataset_hsi_2s``.

    Returns:
        np.ndarray of bool with ``len(sample_ids)`` entries.
    """
    global _REFERENCE_FLAGS
    if _REFERENCE_FLAGS is None:
        _REFERENCE_FLAGS = load_reference_flags()
    return np.array([bool(_REFERENCE_FLAGS.get(str(sid), False)) for sid in sample_ids])


def _load_targets_2s(gt: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame containing only the 2-step target columns."""
    return gt[["sample_id"] + Y_TARGETS_2S]


# ── Sequence builder for 2-step prediction ────────────────────────────────────
def build_slice_sequences_hsi_2s(
    ground_truth: pd.DataFrame,
    targets: pd.DataFrame,
    seq_length: int = 2,
    hsi_dir: Path | None = None,
    std_shape: tuple[int, int, int] = (64, 52, 112),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[Tuple[int, int, int]]]:
    """Build temporal sequences for 2-step prediction (t-1, t) -> (t+1 targets).

    Args:
        ground_truth: joined DataFrame from ``load_ground_truth()``.
        targets:      target DataFrame with moisture content.
        seq_length:   number of consecutive time steps per sequence (must be 2).
        hsi_dir:      directory containing ``{sample_id}.mat`` files.
        std_shape:    target spatial-spectral shape for padding.

    Returns:
        (X_hsi, X_meta, is_padded, y, batch_ids, sample_ids, slice_info)
        X_hsi shape: ``(n_sequences, seq_length, 64, 52, 112)`` - steps t-1 and t
        X_meta shape: ``(n_sequences, 6)`` - meta[t-1] + meta[t] flattened
        y shape: ``(n_sequences, 1)`` - moisture content at t+1
    """
    if hsi_dir is None:
        hsi_dir = HSI_DIR

    if seq_length != 2:
        raise ValueError(f"seq_length must be 2 for 2-step prediction, got {seq_length}")

    meta_cols = ["Drying Time", "Drying Temp", "AirVelocity"]
    sequences: List[dict] = []

    for batch in sorted(ground_truth["Batch"].unique()):
        batch_df = ground_truth[ground_truth["Batch"] == batch]
        for slice_id in sorted(batch_df["Slice ID"].unique()):
            slice_df = batch_df[batch_df["Slice ID"] == slice_id].sort_values("Time step")
            indices = slice_df.index
            n_steps = len(slice_df)

            # We need t-1, t for input and t+1 for target
            # So we can only form sequences up to n_steps - 2
            for i in range(n_steps - 2):
                # Input: steps t-1 (i) and t (i+1)
                meta_t_minus_1 = slice_df.loc[indices[i], meta_cols].values.astype(np.float32)
                meta_t = slice_df.loc[indices[i + 1], meta_cols].values.astype(np.float32)

                # Target: t+1 (i+2)
                target_t_plus_1 = targets.iloc[indices[i + 2]]
                y_vals = target_t_plus_1[Y_TARGETS_2S].values.astype(np.float32)

                # Sample info at t+1
                sample_id_t_plus_1 = str(slice_df.loc[indices[i + 2], "sample_id"])

                # Build history for steps t-1 and t
                history_meta: List[np.ndarray] = []
                history_hsi: List[np.ndarray | None] = []
                is_padded_step: List[float] = []

                for j in range(1, seq_length):
                    prev_idx = (i + 1) - j  # Start from i (t), go back to i-1 (t-1)
                    if prev_idx >= 0:
                        prev_id = str(slice_df.loc[indices[prev_idx], "sample_id"])
                        prev_meta = slice_df.loc[indices[prev_idx], meta_cols].values.astype(np.float32)
                        history_meta.append(prev_meta)
                        is_padded_step.append(0.0)
                        prev_mat = hsi_dir / f"{prev_id}.mat"
                        prev_hsi = _load_hsi_mat(prev_mat)
                        if prev_hsi is not None:
                            history_hsi.append(_pad_to_target(prev_hsi, std_shape))
                        else:
                            history_hsi.append(None)
                    else:
                        history_meta.append(np.zeros_like(meta_t))
                        history_hsi.append(np.zeros(std_shape, dtype=np.float32))
                        is_padded_step.append(1.0)

                history_meta.reverse()
                history_hsi.reverse()
                is_padded_step.reverse()

                # Pad mask for 2 input steps
                pad_mask = np.array(is_padded_step)

                sequences.append({
                    "meta": np.array(history_meta + [meta_t]),  # Only t-1, t
                    "hsi": history_hsi + [None, None],  # placeholders for t-1, t
                    "y": y_vals,
                    "batch": int(batch_df.loc[indices[i + 2], "Batch"]),
                    "sample_id": sample_id_t_plus_1,
                    "slice_id": int(slice_df.loc[indices[i + 2], "Slice ID"]),
                    "time_step": int(slice_df.loc[indices[i + 2], "Time step"]),
                    "is_padded": pad_mask,
                })

    # Now fill in the HSI images for t-1 and t steps
    for seq in sequences:
        for j in range(seq_length):
            if seq["hsi"][j] is None:
                # Extract the correct sample_id from history_meta
                # We stored sample info at t+1, need to find t-1 and t sample IDs
                pass  # Will be filled below

    # Rebuild with proper sample IDs
    sequences: List[dict] = []

    for batch in sorted(ground_truth["Batch"].unique()):
        batch_df = ground_truth[ground_truth["Batch"] == batch]
        for slice_id in sorted(batch_df["Slice ID"].unique()):
            slice_df = batch_df[batch_df["Slice ID"] == slice_id].sort_values("Time step")
            indices = slice_df.index
            n_steps = len(slice_df)

            for i in range(n_steps - 2):
                meta_t_minus_1 = slice_df.loc[indices[i], meta_cols].values.astype(np.float32)
                meta_t = slice_df.loc[indices[i + 1], meta_cols].values.astype(np.float32)
                meta_t_plus_1 = slice_df.loc[indices[i + 2], meta_cols].values.astype(np.float32)

                target_t_plus_1 = targets.iloc[indices[i + 2]]
                y_vals = target_t_plus_1[Y_TARGETS_2S].values.astype(np.float32)

                sample_id_t_plus_1 = str(slice_df.loc[indices[i + 2], "sample_id"])

                # Get sample IDs for t-1 and t
                sample_id_t_minus_1 = str(slice_df.loc[indices[i], "sample_id"])
                sample_id_t = str(slice_df.loc[indices[i + 1], "sample_id"])

                # Load HSI for t-1 and t
                hsi_t_minus_1 = _load_hsi_mat(hsi_dir / f"{sample_id_t_minus_1}.mat")
                hsi_t = _load_hsi_mat(hsi_dir / f"{sample_id_t}.mat")

                if hsi_t_minus_1 is None:
                    hsi_t_minus_1 = np.zeros(std_shape, dtype=np.float32)
                if hsi_t is None:
                    hsi_t = np.zeros(std_shape, dtype=np.float32)

                sequences.append({
                    "meta": np.array([meta_t_minus_1, meta_t]),  # Only t-1, t (no future metadata)
                    "hsi": [_pad_to_target(hsi_t_minus_1, std_shape), _pad_to_target(hsi_t, std_shape)],
                    "y": y_vals,
                    "batch": int(batch_df.loc[indices[i + 2], "Batch"]),
                    "sample_id": sample_id_t_plus_1,
                    "slice_id": int(slice_df.loc[indices[i + 2], "Slice ID"]),
                    "time_step": int(slice_df.loc[indices[i + 2], "Time step"]),
                    "is_padded": np.array([0.0, 0.0]),  # 2 input steps only
                })

    X_meta = np.array([s["meta"] for s in sequences])  # (n, seq_length, 3) per-timestep
    X_hsi = np.array([s["hsi"] for s in sequences])
    is_padded = np.array([s["is_padded"] for s in sequences])
    y = np.array([s["y"] for s in sequences])
    batch_ids = np.array([s["batch"] for s in sequences], dtype=np.int64)
    sample_ids = [str(s["sample_id"]) for s in sequences]
    slice_info = [
        (s["batch"], s["slice_id"], s["time_step"]) for s in sequences
    ]

    n_batches = len({s["batch"] for s in sequences})
    total_steps = len(sequences) * seq_length
    total_padded = int(is_padded.sum())
    logger.info(
        "Built %d HSI-2s sequences (length=%d) from %d batches -- "
        "%d/%d steps (%.1f%%) padded",
        len(sequences), seq_length, n_batches,
        total_padded, total_steps, 100 * total_padded / total_steps,
    )
    return X_hsi, X_meta, is_padded, y, batch_ids, sample_ids, slice_info


# ── Convenience loader ───────────────────────────────────────────────────────
def load_dataset_hsi_2s(
    seq_length: int = 2,
    hsi_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[Tuple[int, int, int]]]:
    """One-call 2-step HSI loader: ground truth -> sequences.

    Returns:
        (X_hsi, X_meta, is_padded, y, batch_ids, sample_ids, slice_info)
        y has shape (n_sequences, 1) for moisture content prediction
    """
    ground_truth = load_ground_truth()
    targets = _load_targets_2s(ground_truth)
    return build_slice_sequences_hsi_2s(ground_truth, targets, seq_length, hsi_dir)


# ── LOBO CV for sequences ────────────────────────────────────────────────────
def lobos_folds_for_sequences(
    batch_ids: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Yield (train_mask, val_mask, val_batch_id) for sequence data.

    Args:
        batch_ids: Array of batch IDs for each sequence (from load_dataset_hsi_2s)

    Returns:
        List of (train_mask, val_mask, val_batch_id) tuples
    """
    all_batches = sorted(np.unique(batch_ids))
    n_sequences = len(batch_ids)
    result = []

    for val_batch in all_batches:
        val_mask = batch_ids == val_batch
        train_mask = ~val_mask
        result.append((train_mask, val_mask, int(val_batch)))

    return result


# ── Inner validation split for sequences ──────────────────────────────────────
def make_inner_val_mask_for_sequences(
    batch_ids: np.ndarray,
    train_mask: np.ndarray,
    n_val_batches: int = 2,
    seed: int = 42,
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
    val_batches = rng.choice(train_batches, size=n_val_batches, replace=False)
    val_batches = set(val_batches.tolist())

    inner_val_mask = train_mask & np.isin(batch_ids, list(val_batches))
    inner_train_mask = train_mask & ~np.isin(batch_ids, list(val_batches))
    return inner_train_mask, inner_val_mask


def get_lobo_folds_with_inner_val(
    batch_ids: np.ndarray,
    n_val_batches: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Generate LOBO CV folds with inner validation split.

    Args:
        batch_ids: Array of batch IDs for all sequences.
        n_val_batches: Number of training batches to hold out as inner validation.
        seed: Random seed for reproducible inner split.

    Returns:
        List of dicts with keys:
            - 'val_batch': the test batch ID
            - 'train_mask': mask for outer training batches
            - 'val_mask': mask for test batch
            - 'inner_train_mask': mask for inner training (87.5% of train)
            - 'inner_val_mask': mask for inner validation (12.5% of train)
            - 'inner_val_batches': list of batch IDs used for inner validation
    """
    all_batches = sorted(np.unique(batch_ids))
    result = []

    for val_batch in all_batches:
        val_mask = batch_ids == val_batch
        train_mask = ~val_mask

        # Split training into inner-train and inner-val
        inner_train_mask, inner_val_mask = make_inner_val_mask_for_sequences(
            batch_ids, train_mask, n_val_batches=n_val_batches, seed=seed
        )
        inner_val_batches = sorted(
            [b for b in all_batches if inner_val_mask[batch_ids == b].any()]
        )

        result.append({
            'val_batch': int(val_batch),
            'train_mask': train_mask,
            'val_mask': val_mask,
            'inner_train_mask': inner_train_mask,
            'inner_val_mask': inner_val_mask,
            'inner_val_batches': inner_val_batches,
        })

    return result


# ── LOBO CV (imported alias) ──────────────────────────────────────────────────
def lobos_folds_for(
    ground_truth: pd.DataFrame,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Yield (train_mask, val_mask, val_batch_id) using actual batch IDs."""
    all_batches = sorted(ground_truth["Batch"].unique())
    all_indices = ground_truth.index.values
    result = []
    for val_batch in all_batches:
        val_mask = ground_truth.loc[all_indices, "Batch"] == val_batch
        train_mask = ~val_mask
        result.append((train_mask.values, val_mask.values, int(val_batch)))
    return result
