"""2-Step Sequential CNN-LSTM for HSI-based cocoyam moisture prediction.

This model takes two consecutive time steps (t-1 and t) as input and predicts
moisture content at t+1.

Architecture:
1. TimeDistributed 2D CNN for spatial-spectral feature extraction from HSI sequence
2. Per-timestep spatial projection to a fixed embedding size
3. TimeDistributed metadata encoding, aligned per timestep
4. Per-timestep concatenation of spatial + metadata embeddings (no global flatten/reshape)
5. LSTM for temporal dynamics over the correctly aligned 2-step sequence
6. Single Dense(1) layer for moisture prediction

Training:
- Loss: MSE for moisture prediction
- Optimizer: Adam with configurable learning rate
- Early stopping: patience-based

Note: This is a custom research architecture trained from scratch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

tf.random.set_seed(50)
np.random.seed(50)


def load_fold_hyperparams(fold_idx: int, project_root: Path | None = None) -> dict:
    """Load best hyperparameters for a given fold.

    Args:
        fold_idx: 1-based fold index (1-16)
        project_root: Project root directory

    Returns:
        Dictionary of best hyperparameters for this fold

    Raises:
        FileNotFoundError: If hyperparameter file doesn't exist
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    hyperparams_dir = project_root / "outputs" / "hsi_2s" / "hyperparams" / f"fold_{fold_idx}"
    params_file = hyperparams_dir / "best_params.json"

    if not params_file.exists():
        raise FileNotFoundError(
            f"Hyperparameter file not found: {params_file}. "
            f"Run tune_cnn_lstm_hsi_2s.py first to tune hyperparameters."
        )

    with open(params_file, "r") as f:
        data = json.load(f)

    return data["best_params"]


class CNNLSTMHSI2SModel:
    """2-Step Sequential CNN-LSTM for HSI-based cocoyam moisture prediction.

    CUSTOM ARCHITECTURE TRAINED FROM SCRATCH (not a foundation model).

    Input:
        image_input: (batch_size, 2, 64, 52, 112) - HSI at t-1 and t
        meta_input: (batch_size, 2, 3) - metadata at t-1 and t, kept per-timestep

    Output:
        Prediction (batch_size, 1) for t+1 moisture content

    Architecture:
        1. TimeDistributed Conv2D on HSI sequence (2 steps)
        2. TimeDistributed Flatten + Dense projection -> per-step spatial embedding
        3. TimeDistributed Dense encoding -> per-step metadata embedding
        4. Concatenate spatial + metadata embeddings per timestep (axis=-1, no reshape)
        5. LSTM over the resulting (batch, 2, spatial_embed_dim + meta_embed_dim) sequence
        6. Single Dense(1) layer for moisture prediction

    Note:
        Metadata is now passed per timestep as shape (2, 3), not flattened to (6,).
        This guarantees the LSTM's two timesteps each receive their own correctly
        aligned spatial and metadata embedding, with no cross-timestep mixing.
    """

    def __init__(
        self,
        n_bands: int = 112,
        hsi_h: int = 64,
        hsi_w: int = 52,
        meta_dim_per_step: int = 3,  # 3 metadata features per timestep
        seq_length: int = 2,
        n_conv_layers: int = 2,
        conv_filters: int = 64,
        spatial_embed_dim: int = 128,
        meta_embed_dim: int = 16,
        n_lstm_layers: int = 2,
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 16,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        random_state: int = 50,
    ):
        """Initialize 2-step CNN-LSTM model.

        Args:
            n_bands: Number of spectral bands (default 112)
            hsi_h: Height of HSI image (default 64)
            hsi_w: Width of HSI image (default 52)
            meta_dim_per_step: Number of metadata features per timestep (default 3)
            seq_length: Number of time steps in sequence (default 2)
            n_conv_layers: Number of Conv2D layers (default 2)
            conv_filters: Number of filters in each Conv2D layer (default 64)
            spatial_embed_dim: Dimension of per-step spatial projection (default 128)
            meta_embed_dim: Dimension of per-step metadata projection (default 16)
            n_lstm_layers: Number of LSTM layers (default 2)
            lstm_units: Number of units in LSTM layers (default 128)
            dropout_rate: Dropout rate (default 0.3)
            learning_rate: Learning rate for optimizer (default 0.001)
            batch_size: Batch size for training (default 16)
            epochs: Maximum training epochs (default 100)
            early_stopping_patience: Early stopping patience (default 15)
            random_state: Random seed for reproducibility (default 50)
        """
        self.n_bands = n_bands
        self.hsi_h = hsi_h
        self.hsi_w = hsi_w
        self.meta_dim_per_step = meta_dim_per_step
        self.seq_length = seq_length
        self.n_conv_layers = n_conv_layers
        self.conv_filters = conv_filters
        self.spatial_embed_dim = spatial_embed_dim
        self.meta_embed_dim = meta_embed_dim
        self.n_lstm_layers = n_lstm_layers
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        self._model: keras.Model | None = None
        self._history: Any | None = None

        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def _build_spatial_spectral_enhancement(self, image_input: layers.Layer) -> layers.Layer:
        """Build 2D CNN with TimeDistributed for spatial-spectral enhancement.

        Uses 2D convolution to extract spatial features from HSI cubes at each time step.
        Output retains the timestep axis: (batch, seq_length, conv_feature_dim).
        """
        x = image_input

        for i in range(self.n_conv_layers):
            x = layers.TimeDistributed(
                layers.Conv2D(
                    filters=self.conv_filters,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="relu",
                    name=f"cnn_spatial_conv{i + 1}",
                ),
                name=f"cnn_spatial_td{i + 1}",
            )(x)
            x = layers.TimeDistributed(
                layers.BatchNormalization(name=f"cnn_spatial_bn{i + 1}"),
                name=f"cnn_spatial_bn_td{i + 1}",
            )(x)
            x = layers.TimeDistributed(
                layers.Dropout(self.dropout_rate, name=f"cnn_spatial_drop{i + 1}"),
                name=f"cnn_spatial_drop_td{i + 1}",
            )(x)
            x = layers.TimeDistributed(
                layers.AveragePooling2D(pool_size=(2, 2), name=f"cnn_spatial_pool{i + 1}"),
                name=f"cnn_spatial_pool_td{i + 1}",
            )(x)

        # Flatten spatial features WITHIN each time step (timestep axis preserved)
        x = layers.TimeDistributed(
            layers.Flatten(name="cnn_spatial_flat"),
            name="cnn_spatial_flat_td",
        )(x)
        # x shape: (batch, seq_length, per_step_flat_dim)
        return x

    def _build_model(self) -> keras.Model:
        """Build the complete 2-step HSI CNN-LSTM model with per-timestep fusion.

        Input:
            image_input: (2, 64, 52, 112) - HSI at t-1 and t
            meta_input: (2, 3) - metadata at t-1 and t, per-timestep

        Output:
            Moisture prediction (batch_size, 1) at t+1
        """
        # Inputs
        image_input = layers.Input(
            shape=(self.seq_length, self.hsi_h, self.hsi_w, self.n_bands),
            name="image_input",
        )
        meta_input = layers.Input(
            shape=(self.seq_length, self.meta_dim_per_step),
            name="meta_input",
        )

        # Branch 1: Spatial-Spectral feature extraction (TimeDistributed), timestep axis preserved
        spatial_features = self._build_spatial_spectral_enhancement(image_input)
        # spatial_features shape: (batch, seq_length, per_step_flat_dim)

        # Project spatial features to a fixed embedding size, matched in scale to metadata
        spatial_embed = layers.TimeDistributed(
            layers.Dense(self.spatial_embed_dim, activation="relu", name="spatial_proj"),
            name="spatial_proj_td",
        )(spatial_features)
        # spatial_embed shape: (batch, seq_length, spatial_embed_dim)

        # Branch 2: Metadata encoding, applied identically and independently per timestep
        meta_encoded = layers.TimeDistributed(
            layers.Dense(32, activation="relu", name="meta_dense1"),
            name="meta_dense1_td",
        )(meta_input)
        meta_encoded = layers.TimeDistributed(
            layers.Dense(self.meta_embed_dim, activation="relu", name="meta_dense2"),
            name="meta_dense2_td",
        )(meta_encoded)
        # meta_encoded shape: (batch, seq_length, meta_embed_dim)

        # Fuse spatial + metadata embeddings PER TIMESTEP (feature axis concat, no reshape)
        fused = layers.Concatenate(axis=-1, name="fuse_spatial_meta_per_step")(
            [spatial_embed, meta_encoded]
        )
        # fused shape: (batch, seq_length, spatial_embed_dim + meta_embed_dim)
        # Each of the 2 timesteps now carries its OWN correctly aligned spatial + meta embedding.

        # Temporal encoder (LSTM) - processes the aligned 2-timestep sequence directly
        x = fused
        for i in range(self.n_lstm_layers):
            return_sequences = i < self.n_lstm_layers - 1
            x = layers.LSTM(
                self.lstm_units,
                return_sequences=return_sequences,
                name=f"lstm_{i + 1}",
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f"lstm_drop_{i + 1}")(x)

        # Single output head for moisture prediction
        output = layers.Dense(1, activation="linear", name="output_moisture")(x)

        model = keras.Model(
            inputs={"image_input": image_input, "meta_input": meta_input},
            outputs=output,
            name="CNNLSTMHSI_2S",
        )

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"],
        )

        return model

    def fit(
        self,
        X_hsi: np.ndarray,
        y: np.ndarray,
        X_meta: np.ndarray | None = None,
        validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        verbose: int = 1,
    ) -> "CNNLSTMHSI2SModel":
        """Train the 2-step HSI CNN-LSTM model.

        Args:
            X_hsi: Training images of shape (n_samples, 2, 64, 52, 112)
            y: Training targets of shape (n_samples, 1) for moisture content
            X_meta: Training metadata of shape (n_samples, 2, 3), per-timestep
            validation_data: Optional (X_hsi_val, y_val, X_meta_val)
            verbose: Verbosity mode (0=quiet, 1=progress bar, 2=one line per epoch)

        Returns:
            self
        """
        if self._model is None:
            self._model = self._build_model()

        if X_meta is None:
            X_meta = np.zeros(
                (X_hsi.shape[0], self.seq_length, self.meta_dim_per_step), dtype=np.float32
            )

        val_input = None
        if validation_data is not None:
            X_hsi_val, y_val, X_meta_val = validation_data
            if X_meta_val is None:
                X_meta_val = np.zeros(
                    (X_hsi_val.shape[0], self.seq_length, self.meta_dim_per_step),
                    dtype=np.float32,
                )
            val_input = (
                {"image_input": X_hsi_val, "meta_input": X_meta_val},
                y_val,
            )

        callbacks = []
        if val_input is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=verbose,
                )
            )

        train_input = {"image_input": X_hsi, "meta_input": X_meta}
        self._history = self._model.fit(
            train_input,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=val_input,
            callbacks=callbacks,
            verbose=verbose,
        )

        return self

    def predict(
        self, X_hsi: np.ndarray, X_meta: np.ndarray | None = None, verbose: int = 0
    ) -> np.ndarray:
        """Make predictions.

        Args:
            X_hsi: Input images of shape (n_samples, 2, 64, 52, 112)
            X_meta: Input metadata of shape (n_samples, 2, 3), per-timestep
            verbose: Verbosity mode

        Returns:
            Predictions of shape (n_samples, 1) for moisture content
        """
        assert self._model is not None, "Model must be fitted before predicting"

        if X_meta is None:
            X_meta = np.zeros(
                (X_hsi.shape[0], self.seq_length, self.meta_dim_per_step), dtype=np.float32
            )

        preds = self._model.predict(
            {"image_input": X_hsi, "meta_input": X_meta}, verbose=verbose
        )

        preds = np.clip(preds, 0, None)

        return preds

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "n_bands": self.n_bands,
            "hsi_h": self.hsi_h,
            "hsi_w": self.hsi_w,
            "meta_dim_per_step": self.meta_dim_per_step,
            "seq_length": self.seq_length,
            "n_conv_layers": self.n_conv_layers,
            "conv_filters": self.conv_filters,
            "spatial_embed_dim": self.spatial_embed_dim,
            "meta_embed_dim": self.meta_embed_dim,
            "n_lstm_layers": self.n_lstm_layers,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "CNNLSTMHSI2SModel":
        """Set model parameters."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def history(self) -> dict:
        """Get training history."""
        if self._history is None:
            return {}
        return {
            "loss": self._history.history.get("loss", []),
            "mae": self._history.history.get("mae", []),
            "val_loss": self._history.history.get("val_loss", []),
            "val_mae": self._history.history.get("val_mae", []),
        }