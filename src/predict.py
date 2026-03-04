"""
predict.py - AegisNet Inference Engine.

Loads a trained Autoencoder checkpoint + scaler and exposes a clean
prediction interface for production use.

Usage:
    from src.predict import AegisNetPredictor
    predictor = AegisNetPredictor()
    result = predictor.predict([0.1, 0.5, ...])
"""

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

# Import the model definition so we can reconstruct the architecture.
from src.model_autoencoder import Autoencoder

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = Path("models/autoencoder_phase1.pt")
SCALER_PATH      = Path("models/scaler.pkl")

# ── Decision threshold ────────────────────────────────────────────────────────
# Reconstruction error above this value is flagged as anomalous.
# Tune this value using the ROC curve from training (e.g., at desired FPR).
ANOMALY_THRESHOLD: float = 0.5


class AegisNetPredictor:
    """
    Production inference wrapper for the AegisNet Autoencoder.

    Handles:
      - Checkpoint loading and architecture reconstruction
      - Scaler loading and feature normalisation
      - Input validation (shape, NaN)
      - CPU/GPU device selection
      - Threshold-based anomaly decision
    """

    def __init__(
        self,
        checkpoint_path: str | Path = MODEL_CHECKPOINT,
        scaler_path: str | Path = SCALER_PATH,
        threshold: float = ANOMALY_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"AegisNetPredictor using device: {self.device}")

        self._load_scaler(scaler_path)
        self._load_model(checkpoint_path)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_scaler(self, path: str | Path) -> None:
        """Load the sklearn StandardScaler saved during training."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Scaler not found at '{path}'. "
                "Run train.py first to generate models/scaler.pkl."
            )
        self.scaler = joblib.load(path)
        logger.info(f"Scaler loaded from {path}")

    def _load_model(self, path: str | Path) -> None:
        """Reconstruct the Autoencoder architecture and load saved weights."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{path}'. "
                "Run train.py first to generate models/autoencoder_phase1.pt."
            )

        # Load checkpoint to CPU first to keep device-agnostic.
        checkpoint = torch.load(path, map_location="cpu")

        input_dim   = checkpoint["input_dim"]
        hidden_dims = checkpoint["hidden_dims"]

        self.input_dim   = input_dim
        self.feature_cols = checkpoint.get("feature_cols", [])

        # Rebuild the exact same architecture used during training.
        self.model = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()  # Disable dropout / batchnorm training behaviour.

        logger.info(
            f"Model loaded from {path} "
            f"(input_dim={input_dim}, layers={hidden_dims}, "
            f"epoch={checkpoint.get('epoch', '?')}, "
            f"final_loss={checkpoint.get('train_loss', '?'):.6f})"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, features: list[float]) -> dict[str, Any]:
        """
        Run inference on a single network flow sample.

        Args:
            features: List of float values representing one network flow.
                      Length must match the model's input_dim.

        Returns:
            {
                "anomaly_score": float,   # MSE reconstruction error
                "is_anomaly":   bool,     # True if score > threshold
                "threshold":    float,    # Decision threshold in use
                "input_dim":    int,      # Expected feature count
            }

        Raises:
            ValueError: If input length is wrong or contains NaN/Inf.
        """
        # ── Input validation ──────────────────────────────────────────────
        if len(features) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {len(features)}."
            )

        arr = np.array(features, dtype=np.float64)

        if np.any(np.isnan(arr)):
            raise ValueError("Input contains NaN values.")

        if np.any(np.isinf(arr)):
            raise ValueError("Input contains infinite values.")

        # ── Scale ─────────────────────────────────────────────────────────
        # Reshape to (1, n_features) for the scaler's expected 2-D input.
        arr_scaled = self.scaler.transform(arr.reshape(1, -1)).astype(np.float32)

        # ── Inference ─────────────────────────────────────────────────────
        tensor = torch.tensor(arr_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            error = self.model.reconstruction_error(tensor)

        anomaly_score = float(error.item())
        is_anomaly    = anomaly_score > self.threshold

        return {
            "anomaly_score": round(anomaly_score, 6),
            "is_anomaly":    is_anomaly,
            "threshold":     self.threshold,
            "input_dim":     self.input_dim,
        }

    def predict_batch(self, feature_matrix: list[list[float]]) -> list[dict[str, Any]]:
        """
        Run inference on multiple samples at once.

        Args:
            feature_matrix: List of feature lists.

        Returns:
            List of prediction dicts (same format as predict()).
        """
        return [self.predict(row) for row in feature_matrix]
