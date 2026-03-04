"""
train.py – Autoencoder training pipeline for AegisNet Phase 1.

Run:
    python src/train.py

What this script does:
  1. Loads and preprocesses the CIC-IDS2017 dataset.
  2. Trains the autoencoder on NORMAL traffic only.
  3. Evaluates reconstruction error on the full test set.
  4. Plots and saves:
       - Training loss curve
       - Anomaly score distribution
       - ROC curve
"""

import os
import sys

# Allow importing from src/ when script is run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib

from src.data_loader import load_cicids
from src.preprocessing import clean_features, get_splits
from src.model_autoencoder import Autoencoder
from src.utils import set_seed, get_device, get_logger, plot_loss_curve, save_fig

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH     = "data/raw/cic_ids.csv"
HIDDEN_DIMS   = [256, 128, 64, 32]   # Encoder layer sizes; decoder is reversed
DROPOUT       = 0.2
BATCH_SIZE    = 512
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-5                  # L2 regularisation to prevent overfitting
SEED          = 42
RESULTS_DIR   = "experiments/phase1_baseline"

logger = get_logger("train")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss  = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return per-sample reconstruction error for the test set."""
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    errors = model.reconstruction_error(X_tensor)
    return errors.cpu().numpy()


def plot_score_distribution(
    errors: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
) -> None:
    """Visualise anomaly score distribution for normal vs attack samples."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = 100

    ax.hist(errors[y_test == 0], bins=bins, alpha=0.65,
            label="Normal", color="#2196F3", density=True)
    ax.hist(errors[y_test == 1], bins=bins, alpha=0.65,
            label="Attack", color="#F44336", density=True)

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Anomaly Score Distribution – Normal vs Attack", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    save_fig(fig, save_path)
    logger.info(f"Score distribution saved -> {save_path}")


def plot_roc(
    errors: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
) -> None:
    """Plot ROC curve and report AUC."""
    fpr, tpr, _ = roc_curve(y_test, errors)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, lw=2, color="#4CAF50",
            label=f"Autoencoder (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve – Autoencoder Anomaly Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    save_fig(fig, save_path)
    logger.info(f"ROC AUC = {roc_auc:.4f}  |  Saved -> {save_path}")


def main():
    set_seed(SEED)
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("Loading data …")
    df = load_cicids(DATA_PATH)
    df = clean_features(df)
    X_train, X_test, y_test, scaler, feature_cols = get_splits(df)

    input_dim = X_train.shape[1]
    logger.info(f"Input dimensionality: {input_dim}")

    # ── 2. Dataset / DataLoader ───────────────────────────────────────────────
    train_tensor  = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ── 3. Model / Optimiser / Loss ───────────────────────────────────────────
    model     = Autoencoder(input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    logger.info(f"Model:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    # ── 4. Training loop ──────────────────────────────────────────────────────
    train_losses = []
    logger.info("Starting training …")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch [{epoch:>3}/{EPOCHS}]  Loss: {loss:.6f}")

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    logger.info("Evaluating on test set …")
    errors = evaluate(model, X_test, device)

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    plot_loss_curve(train_losses, save_path=os.path.join(RESULTS_DIR, "loss_curve.png"))
    plot_score_distribution(errors, y_test, save_path=os.path.join(RESULTS_DIR, "score_dist.png"))
    plot_roc(errors, y_test, save_path=os.path.join(RESULTS_DIR, "roc_curve.png"))

    # Save scaler as .pkl for production inference (predict.py / api.py)
    scaler_path = os.path.join("models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved -> {scaler_path}")

    # Save model checkpoint
    ckpt_path = os.path.join("models", "autoencoder_phase1.pt")
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state":  model.state_dict(),
        "input_dim":    input_dim,
        "hidden_dims":  HIDDEN_DIMS,
        "feature_cols": feature_cols,
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "epoch":        EPOCHS,
        "train_loss":   train_losses[-1],
    }, ckpt_path)
    logger.info(f"Model checkpoint saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
