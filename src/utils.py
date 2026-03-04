"""
utils.py – Shared utility functions for AegisNet.

Centralising helpers here keeps other modules clean and avoids duplication.
"""

import os
import random
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a named logger that writes to both stdout and a file.

    Args:
        name:    Logger name (usually __name__ of the calling module).
        log_dir: Directory where the log file is saved.

    Returns:
        Configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    if logger.handlers:           # Prevent duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, path: str) -> None:
    """Save a matplotlib figure, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"Figure saved -> {path}")


def plot_loss_curve(train_losses: list, save_path: str = None) -> plt.Figure:
    """Plot training loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        save_fig(fig, save_path)
    return fig
