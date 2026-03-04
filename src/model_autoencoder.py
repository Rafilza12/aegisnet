"""
model_autoencoder.py – Symmetric Autoencoder for unsupervised anomaly detection.

Architecture:
  - Encoder: Linear → BatchNorm → ReLU (repeated, shrinking)
  - Bottleneck: smallest latent representation
  - Decoder: mirrors the encoder (expanding)

Why this design?
  - Trained purely on normal traffic, the AE learns to reconstruct benign
    patterns. Anomalies fall outside this learned manifold → high MSE.
  - BatchNorm stabilises training on tabular data with varied feature scales.
  - Dropout acts as regularisation to prevent the AE from becoming an
    identity function (which would make every reconstruction error = 0).
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Symmetric Autoencoder for network flow anomaly detection.

    Args:
        input_dim:   Number of input features.
        hidden_dims: List of encoder layer sizes (decoder is the reverse).
                     Default shrinks from input → 128 → 64 → 32.
        dropout:     Dropout probability applied in each block.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_layers = []
        dec_dims = list(reversed(hidden_dims))
        for i, out_dim in enumerate(dec_dims[1:], start=1):
            dec_layers.extend([
                nn.Linear(dec_dims[i - 1], out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        # Final reconstruction layer: no activation → unbounded output
        dec_layers.append(nn.Linear(dec_dims[-1], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode; return reconstruction."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent bottleneck representation."""
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample mean squared reconstruction error.

        Higher error → sample looks more anomalous.
        """
        recon = self.forward(x)
        # Mean over feature dimension; keeps batch dimension intact
        return torch.mean((x - recon) ** 2, dim=1)
