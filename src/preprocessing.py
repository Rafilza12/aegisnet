"""
preprocessing.py – Feature engineering and preprocessing for AegisNet.

Responsibilities:
  - Separate numeric features from labels
  - Remove constant / infinite-value columns
  - Scale features with StandardScaler
  - Split data into train / test sets (normal-only train for autoencoder)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Label used in CIC-IDS2017 for benign traffic.
BENIGN_LABEL = "BENIGN"


def clean_features(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """
    Remove non-numeric and degenerate columns.

    'Degenerate' means constant or all-infinite, which provide no signal.

    Args:
        df:        Full DataFrame with labels.
        label_col: Name of the label column (excluded from cleaning).

    Returns:
        DataFrame with only usable numeric feature columns + label column.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    df_feats = df[feature_cols].copy()

    # Keep only numeric columns
    df_feats = df_feats.select_dtypes(include=[np.number])

    # Replace inf/-inf with NaN so they can be caught by dropna
    df_feats.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns where ≥50% of values are NaN (low-quality features)
    threshold = 0.5
    valid_cols = df_feats.columns[df_feats.isna().mean() < threshold].tolist()
    dropped = set(df_feats.columns) - set(valid_cols)
    if dropped:
        logger.info(f"Dropped low-quality columns ({len(dropped)}): {dropped}")

    # Drop columns with zero variance (constant features add no information)
    df_feats = df_feats[valid_cols]
    constant_cols = df_feats.columns[df_feats.std() == 0].tolist()
    if constant_cols:
        logger.info(f"Dropped constant columns ({len(constant_cols)}): {constant_cols}")
        df_feats.drop(columns=constant_cols, inplace=True)

    # Fill remaining NaNs with column median (robust to outliers)
    df_feats.fillna(df_feats.median(), inplace=True)

    # Re-attach labels
    df_clean = df_feats.copy()
    df_clean[label_col] = df[label_col].values
    logger.info(f"Clean feature set: {df_clean.shape}")
    return df_clean


def get_splits(
    df: pd.DataFrame,
    label_col: str = "Label",
    benign_label: str = BENIGN_LABEL,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Create train/test splits appropriate for autoencoder anomaly detection.

    The autoencoder is trained ONLY on normal (BENIGN) traffic so it learns
    to reconstruct the normal distribution. At test time we use the full set
    (normal + attacks) and flag samples with high reconstruction error.

    Returns:
        X_train   – Normal-only samples for training (numpy array, scaled).
        X_test    – Full test set (numpy array, scaled).
        y_test    – Binary labels for test set (0 = normal, 1 = attack).
        scaler    – Fitted StandardScaler (save this for inference).
        feature_names – List of feature column names.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    features = df[feature_cols].values
    labels = df[label_col].values

    # Binary labels: 0 = normal, 1 = anomaly/attack
    y_binary = (labels != benign_label).astype(int)

    # --- Fit scaler on normal traffic only to avoid data leakage ---
    normal_mask = y_binary == 0
    scaler = StandardScaler()
    scaler.fit(features[normal_mask])
    X_scaled = scaler.transform(features)

    # --- Split: preserve original label distribution in test set ---
    X_train_pool = X_scaled[normal_mask]     # Train only on normal traffic
    X_full = X_scaled
    y_full = y_binary

    # Hold out a portion of the full dataset for evaluation
    X_test_full, _, y_test_full, _ = train_test_split(
        X_full, y_full,
        test_size=(1 - test_size),
        random_state=random_state,
        stratify=y_full,
    )

    # Train set: random sample from normal pool (80% of normal traffic)
    X_train, _, = train_test_split(
        X_train_pool,
        test_size=0.2,
        random_state=random_state,
    )

    logger.info(
        f"Train size (normal only): {X_train.shape[0]} | "
        f"Test size: {X_test_full.shape[0]} "
        f"(normal: {(y_test_full == 0).sum()}, attack: {(y_test_full == 1).sum()})"
    )

    return X_train, X_test_full, y_test_full, scaler, feature_cols
