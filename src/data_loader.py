"""
data_loader.py – Loads raw CIC-IDS2017 CSV data into a clean DataFrame.

Responsibilities:
  - Read CSV
  - Sanitise column names
  - Drop obvious junk (all-NaN rows, duplicate rows)
  - Report basic statistics
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Column name the dataset uses for ground-truth labels (may vary between
# CIC-IDS2017 versions – override via the label_col argument if needed).
DEFAULT_LABEL_COL = "Label"


def load_cicids(
    path: str,
    label_col: str = DEFAULT_LABEL_COL,
    nrows: int = None,
) -> pd.DataFrame:
    """
    Load a CIC-IDS2017 CSV file and perform minimal sanity cleaning.

    Args:
        path:      Path to the raw CSV file.
        label_col: Name of the column containing class labels.
        nrows:     If set, load only the first N rows (useful for quick tests).

    Returns:
        Cleaned DataFrame with sanitised column names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")

    # --- Sanitise column names ---
    # CIC-IDS2017 has leading/trailing spaces in column names.
    df.columns = df.columns.str.strip()

    # --- Drop fully-empty rows ---
    df.dropna(how="all", inplace=True)

    # --- Drop complete duplicates ---
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Dropped {removed} duplicate rows.")

    # --- Validate label column exists ---
    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Clean shape: {df.shape}")
    logger.info(f"Class distribution:\n{df[label_col].value_counts()}")

    return df
