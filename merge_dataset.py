"""
merge_dataset.py – Merge all CIC-IDS2017 CSV files into one clean dataset.

Source folder : C:\\Users\\rafil\\OneDrive\\Tugas KUliah\\MachineLearningCVE
Output file   : data/raw/cic_ids.csv

Usage:
    python merge_dataset.py
"""

import glob
import os
from pathlib import Path

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
SOURCE_DIR  = r"C:\Users\rafil\OneDrive\Tugas KUliah\MachineLearningCVE"
OUTPUT_PATH = Path("data/raw/cic_ids.csv")
# ──────────────────────────────────────────────────────────────────────────────


def load_csv(filepath: str) -> pd.DataFrame | None:
    """
    Load a single CSV file safely.

    Returns None if the file cannot be loaded so a single bad file
    does not abort the entire merge process.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
        return df
    except Exception as exc:
        print(f"  [ERROR] Failed to load '{filepath}': {exc}")
        return None


def main():
    # ── 1. Discover all CSV files ──────────────────────────────────────────
    pattern   = os.path.join(SOURCE_DIR, "*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"[ERROR] No CSV files found in: {SOURCE_DIR}")
        print("  Check the path and make sure CSV files exist there.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in:\n  {SOURCE_DIR}\n")

    # ── 2. Load each file, collecting valid DataFrames ────────────────────
    frames = []
    for path in csv_files:
        filename = os.path.basename(path)
        print(f"  Loading: {filename} ...", end=" ", flush=True)

        df = load_csv(path)
        if df is None:
            # Error already printed inside load_csv(); skip this file.
            continue

        # Sanitise column names immediately so all frames are consistent
        df.columns = df.columns.str.strip()

        print(f"OK  ({len(df):,} rows × {len(df.columns)} cols)")
        frames.append(df)

    if not frames:
        print("\n[ERROR] No files could be loaded. Aborting.")
        return

    print(f"\nSuccessfully loaded {len(frames)} / {len(csv_files)} file(s).")

    # ── 3. Concatenate all DataFrames ─────────────────────────────────────
    # ignore_index=True avoids duplicate index values across files.
    # sort=False preserves the original column order of the first file.
    print("\nConcatenating …")
    merged = pd.concat(frames, ignore_index=True, sort=False)

    # ── 4. Remove duplicate header rows ───────────────────────────────────
    # Some CIC-IDS2017 splits contain literal column-name rows baked into
    # the data (a common artefact when files are created by splitting a
    # spreadsheet). Detect them: if a cell in any numeric column contains
    # the column *name* as a string, the whole row is a phantom header.
    first_col = merged.columns[0]
    phantom_mask = merged[first_col].astype(str) == first_col
    phantom_count = phantom_mask.sum()
    if phantom_count:
        merged = merged[~phantom_mask].reset_index(drop=True)
        print(f"Removed {phantom_count:,} duplicate header row(s).")

    # ── 5. Drop completely empty columns ──────────────────────────────────
    # Columns that are all-NaN across every row carry no information.
    before_cols = len(merged.columns)
    merged.dropna(axis=1, how="all", inplace=True)
    dropped_cols = before_cols - len(merged.columns)
    if dropped_cols:
        print(f"Dropped {dropped_cols} completely empty column(s).")

    # ── 6. Numeric type coercion ───────────────────────────────────────────
    # After removing phantom headers, all columns that should be numeric
    # may still be object dtype. Coerce them back; non-convertible values
    # become NaN (e.g., the orphaned string cells in mixed files).
    label_col = "Label"
    for col in merged.columns:
        if col == label_col:
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # ── 7. Ensure output directory exists ─────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── 8. Save merged dataset ────────────────────────────────────────────
    print(f"\nSaving merged dataset to: {OUTPUT_PATH} …")
    merged.to_csv(OUTPUT_PATH, index=False)

    # ── 9. Final report ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Merge complete!")
    print(f"  Rows    : {len(merged):,}")
    print(f"  Columns : {len(merged.columns)}")
    print(f"  Output  : {OUTPUT_PATH.resolve()}")

    if label_col in merged.columns:
        print(f"\n  Class distribution ('{label_col}' column):")
        for label, count in merged[label_col].value_counts().items():
            pct = 100 * count / len(merged)
            print(f"    {label:<35} {count:>8,}  ({pct:5.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()
