# AegisNet – Notebook 01: Exploratory Data Analysis
# =============================================================
# Paste each section into a separate Jupyter cell.
# =============================================================

# ─── Cell 1 – Imports ─────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

DATA_PATH  = "../data/raw/cic_ids.csv"
LABEL_COL  = "Label"

print("Libraries loaded.")


# ─── Cell 2 – Load Data ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()          # CIC-IDS2017 has leading spaces

print(f"Shape     : {df.shape}")
print(f"\nColumns ({len(df.columns)}):\n{list(df.columns)}")
df.head(3)


# ─── Cell 3 – Class Distribution ─────────────────────────────────────────────
label_counts = df[LABEL_COL].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
label_counts.plot(kind="bar", ax=axes[0], color=sns.color_palette("muted", len(label_counts)))
axes[0].set_title("Class Distribution (count)")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

# Pie chart – proportions
label_counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%", startangle=140)
axes[1].set_ylabel("")
axes[1].set_title("Class Proportions")

plt.tight_layout()
plt.suptitle("CIC-IDS2017 – Label Distribution", y=1.02, fontsize=14, fontweight="bold")
plt.show()

print(label_counts.to_string())


# ─── Cell 4 – Missing Values ──────────────────────────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
missing    = numeric_df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_report = pd.DataFrame({"Missing": missing, "Missing%": missing_pct})
missing_report = missing_report[missing_report["Missing"] > 0].sort_values("Missing%", ascending=False)

print(f"Columns with missing values: {len(missing_report)}")
if not missing_report.empty:
    print(missing_report.to_string())
    # Visual
    fig, ax = plt.subplots(figsize=(12, 4))
    missing_report["Missing%"].plot(kind="bar", ax=ax, color="#e57373")
    ax.set_title("Missing Value Rate per Column")
    ax.set_ylabel("Missing %")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found.")


# ─── Cell 5 – Correlation Heatmap ────────────────────────────────────────────
# Use a sample for speed on large datasets; correlation is stable with ~20 k rows.
SAMPLE_N = min(20_000, len(df))
sample   = df.sample(SAMPLE_N, random_state=42)
num_cols = sample.select_dtypes(include=[np.number])

# Replace inf values before computing correlation
num_cols  = num_cols.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
corr_mat  = num_cols.corr()

# Keep the 20 most variable features to keep the heatmap readable
top_cols  = num_cols.std().nlargest(20).index
corr_top  = corr_mat.loc[top_cols, top_cols]

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_top, dtype=bool))   # Upper triangle mask
sns.heatmap(
    corr_top, mask=mask, cmap="coolwarm", center=0,
    annot=True, fmt=".1f", linewidths=0.5,
    annot_kws={"size": 7}, ax=ax
)
ax.set_title("Feature Correlation Heatmap (Top-20 Highest Variance Features)", fontsize=13)
plt.tight_layout()
plt.show()


# ─── Cell 6 – Feature Distributions ──────────────────────────────────────────
# Pick 3 interesting numeric features (adjust to match your column names)
features_to_plot = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
]

# Verify columns exist; fall back to first 3 numeric columns if not
available = [c for c in features_to_plot if c in df.columns]
if len(available) < 3:
    available = num_cols.columns[:3].tolist()
    print(f"[INFO] Default features used: {available}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, feat in zip(axes, available):
    # Clip extreme outliers at 99th percentile for a clean plot
    q99 = sample[feat].quantile(0.99)
    clipped = sample[feat].clip(upper=q99)
    clipped.hist(bins=60, ax=ax, edgecolor="white", color="#42A5F5")
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

plt.suptitle("Distribution of Selected Features (clipped at 99th pct)", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


# ─── Cell 7 – Summary Statistics ─────────────────────────────────────────────
print("Summary statistics (numeric columns):")
df.select_dtypes(include=[np.number]).describe().T
