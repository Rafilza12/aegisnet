# AegisNet

**Adaptive Real-Time Network Anomaly Detection System**

> Phase 1 – Foundation & Baseline Autoencoder

---

## Project Goal

AegisNet is a research-grade system that detects network anomalies (attacks, outliers) in flow-based network traffic using deep learning. Phase 1 establishes a clean ML foundation using an unsupervised **Autoencoder** trained exclusively on normal traffic. Future phases will extend this toward transformer-based models and a real-time Go inference engine.

---

## Project Structure

```
aegisnet/
├── data/
│   ├── raw/          ← Place raw CSVs here (e.g., cic_ids.csv)
│   └── processed/    ← Cleaned/normalised data outputs
│
├── notebooks/
│   └── 01_eda.py     ← EDA code (paste into Jupyter cell-by-cell)
│
├── src/
│   ├── data_loader.py       ← CSV loading & sanity cleaning
│   ├── preprocessing.py     ← Feature engineering & train/test split
│   ├── model_autoencoder.py ← PyTorch Autoencoder model
│   ├── train.py             ← Full training + evaluation pipeline
│   └── utils.py             ← Logging, device, seeding, plotting
│
├── experiments/      ← Saved plots & per-run metrics
├── models/           ← Saved model checkpoints (.pt)
├── logs/             ← Per-module log files
├── gpu_check.py      ← Quick GPU sanity check script
└── requirements.txt
```

---

## Quick Start

### 1. Create and activate virtual environment

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA support

```bash
# CUDA 11.8 (adjust cu118 → cu121 if using CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU setup

```bash
python gpu_check.py
```

### 5. Place dataset

Download CIC-IDS2017 from [UNB ISCX](https://www.unb.ca/cic/datasets/ids-2017.html) and place any CSV file at:

```
data/raw/cic_ids.csv
```

### 6. Run training

```bash
# From the aegisnet/ directory
python src/train.py
```

Results (loss curve, anomaly score distribution, ROC curve) will be saved to `experiments/phase1_baseline/`.

---

## EDA Notebook

Open Jupyter and create a new notebook in `notebooks/`. Copy the contents of `notebooks/01_eda.py` cell-by-cell. The notebook assumes the dataset is at `../data/raw/cic_ids.csv`.

---

## Architecture

```
Input Features
    │
    ▼
┌────────────┐
│  Encoder   │  Linear → BatchNorm → ReLU → Dropout (repeated)
│            │  256 → 128 → 64 → 32
└────────────┘
    │
    ▼ Bottleneck (32-dim latent representation)
    │
┌────────────┐
│  Decoder   │  32 → 64 → 128 → 256 → Input Dim
│            │  (mirror of encoder)
└────────────┘
    │
    ▼
Reconstruction

Anomaly Score = MSE(input, reconstruction)
  ↑ High score → likely attack
```

**Key design decision**: The scaler is fitted *only on normal traffic* (not the entire dataset) to avoid data leakage from the attack distribution into the normalisation step.

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation + Baseline Autoencoder | ✅ In Progress |
| 2 | Feature Selection + Advanced Models (VAE, IsolationForest) | ⬜ Planned |
| 3 | Hyperparameter tuning + SHAP explainability | ⬜ Planned |
| 4 | Real-time inference pipeline (Go integration) | ⬜ Planned |
| 5 | Production system + streaming data | ⬜ Planned |

---

## References

- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- Hinton & Salakhutdinov (2006) – *Reducing the Dimensionality of Data with Neural Networks*
- Chandola et al. (2009) – *Anomaly Detection: A Survey*
