# GraphSAGE vs. XGBoost for Fraud Detection on IEEE-CIS

A latency-aware comparative study of **inductive GraphSAGE** against a tuned **XGBoost** baseline for financial fraud detection on the IEEE-CIS Transaction Dataset.

## Research Question

> *Does a GraphSAGE model that exploits shared-entity edges between transactions achieve materially higher AUC than an XGBoost tabular baseline on the IEEE-CIS dataset, and at what inference-latency cost per 1,000 transactions?*

## Dataset

**IEEE-CIS Fraud Detection Dataset** (Kaggle / Vesta Corporation)  
- **Source:** [https://www.kaggle.com/competitions/ieee-fraud-detection/data](https://www.kaggle.com/competitions/ieee-fraud-detection/data)  
- **Scale:** ~590,540 transactions, ~432 raw features, 3.5% positive fraud rate  
- **Files used:** `train_transaction.csv` joined with `train_identity.csv` on `TransactionID`

### Local Data Setup

Download and extract the dataset into a sibling directory:
```
Fraud Detection Paper/
├── ieee-fraud-detection/
│   ├── train_transaction.csv
│   └── train_identity.csv
└── graphsage-vs-xgboost-fraud-detection/   <-- this repository
```

## Repository Structure

```
graphsage-vs-xgboost-fraud-detection/
├── README.md                               <- This file
├── requirements.txt                        <- Python dependencies
├── initial_report.txt                      <- LaTeX technical proposal
├── initial_report.tex                      <- LaTeX source (copy of .txt)
├── main.py                                 <- CLI orchestrator script
├── experiments_and_visualizations.ipynb     <- Interactive Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── config.py                           <- Seeds, hyperparameters, paths
│   ├── preprocessing.py                    <- Temporal split & leak-free scaling
│   ├── graph_builder.py                    <- Causal graph with anti-supernode capping
│   ├── models.py                           <- LR, RF, XGBoost, GraphSAGE, GAT
│   ├── benchmark.py                        <- Metrics, latency timer, t-tests
│   └── visualization.py                    <- Publication-quality figures (300 DPI)
├── models/
│   ├── native/                             <- PyTorch (.pt), XGBoost (.json), sklearn (.joblib)
│   └── onnx/                               <- ONNX Runtime artifacts (.onnx)
└── results/
    ├── tables/                             <- CSV + LaTeX (.tex) tables
    └── figures/                            <- PNG figures at 300 DPI
```

## Models Evaluated

| # | Model | Type | Serialization Format | Purpose |
|---|-------|------|---------------------|---------|
| 1 | **XGBoost** | Tabular (GBDT) | `.json` (native) + `.onnx` | Primary baseline |
| 2 | **Random Forest** | Tabular (Ensemble) | `.joblib` (native) + `.onnx` | Second tree baseline |
| 3 | **Logistic Regression** | Tabular (Linear) | `.joblib` (native) + `.onnx` | Linear floor |
| 4 | **GraphSAGE** | Graph Neural Network | `.pt` (PyTorch state dict) | Proposed model |
| 5 | **GAT** | Graph Neural Network | `.pt` (PyTorch state dict) | Appendix GNN baseline |
| 6 | **Hybrid (GraphSAGE + XGBoost)** | Hybrid | `.pt` + `.json` | Advanced extension |

### Loading Saved Models

**Tabular models (ONNX):**
```python
import onnxruntime as ort
session = ort.InferenceSession("models/onnx/xgboost_model.onnx")
predictions = session.run(None, {"input": X_test.astype(np.float32)})
```

**Tabular models (Native):**
```python
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model("models/native/xgboost_model.json")

import joblib
rf_model = joblib.load("models/native/random_forest_model.joblib")
lr_model = joblib.load("models/native/logistic_regression_model.joblib")
```

**Graph models (PyTorch):**
```python
import torch
from src.models import FraudGraphSAGE
model = FraudGraphSAGE(in_channels=N_FEATURES, hidden_channels=128)
model.load_state_dict(torch.load("models/native/graphsage_model.pt"))
model.eval()
```

## Evaluation Protocol

- **Split:** Strict chronological 80% Train / 10% Validation / 10% Test (no shuffling)
- **CV:** `TimeSeriesSplit(n_splits=5)` for statistical hypothesis testing
- **Metrics:** ROC-AUC (primary), PR-AUC, F1-Score, Balanced Accuracy
- **Statistical Test:** Paired t-test with Bonferroni correction across 5 temporal folds
- **Latency:** Mean ± Std inference time per 1,000 transactions (10 timed runs)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run via Jupyter Notebook (Recommended)
```bash
jupyter notebook experiments_and_visualizations.ipynb
```

### 3. Run via Command Line
```bash
# Prototype mode (first 100k transactions for rapid testing)
python main.py --mode prototype --seed 42

# Full benchmark (all 590k transactions)
python main.py --mode full --seed 42
```

## Generating the PDF Report

The technical proposal is written in LaTeX. To compile it into a PDF document:

```bash
pdflatex initial_report.txt
pdflatex initial_report.txt
```

The command is run **twice** so that LaTeX can resolve the table of contents, page numbers, and cross-references correctly. If your LaTeX distribution requires the `.tex` extension, use `initial_report.tex` instead (the content is identical).

## Key Design Decisions

1. **`DeviceType` excluded from graph edges:** This column has only 2 values (`desktop`/`mobile`), which would create ~5.1 billion edges and crash memory. We use `card1`, `DeviceInfo`, and `P_emaildomain` with frequency capping (`max_freq=500`) instead.
2. **Causal edge direction enforced:** Edges only point from earlier transactions to later ones (`t_src <= t_dst`) to prevent temporal data leakage.
3. **Memory optimization:** Automatic dtype downcasting (`float64→float32`, `int64→int32`) reduces RAM from ~1.8GB to ~750MB for the full dataset.
4. **Staged execution:** Prototype on 100k transactions first, then scale to 590k.

## License

This project is part of a Master's degree research study.
