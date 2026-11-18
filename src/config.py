"""
Global configuration, hyperparameters, and deterministic seeding.

All experiment parameters are centralized here to ensure full reproducibility
and to allow easy switching between prototype (100k) and full (590k) modes.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Fix all random seeds for deterministic reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select the best available compute device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
        print(f"[CONFIG] Using GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")
    else:
        device = torch.device("cpu")
        print("[CONFIG] No GPU detected. Using CPU.")
    return device


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the config works from any CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "..", "ieee-fraud-detection"))

CONFIG = {
    # --- Reproducibility ---
    "seed": 42,

    # --- Paths ---
    "project_root": _PROJECT_ROOT,
    "data_dir": _DATA_DIR,
    "results_dir": os.path.join(_PROJECT_ROOT, "results"),
    "figures_dir": os.path.join(_PROJECT_ROOT, "results", "figures"),
    "tables_dir": os.path.join(_PROJECT_ROOT, "results", "tables"),
    "models_native_dir": os.path.join(_PROJECT_ROOT, "models", "native"),
    "models_onnx_dir": os.path.join(_PROJECT_ROOT, "models", "onnx"),

    # --- Data ---
    "subset_size": None,  # Set to 100_000 for prototype mode; None for full 590k
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,

    # --- Graph construction ---
    # Entity columns used for edge creation (DeviceType excluded: only 2 values -> OOM)
    "edge_cols": ["card1", "DeviceInfo", "P_emaildomain"],
    "max_entity_freq": 500,         # Cap high-frequency entities to prevent super-nodes
    "max_neighbors_per_node": 20,   # Max prior transactions linked per node

    # --- GraphSAGE architecture ---
    "hidden_dim": 128,
    "dropout": 0.3,
    "lr": 1e-3,
    "epochs": 25,
    "patience": 5,
    "batch_size": 1024,
    "fan_out": [15, 10],  # NeighborLoader sampling sizes [hop1, hop2]

    # --- GAT architecture (appendix baseline) ---
    "gat_hidden_dim": 32,
    "gat_heads": 4,

    # --- Latency benchmarking ---
    "latency_runs": 10,
    "latency_warmup": 3,
    "latency_batch_size": 1000,

    # --- Cross-validation ---
    "cv_splits": 5,  # TimeSeriesSplit(n_splits=5)

    # --- Visualization ---
    "figure_dpi": 300,
    "figure_format": "png",
}
