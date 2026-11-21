"""
Model definitions and training routines for all five benchmarked architectures.

Models:
  1. XGBoost           - Primary tabular baseline (histogram GBDT)
  2. Random Forest      - Second tree baseline (200 trees)
  3. Logistic Regression - Linear floor
  4. GraphSAGE          - Proposed inductive GNN (2-layer SAGEConv)
  5. GAT                - Appendix GNN baseline (2-layer, 4-head attention)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.loader import NeighborLoader
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# =========================================================================
# 1. GraphSAGE Architecture
# =========================================================================
class FraudGraphSAGE(nn.Module):
    """
    Two-layer inductive GraphSAGE with BatchNorm, Dropout, and a
    single-output classification head for binary fraud detection.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)

    def get_embeddings(self, x, edge_index):
        """Extract 128-dim node embeddings (for the hybrid model extension)."""
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        return x  # shape: [num_nodes, hidden_channels]


# =========================================================================
# 2. GAT Architecture (Appendix Baseline)
# =========================================================================
class FraudGAT(nn.Module):
    """
    Two-layer Graph Attention Network with multi-head attention.
    Included as an appendix baseline to address reviewer objection:
    'Why GraphSAGE instead of GAT?'
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.classifier(x).squeeze(-1)


# =========================================================================
# 3. Tabular Baselines Training
# =========================================================================
def train_tabular_baselines(data_dict: dict, config: dict) -> dict:
    """
    Train XGBoost, Random Forest, and Logistic Regression on tabular features.

    Returns:
        dict mapping model name -> fitted model object.
    """
    X_tab = data_dict["X_tabular"]
    X_neu = data_dict["X_neural"]
    y = data_dict["y"]
    tr = data_dict["train_idx"]
    va = data_dict["val_idx"]
    seed = config["seed"]

    pos_weight = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    print(f"[MODELS] Positive class weight: {pos_weight:.2f}")

    models = {}

    # --- XGBoost (Primary Baseline) ---
    print("[MODELS] Training XGBoost ...")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(
        X_tab[tr], y[tr],
        eval_set=[(X_tab[va], y[va])],
        verbose=False,
    )
    t_xgb = time.time() - t0
    print(f"  XGBoost trained in {t_xgb:.1f}s (best iteration: {xgb_model.best_iteration})")
    models["XGBoost"] = xgb_model

    # --- Logistic Regression (Linear Floor: fast lbfgs solver) ---
    print("[MODELS] Training Logistic Regression ...")
    t0 = time.time()
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=200,
        random_state=seed,
        solver="lbfgs",
    )
    lr_model.fit(X_neu[tr], y[tr])
    t_lr = time.time() - t0
    print(f"  Logistic Regression trained in {t_lr:.1f}s")
    models["LogisticRegression"] = lr_model

    # --- Random Forest (Second Tree Baseline) ---
    print("[MODELS] Training Random Forest ...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    rf_model.fit(X_tab[tr], y[tr])
    t_rf = time.time() - t0
    print(f"  Random Forest trained in {t_rf:.1f}s")
    models["RandomForest"] = rf_model

    training_times = {
        "XGBoost": float(t_xgb),
        "LogisticRegression": float(t_lr),
        "RandomForest": float(t_rf),
    }

    return models, training_times


# =========================================================================
# 4. GraphSAGE Training Loop
# =========================================================================
def train_graphsage(graph_data, config: dict, device: torch.device):
    """
    Train GraphSAGE with NeighborLoader mini-batches, weighted BCE loss,
    gradient clipping, and early stopping.

    Returns:
        tuple: (Trained FraudGraphSAGE model, training history dict, elapsed training time in seconds)
    """
    print("[MODELS] Training GraphSAGE ...")
    t0 = time.time()

    train_loader = NeighborLoader(
        graph_data,
        num_neighbors=config["fan_out"],
        batch_size=config["batch_size"],
        input_nodes=graph_data.train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        graph_data,
        num_neighbors=config["fan_out"],
        batch_size=config["batch_size"],
        input_nodes=graph_data.val_mask,
        shuffle=False,
    )

    model = FraudGraphSAGE(
        in_channels=graph_data.num_features,
        hidden_channels=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    # Compute class-weighted loss
    y_train = graph_data.y[graph_data.train_mask]
    pos_w = (y_train == 0).sum().float() / (y_train == 1).sum().float()
    pos_weight = torch.tensor([pos_w], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_auc = 0.0
    best_weights = None
    patience_counter = 0

    history = {
        "epochs": [],
        "train_loss": [],
        "val_auc": [],
        "best_epoch": 0,
        "best_val_auc": 0.0,
    }

    for epoch in range(1, config["epochs"] + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[: batch.batch_size]
            loss = criterion(out, batch.y[: batch.batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                probs = torch.sigmoid(model(batch.x, batch.edge_index)[: batch.batch_size])
                val_preds.append(probs.cpu().numpy())
                val_targets.append(batch.y[: batch.batch_size].cpu().numpy())

        val_auc = roc_auc_score(np.concatenate(val_targets), np.concatenate(val_preds))

        history["epochs"].append(epoch)
        history["train_loss"].append(float(avg_loss))
        history["val_auc"].append(float(val_auc))

        print(f"  Epoch {epoch:02d}/{config['epochs']} | "
              f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            history["best_epoch"] = epoch
            history["best_val_auc"] = float(val_auc)
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping at epoch {epoch} (Best Val AUC: {best_val_auc:.4f})")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model = model.to(device)

    t_graphsage = float(time.time() - t0)
    print(f"[MODELS] GraphSAGE training complete in {t_graphsage:.1f}s. Best Val AUC: {best_val_auc:.4f}")
    return model, history, t_graphsage


# =========================================================================
# 5. GAT Training Loop (same structure, different architecture)
# =========================================================================
def train_gat(graph_data, config: dict, device: torch.device):
    """
    Train GAT model (appendix baseline) with identical protocol to GraphSAGE.

    Returns:
        tuple: (Trained FraudGAT model, elapsed training time in seconds)
    """
    print("[MODELS] Training GAT (appendix baseline) ...")
    t0 = time.time()

    train_loader = NeighborLoader(
        graph_data,
        num_neighbors=config["fan_out"],
        batch_size=config["batch_size"],
        input_nodes=graph_data.train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        graph_data,
        num_neighbors=config["fan_out"],
        batch_size=config["batch_size"],
        input_nodes=graph_data.val_mask,
        shuffle=False,
    )

    model = FraudGAT(
        in_channels=graph_data.num_features,
        hidden_channels=config["gat_hidden_dim"],
        heads=config["gat_heads"],
        dropout=config["dropout"],
    ).to(device)

    y_train = graph_data.y[graph_data.train_mask]
    pos_w = (y_train == 0).sum().float() / (y_train == 1).sum().float()
    pos_weight = torch.tensor([pos_w], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_auc = 0.0
    best_weights = None
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[: batch.batch_size]
            loss = criterion(out, batch.y[: batch.batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                probs = torch.sigmoid(model(batch.x, batch.edge_index)[: batch.batch_size])
                val_preds.append(probs.cpu().numpy())
                val_targets.append(batch.y[: batch.batch_size].cpu().numpy())

        val_auc = roc_auc_score(np.concatenate(val_targets), np.concatenate(val_preds))

        print(f"  Epoch {epoch:02d}/{config['epochs']} | "
              f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping at epoch {epoch} (Best Val AUC: {best_val_auc:.4f})")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model = model.to(device)

    t_gat = float(time.time() - t0)
    print(f"[MODELS] GAT training complete in {t_gat:.1f}s. Best Val AUC: {best_val_auc:.4f}")
    return model, t_gat
