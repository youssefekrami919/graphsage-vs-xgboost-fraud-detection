"""
Model serialization (Native + ONNX) and Hybrid Model pipeline.

Features:
  1. Save & Load Native checkpoints:
     - PyTorch (.pt) for GraphSAGE and GAT.
     - Native JSON for XGBoost.
     - Joblib for Random Forest and Logistic Regression.
  2. Export Tabular models to ONNX (.onnx) for low-latency production deployment.
  3. Hybrid Model Pipeline:
     - Extracts 128-dimensional topological node embeddings from trained GraphSAGE.
     - Concatenates embeddings with tabular features: X_hybrid = [X_tabular || Z_embeddings].
     - Trains a secondary XGBoost model on the enriched manifold.
"""

import os
import time
import joblib
import numpy as np
import torch
import xgboost as xgb


# =========================================================================
# 1. Native Model Checkpointing
# =========================================================================
def save_native_models(
    tabular_models: dict,
    graphsage_model,
    gat_model,
    config: dict,
    hybrid_model=None,
):
    """
    Save all models in their robust, native framework formats.
    """
    native_dir = config["models_native_dir"]
    os.makedirs(native_dir, exist_ok=True)

    # 1. XGBoost -> native JSON
    if "XGBoost" in tabular_models:
        xgb_path = os.path.join(native_dir, "xgboost_model.json")
        tabular_models["XGBoost"].save_model(xgb_path)
        print(f"[SERIALIZE] Saved Native XGBoost -> {xgb_path}")

    # 2. Random Forest -> joblib
    if "RandomForest" in tabular_models:
        rf_path = os.path.join(native_dir, "random_forest_model.joblib")
        joblib.dump(tabular_models["RandomForest"], rf_path)
        print(f"[SERIALIZE] Saved Native Random Forest -> {rf_path}")

    # 3. Logistic Regression -> joblib
    if "LogisticRegression" in tabular_models:
        lr_path = os.path.join(native_dir, "logistic_regression_model.joblib")
        joblib.dump(tabular_models["LogisticRegression"], lr_path)
        print(f"[SERIALIZE] Saved Native Logistic Regression -> {lr_path}")

    # 4. GraphSAGE -> PyTorch state_dict (.pt)
    if graphsage_model is not None:
        gs_path = os.path.join(native_dir, "graphsage_model.pt")
        torch.save(graphsage_model.state_dict(), gs_path)
        print(f"[SERIALIZE] Saved Native GraphSAGE -> {gs_path}")

    # 5. GAT -> PyTorch state_dict (.pt)
    if gat_model is not None:
        gat_path = os.path.join(native_dir, "gat_model.pt")
        torch.save(gat_model.state_dict(), gat_path)
        print(f"[SERIALIZE] Saved Native GAT -> {gat_path}")

    # 6. Hybrid Model -> native JSON
    if hybrid_model is not None:
        hyb_path = os.path.join(native_dir, "hybrid_xgboost_model.json")
        hybrid_model.save_model(hyb_path)
        print(f"[SERIALIZE] Saved Native Hybrid XGBoost -> {hyb_path}")


# =========================================================================
# 2. ONNX Model Export for Tabular Models
# =========================================================================
def export_tabular_to_onnx(tabular_models: dict, n_features: int, config: dict):
    """
    Export scikit-learn and XGBoost models to ONNX binary graph format
    for cross-platform, sub-millisecond production inference.
    """
    onnx_dir = config["models_onnx_dir"]
    os.makedirs(onnx_dir, exist_ok=True)

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, n_features]))]

        # Export Logistic Regression
        if "LogisticRegression" in tabular_models:
            lr_onnx = convert_sklearn(tabular_models["LogisticRegression"], initial_types=initial_type)
            lr_path = os.path.join(onnx_dir, "logistic_regression.onnx")
            with open(lr_path, "wb") as f:
                f.write(lr_onnx.SerializeToString())
            print(f"[SERIALIZE] Exported ONNX Logistic Regression -> {lr_path}")

        # Export Random Forest
        if "RandomForest" in tabular_models:
            rf_onnx = convert_sklearn(tabular_models["RandomForest"], initial_types=initial_type)
            rf_path = os.path.join(onnx_dir, "random_forest.onnx")
            with open(rf_path, "wb") as f:
                f.write(rf_onnx.SerializeToString())
            print(f"[SERIALIZE] Exported ONNX Random Forest -> {rf_path}")

        # Export XGBoost
        if "XGBoost" in tabular_models:
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType as OxmlFloatTensorType
            xgb_initial_type = [("float_input", OxmlFloatTensorType([None, n_features]))]
            xgb_onnx = onnxmltools.convert_xgboost(tabular_models["XGBoost"], initial_types=xgb_initial_type)
            xgb_path = os.path.join(onnx_dir, "xgboost.onnx")
            with open(xgb_path, "wb") as f:
                f.write(xgb_onnx.SerializeToString())
            print(f"[SERIALIZE] Exported ONNX XGBoost -> {xgb_path}")

    except Exception as e:
        print(f"[SERIALIZE] Notice: ONNX export skipped or encountered optional package absence: {e}")
        print("            Native formats (.json, .joblib, .pt) remain fully intact and operational.")


# =========================================================================
# 3. Hybrid Pipeline: GraphSAGE Embeddings + XGBoost
# =========================================================================
def train_hybrid_model(
    graphsage_model,
    graph_data,
    data_dict: dict,
    config: dict,
    device: torch.device,
) -> tuple:
    """
    Executes the proposed Hybrid Architecture:
      1. Forward propagation through GraphSAGE to extract 128-dim node embeddings.
      2. Concatenates embeddings with tabular features: X_hybrid = [X_tabular || Z_embeddings].
      3. Fits an XGBoost model on the enriched manifold.
    """
    print("[HYBRID] Extracting GraphSAGE topological node embeddings ...")
    graphsage_model.eval()

    from torch_geometric.loader import NeighborLoader
    all_nodes = torch.arange(graph_data.num_nodes, dtype=torch.long)
    loader = NeighborLoader(
        graph_data,
        num_neighbors=config["fan_out"],
        batch_size=config["batch_size"],
        input_nodes=all_nodes,
        shuffle=False,
    )

    embeddings_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = graphsage_model.get_embeddings(batch.x, batch.edge_index)[: batch.batch_size]
            embeddings_list.append(emb.cpu().numpy())

    Z_embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)
    print(f"  -> Extracted embeddings shape: {Z_embeddings.shape}")

    # Concatenate with tabular features
    X_tab = data_dict["X_tabular"]
    X_hybrid = np.hstack([X_tab, Z_embeddings]).astype(np.float32)
    print(f"  -> Hybrid feature matrix shape: {X_hybrid.shape}")

    # Train secondary XGBoost on X_hybrid
    tr = data_dict["train_idx"]
    va = data_dict["val_idx"]
    y = data_dict["y"]

    pos_weight = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))

    print("[HYBRID] Training secondary XGBoost on hybrid features ...")
    hybrid_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=config["seed"],
        n_jobs=-1,
        verbosity=0,
    )
    t0 = time.time()
    hybrid_xgb.fit(
        X_hybrid[tr], y[tr],
        eval_set=[(X_hybrid[va], y[va])],
        verbose=False,
    )
    t_hybrid = float(time.time() - t0)
    print(f"  -> Hybrid XGBoost trained in {t_hybrid:.1f}s (best iteration: {hybrid_xgb.best_iteration})")

    return hybrid_xgb, X_hybrid, t_hybrid
