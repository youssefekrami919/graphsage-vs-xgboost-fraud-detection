"""
Comprehensive benchmarking, latency profiling, statistical significance testing,
and LaTeX table generation.

Features:
  1. Classification metrics: ROC-AUC, PR-AUC, F1-Score, Balanced Accuracy.
  2. Dual latency profiling: Batch (1,000 tx) and Single-stream (1 tx).
  3. Latency decomposition for GNNs: Sampling time vs. Neural forward pass.
  4. TimeSeriesSplit(n_splits=5) cross-validation and Paired t-test with Bonferroni correction.
  5. Cold-start & Isolated nodes (degree = 0 vs. degree >= 3) evaluation.
  6. Automated table export to CSV and LaTeX (.tex) with booktabs formatting.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
)
import xgboost as xgb


# =========================================================================
# 1. Classification Metrics
# =========================================================================
def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute standard academic fraud classification metrics.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Continuous predicted fraud probabilities in [0, 1].
        threshold: Decision threshold for discrete classification (default: 0.5).

    Returns:
        Dictionary of rounded metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced-Acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def predict_gnn_probabilities(model, graph_data, mask: torch.Tensor, config: dict, device: torch.device) -> tuple:
    """
    Generate out-of-sample predictions and retrieve ground-truth labels for a GNN model
    using NeighborLoader mini-batch inference.
    """
    model.eval()
    loader = NeighborLoader(
        graph_data,
        num_neighbors=config["fan_out"],
        batch_size=config["batch_size"],
        input_nodes=mask,
        shuffle=False,
    )

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)[: batch.batch_size]
            probs = torch.sigmoid(out)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch.y[: batch.batch_size].cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    return y_true, y_prob


# =========================================================================
# 2. Latency Benchmarking (Batch 1k & Single Stream 1 tx)
# =========================================================================
def benchmark_tabular_latency(model, X_sample: np.ndarray, n_runs: int = 10, n_warmup: int = 3) -> tuple:
    """
    Measures Mean +/- Std inference latency (ms) for tabular models.
    """
    for _ in range(n_warmup):
        _ = model.predict_proba(X_sample)

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict_proba(X_sample)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def benchmark_gnn_latency(
    model,
    graph_data,
    test_indices: np.ndarray,
    batch_size: int,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Measures end-to-end GNN latency (ms) with explicit decomposition:
      - tau_sampling: dynamic NeighborLoader subgraph collation time.
      - tau_forward: neural tensor forward pass time.
      - tau_total: sum of sampling and forward pass.
    """
    model.eval()
    sample_nodes = torch.tensor(test_indices[:batch_size], dtype=torch.long)
    loader = NeighborLoader(
        graph_data,
        num_neighbors=config["fan_out"],
        batch_size=batch_size,
        input_nodes=sample_nodes,
        shuffle=False,
    )

    # Warm-up cycles
    with torch.no_grad():
        for _ in range(config["latency_warmup"]):
            for batch in loader:
                batch = batch.to(device)
                _ = model(batch.x, batch.edge_index)[: batch.batch_size]
                if device.type == "cuda":
                    torch.cuda.synchronize()

    sampling_times = []
    forward_times = []
    total_times = []

    with torch.no_grad():
        for _ in range(config["latency_runs"]):
            t_start = time.perf_counter()
            for batch in loader:
                t_sampled = time.perf_counter()
                batch = batch.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_forward_start = time.perf_counter()

                _ = model(batch.x, batch.edge_index)[: batch.batch_size]

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_forward_end = time.perf_counter()

                sampling_times.append((t_sampled - t_start) * 1000.0)
                forward_times.append((t_forward_end - t_forward_start) * 1000.0)
                total_times.append((t_forward_end - t_start) * 1000.0)

    return {
        "total_mean": float(np.mean(total_times)),
        "total_std": float(np.std(total_times)),
        "sampling_mean": float(np.mean(sampling_times)),
        "sampling_std": float(np.std(sampling_times)),
        "forward_mean": float(np.mean(forward_times)),
        "forward_std": float(np.std(forward_times)),
    }


# =========================================================================
# 3. Statistical Significance Testing (Paired t-test + Bonferroni)
# =========================================================================
def paired_ttest_bonferroni(scores_a: list, scores_b: list, num_comparisons: int = 3) -> dict:
    """
    Paired t-test over cross-validation fold scores with Bonferroni correction.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    diff = scores_a - scores_b

    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    p_bonferroni = min(float(p_val) * num_comparisons, 1.0)

    return {
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "t_statistic": float(t_stat),
        "raw_p_value": float(p_val),
        "bonferroni_p_value": float(p_bonferroni),
        "significant_at_0.05": bool(p_bonferroni < 0.05),
    }


def run_temporal_cross_validation(
    data_dict: dict,
    graph_data,
    config: dict,
    device: torch.device,
    n_splits: int = 5,
) -> dict:
    """
    Executes expanding-window TimeSeriesSplit(n_splits=5) on the training set
    to validate stability and generate fold scores for statistical hypothesis testing.
    """
    from src.models import train_graphsage, FraudGraphSAGE

    train_indices = data_dict["train_idx"]
    X_tab = data_dict["X_tabular"]
    y = data_dict["y"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    xgb_auc_scores = []
    graphsage_auc_scores = []

    print(f"[BENCHMARK] Running {n_splits}-fold TimeSeriesSplit Cross-Validation ...")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_indices), start=1):
        actual_tr = train_indices[tr_idx]
        actual_val = train_indices[val_idx]

        print(f"  Fold {fold}/{n_splits} | Train: {len(actual_tr):,} | Val: {len(actual_val):,}")

        # Train fast XGBoost fold
        pos_weight = float((y[actual_tr] == 0).sum() / max((y[actual_tr] == 1).sum(), 1))
        xgb_cv = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=pos_weight,
            tree_method="hist",
            eval_metric="auc",
            random_state=config["seed"] + fold,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_cv.fit(X_tab[actual_tr], y[actual_tr], eval_set=[(X_tab[actual_val], y[actual_val])], verbose=False)
        xgb_prob = xgb_cv.predict_proba(X_tab[actual_val])[:, 1]
        xgb_auc = float(roc_auc_score(y[actual_val], xgb_prob))
        xgb_auc_scores.append(xgb_auc)

        # Prepare temporary fold masks for GraphSAGE
        fold_train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        fold_val_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        fold_train_mask[actual_tr] = True
        fold_val_mask[actual_val] = True

        fold_graph = graph_data.clone()
        fold_graph.train_mask = fold_train_mask
        fold_graph.val_mask = fold_val_mask

        # Fast GraphSAGE fold (epochs=10 for CV speed)
        fold_config = config.copy()
        fold_config["epochs"] = 10
        fold_config["patience"] = 3
        gs_model = train_graphsage(fold_graph, fold_config, device)

        y_true_gs, y_prob_gs = predict_gnn_probabilities(
            gs_model, fold_graph, fold_val_mask, fold_config, device
        )
        gs_auc = float(roc_auc_score(y_true_gs, y_prob_gs))
        graphsage_auc_scores.append(gs_auc)

        print(f"    Fold {fold} Results -> XGBoost AUC: {xgb_auc:.4f} | GraphSAGE AUC: {gs_auc:.4f}")

    # Paired t-test
    ttest_results = paired_ttest_bonferroni(graphsage_auc_scores, xgb_auc_scores, num_comparisons=3)
    return {
        "xgb_fold_auc": xgb_auc_scores,
        "graphsage_fold_auc": graphsage_auc_scores,
        "ttest": ttest_results,
    }


# =========================================================================
# 4. Cold-Start / Isolated Nodes Robustness Analysis
# =========================================================================
def evaluate_cold_start_robustness(
    tabular_models: dict,
    graphsage_model,
    data_dict: dict,
    graph_data,
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    """
    Evaluates performance on Isolated Nodes (degree = 0) vs. Connected Nodes (degree >= 3)
    to study model degradation under cold-start conditions.
    """
    test_idx = data_dict["test_idx"]
    edge_index = graph_data.edge_index
    n_nodes = graph_data.num_nodes

    # In-degree computation
    degrees = torch.zeros(n_nodes, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
    degrees_np = degrees.numpy()

    test_degrees = degrees_np[test_idx]
    isolated_mask = test_degrees == 0
    connected_mask = test_degrees >= 3

    isolated_indices = test_idx[isolated_mask]
    connected_indices = test_idx[connected_mask]

    print(f"[BENCHMARK] Test Subsets -> Isolated (deg=0): {len(isolated_indices):,} | "
          f"Connected (deg>=3): {len(connected_indices):,}")

    rows = []

    # Helper mask for GNN loader
    def eval_subset(name, subset_indices):
        y_true_sub = data_dict["y"][subset_indices]
        if (y_true_sub == 1).sum() == 0 or (y_true_sub == 0).sum() == 0:
            print(f"  Warning: Subset {name} has only one class.")
            return

        # XGBoost
        xgb_prob = tabular_models["XGBoost"].predict_proba(data_dict["X_tabular"][subset_indices])[:, 1]
        xgb_m = compute_classification_metrics(y_true_sub, xgb_prob)
        rows.append({"Topology": name, "Model": "XGBoost", "Count": len(subset_indices), **xgb_m})

        # GraphSAGE
        sub_mask = torch.zeros(n_nodes, dtype=torch.bool)
        sub_mask[subset_indices] = True
        y_true_gs, gs_prob = predict_gnn_probabilities(graphsage_model, graph_data, sub_mask, config, device)
        gs_m = compute_classification_metrics(y_true_gs, gs_prob)
        rows.append({"Topology": name, "Model": "GraphSAGE", "Count": len(subset_indices), **gs_m})

    eval_subset("Isolated (Degree = 0)", isolated_indices)
    eval_subset("Connected (Degree >= 3)", connected_indices)

    df_robustness = pd.DataFrame(rows)

    # Wide format matching Table 7
    table7_rows = []
    for model_name in ["XGBoost", "GraphSAGE"]:
        iso_row = df_robustness[(df_robustness["Topology"] == "Isolated (Degree = 0)") & (df_robustness["Model"] == model_name)]
        conn_row = df_robustness[(df_robustness["Topology"] == "Connected (Degree >= 3)") & (df_robustness["Model"] == model_name)]
        if len(iso_row) > 0 and len(conn_row) > 0:
            iso_pr = iso_row["PR-AUC"].values[0]
            iso_roc = iso_row["ROC-AUC"].values[0]
            conn_pr = conn_row["PR-AUC"].values[0]
            conn_roc = conn_row["ROC-AUC"].values[0]
            table7_rows.append({
                "Model": model_name,
                "Isolated Count": int(iso_row["Count"].values[0]),
                "Isolated PR-AUC": float(iso_pr),
                "Isolated ROC-AUC": float(iso_roc),
                "Connected Count": int(conn_row["Count"].values[0]),
                "Connected PR-AUC": float(conn_pr),
                "Connected ROC-AUC": float(conn_roc),
                "Delta PR-AUC (Conn - Iso)": float(conn_pr - iso_pr),
            })

    df_table7 = pd.DataFrame(table7_rows)
    if len(table7_rows) == 2:
        delta_row = {
            "Model": "Delta (GraphSAGE - XGBoost)",
            "Isolated Count": 0,
            "Isolated PR-AUC": float(table7_rows[1]["Isolated PR-AUC"] - table7_rows[0]["Isolated PR-AUC"]),
            "Isolated ROC-AUC": float(table7_rows[1]["Isolated ROC-AUC"] - table7_rows[0]["Isolated ROC-AUC"]),
            "Connected Count": 0,
            "Connected PR-AUC": float(table7_rows[1]["Connected PR-AUC"] - table7_rows[0]["Connected PR-AUC"]),
            "Connected ROC-AUC": float(table7_rows[1]["Connected ROC-AUC"] - table7_rows[0]["Connected ROC-AUC"]),
            "Delta PR-AUC (Conn - Iso)": float(table7_rows[1]["Delta PR-AUC (Conn - Iso)"] - table7_rows[0]["Delta PR-AUC (Conn - Iso)"]),
        }
        df_table7 = pd.concat([df_table7, pd.DataFrame([delta_row])], ignore_index=True)

    export_results_table(
        df_table7,
        "coldstart_metrics",
        config,
        caption="Cold-Start and Structural Topology Robustness Metrics (Table 7)",
        label="coldstart_metrics"
    )

    return df_robustness


# =========================================================================
# 5. Table Export to CSV and LaTeX
# =========================================================================
def export_results_table(df: pd.DataFrame, filename_base: str, config: dict, caption: str = "", label: str = ""):
    """
    Save DataFrame to both CSV and publication-ready LaTeX (.tex) with booktabs formatting.
    """
    tables_dir = config["tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)

    csv_path = os.path.join(tables_dir, f"{filename_base}.csv")
    tex_path = os.path.join(tables_dir, f"{filename_base}.tex")

    # Save CSV
    df.to_csv(csv_path, index=False)

    # Save LaTeX table
    latex_code = df.to_latex(
        index=False,
        caption=caption if caption else None,
        label=f"tab:{label}" if label else None,
        float_format="%.4f",
        column_format="l" + "r" * (len(df.columns) - 1),
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"[BENCHMARK] Exported table -> {csv_path} and {tex_path}")


# =========================================================================
# 6. Cost-Sensitive Financial Savings (Hook 3 / Table 8)
# =========================================================================
def compute_financial_savings(
    y_true: np.ndarray,
    y_prob_dict: dict,
    transaction_amt: np.ndarray,
    config: dict,
    c_fp: float = 5.0,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Computes cost-sensitive financial savings based on actual transaction amounts ($).

    Formulation:
      - Saved Fraud Dollars: sum of TransactionAmt for True Positives (fraud correctly prevented).
      - False-Positive Cost: c_fp * count(FP) (operational cost of manual investigation/alert fatigue).
      - False-Negative Cost: sum of TransactionAmt for False Negatives (fraud slip loss).
      - Net Financial Savings: Saved Fraud Dollars - False-Positive Cost.
      - Detection Rate (%): Saved Fraud Dollars / Total Fraud Dollars * 100.
    """
    total_fraud_dollars = float(transaction_amt[y_true == 1].sum())

    rows = []
    for model_name, y_prob in y_prob_dict.items():
        y_pred = (y_prob >= threshold).astype(int)
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        saved_dollars = float(transaction_amt[tp_mask].sum())
        fp_cost = float(c_fp * fp_mask.sum())
        fn_loss = float(transaction_amt[fn_mask].sum())
        net_savings = float(saved_dollars - fp_cost)
        detection_rate = float((saved_dollars / max(total_fraud_dollars, 1e-6)) * 100.0)

        rows.append({
            "Model": model_name,
            "Saved Fraud ($)": saved_dollars,
            "FP Cost ($)": fp_cost,
            "FN Loss ($)": fn_loss,
            "Net Savings ($)": net_savings,
            "Fraud Dollar Recall (%)": detection_rate,
            "TP Count": int(tp_mask.sum()),
            "FP Count": int(fp_mask.sum()),
        })

    df_financial = pd.DataFrame(rows)

    if "XGBoost" in y_prob_dict and "GraphSAGE" in y_prob_dict:
        xgb_row = df_financial[df_financial["Model"] == "XGBoost"].iloc[0]
        gs_row = df_financial[df_financial["Model"] == "GraphSAGE"].iloc[0]
        delta_row = {
            "Model": "Delta (GraphSAGE - XGBoost)",
            "Saved Fraud ($)": float(gs_row["Saved Fraud ($)"] - xgb_row["Saved Fraud ($)"]),
            "FP Cost ($)": float(gs_row["FP Cost ($)"] - xgb_row["FP Cost ($)"]),
            "FN Loss ($)": float(gs_row["FN Loss ($)"] - xgb_row["FN Loss ($)"]),
            "Net Savings ($)": float(gs_row["Net Savings ($)"] - xgb_row["Net Savings ($)"]),
            "Fraud Dollar Recall (%)": float(gs_row["Fraud Dollar Recall (%)"] - xgb_row["Fraud Dollar Recall (%)"]),
            "TP Count": int(gs_row["TP Count"] - xgb_row["TP Count"]),
            "FP Count": int(gs_row["FP Count"] - xgb_row["FP Count"]),
        }
        df_financial = pd.concat([df_financial, pd.DataFrame([delta_row])], ignore_index=True)

    export_results_table(
        df_financial,
        "financial_savings",
        config,
        caption="Cost-Sensitive Financial Savings and Loss Prevention Breakdown (Table 8)",
        label="financial_savings"
    )

    return df_financial


def sweep_cost_sensitivity(
    y_true: np.ndarray,
    y_prob_dict: dict,
    transaction_amt: np.ndarray,
    config: dict,
    c_fp_min: float = 1.0,
    c_fp_max: float = 25.0,
    n_points: int = 25,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Parametric sweep of operational alert cost c_fp in [$1.0, $25.0].
    Evaluates net financial savings and identifies break-even thresholds.
    """
    c_fp_values = np.linspace(c_fp_min, c_fp_max, n_points)
    sweep_records = []

    for c_fp in c_fp_values:
        row = {"c_fp": float(c_fp)}
        for model_name, y_prob in y_prob_dict.items():
            y_pred = (y_prob >= threshold).astype(int)
            tp_mask = (y_true == 1) & (y_pred == 1)
            fp_mask = (y_true == 0) & (y_pred == 1)

            saved_dollars = float(transaction_amt[tp_mask].sum())
            fp_cost = float(c_fp * fp_mask.sum())
            net_savings = float(saved_dollars - fp_cost)
            row[f"NetSavings_{model_name}"] = net_savings

        sweep_records.append(row)

    df_sweep = pd.DataFrame(sweep_records)
    export_results_table(
        df_sweep,
        "cost_sensitivity_sweep",
        config,
        caption="Cost Sensitivity Parameter Sweep across c_FP ($1 to $25)",
        label="cost_sensitivity_sweep"
    )
    return df_sweep


# =========================================================================
# 7. Additional Publication Tables (Table 6, Table 9, Training Times)
# =========================================================================
def export_latency_table(
    tabular_latency: dict,
    gnn_latency_1k: dict,
    gnn_latency_1tx: dict,
    config: dict,
) -> pd.DataFrame:
    """
    Export Table 6: Inference Latency Benchmark across models with
    Batch (1k tx), Single-Stream (1 tx), and GNN sampling/forward decomposition.
    """
    rows = []
    # Tabular models
    for m, vals in tabular_latency.items():
        rows.append({
            "Model": m,
            "Batch 1k Latency (ms)": f"{vals['batch_mean']:.2f} +/- {vals['batch_std']:.2f}",
            "Single Stream Latency (ms)": f"{vals['single_mean']:.2f} +/- {vals['single_std']:.2f}",
            "Sampling Latency tau_sampling (ms)": "N/A",
            "Forward Latency tau_forward (ms)": f"{vals['single_mean']:.2f}",
        })

    # GNN models
    for m in gnn_latency_1k.keys():
        b_mean = gnn_latency_1k[m]["total_mean"]
        b_std = gnn_latency_1k[m]["total_std"]
        s_mean = gnn_latency_1tx[m]["total_mean"]
        s_std = gnn_latency_1tx[m]["total_std"]
        samp = gnn_latency_1tx[m]["sampling_mean"]
        fwd = gnn_latency_1tx[m]["forward_mean"]

        rows.append({
            "Model": m,
            "Batch 1k Latency (ms)": f"{b_mean:.2f} +/- {b_std:.2f}",
            "Single Stream Latency (ms)": f"{s_mean:.2f} +/- {s_std:.2f}",
            "Sampling Latency tau_sampling (ms)": f"{samp:.2f}",
            "Forward Latency tau_forward (ms)": f"{fwd:.2f}",
        })

    df_lat = pd.DataFrame(rows)
    export_results_table(
        df_lat,
        "latency_benchmark",
        config,
        caption="Inference Latency Benchmark across Tabular and GNN Architectures (Table 6)",
        label="latency_benchmark"
    )
    return df_lat


def export_overall_metrics_table(
    metrics_dict: dict,
    training_times: dict,
    ttest_dict: dict,
    config: dict,
) -> pd.DataFrame:
    """
    Export Table 9: Full Comparative Performance Metrics across All Architectures.
    """
    rows = []
    for model_name, m in metrics_dict.items():
        row = {
            "Model": model_name,
            "PR-AUC": float(m["PR-AUC"]),
            "ROC-AUC": float(m["ROC-AUC"]),
            "F1-Score": float(m["F1-Score"]),
            "Balanced-Acc": float(m["Balanced-Acc"]),
            "Training Duration (s)": float(training_times.get(model_name, 0.0)),
        }
        if model_name == "XGBoost":
            row["p-value vs XGBoost"] = "Baseline"
        elif model_name == "GraphSAGE" and ttest_dict is not None and "ttest" in ttest_dict:
            p_bon = ttest_dict["ttest"]["bonferroni_p_value"]
            row["p-value vs XGBoost"] = f"{p_bon:.4e}" if p_bon < 0.001 else f"{p_bon:.4f}"
        else:
            row["p-value vs XGBoost"] = "--"
        rows.append(row)

    df_overall = pd.DataFrame(rows)
    export_results_table(
        df_overall,
        "overall_metrics",
        config,
        caption="Comparative Performance Benchmark on Inductive Test Set (Table 9)",
        label="overall_metrics"
    )
    return df_overall


def export_training_times_table(training_times: dict, config: dict) -> pd.DataFrame:
    """
    Export dedicated training times summary table.
    """
    rows = [{"Model": k, "Training Duration (s)": float(v), "Training Duration (min)": float(v / 60.0)}
            for k, v in training_times.items()]
    df_times = pd.DataFrame(rows)
    export_results_table(
        df_times,
        "training_times_summary",
        config,
        caption="Computational Training Overhead Summary across Architectures",
        label="training_times"
    )
    return df_times
