"""
Publication-quality visualization generation (300 DPI) for academic reports and papers.

All generated plots follow Elsevier/IEEE academic styling guidelines:
  Fig 1: Log-log degree distribution (anti-supernode proof).
  Fig 2: Chronological transaction volume and fraud rate over time (leak-free split visualization).
  Fig 3: Dual Precision-Recall (PR) and ROC curves across all models.
  Fig 4: Accuracy vs. Inference Latency Trade-off (The Pareto Frontier plot).
  Fig 5: GNN Latency Decomposition (Neighborhood sampling vs. Forward pass).
  Fig 6: Edge-Type Ablation Study (card1 vs. DeviceInfo vs. P_emaildomain vs. Combined).
  Fig 7: Cold-Start / Isolated Nodes Robustness Analysis (degree = 0 vs. degree >= 3).
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch


# Configure global publication aesthetics
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

MODEL_COLORS = {
    "XGBoost": "#1f77b4",          # Deep Blue
    "RandomForest": "#2ca02c",     # Forest Green
    "LogisticRegression": "#7f7f7f", # Neutral Grey
    "GraphSAGE": "#d62728",        # Crimson Red (Proposed)
    "GAT": "#9467bd",              # Purple (Appendix)
    "Hybrid": "#ff7f0e",           # Amber Orange
}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================================================================
# Fig 1: Log-Log Degree Distribution
# =========================================================================
def plot_degree_distribution(graph_data, config: dict) -> str:
    """
    Plot node in-degree distribution on log-log scale to demonstrate
    scale-free topology and confirm capping of super-nodes.
    """
    _ensure_dir(config["figures_dir"])
    n_nodes = graph_data.num_nodes
    edge_index = graph_data.edge_index

    degrees = torch.zeros(n_nodes, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
    degrees_np = degrees.numpy()

    # Degree frequency counts
    unique_deg, counts = np.unique(degrees_np, return_counts=True)
    non_zero = unique_deg > 0

    fig, ax = plt.subplots(figsize=(7, 5), dpi=config["figure_dpi"])
    ax.scatter(unique_deg[non_zero], counts[non_zero], color="#1f77b4", alpha=0.7, edgecolors="none", s=25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("In-Degree $k$ (Log Scale)")
    ax.set_ylabel("Node Frequency $N(k)$ (Log Scale)")
    ax.set_title("Causal Transaction Graph: In-Degree Distribution")

    # Annotate max degree and isolated nodes
    isolated_pct = (degrees_np == 0).mean() * 100
    ax.axvline(x=config["max_neighbors_per_node"], color="crimson", linestyle=":", label=f"Degree Cap ($k={config['max_neighbors_per_node']}$)")
    ax.text(0.05, 0.15, f"Isolated nodes ($k=0$): {isolated_pct:.1f}%\nMax degree: {degrees_np.max()}",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    ax.legend(loc="upper right")

    save_path = os.path.join(config["figures_dir"], "fig1_degree_distribution.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 2: Chronological Fraud & Volume Distribution
# =========================================================================
def plot_temporal_fraud_distribution(data_dict: dict, config: dict) -> str:
    """
    Visualize transaction volume and fraud rate over TransactionDT with
    clear demarcation of Train (80%), Val (10%), and Test (10%) splits.
    """
    _ensure_dir(config["figures_dir"])
    timestamps = data_dict["timestamps"]
    y = data_dict["y"]

    # Bin timestamps into 50 chronological intervals
    bins = np.linspace(timestamps.min(), timestamps.max(), 50)
    bin_indices = np.digitize(timestamps, bins)

    bin_df = pd.DataFrame({"bin": bin_indices, "isFraud": y})
    grouped = bin_df.groupby("bin").agg(
        volume=("isFraud", "count"),
        fraud_rate=("isFraud", "mean")
    ).reset_index()

    train_end_t = timestamps[data_dict["train_idx"][-1]]
    val_end_t = timestamps[data_dict["val_idx"][-1]]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=config["figure_dpi"])
    ax2 = ax1.twinx()

    # Transaction volume bars
    ax1.bar(bins[:len(grouped)], grouped["volume"], width=(bins[1]-bins[0])*0.85,
            color="#aec7e8", alpha=0.6, label="Transaction Volume")
    ax1.set_xlabel("TransactionDT (Chronological Seconds)")
    ax1.set_ylabel("Transaction Count per Interval", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    # Fraud rate line
    ax2.plot(bins[:len(grouped)], grouped["fraud_rate"] * 100, color="crimson",
             linewidth=2.0, marker="o", markersize=4, label="Fraud Rate (%)")
    ax2.set_ylabel("Observed Fraud Rate (%)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    # Split boundaries
    ax1.axvline(train_end_t, color="black", linestyle="--", linewidth=1.5)
    ax1.axvline(val_end_t, color="black", linestyle="--", linewidth=1.5)

    y_top = grouped["volume"].max() * 0.9
    ax1.text(train_end_t * 0.5, y_top, "Train Split (80%)", horizontalalignment="center", fontweight="bold")
    ax1.text((train_end_t + val_end_t) * 0.5, y_top, "Val (10%)", horizontalalignment="center", fontweight="bold")
    ax1.text((val_end_t + timestamps.max()) * 0.5, y_top, "Test (10%)", horizontalalignment="center", fontweight="bold")

    plt.title("Temporal Evolution of Transaction Volume and Fraud Prevalence")
    save_path = os.path.join(config["figures_dir"], "fig2_temporal_distribution.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 3: PR Curves and ROC Curves (Dual Subplots)
# =========================================================================
def plot_pr_and_roc_curves(predictions_dict: dict, y_true: np.ndarray, config: dict) -> str:
    """
    Generate side-by-side Precision-Recall and ROC curves across all evaluated models.
    """
    _ensure_dir(config["figures_dir"])
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=config["figure_dpi"])

    for model_name, y_prob in predictions_dict.items():
        color = MODEL_COLORS.get(model_name, "#333333")

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
                   color=color, linewidth=2.0)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{model_name} (ROC-AUC = {roc_auc:.4f})",
                    color=color, linewidth=2.0)

    # Baseline references
    prevalence = y_true.mean()
    ax_pr.axhline(prevalence, color="gray", linestyle=":", label=f"Random Chance ({prevalence:.2%})")
    ax_pr.set_xlabel("Recall (True Positive Rate)")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)")
    ax_pr.set_title("Precision-Recall (PR) Curves")
    ax_pr.legend(loc="upper right", frameon=True)

    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Chance (0.5000)")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax_roc.legend(loc="lower right", frameon=True)

    save_path = os.path.join(config["figures_dir"], "fig3_pr_roc_curves.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 4: Accuracy vs. Inference Latency (The Central Pareto Frontier Plot)
# =========================================================================
def plot_accuracy_latency_tradeoff(summary_df: pd.DataFrame, config: dict) -> str:
    """
    The defining graphical abstract of the paper:
      X-axis: Inference Latency per 1,000 transactions in ms (Log Scale).
      Y-axis: PR-AUC (or ROC-AUC).
    Demonstrates the Pareto frontier between XGBoost and GraphSAGE.
    """
    _ensure_dir(config["figures_dir"])
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=config["figure_dpi"])

    for _, row in summary_df.iterrows():
        name = row["Model"]
        color = MODEL_COLORS.get(name, "#333333")
        x_val = row["Latency_per_1k_ms"]
        x_err = row.get("Latency_Std_ms", 0.0)
        y_val = row["PR-AUC"]

        ax.errorbar(
            x_val, y_val,
            xerr=x_err,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            markersize=10,
            label=f"{name} ({y_val:.4f}, {x_val:.1f}ms)"
        )
        ax.annotate(
            f" {name}",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
            fontweight="bold",
            color=color
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency per 1,000 Transactions (ms, Log Scale)")
    ax.set_ylabel("Precision-Recall AUC (PR-AUC)")
    ax.set_title("Accuracy vs. Inference Latency Trade-Off on IEEE-CIS")
    ax.legend(loc="lower left", frameon=True)

    save_path = os.path.join(config["figures_dir"], "fig4_accuracy_latency_tradeoff.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 5: GNN Latency Decomposition (Stacked Bar Chart)
# =========================================================================
def plot_latency_breakdown(gnn_breakdown_dict: dict, config: dict) -> str:
    """
    Stacked bar chart detailing Sampling Time vs. Neural Forward Pass Time.
    """
    _ensure_dir(config["figures_dir"])
    models = list(gnn_breakdown_dict.keys())
    sampling_times = [gnn_breakdown_dict[m]["sampling_mean"] for m in models]
    forward_times = [gnn_breakdown_dict[m]["forward_mean"] for m in models]

    x = np.arange(len(models))
    width = 0.45

    fig, ax = plt.subplots(figsize=(7, 5), dpi=config["figure_dpi"])
    p1 = ax.bar(x, sampling_times, width, label="Neighbor Sampling (I/O & Traversal)", color="#e7ba52")
    p2 = ax.bar(x, forward_times, width, bottom=sampling_times, label="Neural Forward Pass (Tensor Compute)", color="#1f77b4")

    ax.set_ylabel("Latency per 1,000 Transactions (ms)")
    ax.set_title("GNN Latency Decomposition: Sampling vs. Forward Pass")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.legend(loc="upper right")

    # Add percentage labels inside bars
    for i in range(len(models)):
        total = sampling_times[i] + forward_times[i]
        samp_pct = (sampling_times[i] / total) * 100
        fwd_pct = (forward_times[i] / total) * 100
        ax.text(x[i], sampling_times[i] * 0.5, f"{samp_pct:.1f}%", ha="center", va="center", color="black", fontweight="bold")
        ax.text(x[i], sampling_times[i] + forward_times[i] * 0.5, f"{fwd_pct:.1f}%", ha="center", va="center", color="white", fontweight="bold")

    save_path = os.path.join(config["figures_dir"], "fig5_latency_breakdown.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 6: Edge-Type Ablation Study Chart
# =========================================================================
def plot_edge_ablation(ablation_df: pd.DataFrame, config: dict) -> str:
    """
    Bar chart comparing PR-AUC and ROC-AUC across edge configurations:
      card1 only vs. DeviceInfo only vs. P_emaildomain only vs. Combined.
    """
    _ensure_dir(config["figures_dir"])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=config["figure_dpi"])

    x = np.arange(len(ablation_df))
    width = 0.35

    ax.bar(x - width/2, ablation_df["PR-AUC"], width, label="PR-AUC", color="#1f77b4")
    ax.bar(x + width/2, ablation_df["ROC-AUC"], width, label="ROC-AUC", color="#aec7e8")

    ax.set_ylabel("Classification Metric Score")
    ax.set_title("Edge-Type Ablation Study: Impact on GraphSAGE Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["Edge_Configuration"], rotation=15, ha="right", fontweight="bold")
    ax.legend(loc="lower right")

    save_path = os.path.join(config["figures_dir"], "fig6_edge_ablation.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 7: Cold-Start / Isolated Nodes Comparison
# =========================================================================
def plot_cold_start_comparison(robustness_df: pd.DataFrame, config: dict) -> str:
    """
    Grouped bar chart comparing XGBoost vs. GraphSAGE across
    Isolated Nodes (deg=0) vs. Connected Nodes (deg>=3).
    """
    _ensure_dir(config["figures_dir"])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=config["figure_dpi"])

    sns.barplot(
        data=robustness_df,
        x="Topology",
        y="PR-AUC",
        hue="Model",
        palette=[MODEL_COLORS["XGBoost"], MODEL_COLORS["GraphSAGE"]],
        ax=ax
    )

    ax.set_ylabel("PR-AUC")
    ax.set_title("Cold-Start Robustness: Isolated vs. Connected Transactions")
    ax.legend(title="Model", frameon=True)

    save_path = os.path.join(config["figures_dir"], "fig7_cold_start_analysis.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 04 (Paper): GraphSAGE Epoch Training Curves
# =========================================================================
def plot_graphsage_training_curves(history: dict, config: dict) -> str:
    """
    Plot training loss and validation AUC curves across epochs with early stopping marker.
    """
    _ensure_dir(config["figures_dir"])
    epochs = history.get("epochs", [])
    if not epochs:
        print("[VIZ] Warning: Empty training history for GraphSAGE.")
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=config["figure_dpi"])

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#d62728", lw=2, label="Train Loss (Weighted BCE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("GraphSAGE Optimization Loss")
    ax1.legend(loc="upper right")

    # Validation AUC curve
    ax2.plot(epochs, history["val_auc"], "s-", color="#1f77b4", lw=2, label="Validation ROC-AUC")
    best_epoch = history.get("best_epoch", epochs[-1])
    best_val_auc = history.get("best_val_auc", max(history["val_auc"]))
    ax2.axvline(x=best_epoch, color="forestgreen", linestyle="--", label=f"Best Checkpoint (Ep {best_epoch}, AUC {best_val_auc:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title("GraphSAGE Validation ROC-AUC Trajectory")
    ax2.legend(loc="lower right")

    save_path = os.path.join(config["figures_dir"], "fig04_graphsage_training_curves.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 07 (Paper): Financial Savings Decomposition (Hook 3)
# =========================================================================
def plot_financial_savings_decomposition(df_financial: pd.DataFrame, config: dict) -> str:
    """
    Grouped bar chart showing Saved Fraud Dollars ($) vs. False-Positive Investigation Cost ($)
    and Net Financial Savings ($) across models.
    """
    _ensure_dir(config["figures_dir"])
    # Filter out delta row if present
    plot_df = df_financial[~df_financial["Model"].str.contains("Delta", case=False, na=False)].copy()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=config["figure_dpi"])
    x = np.arange(len(plot_df))
    width = 0.25

    ax.bar(x - width, plot_df["Saved Fraud ($)"], width, label="Saved Fraud ($)", color="#2ca02c")
    ax.bar(x, plot_df["FP Cost ($)"], width, label="False-Positive Cost ($5/alert)", color="#d62728")
    ax.bar(x + width, plot_df["Net Savings ($)"], width, label="Net Financial Savings ($)", color="#1f77b4")

    ax.set_ylabel("USD ($)")
    ax.set_title("Cost-Sensitive Financial Savings Breakdown across Architectures")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=15, ha="right", fontweight="bold")
    ax.legend(loc="upper left")

    # Add numeric labels on bars
    for i in range(len(plot_df)):
        net_val = plot_df["Net Savings ($)"].iloc[i]
        ax.text(x[i] + width, net_val + (max(plot_df["Saved Fraud ($)"]) * 0.02),
                f"${net_val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    save_path = os.path.join(config["figures_dir"], "fig07_financial_savings_decomposition.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 08 (Paper): Cost Sensitivity Curve (c_FP Sweep)
# =========================================================================
def plot_cost_sensitivity(df_sweep: pd.DataFrame, config: dict) -> str:
    """
    Plots Net Financial Savings vs. False-Positive alert cost c_FP ($1 to $25),
    showing break-even points and financial robustness frontiers.
    """
    _ensure_dir(config["figures_dir"])
    fig, ax = plt.subplots(figsize=(9, 6), dpi=config["figure_dpi"])

    c_fp = df_sweep["c_fp"]
    colors = {
        "XGBoost": "#1f77b4",
        "RandomForest": "#2ca02c",
        "LogisticRegression": "#7f7f7f",
        "GraphSAGE": "#d62728",
        "Hybrid": "#ff7f0e",
    }

    for col in df_sweep.columns:
        if col.startswith("NetSavings_"):
            model_name = col.replace("NetSavings_", "")
            c = colors.get(model_name, "#333333")
            ax.plot(c_fp, df_sweep[col], label=model_name, color=c, lw=2.2)

    ax.axhline(0, color="black", linestyle="--", alpha=0.5, label="Break-Even ($0)")
    ax.set_xlabel("Operational Cost per Alert $c_{FP}$ ($)")
    ax.set_ylabel("Net Financial Savings ($)")
    ax.set_title("Operational Robustness: Net Savings under Varying Alert Costs ($1 - $25)")
    ax.legend(loc="upper right", frameon=True)

    save_path = os.path.join(config["figures_dir"], "fig08_cost_sensitivity.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Fig 09 (Paper): XGBoost Feature Importance (Explainability Pillar)
# =========================================================================
def plot_feature_importance(xgb_model, feature_names: list, config: dict, top_k: int = 20) -> str:
    """
    Plots the top K features ranked by Gain from the trained XGBoost model,
    providing empirical evidence for tabular dominance.
    """
    _ensure_dir(config["figures_dir"])
    booster = xgb_model.get_booster()
    score_dict = booster.get_score(importance_type="gain")

    # Map f0, f1... to original feature names
    named_scores = {}
    for feat_id, gain in score_dict.items():
        if feat_id.startswith("f") and feat_id[1:].isdigit():
            idx = int(feat_id[1:])
            name = feature_names[idx] if idx < len(feature_names) else feat_id
        else:
            name = feat_id
        named_scores[name] = gain

    sorted_feats = sorted(named_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    names = [x[0] for x in reversed(sorted_feats)]
    scores = [x[1] for x in reversed(sorted_feats)]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=config["figure_dpi"])
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color="#1f77b4", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Gain (Relative Information Contribution)")
    ax.set_title(f"XGBoost Feature Importance: Top {top_k} Features by Gain")

    save_path = os.path.join(config["figures_dir"], "fig09_feature_importance.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved -> {save_path}")
    return save_path


# =========================================================================
# Paper Figure Aliases (Appendix A.2 Compliance)
# =========================================================================
def create_paper_figure_aliases(config: dict):
    """
    Creates duplicate alias copies matching the exact numbering scheme in Appendix A.2
    without deleting any existing figures.
    """
    fdir = config["figures_dir"]
    mappings = [
        ("fig2_temporal_distribution.png", "fig01_chronological_split.png"),
        ("fig1_degree_distribution.png", "fig02_graph_construction.png"),
        ("fig3_pr_roc_curves.png", "fig03_roc_pr_curves.png"),
        ("fig5_latency_breakdown.png", "fig05_latency_decomposition.png"),
        ("fig7_cold_start_analysis.png", "fig06_coldstart_comparison.png"),
    ]
    for src_name, dst_name in mappings:
        src_path = os.path.join(fdir, src_name)
        dst_path = os.path.join(fdir, dst_name)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            print(f"[VIZ] Created paper alias: {dst_name} -> {src_name}")
