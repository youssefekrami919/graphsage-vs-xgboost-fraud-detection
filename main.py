"""
Main CLI Orchestrator for IEEE-CIS GraphSAGE vs. XGBoost Benchmark.

Usage:
  # Rapid prototype on 100k transactions:
  python main.py --mode prototype --seed 42

  # Full benchmark on all 590k transactions:
  python main.py --mode full --seed 42

  # Run with TimeSeriesSplit(n_splits=5) cross-validation:
  python main.py --mode prototype --run_cv
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

from src.config import CONFIG, seed_everything, get_device
from src.preprocessing import load_and_preprocess
from src.graph_builder import build_causal_transaction_graph, get_graph_statistics
from src.models import (
    train_tabular_baselines,
    train_graphsage,
    train_gat,
)
from src.benchmark import (
    compute_classification_metrics,
    predict_gnn_probabilities,
    benchmark_tabular_latency,
    benchmark_gnn_latency,
    run_temporal_cross_validation,
    evaluate_cold_start_robustness,
    export_results_table,
)
from src.visualization import (
    plot_degree_distribution,
    plot_temporal_fraud_distribution,
    plot_pr_and_roc_curves,
    plot_accuracy_latency_tradeoff,
    plot_latency_breakdown,
    plot_cold_start_comparison,
)
from src.serialization import (
    save_native_models,
    export_tabular_to_onnx,
    train_hybrid_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GraphSAGE vs. XGBoost on IEEE-CIS Benchmark")
    parser.add_argument("--mode", type=str, choices=["prototype", "full"], default="prototype",
                        help="prototype: first 100k transactions; full: all 590,540 transactions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_cv", action="store_true", help="Run 5-fold TimeSeriesSplit cross-validation")
    parser.add_argument("--skip_gat", action="store_true", help="Skip GAT appendix baseline to save time")
    parser.add_argument("--skip_hybrid", action="store_true", help="Skip hybrid GraphSAGE+XGBoost model")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print(" IEEE-CIS FRAUD DETECTION: GRAPHSAGE VS. XGBOOST BENCHMARK")
    print(f" Mode: {args.mode.upper()} | Seed: {args.seed}")
    print("=" * 75)

    # 1. Configuration & Seeding
    config = CONFIG.copy()
    config["seed"] = args.seed
    if args.mode == "prototype":
        config["subset_size"] = 100_000
    else:
        config["subset_size"] = None

    seed_everything(config["seed"])
    device = get_device()

    t_pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Data Preprocessing & Chronological Split
    # ------------------------------------------------------------------
    print("\n>>> STAGE 1: Data Ingestion & Leak-Free Preprocessing")
    data_dict = load_and_preprocess(config)

    # Visualization: Temporal volume and fraud distribution
    plot_temporal_fraud_distribution(data_dict, config)

    # ------------------------------------------------------------------
    # Step 2: Causal Graph Construction
    # ------------------------------------------------------------------
    print("\n>>> STAGE 2: Causal Graph Construction (Anti-Supernode)")
    graph_data = build_causal_transaction_graph(data_dict, config)
    graph_stats = get_graph_statistics(graph_data)

    # Visualization: Degree distribution (Log-log)
    plot_degree_distribution(graph_data, config)

    # ------------------------------------------------------------------
    # Step 3: Tabular Baselines Training
    # ------------------------------------------------------------------
    print("\n>>> STAGE 3: Training Tabular Baselines (XGBoost, RF, LR)")
    tabular_models = train_tabular_baselines(data_dict, config)

    # ------------------------------------------------------------------
    # Step 4: Inductive GraphSAGE Training
    # ------------------------------------------------------------------
    print("\n>>> STAGE 4: Training Proposed Inductive GraphSAGE")
    graphsage_model = train_graphsage(graph_data, config, device)

    # ------------------------------------------------------------------
    # Step 5: Appendix GAT Training (Optional)
    # ------------------------------------------------------------------
    gat_model = None
    if not args.skip_gat:
        print("\n>>> STAGE 5: Training Appendix Baseline (GAT)")
        gat_model = train_gat(graph_data, config, device)
    else:
        print("\n>>> STAGE 5: Skipping GAT baseline (--skip_gat flag set)")

    # ------------------------------------------------------------------
    # Step 6: Hybrid Model Training (Optional Extension)
    # ------------------------------------------------------------------
    hybrid_model = None
    X_hybrid = None
    if not args.skip_hybrid:
        print("\n>>> STAGE 6: Training Proposed Hybrid Model (GraphSAGE + XGBoost)")
        hybrid_model, X_hybrid = train_hybrid_model(graphsage_model, graph_data, data_dict, config, device)
    else:
        print("\n>>> STAGE 6: Skipping Hybrid Model (--skip_hybrid flag set)")

    # ------------------------------------------------------------------
    # Step 7: Classification Performance Evaluation on Held-Out Test Set
    # ------------------------------------------------------------------
    print("\n>>> STAGE 7: Out-of-Sample Evaluation on Held-Out Test Split (10%)")
    test_idx = data_dict["test_idx"]
    y_test = data_dict["y"][test_idx]
    X_tab_test = data_dict["X_tabular"][test_idx]
    X_neu_test = data_dict["X_neural"][test_idx]

    predictions_dict = {}
    metrics_rows = []

    # 1. XGBoost
    xgb_prob = tabular_models["XGBoost"].predict_proba(X_tab_test)[:, 1]
    predictions_dict["XGBoost"] = xgb_prob
    metrics_rows.append({"Model": "XGBoost", **compute_classification_metrics(y_test, xgb_prob)})

    # 2. Random Forest
    rf_prob = tabular_models["RandomForest"].predict_proba(X_tab_test)[:, 1]
    predictions_dict["RandomForest"] = rf_prob
    metrics_rows.append({"Model": "RandomForest", **compute_classification_metrics(y_test, rf_prob)})

    # 3. Logistic Regression
    lr_prob = tabular_models["LogisticRegression"].predict_proba(X_neu_test)[:, 1]
    predictions_dict["LogisticRegression"] = lr_prob
    metrics_rows.append({"Model": "LogisticRegression", **compute_classification_metrics(y_test, lr_prob)})

    # 4. GraphSAGE
    _, gs_prob = predict_gnn_probabilities(graphsage_model, graph_data, graph_data.test_mask, config, device)
    predictions_dict["GraphSAGE"] = gs_prob
    metrics_rows.append({"Model": "GraphSAGE", **compute_classification_metrics(y_test, gs_prob)})

    # 5. GAT (if trained)
    if gat_model is not None:
        _, gat_prob = predict_gnn_probabilities(gat_model, graph_data, graph_data.test_mask, config, device)
        predictions_dict["GAT"] = gat_prob
        metrics_rows.append({"Model": "GAT", **compute_classification_metrics(y_test, gat_prob)})

    # 6. Hybrid Model (if trained)
    if hybrid_model is not None:
        hyb_prob = hybrid_model.predict_proba(X_hybrid[test_idx])[:, 1]
        predictions_dict["Hybrid"] = hyb_prob
        metrics_rows.append({"Model": "Hybrid (SAGE+XGB)", **compute_classification_metrics(y_test, hyb_prob)})

    table1_df = pd.DataFrame(metrics_rows)
    print("\n--- TABLE 1: CLASSIFICATION PERFORMANCE METRICS ---")
    print(table1_df.to_string(index=False))
    export_results_table(table1_df, "table1_classification_metrics", config,
                         caption="Classification Performance on Held-Out Test Set (IEEE-CIS)",
                         label="classification_metrics")

    # Visualization: PR and ROC curves
    plot_pr_and_roc_curves(predictions_dict, y_test, config)

    # ------------------------------------------------------------------
    # Step 8: Latency Benchmarking (Batch 1,000 tx & Single tx)
    # ------------------------------------------------------------------
    print("\n>>> STAGE 8: Dual-Latency Benchmarking (Batch 1k & Single Stream 1 tx)")
    latency_rows = []
    gnn_breakdown = {}

    sample_1k_tab = X_tab_test[:1000]
    sample_1k_neu = X_neu_test[:1000]

    # Tabular 1k latency
    xgb_lat_mean, xgb_lat_std = benchmark_tabular_latency(tabular_models["XGBoost"], sample_1k_tab)
    rf_lat_mean, rf_lat_std = benchmark_tabular_latency(tabular_models["RandomForest"], sample_1k_tab)
    lr_lat_mean, lr_lat_std = benchmark_tabular_latency(tabular_models["LogisticRegression"], sample_1k_neu)

    # Tabular 1 tx latency
    xgb_1_mean, xgb_1_std = benchmark_tabular_latency(tabular_models["XGBoost"], sample_1k_tab[:1])

    # GraphSAGE latency
    gs_lat = benchmark_gnn_latency(graphsage_model, graph_data, test_idx, 1000, config, device)
    gnn_breakdown["GraphSAGE"] = gs_lat

    latency_rows.append({
        "Model": "XGBoost",
        "Latency_per_1k_ms": xgb_lat_mean,
        "Latency_Std_ms": xgb_lat_std,
        "Single_tx_ms": xgb_1_mean,
        "PR-AUC": float(table1_df.loc[table1_df['Model'] == 'XGBoost', 'PR-AUC'].values[0]),
    })
    latency_rows.append({
        "Model": "RandomForest",
        "Latency_per_1k_ms": rf_lat_mean,
        "Latency_Std_ms": rf_lat_std,
        "Single_tx_ms": float(benchmark_tabular_latency(tabular_models["RandomForest"], sample_1k_tab[:1])[0]),
        "PR-AUC": float(table1_df.loc[table1_df['Model'] == 'RandomForest', 'PR-AUC'].values[0]),
    })
    latency_rows.append({
        "Model": "LogisticRegression",
        "Latency_per_1k_ms": lr_lat_mean,
        "Latency_Std_ms": lr_lat_std,
        "Single_tx_ms": float(benchmark_tabular_latency(tabular_models["LogisticRegression"], sample_1k_neu[:1])[0]),
        "PR-AUC": float(table1_df.loc[table1_df['Model'] == 'LogisticRegression', 'PR-AUC'].values[0]),
    })
    latency_rows.append({
        "Model": "GraphSAGE",
        "Latency_per_1k_ms": gs_lat["total_mean"],
        "Latency_Std_ms": gs_lat["total_std"],
        "Single_tx_ms": float(benchmark_gnn_latency(graphsage_model, graph_data, test_idx, 1, config, device)["total_mean"]),
        "PR-AUC": float(table1_df.loc[table1_df['Model'] == 'GraphSAGE', 'PR-AUC'].values[0]),
    })

    if gat_model is not None:
        gat_lat = benchmark_gnn_latency(gat_model, graph_data, test_idx, 1000, config, device)
        gnn_breakdown["GAT"] = gat_lat
        latency_rows.append({
            "Model": "GAT",
            "Latency_per_1k_ms": gat_lat["total_mean"],
            "Latency_Std_ms": gat_lat["total_std"],
            "Single_tx_ms": float(benchmark_gnn_latency(gat_model, graph_data, test_idx, 1, config, device)["total_mean"]),
            "PR-AUC": float(table1_df.loc[table1_df['Model'] == 'GAT', 'PR-AUC'].values[0]),
        })

    table2_df = pd.DataFrame(latency_rows)
    print("\n--- TABLE 2: INFERENCE LATENCY BENCHMARK (ms) ---")
    print(table2_df.to_string(index=False))
    export_results_table(table2_df, "table2_latency_benchmark", config,
                         caption="Inference Latency per 1,000 Transactions and Single-Sample Latency (Mean $\\pm$ Std in ms)",
                         label="latency_benchmark")

    # Visualizations: Latency vs Accuracy & Latency Breakdown
    plot_accuracy_latency_tradeoff(table2_df, config)
    plot_latency_breakdown(gnn_breakdown, config)

    # ------------------------------------------------------------------
    # Step 9: Cold-Start & Isolated Nodes Robustness Analysis
    # ------------------------------------------------------------------
    print("\n>>> STAGE 9: Cold-Start & Isolated Nodes Analysis")
    df_robustness = evaluate_cold_start_robustness(
        tabular_models, graphsage_model, data_dict, graph_data, config, device
    )
    if not df_robustness.empty:
        print(df_robustness.to_string(index=False))
        export_results_table(df_robustness, "table3_cold_start_robustness", config,
                             caption="Performance Comparison on Isolated vs. Connected Transactions",
                             label="cold_start_robustness")
        plot_cold_start_comparison(df_robustness, config)

    # ------------------------------------------------------------------
    # Step 10: TimeSeriesSplit Cross-Validation (Optional)
    # ------------------------------------------------------------------
    if args.run_cv:
        print("\n>>> STAGE 10: 5-Fold TimeSeriesSplit Cross-Validation")
        cv_results = run_temporal_cross_validation(data_dict, graph_data, config, device, n_splits=config["cv_splits"])
        print(f"  XGBoost 5-Fold AUC:   {[round(x, 4) for x in cv_results['xgb_fold_auc']]}")
        print(f"  GraphSAGE 5-Fold AUC: {[round(x, 4) for x in cv_results['graphsage_fold_auc']]}")
        print("  Paired t-test results:")
        for k, v in cv_results["ttest"].items():
            print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # Step 11: Model Serialization (Native & ONNX)
    # ------------------------------------------------------------------
    print("\n>>> STAGE 11: Model Serialization (Native Checkpoints & ONNX)")
    save_native_models(tabular_models, graphsage_model, gat_model, config, hybrid_model=hybrid_model)
    export_tabular_to_onnx(tabular_models, n_features=data_dict["X_tabular"].shape[1], config=config)

    t_total = time.time() - t_pipeline_start
    print("\n" + "=" * 75)
    print(f" BENCHMARK RUN COMPLETE IN {t_total / 60:.1f} MINUTES ({t_total:.1f}s)")
    print(f" Results exported to: {config['results_dir']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
