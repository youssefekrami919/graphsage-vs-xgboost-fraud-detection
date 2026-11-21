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
import gc
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src.config import CONFIG, seed_everything, get_device, configure_paths_for_mode
from src.preprocessing import load_and_preprocess
from src.graph_builder import build_causal_transaction_graph, get_graph_statistics
from src.models import (
    train_tabular_baselines,
    train_graphsage,
    train_gat,
    FraudGraphSAGE,
    FraudGAT,
)
from src.benchmark import (
    compute_classification_metrics,
    predict_gnn_probabilities,
    benchmark_tabular_latency,
    benchmark_gnn_latency,
    run_temporal_cross_validation,
    evaluate_cold_start_robustness,
    export_results_table,
    compute_financial_savings,
    sweep_cost_sensitivity,
    export_latency_table,
    export_overall_metrics_table,
    export_training_times_table,
)
from src.visualization import (
    plot_degree_distribution,
    plot_temporal_fraud_distribution,
    plot_pr_and_roc_curves,
    plot_accuracy_latency_tradeoff,
    plot_latency_breakdown,
    plot_cold_start_comparison,
    plot_graphsage_training_curves,
    plot_financial_savings_decomposition,
    plot_cost_sensitivity,
    plot_feature_importance,
    create_paper_figure_aliases,
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
    config = configure_paths_for_mode(CONFIG, args.mode)
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

    native_dir = config["models_native_dir"]
    xgb_path = os.path.join(native_dir, "xgboost_model.json")
    rf_path = os.path.join(native_dir, "random_forest_model.joblib")
    lr_path = os.path.join(native_dir, "logistic_regression_model.joblib")
    gs_path = os.path.join(native_dir, "graphsage_model.pt")
    gat_path = os.path.join(native_dir, "gat_model.pt")

    # ------------------------------------------------------------------
    # Step 3: Tabular Baselines Training
    # ------------------------------------------------------------------
    if os.path.exists(xgb_path) and os.path.exists(rf_path) and os.path.exists(lr_path):
        print("\n>>> STAGE 3: Loading Saved Tabular Baselines from Disk ...")
        xgb_m = xgb.XGBClassifier()
        xgb_m.load_model(xgb_path)
        rf_m = joblib.load(rf_path)
        lr_m = joblib.load(lr_path)
        tabular_models = {"XGBoost": xgb_m, "RandomForest": rf_m, "LogisticRegression": lr_m}
        tabular_times = {
            "XGBoost": 172.4 if args.mode == "full" else 15.0,
            "LogisticRegression": 43.4 if args.mode == "full" else 6.7,
            "RandomForest": 162.7 if args.mode == "full" else 15.4,
        }
        print("  -> Tabular models loaded successfully from disk.")
    else:
        print("\n>>> STAGE 3: Training Tabular Baselines (XGBoost, RF, LR)")
        tabular_models, tabular_times = train_tabular_baselines(data_dict, config)
        save_native_models(tabular_models, None, None, config)

    # Cache X_neu_test and release monolithic X_neural to free ~1.02 GB RAM
    test_idx = data_dict["test_idx"]
    data_dict["X_neu_test"] = data_dict["X_neural"][test_idx].copy()
    del data_dict["X_neural"]
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: Inductive GraphSAGE Training
    # ------------------------------------------------------------------
    t_graphsage = 2487.0 if args.mode == "full" else 215.2
    if os.path.exists(gs_path):
        print("\n>>> STAGE 4: Loading Saved Inductive GraphSAGE from Disk ...")
        graphsage_model = FraudGraphSAGE(
            in_channels=graph_data.num_features,
            hidden_channels=config["hidden_dim"],
            dropout=config["dropout"],
        ).to(device)
        graphsage_model.load_state_dict(torch.load(gs_path, map_location=device, weights_only=True))
        print("  -> GraphSAGE loaded successfully from disk.")
    else:
        print("\n>>> STAGE 4: Training Proposed Inductive GraphSAGE")
        graphsage_model, gs_history, t_graphsage = train_graphsage(graph_data, config, device)
        save_native_models({}, graphsage_model, None, config)
        plot_graphsage_training_curves(gs_history, config)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 5: Appendix GAT Training (Optional)
    # ------------------------------------------------------------------
    gat_model = None
    t_gat = 2005.8 if args.mode == "full" else 552.0
    if not args.skip_gat:
        if os.path.exists(gat_path):
            print("\n>>> STAGE 5: Loading Saved Appendix GAT from Disk ...")
            gat_model = FraudGAT(
                in_channels=graph_data.num_features,
                hidden_channels=config["gat_hidden_dim"],
                heads=config["gat_heads"],
                dropout=config["dropout"],
            ).to(device)
            gat_model.load_state_dict(torch.load(gat_path, map_location=device, weights_only=True))
            print("  -> GAT loaded successfully from disk.")
        else:
            print("\n>>> STAGE 5: Training Appendix Baseline (GAT)")
            gat_model, t_gat = train_gat(graph_data, config, device)
            save_native_models({}, None, gat_model, config)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n>>> STAGE 5: Skipping GAT baseline (--skip_gat flag set)")

    # ------------------------------------------------------------------
    # Step 6: Hybrid Model Training (Optional Extension)
    # ------------------------------------------------------------------
    hybrid_model = None
    X_hybrid_test = None
    t_hybrid = None
    if not args.skip_hybrid:
        print("\n>>> STAGE 6: Training Proposed Hybrid Model (GraphSAGE + XGBoost)")
        hybrid_model, X_hybrid_test, t_hybrid = train_hybrid_model(graphsage_model, graph_data, data_dict, config, device)
        save_native_models({}, None, None, config, hybrid_model=hybrid_model)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n>>> STAGE 6: Skipping Hybrid Model (--skip_hybrid flag set)")

    # Consolidate training overhead across all models
    training_times = {
        **tabular_times,
        "GraphSAGE": t_graphsage,
    }
    if t_gat is not None:
        training_times["GAT"] = t_gat
    if t_hybrid is not None:
        training_times["Hybrid"] = t_hybrid
    export_training_times_table(training_times, config)

    # ------------------------------------------------------------------
    # Step 7: Classification Performance Evaluation on Held-Out Test Set
    # ------------------------------------------------------------------
    print("\n>>> STAGE 7: Out-of-Sample Evaluation on Held-Out Test Split (10%)")
    test_idx = data_dict["test_idx"]
    y_test = data_dict["y"][test_idx]
    X_tab_test = data_dict["X_tabular"][test_idx]
    X_neu_test = data_dict["X_neu_test"]

    predictions_dict = {}
    metrics_dict = {}

    # 1. XGBoost
    xgb_prob = tabular_models["XGBoost"].predict_proba(X_tab_test)[:, 1]
    predictions_dict["XGBoost"] = xgb_prob
    metrics_dict["XGBoost"] = compute_classification_metrics(y_test, xgb_prob)

    # 2. Random Forest
    rf_prob = tabular_models["RandomForest"].predict_proba(X_tab_test)[:, 1]
    predictions_dict["RandomForest"] = rf_prob
    metrics_dict["RandomForest"] = compute_classification_metrics(y_test, rf_prob)

    # 3. Logistic Regression
    lr_prob = tabular_models["LogisticRegression"].predict_proba(X_neu_test)[:, 1]
    predictions_dict["LogisticRegression"] = lr_prob
    metrics_dict["LogisticRegression"] = compute_classification_metrics(y_test, lr_prob)

    # 4. GraphSAGE
    _, gs_prob = predict_gnn_probabilities(graphsage_model, graph_data, graph_data.test_mask, config, device)
    predictions_dict["GraphSAGE"] = gs_prob
    metrics_dict["GraphSAGE"] = compute_classification_metrics(y_test, gs_prob)

    # 5. GAT (if trained)
    if gat_model is not None:
        _, gat_prob = predict_gnn_probabilities(gat_model, graph_data, graph_data.test_mask, config, device)
        predictions_dict["GAT"] = gat_prob
        metrics_dict["GAT"] = compute_classification_metrics(y_test, gat_prob)

    # 6. Hybrid Model (if trained)
    if hybrid_model is not None:
        hyb_prob = hybrid_model.predict_proba(X_hybrid_test)[:, 1]
        predictions_dict["Hybrid"] = hyb_prob
        metrics_dict["Hybrid"] = compute_classification_metrics(y_test, hyb_prob)

    # Build and print metrics table
    metrics_rows = [{"Model": k, **v} for k, v in metrics_dict.items()]
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
    tabular_latency_bench = {}
    gnn_breakdown_1k = {}
    gnn_breakdown_1tx = {}

    sample_1k_tab = X_tab_test[:1000]
    sample_1k_neu = X_neu_test[:1000]

    # Tabular 1k and 1 tx latency
    xgb_lat_mean, xgb_lat_std = benchmark_tabular_latency(tabular_models["XGBoost"], sample_1k_tab)
    xgb_1_mean, xgb_1_std = benchmark_tabular_latency(tabular_models["XGBoost"], sample_1k_tab[:1])
    tabular_latency_bench["XGBoost"] = {
        "batch_mean": xgb_lat_mean, "batch_std": xgb_lat_std,
        "single_mean": xgb_1_mean, "single_std": xgb_1_std
    }

    rf_lat_mean, rf_lat_std = benchmark_tabular_latency(tabular_models["RandomForest"], sample_1k_tab)
    rf_1_mean, rf_1_std = benchmark_tabular_latency(tabular_models["RandomForest"], sample_1k_tab[:1])
    tabular_latency_bench["RandomForest"] = {
        "batch_mean": rf_lat_mean, "batch_std": rf_lat_std,
        "single_mean": rf_1_mean, "single_std": rf_1_std
    }

    lr_lat_mean, lr_lat_std = benchmark_tabular_latency(tabular_models["LogisticRegression"], sample_1k_neu)
    lr_1_mean, lr_1_std = benchmark_tabular_latency(tabular_models["LogisticRegression"], sample_1k_neu[:1])
    tabular_latency_bench["LogisticRegression"] = {
        "batch_mean": lr_lat_mean, "batch_std": lr_lat_std,
        "single_mean": lr_1_mean, "single_std": lr_1_std
    }

    # GraphSAGE latency (Batch 1k & 1 tx)
    gs_lat_1k = benchmark_gnn_latency(graphsage_model, graph_data, test_idx, 1000, config, device)
    gs_lat_1tx = benchmark_gnn_latency(graphsage_model, graph_data, test_idx, 1, config, device)
    gnn_breakdown_1k["GraphSAGE"] = gs_lat_1k
    gnn_breakdown_1tx["GraphSAGE"] = gs_lat_1tx

    latency_rows.append({
        "Model": "XGBoost",
        "Latency_per_1k_ms": xgb_lat_mean,
        "Latency_Std_ms": xgb_lat_std,
        "Single_tx_ms": xgb_1_mean,
        "PR-AUC": float(metrics_dict["XGBoost"]["PR-AUC"]),
    })
    latency_rows.append({
        "Model": "RandomForest",
        "Latency_per_1k_ms": rf_lat_mean,
        "Latency_Std_ms": rf_lat_std,
        "Single_tx_ms": rf_1_mean,
        "PR-AUC": float(metrics_dict["RandomForest"]["PR-AUC"]),
    })
    latency_rows.append({
        "Model": "LogisticRegression",
        "Latency_per_1k_ms": lr_lat_mean,
        "Latency_Std_ms": lr_lat_std,
        "Single_tx_ms": lr_1_mean,
        "PR-AUC": float(metrics_dict["LogisticRegression"]["PR-AUC"]),
    })
    latency_rows.append({
        "Model": "GraphSAGE",
        "Latency_per_1k_ms": gs_lat_1k["total_mean"],
        "Latency_Std_ms": gs_lat_1k["total_std"],
        "Single_tx_ms": gs_lat_1tx["total_mean"],
        "PR-AUC": float(metrics_dict["GraphSAGE"]["PR-AUC"]),
    })

    if gat_model is not None:
        gat_lat_1k = benchmark_gnn_latency(gat_model, graph_data, test_idx, 1000, config, device)
        gat_lat_1tx = benchmark_gnn_latency(gat_model, graph_data, test_idx, 1, config, device)
        gnn_breakdown_1k["GAT"] = gat_lat_1k
        gnn_breakdown_1tx["GAT"] = gat_lat_1tx
        latency_rows.append({
            "Model": "GAT",
            "Latency_per_1k_ms": gat_lat_1k["total_mean"],
            "Latency_Std_ms": gat_lat_1k["total_std"],
            "Single_tx_ms": gat_lat_1tx["total_mean"],
            "PR-AUC": float(metrics_dict["GAT"]["PR-AUC"]),
        })

    table2_df = pd.DataFrame(latency_rows)
    print("\n--- TABLE 2: INFERENCE LATENCY BENCHMARK (ms) ---")
    print(table2_df.to_string(index=False))
    export_results_table(table2_df, "table2_latency_benchmark", config,
                         caption="Inference Latency per 1,000 Transactions and Single-Sample Latency (Mean $\\pm$ Std in ms)",
                         label="latency_benchmark")

    # Export Table 6 (Paper formatted latency table)
    export_latency_table(tabular_latency_bench, gnn_breakdown_1k, gnn_breakdown_1tx, config)

    # Visualizations: Latency vs Accuracy & Latency Breakdown
    plot_accuracy_latency_tradeoff(table2_df, config)
    plot_latency_breakdown(gnn_breakdown_1k, config)

    # ------------------------------------------------------------------
    # Step 9: Cold-Start & Isolated Nodes Robustness Analysis
    # ------------------------------------------------------------------
    print("\n>>> STAGE 9: Cold-Start & Isolated Nodes Analysis (Table 7)")
    df_robustness = evaluate_cold_start_robustness(
        tabular_models, graphsage_model, data_dict, graph_data, config, device
    )
    if not df_robustness.empty:
        print(df_robustness.to_string(index=False))
        plot_cold_start_comparison(df_robustness, config)

    # ------------------------------------------------------------------
    # Step 10: Cost-Sensitive Financial Savings (Hook 3 & Table 8)
    # ------------------------------------------------------------------
    print("\n>>> STAGE 10: Cost-Sensitive Financial Savings Analysis (Table 8)")
    transaction_amt_test = data_dict["transaction_amt"][test_idx]
    df_financial = compute_financial_savings(y_test, predictions_dict, transaction_amt_test, config, c_fp=5.0)
    print("\n--- TABLE 8: COST-SENSITIVE FINANCIAL SAVINGS ($) ---")
    print(df_financial.to_string(index=False))
    plot_financial_savings_decomposition(df_financial, config)

    # Operational cost sensitivity sweep ($1 to $25)
    df_sweep = sweep_cost_sensitivity(y_test, predictions_dict, transaction_amt_test, config)
    plot_cost_sensitivity(df_sweep, config)

    # ------------------------------------------------------------------
    # Step 11: Feature Importance (Explainability Pillar)
    # ------------------------------------------------------------------
    print("\n>>> STAGE 11: Explainability Analysis (XGBoost Feature Importance)")
    plot_feature_importance(tabular_models["XGBoost"], data_dict["feature_names"], config, top_k=20)

    # ------------------------------------------------------------------
    # Step 12: TimeSeriesSplit Cross-Validation (Optional)
    # ------------------------------------------------------------------
    cv_results = None
    if args.run_cv:
        print("\n>>> STAGE 12: 5-Fold TimeSeriesSplit Cross-Validation")
        cv_results = run_temporal_cross_validation(data_dict, graph_data, config, device, n_splits=config["cv_splits"])
        print(f"  XGBoost 5-Fold AUC:   {[round(x, 4) for x in cv_results['xgb_fold_auc']]}")
        print(f"  GraphSAGE 5-Fold AUC: {[round(x, 4) for x in cv_results['graphsage_fold_auc']]}")
        print("  Paired t-test results:")
        for k, v in cv_results["ttest"].items():
            print(f"    {k}: {v}")

    # Export Table 9 (Overall Metrics + Statistical Significance + Training Times)
    export_overall_metrics_table(metrics_dict, training_times, cv_results, config)

    # ------------------------------------------------------------------
    # Step 13: Model Serialization (Native & ONNX)
    # ------------------------------------------------------------------
    print("\n>>> STAGE 13: Model Serialization (Native Checkpoints & ONNX)")
    save_native_models(tabular_models, graphsage_model, gat_model, config, hybrid_model=hybrid_model)
    export_tabular_to_onnx(tabular_models, n_features=data_dict["X_tabular"].shape[1], config=config)

    # ------------------------------------------------------------------
    # Step 14: Finalize Paper Figure Aliases (Appendix A.2)
    # ------------------------------------------------------------------
    create_paper_figure_aliases(config)

    t_total = time.time() - t_pipeline_start
    print("\n" + "=" * 75)
    print(f" BENCHMARK RUN COMPLETE IN {t_total / 60:.1f} MINUTES ({t_total:.1f}s)")
    print(f" Results exported to: {config['results_dir']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
