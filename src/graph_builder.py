"""
Causal transaction graph construction with anti-supernode frequency capping.

Design principles:
  1. Edges connect transactions sharing high-cardinality entities
     (card1, DeviceInfo, P_emaildomain).
  2. DeviceType is EXCLUDED: only 2 values -> billions of edges -> OOM.
  3. Entities with frequency > max_entity_freq (500) are filtered out
     to prevent mega-hub nodes (e.g., gmail.com).
  4. All edges are strictly causal: t_source <= t_target.
  5. Each node links to at most max_neighbors_per_node (20) prior transactions.
  6. Vectorized implementation using pandas.groupby for performance.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def build_causal_transaction_graph(data_dict: dict, config: dict) -> Data:
    """
    Construct a directed causal graph where transactions are nodes and edges
    connect transactions sharing high-signal entity identifiers.

    Args:
        data_dict: Output from preprocessing.load_and_preprocess().
        config: Global CONFIG dictionary.

    Returns:
        torch_geometric.data.Data object with node features, edge_index, labels, masks.
    """
    raw_entities = data_dict["raw_entities"]
    timestamps = data_dict["timestamps"]
    n_nodes = len(data_dict["y"])
    max_neighbors = config["max_neighbors_per_node"]

    all_src = []
    all_dst = []
    edge_records = []

    print("[GRAPH] Building causal edges from entity columns ...")
    for col in config["edge_cols"]:
        print(f"  Processing column: {col}")

        series = raw_entities[col].copy()
        # Drop NaN entries (no entity -> no edge)
        valid_mask = series.notna()
        valid_series = series[valid_mask].astype(str)

        # Count frequencies and filter
        freq = valid_series.value_counts()
        # Keep entities with 2 <= frequency <= max_entity_freq
        valid_entities = freq[(freq > 1) & (freq <= config["max_entity_freq"])].index
        filtered = valid_series[valid_series.isin(valid_entities)]

        raw_count = len(freq)
        filt_count = len(valid_entities)
        cov_count = len(filtered)

        print(f"    Entities before filtering: {raw_count:,}")
        print(f"    Entities after filtering:  {filt_count:,}")
        print(f"    Transactions covered:      {cov_count:,}")

        # Group transaction indices by shared entity value
        # Use vectorized approach: build a DataFrame and use groupby
        edge_df = pd.DataFrame({
            "node_idx": filtered.index.values,
            "entity": filtered.values
        })

        col_src = []
        col_dst = []

        for _, group in edge_df.groupby("entity"):
            indices = group["node_idx"].values
            # Indices are already in chronological order (DataFrame was sorted by TransactionDT)
            k = len(indices)
            for i in range(1, k):
                # Connect to at most max_neighbors prior transactions
                start_j = max(0, i - max_neighbors)
                for j in range(start_j, i):
                    src_node = indices[j]
                    dst_node = indices[i]
                    # Enforce causal ordering (should be guaranteed by sort, but be safe)
                    if timestamps[src_node] <= timestamps[dst_node]:
                        col_src.append(src_node)
                        col_dst.append(dst_node)

        col_edge_count = len(col_src)
        print(f"    Edges from {col}: {col_edge_count:,}")
        edge_records.append({
            "Entity Identifier": col,
            "Raw Entities": raw_count,
            "Filtered Entities (2<=f<=500)": filt_count,
            "Transactions Covered": cov_count,
            "Generated Directed Edges": col_edge_count,
        })

        all_src.extend(col_src)
        all_dst.extend(col_dst)

    # ------------------------------------------------------------------
    # Deduplicate edges (a pair may share multiple entity columns)
    # ------------------------------------------------------------------
    total_raw = len(all_src)
    print(f"[GRAPH] Total raw edges (before dedup): {total_raw:,}")
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    unique_edge_count = edge_index.shape[1]
    print(f"[GRAPH] Unique edges (after dedup):     {unique_edge_count:,}")

    # Build Table 2: Edge Statistics DataFrame
    stats_df = pd.DataFrame(edge_records)
    stats_df["Percentage of Graph Edges (%)"] = (
        (stats_df["Generated Directed Edges"] / max(total_raw, 1)) * 100.0
    ).round(2)

    total_row = pd.DataFrame([{
        "Entity Identifier": "Total (Deduplicated)",
        "Raw Entities": stats_df["Raw Entities"].sum(),
        "Filtered Entities (2<=f<=500)": stats_df["Filtered Entities (2<=f<=500)"].sum(),
        "Transactions Covered": stats_df["Transactions Covered"].sum(),
        "Generated Directed Edges": unique_edge_count,
        "Percentage of Graph Edges (%)": 100.0,
    }])
    stats_df = pd.concat([stats_df, total_row], ignore_index=True)

    # Export Table 2 to results directory if configured
    if "tables_dir" in config:
        os.makedirs(config["tables_dir"], exist_ok=True)
        csv_p = os.path.join(config["tables_dir"], "edge_statistics.csv")
        tex_p = os.path.join(config["tables_dir"], "edge_statistics.tex")
        stats_df.to_csv(csv_p, index=False)
        with open(tex_p, "w", encoding="utf-8") as f:
            f.write(stats_df.to_latex(
                index=False,
                caption="Causal Transaction Graph Construction Statistics across Entity Identifiers (Table 2)",
                label="tab:edge_stats",
                column_format="lrrrrr"
            ))
        print(f"[GRAPH] Table 2 saved -> {csv_p} and {tex_p}")

    # ------------------------------------------------------------------
    # Build node masks for train / val / test
    # ------------------------------------------------------------------
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[data_dict["train_idx"]] = True
    val_mask[data_dict["val_idx"]] = True
    test_mask[data_dict["test_idx"]] = True

    # ------------------------------------------------------------------
    # Assemble PyG Data object
    # ------------------------------------------------------------------
    graph_data = Data(
        x=torch.tensor(data_dict["X_neural"], dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(data_dict["y"], dtype=torch.float32),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # ------------------------------------------------------------------
    # Report graph topology statistics
    # ------------------------------------------------------------------
    degrees = torch.zeros(n_nodes, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
    isolated = (degrees == 0).sum().item()
    connected = (degrees >= 3).sum().item()

    print(f"[GRAPH] Final Graph Statistics:")
    print(f"  Nodes:            {graph_data.num_nodes:,}")
    print(f"  Directed Edges:   {graph_data.num_edges:,}")
    print(f"  Avg In-Degree:    {degrees.float().mean():.2f}")
    print(f"  Max In-Degree:    {degrees.max().item()}")
    print(f"  Isolated Nodes:   {isolated:,} ({isolated / n_nodes * 100:.1f}%)")
    print(f"  Connected (d>=3): {connected:,} ({connected / n_nodes * 100:.1f}%)")
    graph_data.edge_stats_df = stats_df
    print("[GRAPH] Done.")

    return graph_data


def get_graph_statistics(graph_data: Data) -> dict:
    """
    Extract topology statistics for the edge-type ablation study and reporting.

    Returns:
        dict with node_count, edge_count, avg_degree, max_degree,
              isolated_count, connected_count.
    """
    n = graph_data.num_nodes
    edge_index = graph_data.edge_index

    degrees = torch.zeros(n, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))

    return {
        "node_count": n,
        "edge_count": graph_data.num_edges,
        "avg_degree": float(degrees.float().mean()),
        "max_degree": int(degrees.max().item()),
        "isolated_count": int((degrees == 0).sum().item()),
        "connected_count": int((degrees >= 3).sum().item()),
    }
