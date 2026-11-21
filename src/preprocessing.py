"""
Leak-free data preprocessing with chronological splitting and memory optimization.

Key safeguards:
  1. Chronological sort by TransactionDT (no shuffling).
  2. TransactionID dropped from features to prevent spurious memorization.
  3. StandardScaler fitted ONLY on training indices (no val/test leakage).
  4. Automatic dtype downcasting to minimize RAM on 16GB hardware.
  5. Two feature matrices produced:
     - X_tabular: missing values filled with -999 (native for tree models).
     - X_neural:  missing values filled with training median + StandardScaler.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory footprint by downcasting numeric columns.
    float64 -> float32, int64 -> int32/int16 where safe.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    return df


def load_and_preprocess(config: dict) -> dict:
    """
    Load IEEE-CIS data, merge tables, apply leak-free preprocessing,
    and return chronologically split feature matrices.

    Returns:
        dict with keys: X_tabular, X_neural, y, raw_entities, timestamps,
                        train_idx, val_idx, test_idx, feature_names
    """
    data_dir = config["data_dir"]

    # ------------------------------------------------------------------
    # 1. Load and merge the two CSV tables
    # ------------------------------------------------------------------
    print("[PREPROCESSING] Loading train_transaction.csv ...")
    train_tx = pd.read_csv(
        f"{data_dir}/train_transaction.csv",
        dtype={col: np.float32 for col in [f"V{i}" for i in range(1, 340)]},
    )
    print(f"  -> Transactions loaded: {len(train_tx):,} rows")

    print("[PREPROCESSING] Loading train_identity.csv ...")
    train_id = pd.read_csv(f"{data_dir}/train_identity.csv")
    print(f"  -> Identities loaded:   {len(train_id):,} rows")

    print("[PREPROCESSING] Merging on TransactionID (LEFT JOIN) ...")
    df = train_tx.merge(train_id, on="TransactionID", how="left")
    del train_tx, train_id  # Free memory immediately
    print(f"  -> Merged shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Sort strictly by TransactionDT (chronological order)
    # ------------------------------------------------------------------
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Optional subsetting for prototype mode
    # ------------------------------------------------------------------
    if config["subset_size"] is not None:
        df = df.iloc[: config["subset_size"]].copy()
        print(f"[PREPROCESSING] Prototype mode: using first {config['subset_size']:,} transactions")

    # ------------------------------------------------------------------
    # 4. Memory optimization via dtype downcasting
    # ------------------------------------------------------------------
    mem_before = df.memory_usage(deep=True).sum() / (1024 ** 2)
    df = _downcast_dtypes(df)
    mem_after = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[PREPROCESSING] Memory: {mem_before:.0f} MB -> {mem_after:.0f} MB "
          f"(saved {mem_before - mem_after:.0f} MB)")

    # ------------------------------------------------------------------
    # 5. Preserve raw entity columns for graph construction
    # ------------------------------------------------------------------
    raw_entities = df[config["edge_cols"]].copy()
    timestamps = df["TransactionDT"].values
    transaction_amt = df["TransactionAmt"].values.astype(np.float32)
    y = df["isFraud"].values.astype(np.float32)

    print(f"[PREPROCESSING] Fraud rate: {y.mean() * 100:.2f}% "
          f"({int(y.sum()):,} / {len(y):,})")

    # ------------------------------------------------------------------
    # 6. Prepare feature columns (drop non-predictive columns)
    # ------------------------------------------------------------------
    drop_cols = ["TransactionID", "isFraud", "TransactionDT"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].copy()
    del df  # Free the full dataframe

    # Identify categorical vs numerical columns
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # ------------------------------------------------------------------
    # 7. Chronological 80 / 10 / 10 split (strict, no shuffling)
    # ------------------------------------------------------------------
    n = len(X_df)
    train_end = int(n * config["train_ratio"])
    val_end = int(n * (config["train_ratio"] + config["val_ratio"]))

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    print(f"[PREPROCESSING] Split sizes -> Train: {len(train_idx):,} | "
          f"Val: {len(val_idx):,} | Test: {len(test_idx):,}")

    # ------------------------------------------------------------------
    # 8. Encode categoricals via LabelEncoder
    # ------------------------------------------------------------------
    print("[PREPROCESSING] Encoding categorical features ...")
    label_encoders = {}
    for col in cat_cols:
        X_df[col] = X_df[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col])
        label_encoders[col] = le

    # ------------------------------------------------------------------
    # 9. Build X_tabular (for tree models: impute with -999)
    # ------------------------------------------------------------------
    X_num_raw = X_df[num_cols].fillna(-999.0).values.astype(np.float32)
    X_cat_raw = X_df[cat_cols].values.astype(np.float32)
    X_tabular = np.hstack([X_num_raw, X_cat_raw])

    # ------------------------------------------------------------------
    # 10. Build X_neural (for neural models: median imputation + StandardScaler)
    #     Scaler is fitted ONLY on training data to prevent leakage.
    # ------------------------------------------------------------------
    print("[PREPROCESSING] Scaling numerical features (fit on train only) ...")

    # Compute training medians for imputation
    train_medians = X_df[num_cols].iloc[train_idx].median()
    X_num_imputed = X_df[num_cols].fillna(train_medians).values.astype(np.float32)

    # Fit scaler only on training rows
    scaler = StandardScaler()
    X_num_scaled = np.zeros_like(X_num_imputed)
    X_num_scaled[train_idx] = scaler.fit_transform(X_num_imputed[train_idx])
    X_num_scaled[val_idx] = scaler.transform(X_num_imputed[val_idx])
    X_num_scaled[test_idx] = scaler.transform(X_num_imputed[test_idx])

    X_neural = np.hstack([X_num_scaled, X_cat_raw]).astype(np.float32)

    del X_df, X_num_raw, X_num_imputed, X_num_scaled, X_cat_raw  # Free memory

    print(f"[PREPROCESSING] Feature dimensions -> X_tabular: {X_tabular.shape}, "
          f"X_neural: {X_neural.shape}")
    print("[PREPROCESSING] Done.")

    return {
        "X_tabular": X_tabular,
        "X_neural": X_neural,
        "y": y,
        "raw_entities": raw_entities,
        "timestamps": timestamps,
        "transaction_amt": transaction_amt,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "feature_names": num_cols + cat_cols,
    }
