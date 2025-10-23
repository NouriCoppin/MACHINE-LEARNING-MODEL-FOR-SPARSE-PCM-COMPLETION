import pandas as pd
import numpy as np

def load_pairs_csv(path: str):
    df = pd.read_csv(path)
    assert {"source_id","target_id","label"}.issubset(df.columns)
    n = int(max(df["source_id"].max(), df["target_id"].max())) + 1
    return df, n

def split_edges(df, val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(len(df)*val_frac)
    n_test = int(len(df)*test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val+n_test]
    train_idx = idx[n_val+n_test:]
    return df.iloc[train_idx].reset_index(drop=True), \               df.iloc[val_idx].reset_index(drop=True), \               df.iloc[test_idx].reset_index(drop=True)
