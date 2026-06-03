from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _train_test_split_indices(n_items: int, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n_items))
    rng.shuffle(idx)
    n_test = int(round(float(test_size) * int(n_items)))
    n_test = max(1, min(int(n_items) - 1, n_test))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return train_idx, test_idx


# -----------------------------------------------------------------------------
# Legacy single train/test split helpers (kept for backward compatibility)
# -----------------------------------------------------------------------------

def split_random_row(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    return _train_test_split_indices(len(df), test_size=test_size, seed=seed)


def split_strict_pair(df: pd.DataFrame, pair_key: str, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    keys = df[pair_key].astype(str).values
    unique_keys = np.unique(keys)
    if len(unique_keys) < 2:
        raise ValueError("split_strict_pair requires at least two unique pair keys")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_keys)
    n_test = int(round(float(test_size) * len(unique_keys)))
    n_test = max(1, min(len(unique_keys) - 1, n_test))
    test_keys = set(unique_keys[:n_test])
    train_keys = set(unique_keys[n_test:])
    train_mask = np.array([k in train_keys for k in keys])
    test_mask = np.array([k in test_keys for k in keys])
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if not set(df.loc[train_idx, pair_key]).isdisjoint(set(df.loc[test_idx, pair_key])):
        raise ValueError("split_strict_pair produced overlapping pair keys")
    return train_idx, test_idx


# -----------------------------------------------------------------------------
# New: 5-fold CV helpers
# -----------------------------------------------------------------------------

def kfold_random_row(df: pd.DataFrame, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(df))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    folds = np.array_split(idx, int(k))
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(int(k)):
        test_idx = np.sort(folds[i])
        train_idx = np.sort(np.concatenate([folds[j] for j in range(int(k)) if j != i]))
        out.append((train_idx, test_idx))
    return out


def kfold_strict_pair(df: pd.DataFrame, pair_key: str, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if int(k) < 2:
        raise ValueError("kfold_strict_pair requires k >= 2")
    keys = df[pair_key].astype(str).values
    unique_keys = np.unique(keys)
    if len(unique_keys) < int(k):
        raise ValueError("kfold_strict_pair requires at least k unique pair keys")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_keys)
    folds = np.array_split(unique_keys, int(k))
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(int(k)):
        test_keys = set(folds[i].tolist())
        train_keys = set(np.concatenate([folds[j] for j in range(int(k)) if j != i]).tolist())
        train_mask = np.array([k in train_keys for k in keys])
        test_mask = np.array([k in test_keys for k in keys])
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if not set(df.loc[train_idx, pair_key]).isdisjoint(set(df.loc[test_idx, pair_key])):
            raise ValueError("kfold_strict_pair produced overlapping pair keys")
        out.append((train_idx, test_idx))
    return out
