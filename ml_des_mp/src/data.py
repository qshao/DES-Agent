from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import pandas as pd

@dataclass(frozen=True)
class DatasetColumns:
    smiles1: str
    smiles2: str
    t1: str
    t2: str
    frac1: str
    tm: str

def load_dataset(csv_path: str, cols: DatasetColumns) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = [cols.smiles1, cols.smiles2, cols.t1, cols.t2, cols.frac1, cols.tm]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    out = df[needed].copy()
    out = out.dropna().reset_index(drop=True)
    # Ensure numeric
    out[cols.t1] = pd.to_numeric(out[cols.t1], errors="coerce")
    out[cols.t2] = pd.to_numeric(out[cols.t2], errors="coerce")
    out[cols.frac1] = pd.to_numeric(out[cols.frac1], errors="coerce")
    out[cols.tm] = pd.to_numeric(out[cols.tm], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out

def add_pair_key(df: pd.DataFrame, smiles1_col: str, smiles2_col: str) -> pd.Series:
    # Unordered pair key: ensures strict split doesn't leak (A,B) into train when (B,A) in test.
    s1 = df[smiles1_col].astype(str)
    s2 = df[smiles2_col].astype(str)
    a = s1.where(s1 <= s2, s2)
    b = s2.where(s1 <= s2, s1)
    return a + "||" + b
