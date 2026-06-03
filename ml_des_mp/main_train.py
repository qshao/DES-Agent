from __future__ import annotations

import argparse
from pathlib import Path
import json
import os

import numpy as np
import torch
import yaml

from src.data import DatasetColumns, add_pair_key, load_dataset
from src.embeddings.factory import build_embedder
from src.splits import kfold_random_row, kfold_strict_pair
from src.train import fit_one_run
from src.train_gnn import fit_one_run_gnn
from src.utils import ensure_dir, get_device, resolve_path, set_seed


def _to_tensors(df, cols: DatasetColumns, device: torch.device):
    T1 = torch.tensor(df[cols.t1].values.astype(np.float32), device=device)
    T2 = torch.tensor(df[cols.t2].values.astype(np.float32), device=device)
    r = torch.tensor(df[cols.frac1].values.astype(np.float32), device=device)
    y = df[cols.tm].values.astype(np.float32)
    return T1, T2, r, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    config_path = resolve_path(args.config, base_dir=Path(__file__).resolve().parent)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed0 = int(cfg.get("seed", 42))
    set_seed(seed0)
    device = get_device(cfg.get("device", "cuda"))

    out_dir = resolve_path(cfg["output"]["dir"], base_dir=config_path.parent)
    ensure_dir(out_dir)

    cols = DatasetColumns(
        smiles1=cfg["data"]["smiles1_col"],
        smiles2=cfg["data"]["smiles2_col"],
        t1=cfg["data"]["t1_col"],
        t2=cfg["data"]["t2_col"],
        frac1=cfg["data"]["frac1_col"],
        tm=cfg["data"]["tm_col"],
    )
    csv_path = resolve_path(cfg["data"]["csv_path"], base_dir=config_path.parent)
    df = load_dataset(csv_path, cols)
    df["pair_key"] = add_pair_key(df, cols.smiles1, cols.smiles2)

    # Build embedder once (method is controlled by config.yaml)
    emb_bundle = build_embedder(cfg["embedding"], device=device)
    method = emb_bundle.kind

    # Prepare embeddings (fixed-feature), or use end-to-end GNN.
    if method != "gnn":
        smiles1_all = df[cols.smiles1].tolist()
        smiles2_all = df[cols.smiles2].tolist()
        X1_all = emb_bundle.embedder.embed(smiles1_all)
        X2_all = emb_bundle.embedder.embed(smiles2_all)
        emb_dim_in = int(X1_all.shape[1])
    else:
        X1_all = None
        X2_all = None
        emb_dim_in = None

    split_methods = list(cfg["splits"].get("split_methods", ["random_row", "strict_pair"]))
    k_folds = int(cfg["splits"].get("k_folds", 5))

    summary = {}

    # shared tensors for T1/T2/r/y
    T1_all, T2_all, r_all, y_all = _to_tensors(df, cols, device)

    for split_method in split_methods:
        if split_method == "random_row":
            folds = kfold_random_row(df, k=k_folds, seed=seed0)
        elif split_method == "strict_pair":
            folds = kfold_strict_pair(df, "pair_key", k=k_folds, seed=seed0)
        else:
            raise ValueError(f"Unknown split method: {split_method}")

        train_maes = []
        test_maes = []

        for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
            run_name = f"{method}_{split_method}_fold{fold_i:02d}of{k_folds:02d}"

            if method == "gnn":
                res = fit_one_run_gnn(
                    smiles1_train=df.loc[train_idx, cols.smiles1].tolist(),
                    smiles2_train=df.loc[train_idx, cols.smiles2].tolist(),
                    T1_train=T1_all[train_idx],
                    T2_train=T2_all[train_idx],
                    r_train=r_all[train_idx],
                    y_train=y_all[train_idx],
                    smiles1_test=df.loc[test_idx, cols.smiles1].tolist(),
                    smiles2_test=df.loc[test_idx, cols.smiles2].tolist(),
                    T1_test=T1_all[test_idx],
                    T2_test=T2_all[test_idx],
                    r_test=r_all[test_idx],
                    y_test=y_all[test_idx],
                    cfg=cfg,
                    out_dir=out_dir,
                    device=device,
                    run_name=run_name,
                )
            else:
                X1_train = torch.tensor(X1_all[train_idx], device=device)
                X2_train = torch.tensor(X2_all[train_idx], device=device)
                X1_test = torch.tensor(X1_all[test_idx], device=device)
                X2_test = torch.tensor(X2_all[test_idx], device=device)

                res = fit_one_run(
                    X1_train=X1_train,
                    X2_train=X2_train,
                    T1_train=T1_all[train_idx],
                    T2_train=T2_all[train_idx],
                    r_train=r_all[train_idx],
                    Tm_train=y_all[train_idx],
                    X1_test=X1_test,
                    X2_test=X2_test,
                    T1_test=T1_all[test_idx],
                    T2_test=T2_all[test_idx],
                    r_test=r_all[test_idx],
                    Tm_test=y_all[test_idx],
                    emb_dim_in=emb_dim_in,
                    cfg=cfg,
                    out_dir=out_dir,
                    device=device,
                    run_name=run_name,
                )

            # Save history (small)
            hist_path = os.path.join(out_dir, f"{run_name}_history.npz")
            np.savez(hist_path, **{k: np.array(v, dtype=np.float32) for k, v in res.history.items()})

            train_maes.append(res.train_mae)
            test_maes.append(res.test_mae)

            print(
                f"{split_method} | fold {fold_i:02d}/{k_folds:02d} | "
                f"train MAE: {res.train_mae:.3f} K | test MAE: {res.test_mae:.3f} K"
            )

        avg_train = float(np.mean(train_maes))
        std_train = float(np.std(train_maes, ddof=1)) if len(train_maes) > 1 else 0.0
        avg_test = float(np.mean(test_maes))
        std_test = float(np.std(test_maes, ddof=1)) if len(test_maes) > 1 else 0.0

        summary[split_method] = {
            "train_mae": train_maes,
            "test_mae": test_maes,
            "avg_train_mae": avg_train,
            "std_train_mae": std_train,
            "avg_test_mae": avg_test,
            "std_test_mae": std_test,
        }

        print(
            f"{split_method} | AVG | "
            f"train MAE: {avg_train:.3f} ± {std_train:.3f} K | "
            f"test MAE: {avg_test:.3f} ± {std_test:.3f} K"
        )

    with open(os.path.join(out_dir, f"summary_{method}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
