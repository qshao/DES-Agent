from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, r2_score

from src.data import DatasetColumns, load_dataset, add_pair_key
from src.embeddings.factory import build_embedder
from src.models.model import DESPhysicsModel, DESPhysicsModelWithProjector, ProjectorConfig
from src.splits import kfold_random_row, kfold_strict_pair
from src.train import _predict_Tm_from_params
from src.utils import get_device


def _safe_slug(s: str, max_len: int = 120) -> str:
    s = re.sub(r"\s+", "_", str(s))
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    if len(s) > max_len:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
        s = s[: max_len - 11] + "_" + h
    return s


RUN_RE = re.compile(
    r"^(?P<method>[A-Za-z0-9_]+)_(?P<split>random_row|strict_pair)_fold(?P<fold>\d+)of(?P<k>\d+)_best\.pt$"
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    split_method: str
    fold_i: int
    k_folds: int
    ckpt_path: str
    history_path: str
    run_stem: str


@dataclass(frozen=True)
class CanonicalPair:
    key: str
    s1: str
    s2: str
    T1: float
    T2: float


@dataclass
class EvalBundle:
    run: RunSpec
    ckpt: dict
    model: torch.nn.Module
    test_idx: np.ndarray
    y_pred_test: np.ndarray


@dataclass(frozen=True)
class CanonicalCurveAnchor:
    pair_key: str
    emb_smiles1: str
    emb_smiles2: str
    T1: float
    T2: float


@dataclass
class PairCurveResult:
    y_curve: np.ndarray
    d1: float
    d2: float
    W: float


def _parse_run_spec(ckpt_path: str) -> RunSpec:
    base = os.path.basename(ckpt_path)
    m = RUN_RE.match(base)
    if not m:
        raise ValueError(
            "Checkpoint filename must look like '<embedder>_<split_method>_foldXXofYY_best.pt', "
            f"but got: {base}"
        )
    run_stem = base[: -len("_best.pt")]
    return RunSpec(
        method=str(m.group("method")),
        split_method=str(m.group("split")),
        fold_i=int(m.group("fold")),
        k_folds=int(m.group("k")),
        ckpt_path=ckpt_path,
        history_path=os.path.join(os.path.dirname(ckpt_path), f"{run_stem}_history.npz"),
        run_stem=run_stem,
    )


def _companion_run_spec(primary: RunSpec) -> Optional[RunSpec]:
    other_split = "strict_pair" if primary.split_method == "random_row" else "random_row"
    pattern = os.path.join(
        os.path.dirname(primary.ckpt_path),
        f"{primary.method}_{other_split}_fold{primary.fold_i:02d}of{primary.k_folds:02d}_best.pt",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return _parse_run_spec(matches[0])


def plot_history(history: dict, out_dir: str, prefix: str, mae_ylim: float = 50.0):
    steps = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(steps, history["train_loss"], label="train")
    plt.plot(steps, history["test_loss"], label="test")
    plt.xlabel("logged step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(steps, history["train_mae"], label="train")
    plt.plot(steps, history["test_mae"], label="test")
    plt.xlabel("logged step")
    plt.ylabel("MAE (K)")
    plt.ylim(0, float(mae_ylim))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_mae.png"), dpi=200)
    plt.close()


def _annot_metrics(ax, y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    txt = f"R²: {r2:.3f}\nMAE: {mae:.2f} K"
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str):
    assert len(y_true) == len(y_pred), "y_true and y_pred must be the same length."

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=16, alpha=0.65, label="Historical test fold")
    lims = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    ax.plot(lims, lims, label="y=x")
    ax.set_xlabel("Actual Tm (K)")
    ax.set_ylabel("Predicted Tm (K)")
    ax.set_title(title)
    _annot_metrics(ax, y_true, y_pred)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_melting_curve(
    *,
    title: str,
    x_grid: np.ndarray,
    y_curve: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    textbox: str,
    out_path: str,
):
    fig, ax = plt.subplots(figsize=(12.8, 3.6))
    if len(x_train) > 0:
        ax.scatter(x_train, y_train, marker="x", s=55, linewidths=1.8, label="Train Data")
    if len(x_test) > 0:
        ax.scatter(x_test, y_test, s=55, alpha=0.9, edgecolors="k", linewidths=0.6, label="Test Data")
    ax.plot(x_grid, y_curve, linewidth=2.0, label="Surrogate Model")

    ax.set_xlabel(r"Mole Fraction ($x_1$)")
    ax.set_ylabel(r"Temperature ($T_{melt}$, K)")
    ax.set_title(title)

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.35)

    ax.text(
        0.5,
        0.83,
        textbox,
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_melting_curve_both_splits(
    *,
    title: str,
    x_grid: np.ndarray,
    y_curve: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
    x_test_random: np.ndarray,
    y_test_random: np.ndarray,
    x_test_strict: np.ndarray,
    y_test_strict: np.ndarray,
    textbox: str,
    out_path: str,
):
    fig, ax = plt.subplots(figsize=(12.8, 3.6))

    if len(x_all) > 0:
        ax.scatter(x_all, y_all, s=28, alpha=0.25, label="All Data")

    if len(x_test_random) > 0:
        ax.scatter(
            x_test_random,
            y_test_random,
            s=60,
            alpha=0.95,
            edgecolors="k",
            linewidths=0.6,
            label="Test (Random Row)",
        )

    if len(x_test_strict) > 0:
        ax.scatter(
            x_test_strict,
            y_test_strict,
            marker="s",
            s=55,
            alpha=0.95,
            edgecolors="k",
            linewidths=0.6,
            label="Test (Strict Pair)",
        )

    ax.plot(x_grid, y_curve, linewidth=2.0, label="Surrogate Model")

    ax.set_xlabel(r"Mole Fraction ($x_1$)")
    ax.set_ylabel(r"Temperature ($T_{melt}$, K)")
    ax.set_title(title)

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.35)

    ax.text(
        0.5,
        0.83,
        textbox,
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_ckpt_and_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if "emb_dim_in" not in ckpt or "model_state" not in ckpt:
        raise ValueError(
            "This checkpoint doesn't look like a fixed-feature DES checkpoint. "
            "If this was trained with embedding.method=gnn, use a GNN-specific interpretation script."
        )

    if "projector_cfg" in ckpt:
        proj_cfg = ProjectorConfig(**ckpt["projector_cfg"])
        model = DESPhysicsModel(emb_dim_in=int(ckpt["emb_dim_in"]), projector=proj_cfg).to(device)
    else:
        model = DESPhysicsModel(emb_dim_in=int(ckpt["emb_dim_in"])).to(device)

    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError:
        if "projector_cfg" in ckpt:
            proj_cfg = ProjectorConfig(**ckpt["projector_cfg"])
            legacy = DESPhysicsModelWithProjector(emb_dim_in=int(ckpt["emb_dim_in"]), projector=proj_cfg).to(device)
            legacy.load_state_dict(ckpt["model_state"])
            model = legacy
        else:
            raise

    model.eval()
    return ckpt, model


def _load_history(history_path: str) -> Optional[dict]:
    if not os.path.exists(history_path):
        return None
    h = np.load(history_path, allow_pickle=True)
    return {k: h[k].tolist() for k in h.files}


def _canonicalize_pair(df_pair, cols: DatasetColumns, pair_key: str) -> CanonicalPair:
    row0 = df_pair.iloc[0]
    s1_row0 = str(row0[cols.smiles1])
    s2_row0 = str(row0[cols.smiles2])
    if s1_row0 <= s2_row0:
        canon_s1 = s1_row0
        canon_s2 = s2_row0
        T1 = float(row0[cols.t1])
        T2 = float(row0[cols.t2])
    else:
        canon_s1 = s2_row0
        canon_s2 = s1_row0
        T1 = float(row0[cols.t2])
        T2 = float(row0[cols.t1])

    return CanonicalPair(
        key=str(row0[pair_key]),
        s1=canon_s1,
        s2=canon_s2,
        T1=T1,
        T2=T2,
    )


def _map_row_to_canonical_x(row, cols: DatasetColumns, canon: CanonicalPair) -> float:
    s1 = str(row[cols.smiles1])
    s2 = str(row[cols.smiles2])
    r = float(row[cols.frac1])
    if s1 == canon.s1 and s2 == canon.s2:
        return r
    if s1 == canon.s2 and s2 == canon.s1:
        return 1.0 - r
    return r


def _build_curve_anchor(df_pair, cols: DatasetColumns, canon: CanonicalPair) -> CanonicalCurveAnchor:
    mask_forward = (
        (df_pair[cols.smiles1].astype(str) == canon.s1)
        & (df_pair[cols.smiles2].astype(str) == canon.s2)
    )
    if bool(mask_forward.any()):
        row = df_pair.loc[mask_forward].iloc[0]
        return CanonicalCurveAnchor(
            pair_key=canon.key,
            emb_smiles1=canon.s1,
            emb_smiles2=canon.s2,
            T1=float(row[cols.t1]),
            T2=float(row[cols.t2]),
        )

    mask_reverse = (
        (df_pair[cols.smiles1].astype(str) == canon.s2)
        & (df_pair[cols.smiles2].astype(str) == canon.s1)
    )
    if bool(mask_reverse.any()):
        row = df_pair.loc[mask_reverse].iloc[0]
        return CanonicalCurveAnchor(
            pair_key=canon.key,
            emb_smiles1=canon.s1,
            emb_smiles2=canon.s2,
            T1=float(row[cols.t2]),
            T2=float(row[cols.t1]),
        )

    raise ValueError(f"Could not build canonical curve anchor for pair_key={canon.key}")


def _predict_curve_for_anchor(
    *,
    model,
    device,
    embedder,
    anchor: CanonicalCurveAnchor,
    x_grid: np.ndarray,
) -> PairCurveResult:
    x1 = embedder.embed([anchor.emb_smiles1])
    x2 = embedder.embed([anchor.emb_smiles2])

    x1_t = torch.tensor(x1, device=device)
    x2_t = torch.tensor(x2, device=device)

    T1_t = torch.tensor(np.full_like(x_grid, anchor.T1, dtype=np.float32), device=device)
    T2_t = torch.tensor(np.full_like(x_grid, anchor.T2, dtype=np.float32), device=device)
    r_t = torch.tensor(x_grid.astype(np.float32), device=device)

    with torch.no_grad():
        d1, d2, W = model.forward_params(x1_t, x2_t)
        d1g = d1.repeat(len(x_grid))
        d2g = d2.repeat(len(x_grid))
        Wg = W.repeat(len(x_grid))
        y_curve = _predict_Tm_from_params(d1g, d2g, Wg, T1_t, T2_t, r_t).detach().cpu().numpy()

    return PairCurveResult(y_curve=y_curve, d1=float(d1.item()), d2=float(d2.item()), W=float(W.item()))


def _predict_rows(*, model, device, X1: np.ndarray, X2: np.ndarray, T1: np.ndarray, T2: np.ndarray, r: np.ndarray) -> np.ndarray:
    X1_t = torch.tensor(X1, device=device)
    X2_t = torch.tensor(X2, device=device)
    T1_t = torch.tensor(T1.astype(np.float32), device=device)
    T2_t = torch.tensor(T2.astype(np.float32), device=device)
    r_t = torch.tensor(r.astype(np.float32), device=device)

    with torch.no_grad():
        d1, d2, W = model.forward_params(X1_t, X2_t)
        y_pred = _predict_Tm_from_params(d1, d2, W, T1_t, T2_t, r_t).detach().cpu().numpy()
    return y_pred


def _get_fold_indices(df, split_method: str, pair_key_col: str, k_folds: int, seed: int, fold_i: int) -> Tuple[np.ndarray, np.ndarray]:
    if split_method == "random_row":
        folds = kfold_random_row(df, k=k_folds, seed=seed)
    elif split_method == "strict_pair":
        folds = kfold_strict_pair(df, pair_key=pair_key_col, k=k_folds, seed=seed)
    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    if not (1 <= fold_i <= len(folds)):
        raise ValueError(f"fold_i must be within [1, {len(folds)}], got {fold_i}")
    return folds[fold_i - 1]


def _make_eval_bundle(
    *,
    run: RunSpec,
    cfg: dict,
    device: torch.device,
    df,
    pair_key_col: str,
    X1: np.ndarray,
    X2: np.ndarray,
    T1_all: np.ndarray,
    T2_all: np.ndarray,
    r_all: np.ndarray,
) -> EvalBundle:
    ckpt, model = load_ckpt_and_model(run.ckpt_path, device)
    seed0 = int(ckpt.get("cfg", cfg).get("seed", cfg.get("seed", 42)))
    _, test_idx = _get_fold_indices(
        df=df,
        split_method=run.split_method,
        pair_key_col=pair_key_col,
        k_folds=run.k_folds,
        seed=seed0,
        fold_i=run.fold_i,
    )
    y_pred_test = _predict_rows(
        model=model,
        device=device,
        X1=X1[test_idx],
        X2=X2[test_idx],
        T1=T1_all[test_idx],
        T2=T2_all[test_idx],
        r=r_all[test_idx],
    )
    return EvalBundle(run=run, ckpt=ckpt, model=model, test_idx=np.asarray(test_idx), y_pred_test=y_pred_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="runs/interpret")
    ap.add_argument("--mae_ylim", type=float, default=50.0)
    ap.add_argument("--skip_diagnostics", action="store_true")
    ap.add_argument("--max_pairs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "cuda"))
    os.makedirs(args.out_dir, exist_ok=True)

    primary_run = _parse_run_spec(args.ckpt)
    companion_run = _companion_run_spec(primary_run)

    primary_ckpt, primary_model = load_ckpt_and_model(primary_run.ckpt_path, device)
    train_cfg = primary_ckpt.get("cfg", cfg)

    primary_history = _load_history(primary_run.history_path)
    if primary_history is not None:
        plot_history(primary_history, args.out_dir, prefix=f"history__{primary_run.split_method}", mae_ylim=float(args.mae_ylim))

    if companion_run is not None:
        companion_history = _load_history(companion_run.history_path)
        if companion_history is not None:
            plot_history(companion_history, args.out_dir, prefix=f"history__{companion_run.split_method}", mae_ylim=float(args.mae_ylim))
    else:
        warnings.warn(
            "Matching companion checkpoint for the other split was not found in the same directory. "
            "Only the split from --ckpt will have diagnostic and per-split curve outputs."
        )

    cols = DatasetColumns(
        smiles1=train_cfg["data"]["smiles1_col"],
        smiles2=train_cfg["data"]["smiles2_col"],
        t1=train_cfg["data"]["t1_col"],
        t2=train_cfg["data"]["t2_col"],
        frac1=train_cfg["data"]["frac1_col"],
        tm=train_cfg["data"]["tm_col"],
    )
    df = load_dataset(train_cfg["data"]["csv_path"], cols)
    pair_key_col = "_pair_key"
    df[pair_key_col] = add_pair_key(df, cols.smiles1, cols.smiles2)

    emb_bundle = build_embedder(train_cfg["embedding"], device=device)
    if emb_bundle.kind == "gnn":
        raise ValueError("interpret_updated.py supports fixed-feature embedders only (chemberta/morgan/rdkit).")

    X1 = emb_bundle.embedder.embed(df[cols.smiles1].tolist())
    X2 = emb_bundle.embedder.embed(df[cols.smiles2].tolist())

    expected_dim = int(primary_ckpt["emb_dim_in"])
    if X1.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dim mismatch: checkpoint expects emb_dim_in={expected_dim}, "
            f"but embedder '{emb_bundle.kind}' produced dim={X1.shape[1]}."
        )

    T1_all = df[cols.t1].values.astype(np.float32)
    T2_all = df[cols.t2].values.astype(np.float32)
    r_all = df[cols.frac1].values.astype(np.float32)
    y_all = df[cols.tm].values.astype(np.float32)

    evals: Dict[str, EvalBundle] = {}
    primary_bundle = _make_eval_bundle(
        run=primary_run,
        cfg=cfg,
        device=device,
        df=df,
        pair_key_col=pair_key_col,
        X1=X1,
        X2=X2,
        T1_all=T1_all,
        T2_all=T2_all,
        r_all=r_all,
    )
    primary_bundle.ckpt = primary_ckpt
    primary_bundle.model = primary_model
    primary_bundle.y_pred_test = _predict_rows(
        model=primary_model,
        device=device,
        X1=X1[primary_bundle.test_idx],
        X2=X2[primary_bundle.test_idx],
        T1=T1_all[primary_bundle.test_idx],
        T2=T2_all[primary_bundle.test_idx],
        r=r_all[primary_bundle.test_idx],
    )
    evals[primary_run.split_method] = primary_bundle

    if companion_run is not None:
        evals[companion_run.split_method] = _make_eval_bundle(
            run=companion_run,
            cfg=cfg,
            device=device,
            df=df,
            pair_key_col=pair_key_col,
            X1=X1,
            X2=X2,
            T1_all=T1_all,
            T2_all=T2_all,
            r_all=r_all,
        )

    if not args.skip_diagnostics:
        for split_name, bundle in evals.items():
            out_path = os.path.join(args.out_dir, f"pred_vs_actual__{split_name}__historical_test_fold.png")
            title = f"Pred vs Actual (historical fold {bundle.run.fold_i:02d}/{bundle.run.k_folds:02d}) — {split_name}"
            plot_pred_vs_actual(
                y_true=y_all[bundle.test_idx],
                y_pred=bundle.y_pred_test,
                out_path=out_path,
                title=title,
            )

    curves_dir = os.path.join(args.out_dir, "melting_curves")
    os.makedirs(curves_dir, exist_ok=True)
    for sub in ["random_row", "strict_pair", "both_splits"]:
        os.makedirs(os.path.join(curves_dir, sub), exist_ok=True)

    unique_pairs = df[pair_key_col].unique().tolist()
    if args.max_pairs is not None:
        unique_pairs = unique_pairs[: int(args.max_pairs)]

    x_grid = np.linspace(0.0, 1.0, 201, dtype=np.float32)

    rr_test_set = set(map(int, evals["random_row"].test_idx.tolist())) if "random_row" in evals else set()
    sp_test_set = set(map(int, evals["strict_pair"].test_idx.tolist())) if "strict_pair" in evals else set()

    curve_bundle = evals.get(primary_run.split_method)
    curve_model_label = f"curve model: {primary_run.split_method} fold {primary_run.fold_i:02d}/{primary_run.k_folds:02d}"

    for j, pkey in enumerate(unique_pairs):
        df_pair = df[df[pair_key_col] == pkey].copy()
        if len(df_pair) == 0:
            continue

        canon = _canonicalize_pair(df_pair, cols, pair_key_col)
        anchor = _build_curve_anchor(df_pair, cols, canon)
        curve_res = _predict_curve_for_anchor(
            model=curve_bundle.model,
            device=device,
            embedder=emb_bundle.embedder,
            anchor=anchor,
            x_grid=x_grid,
        )

        xs, ys, global_indices = [], [], []
        for idx, row in df_pair.iterrows():
            xs.append(_map_row_to_canonical_x(row, cols, canon))
            ys.append(float(row[cols.tm]))
            global_indices.append(int(idx))

        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        global_indices = np.asarray(global_indices, dtype=np.int64)

        order = np.argsort(xs, kind="stable")
        xs = xs[order]
        ys = ys[order]
        global_indices = global_indices[order]

        rr_is_test = np.array([int(i) in rr_test_set for i in global_indices], dtype=bool)
        sp_is_test = np.array([int(i) in sp_test_set for i in global_indices], dtype=bool)

        textbox = (
            f"delta1: {curve_res.d1:.2f}\n"
            f"delta2: {curve_res.d2:.2f}\n"
            f"W: {curve_res.W:.2f}\n"
            f"{curve_model_label}"
        )
        title = f"System: {canon.s1} + {canon.s2}"
        fname_base = f"{j:05d}__{_safe_slug(pkey)}"

        if "random_row" in evals:
            plot_melting_curve(
                title=title,
                x_grid=x_grid,
                y_curve=curve_res.y_curve,
                x_train=xs[~rr_is_test],
                y_train=ys[~rr_is_test],
                x_test=xs[rr_is_test],
                y_test=ys[rr_is_test],
                textbox=textbox,
                out_path=os.path.join(curves_dir, "random_row", f"{fname_base}.png"),
            )

        if "strict_pair" in evals:
            plot_melting_curve(
                title=title,
                x_grid=x_grid,
                y_curve=curve_res.y_curve,
                x_train=xs[~sp_is_test],
                y_train=ys[~sp_is_test],
                x_test=xs[sp_is_test],
                y_test=ys[sp_is_test],
                textbox=textbox,
                out_path=os.path.join(curves_dir, "strict_pair", f"{fname_base}.png"),
            )

        plot_melting_curve_both_splits(
            title=title,
            x_grid=x_grid,
            y_curve=curve_res.y_curve,
            x_all=xs,
            y_all=ys,
            x_test_random=xs[rr_is_test],
            y_test_random=ys[rr_is_test],
            x_test_strict=xs[sp_is_test],
            y_test_strict=ys[sp_is_test],
            textbox=textbox,
            out_path=os.path.join(curves_dir, "both_splits", f"{fname_base}.png"),
        )

    print(f"Saved interpretation plots to: {args.out_dir}")
    print(f"Melting curves saved under: {os.path.join(args.out_dir, 'melting_curves')}")


if __name__ == "__main__":
    main()
