"""Train the ChemBERTa-embedding QSPR melting-point ensemble.

Featurizes the assembled training set (``ml_des_mp/mp_train.csv``) with the same
ChemBERTa embedder used elsewhere in the project, then trains a deep ensemble of
MLP heads. The ensemble mean is the melting point; the ensemble spread is the
epistemic uncertainty surfaced as a confidence in ``property_resolution``.

Usage:
    python -m ml_des_mp.train_mp_qspr            # train + save artifact
    python -m ml_des_mp.train_mp_qspr --members 5 --epochs 400
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from des_multi_agent.paths import ensure_ml_des_path
from des_multi_agent.predictors.melting_point import MPRegressor

ensure_ml_des_path()
from src.embeddings.chemberta import ChemBERTaEmbedder  # noqa: E402

EMBEDDING_MODEL = "DeepChem/ChemBERTa-77M-MTR"
# 128 matches the DES embedder config so the two ChemBERTa instances can be
# shared (only ~1/3000 training SMILES exceeds 128 chars).
MAX_LENGTH = 128
EMB_CACHE_DIR = ".cache/chemberta_emb"


def _load_rows(csv_path: Path):
    smiles, tm = [], []
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        smiles.append(r["smiles"])
        tm.append(float(r["tm_k"]))
    return smiles, np.array(tm, dtype=np.float64)


def _train_member(X, y, seed, epochs, lr, device, emb_dim, hidden, dropout):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = X.shape[0]
    idx = torch.randint(0, n, (n,), generator=g)  # bootstrap for ensemble diversity
    xb, yb = X[idx].to(device), y[idx].to(device)
    torch.manual_seed(seed)
    model = MPRegressor(emb_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    return model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ml_des_mp/mp_train.csv")
    ap.add_argument("--out", default="artifacts/melting_points/qspr_model.pt")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--calib-frac", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage (0.1 -> 90% intervals)")
    ap.add_argument("--refit-full", action="store_true", default=True,
                    help="deploy a model refit on all data (metrics/calibration stay from the held-out split)")
    ap.add_argument("--no-refit-full", dest="refit_full", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles, tm = _load_rows(Path(args.csv))
    print(f"Loaded {len(smiles)} compounds; embedding with {EMBEDDING_MODEL} "
          f"(max_length={MAX_LENGTH}) on {device}...")
    embedder = ChemBERTaEmbedder(
        model_name=EMBEDDING_MODEL, pooling="mean", batch_size=64,
        max_length=MAX_LENGTH, device=device, cache_dir=EMB_CACHE_DIR,
    )
    emb = np.asarray(embedder.embed(smiles), dtype=np.float32)

    # train / calibration / test split: train fits the ensemble, calibration
    # fits the conformal quantile + confidence scale, test reports honest metrics
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(smiles))
    n_test = int(len(smiles) * args.test_frac)
    n_calib = int(len(smiles) * args.calib_frac)
    test_idx = perm[:n_test]
    calib_idx = perm[n_test:n_test + n_calib]
    train_idx = perm[n_test + n_calib:]

    feat_mean = emb[train_idx].mean(0)
    feat_std = emb[train_idx].std(0) + 1e-6
    tm_mean = float(tm[train_idx].mean())
    tm_std = float(tm[train_idx].std() + 1e-6)

    def _x(idx):
        return torch.tensor((emb[idx] - feat_mean) / feat_std, dtype=torch.float32).to(device)

    Xtr = torch.tensor((emb[train_idx] - feat_mean) / feat_std, dtype=torch.float32)
    ytr = torch.tensor((tm[train_idx] - tm_mean) / tm_std, dtype=torch.float32)

    emb_dim = emb.shape[1]
    members = [
        _train_member(Xtr, ytr, args.seed + k, args.epochs, args.lr, device,
                      emb_dim, args.hidden, args.dropout)
        for k in range(args.members)
    ]

    def _ensemble(idx):
        with torch.no_grad():
            preds = torch.stack([m(_x(idx)) for m in members])  # (members, n)
            mean_k = (preds.mean(0) * tm_std + tm_mean).cpu().numpy()
            std_k = (preds.std(0, unbiased=False) * tm_std).cpu().numpy()
        return mean_k, std_k

    # --- conformal calibration on the calibration split ---
    alpha = args.alpha
    cmean, cstd = _ensemble(calib_idx)
    cresid = np.abs(tm[calib_idx] - cmean)
    norm_resid = cresid / np.maximum(cstd, 1e-6)
    # finite-sample-adjusted (1-alpha) quantile of normalized residuals
    n_cal = len(calib_idx)
    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    conformal_q = float(np.quantile(norm_resid, q_level))
    # data-calibrated confidence scale: 90th-pct ensemble std on calibration
    std_scale_k = float(np.quantile(cstd, 0.90))

    # --- honest test metrics + conformal coverage ---
    tmean, tstd = _ensemble(test_idx)
    yte = tm[test_idx]
    err = tmean - yte
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    covered = np.abs(err) <= conformal_q * np.maximum(tstd, 1e-6)
    coverage = float(covered.mean())
    half_width = float((conformal_q * np.maximum(tstd, 1e-6)).mean())
    print(f"Test (n={n_test}): RMSE={rmse:.1f} K  MAE={mae:.1f} K | "
          f"conformal {int((1-alpha)*100)}% coverage={coverage:.2f} "
          f"(target {1-alpha:.2f}), mean half-width={half_width:.1f} K | "
          f"std_scale={std_scale_k:.1f} K, conformal_q={conformal_q:.2f}")

    # Deploy a model refit on ALL data for best accuracy; honest metrics and the
    # conformal calibration above come from the held-out split.
    if args.refit_full:
        print("Refitting deployed ensemble on all data...")
        feat_mean = emb.mean(0)
        feat_std = emb.std(0) + 1e-6
        tm_mean = float(tm.mean())
        tm_std = float(tm.std() + 1e-6)
        Xall = torch.tensor((emb - feat_mean) / feat_std, dtype=torch.float32)
        yall = torch.tensor((tm - tm_mean) / tm_std, dtype=torch.float32)
        members = [
            _train_member(Xall, yall, args.seed + 100 + k, args.epochs, args.lr, device,
                          emb_dim, args.hidden, args.dropout)
            for k in range(args.members)
        ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "member_states": [{k: v.cpu() for k, v in m.state_dict().items()} for m in members],
            "emb_dim": emb_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "feat_mean": feat_mean.tolist(),
            "feat_std": feat_std.tolist(),
            "tm_mean": tm_mean,
            "tm_std": tm_std,
            "embedding_model": EMBEDDING_MODEL,
            "max_length": MAX_LENGTH,
            "emb_cache_dir": EMB_CACHE_DIR,
            "calibration": {
                "std_scale_k": std_scale_k,
                "conformal_q": conformal_q,
                "alpha": alpha,
                "coverage": coverage,
                "half_width_k": half_width,
            },
            "metrics": {"rmse_k": rmse, "mae_k": mae, "n_test": n_test},
        },
        out,
    )
    print(f"Saved QSPR ensemble ({args.members} members) to {out}")


if __name__ == "__main__":
    main()
