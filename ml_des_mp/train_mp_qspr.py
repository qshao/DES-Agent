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
MAX_LENGTH = 256


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
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles, tm = _load_rows(Path(args.csv))
    print(f"Loaded {len(smiles)} compounds; embedding with {EMBEDDING_MODEL} on {device}...")
    embedder = ChemBERTaEmbedder(
        model_name=EMBEDDING_MODEL, pooling="mean", batch_size=64,
        max_length=MAX_LENGTH, device=device,
    )
    emb = np.asarray(embedder.embed(smiles), dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(smiles))
    n_test = int(len(smiles) * args.test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    feat_mean = emb[train_idx].mean(0)
    feat_std = emb[train_idx].std(0) + 1e-6
    tm_mean = float(tm[train_idx].mean())
    tm_std = float(tm[train_idx].std() + 1e-6)

    Xtr = torch.tensor((emb[train_idx] - feat_mean) / feat_std, dtype=torch.float32)
    ytr = torch.tensor((tm[train_idx] - tm_mean) / tm_std, dtype=torch.float32)
    Xte = torch.tensor((emb[test_idx] - feat_mean) / feat_std, dtype=torch.float32).to(device)

    emb_dim = emb.shape[1]
    members = [
        _train_member(Xtr, ytr, args.seed + k, args.epochs, args.lr, device,
                      emb_dim, args.hidden, args.dropout)
        for k in range(args.members)
    ]

    # held-out metrics
    with torch.no_grad():
        preds = torch.stack([m(Xte) for m in members])          # (members, n_test)
        mean_k = (preds.mean(0) * tm_std + tm_mean).cpu().numpy()
        std_k = (preds.std(0, unbiased=False) * tm_std).cpu().numpy()
    yte = tm[test_idx]
    err = mean_k - yte
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    # correlation between predicted uncertainty and absolute error (calibration)
    cal = float(np.corrcoef(std_k, np.abs(err))[0, 1]) if n_test > 2 else float("nan")
    print(f"Held-out (n={n_test}): RMSE={rmse:.1f} K  MAE={mae:.1f} K  "
          f"mean_pred_std={std_k.mean():.1f} K  unc-err corr={cal:.2f}")

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
            "metrics": {"rmse_k": rmse, "mae_k": mae, "n_test": n_test},
        },
        out,
    )
    print(f"Saved QSPR ensemble ({args.members} members) to {out}")


if __name__ == "__main__":
    main()
