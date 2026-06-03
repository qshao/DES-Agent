from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch

from src.utils import get_device, resolve_path
from src.data import load_dataset, DatasetColumns
from src.embeddings.factory import build_embedder
from src.models.model import DESPhysicsModel, DESPhysicsModelWithProjector, ProjectorConfig
from src.train import _predict_Tm_from_params


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Preferred path: use the unified model with an optional projector.
    if "projector_cfg" in ckpt:
        proj_cfg = ProjectorConfig(**ckpt["projector_cfg"])
        model = DESPhysicsModel(emb_dim_in=int(ckpt["emb_dim_in"]), projector=proj_cfg).to(device)
    else:
        model = DESPhysicsModel(emb_dim_in=int(ckpt["emb_dim_in"])).to(device)

    # Backward-compat: if the state dict doesn't match (very old checkpoints), fall back.
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
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True, help="External validation CSV (same column names as config)")
    ap.add_argument("--out_csv", default=None, help="If provided, write predictions here")
    args = ap.parse_args()

    config_path = resolve_path(args.config, base_dir=Path(__file__).resolve().parent)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = get_device(cfg.get("device", "cuda"))

    cols = DatasetColumns(
        smiles1=cfg["data"]["smiles1_col"],
        smiles2=cfg["data"]["smiles2_col"],
        t1=cfg["data"]["t1_col"],
        t2=cfg["data"]["t2_col"],
        frac1=cfg["data"]["frac1_col"],
        tm=cfg["data"]["tm_col"],
    )

    csv_path = resolve_path(args.csv, base_dir=config_path.parent)
    df = load_dataset(csv_path, cols)
    ckpt_path = resolve_path(args.ckpt, base_dir=config_path.parent)
    model = load_model(str(ckpt_path), device)

    # Embedding method is still controlled by config.yaml
    emb_bundle = build_embedder(cfg["embedding"], device=device)

    if emb_bundle.kind == "gnn":
        raise ValueError(
            "predict.py supports fixed-feature embedders (chemberta/morgan/rdkit). "
            "For GNN, use the trained fold checkpoint(s) and run an end-to-end inference script."
        )

    X1 = emb_bundle.embedder.embed(df[cols.smiles1].tolist())
    X2 = emb_bundle.embedder.embed(df[cols.smiles2].tolist())

    X1_t = torch.tensor(X1, device=device)
    X2_t = torch.tensor(X2, device=device)
    T1 = torch.tensor(df[cols.t1].values.astype(np.float32), device=device)
    T2 = torch.tensor(df[cols.t2].values.astype(np.float32), device=device)
    r = torch.tensor(df[cols.frac1].values.astype(np.float32), device=device)
    y_true = df[cols.tm].values.astype(np.float32)

    with torch.no_grad():
        d1, d2, W = model.forward_params(X1_t, X2_t)
        y_pred = _predict_Tm_from_params(d1, d2, W, T1, T2, r).detach().cpu().numpy()

    out = df.copy()
    out["Tm_pred_K"] = y_pred

    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"Wrote predictions to {args.out_csv}")
    else:
        print(out[[cols.smiles1, cols.smiles2, cols.frac1, cols.tm, "Tm_pred_K"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
