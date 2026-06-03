from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import os
import numpy as np
import torch
import torch.optim as optim

from .embeddings.gnn import GNNConfig, batch_smiles
from .models.model_gnn import DESPhysicsGNNModel


def _mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))


def _predict_Tm_from_params(d1, d2, W, T1, T2, r, R=8.314):
    eps = 1e-8
    r_c = torch.clamp(r, min=eps, max=1.0 - eps)
    T_ref = (T1 + T2) / 2.0
    ln_gamma1 = (W / (R * T_ref)) * (1 - r_c) ** 2
    ln_gamma2 = (W / (R * T_ref)) * (r_c) ** 2
    ln_a1 = torch.log(r_c) + ln_gamma1
    ln_a2 = torch.log(1 - r_c) + ln_gamma2
    denom1 = torch.clamp(1.0 - (R / d1) * ln_a1, min=0.1)
    denom2 = torch.clamp(1.0 - (R / d2) * ln_a2, min=0.1)
    return torch.max(T1 / denom1, T2 / denom2)


def _mae_batches(model: DESPhysicsGNNModel, b1, b2, T1, T2, r, y_true: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        d1, d2, W = model.forward_params_batches(b1, b2)
        y_pred = _predict_Tm_from_params(d1, d2, W, T1, T2, r).detach().cpu().numpy()
    return float(_mean_absolute_error(y_true, y_pred))


@dataclass
class TrainResult:
    train_mae: float
    test_mae: float
    history: Dict[str, List[float]]
    best_ckpt_path: str


def fit_one_run_gnn(
    *,
    smiles1_train: List[str],
    smiles2_train: List[str],
    T1_train: torch.Tensor,
    T2_train: torch.Tensor,
    r_train: torch.Tensor,
    y_train: np.ndarray,
    smiles1_test: List[str],
    smiles2_test: List[str],
    T1_test: torch.Tensor,
    T2_test: torch.Tensor,
    r_test: torch.Tensor,
    y_test: np.ndarray,
    cfg: Dict[str, Any],
    out_dir: str,
    device: torch.device,
    run_name: str,
) -> TrainResult:
    g = cfg["embedding"]["gnn"]
    gcfg = GNNConfig(
        emb_dim=int(g.get("emb_dim", 256)),
        num_layers=int(g.get("num_layers", 5)),
        hidden_dim=int(g.get("hidden_dim", 128)),
        dropout=float(g.get("dropout", 0.2)),
        conv=str(g.get("conv", "gin")),
        readout=str(g.get("readout", "mean")),
    )

    model = DESPhysicsGNNModel(gnn_cfg=gcfg).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    epochs = int(cfg["training"]["epochs"])
    grad_clip = float(cfg["training"]["grad_clip"])
    lambda_reg = float(cfg["training"]["lambda_reg"])
    log_every = int(cfg["training"].get("log_every", 1))

    b1_train = batch_smiles(smiles1_train, device)
    b2_train = batch_smiles(smiles2_train, device)
    b1_test = batch_smiles(smiles1_test, device)
    b2_test = batch_smiles(smiles2_test, device)

    best_test_mae = float("inf")
    best_path = os.path.join(out_dir, f"{run_name}_best.pt")
    history = {"train_loss": [], "test_loss": [], "train_mae": [], "test_mae": []}

    y_train_np = np.asarray(y_train, dtype=np.float32)
    y_test_np = np.asarray(y_test, dtype=np.float32)

    y_train_t = torch.tensor(y_train_np, device=device)
    y_test_t = torch.tensor(y_test_np, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss = model.physics_loss_batches(
            b1_train,
            b2_train,
            T1_train,
            T2_train,
            r_train,
            y_train_t,
            lambda_reg=lambda_reg,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_loss = model.physics_loss_batches(
                    b1_test,
                    b2_test,
                    T1_test,
                    T2_test,
                    r_test,
                    y_test_t,
                    lambda_reg=lambda_reg,
                )

            train_mae = _mae_batches(model, b1_train, b2_train, T1_train, T2_train, r_train, y_train_np)
            test_mae = _mae_batches(model, b1_test, b2_test, T1_test, T2_test, r_test, y_test_np)

            history["train_loss"].append(float(loss.item()))
            history["test_loss"].append(float(test_loss.item()))
            history["train_mae"].append(train_mae)
            history["test_mae"].append(test_mae)

            if test_mae < best_test_mae:
                best_test_mae = test_mae
                torch.save({"model_state": model.state_dict(), "cfg": cfg}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    final_train_mae = _mae_batches(model, b1_train, b2_train, T1_train, T2_train, r_train, y_train_np)
    final_test_mae = _mae_batches(model, b1_test, b2_test, T1_test, T2_test, r_test, y_test_np)

    return TrainResult(train_mae=final_train_mae, test_mae=final_test_mae, history=history, best_ckpt_path=best_path)
