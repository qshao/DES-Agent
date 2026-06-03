from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import os
import numpy as np
import torch
import torch.optim as optim

from .models.model import DESPhysicsModel, ProjectorConfig


def _mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))


@dataclass
class TrainResult:
    train_mae: float
    test_mae: float
    history: Dict[str, List[float]]
    best_ckpt_path: str


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


def _mae(model: DESPhysicsModel, x1, x2, T1, T2, r, Tm_true) -> float:
    model.eval()
    with torch.no_grad():
        d1, d2, W = model.forward_params(x1, x2)
        Tm_pred = _predict_Tm_from_params(d1, d2, W, T1, T2, r).detach().cpu().numpy()
    return float(_mean_absolute_error(Tm_true, Tm_pred))


def fit_one_run(
    *,
    X1_train: torch.Tensor,
    X2_train: torch.Tensor,
    T1_train: torch.Tensor,
    T2_train: torch.Tensor,
    r_train: torch.Tensor,
    Tm_train: np.ndarray,
    X1_test: torch.Tensor,
    X2_test: torch.Tensor,
    T1_test: torch.Tensor,
    T2_test: torch.Tensor,
    r_test: torch.Tensor,
    Tm_test: np.ndarray,
    emb_dim_in: int,
    cfg: Dict[str, Any],
    out_dir: str,
    device: torch.device,
    run_name: str,
) -> TrainResult:
    emb_method = str(cfg.get("embedding", {}).get("method", "")).lower()
    projector_cfg = None
    if emb_method == "rdkit":
        proj = cfg.get("embedding", {}).get("rdkit_projector", None)
        if proj is not None:
            projector_cfg = ProjectorConfig(**proj)

    model = DESPhysicsModel(emb_dim_in=int(emb_dim_in), projector=projector_cfg).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    epochs = int(cfg["training"]["epochs"])
    grad_clip = float(cfg["training"]["grad_clip"])
    lambda_reg = float(cfg["training"]["lambda_reg"])
    log_every = int(cfg["training"].get("log_every", 1))

    best_test_mae = float("inf")
    best_path = os.path.join(out_dir, f"{run_name}_best.pt")
    history = {"train_loss": [], "test_loss": [], "train_mae": [], "test_mae": []}

    Tm_train_np = np.asarray(Tm_train, dtype=np.float32)
    Tm_test_np = np.asarray(Tm_test, dtype=np.float32)

    Tm_train_t = torch.tensor(Tm_train_np, device=device)
    Tm_test_t = torch.tensor(Tm_test_np, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss = model.physics_loss(
            X1_train,
            X2_train,
            T1_train,
            T2_train,
            r_train,
            Tm_train_t,
            lambda_reg=lambda_reg,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_loss = model.physics_loss(
                    X1_test,
                    X2_test,
                    T1_test,
                    T2_test,
                    r_test,
                    Tm_test_t,
                    lambda_reg=lambda_reg,
                )

            train_mae = _mae(model, X1_train, X2_train, T1_train, T2_train, r_train, Tm_train_np)
            test_mae = _mae(model, X1_test, X2_test, T1_test, T2_test, r_test, Tm_test_np)

            history["train_loss"].append(float(loss.item()))
            history["test_loss"].append(float(test_loss.item()))
            history["train_mae"].append(train_mae)
            history["test_mae"].append(test_mae)

            if test_mae < best_test_mae:
                best_test_mae = test_mae
                payload = {
                    "model_state": model.state_dict(),
                    "emb_dim_in": int(emb_dim_in),
                    "cfg": cfg,
                }
                if projector_cfg is not None:
                    payload["projector_cfg"] = projector_cfg.__dict__
                torch.save(payload, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_train_mae = _mae(model, X1_train, X2_train, T1_train, T2_train, r_train, Tm_train_np)
    final_test_mae = _mae(model, X1_test, X2_test, T1_test, T2_test, r_test, Tm_test_np)

    return TrainResult(train_mae=final_train_mae, test_mae=final_test_mae, history=history, best_ckpt_path=best_path)
