from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from ..embeddings.gnn import GNNConfig, MoleculeGNN
from .physics_core import PhysicsSiameseNet, m_physics_loss


# -----------------------------------------------------------------------------
# New default GNN model (NO projector)
# -----------------------------------------------------------------------------

class DESPhysicsGNNModel(nn.Module):
    """End-to-end: shared molecule GNN -> PhysicsSiameseNet.

    Projector removed: PhysicsSiameseNet consumes the GNN embeddings directly.
    """

    def __init__(self, gnn_cfg: GNNConfig):
        super().__init__()
        self.gnn_cfg = gnn_cfg
        self.gnn = MoleculeGNN(gnn_cfg)
        self.physics = PhysicsSiameseNet(emb_dim=int(gnn_cfg.emb_dim))

    def forward_params_batches(self, b1: Batch, b2: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.gnn(b1)
        e2 = self.gnn(b2)
        return self.physics(e1, e2)

    def physics_loss_batches(
        self,
        b1: Batch,
        b2: Batch,
        T1: torch.Tensor,
        T2: torch.Tensor,
        r: torch.Tensor,
        Tm: torch.Tensor,
        lambda_reg: float,
    ) -> torch.Tensor:
        d1, d2, W = self.forward_params_batches(b1, b2)
        loss = m_physics_loss(d1, d2, W, T1, T2, r, Tm)
        anchor_penalty = float(lambda_reg) * torch.mean(W**2)
        return loss + anchor_penalty


# -----------------------------------------------------------------------------
# Legacy compatibility (old checkpoints that included a projector)
# -----------------------------------------------------------------------------

def _make_mlp(in_dim: int, hidden_sizes: List[int], out_dim: int, dropout: float, activation: str) -> nn.Sequential:
    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    Act = acts.get(str(activation).lower(), nn.ReLU)
    layers: List[nn.Module] = []
    d = int(in_dim)
    for h in list(hidden_sizes):
        layers += [nn.Linear(d, int(h)), Act(), nn.Dropout(float(dropout))]
        d = int(h)
    layers += [nn.Linear(d, int(out_dim))]
    return nn.Sequential(*layers)


@dataclass
class ProjectorConfig:
    out_dim: int
    hidden_sizes: List[int]
    dropout: float
    activation: str = "relu"


class DESPhysicsGNNModelWithProjector(nn.Module):
    """Legacy: GNN -> projector -> PhysicsSiameseNet (for old checkpoints)."""

    def __init__(self, gnn_cfg: GNNConfig, projector: ProjectorConfig):
        super().__init__()
        self.gnn_cfg = gnn_cfg
        self.gnn = MoleculeGNN(gnn_cfg)
        self.projector = _make_mlp(
            int(gnn_cfg.emb_dim),
            list(projector.hidden_sizes),
            int(projector.out_dim),
            float(projector.dropout),
            str(projector.activation),
        )
        self.physics = PhysicsSiameseNet(emb_dim=int(projector.out_dim))

    def forward_params_batches(self, b1: Batch, b2: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.gnn(b1)
        e2 = self.gnn(b2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)
        return self.physics(z1, z2)

    def physics_loss_batches(
        self,
        b1: Batch,
        b2: Batch,
        T1: torch.Tensor,
        T2: torch.Tensor,
        r: torch.Tensor,
        Tm: torch.Tensor,
        lambda_reg: float,
    ) -> torch.Tensor:
        d1, d2, W = self.forward_params_batches(b1, b2)
        loss = m_physics_loss(d1, d2, W, T1, T2, r, Tm)
        anchor_penalty = float(lambda_reg) * torch.mean(W**2)
        return loss + anchor_penalty
