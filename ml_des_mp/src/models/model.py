from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from .physics_core import PhysicsSiameseNet, m_physics_loss


# -----------------------------------------------------------------------------
# Default model (projector optional)
# -----------------------------------------------------------------------------

class DESPhysicsModel(nn.Module):
    """End-to-end model: (fixed) embeddings -> (optional) projector -> PhysicsSiameseNet.

    The projector is optional and should generally be avoided.
    However, RDKit descriptors often benefit from a small projector because raw
    descriptor scales vary widely.
    """

    def __init__(self, emb_dim_in: int, projector: "ProjectorConfig | None" = None):
        super().__init__()
        self.emb_dim_in = int(emb_dim_in)

        if projector is None:
            self.projector: nn.Module = nn.Identity()
            emb_dim = self.emb_dim_in
            self.projector_cfg = None
        else:
            self.projector = _make_mlp(
                int(emb_dim_in),
                list(projector.hidden_sizes),
                int(projector.out_dim),
                float(projector.dropout),
                str(projector.activation),
            )
            emb_dim = int(projector.out_dim)
            self.projector_cfg = projector

        self.physics = PhysicsSiameseNet(emb_dim=emb_dim)

    def forward_params(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.projector(x1)
        z2 = self.projector(x2)
        # PhysicsSiameseNet is kept unchanged; pairing/symmetry is handled inside it.
        return self.physics(z1, z2)

    def physics_loss(self, x1, x2, T1, T2, r, Tm, lambda_reg: float) -> torch.Tensor:
        d1, d2, W = self.forward_params(x1, x2)
        loss = m_physics_loss(d1, d2, W, T1, T2, r, Tm)
        # Keep the same anchor penalty idea as original training loop
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


class DESPhysicsModelWithProjector(nn.Module):
    """Legacy model: embeddings -> projector -> PhysicsSiameseNet.

    Kept only to load old checkpoints trained before projector removal.
    """

    def __init__(self, emb_dim_in: int, projector: ProjectorConfig):
        super().__init__()
        self.projector = _make_mlp(
            int(emb_dim_in),
            list(projector.hidden_sizes),
            int(projector.out_dim),
            float(projector.dropout),
            str(projector.activation),
        )
        self.physics = PhysicsSiameseNet(emb_dim=int(projector.out_dim))

    def forward_params(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.projector(x1)
        z2 = self.projector(x2)
        return self.physics(z1, z2)

    def physics_loss(self, x1, x2, T1, T2, r, Tm, lambda_reg: float) -> torch.Tensor:
        d1, d2, W = self.forward_params(x1, x2)
        loss = m_physics_loss(d1, d2, W, T1, T2, r, Tm)
        anchor_penalty = float(lambda_reg) * torch.mean(W**2)
        return loss + anchor_penalty
