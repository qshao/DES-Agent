"""ChemBERTa-embedding QSPR model for pure-component melting points.

Defines the regression-head architecture (shared by training and inference) and
a loader that runs a deep ensemble to produce a melting point plus an epistemic
uncertainty estimate. The artifact is optional: if it is missing the loader
returns ``None`` so ``property_resolution`` falls back to the heuristic layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# Default artifact location, parallel to the experimental lookup table.
QSPR_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "melting_points" / "qspr_model.pt"
)


class MPRegressor(nn.Module):
    """Small MLP head mapping a molecular embedding to a (standardized) Tm."""

    def __init__(self, emb_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class QSPRPrediction:
    tm_k: float
    std_k: float
    ci_k: float | None = None  # conformal-calibrated half-width (e.g. 90% interval)


class MeltingPointQSPR:
    """Loaded deep ensemble + ChemBERTa embedder for melting-point inference."""

    def __init__(self, members, feat_mean, feat_std, tm_mean, tm_std, embedder, device,
                 std_scale_k=None, conformal_q=None):
        self._members = members
        self._feat_mean = feat_mean
        self._feat_std = feat_std
        self._tm_mean = float(tm_mean)
        self._tm_std = float(tm_std)
        self._embedder = embedder
        self._device = device
        # data-calibrated confidence scale + conformal interval multiplier
        self.std_scale_k = float(std_scale_k) if std_scale_k else None
        self.conformal_q = float(conformal_q) if conformal_q else None

    @torch.no_grad()
    def predict(self, smiles: str) -> QSPRPrediction:
        emb = torch.tensor(self._embedder.embed([smiles]), device=self._device, dtype=torch.float32)
        x = (emb - self._feat_mean) / self._feat_std
        preds = torch.stack([m(x) for m in self._members])  # (n_members, 1)
        preds_k = preds * self._tm_std + self._tm_mean
        if len(self._members) > 1:
            std_k = float(preds_k.std(unbiased=False).item())
            ci_k = self.conformal_q * std_k if self.conformal_q else None
        else:
            # Single-member ensemble: no inter-model variance available.
            # Use inf so _qspr_confidence() maps to the minimum confidence floor
            # rather than false-maximum certainty (std=0 → conf=0.85).
            std_k = float("inf")
            ci_k = None
        return QSPRPrediction(tm_k=float(preds_k.mean().item()), std_k=std_k, ci_k=ci_k)


def _build_embedder(model_name: str, max_length: int, device: torch.device, cache_dir: str | None = None):
    # Shared factory: reuses the DES embedder instance when the configuration
    # (model, pooling, batch_size, max_length, device, cache_dir) matches.
    from ..embedding_factory import get_chemberta_embedder

    return get_chemberta_embedder(
        model_name=model_name, pooling="mean", batch_size=64,
        max_length=max_length, device=device, cache_dir=cache_dir,
    )


def load_qspr_model(path: str | Path = QSPR_MODEL_PATH, device: str = "cpu") -> MeltingPointQSPR | None:
    import warnings as _warnings

    path = Path(path)
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        dev = torch.device(device)
        emb_dim = int(ckpt["emb_dim"])
        members = []
        for state in ckpt["member_states"]:
            m = MPRegressor(emb_dim, hidden=int(ckpt["hidden"]), dropout=float(ckpt["dropout"]))
            m.load_state_dict(state)
            m.to(dev).eval()
            members.append(m)
        embedder = _build_embedder(
            ckpt["embedding_model"], int(ckpt["max_length"]), dev,
            cache_dir=ckpt.get("emb_cache_dir", ".cache/chemberta_emb"),
        )
        calib = ckpt.get("calibration", {})
        return MeltingPointQSPR(
            members=members,
            feat_mean=torch.tensor(ckpt["feat_mean"], device=dev, dtype=torch.float32),
            feat_std=torch.tensor(ckpt["feat_std"], device=dev, dtype=torch.float32),
            tm_mean=ckpt["tm_mean"],
            tm_std=ckpt["tm_std"],
            embedder=embedder,
            device=dev,
            std_scale_k=calib.get("std_scale_k"),
            conformal_q=calib.get("conformal_q"),
        )
    except Exception as exc:
        _warnings.warn(
            f"QSPR checkpoint at {path} exists but failed to load ({exc}); "
            "falling back to heuristic melting-point estimation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
