from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from des_multi_agent import prediction


def test_ratio_grid_covers_requested_range():
    grid = prediction.build_ratio_grid()
    assert grid[0] == 0.1
    assert grid[-1] == 0.9
    assert len(grid) >= 9


@dataclass
class _DummyBundle:
    kind: str = "morgan"
    embedder: object | None = None
    gnn_wrapper: object | None = None
    dim: int = 2


class _DummyEmbedder:
    dim = 2

    def embed(self, smiles):
        return [[1.0, 2.0] for _ in smiles]


class _DummyModel:
    def forward_params(self, x1, x2):
        return (
            torch.tensor([300.0], dtype=torch.float32),
            torch.tensor([320.0], dtype=torch.float32),
            torch.tensor([10.0], dtype=torch.float32),
        )


def test_predict_curve_uses_backend_adapter(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.write_text("placeholder", encoding="utf-8")
    cfg_path.write_text(
        """
device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(prediction, "_load_embedder", lambda cfg, device: _DummyBundle(embedder=_DummyEmbedder()))
    monkeypatch.setattr(prediction, "load_model", lambda ckpt_path, device: _DummyModel())

    curve = prediction.predict_curve(
        "CCO",
        "O",
        t1_k=298.15,
        t2_k=273.15,
        checkpoint_path=str(ckpt_path),
        config_path=str(cfg_path),
    )
    assert len(curve.ratios) == len(grid := prediction.build_ratio_grid())
    assert curve.tm_pred_k and len(curve.tm_pred_k) == len(grid)
    assert Path(curve.checkpoint_path) == ckpt_path
