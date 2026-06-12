"""Tests for the prediction-path performance work: load-once caching,
vectorized ratio-curve evaluation, and a device override for the DES ML stage.

These pin behavior that must stay numerically identical to the original
per-candidate reload path while eliminating redundant model/embedder loads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from des_multi_agent import prediction


@dataclass
class _DummyBundle:
    kind: str = "morgan"
    embedder: object | None = None
    gnn_wrapper: object | None = None
    dim: int = 2


class _DummyEmbedder:
    dim = 2

    def __init__(self):
        self.embed_calls: list[str] = []

    def embed(self, smiles):
        self.embed_calls.extend(smiles)
        return [[1.0, 2.0] for _ in smiles]


class _DummyModel:
    """Returns fixed physics params so the ratio curve is deterministic."""

    d1, d2, w = 4000.0, 4200.0, 1500.0

    def forward_params(self, x1, x2):
        return (
            torch.tensor([self.d1], dtype=torch.float32),
            torch.tensor([self.d2], dtype=torch.float32),
            torch.tensor([self.w], dtype=torch.float32),
        )


@pytest.fixture(autouse=True)
def _clear_caches():
    prediction.clear_prediction_caches()
    yield
    prediction.clear_prediction_caches()


def _write_cfg(tmp_path, device="cpu"):
    cfg_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.write_text("placeholder", encoding="utf-8")
    cfg_path.write_text(
        f"""
device: {device}
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )
    return cfg_path, ckpt_path


def test_model_and_embedder_and_compat_loaded_once_across_calls(monkeypatch, tmp_path):
    cfg_path, ckpt_path = _write_cfg(tmp_path)
    counts = {"model": 0, "embedder": 0, "compat": 0}

    def fake_load_model(ckpt, device):
        counts["model"] += 1
        return _DummyModel()

    def fake_load_embedder(cfg, device):
        counts["embedder"] += 1
        return _DummyBundle(embedder=_DummyEmbedder())

    def fake_compat(ckpt, cfg):
        counts["compat"] += 1
        return []

    monkeypatch.setattr(prediction, "load_model", fake_load_model)
    monkeypatch.setattr(prediction, "_load_embedder", fake_load_embedder)
    monkeypatch.setattr(prediction, "check_checkpoint_config_compat", fake_compat)

    for b in ("O", "CCO", "CCN"):
        prediction.predict_curve(
            "CCO", b, t1_k=298.15, t2_k=273.15,
            checkpoint_path=str(ckpt_path), config_path=str(cfg_path),
        )

    assert counts == {"model": 1, "embedder": 1, "compat": 1}


def test_ratio_curve_matches_reference_formula(monkeypatch, tmp_path):
    cfg_path, ckpt_path = _write_cfg(tmp_path)
    monkeypatch.setattr(prediction, "_load_embedder", lambda cfg, device: _DummyBundle(embedder=_DummyEmbedder()))
    monkeypatch.setattr(prediction, "load_model", lambda ckpt, device: _DummyModel())
    monkeypatch.setattr(prediction, "check_checkpoint_config_compat", lambda ckpt, cfg: [])

    t1, t2 = 298.15, 273.15
    curve = prediction.predict_curve(
        "CCO", "O", t1_k=t1, t2_k=t2,
        checkpoint_path=str(ckpt_path), config_path=str(cfg_path),
    )

    R = 8.314
    d1, d2, w = _DummyModel.d1, _DummyModel.d2, _DummyModel.w
    eps = 1e-8
    t_ref = (t1 + t2) / 2.0
    for ratio, got in zip(curve.ratios, curve.tm_pred_k):
        r = min(max(ratio, eps), 1.0 - eps)
        ln_a1 = math.log(r) + (w / (R * t_ref)) * (1 - r) ** 2
        ln_a2 = math.log(1 - r) + (w / (R * t_ref)) * r ** 2
        denom1 = max(1.0 - (R / d1) * ln_a1, 0.1)
        denom2 = max(1.0 - (R / d2) * ln_a2, 0.1)
        expected = max(t1 / denom1, t2 / denom2)
        assert got == pytest.approx(expected, rel=1e-5)


def test_constant_component_a_embedded_once_across_candidates(monkeypatch, tmp_path):
    cfg_path, ckpt_path = _write_cfg(tmp_path)
    embedder = _DummyEmbedder()
    monkeypatch.setattr(prediction, "_load_embedder", lambda cfg, device: _DummyBundle(embedder=embedder))
    monkeypatch.setattr(prediction, "load_model", lambda ckpt, device: _DummyModel())
    monkeypatch.setattr(prediction, "check_checkpoint_config_compat", lambda ckpt, cfg: [])

    for b in ("O", "CCN", "CCC"):
        prediction.predict_curve(
            "CCO", b, t1_k=298.15, t2_k=273.15,
            checkpoint_path=str(ckpt_path), config_path=str(cfg_path),
        )

    # component_a "CCO" embedded exactly once; each distinct B embedded once
    assert embedder.embed_calls.count("CCO") == 1
    assert sorted(embedder.embed_calls) == ["CCC", "CCN", "CCO", "O"]


def test_des_device_override_forces_cpu(monkeypatch, tmp_path):
    # config requests cuda, but the override must win regardless of GPU presence
    cfg_path, ckpt_path = _write_cfg(tmp_path, device="cuda")
    monkeypatch.setenv("DES_ML_DEVICE", "cpu")
    seen = {}

    def fake_load_model(ckpt, device):
        seen["model_device"] = device
        return _DummyModel()

    def fake_load_embedder(cfg, device):
        seen["embedder_device"] = device
        return _DummyBundle(embedder=_DummyEmbedder())

    monkeypatch.setattr(prediction, "load_model", fake_load_model)
    monkeypatch.setattr(prediction, "_load_embedder", fake_load_embedder)
    monkeypatch.setattr(prediction, "check_checkpoint_config_compat", lambda ckpt, cfg: [])

    prediction.predict_curve(
        "CCO", "O", t1_k=298.15, t2_k=273.15,
        checkpoint_path=str(ckpt_path), config_path=str(cfg_path),
    )

    assert seen["model_device"] == torch.device("cpu")
    assert seen["embedder_device"] == torch.device("cpu")
