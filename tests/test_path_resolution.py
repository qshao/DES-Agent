from pathlib import Path

import torch

from des_multi_agent import prediction
from des_multi_agent.paths import resolve_existing_path


class _DummyBundle:
    kind = "morgan"

    def __init__(self):
        self.embedder = self
        self.gnn_wrapper = None
        self.dim = 2

    def embed(self, smiles):
        return [[1.0, 2.0] for _ in smiles]


class _DummyModel:
    def forward_params(self, x1, x2):
        return (
            torch.tensor([300.0], dtype=torch.float32),
            torch.tensor([320.0], dtype=torch.float32),
            torch.tensor([10.0], dtype=torch.float32),
        )


def test_resolve_existing_path_finds_repo_relative_config_from_any_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_existing_path("ml_des_mp/config.yaml")
    assert resolved.name == "config.yaml"
    assert resolved.parent.name == "ml_des_mp"


def test_predict_curve_resolves_paths_from_any_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(prediction, "get_device", lambda device_str: torch.device("cpu"))
    monkeypatch.setattr(prediction, "_load_embedder", lambda cfg, device: _DummyBundle())
    monkeypatch.setattr(prediction, "load_model", lambda ckpt_path, device: _DummyModel())

    curve = prediction.predict_curve(
        "CCO",
        "O",
        t1_k=298.15,
        t2_k=273.15,
        checkpoint_path=str(ckpt_path),
        config_path="ml_des_mp/config.yaml",
    )
    assert Path(curve.checkpoint_path) == ckpt_path.resolve()
    assert curve.ratios[0] == 0.1
    assert len(curve.tm_pred_k) == len(curve.ratios)


def test_predict_cli_resolves_checkpoint_relative_to_config_dir(monkeypatch, tmp_path):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
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
    ckpt_path = config_dir / "model.pt"
    ckpt_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prediction, "get_device", lambda device_str: torch.device("cpu"))
    monkeypatch.setattr(prediction, "_load_embedder", lambda cfg, device: _DummyBundle())

    captured = {}

    def fake_load_model(path, device):
        captured["path"] = Path(path)
        return _DummyModel()

    monkeypatch.setattr(prediction, "load_model", fake_load_model)

    curve = prediction.predict_curve(
        "CCO",
        "O",
        t1_k=298.15,
        t2_k=273.15,
        checkpoint_path="model.pt",
        config_path=str(config_path),
    )
    assert captured["path"] == ckpt_path.resolve()
    assert Path(curve.checkpoint_path) == ckpt_path.resolve()
