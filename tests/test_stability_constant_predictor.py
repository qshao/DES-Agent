from pathlib import Path

from des_multi_agent.predictors.stability_constants import predict_log_k


def test_predict_log_k_uses_local_artifact(tmp_path: Path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        """
{
  "model_name": "stabilityconstant-ml-models",
  "units": "log K",
  "bias": 5.0,
  "coefficients": {
    "ligand_hbd": 0.3,
    "ligand_hba": 0.35,
    "ligand_tpsa": 0.04,
    "ligand_rings": 0.15,
    "abs_metal_charge": 0.4
  }
}
""".strip(),
        encoding="utf-8",
    )
    result = predict_log_k("Cu2+", "NCCN", model_path=model_path, allow_fallback=False)
    assert result.units == "log K"
    assert result.model_name == "stabilityconstant-ml-models"
    assert result.source == "artifact"
    assert result.metadata["metal_ion"] == "Cu2+"
    assert result.metadata["ligand"] == "NCCN"
    assert isinstance(result.value, float)


def test_predict_log_k_uses_bundled_artifact_by_default():
    result = predict_log_k("Cu2+", "NCCN", model_path=None, allow_fallback=True)
    assert result.source == "artifact"
    assert isinstance(result.value, float)
