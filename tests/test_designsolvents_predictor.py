from pathlib import Path

from des_multi_agent.predictors.designsolvents import predict_viscosity


def test_predict_viscosity_uses_local_artifact(tmp_path: Path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        """
{
  "model_name": "DESignSolvents",
  "units": "mPa*s",
  "bias": 8.0,
  "coefficients": {
    "total_heavy_atoms": 0.1,
    "total_hbd": 0.5,
    "total_hba": 0.25,
    "total_logp": -0.2,
    "total_rings": 0.4
  }
}
""".strip(),
        encoding="utf-8",
    )
    result = predict_viscosity("CCO", "OCCO", model_path=model_path, allow_fallback=False)
    assert result.units == "mPa*s"
    assert result.model_name == "DESignSolvents"
    assert result.source == "artifact"
    assert result.metadata["component_a"] == "CCO"
    assert result.metadata["component_b"] == "OCCO"
    assert result.value > 0.0


def test_predict_viscosity_uses_bundled_artifact_by_default():
    result = predict_viscosity("CCO", "OCCO", model_path=None, allow_fallback=True)
    assert result.source == "artifact"
    assert result.value > 0.0
