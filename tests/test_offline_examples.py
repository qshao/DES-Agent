from pathlib import Path


def test_offline_artifacts_exist():
    assert Path("des_multi_agent/artifacts/manifest.yaml").exists()
    assert Path("artifacts/designsolvents/viscosity/model.json").exists()
    assert Path("artifacts/stability_constants/model.json").exists()
