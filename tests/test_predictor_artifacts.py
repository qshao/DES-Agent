from pathlib import Path

from des_multi_agent.predictors.artifacts import default_artifact_root, load_manifest, require_artifact, resolve_artifact


def test_load_manifest_and_require_artifact(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    artifact_root = tmp_path / "artifacts"
    model_path = artifact_root / "designsolvents" / "viscosity" / "model.json"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("{}", encoding="utf-8")
    manifest.write_text(
        """
workflows:
  des_viscosity:
    artifacts:
      model: designsolvents/viscosity/model.json
""".strip(),
        encoding="utf-8",
    )
    data = load_manifest(manifest)
    assert data["workflows"]["des_viscosity"]["artifacts"]["model"] == "designsolvents/viscosity/model.json"
    assert require_artifact(artifact_root, "designsolvents/viscosity/model.json") == model_path


def test_resolve_artifact_uses_manifest(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    model_path = artifact_root / "stability_constants" / "model.json"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
workflows:
  metal_binding:
    artifacts:
      model: stability_constants/model.json
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("des_multi_agent.predictors.artifacts.MANIFEST_PATH", manifest)
    monkeypatch.setattr("des_multi_agent.predictors.artifacts.DEFAULT_ARTIFACT_ROOT", artifact_root)
    assert resolve_artifact(None, "metal_binding") == model_path


def test_resolve_artifact_rejects_missing_explicit_path(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    model_path = artifact_root / "stability_constants" / "model.json"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
workflows:
  metal_binding:
    artifacts:
      model: stability_constants/model.json
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("des_multi_agent.predictors.artifacts.MANIFEST_PATH", manifest)
    monkeypatch.setattr("des_multi_agent.predictors.artifacts.DEFAULT_ARTIFACT_ROOT", artifact_root)
    missing = tmp_path / "missing.json"
    try:
        resolve_artifact(missing, "metal_binding")
    except FileNotFoundError as exc:
        assert "Missing explicit local artifact" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for missing explicit artifact")
