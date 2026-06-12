from pathlib import Path

import pytest

from des_multi_agent.doctor import DoctorIssue, DoctorResult, format_doctor_report, run_doctor
from des_multi_agent.summary import build_command_summary, render_command_summary
from des_multi_agent import user_config as user_config_module


def _write_example_folder(root: Path, folder_name: str) -> None:
    folder = root / "examples" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "input.txt", "output.txt", "run.sh"):
        (folder / name).write_text("ok\n", encoding="utf-8")


def _write_benchmark_folder(root: Path, folder_name: str) -> None:
    folder = root / "tests" / "fixtures" / "example_benchmark_baselines" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "input.txt", "output.txt"):
        (folder / name).write_text("ok\n", encoding="utf-8")


def _write_healthy_repo(root: Path) -> None:
    (root / "ml_des_mp").mkdir()
    (root / "ml_des_mp" / "config.yaml").write_text("config: true\n", encoding="utf-8")
    (root / "llm.example.yaml").write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    (root / "README.md").write_text("example benchmark suite doctor\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "tutorial.md").write_text("doctor\n", encoding="utf-8")
    (root / "examples").mkdir()
    (root / "examples" / "README.md").write_text("doctor\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_benchmarks_examples.py").write_text("ok\n", encoding="utf-8")
    for folder_name in (
        "des_viscosity",
        "viscosity_template",
        "metal_binding",
        "ligand_binding_template",
        "gemma4_12b",
        "nemotron_3_nano",
        "qwen3_6",
        "lidocaine_gemma4_12b",
        "plain_language_gemma4_12b",
        "plain_language_metal_binding_gemma4_12b",
        "task_router",
        "des_run_memory_feedback",
    ):
        _write_example_folder(root, folder_name)
    for folder_name in (
        "des_viscosity",
        "gemma4_12b",
        "lidocaine_gemma4_12b",
        "ligand_binding_template",
        "metal_binding",
        "nemotron_3_nano",
        "plain_language_gemma4_12b",
        "plain_language_metal_binding_gemma4_12b",
        "qwen3_6",
        "task_router",
        "viscosity_template",
    ):
        _write_benchmark_folder(root, folder_name)


def test_doctor_reports_ok_when_repository_is_healthy(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_healthy_repo(repo_root)

    result = run_doctor(repo_root)

    assert result.errors == []
    assert result.warnings == []
    assert result.exit_code == 0
    assert "doctor: ok" in format_doctor_report(result)


def test_doctor_optional_checks_warn_when_local_paths_are_missing(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_healthy_repo(repo_root)

    result = run_doctor(repo_root, optional_checks=("checkpoint", "discovery", "artifacts"))

    assert result.errors == []
    assert result.warnings
    assert any("checkpoint" in issue.message for issue in result.warnings)
    assert any("discovery" in issue.message for issue in result.warnings)
    assert any("artifacts" in issue.message for issue in result.warnings)


def test_doctor_reports_missing_core_files(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    result = run_doctor(repo_root)

    assert result.errors
    assert any("ml_des_mp/config.yaml" in issue.message for issue in result.errors)
    assert any("examples/README.md" in issue.message for issue in result.errors)
    assert result.warnings
    assert result.exit_code == 1


def test_doctor_result_groups_errors_and_warnings():
    result = DoctorResult(
        errors=[DoctorIssue(severity="error", message="missing required file: ml_des_mp/config.yaml")],
        warnings=[DoctorIssue(severity="warning", message="README.md does not mention doctor")],
    )

    text = format_doctor_report(result)

    assert text.startswith("doctor: issues found")
    assert "errors:" in text
    assert "warnings:" in text


def test_doctor_deduplicates_optional_checks(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_healthy_repo(repo_root)

    result = run_doctor(repo_root, optional_checks=("checkpoint", "checkpoint"))

    assert len([issue for issue in result.warnings if "checkpoint" in issue.message]) == 1


def test_doctor_rejects_unsupported_optional_check(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_healthy_repo(repo_root)

    with pytest.raises(ValueError, match="unsupported optional doctor check"):
        run_doctor(repo_root, optional_checks=("bogus",))


def test_doctor_optional_checks_are_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples = Path("examples/README.md").read_text(encoding="utf-8")

    assert "doctor --check checkpoint" in readme
    assert "doctor --check discovery" in tutorial
    assert "doctor --check artifacts" in examples
    assert "doctor --check config" in readme
    assert "doctor --check llm" in examples


def test_standardized_run_directory_is_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples = Path("examples/README.md").read_text(encoding="utf-8")

    assert "--output-dir runs/run_001" in readme
    assert "report.txt" in tutorial
    assert "run.json" in examples


def test_doctor_summary_mentions_error_and_warning_counts():
    result = DoctorResult(
        errors=[DoctorIssue(severity="error", message="missing required file: ml_des_mp/config.yaml")],
        warnings=[DoctorIssue(severity="warning", message="README.md does not mention doctor")],
    )

    text = render_command_summary(build_command_summary("doctor", result))

    assert "summary:" in text
    assert "status: issues found" in text
    assert "errors: 1" in text
    assert "warnings: 1" in text


def test_doctor_config_check_warns_for_malformed_user_config(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_healthy_repo(repo_root)
    config_path = tmp_path / "bad-config.yaml"
    config_path.write_text("checkpoint_path: [", encoding="utf-8")
    monkeypatch.setattr(user_config_module, "get_user_config_path", lambda: config_path)

    result = run_doctor(repo_root, optional_checks=("config",))

    assert result.errors == []
    assert any("invalid YAML" in issue.message for issue in result.warnings)


def test_doctor_config_check_warns_for_unknown_keys_and_missing_paths(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_healthy_repo(repo_root)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("checkpoint_path: missing.pt\nunknown_key: true\n", encoding="utf-8")
    monkeypatch.setattr(user_config_module, "get_user_config_path", lambda: config_path)

    result = run_doctor(repo_root, optional_checks=("config",))

    messages = "\n".join(issue.message for issue in result.warnings)
    assert "unknown key" in messages
    assert "checkpoint_path path does not exist" in messages
