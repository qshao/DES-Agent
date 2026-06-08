from pathlib import Path

from des_multi_agent.doctor import DoctorIssue, DoctorResult, format_doctor_report, run_doctor


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


def test_doctor_reports_ok_when_repository_is_healthy(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "ml_des_mp").mkdir()
    (repo_root / "ml_des_mp" / "config.yaml").write_text("config: true\n", encoding="utf-8")
    (repo_root / "llm.example.yaml").write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    (repo_root / "README.md").write_text("example benchmark suite doctor\n", encoding="utf-8")
    (repo_root / "docs").mkdir()
    (repo_root / "docs" / "tutorial.md").write_text("doctor\n", encoding="utf-8")
    (repo_root / "examples").mkdir()
    (repo_root / "examples" / "README.md").write_text("doctor\n", encoding="utf-8")
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_benchmarks_examples.py").write_text("ok\n", encoding="utf-8")
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
        _write_example_folder(repo_root, folder_name)
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
        _write_benchmark_folder(repo_root, folder_name)

    result = run_doctor(repo_root)

    assert result.errors == []
    assert result.warnings == []
    assert result.exit_code == 0
    assert "doctor: ok" in format_doctor_report(result)


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
