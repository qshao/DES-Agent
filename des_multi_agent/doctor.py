from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT


EXAMPLE_FOLDERS = (
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
)

BENCHMARK_FOLDERS = (
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
)


@dataclass(frozen=True)
class DoctorIssue:
    severity: str
    message: str


@dataclass(frozen=True)
class DoctorResult:
    errors: list[DoctorIssue]
    warnings: list[DoctorIssue]

    @property
    def exit_code(self) -> int:
        return 1 if self.errors else 0


def _format_issue_section(label: str, issues: list[DoctorIssue]) -> list[str]:
    if not issues:
        return []
    lines = [f"{label}:"]
    for issue in issues:
        lines.append(f"- {issue.message}")
    return lines


def format_doctor_report(result: DoctorResult) -> str:
    lines = ["doctor: ok" if not result.errors else "doctor: issues found"]
    lines.extend(_format_issue_section("errors", result.errors))
    lines.extend(_format_issue_section("warnings", result.warnings))
    return "\n".join(lines)


def _add_issue(collection: list[DoctorIssue], severity: str, message: str) -> None:
    collection.append(DoctorIssue(severity=severity, message=message))


def _check_file_exists(root: Path, relative_path: str, errors: list[DoctorIssue]) -> None:
    path = root / relative_path
    if not path.exists():
        _add_issue(errors, "error", f"missing required file: {relative_path}")


def _check_text_contains(root: Path, relative_path: str, needle: str, warnings: list[DoctorIssue]) -> None:
    path = root / relative_path
    if not path.exists():
        _add_issue(warnings, "warning", f"cannot inspect missing file: {relative_path}")
        return
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        _add_issue(warnings, "warning", f"{relative_path} does not mention {needle}")


def _check_example_folder(root: Path, folder_name: str, errors: list[DoctorIssue]) -> None:
    folder = root / "examples" / folder_name
    if not folder.exists():
        _add_issue(errors, "error", f"missing example folder: examples/{folder_name}")
        return
    for required_name in ("README.md", "input.txt", "output.txt", "run.sh"):
        if not (folder / required_name).exists():
            _add_issue(errors, "error", f"missing example artifact: examples/{folder_name}/{required_name}")


def _check_benchmark_baselines(root: Path, errors: list[DoctorIssue]) -> None:
    baseline_root = root / "tests" / "fixtures" / "example_benchmark_baselines"
    if not baseline_root.exists():
        _add_issue(errors, "error", "missing benchmark baseline directory: tests/fixtures/example_benchmark_baselines")
        return
    for folder_name in BENCHMARK_FOLDERS:
        for required_name in ("README.md", "input.txt", "output.txt"):
            if not (baseline_root / folder_name / required_name).exists():
                _add_issue(errors, "error", f"missing benchmark baseline file: tests/fixtures/example_benchmark_baselines/{folder_name}/{required_name}")


def run_doctor(repo_root: str | Path = PROJECT_ROOT) -> DoctorResult:
    root = Path(repo_root)
    errors: list[DoctorIssue] = []
    warnings: list[DoctorIssue] = []

    _check_file_exists(root, "ml_des_mp/config.yaml", errors)
    _check_file_exists(root, "llm.example.yaml", errors)
    _check_file_exists(root, "README.md", errors)
    _check_file_exists(root, "docs/tutorial.md", errors)
    _check_file_exists(root, "examples/README.md", errors)
    _check_file_exists(root, "tests/test_benchmarks_examples.py", errors)
    _check_benchmark_baselines(root, errors)

    for folder_name in EXAMPLE_FOLDERS:
        _check_example_folder(root, folder_name, errors)

    # Lightweight doc consistency checks.
    _check_text_contains(root, "README.md", "example benchmark suite", warnings)
    _check_text_contains(root, "README.md", "doctor", warnings)
    _check_text_contains(root, "docs/tutorial.md", "doctor", warnings)
    _check_text_contains(root, "examples/README.md", "doctor", warnings)

    return DoctorResult(errors=errors, warnings=warnings)
