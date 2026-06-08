# Doctor Optional Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing `doctor` command with opt-in local setup checks for checkpoint, discovery, and artifact paths, while keeping optional failures as warnings and preserving the default behavior.

**Architecture:** Keep the current doctor pipeline in `des_multi_agent/doctor.py` and add a small optional-check dispatch layer that runs only when `doctor --check ...` is used. Parse repeated optional checks in the CLI, deduplicate them, and pass them through to the doctor runner. Keep all checks local, read-only, and deterministic; the default `doctor` behavior must remain unchanged.

**Tech Stack:** Python 3.13, `argparse`, `pathlib`, `pytest`, existing `des_multi_agent.cli` and repo doc conventions.

---

### Task 1: Add optional doctor check dispatch in the doctor module

**Files:**
- Modify: `des_multi_agent/doctor.py`
- Test: `tests/test_doctor.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.doctor import run_doctor


def test_doctor_optional_checks_warn_when_local_paths_are_missing(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "ml_des_mp").mkdir()
    (repo_root / "ml_des_mp" / "config.yaml").write_text("config: true\n", encoding="utf-8")
    (repo_root / "llm.example.yaml").write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    (repo_root / "README.md").write_text("doctor example benchmark suite\n", encoding="utf-8")
    (repo_root / "docs").mkdir()
    (repo_root / "docs" / "tutorial.md").write_text("doctor\n", encoding="utf-8")
    (repo_root / "examples").mkdir()
    (repo_root / "examples" / "README.md").write_text("doctor\n", encoding="utf-8")
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_benchmarks_examples.py").write_text("ok\n", encoding="utf-8")

    result = run_doctor(repo_root, optional_checks=("checkpoint", "discovery", "artifacts"))

    assert result.errors == []
    assert result.warnings
    assert any("checkpoint" in issue.message for issue in result.warnings)
    assert any("discovery" in issue.message for issue in result.warnings)
    assert any("artifacts" in issue.message for issue in result.warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```

Expected: FAIL because `run_doctor(..., optional_checks=...)` does not yet exist.

- [ ] **Step 3: Write minimal implementation**

```python
from collections.abc import Sequence


OPTIONAL_CHECKS = ("checkpoint", "discovery", "artifacts")
DEFAULT_CHECKPOINT = "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt"
DEFAULT_DISCOVERY_FIXTURE = "tests/fixtures/discovery"
DEFAULT_ARTIFACTS = (
    "artifacts/README.md",
    "artifacts/designsolvents/viscosity/model.json",
    "artifacts/stability_constants/model.json",
)


def _check_optional_checkpoint(root: Path, warnings: list[DoctorIssue]) -> None:
    _check_file_exists(root, DEFAULT_CHECKPOINT, warnings)


def _check_optional_discovery(root: Path, warnings: list[DoctorIssue]) -> None:
    _check_file_exists(root, f"{DEFAULT_DISCOVERY_FIXTURE}/literature.yaml", warnings)
    _check_file_exists(root, f"{DEFAULT_DISCOVERY_FIXTURE}/library.yaml", warnings)


def _check_optional_artifacts(root: Path, warnings: list[DoctorIssue]) -> None:
    for relative_path in DEFAULT_ARTIFACTS:
        _check_file_exists(root, relative_path, warnings)


def run_doctor(repo_root: str | Path = PROJECT_ROOT, optional_checks: Sequence[str] = ()) -> DoctorResult:
    root = Path(repo_root)
    errors: list[DoctorIssue] = []
    warnings: list[DoctorIssue] = []

    # existing core checks stay unchanged here

    for check_name in dict.fromkeys(optional_checks):
        if check_name == "checkpoint":
            _check_optional_checkpoint(root, warnings)
        elif check_name == "discovery":
            _check_optional_discovery(root, warnings)
        elif check_name == "artifacts":
            _check_optional_artifacts(root, warnings)
        else:
            raise ValueError(f"unsupported optional doctor check: {check_name}")

    return DoctorResult(errors=errors, warnings=warnings)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```

Expected: PASS once optional check dispatch is wired in.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/doctor.py tests/test_doctor.py
git commit -m "feat: add optional doctor checks"
```

### Task 2: Wire `doctor --check` into the CLI

**Files:**
- Modify: `des_multi_agent/cli.py:1-220`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from des_multi_agent.cli import main


def test_doctor_command_accepts_optional_checks(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["doctor", "--check", "checkpoint", "--check", "discovery", "--check", "artifacts"])
    assert exc.value.code in (0, 1)
    out = capsys.readouterr().out
    assert "doctor:" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_cli.py -q
```

Expected: FAIL because `doctor --check` is not yet parsed.

- [ ] **Step 3: Write minimal implementation**

```python
doctor_parser = subparsers.add_parser("doctor", help="Check local repo and example setup")
doctor_parser.add_argument(
    "--check",
    action="append",
    choices=("checkpoint", "discovery", "artifacts"),
    default=[],
    help="Run optional local setup checks; may be passed multiple times",
)
doctor_parser.set_defaults(command="doctor")


if getattr(args, "command", None) == "doctor":
    result = run_doctor(PROJECT_ROOT, optional_checks=args.check)
    print(format_doctor_report(result))
    raise SystemExit(result.exit_code)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_cli.py -q
```

Expected: PASS after CLI parsing is added.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: wire doctor optional checks into cli"
```

### Task 3: Document the optional checks and keep the repo docs aligned

**Files:**
- Modify: `README.md:1-220`
- Modify: `docs/tutorial.md:1-220`
- Modify: `examples/README.md:1-220`
- Test: `tests/test_doctor.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_doctor_optional_checks_are_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples = Path("examples/README.md").read_text(encoding="utf-8")
    assert "doctor --check checkpoint" in readme
    assert "doctor --check discovery" in tutorial
    assert "doctor --check artifacts" in examples
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```

Expected: FAIL until the docs mention the new optional checks.

- [ ] **Step 3: Write minimal implementation**

```markdown
Run `python -m des_multi_agent.cli doctor --check checkpoint --check discovery --check artifacts`
to verify optional local setup paths before a workflow.
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```

Expected: PASS once the docs mention the new optional checks.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md tests/test_doctor.py
git commit -m "docs: describe doctor optional checks"
```

### Task 4: Verify the full doctor path is stable

**Files:**
- Test: `tests/test_doctor.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add or update regression coverage for duplicates and invalid names**

```python
import pytest

from des_multi_agent.doctor import run_doctor


def test_doctor_duplicate_optional_checks_are_deduplicated(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "ml_des_mp").mkdir()
    (repo_root / "ml_des_mp" / "config.yaml").write_text("config: true\n", encoding="utf-8")
    (repo_root / "llm.example.yaml").write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    (repo_root / "README.md").write_text("doctor example benchmark suite\n", encoding="utf-8")
    (repo_root / "docs").mkdir()
    (repo_root / "docs" / "tutorial.md").write_text("doctor\n", encoding="utf-8")
    (repo_root / "examples").mkdir()
    (repo_root / "examples" / "README.md").write_text("doctor\n", encoding="utf-8")
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_benchmarks_examples.py").write_text("ok\n", encoding="utf-8")

    result = run_doctor(repo_root, optional_checks=("checkpoint", "checkpoint"))

    assert len([issue for issue in result.warnings if "checkpoint" in issue.message]) == 1


def test_doctor_rejects_unsupported_optional_check(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError, match="unsupported optional doctor check"):
        run_doctor(repo_root, optional_checks=("bogus",))
```

- [ ] **Step 2: Run the doctor-focused test file**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run:
```bash
python -m pytest -q
```

Expected: PASS with the existing third-party warnings only.

- [ ] **Step 4: Commit**

```bash
git add des_multi_agent/doctor.py des_multi_agent/cli.py tests/test_doctor.py tests/test_cli.py README.md docs/tutorial.md examples/README.md
git commit -m "feat: add doctor optional checks"
```

## Self-Review

**Spec coverage**
- Optional `doctor --check` plumbing: Task 1 and Task 2
- Named optional checks `checkpoint`, `discovery`, `artifacts`: Task 1
- Warning-only behavior for optional failures: Task 1 and Task 4
- Duplicate optional checks deduped: Task 4
- Unsupported check names rejected: Task 2 and Task 4
- Docs and examples updated: Task 3

**Placeholder scan**
- No TBD or TODO placeholders
- No undefined helper names
- No vague “add validation” steps without concrete code snippets

**Type consistency**
- `run_doctor(repo_root, optional_checks=())` is used consistently across tasks
- The CLI passes `args.check` directly into the doctor runner
- Optional check names are the same everywhere: `checkpoint`, `discovery`, `artifacts`

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-08-doctor-optional-checks.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
