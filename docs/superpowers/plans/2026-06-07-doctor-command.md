# Doctor Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast, read-only `doctor` subcommand that checks the core repo and checked-in example folders, reports all issues in one pass, and distinguishes errors from warnings.

**Architecture:** Implement the checks in a focused `des_multi_agent/doctor.py` module that returns structured results, then wire a new `doctor` CLI subcommand to print a terminal-friendly report and exit nonzero on errors. Keep the checks local and deterministic, limited to file existence, readability, and doc/example consistency. Update the docs so `doctor` becomes the first setup step before running demos.

**Tech Stack:** Python 3.13, `argparse`, `pathlib`, `pytest`, existing `des_multi_agent.cli` and repo doc conventions.

---

### Task 1: Add the doctor result model and checks

**Files:**
- Create: `des_multi_agent/doctor.py`
- Create: `tests/test_doctor.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.doctor import run_doctor, format_doctor_report


def test_doctor_reports_all_example_folders(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "ml_des_mp").mkdir(parents=True)
    (repo_root / "ml_des_mp" / "config.yaml").write_text("config: true\n", encoding="utf-8")
    (repo_root / "llm.example.yaml").write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    (repo_root / "examples" / "demo").mkdir(parents=True)
    for name in ("README.md", "input.txt", "output.txt", "run.sh"):
        (repo_root / "examples" / "demo" / name).write_text("ok\n", encoding="utf-8")
    (repo_root / "README.md").write_text("example benchmark suite\n", encoding="utf-8")
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "docs" / "tutorial.md").write_text("example benchmark suite\n", encoding="utf-8")
    (repo_root / "examples" / "README.md").write_text("demo\n", encoding="utf-8")
    (repo_root / "tests" / "fixtures" / "example_benchmark_baselines").mkdir(parents=True)

    result = run_doctor(repo_root)

    assert result.errors == []
    assert result.warnings == []
    assert "doctor: ok" in format_doctor_report(result)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```
Expected: FAIL because `des_multi_agent.doctor` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


def format_doctor_report(result: DoctorResult) -> str:
    lines = ["doctor: ok" if not result.errors else "doctor: issues found"]
    if result.errors:
        lines.append("errors:")
        for issue in result.errors:
            lines.append(f"- {issue.message}")
    if result.warnings:
        lines.append("warnings:")
        for issue in result.warnings:
            lines.append(f"- {issue.message}")
    return "\n".join(lines)


def run_doctor(repo_root: str | Path) -> DoctorResult:
    root = Path(repo_root)
    errors: list[DoctorIssue] = []
    warnings: list[DoctorIssue] = []
    # minimal checks for core files and example folders
    return DoctorResult(errors=errors, warnings=warnings)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```
Expected: PASS once the doctor model and baseline checks are wired in.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/doctor.py tests/test_doctor.py
git commit -m "feat: add doctor checks"
```

### Task 2: Wire `doctor` into the CLI

**Files:**
- Modify: `des_multi_agent/cli.py:1-220`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.cli import main


def test_doctor_command_prints_report(capsys):
    try:
        main(["doctor"])
    except SystemExit as exc:
        assert exc.code in (0, 1)
    out = capsys.readouterr().out
    assert "doctor:" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_cli.py -q
```
Expected: FAIL because `doctor` is not yet a CLI subcommand.

- [ ] **Step 3: Write minimal implementation**

```python
from .doctor import format_doctor_report, run_doctor

# inside build_parser()
doctor_parser = subparsers.add_parser("doctor", help="Check local repo and example setup")
doctor_parser.set_defaults(command="doctor")

# inside main()
if getattr(args, "command", None) == "doctor":
    result = run_doctor(PROJECT_ROOT)
    print(format_doctor_report(result))
    raise SystemExit(result.exit_code)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_cli.py -q
```
Expected: PASS after CLI routing is added.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: add doctor cli subcommand"
```

### Task 3: Document `doctor` and verify repo/example coverage

**Files:**
- Modify: `README.md:1-220`
- Modify: `docs/tutorial.md:1-220`
- Modify: `examples/README.md:1-220`
- Modify: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_doctor_is_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples = Path("examples/README.md").read_text(encoding="utf-8")
    assert "doctor" in readme
    assert "doctor" in tutorial
    assert "doctor" in examples
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_demo_des_search.py -q
```
Expected: FAIL until the docs mention the new command.

- [ ] **Step 3: Write minimal implementation**

```markdown
## Doctor

Run `python -m des_multi_agent.cli doctor` before your first demo to check that the repo, examples, and benchmark fixtures are present.
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_demo_des_search.py -q
```
Expected: PASS after the docs are updated.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md tests/test_demo_des_search.py
git commit -m "docs: add doctor usage guidance"
```

### Task 4: Final verification

**Files:**
- No new files
- Verify: `des_multi_agent/doctor.py`, `des_multi_agent/cli.py`, `tests/test_doctor.py`, `tests/test_cli.py`, `tests/test_demo_des_search.py`

- [ ] **Step 1: Run focused tests**

Run:
```bash
python -m pytest tests/test_doctor.py tests/test_cli.py tests/test_demo_des_search.py -q
```
Expected: PASS.

- [ ] **Step 2: Run the full suite**

Run:
```bash
python -m pytest -q
```
Expected: PASS with the existing third-party warnings only.

- [ ] **Step 3: Commit the verified slice**

```bash
git add des_multi_agent/doctor.py des_multi_agent/cli.py tests/test_doctor.py tests/test_cli.py README.md docs/tutorial.md examples/README.md tests/test_demo_des_search.py
git commit -m "feat: add doctor command"
```
