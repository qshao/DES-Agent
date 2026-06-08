# Compare Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only `compare-runs` subcommand that compares two saved run artifacts from the same workflow and prints a compact terminal report showing how the top ranked candidates changed.

**Architecture:** Implement a focused comparison module that reuses the existing run-memory loader, rejects malformed or mismatched inputs, and renders a compact diff-style terminal report for the top ranked candidates only. Wire the new subcommand into the CLI and document when to use it in the setup guide.

**Tech Stack:** Python 3.13, `argparse`, `pathlib`, `pytest`, existing `des_multi_agent.run_memory` and `des_multi_agent.cli` modules.

---

### Task 1: Add the comparison model and diff logic

**Files:**
- Create: `des_multi_agent/compare_runs.py`
- Create: `tests/test_compare_runs.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from des_multi_agent.compare_runs import compare_saved_runs, format_compare_report


def _write_memory(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def test_compare_saved_runs_marks_rank_changes(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_memory(
        run_a / "run.memory.json",
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )
    _write_memory(
        run_b / "run.memory.json",
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "CC(=O)O", "rank": 1, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CN", "rank": 2, "min_tm_k": 241.11, "trust_score": 0.80, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )

    comparison = compare_saved_runs(run_a, run_b)

    assert comparison.workflow == "des"
    assert any(item.status == "moved" and item.smiles_b == "CC(=O)O" for item in comparison.rows)
    assert any(item.status == "removed" and item.smiles_b == "O" for item in comparison.rows)
    assert any(item.status == "new" and item.smiles_b == "CN" for item in comparison.rows)
    assert "compare-runs" in format_compare_report(comparison)


def test_compare_saved_runs_rejects_mismatched_workflow(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_memory(
        run_a / "run.memory.json",
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": []
        }""",
    )
    _write_memory(
        run_b / "run.memory.json",
        """{
          "workflow": "metal-binding",
          "component_a": null,
          "n": null,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "NCCN", "rank": 1, "min_tm_k": null, "trust_score": null, "uncertainty_flag": "", "source": "heuristic", "source_id": ""}
          ]
        }""",
    )

    with pytest.raises(ValueError, match="workflow"):
        compare_saved_runs(run_a, run_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_compare_runs.py -q
```
Expected: FAIL because `des_multi_agent.compare_runs` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .run_memory import load_run_memory


@dataclass(frozen=True)
class CompareRow:
    smiles_b: str
    left_rank: int | None
    right_rank: int | None
    status: str


@dataclass(frozen=True)
class CompareResult:
    workflow: str
    left_path: Path
    right_path: Path
    rows: list[CompareRow]


def compare_saved_runs(left: str | Path, right: str | Path) -> CompareResult:
    left_memory = load_run_memory(left)
    right_memory = load_run_memory(right)
    if left_memory.workflow != right_memory.workflow:
        raise ValueError("compare-runs requires both saved runs to have the same workflow")
    # build top-rank diff rows
    return CompareResult(workflow=left_memory.workflow, left_path=Path(left), right_path=Path(right), rows=[])


def format_compare_report(result: CompareResult) -> str:
    lines = [f"compare-runs: {result.workflow}"]
    for row in result.rows:
        lines.append(f"{row.smiles_b} | {row.status} | {row.left_rank} -> {row.right_rank}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_compare_runs.py -q
```
Expected: PASS once the comparison logic and formatting are complete.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/compare_runs.py tests/test_compare_runs.py
git commit -m "feat: add run comparison logic"
```

### Task 2: Wire `compare-runs` into the CLI

**Files:**
- Modify: `des_multi_agent/cli.py:1-260`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import des_multi_agent.cli as cli_module


def test_compare_runs_subcommand_prints_report(monkeypatch, capsys, tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "run.memory.json").write_text("{}", encoding="utf-8")
    (right / "run.memory.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli_module, "compare_saved_runs", lambda a, b: type("R", (), {"workflow": "des", "rows": []})())
    monkeypatch.setattr(cli_module, "format_compare_report", lambda result: "compare-runs report")

    cli_module.main(["compare-runs", str(left), str(right)])
    out = capsys.readouterr().out.strip()
    assert out == "compare-runs report"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_cli.py -q
```
Expected: FAIL because `compare-runs` is not yet a CLI subcommand.

- [ ] **Step 3: Write minimal implementation**

```python
from .compare_runs import compare_saved_runs, format_compare_report

# inside build_parser()
compare_runs_parser = subparsers.add_parser("compare-runs", help="Compare two saved runs from the same workflow")
compare_runs_parser.add_argument("left", help="Left run folder or run.memory.json file")
compare_runs_parser.add_argument("right", help="Right run folder or run.memory.json file")
compare_runs_parser.set_defaults(command="compare-runs")

# inside main()
if getattr(args, "command", None) == "compare-runs":
    try:
        result = compare_saved_runs(args.left, args.right)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(format_compare_report(result))
    return
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
git commit -m "feat: add compare-runs cli subcommand"
```

### Task 3: Document `compare-runs`

**Files:**
- Modify: `README.md:1-220`
- Modify: `docs/tutorial.md:1-260`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_compare_runs_is_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    assert "compare-runs" in readme
    assert "compare-runs" in tutorial
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_demo_des_search.py -q
```
Expected: FAIL until the docs mention the new command.

- [ ] **Step 3: Write minimal implementation**

```markdown
## Compare Runs

Use `python -m des_multi_agent.cli compare-runs <run-a> <run-b>` to compare two saved runs from the same workflow and see which top candidates moved, disappeared, or appeared.
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_demo_des_search.py -q
```
Expected: PASS after the docs are updated.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md
git commit -m "docs: add compare runs guidance"
```

### Task 4: Final verification

**Files:**
- Verify: `des_multi_agent/compare_runs.py`, `des_multi_agent/cli.py`, `tests/test_compare_runs.py`, `tests/test_cli.py`, `README.md`, `docs/tutorial.md`

- [ ] **Step 1: Run focused tests**

Run:
```bash
python -m pytest tests/test_compare_runs.py tests/test_cli.py tests/test_demo_des_search.py -q
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
git add des_multi_agent/compare_runs.py des_multi_agent/cli.py tests/test_compare_runs.py tests/test_cli.py README.md docs/tutorial.md
git commit -m "feat: add compare-runs command"
```
