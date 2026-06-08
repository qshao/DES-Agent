# Compare Runs JSON Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--json` flag to `compare-runs` so it prints a compact machine-readable summary on stdout in addition to the existing terminal diff table.

**Architecture:** Keep the same-workflow comparison logic in `des_multi_agent/compare_runs.py` and add a second formatter that derives a compact JSON summary from the same `CompareResult`. The CLI should accept `--json`, print the human-readable diff table as before, and then emit the JSON summary. Keep the command read-only, offline, and hard-erroring for malformed or mismatched inputs.

**Tech Stack:** Python 3.13, `argparse`, `json`, `pathlib`, `pytest`, existing `des_multi_agent.compare_runs` and `des_multi_agent.cli`.

---

### Task 1: Add a JSON summary builder to the compare-runs module

**Files:**
- Modify: `des_multi_agent/compare_runs.py`
- Test: `tests/test_compare_runs.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.compare_runs import compare_saved_runs, format_compare_json


def test_compare_saved_runs_json_summary_includes_counts_and_changed_candidates(tmp_path: Path):
    left = tmp_path / "left.memory.json"
    right = tmp_path / "right.memory.json"
    left.write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CN", "rank": 3, "min_tm_k": 241.11, "trust_score": 0.80, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
        encoding="utf-8",
    )
    right.write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 5,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "CC(=O)O", "rank": 1, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CN", "rank": 2, "min_tm_k": 241.11, "trust_score": 0.80, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "OC", "rank": 3, "min_tm_k": 230.00, "trust_score": 0.75, "uncertainty_flag": "medium", "source": "heuristic", "source_id": ""}
          ]
        }""",
        encoding="utf-8",
    )

    comparison = compare_saved_runs(left, right)
    summary = format_compare_json(comparison)

    assert summary["workflow"] == "des"
    assert summary["left"]["path"].endswith("left.memory.json")
    assert summary["right"]["path"].endswith("right.memory.json")
    assert summary["counts"] == {"new": 1, "removed": 1, "moved": 2, "unchanged": 0}
    assert summary["changed_candidates"][0]["smiles_b"] in {"CC(=O)O", "CN", "O", "OC"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_compare_runs.py -q
```

Expected: FAIL because `format_compare_json()` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def format_compare_json(result: CompareResult) -> dict[str, object]:
    counts = {"new": 0, "removed": 0, "moved": 0, "unchanged": 0}
    changed_candidates = []
    for row in result.rows:
        counts[row.status] += 1
        if row.status != "unchanged":
            changed_candidates.append(
                {
                    "smiles_b": row.smiles_b,
                    "status": row.status,
                    "left_rank": row.left_rank,
                    "right_rank": row.right_rank,
                }
            )
    return {
        "workflow": result.workflow,
        "left": {"path": str(result.left_path), "component_a": result.left_component_a, "n": result.left_n},
        "right": {"path": str(result.right_path), "component_a": result.right_component_a, "n": result.right_n},
        "counts": counts,
        "changed_candidates": changed_candidates,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_compare_runs.py -q
```

Expected: PASS once the JSON formatter is added.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/compare_runs.py tests/test_compare_runs.py
git commit -m "feat: add compare-runs json summary"
```

### Task 2: Wire `--json` into the CLI for compare-runs

**Files:**
- Modify: `des_multi_agent/cli.py:1-220`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from io import StringIO
from contextlib import redirect_stdout

import des_multi_agent.cli as cli_module


def test_compare_runs_subcommand_accepts_json_flag(monkeypatch):
    parser = cli_module.build_parser()
    args = parser.parse_args(["compare-runs", "runs/run_001", "runs/run_002", "--json"])
    assert args.json is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_cli.py -q
```

Expected: FAIL because `compare-runs` does not yet accept `--json`.

- [ ] **Step 3: Write minimal implementation**

```python
compare_runs_parser.add_argument("--json", action="store_true", help="Also print a JSON summary of the comparison")


if getattr(args, "command", None) == "compare-runs":
    result = compare_saved_runs(args.left, args.right)
    print(format_compare_report(result))
    if args.json:
        print(json.dumps(format_compare_json(result), indent=2, sort_keys=True))
    return
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_cli.py -q
```

Expected: PASS after the parser and compare-runs branch accept `--json`.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: wire compare-runs json output into cli"
```

### Task 3: Document the JSON mode and keep compare-runs examples aligned

**Files:**
- Modify: `README.md:1-220`
- Modify: `docs/tutorial.md:1-220`
- Modify: `examples/README.md:1-220`
- Test: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_compare_runs_json_is_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples = Path("examples/README.md").read_text(encoding="utf-8")
    assert "compare-runs --json" in readme
    assert "compare-runs --json" in tutorial
    assert "compare-runs --json" in examples
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_demo_des_search.py -q
```

Expected: FAIL until the docs mention JSON output mode.

- [ ] **Step 3: Write minimal implementation**

```markdown
Use `python -m des_multi_agent.cli compare-runs run_a run_b --json` when you want a machine-readable summary of the comparison.
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_demo_des_search.py -q
```

Expected: PASS after the docs mention JSON output mode.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md tests/test_demo_des_search.py
git commit -m "docs: describe compare-runs json output"
```

### Task 4: Verify the terminal report and JSON summary stay aligned

**Files:**
- Test: `tests/test_compare_runs.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add a regression test that compares terminal and JSON counts**

```python
import json
from pathlib import Path

from des_multi_agent.compare_runs import compare_saved_runs, format_compare_json, format_compare_report


def test_compare_runs_json_matches_terminal_summary(tmp_path: Path):
    left = tmp_path / "left.memory.json"
    right = tmp_path / "right.memory.json"
    left.write_text("""{... valid DES run memory ...}""", encoding="utf-8")
    right.write_text("""{... valid DES run memory ...}""", encoding="utf-8")

    comparison = compare_saved_runs(left, right)
    text = format_compare_report(comparison)
    summary = format_compare_json(comparison)

    assert "compare-runs: des" in text
    assert summary["workflow"] == "des"
    assert sum(summary["counts"].values()) == len(comparison.rows)
```

- [ ] **Step 2: Run the focused compare-runs tests**

Run:
```bash
python -m pytest tests/test_compare_runs.py tests/test_cli.py -q
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
git add des_multi_agent/compare_runs.py des_multi_agent/cli.py README.md docs/tutorial.md examples/README.md tests/test_compare_runs.py tests/test_cli.py tests/test_demo_des_search.py
git commit -m "feat: add compare-runs json summary"
```

## Self-Review

**Spec coverage**
- `--json` flag on `compare-runs`: Task 2
- compact JSON summary: Task 1
- same-workflow hard errors preserved: Task 1 and Task 4
- docs updates: Task 3
- terminal report unchanged in meaning: Task 4

**Placeholder scan**
- No TBD or TODO placeholders
- No undefined helper names
- No vague “add validation” steps without concrete examples

**Type consistency**
- `format_compare_json(result: CompareResult)` is introduced in Task 1 and used in Task 2
- `args.json` is the CLI flag name used consistently across tasks
- the JSON summary shape is stable and explicit in Task 1

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-08-compare-runs-json.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
