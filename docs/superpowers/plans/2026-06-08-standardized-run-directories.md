# Standardized Run Directories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standard flat output directory for DES runs so each run writes `report.txt`, `run.json`, `run.csv`, `run.manifest.json`, and optional `run.memory.json` into one predictable folder.

**Architecture:** Extend the existing DES orchestrator so it writes the human-readable report and machine-readable artifacts into a resolved output directory. Add `--output-dir` to the DES CLI path, keep non-DES workflows unchanged, and preserve the current default behavior when the new flag is not used. Treat the output directory as the canonical run home so later `label-run`, `reuse-run`, and `compare-runs` commands can consume it directly.

**Tech Stack:** Python 3.13, `argparse`, `pathlib`, `pytest`, existing `des_multi_agent.cli`, `des_multi_agent.orchestrator`, `des_multi_agent.exporting`, and `des_multi_agent.run_memory`.

---

### Task 1: Add output-directory plumbing to the DES orchestrator and exporter

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/exporting.py`
- Test: `tests/test_exports.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.exporting import export_des_run_bundle


def test_export_bundle_writes_into_requested_output_dir(tmp_path: Path):
    output_dir = tmp_path / "runs" / "run_001"
    payload = {
        "workflow": "des",
        "component_a": "CCO",
        "n": 1,
        "results": [
            {
                "smiles_b": "O",
                "is_des": True,
                "min_tm_k": 200.0,
                "rank": 1,
                "source": "mock",
                "source_id": "mock-demo",
                "trust_score": 0.9,
                "uncertainty_flag": "low",
            }
        ],
    }

    paths = export_des_run_bundle(output_dir, payload)

    assert paths["json"] == output_dir / "run.json"
    assert paths["csv"] == output_dir / "run.csv"
    assert paths["manifest"] == output_dir / "run.manifest.json"
    assert (output_dir / "run.json").exists()
    assert (output_dir / "run.csv").exists()
    assert (output_dir / "run.manifest.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_exports.py -q
```

Expected: FAIL if the exporter or orchestrator still assumes the old loose output layout.

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path

from .exporting import export_des_run_bundle
from .reporting import format_report


def _resolve_des_output_dir(output_dir: str | Path | None, save_run_memory_path: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if save_run_memory_path is not None:
        return Path(save_run_memory_path).parent
    return Path.cwd()


def _write_report(output_dir: Path, report_text: str) -> Path:
    report_path = output_dir / "report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


# inside the DES branch after outcome is computed:
output_dir = _resolve_des_output_dir(args.output_dir, args.save_run_memory)
report_text = format_report(...)
_write_report(output_dir, report_text)
export_des_run_bundle(output_dir, outcome_payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_exports.py -q
```

Expected: PASS once the exporter writes into the requested run directory.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/exporting.py tests/test_exports.py
git commit -m "feat: standardize des run output directory"
```

### Task 2: Add `--output-dir` to the DES CLI path

**Files:**
- Modify: `des_multi_agent/cli.py:1-220`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.cli import build_parser


def test_des_cli_accepts_output_dir():
    parser = build_parser()
    args = parser.parse_args([
        "--workflow",
        "des",
        "--component-a",
        "CCO",
        "--checkpoint-path",
        "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        "--output-dir",
        "runs/run_001",
    ])

    assert args.output_dir == "runs/run_001"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_cli.py -q
```

Expected: FAIL because `--output-dir` is not yet defined.

- [ ] **Step 3: Write minimal implementation**

```python
parser.add_argument("--output-dir", default=None, help="Optional directory where DES run artifacts are written")


# in the DES call path
output_dir = resolve_existing_path(args.output_dir) if args.output_dir else None
...
outcome = run_search_report(..., output_dir=output_dir, ...)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_cli.py -q
```

Expected: PASS after the parser and DES path accept the new flag.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: add output dir flag for des runs"
```

### Task 3: Keep run-memory and reuse tools aligned with the standardized directory

**Files:**
- Modify: `des_multi_agent/run_memory.py`
- Modify: `des_multi_agent/label_run.py`
- Modify: `des_multi_agent/compare_runs.py`
- Test: `tests/test_run_memory.py`
- Test: `tests/test_label_run.py`
- Test: `tests/test_compare_runs.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.run_memory import load_run_memory, write_run_memory


def test_run_memory_round_trips_in_a_run_directory(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    memory_path = run_dir / "run.memory.json"
    memory = load_run_memory("tests/fixtures/example_run_memory.json")
    write_run_memory(memory_path, memory)

    assert memory_path.exists()
    assert load_run_memory(run_dir).workflow == "des"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_run_memory.py -q
```

Expected: FAIL if the helper path resolution still assumes ad hoc file placement only.

- [ ] **Step 3: Write minimal implementation**

```python
def resolve_run_memory_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "run.memory.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Run memory file not found: {candidate}")
    return candidate
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_run_memory.py -q
```

Expected: PASS after directory resolution remains compatible with the standardized run folder.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/run_memory.py des_multi_agent/label_run.py des_multi_agent/compare_runs.py tests/test_run_memory.py tests/test_label_run.py tests/test_compare_runs.py
git commit -m "feat: keep run memory tools aligned with run directories"
```

### Task 4: Update docs and example guidance for the run directory convention

**Files:**
- Modify: `README.md:1-220`
- Modify: `docs/tutorial.md:1-220`
- Modify: `examples/README.md:1-220`
- Test: `tests/test_doctor.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_standardized_run_directory_is_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples = Path("examples/README.md").read_text(encoding="utf-8")

    assert "--output-dir" in readme
    assert "report.txt" in tutorial
    assert "run.json" in examples
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```

Expected: FAIL until the docs explain the standardized run directory layout.

- [ ] **Step 3: Write minimal implementation**

```markdown
Use `--output-dir runs/run_001` to write a flat run folder containing `report.txt`, `run.json`, `run.csv`, and `run.manifest.json`.
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_doctor.py -q
```

Expected: PASS once the docs mention the run directory layout and the new flag.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md tests/test_doctor.py
git commit -m "docs: describe standardized des run directories"
```

### Task 5: Verify full-suite behavior and clean up any leftover dead code

**Files:**
- Test: `tests/test_exports.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_run_memory.py`
- Test: `tests/test_label_run.py`
- Test: `tests/test_compare_runs.py`
- Potential cleanup: `des_multi_agent/orchestrator.py`, `des_multi_agent/exporting.py`, `des_multi_agent/run_memory.py`

- [ ] **Step 1: Add a regression test that keeps the default behavior unchanged**

```python
from des_multi_agent.cli import main


def test_des_run_without_output_dir_still_works(capsys):
    try:
        main([
            "--workflow",
            "des",
            "--component-a",
            "CCO",
            "--n",
            "1",
            "--checkpoint-path",
            "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
            "--config-path",
            "ml_des_mp/config.yaml",
        ])
    except SystemExit as exc:
        assert exc.code in (0, 1)
    out = capsys.readouterr().out
    assert "smiles_b" in out
```

- [ ] **Step 2: Run the focused suite**

Run:
```bash
python -m pytest tests/test_exports.py tests/test_cli.py tests/test_run_memory.py tests/test_label_run.py tests/test_compare_runs.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run:
```bash
python -m pytest -q
```

Expected: PASS with the existing third-party warnings only.

- [ ] **Step 4: Remove any dead code surfaced by the run-directory work**

If the final review finds any helper or constant that is no longer used after the run-directory plumbing lands, remove it immediately and rerun the focused suite.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/exporting.py des_multi_agent/run_memory.py des_multi_agent/cli.py README.md docs/tutorial.md examples/README.md tests/test_exports.py tests/test_cli.py tests/test_run_memory.py tests/test_label_run.py tests/test_compare_runs.py tests/test_doctor.py
git commit -m "feat: standardize des run directories"
```

## Self-Review

**Spec coverage**
- `--output-dir` flag: Task 2
- flat run directory with `report.txt`: Task 1 and Task 4
- machine-readable artifacts in the same folder: Task 1
- `run.memory.json` in the same folder: Task 3
- reuse of the directory by `label-run`, `reuse-run`, and `compare-runs`: Task 3
- default behavior unchanged when `--output-dir` is omitted: Task 5
- docs updated: Task 4

**Placeholder scan**
- No TBD or TODO placeholders
- No undefined helper names
- No vague “add validation” steps without concrete examples

**Type consistency**
- `output_dir` is the single directory concept used throughout the plan
- `resolve_run_memory_path()` continues to accept either a file or a directory
- the CLI flag name is consistently `--output-dir`

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-08-standardized-run-directories.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
