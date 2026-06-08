# Example Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the checked-in example folders into a pytest-based regression benchmark suite that scores example output consistency against frozen baselines without rerunning external models.

**Architecture:** Add a small test-only helper module that enumerates the example folders, normalizes warning noise, and compares current example outputs against frozen baseline copies stored under `tests/fixtures/`. Keep the benchmark inside `tests/` so it runs with the normal suite and does not require network access or live model execution.

**Tech Stack:** Python, pytest, pathlib, text normalization helpers, existing example folders under `examples/`.

---

### Task 1: Add shared example benchmark helpers and frozen baselines

**Files:**
- Create: `tests/fixtures/example_benchmark_cases.py`
- Create: `tests/fixtures/example_benchmark_baselines/` (baseline copies of existing example artifacts)
- Test: `tests/test_benchmarks_examples.py`

- [ ] **Step 1: Write the failing test**

```python
from tests.fixtures.example_benchmark_cases import (
    BenchmarkCase,
    iter_benchmark_cases,
    normalize_benchmark_output,
)


def test_benchmark_cases_cover_current_examples():
    names = [case.name for case in iter_benchmark_cases()]
    assert "task_router" in names
    assert "plain_language_gemma4_12b" in names
    assert "plain_language_metal_binding_gemma4_12b" in names
    assert "des_viscosity" in names
    assert "metal_binding" in names


def test_normalize_benchmark_output_strips_warning_noise():
    raw = "\n".join([
        "  ",
        "DeprecationWarning: torch_geometric.distributed has been deprecated",
        "actual report line",
        "",
        "",
        "DeprecationWarning: `torch.jit.script` is deprecated.",
    ])
    normalized = normalize_benchmark_output(raw)
    assert normalized == "actual report line"


def test_benchmark_case_paths_exist():
    for case in iter_benchmark_cases():
        assert case.example_input.exists()
        assert case.example_output.exists()
        assert case.example_readme.exists()
        assert case.baseline_input.exists()
        assert case.baseline_output.exists()
        assert case.baseline_readme.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmarks_examples.py -q`
Expected: FAIL with import or missing-helper errors until the helper module and baselines exist.

- [ ] **Step 3: Write minimal implementation**

Create the baseline snapshot directories and the helper module.

```bash
mkdir -p tests/fixtures/example_benchmark_baselines
for name in des_viscosity viscosity_template metal_binding ligand_binding_template gemma4_12b nemotron_3_nano qwen3_6 lidocaine_gemma4_12b plain_language_gemma4_12b plain_language_metal_binding_gemma4_12b task_router; do
  mkdir -p "tests/fixtures/example_benchmark_baselines/$name"
  cp "examples/$name/input.txt" "tests/fixtures/example_benchmark_baselines/$name/input.txt"
  cp "examples/$name/output.txt" "tests/fixtures/example_benchmark_baselines/$name/output.txt"
  cp "examples/$name/README.md" "tests/fixtures/example_benchmark_baselines/$name/README.md"
done
```

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXAMPLES_ROOT = Path("examples")
BASELINE_ROOT = Path(__file__).resolve().parent / "example_benchmark_baselines"
WARNING_MARKERS = (
    "DeprecationWarning: torch_geometric.distributed",
    "DeprecationWarning: `torch.jit.script` is deprecated.",
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    example_dir: Path
    baseline_dir: Path

    @property
    def example_input(self) -> Path:
        return self.example_dir / "input.txt"

    @property
    def example_output(self) -> Path:
        return self.example_dir / "output.txt"

    @property
    def example_readme(self) -> Path:
        return self.example_dir / "README.md"

    @property
    def baseline_input(self) -> Path:
        return self.baseline_dir / "input.txt"

    @property
    def baseline_output(self) -> Path:
        return self.baseline_dir / "output.txt"

    @property
    def baseline_readme(self) -> Path:
        return self.baseline_dir / "README.md"


def normalize_benchmark_output(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] == "":
                continue
            lines.append("")
            continue
        if any(marker in line for marker in WARNING_MARKERS):
            continue
        lines.append(line)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def iter_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for example_dir in sorted(EXAMPLES_ROOT.iterdir()):
        if not example_dir.is_dir():
            continue
        if not (example_dir / "input.txt").exists():
            continue
        if not (example_dir / "output.txt").exists():
            continue
        if not (example_dir / "README.md").exists():
            continue
        baseline_dir = BASELINE_ROOT / example_dir.name
        if not baseline_dir.exists():
            continue
        cases.append(BenchmarkCase(example_dir.name, example_dir, baseline_dir))
    return tuple(cases)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmarks_examples.py -q`
Expected: PASS after the helper module exports the expected symbols and the baseline snapshots are present.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/example_benchmark_cases.py tests/fixtures/example_benchmark_baselines tests/test_benchmarks_examples.py
git commit -m "test: add example benchmark fixtures"
```

### Task 2: Add the pytest benchmark regression cases and aggregate score

**Files:**
- Modify: `tests/test_benchmarks_examples.py`
- Modify: `tests/fixtures/example_benchmark_cases.py`

- [ ] **Step 1: Write the failing test**

```python
from tests.fixtures.example_benchmark_cases import (
    iter_benchmark_cases,
    normalize_benchmark_output,
)


def test_each_example_output_matches_frozen_baseline():
    for case in iter_benchmark_cases():
        actual = normalize_benchmark_output(case.example_output.read_text(encoding="utf-8"))
        expected = normalize_benchmark_output(case.baseline_output.read_text(encoding="utf-8"))
        assert actual == expected, case.name


def test_example_benchmark_score_is_perfect():
    cases = iter_benchmark_cases()
    matched = 0
    for case in cases:
        actual = normalize_benchmark_output(case.example_output.read_text(encoding="utf-8"))
        expected = normalize_benchmark_output(case.baseline_output.read_text(encoding="utf-8"))
        matched += int(actual == expected)
    score = matched / len(cases)
    assert score == 1.0, f"benchmark score was {score:.3f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmarks_examples.py -q`
Expected: FAIL if any example output drifts away from its frozen baseline or if the benchmark case list is incomplete.

- [ ] **Step 3: Write minimal implementation**

Use the helper module from Task 1 and keep the benchmark logic strictly normalized-text comparison.

```python
from tests.fixtures.example_benchmark_cases import iter_benchmark_cases, normalize_benchmark_output


def test_each_example_output_matches_frozen_baseline():
    for case in iter_benchmark_cases():
        actual = normalize_benchmark_output(case.example_output.read_text(encoding="utf-8"))
        expected = normalize_benchmark_output(case.baseline_output.read_text(encoding="utf-8"))
        assert actual == expected, case.name


def test_example_benchmark_score_is_perfect():
    cases = iter_benchmark_cases()
    matched = 0
    for case in cases:
        actual = normalize_benchmark_output(case.example_output.read_text(encoding="utf-8"))
        expected = normalize_benchmark_output(case.baseline_output.read_text(encoding="utf-8"))
        matched += int(actual == expected)
    score = matched / len(cases)
    assert score == 1.0, f"benchmark score was {score:.3f}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmarks_examples.py -q`
Expected: PASS once the benchmark helper and baseline snapshots are in place.

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmarks_examples.py tests/fixtures/example_benchmark_cases.py
git commit -m "test: add example benchmark regression cases"
```

### Task 3: Document the benchmark and keep the examples discoverable

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_docs_mention_example_benchmark_suite():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples_readme = Path("examples/README.md").read_text(encoding="utf-8")
    assert "example benchmark" in readme.lower()
    assert "example benchmark" in tutorial.lower()
    assert "example benchmark" in examples_readme.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmarks_examples.py -q`
Expected: FAIL until the README, tutorial, and examples index mention the benchmark suite.

- [ ] **Step 3: Write minimal implementation**

Add one short paragraph to each doc that says:
- the example folders are also regression benchmarks
- the benchmark lives in `tests/test_benchmarks_examples.py`
- the benchmark uses frozen baseline copies under `tests/fixtures/example_benchmark_baselines/`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_demo_des_search.py tests/test_benchmarks_examples.py -q`
Expected: PASS with the benchmark docs and example references present.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md tests/test_demo_des_search.py
git commit -m "docs: document example benchmark suite"
```

### Task 4: Verify the full suite still passes

**Files:**
- No new files
- Validate: `tests/test_benchmarks_examples.py`

- [ ] **Step 1: Run the focused benchmark slice**

Run: `python -m pytest tests/test_benchmarks_examples.py tests/test_demo_des_search.py -q`
Expected: PASS

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS with the existing third-party warnings only.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: add example benchmark coverage"
```
