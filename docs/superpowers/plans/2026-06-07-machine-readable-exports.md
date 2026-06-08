# Machine-Readable Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic machine-readable exports for every DES run so the workflow writes `run.json`, `run.csv`, and `run.manifest.json` next to the human-readable report.

**Architecture:** Add a small post-processing export layer that receives the completed DES run object, writes the three-file export bundle, and fails clearly if the export directory or required fields are invalid. Keep the terminal report unchanged and keep the export strictly DES-only.

**Tech Stack:** Python 3.13, `json`, `csv`, `pathlib`, `pytest`, existing `des_multi_agent.orchestrator` and `des_multi_agent.cli` modules.

---

### Task 1: Add export payload and file writers

**Files:**
- Create: `des_multi_agent/exporting.py`
- Create: `tests/test_exports.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from des_multi_agent.exporting import export_des_run_bundle


def test_export_des_run_bundle_writes_three_files(tmp_path: Path):
    output_dir = tmp_path / "runs" / "run_001"
    output_dir.mkdir(parents=True)
    run_payload = {
        "workflow": "des",
        "component_a": "CCO",
        "n": 5,
        "results": [
            {
                "smiles_b": "O",
                "is_des": True,
                "min_tm_k": 208.69,
                "rank": 1,
                "source": "heuristic",
                "source_id": "rule",
                "trust_score": 0.95,
                "uncertainty_flag": "low",
            }
        ],
        "memory_notes": ["Loaded 1 prior ranked candidates for ranking bias."],
        "warnings": ["demo warning"],
    }

    exported = export_des_run_bundle(output_dir, run_payload)

    assert (output_dir / "run.json").exists()
    assert (output_dir / "run.csv").exists()
    assert (output_dir / "run.manifest.json").exists()
    assert exported["json"].name == "run.json"
    assert exported["csv"].name == "run.csv"
    assert exported["manifest"].name == "run.manifest.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_exports.py -q
```
Expected: FAIL because `des_multi_agent.exporting` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from csv import DictWriter
from pathlib import Path
import json


def export_des_run_bundle(output_dir: str | Path, run_payload: dict) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "run.json"
    csv_path = output_path / "run.csv"
    manifest_path = output_path / "run.manifest.json"
    json_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True), encoding="utf-8")
    # flatten ranked results into CSV
    return {"json": json_path, "csv": csv_path, "manifest": manifest_path}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_exports.py -q
```
Expected: PASS once the writer is complete.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/exporting.py tests/test_exports.py
git commit -m "feat: add export writers"
```

### Task 2: Wire exports into the DES orchestrator

**Files:**
- Modify: `des_multi_agent/orchestrator.py:1-420`
- Test: `tests/test_exports.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.orchestrator import SearchOutcome, _build_des_export_payload


def test_build_des_export_payload_maps_run_fields():
    outcome = SearchOutcome(
        results=[],
        annotated_results=[],
        candidate_proposals=[],
        candidate_reviews=[],
        brainstorm_candidates=[],
        explanation_notes=[],
        critique_notes=[],
        llm_warnings=[],
        memory_notes=[],
        viscosity_predictions=[],
    )

    payload = _build_des_export_payload(outcome, component_a="CCO", n=5)

    assert payload["workflow"] == "des"
    assert payload["component_a"] == "CCO"
    assert payload["n"] == 5
    assert payload["results"] == []
    assert payload["memory_notes"] == []
    assert payload["warnings"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_exports.py -q
```
Expected: FAIL until the orchestrator exposes the export payload builder.

- [ ] **Step 3: Write minimal implementation**

```python
from .exporting import export_des_run_bundle


def _build_des_export_payload(outcome, component_a: str, n: int) -> dict:
    return {
        "workflow": "des",
        "component_a": component_a,
        "n": n,
        "results": [
            {
                "smiles_b": result.curve.smiles_b,
                "is_des": result.is_des,
                "min_tm_k": result.min_tm_k,
                "rank": index,
                "source": getattr(proposal, "source", ""),
                "source_id": getattr(proposal, "source_id", ""),
                "trust_score": annotated.trust_score,
                "uncertainty_flag": annotated.uncertainty.uncertainty_flag,
            }
            for index, (result, annotated, proposal) in enumerate(
                zip(outcome.results, outcome.annotated_results, outcome.candidate_proposals),
                start=1,
            )
        ],
        "memory_notes": list(getattr(outcome, "memory_notes", [])),
        "warnings": list(getattr(outcome, "llm_warnings", [])),
    }

# after the DES run completes:
run_payload = _build_des_export_payload(outcome, component_a=component_a, n=n)
export_des_run_bundle(output_dir, run_payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_exports.py -q
```
Expected: PASS after the orchestrator is wired.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py
git commit -m "feat: export DES run bundles"
```

### Task 3: Document the export artifacts

**Files:**
- Modify: `README.md:1-220`
- Modify: `docs/tutorial.md:1-260`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_exports_are_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    assert "run.json" in readme
    assert "run.csv" in readme
    assert "run.manifest.json" in readme
    assert "run.json" in tutorial
    assert "run.csv" in tutorial
    assert "run.manifest.json" in tutorial
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_demo_des_search.py -q
```
Expected: FAIL until the docs mention the export files.

- [ ] **Step 3: Write minimal implementation**

```markdown
Every DES run now writes `run.json`, `run.csv`, and `run.manifest.json` next to the human-readable report.
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
git commit -m "docs: add export bundle guidance"
```

### Task 4: Final verification

**Files:**
- Verify: `des_multi_agent/exporting.py`, `des_multi_agent/orchestrator.py`, `tests/test_exports.py`, `README.md`, `docs/tutorial.md`

- [ ] **Step 1: Run focused tests**

Run:
```bash
python -m pytest tests/test_exports.py tests/test_demo_des_search.py -q
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
git add des_multi_agent/exporting.py des_multi_agent/orchestrator.py tests/test_exports.py README.md docs/tutorial.md
git commit -m "feat: add machine-readable exports"
```
