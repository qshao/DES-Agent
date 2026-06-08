# DES Run Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline DES-only memory layer so a user can save one run to a local JSON file and optionally reuse that file in a later DES run to nudge ranking, without changing the underlying melting-temperature prediction pipeline.

**Architecture:** The DES workflow will gain a small local memory schema plus save/load helpers. The CLI will expose an explicit save path for the current run and a reuse path for a later run. Loaded memory will only affect ranking inside the DES workflow; it will not filter candidates automatically, and it will not change the predictor or uncertainty calculations. Reports and docs will show when memory was applied.

**Tech Stack:** Python, JSON, pytest, existing DES workflow, existing ranking/reporting code, existing CLI

---

### Task 1: Add the DES run-memory schema and serialization helpers

**Files:**
- Create: `des_multi_agent/memory_schema.py`
- Create: `des_multi_agent/run_memory.py`
- Test: `tests/test_run_memory.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from des_multi_agent.run_memory import load_run_memory, parse_run_memory


def test_load_run_memory_reads_des_json_from_file(tmp_path: Path):
    memory_path = tmp_path / "run.memory.json"
    memory_path.write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 20,
          "labels": [
            {"smiles_b": "O", "label": "good"},
            {"smiles_b": "CC(=O)O", "label": "bad"}
          ],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
        encoding="utf-8",
    )

    memory = load_run_memory(memory_path)

    assert memory.workflow == "des"
    assert memory.component_a == "CCO"
    assert memory.n == 20
    assert memory.labels[0].smiles_b == "O"
    assert memory.labels[0].label == "good"
    assert memory.labels[1].smiles_b == "CC(=O)O"
    assert memory.labels[1].label == "bad"
    assert memory.ranked_candidates[0].smiles_b == "O"
    assert memory.ranked_candidates[0].rank == 1


def test_load_run_memory_reads_des_json_from_folder(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.memory.json").write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 10,
          "labels": [],
          "ranked_candidates": []
        }""",
        encoding="utf-8",
    )

    memory = load_run_memory(run_dir)

    assert memory.workflow == "des"
    assert memory.component_a == "CCO"
    assert memory.n == 10


def test_parse_run_memory_rejects_metal_binding_workflow():
    with pytest.raises(ValueError, match="workflow must be des"):
        parse_run_memory(
            {
                "workflow": "metal-binding",
                "component_a": None,
                "n": None,
                "labels": [],
                "ranked_candidates": [],
            }
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_memory.py -q`
Expected: FAIL because `des_multi_agent.run_memory` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

`des_multi_agent/memory_schema.py`
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunLabel:
    smiles_b: str
    label: str


@dataclass(frozen=True)
class RunCandidateSummary:
    smiles_b: str
    rank: int
    min_tm_k: float | None = None
    trust_score: float | None = None
    uncertainty_flag: str = ""
    source: str = ""
    source_id: str = ""


@dataclass(frozen=True)
class RunMemory:
    workflow: str
    component_a: str | None
    n: int | None
    labels: list[RunLabel]
    ranked_candidates: list[RunCandidateSummary]
```

`des_multi_agent/run_memory.py`
```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
import json

from .memory_schema import RunCandidateSummary, RunLabel, RunMemory


def resolve_run_memory_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "run.memory.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Run memory file not found: {candidate}")
    return candidate


def parse_run_memory(data: Mapping[str, object]) -> RunMemory:
    if data.get("workflow") != "des":
        raise ValueError("run memory workflow must be des")
    labels: list[RunLabel] = []
    for item in data.get("labels", []):
        labels.append(RunLabel(smiles_b=item["smiles_b"], label=item["label"]))
    ranked_candidates: list[RunCandidateSummary] = []
    for item in data.get("ranked_candidates", []):
        ranked_candidates.append(
            RunCandidateSummary(
                smiles_b=item["smiles_b"],
                rank=int(item["rank"]),
                min_tm_k=item.get("min_tm_k"),
                trust_score=item.get("trust_score"),
                uncertainty_flag=item.get("uncertainty_flag", ""),
                source=item.get("source", ""),
                source_id=item.get("source_id", ""),
            )
        )
    return RunMemory(
        workflow="des",
        component_a=data.get("component_a"),
        n=data.get("n"),
        labels=labels,
        ranked_candidates=ranked_candidates,
    )


def load_run_memory(path: str | Path) -> RunMemory:
    memory_path = resolve_run_memory_path(path)
    data = json.loads(memory_path.read_text(encoding="utf-8"))
    return parse_run_memory(data)


def write_run_memory(path: str | Path, memory: RunMemory) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(memory)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_memory.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/memory_schema.py des_multi_agent/run_memory.py tests/test_run_memory.py
git commit -m "feat: add des run memory schema"
```

### Task 2: Add CLI flags to save a run memory file and reuse a prior run

**Files:**
- Modify: `des_multi_agent/cli.py`
- Modify: `des_multi_agent/orchestrator.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.cli import build_parser


def test_parser_supports_run_memory_flags():
    parser = build_parser()
    args = parser.parse_args([
        "--workflow",
        "des",
        "--component-a",
        "CCO",
        "--checkpoint-path",
        "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        "--save-run-memory",
        "runs/run_001/run.memory.json",
        "--reuse-run",
        "runs/run_000/run.memory.json",
    ])
    assert args.save_run_memory == "runs/run_001/run.memory.json"
    assert args.reuse_run == "runs/run_000/run.memory.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_parser_supports_run_memory_flags -q`
Expected: FAIL because the flags do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
parser.add_argument(
    "--save-run-memory",
    default=None,
    help="Optional path to write a compact JSON memory file for later reuse",
)
parser.add_argument(
    "--reuse-run",
    default=None,
    help="Optional prior DES run folder or run.memory.json file to reuse for ranking",
)
```

Thread the flags into `run_search_report(...)` as `save_run_memory_path` and `reuse_run_path`, and resolve them with `resolve_run_memory_path` only after confirming they are provided.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py des_multi_agent/orchestrator.py tests/test_cli.py
git commit -m "feat: add des run memory cli flags"
```

### Task 3: Apply saved run memory to DES ranking only

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Modify: `des_multi_agent/run_memory.py`
- Test: `tests/test_run_memory.py`
- Test: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.run_memory import parse_run_memory, apply_run_memory_preferences
from des_multi_agent.uncertainty.schemas import AnnotatedResult, MinimumTmUncertainty


def _make_annotated_result(smiles_b: str, score: float) -> AnnotatedResult:
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b=smiles_b,
        ratios=[0.5],
        tm_pred_k=[200.0],
        t1_k=300.0,
        t2_k=250.0,
        checkpoint_path="ckpt.pt",
    )
    result = DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="ok",
        min_tm_k=200.0,
    )
    uncertainty = MinimumTmUncertainty(
        component_a="CCO",
        component_b=smiles_b,
        repeated_values=(200.0,),
        mean_tm_k=200.0,
        std_tm_k=1.0,
        min_tm_k=200.0,
        max_tm_k=200.0,
        trust_score=score,
        uncertainty_flag="low",
        explanation="ok",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )
    return AnnotatedResult(
        result=result,
        uncertainty=uncertainty,
        trust_score=score,
        ranking_score=score,
    )


def test_run_memory_bumps_preferred_candidate():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
            "ranked_candidates": [],
        }
    )
    adjusted, notes = apply_run_memory_preferences(
        annotated_results=[
            _make_annotated_result("CC(=O)O", 0.70),
            _make_annotated_result("O", 0.60),
        ],
        memory=memory,
        component_a="CCO",
    )
    assert adjusted[0].result.curve.smiles_b == "O"
    assert notes == ["Applied reuse memory to 1 preferred candidate and 0 penalized candidates."]


def test_run_memory_skips_different_component_a():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
            "ranked_candidates": [],
        }
    )
    original = [
        _make_annotated_result("CC(=O)O", 0.70),
        _make_annotated_result("O", 0.60),
    ]
    adjusted, notes = apply_run_memory_preferences(
        annotated_results=original,
        memory=memory,
        component_a="CCN",
    )
    assert adjusted == original
    assert notes == ["Reuse memory ignored because it was recorded for CCO, not CCN."]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_memory.py::test_run_memory_bumps_preferred_candidate -q`
Expected: FAIL because `apply_run_memory_preferences` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def apply_run_memory_preferences(annotated_results, memory, component_a: str):
    if memory is None:
        return list(annotated_results), []
    if memory.component_a is not None and memory.component_a != component_a:
        return list(annotated_results), [
            f"Reuse memory ignored because it was recorded for {memory.component_a}, not {component_a}."
        ]
    preferred = {item.smiles_b for item in memory.labels if item.label == "good"}
    penalized = {item.smiles_b for item in memory.labels if item.label == "bad"}
    adjusted = []
    for item in annotated_results:
        smiles_b = item.result.curve.smiles_b
        bonus = 0.15 if smiles_b in preferred else 0.0
        penalty = 0.15 if smiles_b in penalized else 0.0
        adjusted.append(replace(item, ranking_score=item.ranking_score + bonus - penalty))
    notes = [
        f"Applied reuse memory to {len(preferred)} preferred candidate and {len(penalized)} penalized candidates."
    ]
    return rank_annotated_results(adjusted), notes
```

Load memory only for DES runs, apply preferences after uncertainty ranking, and keep filtering behavior unchanged.

Add `memory_notes: list[str] = field(default_factory=list)` to `SearchOutcome` in `des_multi_agent/orchestrator.py`, and add an optional `memory_notes` parameter to `format_report(...)` in `des_multi_agent/reporting.py` so the CLI can print a small reuse note when memory affects ranking.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_memory.py tests/test_demo_des_search.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/reporting.py des_multi_agent/run_memory.py tests/test_run_memory.py tests/test_demo_des_search.py
git commit -m "feat: reuse des run memory for ranking"
```

### Task 4: Document the new save/reuse flow and update examples

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Modify: `examples/plain_language_gemma4_12b/README.md`
- Modify: `examples/plain_language_metal_binding_gemma4_12b/README.md`
- Modify: `examples/lidocaine_gemma4_12b/README.md`

- [ ] **Step 1: Write the failing documentation check**

```python
from pathlib import Path


def test_readme_mentions_run_memory_flags():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "--save-run-memory" in text
    assert "--reuse-run" in text
    assert "run.memory.json" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_readme_mentions_run_memory_flags -q`
Expected: FAIL until the new flags are documented.

- [ ] **Step 3: Write minimal documentation updates**

Add a short save/reuse example like:

```bash
python -m des_multi_agent.cli --workflow des --component-a CCO --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --save-run-memory runs/run_001/run.memory.json
python -m des_multi_agent.cli --workflow des --component-a CCO --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --reuse-run runs/run_001/run.memory.json
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md examples/plain_language_gemma4_12b/README.md examples/plain_language_metal_binding_gemma4_12b/README.md examples/lidocaine_gemma4_12b/README.md
git commit -m "docs: document des run memory reuse"
```

### Task 5: Prove the reuse layer does not change base DES behavior

**Files:**
- Test: `tests/test_run_memory.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.orchestrator import run_search_report


def test_des_without_reuse_still_runs_without_memory():
    outcome = run_search_report(
        component_a="CCO",
        n=3,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        config_path="ml_des_mp/config.yaml",
    )
    assert outcome.results
    assert outcome.memory_notes == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_memory.py::test_des_without_reuse_still_runs_without_memory -q`
Expected: FAIL until the regression is added.

- [ ] **Step 3: Write minimal implementation**

Keep the memory feature opt-in. If no save or reuse path is supplied, the DES run should behave exactly as it did before these changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_memory.py tests/test_cli.py tests/test_demo_des_search.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_run_memory.py tests/test_cli.py tests/test_demo_des_search.py
git commit -m "test: lock des run memory regression coverage"
```
