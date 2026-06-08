# DES Run Memory Implementation Design

**Goal:** Add an offline, same-workflow memory layer for DES runs so a user can save one run and optionally reuse it in a later DES run to influence ranking, without changing the core melting-temperature prediction pipeline.

**Architecture:** Each DES run will write a compact JSON memory file that captures the run summary, ranked candidates, uncertainty summary, provenance, and any user labels. A later DES run can accept either a prior run folder or a direct memory file; the loader will read only DES memory and pass the loaded history into the ranking layer as a soft preference signal. The base predictor, discovery, uncertainty estimation, and report formatting remain unchanged except for a small reuse note in the output when memory was applied.

**Tech Stack:** Python, JSON, pytest, existing DES workflow, existing ranking and reporting code, existing CLI

---

## Task 1: Define the reusable DES memory schema

**Files:**
- Create: `des_multi_agent/run_memory.py`
- Create: `des_multi_agent/memory_schema.py`
- Test: `tests/test_run_memory.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from des_multi_agent.run_memory import load_run_memory, parse_run_memory


def test_load_run_memory_reads_des_json(tmp_path: Path):
    memory_path = tmp_path / "run.memory.json"
    memory_path.write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 20,
          "labels": [
            {"smiles_b": "O", "label": "good"},
            {"smiles_b": "CC(=O)O", "label": "bad"}
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


def test_parse_run_memory_rejects_metal_binding_workflow():
    with pytest.raises(ValueError, match="workflow must be des"):
        parse_run_memory(
            {
                "workflow": "metal-binding",
                "component_a": None,
                "n": None,
                "labels": [],
            }
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_memory.py::test_load_run_memory_reads_des_json -q`
Expected: FAIL because `des_multi_agent.run_memory` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class RunLabel:
    smiles_b: str
    label: str


@dataclass(frozen=True)
class RunMemory:
    workflow: str
    component_a: str | None
    n: int | None
    labels: list[RunLabel]


def parse_run_memory(data: Mapping[str, object]) -> RunMemory:
    if data.get("workflow") != "des":
        raise ValueError("run memory workflow must be des")
    labels: list[RunLabel] = []
    for item in data.get("labels", []):
        labels.append(RunLabel(smiles_b=item["smiles_b"], label=item["label"]))
    return RunMemory(
        workflow="des",
        component_a=data.get("component_a"),
        n=data.get("n"),
        labels=labels,
    )


def load_run_memory(path: str | Path) -> RunMemory:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_run_memory(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_memory.py::test_load_run_memory_reads_des_json -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/run_memory.py des_multi_agent/memory_schema.py tests/test_run_memory.py
git commit -m "feat: add des run memory schema"
```

## Task 2: Write the DES run memory file after a run completes

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Modify: `des_multi_agent/run_memory.py`
- Test: `tests/test_run_memory.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.run_memory import load_run_memory, parse_run_memory, write_run_memory


def test_write_run_memory_round_trips(tmp_path: Path):
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
        }
    )
    memory_path = tmp_path / "run.memory.json"
    write_run_memory(memory_path, memory)

    loaded = load_run_memory(memory_path)
    assert loaded == memory
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_memory.py::test_write_run_memory_round_trips -q`
Expected: FAIL because `write_run_memory` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import asdict
from pathlib import Path
import json


def write_run_memory(path: str | Path, memory) -> Path:
    path = Path(path)
    path.write_text(json.dumps(asdict(memory), indent=2, sort_keys=True), encoding="utf-8")
    return path
```

Update the DES workflow to call the writer after ranking is complete and after any labels have been collected.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_memory.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/reporting.py des_multi_agent/run_memory.py tests/test_run_memory.py
git commit -m "feat: persist des run memory"
```

## Task 3: Load prior run memory into ranking

**Files:**
- Modify: `des_multi_agent/ranking.py`
- Modify: `des_multi_agent/cli.py`
- Modify: `des_multi_agent/orchestrator.py`
- Test: `tests/test_run_memory.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.run_memory import parse_run_memory
from des_multi_agent.ranking import apply_run_memory_ranking


def test_memory_influences_ranking():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
        }
    )
    ranked = apply_run_memory_ranking(["O", "CC(=O)O"], memory)
    assert ranked[0] == "O"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_memory.py::test_memory_influences_ranking -q`
Expected: FAIL because memory-based ranking does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def apply_run_memory_ranking(candidate_smiles: list[str], memory: RunMemory | None) -> list[str]:
    if memory is None:
        return list(candidate_smiles)
    preferred = {item.smiles_b for item in memory.labels if item.label == "good"}
    penalized = {item.smiles_b for item in memory.labels if item.label == "bad"}
    return sorted(
        candidate_smiles,
        key=lambda smiles: (
            0 if smiles in preferred else 1 if smiles in penalized else 2,
            candidate_smiles.index(smiles),
        ),
    )
```

Wire the DES orchestrator so the loaded run memory is only used to adjust ranking order and never to filter out candidates automatically.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_memory.py tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/ranking.py des_multi_agent/cli.py des_multi_agent/orchestrator.py tests/test_run_memory.py tests/test_cli.py
git commit -m "feat: reuse prior des run memory"
```

## Task 4: Document the reuse flag and update examples

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


def test_docs_mention_run_memory():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "--reuse-run" in text
    assert "run.memory.json" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_docs_mention_run_memory -q`
Expected: FAIL until the new reuse flag is documented.

- [ ] **Step 3: Write minimal documentation updates**

Add a short reuse example like:

```bash
python -m des_multi_agent.cli --workflow des --component-a CCO --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --reuse-run runs/run_001
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md examples/plain_language_gemma4_12b/README.md examples/plain_language_metal_binding_gemma4_12b/README.md examples/lidocaine_gemma4_12b/README.md
git commit -m "docs: document des run memory reuse"
```

## Task 5: Verify the reuse layer does not change base DES behavior

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_memory.py::test_des_without_reuse_still_runs_without_memory -q`
Expected: FAIL until the regression is added.

- [ ] **Step 3: Write minimal implementation**

Keep the memory feature opt-in. If no reuse path is supplied, the DES run should behave exactly as it did before these changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_memory.py tests/test_cli.py tests/test_demo_des_search.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_run_memory.py tests/test_cli.py tests/test_demo_des_search.py
git commit -m "test: lock des run memory regression coverage"
```
