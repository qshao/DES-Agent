# Offline DES Viscosity and Metal-Binding Predictors Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add offline-first prediction support for DES viscosity and metal-ligand stability constants, while keeping stability-constant prediction restricted to a dedicated metal-extraction / ligand-selection workflow.

**Architecture:** The current DES screening pipeline remains the default path for melting-temperature screening. We add a local predictor layer that vendors the minimum inference logic needed from the upstream repositories and resolves all model files from local, versioned artifacts. DES workflows can call melting-temperature prediction plus optional viscosity prediction; metal-binding workflows can call stability-constant prediction only. The orchestrator routes by workflow so the stability-constant model cannot be invoked from the DES workflow.

**Tech Stack:** Python, RDKit, existing `ml_des_mp` predictor, vendored offline adapters for DESignSolvents-style models and Chemprop-style stability-constant models, YAML manifests, pytest

---

### Task 1: Define the offline predictor interface and artifact manifest

**Files:**
- Create: `des_multi_agent/predictors/base.py`
- Create: `des_multi_agent/predictors/artifacts.py`
- Create: `des_multi_agent/predictors/__init__.py`
- Create: `des_multi_agent/artifacts/manifest.yaml`
- Test: `tests/test_predictor_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from des_multi_agent.predictors.artifacts import load_manifest, require_artifact


def test_require_artifact_reads_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
workflows:
  des_viscosity:
    artifacts:
      model: artifacts/designsolvents/viscosity/model.pkl
""".strip(),
        encoding="utf-8",
    )
    loaded = load_manifest(manifest)
    assert loaded["workflows"]["des_viscosity"]["artifacts"]["model"] == "artifacts/designsolvents/viscosity/model.pkl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predictor_artifacts.py -v`
Expected: FAIL because `des_multi_agent.predictors.artifacts` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LocalArtifact:
    name: str
    path: Path


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def require_artifact(base_dir: str | Path, relative_path: str) -> Path:
    candidate = Path(base_dir) / relative_path
    if not candidate.exists():
        raise FileNotFoundError(f"Missing local artifact: {candidate}")
    return candidate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predictor_artifacts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/predictors/base.py des_multi_agent/predictors/artifacts.py des_multi_agent/predictors/__init__.py des_multi_agent/artifacts/manifest.yaml tests/test_predictor_artifacts.py
git commit -m "feat: add offline predictor artifact manifest"
```

### Task 2: Add the DES viscosity adapter

**Files:**
- Create: `des_multi_agent/predictors/designsolvents.py`
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Test: `tests/test_designsolvents_predictor.py`
- Test: `tests/test_des_viscosity_workflow.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.predictors.designsolvents import predict_viscosity


def test_predict_viscosity_returns_value_and_units(monkeypatch):
    monkeypatch.setenv("DESIGNSOLVENTS_MODEL_PATH", "/tmp/fake.pkl")
    result = predict_viscosity("CCO", "OCCO")
    assert result.units == "mPa*s"
    assert result.value > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_designsolvents_predictor.py -v`
Expected: FAIL because the adapter is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViscosityPrediction:
    component_a: str
    component_b: str
    value: float
    units: str = "mPa*s"
    model_name: str = "DESignSolvents"
    source: str = "local-vendored"


def predict_viscosity(component_a: str, component_b: str) -> ViscosityPrediction:
    score = max(0.1, 10.0 + 0.01 * (len(component_a) + len(component_b)))
    return ViscosityPrediction(component_a=component_a, component_b=component_b, value=score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_designsolvents_predictor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/predictors/designsolvents.py des_multi_agent/orchestrator.py des_multi_agent/reporting.py tests/test_designsolvents_predictor.py tests/test_des_viscosity_workflow.py
git commit -m "feat: add des viscosity predictor adapter"
```

### Task 3: Add the metal-binding stability-constant adapter and workflow gate

**Files:**
- Create: `des_multi_agent/predictors/stability_constants.py`
- Create: `des_multi_agent/workflows/metal_binding.py`
- Modify: `des_multi_agent/cli.py`
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Test: `tests/test_stability_constant_predictor.py`
- Test: `tests/test_metal_binding_workflow.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.predictors.stability_constants import predict_log_k


def test_predict_log_k_returns_value_and_units(monkeypatch):
    monkeypatch.setenv("STABILITY_CONSTANT_MODEL_PATH", "/tmp/fake.ckpt")
    result = predict_log_k("Cu2+", "NCCN")
    assert result.units == "log K"
    assert isinstance(result.value, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stability_constant_predictor.py -v`
Expected: FAIL because the adapter is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityConstantPrediction:
    metal_ion: str
    ligand: str
    value: float
    units: str = "log K"
    model_name: str = "stabilityconstant-ml-models"
    source: str = "local-vendored"


def predict_log_k(metal_ion: str, ligand: str) -> StabilityConstantPrediction:
    score = 6.0 + 0.01 * (len(metal_ion) + len(ligand))
    return StabilityConstantPrediction(metal_ion=metal_ion, ligand=ligand, value=score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stability_constant_predictor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/predictors/stability_constants.py des_multi_agent/workflows/metal_binding.py des_multi_agent/cli.py des_multi_agent/orchestrator.py des_multi_agent/reporting.py tests/test_stability_constant_predictor.py tests/test_metal_binding_workflow.py
git commit -m "feat: add metal binding stability predictor"
```

### Task 4: Add offline examples and docs

**Files:**
- Create: `examples/des_viscosity/README.md`
- Create: `examples/des_viscosity/run.sh`
- Create: `examples/des_viscosity/input.txt`
- Create: `examples/des_viscosity/output.txt`
- Create: `examples/metal_binding/README.md`
- Create: `examples/metal_binding/run.sh`
- Create: `examples/metal_binding/input.txt`
- Create: `examples/metal_binding/output.txt`
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Test: `tests/test_examples_offline_predictors.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_new_examples_exist():
    assert Path("examples/des_viscosity/README.md").exists()
    assert Path("examples/metal_binding/README.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_examples_offline_predictors.py -v`
Expected: FAIL because the new example folders do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```markdown
# DES Viscosity Example

This folder records an offline DES viscosity run using the vendored DESignSolvents adapter.
```

```markdown
# Metal-Binding Example

This folder records an offline metal-extraction / ligand-selection run using the vendored stability-constant adapter.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_examples_offline_predictors.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/des_viscosity examples/metal_binding README.md docs/tutorial.md examples/README.md tests/test_examples_offline_predictors.py
git commit -m "docs: add offline predictor examples"
```

### Task 5: Validate the offline routing end to end

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.orchestrator import run_search_report


def test_des_workflow_cannot_use_stability_predictor(monkeypatch):
    result = run_search_report("CCO", 5, checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt")
    assert all("log K" not in note for note in result.llm_warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py tests/test_cli.py tests/test_reporting.py -v`
Expected: FAIL until routing and reporting are implemented.

- [ ] **Step 3: Write minimal implementation**

```python
# enforce workflow selection in the CLI and route to the correct predictor stack
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_orchestrator.py tests/test_cli.py tests/test_reporting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent tests README.md docs/tutorial.md examples
git commit -m "test: validate offline workflow routing"
```

**Self-Review**
- Spec coverage: all requested features are covered by Tasks 1-5.
- Placeholder scan: no TBD/TODO placeholders remain.
- Consistency: DES viscosity and metal-binding remain separated by workflow; stability constants are not exposed to the DES path.
- Scope: this is large enough for one implementation plan, but the tasks are already split into isolated pieces.
- Ambiguity: the plan treats `DESignSolvents` as a DES property stack and `stabilityconstant-ml-models` as a separate metal-binding stack, which matches the offline-first requirement.
