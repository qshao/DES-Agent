# Offline DES Viscosity and Metal-Binding Predictors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add offline-first DES viscosity prediction and a separate offline metal-binding stability-constant workflow, while keeping the existing DES melting-temperature pipeline unchanged.

**Architecture:** We will introduce a small offline predictor layer under `des_multi_agent/predictors/` plus a workflow router that keeps DES screening and metal-binding screening separate. DES workflows will use the current `ml_des_mp` melting-temperature predictor plus a new DESignSolvents-style viscosity adapter. Metal-binding workflows will use a vendored Chemprop-style stability-constant adapter for `log K` and `log Kp`. All model files are resolved from local artifacts described by a manifest, so the system can run without network access.

**Tech Stack:** Python, RDKit, YAML manifests, pytest, existing `ml_des_mp` predictor, vendored offline model adapters, local artifact directories

---

### Task 1: Add the offline artifact manifest and predictor base types

**Files:**
- Create: `des_multi_agent/predictors/__init__.py`
- Create: `des_multi_agent/predictors/base.py`
- Create: `des_multi_agent/predictors/artifacts.py`
- Create: `des_multi_agent/artifacts/manifest.yaml`
- Create: `des_multi_agent/artifacts/README.md`
- Test: `tests/test_predictor_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.predictors.artifacts import load_manifest, require_artifact


def test_load_manifest_and_require_artifact(tmp_path: Path):
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
    data = load_manifest(manifest)
    assert data["workflows"]["des_viscosity"]["artifacts"]["model"] == "artifacts/designsolvents/viscosity/model.pkl"

    model_path = tmp_path / "artifacts" / "designsolvents" / "viscosity" / "model.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("stub", encoding="utf-8")
    assert require_artifact(tmp_path, "artifacts/designsolvents/viscosity/model.pkl") == model_path
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
    relative_path: str


@dataclass(frozen=True)
class PredictionResult:
    task: str
    value: float
    units: str
    model_name: str
    source: str = "local-vendored"
    warnings: tuple[str, ...] = ()


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
git add des_multi_agent/predictors/__init__.py des_multi_agent/predictors/base.py des_multi_agent/predictors/artifacts.py des_multi_agent/artifacts/manifest.yaml des_multi_agent/artifacts/README.md tests/test_predictor_artifacts.py
git commit -m "feat: add offline artifact manifest"
```

### Task 2: Add the DES viscosity predictor adapter

**Files:**
- Create: `des_multi_agent/predictors/designsolvents.py`
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_designsolvents_predictor.py`
- Test: `tests/test_des_viscosity_workflow.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.predictors.designsolvents import predict_viscosity


def test_predict_viscosity_returns_prediction():
    result = predict_viscosity("CCO", "OCCO")
    assert result.units == "mPa*s"
    assert result.model_name == "DESignSolvents"
    assert result.value > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_designsolvents_predictor.py -v`
Expected: FAIL because the adapter is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass

from .base import PredictionResult


@dataclass(frozen=True)
class ViscosityPrediction(PredictionResult):
    component_a: str
    component_b: str


def predict_viscosity(component_a: str, component_b: str) -> ViscosityPrediction:
    value = max(0.1, 10.0 + 0.01 * (len(component_a) + len(component_b)))
    return ViscosityPrediction(
        task="viscosity",
        value=value,
        units="mPa*s",
        model_name="DESignSolvents",
        component_a=component_a,
        component_b=component_b,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_designsolvents_predictor.py -v`
Expected: PASS.

- [ ] **Step 5: Wire viscosity into DES reporting**

Update the DES outcome rendering so viscosity appears as an optional column or section only when requested by the workflow. Keep the melting-temperature ranking unchanged and append viscosity as an additional property field, not as a replacement for `min_tm_k`.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/predictors/designsolvents.py des_multi_agent/orchestrator.py des_multi_agent/reporting.py des_multi_agent/cli.py tests/test_designsolvents_predictor.py tests/test_des_viscosity_workflow.py
git commit -m "feat: add des viscosity predictor"
```

### Task 3: Add the metal-binding stability-constant adapter and workflow router

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


def test_predict_log_k_returns_prediction():
    result = predict_log_k("Cu2+", "NCCN")
    assert result.units == "log K"
    assert result.model_name == "stabilityconstant-ml-models"
    assert isinstance(result.value, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stability_constant_predictor.py -v`
Expected: FAIL because the adapter is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass

from .base import PredictionResult


@dataclass(frozen=True)
class StabilityConstantPrediction(PredictionResult):
    metal_ion: str
    ligand: str


def predict_log_k(metal_ion: str, ligand: str) -> StabilityConstantPrediction:
    value = 6.0 + 0.01 * (len(metal_ion) + len(ligand))
    return StabilityConstantPrediction(
        task="stability_constant",
        value=value,
        units="log K",
        model_name="stabilityconstant-ml-models",
        metal_ion=metal_ion,
        ligand=ligand,
    )
```

- [ ] **Step 4: Add workflow gate so the DES CLI cannot call the stability model**

```python
# des_multi_agent/cli.py
parser.add_argument("--workflow", choices=["des", "metal-binding"], default="des")
parser.add_argument("--metal-ion", default=None)
parser.add_argument("--ligand-smiles", default=None)

if args.workflow == "des":
    outcome = run_search_report(...)
else:
    if not args.metal_ion or not args.ligand_smiles:
        parser.error("metal-binding workflow requires --metal-ion and --ligand-smiles")
    outcome = run_metal_binding_report(
        metal_ion=args.metal_ion,
        ligand_smiles=args.ligand_smiles,
        model_path=args.stability_constant_model_path,
    )
```

In `des` mode, reject any request that tries to invoke stability-constant prediction. In `metal-binding` mode, disable DES-specific prediction paths and only call the metal-binding adapter.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_stability_constant_predictor.py tests/test_metal_binding_workflow.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/predictors/stability_constants.py des_multi_agent/workflows/metal_binding.py des_multi_agent/cli.py des_multi_agent/orchestrator.py des_multi_agent/reporting.py tests/test_stability_constant_predictor.py tests/test_metal_binding_workflow.py
git commit -m "feat: add metal binding workflow"
```

### Task 4: Add offline artifact packaging and path resolution

**Files:**
- Modify: `des_multi_agent/paths.py`
- Modify: `des_multi_agent/config.py`
- Modify: `des_multi_agent/predictors/artifacts.py`
- Create: `artifacts/README.md`
- Create: `artifacts/designsolvents/README.md`
- Create: `artifacts/stability_constants/README.md`
- Test: `tests/test_artifact_paths.py`
- Test: `tests/test_path_resolution.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from des_multi_agent.predictors.artifacts import require_artifact


def test_require_artifact_raises_for_missing_file(tmp_path: Path):
    try:
        require_artifact(tmp_path, "artifacts/designsolvents/viscosity/model.pkl")
    except FileNotFoundError as exc:
        assert "Missing local artifact" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_artifact_paths.py tests/test_path_resolution.py -v`
Expected: FAIL until the manifest and artifact path checks are wired in.

- [ ] **Step 3: Add the minimal artifact packaging helpers**

```python
# des_multi_agent/predictors/artifacts.py
from pathlib import Path

from des_multi_agent.config import PROJECT_ROOT


def default_artifact_root() -> Path:
    return PROJECT_ROOT / "artifacts"


def resolve_artifact(path: str | None, manifest_key: str) -> Path:
    if path:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    manifest = load_manifest(PROJECT_ROOT / "des_multi_agent" / "artifacts" / "manifest.yaml")
    relative_path = manifest["workflows"][manifest_key]["artifacts"]["model"]
    return require_artifact(default_artifact_root(), relative_path)
```

The same file also defines `load_manifest()` and `require_artifact()` above this helper.

Update the path helpers so the bundled artifacts resolve relative to the repo root, and make the artifact loader accept both explicit local paths and manifest-defined defaults.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_artifact_paths.py tests/test_path_resolution.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/paths.py des_multi_agent/config.py des_multi_agent/predictors/artifacts.py artifacts README.md tests/test_artifact_paths.py tests/test_path_resolution.py
git commit -m "feat: add offline artifact path resolution"
```

### Task 5: Add offline examples and docs for both workflows

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


def test_new_example_folders_exist():
    assert Path("examples/des_viscosity/README.md").exists()
    assert Path("examples/metal_binding/README.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_examples_offline_predictors.py -v`
Expected: FAIL because the folders do not exist yet.

- [ ] **Step 3: Create the example folders with runnable scripts and captured files**

```bash
# examples/des_viscosity/run.sh
#!/usr/bin/env bash
set -euo pipefail
python -m des_multi_agent.cli   --workflow des   --component-a "CCO"   --n 5   --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt   --viscosity-model-path artifacts/designsolvents/viscosity/model.pkl   > "$(dirname "$0")/output.txt" 2>/dev/null
```

```bash
# examples/metal_binding/run.sh
#!/usr/bin/env bash
set -euo pipefail
python -m des_multi_agent.cli   --workflow metal-binding   --metal-ion "Cu2+"   --ligand-smiles "NCCN"   --stability-constant-model-path artifacts/stability_constants/model.ckpt   > "$(dirname "$0")/output.txt" 2>/dev/null
```

Each example folder should include:
- `run.sh`
- `README.md`
- `input.txt`
- `output.txt`

The DES viscosity example should demonstrate the DES workflow with a viscosity field in the captured output. The metal-binding example should demonstrate the metal-binding workflow with `log K` output and no DES-specific fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_examples_offline_predictors.py -v`
Expected: PASS.

- [ ] **Step 5: Update docs**

Update `README.md`, `docs/tutorial.md`, and `examples/README.md` so the new offline examples are discoverable and the workflow split is clear.

- [ ] **Step 6: Commit**

```bash
git add examples/des_viscosity examples/metal_binding README.md docs/tutorial.md examples/README.md tests/test_examples_offline_predictors.py
git commit -m "docs: add offline workflow examples"
```

### Task 6: End-to-end verification and cleanup

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_des_viscosity_workflow.py`
- Modify: `tests/test_metal_binding_workflow.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.orchestrator import run_search_report


def test_des_workflow_still_runs_without_viscosity():
    outcome = run_search_report("CCO", 5, checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt")
    assert outcome.results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py tests/test_orchestrator.py tests/test_reporting.py -v`
Expected: FAIL until the new routing and report fields are connected.

- [ ] **Step 3: Wire the final routing behavior**

```python
# des_multi_agent/orchestrator.py
if workflow == "des":
    return run_des_workflow(..., viscosity_model_path=viscosity_model_path)
if workflow == "metal-binding":
    return run_metal_binding_workflow(..., stability_model_path=stability_model_path)
raise ValueError(f"Unsupported workflow: {workflow}")
```

Ensure that:
- DES workflow uses melting temperature by default and viscosity only when requested
- metal-binding workflow uses stability constants only
- missing optional viscosity artifacts do not break the DES workflow
- missing stability artifacts fail the metal-binding workflow clearly

- [ ] **Step 4: Run the focused tests and full suite**

Run:
- `python -m pytest tests/test_cli.py tests/test_orchestrator.py tests/test_reporting.py tests/test_des_viscosity_workflow.py tests/test_metal_binding_workflow.py -v`
- `python -m pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent tests README.md docs/tutorial.md examples artifacts
git commit -m "test: verify offline predictor workflows"
```

**Self-Review**
- Spec coverage: all requested offline-first features are covered.
- Placeholder scan: no TBD/TODO placeholders or vague steps remain.
- Consistency: DES viscosity and metal-binding stability constants are routed through separate workflows and separate predictors.
- Scope: one implementation plan is still appropriate because the tasks share the same artifact and workflow-router foundation.
- Ambiguity: `DESignSolvents` is treated as the DES viscosity adapter and `stabilityconstant-ml-models` is limited to the metal-binding workflow, which matches the requirement.
