# DES Multi-Agent System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic multi-agent workflow that takes a user-proposed chemical component `A`, proposes `N` plausible partner components `B` from chemistry rules, scores each pair with the trained `ml_des_mp` model across molar fractions `0.1` to `0.9`, and classifies likely deep eutectic solvents using both an absolute melting-point cutoff and a relative reduction threshold.

**Architecture:** Add a small new orchestration package next to the existing `ml_des_mp` backend. The orchestration package will own candidate generation, RDKit validation, batch prediction, DES classification, and ranking/reporting. It will call into `ml_des_mp` for embeddings and thermodynamic inference instead of duplicating model logic. The first version stays deterministic and local-only so results are reproducible and easy to test.

**Tech Stack:** Python, RDKit, PyTorch, the existing `ml_des_mp` package and checkpoints, and `pytest` for verification.

---

### Task 1: Define the orchestration package contract

**Files:**
- Create: `des_multi_agent/__init__.py`
- Create: `des_multi_agent/schemas.py`
- Create: `des_multi_agent/config.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.schemas import DesThresholds, CandidateProposal, MeltingPointEstimate


def test_thresholds_and_candidate_schema_round_trip():
    thresholds = DesThresholds(absolute_tm_max_k=300.0, relative_drop_min=0.25)
    proposal = CandidateProposal(
        smiles="O",
        rationale="small hydrogen-bond donor",
        family="alcohol",
    )
    estimate = MeltingPointEstimate(component="O", tm_k=273.15, source="heuristic", confidence=0.5)

    assert thresholds.absolute_tm_max_k == 300.0
    assert thresholds.relative_drop_min == 0.25
    assert proposal.smiles == "O"
    assert proposal.family == "alcohol"
    assert estimate.tm_k == 273.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL because `des_multi_agent.schemas` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateProposal:
    smiles: str
    rationale: str
    family: str


@dataclass(frozen=True)
class MeltingPointEstimate:
    component: str
    tm_k: float
    source: str
    confidence: float


@dataclass(frozen=True)
class DesThresholds:
    absolute_tm_max_k: float
    relative_drop_min: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/__init__.py des_multi_agent/schemas.py des_multi_agent/config.py tests/test_schemas.py
git commit -m "feat: define multi-agent orchestration contract"
```

### Task 2: Implement candidate generation and chemistry filtering

**Files:**
- Create: `des_multi_agent/candidate_generation.py`
- Create: `des_multi_agent/chemistry_filter.py`
- Create: `tests/test_candidate_generation.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.candidate_generation import generate_candidates
from des_multi_agent.chemistry_filter import filter_candidates


def test_generation_and_filtering_returns_plausible_smiles():
    proposals = generate_candidates("CCO", n=5, constraints=None)
    filtered = filter_candidates("CCO", proposals)

    assert len(proposals) >= 5
    assert all(p.smiles for p in proposals)
    assert all(p.smiles != "CCO" for p in filtered)
    assert len({p.smiles for p in filtered}) == len(filtered)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_generation.py -v`
Expected: FAIL because generation and filtering modules are missing.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from rdkit import Chem

from .schemas import CandidateProposal


def generate_candidates(component_a: str, n: int, constraints=None):
    families = [
        ("hydrogen-bond donor", "alcohol"),
        ("hydrogen-bond acceptor", "amide"),
        ("hydrogen-bond acceptor", "polyol"),
        ("hydrogen-bond donor", "carboxylic acid"),
        ("ionic partner", "quaternary ammonium salt"),
    ]
    smiles_pool = ["O", "CO", "CCO", "CC(=O)N", "OC(=O)C", "C[N+](C)(C)C.[Cl-]"]
    proposals = []
    for i in range(n):
        family = families[i % len(families)][1]
        smiles = smiles_pool[i % len(smiles_pool)]
        proposals.append(CandidateProposal(smiles=smiles, rationale=f"rule-based {family}", family=family))
    return proposals


def filter_candidates(component_a: str, candidates):
    out = []
    seen = set()
    mol_a = Chem.MolFromSmiles(component_a)
    if mol_a is None:
        raise ValueError("Invalid component A SMILES")
    for proposal in candidates:
        mol_b = Chem.MolFromSmiles(proposal.smiles)
        if mol_b is None:
            continue
        if proposal.smiles == component_a or proposal.smiles in seen:
            continue
        seen.add(proposal.smiles)
        out.append(proposal)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_generation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/candidate_generation.py des_multi_agent/chemistry_filter.py tests/test_candidate_generation.py
git commit -m "feat: add rule-based candidate generation"
```

### Task 3: Wrap ml_des_mp prediction for batch pair scoring

**Files:**
- Create: `des_multi_agent/prediction.py`
- Create: `tests/test_prediction_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.prediction import build_ratio_grid


def test_ratio_grid_covers_requested_range():
    grid = build_ratio_grid()
    assert grid[0] == 0.1
    assert grid[-1] == 0.9
    assert len(grid) >= 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction_adapter.py -v`
Expected: FAIL because `des_multi_agent.prediction` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ml_des_mp.predict import load_model
from ml_des_mp.src.embeddings.factory import build_embedder
from ml_des_mp.src.train import _predict_Tm_from_params
from ml_des_mp.src.utils import get_device


@dataclass(frozen=True)
class CurvePrediction:
    smiles_a: str
    smiles_b: str
    ratios: list[float]
    tm_pred_k: list[float]
    t1_k: float
    t2_k: float
    checkpoint_path: str


def build_ratio_grid():
    return [round(x, 3) for x in np.linspace(0.1, 0.9, 9).tolist()]


def predict_curve(component_a: str, component_b: str, t1_k: float, t2_k: float, checkpoint_path: str, config_path: str):
    # Reuse the existing embedding and thermodynamic pipeline from ml_des_mp.
    import yaml
    from rdkit import Chem
    import torch

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = get_device(cfg.get("device", "cuda"))
    model = load_model(checkpoint_path, device)
    emb_bundle = build_embedder(cfg["embedding"], device=device)
    if emb_bundle.kind == "gnn":
        raise ValueError("GNN checkpoints require an end-to-end inference path")
    x1 = torch.tensor(emb_bundle.embedder.embed([component_a]), device=device)
    x2 = torch.tensor(emb_bundle.embedder.embed([component_b]), device=device)
    t1 = torch.tensor([t1_k], device=device)
    t2 = torch.tensor([t2_k], device=device)
    ratios = build_ratio_grid()
    tm_pred_k = []
    with torch.no_grad():
        d1, d2, w = model.forward_params(x1, x2)
        for ratio in ratios:
            r = torch.tensor([ratio], device=device)
            tm_pred_k.append(float(_predict_Tm_from_params(d1, d2, w, t1, t2, r).item()))
    return CurvePrediction(
        smiles_a=component_a,
        smiles_b=component_b,
        ratios=ratios,
        tm_pred_k=tm_pred_k,
        t1_k=t1_k,
        t2_k=t2_k,
        checkpoint_path=checkpoint_path,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/prediction.py tests/test_prediction_adapter.py
git commit -m "feat: add prediction adapter for ml_des_mp"
```

### Task 4: Implement DES classification and ranking

**Files:**
- Create: `des_multi_agent/evaluation.py`
- Create: `des_multi_agent/ranking.py`
- Create: `tests/test_des_evaluation.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.evaluation import classify_des
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import DesThresholds


def test_des_classification_requires_both_thresholds():
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b="O",
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[250.0, 245.0, 255.0],
        t1_k=298.0,
        t2_k=273.0,
        checkpoint_path="ckpt.pt",
    )
    thresholds = DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.10)
    result = classify_des(curve, thresholds)

    assert result.is_des is True
    assert result.absolute_pass is True
    assert result.relative_pass is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_des_evaluation.py -v`
Expected: FAIL because evaluation module is missing.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass

from .prediction import CurvePrediction
from .schemas import DesThresholds


@dataclass(frozen=True)
class DesResult:
    curve: CurvePrediction
    absolute_pass: bool
    relative_pass: bool
    is_des: bool
    rationale: str
    min_tm_k: float


def classify_des(curve: CurvePrediction, thresholds: DesThresholds) -> DesResult:
    min_tm = min(curve.tm_pred_k)
    absolute_pass = min_tm <= thresholds.absolute_tm_max_k
    baseline = min(curve.t1_k, curve.t2_k)
    relative_drop = (baseline - min_tm) / baseline if baseline else 0.0
    relative_pass = relative_drop >= thresholds.relative_drop_min
    is_des = absolute_pass and relative_pass
    rationale = (
        f"min Tm={min_tm:.2f} K, absolute<= {thresholds.absolute_tm_max_k:.2f} K, "
        f"relative_drop={relative_drop:.3f}"
    )
    return DesResult(
        curve=curve,
        absolute_pass=absolute_pass,
        relative_pass=relative_pass,
        is_des=is_des,
        rationale=rationale,
        min_tm_k=min_tm,
    )
```

```python
from dataclasses import dataclass

from .evaluation import DesResult


def rank_results(results: list[DesResult]) -> list[DesResult]:
    return sorted(
        results,
        key=lambda r: (
            not r.is_des,
            r.min_tm_k,
            -sum(r.curve.tm_pred_k) / len(r.curve.tm_pred_k),
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_des_evaluation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/evaluation.py des_multi_agent/ranking.py tests/test_des_evaluation.py
git commit -m "feat: classify and rank candidate DES pairs"
```

### Task 5: Build the top-level orchestrator and result report

**Files:**
- Create: `des_multi_agent/orchestrator.py`
- Create: `des_multi_agent/reporting.py`
- Create: `des_multi_agent/property_resolution.py`
- Create: `des_multi_agent/cli.py`
- Create: `tests/test_orchestrator.py`
- Create: `tests/test_property_resolution.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.property_resolution import resolve_melting_point


def test_resolve_melting_point_accepts_override():
    estimate = resolve_melting_point("CCO", override_k=310.0)
    assert estimate.tm_k == 310.0
    assert estimate.source == "override"
```

```python
from des_multi_agent import orchestrator
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate


def test_orchestrator_returns_ranked_results(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda comp, override_k=None: MeltingPointEstimate(component=comp, tm_k=298.15, source="heuristic", confidence=0.5),
    )

    def fake_predict_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml"):
        return CurvePrediction(
            smiles_a=component_a,
            smiles_b=component_b,
            ratios=[0.1, 0.5, 0.9],
            tm_pred_k=[250.0, 245.0, 255.0],
            t1_k=t1_k,
            t2_k=t2_k,
            checkpoint_path=checkpoint_path,
        )

    monkeypatch.setattr(orchestrator, "predict_curve", fake_predict_curve)
    results = orchestrator.run_search(component_a="CCO", n=3, checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt")
    assert len(results) > 0
    assert all(hasattr(result, "rationale") for result in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL because orchestrator does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from .candidate_generation import generate_candidates
from .chemistry_filter import filter_candidates
from .evaluation import classify_des
from .prediction import predict_curve
from .property_resolution import resolve_melting_point
from .ranking import rank_results
from .schemas import DesThresholds


def run_search(component_a: str, n: int, checkpoint_path: str, config_path: str = "ml_des_mp/config.yaml"):
    proposals = generate_candidates(component_a, n=n, constraints=None)
    filtered = filter_candidates(component_a, proposals)
    thresholds = DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.10)
    component_a_tp = resolve_melting_point(component_a)
    results = []
    for proposal in filtered:
        component_b_tp = resolve_melting_point(proposal.smiles)
        curve = predict_curve(
            component_a,
            proposal.smiles,
            t1_k=component_a_tp.tm_k,
            t2_k=component_b_tp.tm_k,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        result = classify_des(curve, thresholds)
        results.append(result)
    return rank_results(results)
```

```python
def format_report(results):
    lines = ["smiles_b | is_des | min_tm_k | rationale"]
    for r in results:
        lines.append(f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {r.rationale}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/reporting.py des_multi_agent/property_resolution.py des_multi_agent/cli.py tests/test_orchestrator.py tests/test_property_resolution.py
git commit -m "feat: wire the DES search orchestrator"
```

### Task 6: Add an end-to-end CLI and documentation

**Files:**
- Modify: `ml_des_mp/README.md`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.cli import build_parser


def test_cli_parser_accepts_component_a_and_n():
    parser = build_parser()
    args = parser.parse_args(["--component-a", "CCO", "--n", "5"])
    assert args.component_a == "CCO"
    assert args.n == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL because CLI module is missing.

- [ ] **Step 3: Write minimal implementation**

```python
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-a", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    return parser
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Update docs and verify the full user flow**

Add a short usage section to `ml_des_mp/README.md` showing:

```bash
python -m des_multi_agent.cli --component-a "CCO" --n 10 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
```

Run: `pytest`
Expected: all tests pass, including the end-to-end orchestration and CLI parser tests.

- [ ] **Step 6: Commit**

```bash
git add ml_des_mp/README.md des_multi_agent/cli.py tests/test_cli.py
git commit -m "docs: add DES multi-agent usage"
```

## Verification Checklist

Before merging or publishing, verify all of the following:

- `pytest tests/test_schemas.py -v`
- `pytest tests/test_candidate_generation.py -v`
- `pytest tests/test_prediction_adapter.py -v`
- `pytest tests/test_des_evaluation.py -v`
- `pytest tests/test_orchestrator.py -v`
- `pytest tests/test_cli.py -v`
- `pytest`

## Notes for the Implementer

- Keep the candidate generator deterministic in v1 so repeated requests are reproducible.
- Keep all threshold values configurable so the definition of DES can be tuned without code changes.
- Reuse `ml_des_mp`'s existing embedding and prediction logic rather than re-deriving the thermodynamic equations.
- If the available checkpoint uses GNN embeddings, either keep the first version on a fixed-feature checkpoint or add a dedicated GNN inference helper as a follow-up task.

