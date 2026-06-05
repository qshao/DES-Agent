# Phase 1 Uncertainty Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an uncertainty layer for the minimum melting-temperature prediction that runs the predictor three times, computes a normalized heuristic trust score, and uses both signals to filter, demote, and report candidate pairs.

**Architecture:** Keep the current deterministic DES pipeline unchanged as the source of truth for prediction and classification. Introduce a small `uncertainty/` package that owns repeated-prediction aggregation, heuristic trust scoring, and uncertainty policy application. Integrate that package in the orchestrator so uncertainty can filter or penalize candidate rankings, and surface the new fields in the CLI report without changing the DES label formula.

**Tech Stack:** Python 3.13, RDKit, the existing `des_multi_agent` and `ml_des_mp` packages, and `pytest`

---

### Task 1: Add uncertainty data structures and repeated-prediction helpers

**Files:**
- Create: `des_multi_agent/uncertainty/__init__.py`
- Create: `des_multi_agent/uncertainty/schemas.py`
- Create: `des_multi_agent/uncertainty/model.py`
- Test: `tests/test_uncertainty_model.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.uncertainty.model import estimate_min_tm_uncertainty


def test_estimate_min_tm_uncertainty_runs_three_passes(monkeypatch, tmp_path):
    calls = []

    class Curve:
        tm_pred_k = [250.0, 240.0, 245.0]

    def fake_predict_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path):
        calls.append((component_a, component_b, checkpoint_path, config_path))
        return Curve()

    monkeypatch.setattr("des_multi_agent.uncertainty.model.predict_curve", fake_predict_curve)
    estimate = estimate_min_tm_uncertainty(
        "CCO",
        "O",
        str(tmp_path / "model.pt"),
        str(tmp_path / "config.yaml"),
    )

    assert len(calls) == 3
    assert estimate.tm_min_values == [240.0, 240.0, 240.0]
    assert estimate.tm_min_mean_k == 240.0
    assert estimate.tm_min_std_k == 0.0
    assert estimate.uncertainty_flag == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uncertainty_model.py -v`
Expected: FAIL because `des_multi_agent/uncertainty/model.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from des_multi_agent.prediction import predict_curve


@dataclass(frozen=True)
class UncertaintyEstimate:
    tm_min_values: list[float]
    tm_min_mean_k: float
    tm_min_std_k: float
    tm_min_min_k: float
    tm_min_max_k: float
    trust_score: float
    uncertainty_flag: str
    explanation: str


def estimate_min_tm_uncertainty(component_a: str, component_b: str, checkpoint_path: str, config_path: str) -> UncertaintyEstimate:
    tm_min_values: list[float] = []
    for _ in range(3):
        curve = predict_curve(
            component_a,
            component_b,
            t1_k=298.15,
            t2_k=300.0,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        tm_min_values.append(min(curve.tm_pred_k))

    tm_min_mean_k = mean(tm_min_values)
    tm_min_std_k = pstdev(tm_min_values) if len(tm_min_values) > 1 else 0.0
    if tm_min_std_k < 5.0:
        uncertainty_flag = "low"
    elif tm_min_std_k < 15.0:
        uncertainty_flag = "medium"
    else:
        uncertainty_flag = "high"
    trust_score = max(0.0, min(1.0, 1.0 - tm_min_std_k / 50.0))
    explanation = f"Three repeated minimum-Tm predictions produced std={tm_min_std_k:.2f} K"

    return UncertaintyEstimate(
        tm_min_values=tm_min_values,
        tm_min_mean_k=tm_min_mean_k,
        tm_min_std_k=tm_min_std_k,
        tm_min_min_k=min(tm_min_values),
        tm_min_max_k=max(tm_min_values),
        trust_score=trust_score,
        uncertainty_flag=uncertainty_flag,
        explanation=explanation,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uncertainty_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/uncertainty/__init__.py des_multi_agent/uncertainty/schemas.py des_multi_agent/uncertainty/model.py tests/test_uncertainty_model.py
git commit -m "feat: add uncertainty estimate model"
```

### Task 2: Add heuristic trust scoring and uncertainty policy

**Files:**
- Modify: `des_multi_agent/uncertainty/schemas.py`
- Create: `des_multi_agent/uncertainty/heuristics.py`
- Create: `des_multi_agent/uncertainty/policy.py`
- Create: `des_multi_agent/uncertainty/filtering.py`
- Test: `tests/test_uncertainty_heuristics.py`

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import dataclass

from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.uncertainty.filtering import apply_uncertainty_policy
from des_multi_agent.uncertainty.heuristics import score_candidate_trust
from des_multi_agent.uncertainty.model import UncertaintyEstimate
from des_multi_agent.uncertainty.policy import UncertaintyPolicy
from des_multi_agent.uncertainty.schemas import AnnotatedResult


def _estimate(std_k: float, trust_score: float) -> UncertaintyEstimate:
    return UncertaintyEstimate(
        tm_min_values=[240.0, 240.0, 240.0],
        tm_min_mean_k=240.0,
        tm_min_std_k=std_k,
        tm_min_min_k=240.0,
        tm_min_max_k=240.0,
        trust_score=trust_score,
        uncertainty_flag="low",
        explanation="",
    )


def _result(smiles_b: str, min_tm_k: float) -> DesResult:
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k, min_tm_k, min_tm_k],
        t1_k=298.15,
        t2_k=300.0,
        checkpoint_path="mock://demo",
    )
    return DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="ok",
        min_tm_k=min_tm_k,
    )


def test_score_candidate_trust_is_clamped():
    low_spread = _estimate(2.0, 1.0)
    high_spread = _estimate(20.0, 1.0)
    low = score_candidate_trust("CCO", "OCCO", low_spread, {"component_a": "measured", "component_b": "heuristic"})
    high = score_candidate_trust("CCO", "O", high_spread, {"component_a": "heuristic", "component_b": "heuristic"})

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert low > high


def test_apply_uncertainty_policy_filters_below_threshold():
    strong = AnnotatedResult(result=_result("OCCO", 220.0), uncertainty=_estimate(2.0, 0.90), trust_score=0.90, ranking_score=220.0)
    weak = AnnotatedResult(result=_result("O", 220.0), uncertainty=_estimate(20.0, 0.20), trust_score=0.20, ranking_score=220.0)
    policy = UncertaintyPolicy(mode="filter", min_trust_score=0.5, soft_penalty_weight=0.35, std_high_threshold_k=15.0, std_medium_threshold_k=5.0)

    kept = apply_uncertainty_policy([strong, weak], {"OCCO": strong.uncertainty, "O": weak.uncertainty}, policy)

    assert [item.result.curve.smiles_b for item in kept] == ["OCCO"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uncertainty_heuristics.py -v`
Expected: FAIL because `des_multi_agent/uncertainty/heuristics.py`, `des_multi_agent/uncertainty/policy.py`, and `des_multi_agent/uncertainty/filtering.py` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from rdkit import Chem

from .model import UncertaintyEstimate
from .schemas import AnnotatedResult


@dataclass(frozen=True)
class UncertaintyPolicy:
    mode: str = "penalize"
    min_trust_score: float = 0.55
    soft_penalty_weight: float = 0.35
    std_high_threshold_k: float = 15.0
    std_medium_threshold_k: float = 5.0


def score_candidate_trust(component_a: str, component_b: str, tm_uncertainty: UncertaintyEstimate, neat_component_status: dict) -> float:
    trust = 1.0
    trust -= min(tm_uncertainty.tm_min_std_k / 50.0, 0.5)
    if neat_component_status.get("component_a") != "measured":
        trust -= 0.1
    if neat_component_status.get("component_b") != "measured":
        trust -= 0.1
    if Chem.MolFromSmiles(component_a) is None or Chem.MolFromSmiles(component_b) is None:
        trust -= 0.4
    return max(0.0, min(1.0, trust))


def apply_uncertainty_policy(results: list, uncertainty: dict[str, UncertaintyEstimate], policy: UncertaintyPolicy) -> list[AnnotatedResult]:
    annotated: list[AnnotatedResult] = []
    for result in results:
        estimate = uncertainty[result.curve.smiles_b]
        trust_score = score_candidate_trust(
            result.curve.smiles_a,
            result.curve.smiles_b,
            estimate,
            {"component_a": "heuristic", "component_b": "heuristic"},
        )
        if policy.mode == "filter" and trust_score < policy.min_trust_score:
            continue
        penalty = (1.0 - trust_score) * policy.soft_penalty_weight * max(result.min_tm_k, 1.0)
        ranking_score = result.min_tm_k if policy.mode == "report_only" else result.min_tm_k + penalty
        annotated.append(
            AnnotatedResult(
                result=result,
                uncertainty=estimate,
                trust_score=trust_score,
                ranking_score=ranking_score,
            )
        )
    return sorted(annotated, key=lambda item: (not item.result.is_des, item.ranking_score, -item.trust_score))
```

```python
from __future__ import annotations

from dataclasses import dataclass

from ..evaluation import DesResult
from .model import UncertaintyEstimate


@dataclass(frozen=True)
class AnnotatedResult:
    result: DesResult
    uncertainty: UncertaintyEstimate
    trust_score: float
    ranking_score: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uncertainty_heuristics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/uncertainty/schemas.py des_multi_agent/uncertainty/heuristics.py des_multi_agent/uncertainty/policy.py des_multi_agent/uncertainty/filtering.py tests/test_uncertainty_heuristics.py
git commit -m "feat: add heuristic trust scoring"
```

### Task 3: Integrate uncertainty into orchestration and reporting

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Modify: `examples/demo_des_search.py`
- Test: `tests/test_uncertainty_orchestrator.py`
- Test: `tests/test_uncertainty_reporting.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import CandidateProposal, MeltingPointEstimate
from des_multi_agent.uncertainty.model import UncertaintyEstimate
from des_multi_agent.uncertainty.policy import UncertaintyPolicy


def _curve(smiles_a: str, smiles_b: str, min_tm_k: float) -> CurvePrediction:
    return CurvePrediction(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k, min_tm_k, min_tm_k],
        t1_k=298.15,
        t2_k=300.0,
        checkpoint_path="ckpt.pt",
    )


def _result(smiles_a: str, smiles_b: str, min_tm_k: float) -> DesResult:
    curve = _curve(smiles_a, smiles_b, min_tm_k)
    return DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="ok",
        min_tm_k=min_tm_k,
    )


def _estimate(std_k: float, trust_score: float) -> UncertaintyEstimate:
    return UncertaintyEstimate(
        tm_min_values=[240.0, 240.0, 240.0],
        tm_min_mean_k=240.0,
        tm_min_std_k=std_k,
        tm_min_min_k=240.0,
        tm_min_max_k=240.0,
        trust_score=trust_score,
        uncertainty_flag="low",
        explanation="",
    )


def test_run_search_report_filters_low_trust(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(smiles="OCCO", rationale="polyol", family="polyol"),
            CandidateProposal(smiles="O", rationale="alcohol", family="alcohol"),
        ],
    )
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )
    monkeypatch.setattr(orchestrator, "predict_curve", lambda component_a, component_b, **kwargs: _curve(component_a, component_b, 240.0 if component_b == "OCCO" else 250.0))
    monkeypatch.setattr(orchestrator, "classify_des", lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)))
    monkeypatch.setattr(orchestrator, "estimate_min_tm_uncertainty", lambda component_a, component_b, checkpoint_path, config_path: _estimate(2.0, 0.90) if component_b == "OCCO" else _estimate(20.0, 0.20))

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=2,
        checkpoint_path="ckpt.pt",
        config_path="ml_des_mp/config.yaml",
        uncertainty_policy=UncertaintyPolicy(mode="filter", min_trust_score=0.5, soft_penalty_weight=0.35, std_high_threshold_k=15.0, std_medium_threshold_k=5.0),
    )

    assert [r.curve.smiles_b for r in outcome.results] == ["OCCO"]
    assert outcome.annotated_results[0].trust_score == 0.90
```

```python
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.reporting import format_report
from des_multi_agent.uncertainty.model import UncertaintyEstimate
from des_multi_agent.uncertainty.schemas import AnnotatedResult


def test_report_shows_uncertainty_fields():
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b="OCCO",
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[220.0, 219.0, 221.0],
        t1_k=298.15,
        t2_k=300.0,
        checkpoint_path="ckpt.pt",
    )
    result = DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="ok",
        min_tm_k=219.0,
    )
    estimate = UncertaintyEstimate(
        tm_min_values=[219.0, 220.0, 218.0],
        tm_min_mean_k=219.0,
        tm_min_std_k=1.0,
        tm_min_min_k=218.0,
        tm_min_max_k=220.0,
        trust_score=0.88,
        uncertainty_flag="low",
        explanation="",
    )
    annotated = AnnotatedResult(result=result, uncertainty=estimate, trust_score=0.88, ranking_score=219.0)

    text = format_report([result], annotated_results=[annotated])

    assert "trust=0.88" in text
    assert "std=1.00 K" in text
    assert "low" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_uncertainty_orchestrator.py tests/test_uncertainty_reporting.py -v`
Expected: FAIL because the orchestrator and reporter do not yet understand uncertainty annotations.

- [ ] **Step 3: Write minimal implementation**

```python
from des_multi_agent.uncertainty.filtering import apply_uncertainty_policy, rank_annotated_results
from des_multi_agent.uncertainty.model import estimate_min_tm_uncertainty
from des_multi_agent.uncertainty.policy import UncertaintyPolicy
from des_multi_agent.uncertainty.schemas import AnnotatedResult
```

```python
@dataclass(frozen=True)
class SearchOutcome:
    results: list[DesResult]
    annotated_results: list[AnnotatedResult]
    brainstorm_candidates: list[CandidateBrainstorm]
    explanation_notes: list[ExplanationNote]
    critique_notes: list[CritiqueNote]
    llm_warnings: list[str]
```

```python
def run_search_report(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    thresholds: DesThresholds | None = None,
    llm_cfg: Mapping[str, object] | None = None,
    llm_request_fn=None,
    uncertainty_policy: UncertaintyPolicy | None = None,
):
    checkpoint_path = resolve_existing_path(checkpoint_path)
    config_path = resolve_existing_path(config_path)
    proposals = generate_candidates(component_a, n=n, constraints=None)
    llm_candidates: list[CandidateBrainstorm] = []
    llm_warnings: list[str] = []
    provider = build_llm_provider(llm_cfg, request_fn=llm_request_fn) if llm_cfg else None
    if provider is not None:
        try:
            llm_candidates = provider.brainstorm_candidates(
                component_a,
                None,
                _search_context(component_a, n, str(checkpoint_path), str(config_path)),
            )
        except Exception as exc:
            llm_warnings.append(f"LLM brainstorming failed: {exc}")
    merged = _merge_candidates(proposals, llm_candidates)
    filtered = filter_candidates(component_a, merged)
    thresholds = thresholds or DesThresholds(
        absolute_tm_max_k=DEFAULT_ABSOLUTE_TM_MAX_K,
        relative_drop_min=DEFAULT_RELATIVE_DROP_MIN,
    )
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
    ranked = rank_results(results)
    policy = uncertainty_policy or UncertaintyPolicy()
    uncertainty_by_smiles: dict[str, UncertaintyEstimate] = {}
    for result in ranked:
        uncertainty_by_smiles[result.curve.smiles_b] = estimate_min_tm_uncertainty(
            component_a=component_a,
            component_b=result.curve.smiles_b,
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
        )
    annotated_results = apply_uncertainty_policy(ranked, uncertainty_by_smiles, policy)
    annotated_results = rank_annotated_results(annotated_results)
    ranked_results = [item.result for item in annotated_results]
    explanation_notes: list[ExplanationNote] = []
    critique_notes: list[CritiqueNote] = []
    if provider is not None:
        context = _search_context(component_a, n, str(checkpoint_path), str(config_path))
        try:
            explanation_notes = provider.generate_explanations(ranked, context)
        except Exception as exc:
            llm_warnings.append(f"LLM explanation generation failed: {exc}")
        try:
            critique_notes = provider.critique_results(ranked, context)
        except Exception as exc:
            llm_warnings.append(f"LLM critique generation failed: {exc}")
    return SearchOutcome(
        results=ranked_results,
        annotated_results=annotated_results,
        brainstorm_candidates=llm_candidates,
        explanation_notes=explanation_notes,
        critique_notes=critique_notes,
        llm_warnings=llm_warnings,
    )
```

```python
def format_report(
    results,
    annotated_results=None,
    explanation_notes=None,
    critique_notes=None,
    brainstorm_candidates=None,
    llm_warnings=None,
) -> str:
    lines = ["smiles_b | is_des | min_tm_k | rationale"]
    annotated_by_smiles = {}
    if annotated_results:
        annotated_by_smiles = {item.result.curve.smiles_b: item for item in annotated_results}
    for r in results:
        if r.curve.smiles_b in annotated_by_smiles:
            ann = annotated_by_smiles[r.curve.smiles_b]
            lines.append(
                f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | trust={ann.trust_score:.2f} | std={ann.uncertainty.tm_min_std_k:.2f} K | flag={ann.uncertainty.uncertainty_flag} | {r.rationale}"
            )
        else:
            lines.append(f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {r.rationale}")
    if brainstorm_candidates:
        lines.append("")
        lines.append("LLM brainstorm:")
        for note in brainstorm_candidates:
            lines.append(f"{note.smiles} | {note.family} | {note.rationale}")
    if explanation_notes:
        lines.append("")
        lines.append("LLM explanations:")
        for note in explanation_notes:
            evidence = "; ".join(note.evidence) if note.evidence else "-"
            lines.append(f"{note.smiles} | {note.summary} | {evidence}")
    if critique_notes:
        lines.append("")
        lines.append("LLM critique:")
        for note in critique_notes:
            concerns = "; ".join(note.concerns) if note.concerns else "-"
            lines.append(f"{note.smiles} | {note.assessment} | {concerns}")
    if llm_warnings:
        lines.append("")
        lines.append("LLM warnings:")
        for warning in llm_warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)
```

```python
print(
    format_report(
        outcome.results,
        annotated_results=outcome.annotated_results,
        explanation_notes=outcome.explanation_notes,
        critique_notes=outcome.critique_notes,
        brainstorm_candidates=outcome.brainstorm_candidates,
        llm_warnings=outcome.llm_warnings,
    )
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_uncertainty_orchestrator.py tests/test_uncertainty_reporting.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/reporting.py examples/demo_des_search.py tests/test_uncertainty_orchestrator.py tests/test_uncertainty_reporting.py
git commit -m "feat: integrate uncertainty into DES workflow"
```

### Task 4: Add CLI coverage and update docs

**Files:**
- Modify: `des_multi_agent/cli.py`
- Modify: `README.md`
- Modify: `examples/README.md`
- Modify: `docs/tutorial.md`
- Test: `tests/test_uncertainty_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.cli import build_parser


def test_cli_accepts_uncertainty_arguments():
    parser = build_parser()
    args = parser.parse_args([
        "--component-a", "CCO",
        "--n", "3",
        "--checkpoint-path", "ckpt.pt",
        "--uncertainty-mode", "filter",
        "--min-trust-score", "0.70",
        "--soft-penalty-weight", "0.20",
    ])

    assert args.uncertainty_mode == "filter"
    assert args.min_trust_score == 0.70
    assert args.soft_penalty_weight == 0.20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uncertainty_cli.py -v`
Expected: FAIL because `build_parser()` does not yet expose uncertainty policy arguments.

- [ ] **Step 3: Write minimal implementation**

```python
from des_multi_agent.uncertainty.policy import UncertaintyPolicy
```

```python
parser.add_argument("--uncertainty-mode", choices=["filter", "penalize", "report_only"], default="penalize")
parser.add_argument("--min-trust-score", type=float, default=0.55)
parser.add_argument("--soft-penalty-weight", type=float, default=0.35)
```

```python
uncertainty_policy = UncertaintyPolicy(
    mode=args.uncertainty_mode,
    min_trust_score=args.min_trust_score,
    soft_penalty_weight=args.soft_penalty_weight,
)
outcome = run_search_report(
    component_a=args.component_a,
    n=args.n,
    checkpoint_path=args.checkpoint_path,
    config_path=args.config_path,
    thresholds=thresholds,
    llm_cfg=llm_cfg,
    llm_request_fn=llm_request_fn,
    uncertainty_policy=uncertainty_policy,
)
```

```markdown
## Uncertainty

Phase 1 adds a three-pass minimum-Tm uncertainty estimate and a normalized trust score in the range `0.0` to `1.0`.
Use `--uncertainty-mode filter` to drop low-trust candidates, `--uncertainty-mode penalize` to demote them, or `--uncertainty-mode report_only` to show the values without changing the ranking.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uncertainty_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py README.md examples/README.md docs/tutorial.md tests/test_uncertainty_cli.py
git commit -m "docs: surface uncertainty controls in cli and docs"
```

### Task 5: Final verification

**Files:**
- Modify: `des_multi_agent/uncertainty/__init__.py`
- Modify: `des_multi_agent/uncertainty/schemas.py`
- Modify: `des_multi_agent/uncertainty/model.py`
- Modify: `des_multi_agent/uncertainty/heuristics.py`
- Modify: `des_multi_agent/uncertainty/policy.py`
- Modify: `des_multi_agent/uncertainty/filtering.py`
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Modify: `des_multi_agent/cli.py`
- Modify: `examples/demo_des_search.py`
- Modify: `README.md`
- Modify: `examples/README.md`
- Modify: `docs/tutorial.md`
- Test: `tests/test_uncertainty_model.py`
- Test: `tests/test_uncertainty_heuristics.py`
- Test: `tests/test_uncertainty_orchestrator.py`
- Test: `tests/test_uncertainty_reporting.py`
- Test: `tests/test_uncertainty_cli.py`

- [ ] **Step 1: Run the uncertainty-focused tests**

Run: `python -m pytest tests/test_uncertainty_model.py tests/test_uncertainty_heuristics.py tests/test_uncertainty_orchestrator.py tests/test_uncertainty_reporting.py tests/test_uncertainty_cli.py -q`
Expected: PASS

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add des_multi_agent/uncertainty/__init__.py des_multi_agent/uncertainty/schemas.py des_multi_agent/uncertainty/model.py des_multi_agent/uncertainty/heuristics.py des_multi_agent/uncertainty/policy.py des_multi_agent/uncertainty/filtering.py des_multi_agent/orchestrator.py des_multi_agent/reporting.py des_multi_agent/cli.py examples/demo_des_search.py README.md examples/README.md docs/tutorial.md tests/test_uncertainty_model.py tests/test_uncertainty_heuristics.py tests/test_uncertainty_orchestrator.py tests/test_uncertainty_reporting.py tests/test_uncertainty_cli.py
git commit -m "test: verify phase 1 uncertainty layer"
```
