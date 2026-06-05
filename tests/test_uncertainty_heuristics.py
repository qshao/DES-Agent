from __future__ import annotations

from dataclasses import replace

import pytest

from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.uncertainty.schemas import MinimumTmUncertainty


def _make_result(smiles_b: str, min_tm_k: float, is_des: bool = True) -> DesResult:
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k + 12.0, min_tm_k, min_tm_k + 6.0],
        t1_k=300.0,
        t2_k=275.0,
        checkpoint_path="ckpt.pt",
    )
    return DesResult(
        curve=curve,
        absolute_pass=is_des,
        relative_pass=is_des,
        is_des=is_des,
        rationale="demo",
        min_tm_k=min_tm_k,
    )


def _make_uncertainty(component_b: str, std_tm_k: float, trust_score: float = 0.5) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a="CCO",
        component_b=component_b,
        repeated_values=(240.0, 240.0 + std_tm_k, 240.0 + 2.0 * std_tm_k),
        mean_tm_k=240.0 + std_tm_k,
        std_tm_k=std_tm_k,
        min_tm_k=240.0,
        max_tm_k=240.0 + 2.0 * std_tm_k,
        trust_score=trust_score,
        uncertainty_flag="low",
        explanation="demo",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )


def test_score_candidate_trust_clamps_and_decreases_with_std_and_confidence():
    from des_multi_agent.uncertainty.heuristics import score_candidate_trust

    high_confidence_status = (
        MeltingPointEstimate(component="CCO", tm_k=300.0, source="heuristic", confidence=0.95),
        MeltingPointEstimate(component="O", tm_k=275.0, source="heuristic", confidence=0.9),
    )
    low_confidence_status = (
        MeltingPointEstimate(component="CCO", tm_k=300.0, source="heuristic", confidence=0.2),
        MeltingPointEstimate(component="O", tm_k=275.0, source="heuristic", confidence=0.3),
    )
    low_std = _make_uncertainty("O", std_tm_k=0.25, trust_score=1.2)
    high_std = replace(low_std, std_tm_k=12.0, trust_score=0.05)

    high_trust = score_candidate_trust("CCO", "O", low_std, high_confidence_status)
    low_trust = score_candidate_trust("CCO", "O", high_std, low_confidence_status)

    assert 0.0 <= high_trust <= 1.0
    assert 0.0 <= low_trust <= 1.0
    assert high_trust > low_trust


def test_apply_uncertainty_policy_filters_at_threshold_boundary(monkeypatch):
    from des_multi_agent.uncertainty import filtering
    from des_multi_agent.uncertainty.policy import UncertaintyPolicy

    results = [_make_result("O", 220.0), _make_result("CO", 230.0)]
    uncertainties = {
        "O": _make_uncertainty("O", std_tm_k=0.5, trust_score=0.50),
        "CO": _make_uncertainty("CO", std_tm_k=0.5, trust_score=0.49),
    }
    policy = UncertaintyPolicy(mode="filter", min_trust_score=0.50)

    monkeypatch.setattr(filtering, "score_candidate_trust", lambda *args, **kwargs: uncertainties[args[1]].trust_score)
    monkeypatch.setattr(
        filtering,
        "_lookup_neat_status",
        lambda component_a, component_b: (
            MeltingPointEstimate(component=component_a, tm_k=300.0, source="heuristic", confidence=0.9),
            MeltingPointEstimate(component=component_b, tm_k=275.0, source="heuristic", confidence=0.9),
        ),
    )

    annotated = filtering.apply_uncertainty_policy(results, uncertainties, policy)

    assert [item.result.curve.smiles_b for item in annotated] == ["O"]
    assert annotated[0].trust_score == pytest.approx(0.50)




def test_apply_uncertainty_policy_report_only_preserves_order(monkeypatch):
    from des_multi_agent.uncertainty import filtering
    from des_multi_agent.uncertainty.policy import UncertaintyPolicy

    results = [_make_result("CO", 215.0, is_des=False), _make_result("O", 220.0, is_des=True)]
    uncertainties = {
        "CO": _make_uncertainty("CO", std_tm_k=0.5, trust_score=0.35),
        "O": _make_uncertainty("O", std_tm_k=0.5, trust_score=0.80),
    }
    policy = UncertaintyPolicy(mode="report_only")

    monkeypatch.setattr(filtering, "score_candidate_trust", lambda *args, **kwargs: uncertainties[args[1]].trust_score)
    monkeypatch.setattr(
        filtering,
        "_lookup_neat_status",
        lambda component_a, component_b: (
            MeltingPointEstimate(component=component_a, tm_k=300.0, source="heuristic", confidence=0.9),
            MeltingPointEstimate(component=component_b, tm_k=275.0, source="heuristic", confidence=0.9),
        ),
    )

    annotated = filtering.apply_uncertainty_policy(results, uncertainties, policy)

    assert [item.result.curve.smiles_b for item in annotated] == ["CO", "O"]
    assert all(item.ranking_score > 0 for item in annotated)
def test_apply_uncertainty_policy_penalizes_low_trust(monkeypatch):
    from des_multi_agent.uncertainty import filtering
    from des_multi_agent.uncertainty.policy import UncertaintyPolicy

    results = [_make_result("O", 220.0), _make_result("CO", 215.0)]
    uncertainties = {
        "O": _make_uncertainty("O", std_tm_k=0.5, trust_score=0.80),
        "CO": _make_uncertainty("CO", std_tm_k=0.5, trust_score=0.35),
    }
    policy = UncertaintyPolicy(mode="penalize", min_trust_score=0.50, soft_penalty_weight=2.0)

    monkeypatch.setattr(filtering, "score_candidate_trust", lambda *args, **kwargs: uncertainties[args[1]].trust_score)
    monkeypatch.setattr(
        filtering,
        "_lookup_neat_status",
        lambda component_a, component_b: (
            MeltingPointEstimate(component=component_a, tm_k=300.0, source="heuristic", confidence=0.9),
            MeltingPointEstimate(component=component_b, tm_k=275.0, source="heuristic", confidence=0.9),
        ),
    )

    annotated = filtering.apply_uncertainty_policy(results, uncertainties, policy)

    assert len(annotated) == 2
    assert annotated[0].result.curve.smiles_b == "O"
    assert annotated[0].ranking_score > annotated[1].ranking_score
    assert annotated[1].ranking_score < annotated[1].trust_score


def test_rank_annotated_results_orders_des_then_trust_then_min_tm():
    from des_multi_agent.uncertainty.filtering import rank_annotated_results
    from des_multi_agent.uncertainty.schemas import AnnotatedResult

    low_trust_des = AnnotatedResult(
        result=_make_result("CO", 210.0, is_des=True),
        uncertainty=_make_uncertainty("CO", std_tm_k=1.0, trust_score=0.3),
        trust_score=0.3,
        ranking_score=0.3,
    )
    high_trust_des = AnnotatedResult(
        result=_make_result("O", 220.0, is_des=True),
        uncertainty=_make_uncertainty("O", std_tm_k=1.0, trust_score=0.9),
        trust_score=0.9,
        ranking_score=0.9,
    )
    non_des = AnnotatedResult(
        result=_make_result("N", 200.0, is_des=False),
        uncertainty=_make_uncertainty("N", std_tm_k=1.0, trust_score=1.0),
        trust_score=1.0,
        ranking_score=1.0,
    )

    ranked = rank_annotated_results([low_trust_des, non_des, high_trust_des])

    assert [item.result.curve.smiles_b for item in ranked] == ["O", "CO", "N"]


def test_score_candidate_trust_missing_component_is_conservative():
    from des_multi_agent.uncertainty.heuristics import score_candidate_trust

    uncertainty = _make_uncertainty("O", std_tm_k=0.5, trust_score=0.8)
    explicit_status = (
        MeltingPointEstimate(component="CCO", tm_k=300.0, source="heuristic", confidence=0.9),
        MeltingPointEstimate(component="O", tm_k=275.0, source="heuristic", confidence=0.9),
    )
    missing_status = {
        "CCO": MeltingPointEstimate(component="CCO", tm_k=300.0, source="heuristic", confidence=0.9),
        "X": MeltingPointEstimate(component="X", tm_k=260.0, source="heuristic", confidence=0.1),
    }

    explicit = score_candidate_trust("CCO", "O", uncertainty, explicit_status)
    missing = score_candidate_trust("CCO", "O", uncertainty, missing_status)

    assert missing < explicit


def test_uncertainty_policy_thresholds_affect_ranking(monkeypatch):
    from des_multi_agent.uncertainty import filtering
    from des_multi_agent.uncertainty.policy import UncertaintyPolicy

    results = [_make_result("O", 220.0), _make_result("CO", 220.0)]
    uncertainties = {
        "O": _make_uncertainty("O", std_tm_k=1.0, trust_score=0.8),
        "CO": _make_uncertainty("CO", std_tm_k=6.0, trust_score=0.8),
    }
    policy = UncertaintyPolicy(mode="penalize", min_trust_score=0.5, soft_penalty_weight=0.35, std_high_threshold_k=5.0, std_medium_threshold_k=2.0)

    monkeypatch.setattr(filtering, "score_candidate_trust", lambda *args, **kwargs: uncertainties[args[1]].trust_score)
    monkeypatch.setattr(
        filtering,
        "_lookup_neat_status",
        lambda component_a, component_b: (
            MeltingPointEstimate(component=component_a, tm_k=300.0, source="heuristic", confidence=0.9),
            MeltingPointEstimate(component=component_b, tm_k=275.0, source="heuristic", confidence=0.9),
        ),
    )

    annotated = filtering.apply_uncertainty_policy(results, uncertainties, policy)

    assert [item.result.curve.smiles_b for item in annotated] == ["O", "CO"]
    assert annotated[0].ranking_score > annotated[1].ranking_score
