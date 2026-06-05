from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.schemas import CandidateProposal
from des_multi_agent.uncertainty.policy import UncertaintyPolicy
from des_multi_agent.uncertainty.schemas import MinimumTmUncertainty


def _make_result(smiles_b: str, min_tm_k: float) -> DesResult:
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k + 5.0, min_tm_k, min_tm_k + 10.0],
        t1_k=300.0,
        t2_k=290.0,
        checkpoint_path="ckpt.pt",
    )
    return DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="demo",
        min_tm_k=min_tm_k,
    )


def _make_uncertainty(smiles_b: str, trust_score: float, std_tm_k: float) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a="CCO",
        component_b=smiles_b,
        repeated_values=(240.0, 240.0 + std_tm_k, 240.0 + 2.0 * std_tm_k),
        mean_tm_k=240.0 + std_tm_k,
        std_tm_k=std_tm_k,
        min_tm_k=240.0,
        max_tm_k=240.0 + 2.0 * std_tm_k,
        trust_score=trust_score,
        uncertainty_flag="low" if std_tm_k <= 2.0 else "high",
        explanation="demo",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )


def _patch_core(monkeypatch, results_by_smiles):
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(smiles="CO", rationale="baseline", family="alcohol"),
            CandidateProposal(smiles="O", rationale="baseline", family="alcohol"),
        ],
    )
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(
            component=component,
            tm_k=300.0 if component == "CCO" else 275.0,
            source="heuristic",
            confidence=0.9,
        ),
    )

    def fake_predict_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml"):
        return CurvePrediction(
            smiles_a=component_a,
            smiles_b=component_b,
            ratios=[0.1, 0.5, 0.9],
            tm_pred_k=[results_by_smiles[component_b].min_tm_k + 8.0, results_by_smiles[component_b].min_tm_k, results_by_smiles[component_b].min_tm_k + 3.0],
            t1_k=t1_k,
            t2_k=t2_k,
            checkpoint_path=checkpoint_path,
        )

    monkeypatch.setattr(orchestrator, "predict_curve", fake_predict_curve)
    monkeypatch.setattr(
        orchestrator,
        "classify_des",
        lambda curve, thresholds: _make_result(curve.smiles_b, results_by_smiles[curve.smiles_b].min_tm_k),
    )


def test_run_search_report_filters_low_trust_candidates(monkeypatch):
    _patch_core(
        monkeypatch,
        {
            "CO": _make_uncertainty("CO", trust_score=0.2, std_tm_k=20.0),
            "O": _make_uncertainty("O", trust_score=0.95, std_tm_k=0.5),
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "estimate_min_tm_uncertainty",
        lambda component_a, component_b, checkpoint_path, config_path: {
            "CO": _make_uncertainty("CO", trust_score=0.2, std_tm_k=20.0),
            "O": _make_uncertainty("O", trust_score=0.95, std_tm_k=0.5),
        }[component_b],
    )

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=2,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        uncertainty_policy=UncertaintyPolicy(mode="filter", min_trust_score=0.5),
    )

    assert [result.curve.smiles_b for result in outcome.results] == ["O"]
    assert [item.result.curve.smiles_b for item in outcome.annotated_results] == ["O"]
    assert outcome.annotated_results[0].trust_score > 0.5


def test_run_search_report_penalizes_low_trust_candidates(monkeypatch):
    _patch_core(
        monkeypatch,
        {
            "CO": _make_uncertainty("CO", trust_score=0.1, std_tm_k=18.0),
            "O": _make_uncertainty("O", trust_score=0.95, std_tm_k=0.5),
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "estimate_min_tm_uncertainty",
        lambda component_a, component_b, checkpoint_path, config_path: {
            "CO": _make_uncertainty("CO", trust_score=0.1, std_tm_k=18.0),
            "O": _make_uncertainty("O", trust_score=0.95, std_tm_k=0.5),
        }[component_b],
    )

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=2,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        uncertainty_policy=UncertaintyPolicy(mode="penalize", min_trust_score=0.5, soft_penalty_weight=2.0),
    )

    assert [result.curve.smiles_b for result in outcome.results] == ["O", "CO"]
    assert [item.result.curve.smiles_b for item in outcome.annotated_results] == ["O", "CO"]
    assert outcome.annotated_results[0].ranking_score >= outcome.annotated_results[1].ranking_score
