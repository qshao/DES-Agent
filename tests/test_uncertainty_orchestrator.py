from __future__ import annotations

from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.schemas import CandidateProposal, DesThresholds
from des_multi_agent.uncertainty import MinimumTmUncertainty, UncertaintyPolicy


def _curve(smiles_a: str, smiles_b: str, min_tm_k: float) -> CurvePrediction:
    return CurvePrediction(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k + 5.0, min_tm_k, min_tm_k + 2.0],
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


def _uncertainty(smiles_b: str, trust_score: float, std_tm_k: float) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a="CCO",
        component_b=smiles_b,
        repeated_values=(240.0, 240.0 + std_tm_k, 240.0 + 2.0 * std_tm_k),
        mean_tm_k=240.0 + std_tm_k,
        std_tm_k=std_tm_k,
        min_tm_k=240.0,
        max_tm_k=240.0 + 2.0 * std_tm_k,
        trust_score=trust_score,
        uncertainty_flag="low" if std_tm_k <= 5.0 else "high",
        explanation="demo",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )


def test_run_search_report_filters_low_trust_candidates(monkeypatch, tmp_path):
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
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": _curve(component_a, component_b, 230.0 if component_b == "OCCO" else 210.0),
    )
    monkeypatch.setattr(
        orchestrator,
        "classify_des",
        lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)),
    )
    monkeypatch.setattr(
        orchestrator,
        "estimate_min_tm_uncertainty",
        lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b, 0.9 if component_b == "OCCO" else 0.2, 1.0),
    )

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=2,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="filter", min_trust_score=0.5),
    )

    assert [r.curve.smiles_b for r in outcome.results] == ["OCCO"]
    assert [item.result.curve.smiles_b for item in outcome.annotated_results] == ["OCCO"]
    assert outcome.annotated_results[0].trust_score >= 0.7


def test_run_search_report_penalizes_and_reranks_low_trust_candidates(monkeypatch, tmp_path):
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
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": _curve(component_a, component_b, 230.0 if component_b == "OCCO" else 210.0),
    )
    monkeypatch.setattr(
        orchestrator,
        "classify_des",
        lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)),
    )
    monkeypatch.setattr(
        orchestrator,
        "estimate_min_tm_uncertainty",
        lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b, 0.9 if component_b == "OCCO" else 0.2, 1.0),
    )

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=2,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="penalize", min_trust_score=0.5, soft_penalty_weight=0.35),
    )

    assert [r.curve.smiles_b for r in outcome.results] == ["OCCO", "O"]
    assert outcome.annotated_results[0].trust_score > outcome.annotated_results[1].trust_score
