from pathlib import Path

from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.reporting import format_report
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


def _uncertainty(smiles_b: str) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a="CCO",
        component_b=smiles_b,
        repeated_values=(238.0, 239.0, 240.0),
        mean_tm_k=239.0,
        std_tm_k=1.0,
        min_tm_k=238.0,
        max_tm_k=240.0,
        trust_score=0.88,
        uncertainty_flag="low",
        explanation="demo",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )


def test_run_search_report_includes_discovery_provenance(monkeypatch, tmp_path):
    fixture_dir = Path(__file__).parent / "fixtures" / "discovery"
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(smiles="O", rationale="baseline", family="heuristic", source="heuristic")
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
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": _curve(
            component_a,
            component_b,
            230.0 if component_b == "OCCO" else 220.0,
        ),
    )
    monkeypatch.setattr(orchestrator, "classify_des", lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)))
    monkeypatch.setattr(orchestrator, "estimate_min_tm_uncertainty", lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b))

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
        uncertainty_policy=UncertaintyPolicy(mode="report_only"),
        discovery_path=str(fixture_dir),
    )

    report = format_report(
        outcome.results,
        annotated_results=outcome.annotated_results,
        candidate_proposals=outcome.candidate_proposals,
        llm_warnings=outcome.llm_warnings,
    )

    assert any(candidate.source == "literature" for candidate in outcome.candidate_proposals)
    assert "source=literature" in report
    assert "source=similarity" in report or "source=heuristic" in report




def test_candidate_proposals_are_canonicalized_across_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(smiles="CCO", rationale="heuristic", family="heuristic", source="heuristic")
        ],
    )
    monkeypatch.setattr(orchestrator, "_build_discovery_candidates", lambda component_a, n, discovery_path, llm_warnings: [
        CandidateProposal(smiles="C(C)O", rationale="discovery", family="literature", source="literature", source_id="LIT-002")
    ])
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": _curve(component_a, component_b, 230.0),
    )
    monkeypatch.setattr(orchestrator, "classify_des", lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)))
    monkeypatch.setattr(orchestrator, "estimate_min_tm_uncertainty", lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b))

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
        component_a="c1ccccc1",  # benzene — distinct from the CCO/C(C)O candidates so neither is suppressed as self-pair
        n=1,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="report_only"),
    )

    assert len(outcome.candidate_proposals) == 1
    assert outcome.candidate_proposals[0].smiles in {"CCO", "C(C)O"}
    assert outcome.results[0].curve.smiles_b in {"CCO", "C(C)O"}
def test_run_search_report_falls_back_when_discovery_is_empty(monkeypatch, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(smiles="O", rationale="baseline", family="heuristic", source="heuristic")
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
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": _curve(component_a, component_b, 230.0),
    )
    monkeypatch.setattr(orchestrator, "classify_des", lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)))
    monkeypatch.setattr(orchestrator, "estimate_min_tm_uncertainty", lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b))

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
        n=1,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="report_only"),
        discovery_path=str(empty_dir),
    )

    assert outcome.results
    assert outcome.candidate_proposals
    assert outcome.candidate_proposals[0].source == "heuristic"
    assert any("No discovery candidates" in warning for warning in outcome.llm_warnings)
