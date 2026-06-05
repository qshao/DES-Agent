from pathlib import Path

import pytest

from des_multi_agent import orchestrator
from des_multi_agent.discovery import load_discovery_library
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.schemas import CandidateProposal, DesThresholds
from des_multi_agent.uncertainty import MinimumTmUncertainty, UncertaintyPolicy


def test_load_discovery_library_rejects_malformed_records(tmp_path: Path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "literature.yaml").write_text("- component_a: CCO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="literature.yaml"):
        load_discovery_library(bad_dir)


def test_run_search_report_falls_back_when_discovery_is_empty(monkeypatch, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(
                smiles="O",
                rationale="baseline",
                family="heuristic",
                source="heuristic",
            )
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
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": CurvePrediction(
            smiles_a=component_a,
            smiles_b=component_b,
            ratios=[0.1, 0.5, 0.9],
            tm_pred_k=[230.0, 229.0, 231.0],
            t1_k=t1_k,
            t2_k=t2_k,
            checkpoint_path=checkpoint_path,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "classify_des",
        lambda curve, thresholds: DesResult(
            curve=curve,
            absolute_pass=True,
            relative_pass=True,
            is_des=True,
            rationale="ok",
            min_tm_k=min(curve.tm_pred_k),
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "estimate_min_tm_uncertainty",
        lambda component_a, component_b, checkpoint_path, config_path: MinimumTmUncertainty(
            component_a=component_a,
            component_b=component_b,
            repeated_values=(229.0, 230.0, 231.0),
            mean_tm_k=230.0,
            std_tm_k=1.0,
            min_tm_k=229.0,
            max_tm_k=231.0,
            trust_score=0.9,
            uncertainty_flag="low",
            explanation="demo",
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        ),
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
