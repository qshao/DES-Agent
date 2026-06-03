from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import CandidateProposal, MeltingPointEstimate


class _FakeLLM:
    def brainstorm_candidates(self, component_a, constraints, context):
        return [CandidateBrainstorm(smiles="OCCO", rationale="polyol", family="polyol")]

    def generate_explanations(self, results, context):
        return [ExplanationNote(smiles="OCCO", summary="ranked highly", evidence=["low min Tm"])]

    def critique_results(self, results, context):
        return [CritiqueNote(smiles="OCCO", assessment="advisory only", concerns=["possible outlier"])]


class _FailingLLM:
    def brainstorm_candidates(self, component_a, constraints, context):
        raise RuntimeError("boom")

    def generate_explanations(self, results, context):
        raise RuntimeError("boom")

    def critique_results(self, results, context):
        raise RuntimeError("boom")


def test_llm_candidates_are_merged_but_still_filtered(monkeypatch):
    monkeypatch.setattr(orchestrator, "build_llm_provider", lambda cfg, request_fn=None: _FakeLLM())
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [CandidateProposal(smiles="O", rationale="baseline", family="alcohol")],
    )
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(
            component=component,
            tm_k=300.0,
            source="heuristic",
            confidence=0.5,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda *args, **kwargs: CurvePrediction(
            smiles_a="CCO",
            smiles_b="O",
            ratios=[0.1],
            tm_pred_k=[250.0],
            t1_k=300.0,
            t2_k=300.0,
            checkpoint_path="ckpt.pt",
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
            min_tm_k=250.0,
        ),
    )
    monkeypatch.setattr(orchestrator, "rank_results", lambda results: results)

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=1,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        llm_cfg={
            "enabled": True,
            "provider": "ollama",
            "model_name": "llama3.1",
            "api_base_url": "http://localhost:11434",
        },
    )

    assert len(outcome.results) == 2
    assert outcome.explanation_notes and outcome.explanation_notes[0].summary == "ranked highly"
    assert outcome.critique_notes and outcome.critique_notes[0].assessment == "advisory only"
    assert outcome.llm_warnings == []


def test_llm_failures_are_reported_without_breaking_deterministic_search(monkeypatch):
    monkeypatch.setattr(orchestrator, "build_llm_provider", lambda cfg, request_fn=None: _FailingLLM())
    monkeypatch.setattr(
        orchestrator,
        "generate_candidates",
        lambda component_a, n, constraints=None: [CandidateProposal(smiles="O", rationale="baseline", family="alcohol")],
    )
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(
            component=component,
            tm_k=300.0,
            source="heuristic",
            confidence=0.5,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda *args, **kwargs: CurvePrediction(
            smiles_a="CCO",
            smiles_b="O",
            ratios=[0.1],
            tm_pred_k=[250.0],
            t1_k=300.0,
            t2_k=300.0,
            checkpoint_path="ckpt.pt",
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
            min_tm_k=250.0,
        ),
    )
    monkeypatch.setattr(orchestrator, "rank_results", lambda results: results)

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=1,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        llm_cfg={
            "enabled": True,
            "provider": "ollama",
            "model_name": "llama3.1",
            "api_base_url": "http://localhost:11434",
        },
    )

    assert len(outcome.results) == 1
    assert outcome.llm_warnings
    assert any("brainstorm" in warning for warning in outcome.llm_warnings)
