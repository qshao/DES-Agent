from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm, CandidateReview, CritiqueNote, ExplanationNote
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import CandidateProposal, MeltingPointEstimate
from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty


class _FakeLLM:
    def review_candidate(self, component_a, candidate_smiles, context):
        return CandidateReview(
            smiles=candidate_smiles,
            decision="keep",
            confidence=0.9,
            rationale="acceptable candidate",
            notes=["demo review"],
        )

    def brainstorm_candidates(self, component_a, constraints, context, **kwargs):
        return [CandidateBrainstorm(smiles="OCCO", rationale="polyol", family="polyol")]

    def generate_explanations(self, results, context):
        return [ExplanationNote(smiles="OCCO", summary="ranked highly", evidence=["low min Tm"])]

    def critique_results(self, results, context):
        return [CritiqueNote(smiles="OCCO", assessment="advisory only", concerns=["possible outlier"])]

    def detect_contradictions(self, results, context):
        return []


class _FailingLLM:
    def review_candidate(self, component_a, candidate_smiles, context):
        raise RuntimeError("boom")

    def brainstorm_candidates(self, component_a, constraints, context, **kwargs):
        raise RuntimeError("boom")

    def generate_explanations(self, results, context):
        raise RuntimeError("boom")

    def critique_results(self, results, context):
        raise RuntimeError("boom")

    def detect_contradictions(self, results, context):
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


def test_llm_candidates_are_promoted_to_candidate_proposals(monkeypatch):
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

    assert any(proposal.source == "llm" for proposal in outcome.candidate_proposals)
    assert any(proposal.source_id == "brainstorm" for proposal in outcome.candidate_proposals)



def test_apply_candidate_reviews_deprioritizes_candidate():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="demo", family="polyol", source="heuristic", source_id="rule"),
        CandidateProposal(smiles="CC(=O)O", rationale="demo", family="acid", source="heuristic", source_id="rule"),
    ]
    reviews = {
        "OCCO": CandidateReview(
            smiles="OCCO",
            decision="deprioritize",
            confidence=0.25,
            rationale="Looks plausible but less compelling than alternatives.",
            notes=["low confidence"],
        )
    }
    reviewed, penalties = orchestrator._apply_candidate_reviews(proposals, reviews)
    assert [item.smiles for item in reviewed] == ["OCCO", "CC(=O)O"]
    assert penalties == {"OCCO": 0.25}


def test_apply_candidate_reviews_reject_drops_candidate():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="demo", family="polyol", source="heuristic", source_id="rule"),
        CandidateProposal(smiles="CC(=O)O", rationale="demo", family="acid", source="heuristic", source_id="rule"),
    ]
    reviews = {
        "OCCO": CandidateReview(
            smiles="OCCO",
            decision="reject",
            confidence=0.93,
            rationale="Does not look like a useful DES partner.",
            notes=["too similar to the input"],
        )
    }
    reviewed, penalties = orchestrator._apply_candidate_reviews(proposals, reviews)
    assert [item.smiles for item in reviewed] == ["CC(=O)O"]
    assert penalties == {}


def test_apply_review_penalties_reorders_results():
    def _annotated(smiles: str, ranking_score: float) -> AnnotatedResult:
        curve = CurvePrediction(
            smiles_a="CCO",
            smiles_b=smiles,
            ratios=[0.1],
            tm_pred_k=[250.0],
            t1_k=300.0,
            t2_k=300.0,
            checkpoint_path="ckpt.pt",
        )
        result = DesResult(
            curve=curve,
            absolute_pass=True,
            relative_pass=True,
            is_des=True,
            rationale="ok",
            min_tm_k=250.0,
        )
        uncertainty = MinimumTmUncertainty(
            component_a="CCO",
            component_b=smiles,
            repeated_values=(250.0,),
            mean_tm_k=250.0,
            std_tm_k=0.0,
            min_tm_k=250.0,
            max_tm_k=250.0,
            trust_score=0.9,
            uncertainty_flag="low",
            explanation="demo",
            checkpoint_path="ckpt.pt",
            config_path="cfg.yaml",
        )
        return AnnotatedResult(result=result, uncertainty=uncertainty, trust_score=0.9, ranking_score=ranking_score)

    annotated = [_annotated("OCCO", 0.90), _annotated("CC(=O)O", 0.80)]
    adjusted = orchestrator._apply_review_penalties(annotated, {"OCCO": 0.25})
    assert [item.result.curve.smiles_b for item in adjusted] == ["CC(=O)O", "OCCO"]


def test_review_top_candidates_ignores_wrong_smiles():
    class DummyProvider:
        def review_candidate(self, component_a, candidate_smiles, context):
            return CandidateReview(
                smiles="WRONG",
                decision="keep",
                confidence=0.9,
                rationale="mismatch",
                notes=[],
            )

    warnings = []
    reviews, review_map = orchestrator._review_top_candidates(
        DummyProvider(),
        "CCO",
        [CandidateProposal(smiles="OCCO", rationale="demo", family="polyol", source="heuristic", source_id="rule")],
        "context",
        1,
        warnings,
    )
    assert reviews == []
    assert review_map == {}
    assert warnings and "wrong SMILES" in warnings[0]


def test_build_prior_productive_family_summary_limits_to_top_counts():
    from des_multi_agent.orchestrator import _build_prior_productive_family_summary

    summary = _build_prior_productive_family_summary({"polyol": 4, "amide": 2, "acid": 1, "amine": 1})
    assert summary == {"polyol": 4, "amide": 2, "acid": 1}
