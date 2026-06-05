from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.reporting import format_report
from des_multi_agent.uncertainty.schemas import AnnotatedResult, MinimumTmUncertainty
from examples import demo_des_search


def _make_result(smiles_b: str, min_tm_k: float) -> DesResult:
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k + 5.0, min_tm_k, min_tm_k + 8.0],
        t1_k=300.0,
        t2_k=280.0,
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


def _make_annotation(result: DesResult) -> AnnotatedResult:
    uncertainty = MinimumTmUncertainty(
        component_a=result.curve.smiles_a,
        component_b=result.curve.smiles_b,
        repeated_values=(result.min_tm_k - 1.0, result.min_tm_k, result.min_tm_k + 1.0),
        mean_tm_k=result.min_tm_k,
        std_tm_k=0.82,
        min_tm_k=result.min_tm_k - 1.0,
        max_tm_k=result.min_tm_k + 1.0,
        trust_score=0.88,
        uncertainty_flag="low",
        explanation="demo uncertainty",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )
    return AnnotatedResult(result=result, uncertainty=uncertainty, trust_score=0.88, ranking_score=0.91)


def test_report_shows_uncertainty_fields_and_preserves_llm_sections():
    result = _make_result("OCCO", 219.5)
    annotated = _make_annotation(result)
    report = format_report(
        [result],
        annotated_results=[annotated],
        explanation_notes=[ExplanationNote(smiles="OCCO", summary="ranked highly", evidence=["low min Tm"])],
        critique_notes=[CritiqueNote(smiles="OCCO", assessment="advisory only", concerns=["possible outlier"])],
        brainstorm_candidates=[CandidateBrainstorm(smiles="OCCO", rationale="polyol", family="polyol")],
        llm_warnings=["warning"],
    )

    assert "trust_score" in report
    assert "tm_min_mean_k" in report
    assert "tm_min_std_k" in report
    assert "uncertainty_flag" in report
    assert "LLM brainstorm" in report
    assert "LLM explanations" in report
    assert "LLM critique" in report
    assert "LLM warnings" in report


def test_demo_passes_annotated_results_to_report_formatter(monkeypatch, capsys):
    result = _make_result("OCCO", 219.5)
    annotated = _make_annotation(result)
    captured = {}

    monkeypatch.setattr(
        demo_des_search,
        "run_search_report",
        lambda *args, **kwargs: demo_des_search.SearchOutcome(
            results=[result],
            annotated_results=[annotated],
            brainstorm_candidates=[],
            explanation_notes=[],
            critique_notes=[],
            llm_warnings=[],
        ),
    )

    def fake_format_report(results, annotated_results=None, **kwargs):
        captured["results"] = results
        captured["annotated_results"] = annotated_results
        return "report"

    monkeypatch.setattr(demo_des_search, "format_report", fake_format_report)
    demo_des_search.main(["--component-a", "CCO", "--n", "1", "--checkpoint-path", "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt", "--config-path", "ml_des_mp/config.yaml"])

    out = capsys.readouterr().out
    assert out.strip() == "report"
    assert captured["results"] == [result]
    assert captured["annotated_results"] == [annotated]
