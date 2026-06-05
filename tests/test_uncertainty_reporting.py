from __future__ import annotations

from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.reporting import format_report
from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty


def test_report_shows_uncertainty_fields_and_llm_sections():
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
    uncertainty = MinimumTmUncertainty(
        component_a="CCO",
        component_b="OCCO",
        repeated_values=(218.0, 219.0, 220.0),
        mean_tm_k=219.0,
        std_tm_k=1.0,
        min_tm_k=218.0,
        max_tm_k=220.0,
        trust_score=0.88,
        uncertainty_flag="low",
        explanation="demo",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )
    annotated = AnnotatedResult(result=result, uncertainty=uncertainty, trust_score=0.88, ranking_score=0.88)

    report = format_report(
        [result],
        annotated_results=[annotated],
        explanation_notes=[ExplanationNote(smiles="OCCO", summary="ranked highly", evidence=["low min Tm"])],
        critique_notes=[CritiqueNote(smiles="OCCO", assessment="advisory only", concerns=["possible outlier"])],
        brainstorm_candidates=[CandidateBrainstorm(smiles="OCCO", rationale="polyol", family="polyol")],
    )

    assert "trust=0.88" in report
    assert "std=1.00 K" in report
    assert "flag=low" in report
    assert "LLM brainstorm" in report
    assert "LLM explanations" in report
    assert "LLM critique" in report
