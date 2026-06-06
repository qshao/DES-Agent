from des_multi_agent.llm.schemas import CandidateBrainstorm, CandidateReview, CritiqueNote, ExplanationNote
from des_multi_agent.reporting import format_report


def test_report_can_attach_optional_llm_notes():
    report = format_report(
        [],
        candidate_reviews=[CandidateReview(smiles="OCCO", decision="keep", confidence=0.87, rationale="reviewed", notes=["demo"])],
        explanation_notes=[ExplanationNote(smiles="OCCO", summary="ranked highly", evidence=["low min Tm"])],
        critique_notes=[CritiqueNote(smiles="OCCO", assessment="advisory only", concerns=["possible outlier"])],
        brainstorm_candidates=[CandidateBrainstorm(smiles="OCCO", rationale="polyol", family="polyol")],
    )
    assert "LLM candidate reviews" in report
    assert "LLM brainstorm" in report
    assert "LLM explanations" in report
    assert "LLM critique" in report
