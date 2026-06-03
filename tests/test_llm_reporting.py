from des_multi_agent.llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote
from des_multi_agent.reporting import format_report


def test_report_can_attach_optional_llm_notes():
    report = format_report(
        [],
        explanation_notes=[ExplanationNote(smiles="OCCO", summary="ranked highly", evidence=["low min Tm"])],
        critique_notes=[CritiqueNote(smiles="OCCO", assessment="advisory only", concerns=["possible outlier"])],
        brainstorm_candidates=[CandidateBrainstorm(smiles="OCCO", rationale="polyol", family="polyol")],
    )
    assert "LLM brainstorm" in report
    assert "LLM explanations" in report
    assert "LLM critique" in report
