from __future__ import annotations
from des_multi_agent.chemical_lesson_summary import (
    ChemistryLessonSummary,
    ChemistryLessonSummaryConfig,
    build_chemistry_lesson_summary,
)
from des_multi_agent.evaluation import DesResult
from des_multi_agent.memory_schema import RunCandidateSummary, RunLabel, RunMemory
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import CandidateProposal
from des_multi_agent.uncertainty.schemas import AnnotatedResult, MinimumTmUncertainty
class DummyCurvePrediction(CurvePrediction):
    pass
def _annotated_result(smiles_b: str, is_des: bool = True, trust_score: float = 0.9, ranking_score: float = 1.0) -> AnnotatedResult:
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[250.0, 240.0, 245.0],
        t1_k=298.15,
        t2_k=300.0,
        checkpoint_path="ckpt.pt",
    )
    result = DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=is_des,
        rationale="demo",
        min_tm_k=240.0,
    )
    uncertainty = MinimumTmUncertainty(
        component_a="CCO",
        component_b=smiles_b,
        repeated_values=(240.0, 241.0, 242.0),
        mean_tm_k=241.0,
        std_tm_k=1.0,
        min_tm_k=240.0,
        max_tm_k=242.0,
        trust_score=trust_score,
        uncertainty_flag="low",
        explanation="demo",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )
    return AnnotatedResult(result=result, uncertainty=uncertainty, trust_score=trust_score, ranking_score=ranking_score)
def test_empty_lesson_summary_is_blank():
    summary = build_chemistry_lesson_summary(
        component_a="CCO",
        annotated_results=[],
        candidate_proposals=[],
        run_memories=[],
        prior_pattern_memory=None,
        prior_lesson_summary=None,
        config=ChemistryLessonSummaryConfig(mode="adaptive"),
    )
    assert summary == ChemistryLessonSummary()
    assert summary.cycle_summary == []
    assert summary.run_summary == []
def test_cycle_lesson_from_des_hit_mentions_productive_pattern():
    summary = build_chemistry_lesson_summary(
        component_a="CCO",
        annotated_results=[_annotated_result("OCCO", is_des=True)],
        candidate_proposals=[
            CandidateProposal(
                smiles="OCCO",
                rationale="short diol",
                family="diol",
                source="llm",
                source_id="brainstorm",
            )
        ],
        run_memories=[],
        prior_pattern_memory=None,
        prior_lesson_summary=None,
        config=ChemistryLessonSummaryConfig(mode="adaptive", max_examples=3),
    )
    assert summary.productive_patterns == {"diol": 1}
    assert summary.representative_examples == ["OCCO"]
    assert any("productive" in note.lower() or "diol" in note for note in summary.cycle_summary)
    assert summary.confidence in {"low", "medium"}
def test_saved_labels_feed_lesson_summary_and_ignore_other_component():
    matching_memory = RunMemory(
        workflow="des",
        component_a="CCO",
        n=5,
        labels=[RunLabel(smiles_b="OCCO", label="good"), RunLabel(smiles_b="CC(=O)O", label="bad")],
        ranked_candidates=[
            RunCandidateSummary(smiles_b="OCCO", rank=1, min_tm_k=240.0, trust_score=0.9, uncertainty_flag="low", source="llm", source_id="brainstorm"),
            RunCandidateSummary(smiles_b="CC(=O)O", rank=2, min_tm_k=260.0, trust_score=0.7, uncertainty_flag="low", source="llm", source_id="brainstorm"),
        ],
    )
    other_memory = RunMemory(
        workflow="des",
        component_a="CCC",
        n=5,
        labels=[RunLabel(smiles_b="NCCN", label="good")],
        ranked_candidates=[],
    )
    summary = build_chemistry_lesson_summary(
        component_a="CCO",
        annotated_results=[],
        candidate_proposals=[],
        run_memories=[matching_memory, other_memory],
        prior_pattern_memory=None,
        prior_lesson_summary=None,
        config=ChemistryLessonSummaryConfig(mode="adaptive", max_examples=3),
    )
    joined = "\n".join(summary.run_summary + summary.notes + summary.cycle_summary)
    assert "Prior good labels: OCCO" in joined
    assert "Prior bad labels: CC(=O)O" in joined
    assert "NCCN" not in joined
    assert summary.representative_examples == ["OCCO", "CC(=O)O"]
