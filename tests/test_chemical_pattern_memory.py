from __future__ import annotations


from des_multi_agent.chemical_pattern_memory import (
    ChemicalPatternMemory,
    ChemicalPatternMemoryConfig,
    apply_pattern_memory_bias,
    build_pattern_memory,
)
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import CandidateProposal
from des_multi_agent.uncertainty.schemas import AnnotatedResult, MinimumTmUncertainty


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


def test_empty_pattern_memory_is_inactive():
    memory = build_pattern_memory(
        component_a="CCO",
        annotated_results=[],
        candidate_proposals=[],
        run_memories=[],
        config=ChemicalPatternMemoryConfig(mode="adaptive"),
    )

    assert memory == ChemicalPatternMemory()
    assert memory.prompt_notes == []
    assert memory.ranking_bias_by_smiles == {}


def test_productive_family_from_des_hit_becomes_prompt_note():
    memory = build_pattern_memory(
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
        config=ChemicalPatternMemoryConfig(mode="adaptive", max_examples=3),
    )

    assert memory.productive_families == {"diol": 1}
    assert memory.good_examples == ["OCCO"]
    assert any("diol" in note for note in memory.prompt_notes)
    assert memory.confidence in {"low", "medium"}


def test_pattern_memory_bias_is_capped_and_family_based():
    memory = ChemicalPatternMemory(
        ranking_bias_by_smiles={"OCCO": 0.20},
        ranking_bias_by_family={"diol": 0.10},
        confidence="high",
    )
    proposals = [
        CandidateProposal(
            smiles="OCCO",
            rationale="short diol",
            family="diol",
            source="llm",
            source_id="brainstorm",
        )
    ]

    adjusted, notes = apply_pattern_memory_bias([_annotated_result("OCCO")], proposals, memory)

    assert adjusted[0].ranking_score == 1.30
    assert any("Applied chemical pattern memory" in note for note in notes)


def test_low_confidence_memory_has_smaller_effect():
    memory = ChemicalPatternMemory(
        ranking_bias_by_smiles={"OCCO": 0.20},
        confidence="low",
    )

    adjusted, _ = apply_pattern_memory_bias([_annotated_result("OCCO")], [], memory)

    assert adjusted[0].ranking_score == 1.10
