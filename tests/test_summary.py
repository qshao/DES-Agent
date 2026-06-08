from dataclasses import dataclass

from des_multi_agent.summary import build_command_summary, render_command_summary


@dataclass(frozen=True)
class _FakeDesOutcome:
    results: list[object]
    memory_notes: list[str]
    llm_warnings: list[str]
    candidate_reviews: list[object]
    brainstorm_candidates: list[object]
    explanation_notes: list[object]
    critique_notes: list[object]
    viscosity_predictions: list[object]


def test_format_command_summary_for_des_mentions_counts_and_memory():
    outcome = _FakeDesOutcome(
        results=[object(), object()],
        memory_notes=["Loaded reuse memory from runs/run_001/run.memory.json."],
        llm_warnings=[],
        candidate_reviews=[],
        brainstorm_candidates=[],
        explanation_notes=[],
        critique_notes=[],
        viscosity_predictions=[],
    )

    text = render_command_summary(build_command_summary("des", outcome))

    assert "summary:" in text
    assert "workflow: des" in text
    assert "ranked candidates: 2" in text
    assert "reuse memory: yes" in text
    assert build_command_summary("des", outcome).stream == "stdout"
