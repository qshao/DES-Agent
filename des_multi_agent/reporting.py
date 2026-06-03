from __future__ import annotations

from collections.abc import Sequence

from .llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote


def format_report(
    results,
    explanation_notes: Sequence[ExplanationNote] | None = None,
    critique_notes: Sequence[CritiqueNote] | None = None,
    brainstorm_candidates: Sequence[CandidateBrainstorm] | None = None,
    llm_warnings: Sequence[str] | None = None,
) -> str:
    lines = ["smiles_b | is_des | min_tm_k | rationale"]
    for r in results:
        lines.append(f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {r.rationale}")
    if brainstorm_candidates:
        lines.append("")
        lines.append("LLM brainstorm:")
        for note in brainstorm_candidates:
            lines.append(f"{note.smiles} | {note.family} | {note.rationale}")
    if explanation_notes:
        lines.append("")
        lines.append("LLM explanations:")
        for note in explanation_notes:
            evidence = "; ".join(note.evidence) if note.evidence else "-"
            lines.append(f"{note.smiles} | {note.summary} | {evidence}")
    if critique_notes:
        lines.append("")
        lines.append("LLM critique:")
        for note in critique_notes:
            concerns = "; ".join(note.concerns) if note.concerns else "-"
            lines.append(f"{note.smiles} | {note.assessment} | {concerns}")
    if llm_warnings:
        lines.append("")
        lines.append("LLM warnings:")
        for warning in llm_warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)
