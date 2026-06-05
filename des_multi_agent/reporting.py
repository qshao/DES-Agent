from __future__ import annotations

from collections.abc import Sequence

from .llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote
from .uncertainty import AnnotatedResult


def format_report(
    results,
    annotated_results: Sequence[AnnotatedResult] | None = None,
    explanation_notes: Sequence[ExplanationNote] | None = None,
    critique_notes: Sequence[CritiqueNote] | None = None,
    brainstorm_candidates: Sequence[CandidateBrainstorm] | None = None,
    llm_warnings: Sequence[str] | None = None,
) -> str:
    annotation_by_smiles = {item.result.curve.smiles_b: item for item in annotated_results or []}
    has_annotations = bool(annotation_by_smiles)
    if has_annotations:
        lines = ["smiles_b | is_des | min_tm_k | rationale"]
    else:
        lines = ["smiles_b | is_des | min_tm_k | rationale"]
    for r in results:
        annotation = annotation_by_smiles.get(r.curve.smiles_b)
        if annotation is None:
            lines.append(f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {r.rationale}")
            continue
        lines.append(
            f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | "
            f"trust={annotation.trust_score:.2f} | mean={annotation.uncertainty.mean_tm_k:.2f} K | "
            f"spread={annotation.uncertainty.min_tm_k:.2f}-{annotation.uncertainty.max_tm_k:.2f} K | "
            f"std={annotation.uncertainty.std_tm_k:.2f} K | flag={annotation.uncertainty.uncertainty_flag} | {r.rationale}"
        )
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
