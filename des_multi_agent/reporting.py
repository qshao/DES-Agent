from __future__ import annotations

from collections.abc import Sequence

from .llm.schemas import CandidateBrainstorm, CandidateReview, CritiqueNote, ExplanationNote
from .predictors.designsolvents import ViscosityPrediction
from .schemas import CandidateProposal
from .uncertainty import AnnotatedResult


def format_report(
    results,
    annotated_results: Sequence[AnnotatedResult] | None = None,
    candidate_proposals: Sequence[CandidateProposal] | None = None,
    candidate_reviews: Sequence[CandidateReview] | None = None,
    explanation_notes: Sequence[ExplanationNote] | None = None,
    critique_notes: Sequence[CritiqueNote] | None = None,
    brainstorm_candidates: Sequence[CandidateBrainstorm] | None = None,
    llm_warnings: Sequence[str] | None = None,
    viscosity_predictions: Sequence[ViscosityPrediction] | None = None,
) -> str:
    proposal_by_smiles = {item.smiles: item for item in candidate_proposals or []}
    annotation_by_smiles = {item.result.curve.smiles_b: item for item in annotated_results or []}
    if annotation_by_smiles:
        lines = ["smiles_b | is_des | min_tm_k | source | trust | mean_tm_k | spread_k | std_k | uncertainty_flag | rationale"]
    else:
        lines = ["smiles_b | is_des | min_tm_k | source | rationale"]
    for r in results:
        proposal = proposal_by_smiles.get(r.curve.smiles_b)
        source_text = "heuristic"
        if proposal is not None:
            parts = [f"source={proposal.source}"]
            if proposal.source_id:
                parts.append(f"id={proposal.source_id}")
            if proposal.similarity_score is not None:
                parts.append(f"sim={proposal.similarity_score:.2f}")
            source_text = "; ".join(parts)
        annotation = annotation_by_smiles.get(r.curve.smiles_b)
        if annotation is None:
            lines.append(f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {source_text} | {r.rationale}")
            continue
        lines.append(
            f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {source_text} | "
            f"trust={annotation.trust_score:.2f} | mean={annotation.uncertainty.mean_tm_k:.2f} K | "
            f"spread={annotation.uncertainty.min_tm_k:.2f}-{annotation.uncertainty.max_tm_k:.2f} K | "
            f"std={annotation.uncertainty.std_tm_k:.2f} K | flag={annotation.uncertainty.uncertainty_flag} | {r.rationale}"
        )
    if candidate_reviews:
        lines.append("")
        lines.append("LLM candidate reviews:")
        for note in candidate_reviews:
            notes = "; ".join(note.notes) if note.notes else "-"
            lines.append(f"{note.smiles} | {note.decision} | confidence={note.confidence:.2f} | {note.rationale} | {notes}")
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
    if viscosity_predictions:
        lines.append("")
        lines.append("Viscosity predictions:")
        lines.append("smiles_a | smiles_b | viscosity | units | model | source")
        for pred in viscosity_predictions:
            lines.append(
                f"{pred.metadata.get('component_a', '?')} | {pred.metadata.get('component_b', '?')} | "
                f"{pred.value:.2f} | {pred.units} | {pred.model_name} | {pred.source}"
            )
    if llm_warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in llm_warnings:
            lines.append(f"- {warning}")
    return '\n'.join(lines)


def format_metal_binding_report(outcome) -> str:
    pred = outcome.prediction
    lines = ["metal_ion | ligand_smiles | value | units | model | source"]
    lines.append(
        f"{outcome.metal_ion} | {outcome.ligand_smiles} | {getattr(pred, 'value', float('nan')):.2f} | "
        f"{getattr(pred, 'units', '?')} | {getattr(pred, 'model_name', '?')} | {getattr(pred, 'source', '?')}"
    )
    if getattr(pred, 'warnings', None):
        lines.append("")
        lines.append("Warnings:")
        for warning in pred.warnings:
            lines.append(f"- {warning}")
    if getattr(outcome, 'warnings', None) and not getattr(pred, 'warnings', None):
        lines.append("")
        lines.append("Warnings:")
        for warning in outcome.warnings:
            lines.append(f"- {warning}")
    return '\n'.join(lines)
