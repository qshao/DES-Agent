from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence

from .llm.schemas import CandidateBrainstorm, CandidateReview, CritiqueNote, ExplanationNote
from .predictors.designsolvents import ViscosityPrediction
from .schemas import CandidateProposal
from .smiles_names import display_name
from .uncertainty import AnnotatedResult


def _format_tm_provenance(results, *, resolve_names: bool = True) -> list[str]:
    """Render the provenance of the pure-component melting points used to anchor
    each prediction. Returns an empty list when no result carries provenance, so
    programmatic results without it leave the report unchanged.
    """
    if not results or not any(getattr(r, "t2_source", None) is not None for r in results):
        return []

    def _row(smiles, tm_k, source, confidence) -> str:
        label = display_name(smiles) if resolve_names else smiles
        conf = f"{confidence:.2f}" if confidence is not None else "?"
        tm = f"{tm_k:.2f}" if tm_k is not None else "?"
        return f"{label} | {tm} | {source or '?'} | {conf}"

    lines = ["Melting-point inputs:", "component | Tm_K | source | confidence"]
    first = results[0]
    seen: set[str] = set()
    a_smiles = getattr(first.curve, "smiles_a", None)
    if a_smiles is not None and getattr(first, "t1_source", None) is not None:
        lines.append(_row(a_smiles, getattr(first.curve, "t1_k", None), first.t1_source, first.t1_confidence))
        seen.add(a_smiles)
    for r in results:
        b_smiles = r.curve.smiles_b
        if b_smiles in seen:
            continue
        seen.add(b_smiles)
        lines.append(_row(b_smiles, getattr(r.curve, "t2_k", None), getattr(r, "t2_source", None), getattr(r, "t2_confidence", None)))
    return lines


def _confidence_label(trust_score: float | None, uncertainty_flag: str) -> str:
    if uncertainty_flag == "low":
        return "high confidence"
    if uncertainty_flag == "high":
        return "low confidence — consider experimental verification"
    if trust_score is not None and trust_score >= 0.70:
        return "moderate-high confidence"
    return "moderate confidence — consider experimental verification"


def _format_summary_block(
    results,
    annotated_results: Sequence[AnnotatedResult] | None,
    *,
    resolve_names: bool = True,
) -> str:
    if not results:
        return "No candidates were screened."

    smiles_a = getattr(results[0].curve, "smiles_a", "?")
    component_a_label = display_name(smiles_a) if resolve_names else smiles_a
    n_screened = len(results)
    n_des = sum(1 for r in results if r.is_des)

    lines = [f"=== DES Search: {component_a_label} ==="]
    lines.append(f"Screened {n_screened} candidate(s). "
                 f"{n_des} predicted DES-former(s) (min Tm ≤ 260 K with ≥10% relative drop).")

    top = results[0]
    t1 = getattr(top.curve, "t1_k", None)
    t2 = getattr(top.curve, "t2_k", None)
    baseline = min(t1, t2) if t1 is not None and t2 is not None else None
    rel_drop_pct = (baseline - top.min_tm_k) / baseline * 100 if baseline else 0.0
    top_label = display_name(top.curve.smiles_b) if resolve_names else top.curve.smiles_b
    top_line = (
        f"Top candidate: {top_label} — "
        f"min Tm {top.min_tm_k:.1f} K (Δ{rel_drop_pct:.1f}%)"
    )

    annotation_by_smiles = {a.result.curve.smiles_b: a for a in (annotated_results or [])}
    top_annotation = annotation_by_smiles.get(top.curve.smiles_b)
    if top_annotation is not None:
        label = _confidence_label(top_annotation.trust_score, top_annotation.uncertainty.uncertainty_flag)
        top_line += f" | {label}"

    lines.append(top_line)
    lines.append("=" * (len(lines[0])))
    return "\n".join(lines)


def format_curve_chart(curve, title: str = "") -> str:
    """Render a compact ASCII sparkline of a Tm-vs-ratio curve.

    Height is fixed at 6 rows; the minimum-Tm point is marked with '*',
    other points with '·'.
    """
    ratios = curve.ratios
    tms = curve.tm_pred_k
    if not ratios or not tms:
        return ""

    n = len(ratios)
    tm_min = min(tms)
    tm_max = max(tms)
    height = 6
    span = tm_max - tm_min if tm_max > tm_min else 1.0
    min_idx = tms.index(tm_min)

    # Build grid: rows[0] = highest Tm, rows[-1] = lowest Tm
    grid = [[" "] * n for _ in range(height)]
    for col, tm in enumerate(tms):
        row = int(round((tm_max - tm) / span * (height - 1)))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*" if col == min_idx else "·"

    label = title or getattr(curve, "smiles_b", "")
    header = f"  Tm curve: {label}" if label else "  Tm curve"
    tm_hi = f"{tm_max:.0f} K"
    tm_lo = f"{tm_min:.0f} K"

    chart_lines = [header]
    for i, row in enumerate(grid):
        y_label = tm_hi if i == 0 else (tm_lo if i == height - 1 else "        ")
        chart_lines.append(f"  {y_label:>6} |{''.join(row)}")
    x_labels = "  " + " " * 9 + "  ".join(f"{r:.2f}" for r in ratios)
    chart_lines.append("         " + "-" * n)
    chart_lines.append(x_labels)
    return "\n".join(chart_lines)


def format_report(
    results,
    annotated_results: Sequence[AnnotatedResult] | None = None,
    candidate_proposals: Sequence[CandidateProposal] | None = None,
    candidate_reviews: Sequence[CandidateReview] | None = None,
    explanation_notes: Sequence[ExplanationNote] | None = None,
    critique_notes: Sequence[CritiqueNote] | None = None,
    brainstorm_candidates: Sequence[CandidateBrainstorm] | None = None,
    llm_warnings: Sequence[str] | None = None,
    memory_notes: Sequence[str] | None = None,
    viscosity_predictions: Sequence[ViscosityPrediction] | None = None,
    resolve_names: bool = True,
    show_curves: bool = False,
    contradiction_notes=None,
) -> str:
    proposal_by_smiles = {item.smiles: item for item in candidate_proposals or []}
    annotation_by_smiles = {item.result.curve.smiles_b: item for item in annotated_results or []}

    lines = [_format_summary_block(results, annotated_results, resolve_names=resolve_names), ""]

    if annotation_by_smiles:
        lines.append("compound | is_des | min_tm_k | eutectic_x_b | source | trust | mean_tm_k | spread_k | std_k | confidence | rationale")
    else:
        lines.append("compound | is_des | min_tm_k | eutectic_x_b | source | rationale")

    for r in results:
        compound_label = display_name(r.curve.smiles_b) if resolve_names else r.curve.smiles_b
        proposal = proposal_by_smiles.get(r.curve.smiles_b)
        source_text = "heuristic"
        if proposal is not None:
            parts = [f"source={proposal.source}"]
            if proposal.source_id:
                parts.append(f"id={proposal.source_id}")
            if proposal.similarity_score is not None:
                parts.append(f"sim={proposal.similarity_score:.2f}")
            source_text = "; ".join(parts)
        ensemble_note = ""
        if getattr(r.curve, "ensemble_checkpoint_count", None) is not None and getattr(r.curve, "ensemble_std_k", None) is not None:
            min_idx = min(range(len(r.curve.ratios)), key=lambda i: r.curve.tm_pred_k[i])
            ens_std = r.curve.ensemble_std_k[min_idx]
            ensemble_note = f" | ensemble_folds={r.curve.ensemble_checkpoint_count} ens_std={ens_std:.2f} K"
        eutectic_x_b = getattr(r, "eutectic_ratio_b", None)
        eutectic_str = f"{eutectic_x_b:.2f}" if eutectic_x_b is not None else "?"
        annotation = annotation_by_smiles.get(r.curve.smiles_b)
        if annotation is None:
            lines.append(f"{compound_label} | {r.is_des} | {r.min_tm_k:.2f} | {eutectic_str} | {source_text} | {r.rationale}{ensemble_note}")
            continue
        confidence = _confidence_label(annotation.trust_score, annotation.uncertainty.uncertainty_flag)
        lines.append(
            f"{compound_label} | {r.is_des} | {r.min_tm_k:.2f} | {eutectic_str} | {source_text} | "
            f"trust={annotation.trust_score:.2f} | mean={annotation.uncertainty.mean_tm_k:.2f} K | "
            f"spread={annotation.uncertainty.min_tm_k:.2f}-{annotation.uncertainty.max_tm_k:.2f} K | "
            f"std={annotation.uncertainty.std_tm_k:.2f} K | {confidence} | {r.rationale}{ensemble_note}"
        )
    nonphysical = [r for r in results if getattr(r, "eutectic_physical", True) is False]
    if nonphysical:
        lines.append("")
        lines.append("Physical sanity — non-physical eutectic (predicted Tm above both pure components):")
        for r in nonphysical:
            label = display_name(r.curve.smiles_b) if resolve_names else r.curve.smiles_b
            t1 = getattr(r.curve, "t1_k", None)
            t2 = getattr(r.curve, "t2_k", None)
            pure = f"min(T1,T2)={min(t1, t2):.1f} K" if t1 is not None and t2 is not None else "pure Tm"
            lines.append(f"- {label}: min Tm {r.min_tm_k:.1f} K > {pure} — treat prediction with caution")
    tm_provenance_lines = _format_tm_provenance(results, resolve_names=resolve_names)
    if tm_provenance_lines:
        lines.append("")
        lines.extend(tm_provenance_lines)
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
    if contradiction_notes:
        lines.append("")
        lines.append("LLM contradiction analysis:")
        for note in contradiction_notes:
            lines.append(f"{note.smiles} | {note.agreement} | {note.explanation}")
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
    if memory_notes:
        lines.append("")
        lines.append("Run memory:")
        for note in memory_notes:
            lines.append(f"- {note}")
    if show_curves and results:
        lines.append("")
        lines.append("Melting curves (Tm vs mole fraction of component B):")
        for r in results:
            title = display_name(r.curve.smiles_b) if resolve_names else r.curve.smiles_b
            lines.append(format_curve_chart(r.curve, title=title))
    return '\n'.join(lines)


def format_report_json(results, annotated_results=None, resolve_names: bool = True) -> str:
    """Render results as a JSON array for machine consumption."""
    annotation_by_smiles = {a.result.curve.smiles_b: a for a in (annotated_results or [])}
    rows = []
    for r in results:
        smiles_b = r.curve.smiles_b
        label = display_name(smiles_b) if resolve_names else smiles_b
        row: dict = {
            "smiles_b": smiles_b,
            "compound": label,
            "smiles_a": r.curve.smiles_a,
            "is_des": r.is_des,
            "min_tm_k": r.min_tm_k,
            "eutectic_ratio_b": getattr(r, "eutectic_ratio_b", None),
            "rationale": r.rationale,
        }
        ann = annotation_by_smiles.get(smiles_b)
        if ann is not None:
            row["trust_score"] = ann.trust_score
            row["uncertainty_flag"] = ann.uncertainty.uncertainty_flag
        if getattr(r, "t1_source", None) is not None or getattr(r, "t2_source", None) is not None:
            row["t1_source"] = getattr(r, "t1_source", None)
            row["t2_source"] = getattr(r, "t2_source", None)
            row["t1_confidence"] = getattr(r, "t1_confidence", None)
            row["t2_confidence"] = getattr(r, "t2_confidence", None)
        rows.append(row)
    return json.dumps(rows, indent=2)


def format_report_csv(results, annotated_results=None, resolve_names: bool = True) -> str:
    """Render results as CSV text."""
    annotation_by_smiles = {a.result.curve.smiles_b: a for a in (annotated_results or [])}
    buf = io.StringIO()
    fieldnames = [
        "smiles_b", "compound", "is_des", "min_tm_k", "eutectic_ratio_b",
        "trust_score", "uncertainty_flag",
        "t1_source", "t2_source", "t1_confidence", "t2_confidence",
        "rationale",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    def _conf(value) -> str:
        return f"{value:.2f}" if isinstance(value, (int, float)) else ""

    for r in results:
        smiles_b = r.curve.smiles_b
        label = display_name(smiles_b) if resolve_names else smiles_b
        ann = annotation_by_smiles.get(smiles_b)
        eutectic_x_b = getattr(r, "eutectic_ratio_b", None)
        writer.writerow({
            "smiles_b": smiles_b,
            "compound": label,
            "is_des": r.is_des,
            "min_tm_k": f"{r.min_tm_k:.2f}",
            "eutectic_ratio_b": f"{eutectic_x_b:.2f}" if eutectic_x_b is not None else "",
            "trust_score": f"{ann.trust_score:.2f}" if ann else "",
            "uncertainty_flag": ann.uncertainty.uncertainty_flag if ann else "",
            "t1_source": getattr(r, "t1_source", None) or "",
            "t2_source": getattr(r, "t2_source", None) or "",
            "t1_confidence": _conf(getattr(r, "t1_confidence", None)),
            "t2_confidence": _conf(getattr(r, "t2_confidence", None)),
            "rationale": r.rationale,
        })
    return buf.getvalue()


def format_report_prose(results, annotated_results=None, resolve_names: bool = True) -> str:
    """Render just the plain-language summary block."""
    return _format_summary_block(results, annotated_results, resolve_names=resolve_names)


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


def format_metal_binding_screen_report(outcome) -> str:
    """Render a ranked-candidate report for a MetalBindingScreenOutcome."""
    results = outcome.results
    top = results[0] if results else None
    top_str = f"{top.ligand_smiles} — log K = {top.log_k:.2f}" if top else "none"
    header_lines = [
        f"=== Metal Binding Screen: {outcome.metal_ion} ===",
        f"Screened {outcome.n_screened} candidate(s). "
        f"Top ligand: {top_str}",
        "=" * 52,
        "",
        "ligand | log_k | source | rationale",
    ]
    rows = []
    for r in results:
        src = f"source={r.source}"
        if r.source_id:
            src += f"; id={r.source_id}"
        rows.append(f"{r.ligand_smiles} | {r.log_k:.2f} | {src} | {r.rationale}")

    review_lines: list[str] = []
    if outcome.llm_candidate_reviews:
        review_lines.append("")
        review_lines.append("LLM ligand reviews:")
        for rev in outcome.llm_candidate_reviews:
            notes = "; ".join(rev.notes) if rev.notes else "-"
            review_lines.append(
                f"{rev.smiles} | {rev.decision} | confidence={rev.confidence:.2f} | {rev.rationale} | {notes}"
            )

    brainstorm_lines: list[str] = []
    if outcome.llm_brainstorm:
        brainstorm_lines.append("")
        brainstorm_lines.append("LLM brainstorm:")
        for b in outcome.llm_brainstorm:
            brainstorm_lines.append(f"{b.smiles} | {b.family} | {b.rationale}")

    warning_lines: list[str] = []
    if outcome.warnings:
        warning_lines.append("")
        warning_lines.append("Warnings:")
        for w in outcome.warnings:
            warning_lines.append(f"- {w}")

    return '\n'.join(header_lines + rows + review_lines + brainstorm_lines + warning_lines)


def format_selectivity_des_report(outcome: "SelectivityDesPipelineOutcome") -> str:
    """Render a two-section selectivity-DES pipeline report."""
    n_compatible = sum(1 for r in outcome.ligand_des_results if r.des_compatible)
    header = [
        f"=== Selectivity-DES Pipeline: {outcome.target_metal} over {outcome.competitor_metal} ===",
        f"Outer cycles run: {outcome.n_outer_cycles_run} | Converged: {'yes' if outcome.converged else 'no'}",
        f"Shortlisted ligands: {len(outcome.ligand_des_results)} | DES-compatible: {n_compatible}",
        "=" * 52,
    ]

    sec1 = [
        "",
        "--- Section 1: Selectivity Results ---",
        "",
        "ligand | log_k_target | log_k_competitor | delta_log_k | score | des_compatible",
    ]
    sel_results = outcome.selectivity_outcome.results if outcome.selectivity_outcome else []
    compatible_smiles = {
        r.ligand.ligand_smiles for r in outcome.ligand_des_results if r.des_compatible
    }
    for r in sel_results:
        des_flag = "yes" if r.ligand_smiles in compatible_smiles else "no"
        sec1.append(
            f"{r.ligand_smiles} | {r.log_k_target:.2f} | {r.log_k_competitor:.2f} | "
            f"{r.delta_log_k:.2f} | {r.composite_score:.2f} | {des_flag}"
        )

    sec2 = ["", "--- Section 2: DES Partners ---"]
    for ldr in outcome.ligand_des_results:
        r = ldr.ligand
        compat_str = "YES" if ldr.des_compatible else "NO"
        sec2.append(
            f"\nLigand: {r.ligand_smiles}  "
            f"(score={r.composite_score:.2f}, ΔlogK={r.delta_log_k:.2f}) "
            f"— DES-compatible: {compat_str}"
        )
        if ldr.des_results:
            sec2.append("  partner | min_tm_k | eutectic_ratio | tm_src | rationale")
            for dr in ldr.des_results:
                t2_source = getattr(dr, "t2_source", None)
                t2_confidence = getattr(dr, "t2_confidence", None)
                if isinstance(t2_source, str):
                    conf = f"{t2_confidence:.2f}" if isinstance(t2_confidence, (int, float)) else "?"
                    tm_src = f"{t2_source} ({conf})"
                else:
                    tm_src = "?"
                sec2.append(
                    f"  {dr.curve.smiles_b} | {dr.min_tm_k:.1f} | "
                    f"{dr.eutectic_ratio_b:.2f} | {tm_src} | {dr.rationale}"
                )
        else:
            sec2.append("  No DES partners found.")

    warning_lines: list[str] = []
    if outcome.warnings:
        warning_lines.append("")
        warning_lines.append("Warnings:")
        for w in outcome.warnings:
            warning_lines.append(f"- {w}")

    return "\n".join(header + sec1 + sec2 + warning_lines)


def format_metal_selectivity_report(outcome) -> str:
    """Render a ranked-candidate report for a SelectivityScreenOutcome."""
    results = outcome.results
    top = results[0] if results else None
    if top:
        top_str = (
            f"{top.ligand_smiles} — score={top.composite_score:.2f} "
            f"(ΔlogK={top.delta_log_k:.2f}, logK({outcome.target_metal})={top.log_k_target:.2f})"
        )
    else:
        top_str = "none"
    header_lines = [
        f"=== Metal Selectivity Screen: {outcome.target_metal} over {outcome.competitor_metal} ===",
        f"Screened {outcome.n_screened} candidate(s) over {outcome.n_cycles} cycle(s).",
        f"Top ligand: {top_str}",
        "=" * 52,
        "",
        "ligand | log_k_target | log_k_competitor | delta_log_k | score | source | rationale",
    ]
    rows = []
    for r in results:
        src = f"source={r.source}"
        if r.source_id:
            src += f"; id={r.source_id}"
        rows.append(
            f"{r.ligand_smiles} | {r.log_k_target:.2f} | {r.log_k_competitor:.2f} | "
            f"{r.delta_log_k:.2f} | {r.composite_score:.2f} | {src} | {r.rationale}"
        )

    review_lines: list[str] = []
    if outcome.llm_candidate_reviews:
        review_lines.append("")
        review_lines.append("LLM ligand reviews:")
        for rev in outcome.llm_candidate_reviews:
            notes = "; ".join(rev.notes) if rev.notes else "-"
            review_lines.append(
                f"{rev.smiles} | {rev.decision} | confidence={rev.confidence:.2f} | "
                f"{rev.rationale} | {notes}"
            )

    brainstorm_lines: list[str] = []
    if outcome.llm_brainstorm:
        brainstorm_lines.append("")
        brainstorm_lines.append("LLM brainstorm:")
        for b in outcome.llm_brainstorm:
            brainstorm_lines.append(f"{b.smiles} | {b.family} | {b.rationale}")

    warning_lines: list[str] = []
    if outcome.warnings:
        warning_lines.append("")
        warning_lines.append("Warnings:")
        for w in outcome.warnings:
            warning_lines.append(f"- {w}")

    return "\n".join(header_lines + rows + review_lines + brainstorm_lines + warning_lines)
