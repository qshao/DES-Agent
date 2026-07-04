from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import sys

from .candidate_generation import generate_candidates
from .chemical_lesson_summary import ChemistryLessonSummary, ChemistryLessonSummaryConfig, build_chemistry_lesson_summary
from .chemical_pattern_memory import ChemicalPatternMemory, ChemicalPatternMemoryConfig, apply_pattern_memory_bias, build_pattern_memory, merge_pattern_memories
from .chemistry.hbond import rank_by_hbond as _rank_by_hbond
from .chemistry_filter import canonicalize_smiles, filter_candidates
from .proposal_diversity import ProposalDiversityConfig, apply_proposal_diversity
from .config import DEFAULT_ABSOLUTE_TM_MAX_K, DEFAULT_RELATIVE_DROP_MIN
from .concurrency import run_concurrent
from .discovery import load_discovery_library, literature_lookup, merge_discovery_candidates, similarity_search
from .exporting import export_des_run_bundle
from .reporting import format_report
from .evaluation import DesResult, classify_des
from .llm.config import LLMConfig
from .llm.factory import build_llm_provider
from .llm.schemas import CandidateBrainstorm, CandidateReview, ChemistryAssessment, ChemistryNextStep, ContradictionNote, CritiqueNote, ExplanationNote
from .paths import resolve_existing_path
from .prediction import predict_curve, predict_curve_ensemble
from .run_memory import apply_run_memory_preferences, build_chemistry_advisor_memory_notes, build_run_memory, load_run_memory_history, write_run_memory
from .predictors.designsolvents import ViscosityPrediction, predict_viscosity
from .property_resolution import resolve_melting_point
from .ranking import rank_results, rank_results_composite
from .schemas import CandidateProposal, DesThresholds
from .uncertainty import (
    AnnotatedResult,
    MinimumTmUncertainty,
    UncertaintyPolicy,
    apply_uncertainty_policy,
    estimate_min_tm_uncertainty,
    rank_annotated_results,
)


@dataclass(frozen=True)
class SearchOutcome:
    results: list[DesResult]
    annotated_results: list[AnnotatedResult]
    candidate_proposals: list[CandidateProposal]
    candidate_reviews: list[CandidateReview]
    brainstorm_candidates: list[CandidateBrainstorm]
    explanation_notes: list[ExplanationNote]
    critique_notes: list[CritiqueNote]
    llm_warnings: list[str]
    contradiction_notes: list[ContradictionNote] = field(default_factory=list)
    memory_notes: list[str] = field(default_factory=list)
    advisor_assessments: list[ChemistryAssessment] = field(default_factory=list)
    advisor_next_steps: list[ChemistryNextStep] = field(default_factory=list)
    viscosity_predictions: list[ViscosityPrediction] = field(default_factory=list)
    claim_verdicts: list[object] = field(default_factory=list)
    chemical_pattern_memory: ChemicalPatternMemory = field(default_factory=ChemicalPatternMemory)
    chemistry_lesson_summary: ChemistryLessonSummary = field(default_factory=ChemistryLessonSummary)
    report_text: str = ""


_PROGRESS_STEPS = 6


def _progress(step: int, message: str) -> None:
    print(f"[{step}/{_PROGRESS_STEPS}] {message}", file=sys.stderr, flush=True)


def _merge_candidates(*candidate_groups) -> tuple[list, list[str]]:
    """Merge and deduplicate candidate groups by canonical SMILES.

    Returns (merged_candidates, dedup_notes) where dedup_notes lists any
    SMILES that were collapsed because they resolve to the same compound.
    """
    merged = []
    seen: dict[str, str] = {}  # canonical → first accepted SMILES
    dedup_notes: list[str] = []
    for group in candidate_groups:
        for candidate in group:
            smiles = candidate.smiles.strip()
            if not smiles:
                continue
            try:
                canonical = canonicalize_smiles(smiles)
            except ValueError:
                continue
            if canonical in seen:
                first = seen[canonical]
                if smiles != first:
                    dedup_notes.append(
                        f"Deduplicated candidate: {smiles} collapsed to {first} (same compound from multiple sources)"
                    )
                continue
            seen[canonical] = smiles
            merged.append(candidate)
    return merged, dedup_notes


def _search_context(component_a: str, n: int, checkpoint_path: str, config_path: str) -> str:
    return (
        f"Component A: {component_a}\n"
        f"Requested deterministic candidates: {n}\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"Config: {config_path}"
    )


def _build_iterative_context(
    base_context: str,
    prior_top: list,
    family_ledger: dict[str, int] | None = None,
    diversity_mode: str = "balanced",
    already_evaluated: set[str] | None = None,
    family_scores: dict[str, list[float]] | None = None,
    saturated_families: set[str] | None = None,
    failing_scaffolds: set[str] | None = None,
    family_ucb_scores: dict[str, float] | None = None,
    fg_hit_counts: dict[str, int] | None = None,
    fg_fail_counts: dict[str, int] | None = None,
) -> str:
    if not prior_top:
        return base_context
    lines = "\n".join(
        f"  - {r.curve.smiles_b}: min_tm_k={r.min_tm_k:.1f} K, is_des={r.is_des}"
        for r in prior_top[:5]
    )
    ctx = base_context + f"\nPrior cycle top results (bias generation toward these chemical families):\n{lines}"
    failed = [r for r in prior_top if not r.is_des][:4]
    if failed:
        fail_lines = "\n".join(
            f"  - {r.curve.smiles_b}: min_tm_k={r.min_tm_k:.1f} K (no eutectic)"
            for r in failed
        )
        ctx += f"\nPrior cycle evaluated but did NOT form DES (avoid similar structures):\n{fail_lines}"
    ctx += f"\nBrainstorm diversity mode: {diversity_mode}"
    if family_ledger:
        top_families = _build_prior_productive_family_summary(family_ledger, limit=3)
        fam_parts: list[str] = []
        for fam, count in top_families.items():
            scores = (family_scores or {}).get(fam, [])
            if scores:
                avg_tm = sum(scores) / len(scores)
                fam_parts.append(f"  - {fam}: {count} DES-positive hits, avg min_tm={avg_tm:.1f} K")
            else:
                fam_parts.append(f"  - {fam}: {count} DES-positive hits")
        ctx += "\nTop productive chemical families:\n" + "\n".join(fam_parts)
    if family_ucb_scores:
        # Show families ranked by UCB exploration value so LLM can focus effort
        # on high-value (unexplored or high hit-rate) families.
        ranked_ucb = sorted(family_ucb_scores.items(), key=lambda x: x[1], reverse=True)
        explore_worthy = [(f, v) for f, v in ranked_ucb if v >= 0.5][:4]
        depleted = [(f, v) for f, v in ranked_ucb if v < 0.5][:3]
        if explore_worthy:
            ctx += "\nFamilies worth exploring further (high UCB score = promising or under-sampled):\n" + "\n".join(
                f"  - {fam} (UCB={val:.2f})" for fam, val in explore_worthy
            )
        if depleted:
            ctx += "\nDepleted families (low UCB score = exhausted, avoid repeating):\n" + "\n".join(
                f"  - {fam} (UCB={val:.2f})" for fam, val in depleted
            )
    elif saturated_families:
        sat_list = sorted(saturated_families)[:5]
        ctx += "\nExhausted families (too many failures, avoid repeating):\n" + "\n".join(
            f"  - {fam}" for fam in sat_list
        )
    if failing_scaffolds:
        sc_list = sorted(failing_scaffolds)[:4]
        ctx += "\nFailing scaffold cores (consistently non-DES, avoid similar structures):\n" + "\n".join(
            f"  - {sc}" for sc in sc_list
        )
    if fg_hit_counts or fg_fail_counts:
        # Sub-family SAR: functional groups enriched in hits vs. failures.
        all_fg = set(fg_hit_counts or {}) | set(fg_fail_counts or {})
        fg_rates: list[tuple[str, float, int]] = []
        for fg in all_fg:
            h = (fg_hit_counts or {}).get(fg, 0)
            f = (fg_fail_counts or {}).get(fg, 0)
            total = h + f
            if total >= 2:
                fg_rates.append((fg, h / total, total))
        if fg_rates:
            productive_fg = sorted([(fg, r, n) for fg, r, n in fg_rates if r >= 0.5], key=lambda x: -x[1])[:4]
            avoid_fg = sorted([(fg, r, n) for fg, r, n in fg_rates if r < 0.5], key=lambda x: x[1])[:3]
            if productive_fg:
                ctx += "\nStructural features enriched in DES hits (prefer):\n" + "\n".join(
                    f"  - {fg}: {r*100:.0f}% hit rate ({n} trials)" for fg, r, n in productive_fg
                )
            if avoid_fg:
                ctx += "\nStructural features enriched in failures (avoid):\n" + "\n".join(
                    f"  - {fg}: {r*100:.0f}% hit rate ({n} trials)" for fg, r, n in avoid_fg
                )
    if already_evaluated:
        smiles_list = sorted(already_evaluated)[:20]
        ctx += "\nAlready evaluated — do not re-propose: " + ", ".join(smiles_list)
    return ctx


def _build_prior_productive_family_summary(family_ledger: dict[str, int] | None, limit: int = 3) -> dict[str, int]:
    if not family_ledger:
        return {}
    return dict(sorted(family_ledger.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _merge_llm_diversity_overrides(
    llm_cfg: Mapping[str, object] | LLMConfig | None,
    proposal_diversity_cfg: Mapping[str, object] | None,
) -> Mapping[str, object] | LLMConfig | None:
    if llm_cfg is None or not proposal_diversity_cfg:
        return llm_cfg
    overrides = {
        key: proposal_diversity_cfg[key]
        for key in ("diversity_mode", "max_families", "family_bias_strength")
        if key in proposal_diversity_cfg
    }
    if not overrides:
        return llm_cfg
    if isinstance(llm_cfg, LLMConfig):
        return replace(llm_cfg, **overrides)
    if isinstance(llm_cfg, Mapping):
        merged = dict(llm_cfg)
        merged.update(overrides)
        return merged
    return llm_cfg


def _build_proposal_diversity_config(
    llm_cfg: Mapping[str, object] | LLMConfig | None,
    proposal_diversity_cfg: Mapping[str, object] | None,
) -> ProposalDiversityConfig:
    base_cfg = ProposalDiversityConfig.from_mapping(llm_cfg)
    if not proposal_diversity_cfg:
        return base_cfg
    per_family_budget = proposal_diversity_cfg.get("per_family_budget", base_cfg.per_family_budget)
    return ProposalDiversityConfig(
        max_similarity=float(proposal_diversity_cfg.get("max_similarity", base_cfg.max_similarity)),
        deduplicate_exact=bool(proposal_diversity_cfg.get("deduplicate_exact", base_cfg.deduplicate_exact)),
        deduplicate_near=bool(proposal_diversity_cfg.get("deduplicate_near", base_cfg.deduplicate_near)),
        family_fallback=bool(proposal_diversity_cfg.get("family_fallback", base_cfg.family_fallback)),
        per_family_budget=None if per_family_budget is None else int(per_family_budget),
        diversity_mode=str(proposal_diversity_cfg.get("diversity_mode", base_cfg.diversity_mode)),
        max_families=int(proposal_diversity_cfg.get("max_families", base_cfg.max_families)),
        family_bias_strength=float(proposal_diversity_cfg.get("family_bias_strength", base_cfg.family_bias_strength)),
    )


def _append_pattern_memory_context(context: str, memory: ChemicalPatternMemory | None) -> str:
    if memory is None or not memory.prompt_notes:
        return context
    lines = [context, "", "Chemical lessons from prior predictions:"]
    lines.extend(f"  - {note}" for note in memory.prompt_notes[:6])
    return "\n".join(lines)


def _append_chemistry_lesson_context(context: str, summary: ChemistryLessonSummary | None) -> str:
    if summary is None:
        return context
    notes: list[str] = []
    notes.extend(summary.run_summary[:6])
    if not notes:
        notes.extend(summary.cycle_summary[:6])
    if not notes:
        return context
    lines = [context, "", "Chemistry lessons from prior predictions:"]
    lines.extend(f"  - {note}" for note in notes[:6])
    if summary.next_steps:
        lines.append("  - Next steps: " + "; ".join(summary.next_steps[:2]))
    if summary.warnings:
        lines.append("  - Warnings: " + "; ".join(summary.warnings[:2]))
    return "\n".join(lines)


def _build_chemistry_advisor_context(base_context: str, annotated_results: list[AnnotatedResult], memory_notes: list[str]) -> str:
    lines = [base_context]
    if memory_notes:
        lines.append("Prior run memory:")
        lines.extend(f"  - {note}" for note in memory_notes[:6])
    if annotated_results:
        lines.append("Top ranked results:")
        for item in annotated_results[:5]:
            lines.append(
                f"  - {item.result.curve.smiles_b}: min_tm_k={item.result.min_tm_k:.1f} K, "
                f"trust={item.trust_score:.2f}, is_des={item.result.is_des}"
            )
    return "\n".join(lines)
def _fallback_uncertainty(
    component_a: str,
    component_b: str,
    checkpoint_path: str,
    config_path: str,
    reason: str,
) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a=component_a,
        component_b=component_b,
        repeated_values=(),
        mean_tm_k=float("inf"),
        std_tm_k=float("inf"),
        min_tm_k=float("inf"),
        max_tm_k=float("inf"),
        trust_score=0.0,
        uncertainty_flag="high",
        explanation=reason,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )


def _build_discovery_candidates(component_a: str, n: int, discovery_path: str | None, llm_warnings: list[str]) -> list[CandidateProposal]:
    if not discovery_path:
        return []
    try:
        library = load_discovery_library(discovery_path)
    except Exception as exc:
        llm_warnings.append(f"Discovery loading failed for {discovery_path}: {exc}")
        return []
    literature = literature_lookup(component_a, library)
    similar = similarity_search(component_a, library, limit=n)
    if not literature and not similar:
        llm_warnings.append(f"No discovery candidates found at {discovery_path}; using heuristic generator only.")
    return merge_discovery_candidates(literature, similar)


def _promote_brainstorm_candidates(candidates: list[CandidateBrainstorm]) -> list[CandidateProposal]:
    return [
        CandidateProposal(
            smiles=candidate.smiles,
            rationale=candidate.rationale,
            family=candidate.family,
            source="llm",
            source_id="brainstorm",
        )
        for candidate in candidates
    ]


def _review_top_candidates(
    provider,
    component_a: str,
    candidate_proposals: list[CandidateProposal],
    context: str,
    top_n: int,
    llm_warnings: list[str],
) -> tuple[list[CandidateReview], dict[str, CandidateReview]]:
    review_notes: list[CandidateReview] = []
    review_by_smiles: dict[str, CandidateReview] = {}
    top_proposals = candidate_proposals[: max(0, top_n)]
    results = run_concurrent(top_proposals, lambda p: provider.review_candidate(component_a, p.smiles, context))
    for proposal, res in zip(top_proposals, results):
        if res.error is not None:
            llm_warnings.append(f"LLM candidate review failed for {proposal.smiles}: {res.error}")
            continue
        review = res.value
        if review.smiles != proposal.smiles:
            llm_warnings.append(f"LLM candidate review returned wrong SMILES for {proposal.smiles}")
            continue
        review_notes.append(review)
        review_by_smiles[proposal.smiles] = review
    return review_notes, review_by_smiles


def _apply_candidate_reviews(
    candidate_proposals: list[CandidateProposal],
    reviews: Mapping[str, CandidateReview],
) -> tuple[list[CandidateProposal], dict[str, float]]:
    kept: list[CandidateProposal] = []
    review_penalty_by_smiles: dict[str, float] = {}
    for proposal in candidate_proposals:
        review = reviews.get(proposal.smiles)
        if review is None:
            kept.append(proposal)
            continue
        if review.decision == "reject":
            continue
        if review.decision == "deprioritize":
            review_penalty_by_smiles[proposal.smiles] = 0.25
        kept.append(proposal)
    return kept, review_penalty_by_smiles


def _apply_review_penalties(
    annotated_results: list[AnnotatedResult],
    review_penalty_by_smiles: Mapping[str, float],
) -> list[AnnotatedResult]:
    if not review_penalty_by_smiles:
        return list(annotated_results)
    adjusted: list[AnnotatedResult] = []
    for item in annotated_results:
        penalty = review_penalty_by_smiles.get(item.result.curve.smiles_b)
        if penalty is None:
            adjusted.append(item)
            continue
        adjusted.append(replace(item, ranking_score=max(0.0, item.ranking_score - penalty)))
    return rank_annotated_results(adjusted)


def _apply_hbond_bias(
    annotated_results: list[AnnotatedResult],
    component_a: str,
    *,
    weight: float = 0.10,
) -> list[AnnotatedResult]:
    """Adjust ranking scores by H-bond complementarity with component_a.

    Maps composite_score ∈ [0, 1] to an adjustment ∈ [-weight, +weight] so
    well-complemented candidates rise and poorly-complemented ones fall.
    Never raises; returns the input list unchanged on any error.
    """
    if not annotated_results:
        return annotated_results
    try:
        smiles_list = [item.result.curve.smiles_b for item in annotated_results]
        scored = _rank_by_hbond(component_a, smiles_list)
        adj_by_smiles = {
            smi: (score.composite_score - 0.5) * 2.0 * weight
            for smi, score in scored
        }
        adjusted = [
            replace(item, ranking_score=max(0.0, item.ranking_score + adj_by_smiles.get(item.result.curve.smiles_b, 0.0)))
            for item in annotated_results
        ]
        return rank_annotated_results(adjusted)
    except Exception:
        return annotated_results


_FAIL_FP_CACHE: dict[str, object] = {}  # smi → Morgan fingerprint; persists across cycles


def _apply_tanimoto_diversity_penalty(
    annotated_results: list[AnnotatedResult],
    prior_results_by_smiles: dict,
    *,
    similarity_cutoff: float = 0.70,
    penalty_scale: float = 0.10,
) -> list[AnnotatedResult]:
    """Demote candidates that are too similar to known DES-negative failures.

    For each new candidate, the max Tanimoto similarity to any previously evaluated
    DES-negative SMILES is computed (Morgan radius-2, 2048 bits). Candidates with
    similarity >= similarity_cutoff receive a penalty that scales linearly from 0
    at the cutoff to penalty_scale at similarity=1.0.

    Never raises; returns the input list unchanged on any error.
    """
    if not annotated_results or not prior_results_by_smiles:
        return annotated_results
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        fail_fps = []
        for smi, res in prior_results_by_smiles.items():
            if not res.is_des:
                fp = _FAIL_FP_CACHE.get(smi)
                if fp is None:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                        _FAIL_FP_CACHE[smi] = fp
                if fp is not None:
                    fail_fps.append(fp)

        if not fail_fps:
            return annotated_results

        adjusted = []
        for item in annotated_results:
            mol = Chem.MolFromSmiles(item.result.curve.smiles_b)
            if mol is None:
                adjusted.append(item)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, fail_fps)
            max_sim = max(sims) if sims else 0.0
            if max_sim >= similarity_cutoff:
                penalty = penalty_scale * (max_sim - similarity_cutoff) / (1.0 - similarity_cutoff)
                adjusted.append(replace(item, ranking_score=max(0.0, item.ranking_score - penalty)))
            else:
                adjusted.append(item)
        return rank_annotated_results(adjusted)
    except Exception:
        return annotated_results


def _canonical(smiles: str) -> str:
    """Return canonical SMILES, or the input string if unparseable."""
    try:
        return canonicalize_smiles(smiles)
    except ValueError:
        return smiles


def _grade_partner_reality(
    component_a: str,
    candidate_smiles: list[str],
    llm_smiles: set[str],
) -> tuple[list[object], dict[str, float], set[str]]:
    """Grade LLM-sourced partner proposals against reality.

    Returns (verdicts, penalty_by_smiles, drop_smiles). Non-LLM candidates are
    skipped (real by construction). ``llm_smiles`` must contain canonical forms;
    each candidate is canonicalized before the membership check so the dedup race
    (heuristic wins with canonical form, LLM proposed a non-canonical spelling)
    does not cause grading to be silently skipped. Never raises.
    """
    from .chemistry.claim_grounding import ground_partner_reality

    verdicts: list[object] = []
    penalties: dict[str, float] = {}
    drops: set[str] = set()
    for smi in candidate_smiles:
        if _canonical(smi) not in llm_smiles:
            continue
        rv = ground_partner_reality(component_a, smi)
        verdicts.append(rv)
        if rv.disposition == "demote":
            penalties[smi] = max(penalties.get(smi, 0.0), rv.penalty)
        elif rv.disposition == "drop":
            drops.add(smi)
    return verdicts, penalties, drops


def _predict_viscosity_predictions(
    component_a: str,
    proposals: list[CandidateProposal],
    model_path: str | None,
    llm_warnings: list[str],
) -> list[ViscosityPrediction]:
    predictions: list[ViscosityPrediction] = []
    if model_path is None:
        return predictions
    for proposal in proposals:
        try:
            predictions.append(predict_viscosity(component_a, proposal.smiles, model_path=model_path, allow_fallback=True))
        except Exception as exc:
            llm_warnings.append(f"Viscosity prediction failed for {proposal.smiles}: {exc}")
    return predictions


def _ensemble_ci(result) -> dict:
    """Return CI columns if ensemble std is available, else empty dict."""
    std_k = getattr(result.curve, "ensemble_std_k", None)
    if not std_k:
        return {}
    tms = result.curve.tm_pred_k
    min_idx = min(range(len(tms)), key=lambda i: tms[i])
    std = std_k[min_idx]
    return {
        "ensemble_ci_low_k": tms[min_idx] - 2.0 * std,
        "ensemble_ci_high_k": tms[min_idx] + 2.0 * std,
    }


def _build_des_export_payload(
    outcome: SearchOutcome,
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str,
) -> dict[str, object]:
    proposal_by_smiles = {item.smiles: item for item in outcome.candidate_proposals}
    ranked_results: list[dict[str, object]] = []
    for rank, annotated in enumerate(outcome.annotated_results, start=1):
        result = annotated.result
        proposal = proposal_by_smiles.get(result.curve.smiles_b)
        ranked_results.append(
            {
                "rank": rank,
                "smiles_a": result.curve.smiles_a,
                "smiles_b": result.curve.smiles_b,
                "is_des": result.is_des,
                "absolute_pass": result.absolute_pass,
                "relative_pass": result.relative_pass,
                "min_tm_k": result.min_tm_k,
                "rationale": result.rationale,
                "source": proposal.source if proposal is not None else "heuristic",
                "source_id": proposal.source_id if proposal is not None else "",
                "similarity_score": proposal.similarity_score if proposal is not None else None,
                "reference_note": proposal.reference_note if proposal is not None else "",
                "trust_score": annotated.trust_score,
                "uncertainty_flag": annotated.uncertainty.uncertainty_flag,
                "ranking_score": annotated.ranking_score,
                "uncertainty": asdict(annotated.uncertainty),
                **_ensemble_ci(result),
            }
        )
    return {
        "workflow": "des",
        "component_a": component_a,
        "n": n,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "results": ranked_results,
        "candidate_proposals": [asdict(item) for item in outcome.candidate_proposals],
        "candidate_reviews": [asdict(item) for item in outcome.candidate_reviews],
        "brainstorm_candidates": [asdict(item) for item in outcome.brainstorm_candidates],
        "explanation_notes": [asdict(item) for item in outcome.explanation_notes],
        "critique_notes": [asdict(item) for item in outcome.critique_notes],
        "viscosity_predictions": [asdict(item) for item in outcome.viscosity_predictions],
        "contradiction_notes": [asdict(n) for n in outcome.contradiction_notes],
        "memory_notes": list(outcome.memory_notes),
        "warnings": list(outcome.llm_warnings),
        "chemistry_lesson_summary": asdict(outcome.chemistry_lesson_summary),
    }


def _resolve_des_output_dir(output_dir: str | None, save_run_memory_path: str | None) -> Path | None:
    if output_dir:
        return Path(output_dir)
    if save_run_memory_path:
        return Path(save_run_memory_path).parent
    return None


def _generate_analogue_candidates(
    prior_cycle_top_results: list | None,
    *,
    near_miss_window_k: float = 15.0,
    transform_weights: dict[str, float] | None = None,
) -> list[CandidateProposal]:
    """Feature 2: expand top DES hits and near-misses from the prior cycle.

    Near-misses (not DES but min_tm_k within *near_miss_window_k* of the threshold)
    are often the most informative seeds — one structural modification may push them
    over the DES boundary.  *transform_weights* (name → Laplace-smoothed hit rate)
    biases the transform order toward historically productive transforms.
    """
    if not prior_cycle_top_results:
        return []
    from .analogue_expansion import generate_analogues_tagged
    proposals: list[CandidateProposal] = []

    top_des = [r for r in prior_cycle_top_results if r.is_des][:2]
    for r in top_des:
        seed_smi = r.curve.smiles_b
        for analogue_smi, tx_name in generate_analogues_tagged(seed_smi, max_n=4, transform_weights=transform_weights):
            proposals.append(CandidateProposal(
                smiles=analogue_smi,
                rationale=f"structural analogue of best DES hit {seed_smi} (min_tm={r.min_tm_k:.1f} K, via {tx_name})",
                family="analogue",
                source="analogue",
                source_id=seed_smi,
            ))

    # Near-miss expansion: candidates just above the DES threshold may cross it with
    # a small structural change. Use the lowest min_tm_k among non-DES results as the
    # threshold proxy (works without needing the exact DesThresholds value here).
    non_des = [r for r in prior_cycle_top_results if not r.is_des]
    if non_des:
        best_non_des_tm = min(r.min_tm_k for r in non_des)
        near_misses = sorted(
            [r for r in non_des if r.min_tm_k <= best_non_des_tm + near_miss_window_k],
            key=lambda r: r.min_tm_k,
        )[:2]
        for r in near_misses:
            seed_smi = r.curve.smiles_b
            for analogue_smi, tx_name in generate_analogues_tagged(seed_smi, max_n=3, transform_weights=transform_weights):
                proposals.append(CandidateProposal(
                    smiles=analogue_smi,
                    rationale=f"near-miss analogue of {seed_smi} (min_tm={r.min_tm_k:.1f} K, just above DES threshold, via {tx_name})",
                    family="analogue",
                    source="near_miss_analogue",
                    source_id=seed_smi,
                ))

    return proposals


def _load_candidates_file(path: str) -> list[CandidateProposal]:
    """Read one SMILES or molecule name per line from a file; skip blanks and # comments."""
    from .chemistry.name_resolution import resolve_to_smiles

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    proposals: list[CandidateProposal] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            smiles = resolve_to_smiles(raw)
        except ValueError as exc:
            print(f"[WARNING] candidates file: skipping {raw!r} — {exc}", file=sys.stderr, flush=True)
            continue
        proposals.append(CandidateProposal(smiles=smiles, rationale="from file", family="unknown", source="file", source_id=""))
    return proposals


def run_search_report(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    thresholds: DesThresholds | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
    llm_cfg: Mapping[str, object] | None = None,
    llm_request_fn=None,
    discovery_path: str | None = None,
    viscosity_model_path: str | None = None,
    save_run_memory_path: str | None = None,
    reuse_run_path: str | None = None,
    output_dir: str | None = None,
    ensemble_checkpoints: list[str] | None = None,
    candidates_file: str | None = None,
    viscosity_weight: float = 0.3,
    viscosity_threshold_cp: float | None = None,
    prior_cycle_top_results: list | None = None,
    prior_family_ledger: dict[str, int] | None = None,
    proposal_diversity_cfg: Mapping[str, object] | None = None,
    prior_pattern_memory: ChemicalPatternMemory | None = None,
    prior_chemistry_lesson_summary: ChemistryLessonSummary | None = None,
    chemical_pattern_memory_mode: str = "adaptive",
    pattern_memory_max_examples: int = 3,
    prior_results_by_smiles: dict | None = None,
    prior_uncertainty_by_smiles: dict | None = None,
    prior_evaluated_smiles: set[str] | None = None,
    prior_family_scores: dict[str, list[float]] | None = None,
    prior_saturated_families: set[str] | None = None,
    prior_failing_scaffolds: set[str] | None = None,
    prior_family_ucb_scores: dict[str, float] | None = None,
    prior_transform_weights: dict[str, float] | None = None,
    prior_fg_hit_counts: dict[str, int] | None = None,
    prior_fg_fail_counts: dict[str, int] | None = None,
):
    # D1 — validate component_a SMILES before any expensive work
    try:
        canonicalize_smiles(component_a)
    except ValueError:
        raise ValueError(f"Invalid component_a SMILES: {component_a}")

    checkpoint_path = resolve_existing_path(checkpoint_path)
    config_path = resolve_existing_path(config_path)
    llm_warnings: list[str] = []
    all_dedup_notes: list[str] = []

    # D4 — batch file bypasses LLM candidate generation, but still allows
    # analogue expansion of prior hits in cycle 2+ (Feature 2).
    if candidates_file is not None:
        file_proposals = _load_candidates_file(candidates_file)
        _progress(1, f"Using {len(file_proposals)} candidate(s) from file: {candidates_file}")
        heuristic_candidates = file_proposals
        discovery_candidates: list[CandidateProposal] = []
        analogue_candidates = _generate_analogue_candidates(prior_cycle_top_results, transform_weights=prior_transform_weights)
    else:
        _progress(1, f"Generating candidates for {component_a}...")
        heuristic_candidates = generate_candidates(component_a, n=n, constraints=None)
        discovery_candidates = _build_discovery_candidates(component_a, n, discovery_path, llm_warnings)
        # Feature 2: adaptive heuristic diversity — expand best prior DES hits.
        # Only runs in cycle 2+ (when prior_cycle_top_results is populated).
        analogue_candidates = _generate_analogue_candidates(prior_cycle_top_results, transform_weights=prior_transform_weights)

    candidate_proposals, notes = _merge_candidates(discovery_candidates, heuristic_candidates, analogue_candidates)
    all_dedup_notes.extend(notes)

    pattern_cfg = ChemicalPatternMemoryConfig(mode=chemical_pattern_memory_mode, max_examples=pattern_memory_max_examples)
    reuse_memories: list = []
    reuse_pattern_memory = ChemicalPatternMemory()
    reuse_run_note: str | None = None
    if reuse_run_path:
        reuse_memories = load_run_memory_history(reuse_run_path)
        reuse_pattern_memory = build_pattern_memory(
            component_a=component_a,
            annotated_results=[],
            candidate_proposals=candidate_proposals,
            run_memories=reuse_memories,
            config=pattern_cfg,
        )
        reuse_run_note = f"Loaded reuse memory from {reuse_run_path} ({len(reuse_memories)} run memory file(s))."
    active_pattern_memory = merge_pattern_memories(prior_pattern_memory, reuse_pattern_memory)

    review_context = _append_pattern_memory_context(_search_context(component_a, n, str(checkpoint_path), str(config_path)), active_pattern_memory)
    review_context = _append_chemistry_lesson_context(review_context, prior_chemistry_lesson_summary)
    effective_llm_cfg = _merge_llm_diversity_overrides(llm_cfg, proposal_diversity_cfg)
    provider = build_llm_provider(effective_llm_cfg, request_fn=llm_request_fn) if effective_llm_cfg else None
    llm_candidates: list[CandidateBrainstorm] = []
    candidate_reviews: list[CandidateReview] = []
    review_penalties: dict[str, float] = {}
    if provider is not None and candidates_file is None:
        try:
            brainstorm_context = review_context
            prior_productive_families = _build_prior_productive_family_summary(prior_family_ledger)
            if prior_cycle_top_results:
                brainstorm_context = _build_iterative_context(
                    review_context,
                    prior_cycle_top_results,
                    family_ledger=prior_family_ledger,
                    diversity_mode=getattr(provider, "diversity_mode", "balanced"),
                    already_evaluated=prior_evaluated_smiles,
                    family_scores=prior_family_scores,
                    saturated_families=prior_saturated_families,
                    failing_scaffolds=prior_failing_scaffolds,
                    family_ucb_scores=prior_family_ucb_scores,
                    fg_hit_counts=prior_fg_hit_counts,
                    fg_fail_counts=prior_fg_fail_counts,
                )
            llm_candidates = provider.brainstorm_candidates(
                component_a,
                None,
                brainstorm_context,
                max_families=getattr(provider, "max_families", 6),
                diversity_mode=getattr(provider, "diversity_mode", "balanced"),
                family_bias_strength=getattr(provider, "family_bias_strength", 0.5),
                prior_productive_families=prior_productive_families,
            )
        except Exception as exc:
            llm_warnings.append(f"LLM brainstorming failed: {exc}")
            llm_candidates = []
    candidate_proposals, notes = _merge_candidates(candidate_proposals, _promote_brainstorm_candidates(llm_candidates))
    all_dedup_notes.extend(notes)

    diversity_cfg = _build_proposal_diversity_config(effective_llm_cfg, proposal_diversity_cfg)
    diversity_result = apply_proposal_diversity(component_a, candidate_proposals, diversity_cfg)
    candidate_proposals = diversity_result.accepted
    all_dedup_notes.extend(diversity_result.notes)
    if diversity_result.suggested_families:
        all_dedup_notes.append(
            "Proposal diversity fallback suggested adjacent families: "
            + ", ".join(diversity_result.suggested_families)
        )

    _progress(2, f"Filtering {len(candidate_proposals)} candidate(s)...")
    _filter_kw = {"failing_scaffolds": prior_failing_scaffolds} if prior_failing_scaffolds else {}
    filtered = filter_candidates(component_a, candidate_proposals, **_filter_kw)
    thresholds = thresholds or DesThresholds(
        absolute_tm_max_k=DEFAULT_ABSOLUTE_TM_MAX_K,
        relative_drop_min=DEFAULT_RELATIVE_DROP_MIN,
    )
    if provider is not None and filtered:
        try:
            review_candidates, review_by_smiles = _review_top_candidates(
                provider,
                component_a,
                filtered,
                review_context,
                min(max(n, 0), len(filtered)),
                llm_warnings,
            )
            candidate_reviews = review_candidates
            filtered, review_penalties = _apply_candidate_reviews(filtered, review_by_smiles)
        except Exception as exc:
            llm_warnings.append(f"LLM candidate review pipeline failed: {exc}")
    _progress(3, f"Running ML predictions on {len(filtered)} candidate(s)"
              + (f" (ensemble, {len(ensemble_checkpoints)} folds)" if ensemble_checkpoints else "") + "...")
    component_a_tp = resolve_melting_point(component_a)
    results = []
    tm_estimates: dict = {}
    for proposal in filtered:
        try:
            canonical_b = canonicalize_smiles(proposal.smiles)
        except ValueError:
            canonical_b = proposal.smiles
        if prior_results_by_smiles and canonical_b in prior_results_by_smiles:
            results.append(prior_results_by_smiles[canonical_b])
            continue
        component_b_tp = resolve_melting_point(proposal.smiles)
        try:
            if ensemble_checkpoints:
                curve = predict_curve_ensemble(
                    component_a,
                    proposal.smiles,
                    t1_k=component_a_tp.tm_k,
                    t2_k=component_b_tp.tm_k,
                    checkpoint_paths=ensemble_checkpoints,
                    config_path=config_path,
                )
            else:
                curve = predict_curve(
                    component_a,
                    proposal.smiles,
                    t1_k=component_a_tp.tm_k,
                    t2_k=component_b_tp.tm_k,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                )
        except Exception as exc:
            # F1 — skip this candidate rather than aborting the whole run
            llm_warnings.append(f"Prediction failed for {proposal.smiles}: {exc}")
            continue
        result = classify_des(curve, thresholds)
        result = replace(
            result,
            t1_source=getattr(component_a_tp, "source", None),
            t1_confidence=getattr(component_a_tp, "confidence", None),
            t2_source=getattr(component_b_tp, "source", None),
            t2_confidence=getattr(component_b_tp, "confidence", None),
        )
        tm_estimates[proposal.smiles] = (component_a_tp, component_b_tp)
        results.append(result)
    # H2 — predict viscosity early so it can influence ranking
    viscosity_predictions = _predict_viscosity_predictions(component_a, filtered, viscosity_model_path, llm_warnings)
    visc_by_smiles_b = {p.component_b: p.value for p in viscosity_predictions}
    if visc_by_smiles_b:
        ranked = rank_results_composite(
            results, visc_by_smiles_b,
            viscosity_weight=viscosity_weight,
            viscosity_threshold_cp=viscosity_threshold_cp,
        )
    else:
        ranked = rank_results(results)
    _progress(4, f"Estimating uncertainty for {len(ranked)} result(s)...")
    policy = uncertainty_policy or UncertaintyPolicy()
    uncertainty_by_smiles: dict[str, MinimumTmUncertainty] = {}
    for result in ranked:
        smiles_b = result.curve.smiles_b
        try:
            canonical_b = canonicalize_smiles(smiles_b)
        except ValueError:
            canonical_b = smiles_b
        if prior_uncertainty_by_smiles and canonical_b in prior_uncertainty_by_smiles:
            uncertainty_by_smiles[smiles_b] = prior_uncertainty_by_smiles[canonical_b]
            continue
        pre = tm_estimates.get(smiles_b)
        try:
            uncertainty_by_smiles[smiles_b] = estimate_min_tm_uncertainty(
                component_a,
                smiles_b,
                str(checkpoint_path),
                str(config_path),
                _est_a=pre[0] if pre else None,
                _est_b=pre[1] if pre else None,
            )
        except Exception as exc:
            llm_warnings.append(f"Uncertainty estimation failed for {smiles_b}: {exc}")
            uncertainty_by_smiles[smiles_b] = _fallback_uncertainty(
                component_a,
                smiles_b,
                str(checkpoint_path),
                str(config_path),
                f"Uncertainty estimation failed: {exc}",
            )
    annotated_results = apply_uncertainty_policy(ranked, uncertainty_by_smiles, policy)
    annotated_results = _apply_review_penalties(annotated_results, review_penalties)
    annotated_results = _apply_hbond_bias(annotated_results, component_a)
    annotated_results = _apply_tanimoto_diversity_penalty(annotated_results, prior_results_by_smiles)
    memory_notes: list[str] = list(all_dedup_notes)
    if reuse_run_note is not None:
        memory_notes.insert(0, reuse_run_note)
    if reuse_memories:
        annotated_results, reuse_notes = apply_run_memory_preferences(
            annotated_results=annotated_results,
            memory=reuse_memories,
            component_a=component_a,
        )
        memory_notes.extend(reuse_notes)
    annotated_results, pattern_notes = apply_pattern_memory_bias(
        annotated_results,
        candidate_proposals,
        active_pattern_memory,
    )
    memory_notes.extend(pattern_notes)
    final_results = [item.result for item in annotated_results]
    explanation_notes: list[ExplanationNote] = []
    critique_notes: list[CritiqueNote] = []
    _progress(5, "Running LLM annotations..." if provider is not None else "Skipping LLM annotations (no provider configured).")
    if provider is not None:
        try:
            explanation_notes = provider.generate_explanations(final_results, review_context)
        except Exception as exc:
            llm_warnings.append(f"LLM explanation generation failed: {exc}")
        try:
            critique_notes = provider.critique_results(final_results, review_context)
        except Exception as exc:
            llm_warnings.append(f"LLM critique generation failed: {exc}")
    # Pre-compute plausibility evidence for the contradiction prompt
    contradiction_facts_block = ""
    try:
        from .chemistry.claim_grounding import ground_des_plausibility as _gdp
        plines = []
        for r in final_results:
            pv = _gdp(component_a, r.curve.smiles_b)
            label_desc = f"label={pv.detail.split('label=')[1].split(',')[0]}" if "label=" in pv.detail else pv.detail[:60]
            plines.append(f"- {r.curve.smiles_b}: DES plausibility {pv.status} ({label_desc})")
        if plines:
            contradiction_facts_block = "Deterministic H-bond plausibility:\n" + "\n".join(plines)
    except Exception:
        pass  # non-fatal; fall back to empty facts block
    contradiction_notes: list[ContradictionNote] = []
    if provider is not None:
        try:
            contradiction_notes = provider.detect_contradictions(final_results, review_context, facts_block=contradiction_facts_block)
        except Exception as exc:
            llm_warnings.append(f"LLM contradiction detection failed: {exc}")
    # Deterministic grounding: verify family + DES plausibility claims
    from .chemistry.claim_grounding import ground_des_plausibility, ground_family
    claim_verdicts: list[object] = []
    grounding_penalty_by_smiles: dict[str, float] = {}
    contradicted_family_smiles: set[str] = set()
    drop_smiles: set[str] = set()
    try:
        family_by_smiles = {_canonical(c.smiles): c.family for c in llm_candidates}
        # Accumulate claim-level grounding (DES plausibility + family) tagged by
        # the SMILES it describes, plus the SMILES that plausibility contradicted.
        # The final claim_verdicts / warnings are assembled below, once reality
        # grading has decided which candidates are dropped, so a dropped molecule
        # leaves no stale verdict behind (only the reality verdict explaining it).
        tagged_verdicts: list[tuple[str, object]] = []   # (smiles, verdict)
        tagged_warnings: list[tuple[str, str]] = []       # (smiles, warning)
        plausibility_contradicted: set[str] = set()
        for item in annotated_results:
            smiles_b = item.result.curve.smiles_b
            # Ground DES plausibility
            plausibility_v = ground_des_plausibility(component_a, smiles_b)
            tagged_verdicts.append((smiles_b, plausibility_v))
            if plausibility_v.status == "contradicted":
                plausibility_contradicted.add(smiles_b)
                grounding_penalty_by_smiles[smiles_b] = max(
                    grounding_penalty_by_smiles.get(smiles_b, 0.0),
                    plausibility_v.penalty,
                )
                tagged_warnings.append((
                    smiles_b,
                    f"[GROUNDING] DES plausibility contradicted for {smiles_b}: {plausibility_v.detail}",
                ))
            # Ground family classification if LLM provided one
            family = family_by_smiles.get(_canonical(smiles_b))
            if family:
                family_v = ground_family(smiles_b, family)
                tagged_verdicts.append((smiles_b, family_v))
                if family_v.status == "contradicted":
                    contradicted_family_smiles.add(smiles_b)
                    grounding_penalty_by_smiles[smiles_b] = max(
                        grounding_penalty_by_smiles.get(smiles_b, 0.0),
                        family_v.penalty,
                    )
                    tagged_warnings.append((
                        smiles_b,
                        f"[GROUNDING] Family '{family}' contradicted for {smiles_b}: {family_v.detail}",
                    ))
        ordered_smiles = [item.result.curve.smiles_b for item in annotated_results]
        reality_verdicts, reality_penalties, reality_drops = _grade_partner_reality(
            component_a, ordered_smiles, set(family_by_smiles)
        )
        drop_smiles.update(reality_drops)
        # Emit claim-level grounding, excluding any candidate reality dropped:
        # it is gone from the results, so its plausibility/family verdicts would
        # mislead. Its reality verdict (added below) explains the removal.
        for smi, verdict in tagged_verdicts:
            if smi not in drop_smiles:
                claim_verdicts.append(verdict)
        for smi, warning in tagged_warnings:
            if smi not in drop_smiles:
                llm_warnings.append(warning)
        # Reality verdicts: a "demote" for no complementarity duplicates the DES
        # plausibility "contradicted" already recorded for the same molecule, so
        # suppress that verdict and warning (the penalty is unchanged via max()).
        for verdict in reality_verdicts:
            smi = verdict.candidate_smiles or verdict.claim.split("partner reality: ", 1)[-1]
            if verdict.disposition == "demote" and smi in plausibility_contradicted:
                continue
            claim_verdicts.append(verdict)
        for smi, pen in reality_penalties.items():
            grounding_penalty_by_smiles[smi] = max(grounding_penalty_by_smiles.get(smi, 0.0), pen)
            if smi not in plausibility_contradicted:
                llm_warnings.append(f"[REALITY] no H-bond complementarity — demoted: {smi}")
        for smi in reality_drops:
            llm_warnings.append(f"[REALITY] not a real/sane molecule — dropped: {smi}")
    except Exception as exc:
        llm_warnings.append(f"[GROUNDING] Grounding failed (non-fatal): {exc}")
    if drop_smiles:
        annotated_results = [
            it for it in annotated_results
            if it.result.curve.smiles_b not in drop_smiles
        ]
    if grounding_penalty_by_smiles:
        annotated_results = _apply_review_penalties(annotated_results, grounding_penalty_by_smiles)
    final_results = [item.result for item in annotated_results]
    advisor_memory_notes = build_chemistry_advisor_memory_notes(reuse_memories)
    if active_pattern_memory.prompt_notes:
        advisor_memory_notes = advisor_memory_notes + active_pattern_memory.prompt_notes[:6]
    if prior_chemistry_lesson_summary is not None:
        advisor_memory_notes = advisor_memory_notes + prior_chemistry_lesson_summary.run_summary[:6]
    advisor_assessments: list[ChemistryAssessment] = []
    advisor_next_steps: list[ChemistryNextStep] = []
    if provider is not None and annotated_results:
        advisor_context = _build_chemistry_advisor_context(review_context, annotated_results, advisor_memory_notes)
        advisor_items = annotated_results[: min(5, len(annotated_results))]
        advisor_results = run_concurrent(
            advisor_items,
            lambda item: provider.assess_candidate_chemistry(item.result.curve.smiles_b, advisor_context, advisor_memory_notes),
        )
        for item, res in zip(advisor_items, advisor_results):
            if res.error is not None:
                llm_warnings.append(f"LLM chemistry assessment failed for {item.result.curve.smiles_b}: {res.error}")
                continue
            try:
                advisor_assessments.extend(res.value)
            except Exception as exc:
                llm_warnings.append(f"LLM chemistry assessment failed for {item.result.curve.smiles_b}: {exc}")
        try:
            advisor_next_steps = provider.suggest_next_steps(advisor_context, advisor_memory_notes)
        except Exception as exc:
            llm_warnings.append(f"LLM chemistry next-step guidance failed: {exc}")
    current_pattern_memory = build_pattern_memory(
        component_a=component_a,
        annotated_results=annotated_results,
        candidate_proposals=candidate_proposals,
        run_memories=reuse_memories,
        config=pattern_cfg,
        contradicted_family_smiles=contradicted_family_smiles,
    )
    memory_notes.extend(current_pattern_memory.notes)
    current_lesson_summary = build_chemistry_lesson_summary(
        component_a=component_a,
        annotated_results=annotated_results,
        candidate_proposals=candidate_proposals,
        run_memories=reuse_memories,
        prior_pattern_memory=active_pattern_memory,
        prior_lesson_summary=prior_chemistry_lesson_summary,
        config=ChemistryLessonSummaryConfig(mode="adaptive", max_examples=pattern_memory_max_examples),
    )
    if save_run_memory_path:
        memory = build_run_memory(
            component_a=component_a,
            n=n,
            annotated_results=annotated_results,
            candidate_proposals=candidate_proposals,
        )
        write_run_memory(save_run_memory_path, memory)
        memory_notes.append(f"Wrote run memory to {save_run_memory_path}.")
    export_outcome = SearchOutcome(
        results=final_results,
        annotated_results=annotated_results,
        candidate_proposals=candidate_proposals,
        candidate_reviews=candidate_reviews,
        brainstorm_candidates=llm_candidates,
        explanation_notes=explanation_notes,
        critique_notes=critique_notes,
        llm_warnings=llm_warnings,
        contradiction_notes=contradiction_notes,
        memory_notes=memory_notes,
        advisor_assessments=advisor_assessments,
        advisor_next_steps=advisor_next_steps,
        viscosity_predictions=viscosity_predictions,
        chemical_pattern_memory=current_pattern_memory,
        chemistry_lesson_summary=current_lesson_summary,
        claim_verdicts=claim_verdicts,
    )
    _progress(6, "Building report...")
    report_text = format_report(
        final_results,
        annotated_results=annotated_results,
        candidate_proposals=candidate_proposals,
        candidate_reviews=candidate_reviews,
        explanation_notes=explanation_notes,
        critique_notes=critique_notes,
        brainstorm_candidates=llm_candidates,
        llm_warnings=llm_warnings,
        memory_notes=memory_notes,
        viscosity_predictions=viscosity_predictions,
        contradiction_notes=contradiction_notes,
        advisor_assessments=advisor_assessments,
        advisor_next_steps=advisor_next_steps,
        chemistry_lesson_summary=current_lesson_summary,
        claim_verdicts=claim_verdicts,
    )
    export_output_dir = _resolve_des_output_dir(output_dir, save_run_memory_path)
    if export_output_dir is not None:
        try:
            export_des_run_bundle(
                export_output_dir,
                _build_des_export_payload(
                    export_outcome,
                    component_a=component_a,
                    n=n,
                    checkpoint_path=str(checkpoint_path),
                    config_path=str(config_path),
                ),
                report_text,
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"DES export failed for {export_output_dir}: {exc}") from exc
    return replace(export_outcome, report_text=report_text)


def run_search(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    thresholds: DesThresholds | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
):
    return run_search_report(
        component_a=component_a,
        n=n,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        thresholds=thresholds,
        uncertainty_policy=uncertainty_policy,
    ).results
