from __future__ import annotations

import argparse

from des_multi_agent.chemistry.name_resolution import resolve_to_smiles as _resolve_name
from des_multi_agent.cli import _positive_int, _unit_float, load_llm_config
from des_multi_agent.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm, CandidateReview, CritiqueNote, ExplanationNote
from des_multi_agent.orchestrator import SearchOutcome, run_search_report
from des_multi_agent.paths import resolve_existing_path
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.reporting import format_report
from des_multi_agent.schemas import CandidateProposal
from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty


DEFAULT_CHECKPOINT = PROJECT_ROOT / "ml_des_mp" / "runs" / "chemberta_random_row_fold01of05_best.pt"
DEFAULT_LLM_CONFIG = PROJECT_ROOT / "llm.example.yaml"


def build_parser():
    parser = argparse.ArgumentParser(description="Run a DES screening demo")
    parser.add_argument("--component-a", default="CCO", help="SMILES for component A")
    parser.add_argument("--n", type=_positive_int, default=20, help="Number of candidate partners to propose")
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to a trained ml_des_mp checkpoint",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the ml_des_mp config.yaml file",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Optional YAML file with llm settings; omit for deterministic mode",
    )
    parser.add_argument(
        "--proposal-diversity-mode",
        choices=["explore", "balanced", "exploit"],
        default=None,
        help="Optional proposal diversity mode for DES brainstorming",
    )
    parser.add_argument(
        "--proposal-max-similarity",
        type=_unit_float,
        default=None,
        help="Maximum fingerprint similarity for suppressing near-duplicate proposals",
    )
    parser.add_argument(
        "--proposal-per-family-budget",
        type=_positive_int,
        default=None,
        help="Maximum accepted proposals per family",
    )
    parser.add_argument(
        "--discovery-path",
        default=None,
        help="Optional local discovery directory containing literature.yaml and library.yaml",
    )
    parser.add_argument(
        "--viscosity-model-path",
        default=None,
        help="Optional local DESignSolvents viscosity model artifact",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory where DES run artifacts are written",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run a fully offline mock demo with canned predictions and LLM notes",
    )
    return parser


def resolve_defaults():
    return (
        resolve_existing_path(DEFAULT_CHECKPOINT),
        resolve_existing_path(DEFAULT_CONFIG_PATH),
        resolve_existing_path(DEFAULT_LLM_CONFIG),
    )


def _mock_uncertainty(component_a: str, component_b: str, min_tm: float, trust_score: float, std_tm_k: float) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a=component_a,
        component_b=component_b,
        repeated_values=(min_tm - std_tm_k, min_tm, min_tm + std_tm_k),
        mean_tm_k=min_tm,
        std_tm_k=std_tm_k,
        min_tm_k=min_tm - std_tm_k,
        max_tm_k=min_tm + std_tm_k,
        trust_score=trust_score,
        uncertainty_flag="low" if std_tm_k <= 2.0 else "medium" if std_tm_k <= 8.0 else "high",
        explanation=f"Mock uncertainty for {component_b}",
        checkpoint_path="mock://demo",
        config_path="mock://demo",
    )


def _mock_outcome(component_a: str, n: int) -> SearchOutcome:
    if n < 1:
        n = 1
    catalog = [
        (CandidateBrainstorm(smiles="OCCO", rationale="small polyol that can hydrogen bond", family="polyol"), 208.69),
        (CandidateBrainstorm(smiles="OCC(O)CO", rationale="more strongly hydrogen-bonding polyol", family="polyol"), 223.28),
        (CandidateBrainstorm(smiles="O", rationale="minimal hydrogen-bond donor/acceptor", family="alcohol"), 225.27),
        (CandidateBrainstorm(smiles="CC(=O)N", rationale="amide-like hydrogen-bonding partner", family="amide"), 222.77),
        (CandidateBrainstorm(smiles="CC(=O)O", rationale="acidic hydrogen-bond donor", family="carboxylic acid"), 236.03),
        (CandidateBrainstorm(smiles="CN", rationale="small amine that can pair through hydrogen bonding", family="amine"), 241.11),
        (CandidateBrainstorm(smiles="NC(=O)N", rationale="urea-like strong hydrogen-bonding partner", family="urea"), 219.84),
        (CandidateBrainstorm(smiles="C[N+](C)(C)C.[Cl-]", rationale="ionic salt-like DES partner", family="quaternary ammonium salt"), 214.62),
        (CandidateBrainstorm(smiles="C[N+](C)(C)CCO.[Cl-]", rationale="choline-like ionic partner", family="choline-like"), 216.44),
        (CandidateBrainstorm(smiles="OC[C@H](O)[C@H](O)CO", rationale="sugar-like polyol with multiple donors", family="sugar-like polyol"), 227.90),
        (CandidateBrainstorm(smiles="OCCCO", rationale="short polyol with flexible hydrogen bonding", family="short diol"), 220.18),
        (CandidateBrainstorm(smiles="COCCO", rationale="ether alcohol with strong polarity", family="ether alcohol"), 221.44),
        (CandidateBrainstorm(smiles="O=C1CCCN1", rationale="lactam-like cyclic hydrogen-bonding partner", family="lactam"), 229.67),
        (CandidateBrainstorm(smiles="O=C1NC(=O)NC1", rationale="cyclic urea with multiple donors and acceptors", family="cyclic urea"), 218.94),
        (CandidateBrainstorm(smiles="CS(=O)C", rationale="sulfoxide with strong polarity", family="sulfoxide"), 233.41),
        (CandidateBrainstorm(smiles="CS(=O)(=O)C", rationale="sulfone with high dipole moment", family="sulfone"), 239.02),
        (CandidateBrainstorm(smiles="OC1=CC=CC=N1", rationale="hydroxypyridine with aromatic hydrogen bonding", family="hydroxypyridine"), 224.88),
        (CandidateBrainstorm(smiles="CN(C)C=O", rationale="polar aprotic amide-like partner", family="dimethylformamide-like"), 231.75),
        (CandidateBrainstorm(smiles="CCOCC", rationale="ether-like flexible partner", family="ether"), 226.05),
        (CandidateBrainstorm(smiles="CC(=O)NCCO", rationale="hydroxyamide with multiple interaction sites", family="hydroxyamide"), 217.53),
    ]
    selected = catalog[:n]
    mock_results: list[DesResult] = []
    annotated_results: list[AnnotatedResult] = []
    for index, (candidate, min_tm) in enumerate(selected):
        curve = CurvePrediction(
            smiles_a=component_a,
            smiles_b=candidate.smiles,
            ratios=[0.1, 0.5, 0.9],
            tm_pred_k=[min_tm + 12.0, min_tm, min_tm + 16.0],
            t1_k=298.15,
            t2_k=300.0,
            checkpoint_path="mock://demo",
        )
        result = DesResult(
            curve=curve,
            absolute_pass=True,
            relative_pass=True,
            is_des=True,
            rationale=f"mock min Tm={min_tm:.2f} K, absolute<= 260.00 K, relative_drop=0.20",
            min_tm_k=min_tm,
        )
        mock_results.append(result)
        uncertainty = _mock_uncertainty(component_a, candidate.smiles, min_tm, trust_score=0.95 - index * 0.03, std_tm_k=0.5 + 0.2 * index)
        annotated_results.append(
            AnnotatedResult(
                result=result,
                uncertainty=uncertainty,
                trust_score=uncertainty.trust_score,
                ranking_score=min_tm - index * 0.1,
            )
        )
    explanations = [
        ExplanationNote(smiles="OCCO", summary="Good hydrogen-bonding candidate in mock mode", evidence=["low mock min Tm"]),
        ExplanationNote(smiles="OCC(O)CO", summary="Strong mock candidate for a DES pair", evidence=["strong mock polarity"]),
        ExplanationNote(smiles="CC(=O)N", summary="Amide-like mock candidate with moderate interaction strength", evidence=["intermediate mock min Tm"]),
    ][: len(selected)]
    critique = [
        CritiqueNote(smiles="O", assessment="Useful for demo only", concerns=["small candidate space"]),
    ]
    warnings = ["Mock mode is using canned outputs and does not call the trained model."]
    return SearchOutcome(
        results=mock_results,
        annotated_results=annotated_results,
        candidate_proposals=[
            CandidateProposal(
                smiles=candidate.smiles,
                rationale=candidate.rationale,
                family=candidate.family,
                source="mock",
                source_id="mock-demo",
            )
            for candidate, _ in selected
        ],
        candidate_reviews=[
            CandidateReview(
                smiles="OCCO",
                decision="keep",
                confidence=0.95,
                rationale="Good hydrogen-bonding candidate in mock mode",
                notes=["low mock min Tm"],
            ),
            CandidateReview(
                smiles="OCC(O)CO",
                decision="deprioritize",
                confidence=0.74,
                rationale="Still useful but less compelling than the top candidate",
                notes=["mock review penalty"],
            ),
            CandidateReview(
                smiles="O",
                decision="reject",
                confidence=0.33,
                rationale="Too small to be informative in this mock review",
                notes=["demo only"],
            ),
        ][: len(selected)],
        brainstorm_candidates=[candidate for candidate, _ in selected],
        explanation_notes=explanations,
        critique_notes=critique,
        llm_warnings=warnings,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    component_a = _resolve_name(args.component_a)
    if args.mock:
        outcome = _mock_outcome(component_a, args.n)
    else:
        checkpoint_path = resolve_existing_path(args.checkpoint_path)
        config_path = resolve_existing_path(args.config_path)
        llm_cfg = load_llm_config(args.llm_config) if args.llm_config else None
        discovery_path = resolve_existing_path(args.discovery_path) if args.discovery_path else None
        proposal_diversity_cfg = None
        if any(
            getattr(args, field) is not None
            for field in ("proposal_diversity_mode", "proposal_max_similarity", "proposal_per_family_budget")
        ):
            proposal_diversity_cfg = {}
            if args.proposal_diversity_mode is not None:
                proposal_diversity_cfg["diversity_mode"] = args.proposal_diversity_mode
            if args.proposal_max_similarity is not None:
                proposal_diversity_cfg["max_similarity"] = args.proposal_max_similarity
            if args.proposal_per_family_budget is not None:
                proposal_diversity_cfg["per_family_budget"] = args.proposal_per_family_budget
        outcome = run_search_report(
            component_a=component_a,
            n=args.n,
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            llm_cfg=llm_cfg,
            discovery_path=str(discovery_path) if discovery_path is not None else None,
            viscosity_model_path=args.viscosity_model_path,
            output_dir=args.output_dir,
            proposal_diversity_cfg=proposal_diversity_cfg,
        )
    print(
        format_report(
            outcome.results,
            annotated_results=outcome.annotated_results,
            candidate_proposals=outcome.candidate_proposals,
            candidate_reviews=outcome.candidate_reviews,
            explanation_notes=outcome.explanation_notes,
            critique_notes=outcome.critique_notes,
            brainstorm_candidates=outcome.brainstorm_candidates,
            llm_warnings=outcome.llm_warnings,
            memory_notes=getattr(outcome, "memory_notes", None),
            chemistry_lesson_summary=getattr(outcome, "chemistry_lesson_summary", None),
            viscosity_predictions=getattr(outcome, "viscosity_predictions", None),
        )
    )


if __name__ == "__main__":
    main()
