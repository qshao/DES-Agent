from __future__ import annotations

import argparse

from des_multi_agent.cli import load_llm_config
from des_multi_agent.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote
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
    parser.add_argument("--n", type=int, default=5, help="Number of candidate partners to propose")
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
        "--discovery-path",
        default=None,
        help="Optional local discovery directory containing literature.yaml and library.yaml",
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
        brainstorm_candidates=[candidate for candidate, _ in selected],
        explanation_notes=explanations,
        critique_notes=critique,
        llm_warnings=warnings,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mock:
        outcome = _mock_outcome(args.component_a, args.n)
    else:
        checkpoint_path = resolve_existing_path(args.checkpoint_path)
        config_path = resolve_existing_path(args.config_path)
        llm_cfg = load_llm_config(args.llm_config) if args.llm_config else None
        discovery_path = resolve_existing_path(args.discovery_path) if args.discovery_path else None
        outcome = run_search_report(
            component_a=args.component_a,
            n=args.n,
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            llm_cfg=llm_cfg,
            discovery_path=str(discovery_path) if discovery_path is not None else None,
        )
    print(
        format_report(
            outcome.results,
            annotated_results=outcome.annotated_results,
            candidate_proposals=outcome.candidate_proposals,
            explanation_notes=outcome.explanation_notes,
            critique_notes=outcome.critique_notes,
            brainstorm_candidates=outcome.brainstorm_candidates,
            llm_warnings=outcome.llm_warnings,
        )
    )


if __name__ == "__main__":
    main()
