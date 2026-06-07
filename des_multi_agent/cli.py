from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import DEFAULT_CONFIG_PATH
from .llm.config import LLMConfig
from .orchestrator import run_search_report
from .task_router import route_task
from .paths import resolve_existing_path
from .reporting import format_metal_binding_report, format_report
from .uncertainty import UncertaintyPolicy
from .workflows.metal_binding import run_metal_binding_workflow


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=["des", "metal-binding"], default="des")
    parser.add_argument("--component-a", default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--llm-config", default=None, help="Optional YAML file containing llm settings")
    parser.add_argument("--discovery-path", default=None, help="Optional local discovery directory containing literature.yaml and library.yaml")
    parser.add_argument("--viscosity-model-path", default=None, help="Optional local DESignSolvents viscosity model artifact")
    parser.add_argument("--metal-ion", default=None, help="Metal ion for the metal-binding workflow")
    parser.add_argument("--ligand-smiles", default=None, help="Ligand SMILES for the metal-binding workflow")
    parser.add_argument("--stability-constant-model-path", default=None, help="Optional local stability-constant model artifact")
    parser.add_argument(
        "--uncertainty-mode",
        choices=["filter", "penalize", "report_only"],
        default="penalize",
        help="How to apply uncertainty annotations to the ranked results",
    )
    parser.add_argument(
        "--min-trust-score",
        type=float,
        default=0.55,
        help="Minimum normalized trust score required before filtering or penalization",
    )
    parser.add_argument(
        "--soft-penalty-weight",
        type=float,
        default=0.35,
        help="Penalty weight applied when uncertainty is below the trust threshold",
    )
    parser.add_argument(
        "--std-high-threshold-k",
        type=float,
        default=15.0,
        help="Upper standard-deviation threshold used by the uncertainty policy",
    )
    parser.add_argument(
        "--std-medium-threshold-k",
        type=float,
        default=5.0,
        help="Middle standard-deviation threshold used by the uncertainty policy",
    )
    subparsers = parser.add_subparsers(dest="command")
    task_router_parser = subparsers.add_parser("task-router", help="Translate a plain-language request into a JSON job")
    task_router_parser.add_argument("request", help="Free-form request to translate into a JSON job")
    task_router_parser.set_defaults(command="task-router")
    parser.set_defaults(command=None)
    return parser


def load_llm_config(path: str | Path | None) -> LLMConfig | None:
    if path is None:
        return None
    llm_cfg_path = resolve_existing_path(path)
    with Path(llm_cfg_path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if isinstance(raw, dict) and "llm" in raw and isinstance(raw["llm"], dict):
        raw = raw["llm"]
    if not isinstance(raw, dict):
        raise ValueError(f"LLM config file {llm_cfg_path} must contain a mapping")
    cfg = LLMConfig.from_mapping(raw)
    cfg.validate()
    return cfg


def _build_uncertainty_policy(args):
    return UncertaintyPolicy(
        mode=args.uncertainty_mode,
        min_trust_score=args.min_trust_score,
        soft_penalty_weight=args.soft_penalty_weight,
        std_high_threshold_k=args.std_high_threshold_k,
        std_medium_threshold_k=args.std_medium_threshold_k,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "task-router":
        try:
            response = route_task(args.request)
        except ValueError as exc:
            parser.error(str(exc))
        print(response.to_json())
        return
    try:
        llm_cfg = load_llm_config(args.llm_config)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        uncertainty_policy = _build_uncertainty_policy(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.workflow == "des":
        if not args.component_a:
            parser.error("DES workflow requires --component-a")
        if args.checkpoint_path is None:
            parser.error("DES workflow requires --checkpoint-path")
        checkpoint_path = resolve_existing_path(args.checkpoint_path)
        config_path = resolve_existing_path(args.config_path)
        outcome = run_search_report(
            component_a=args.component_a,
            n=args.n,
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            llm_cfg=llm_cfg,
            discovery_path=args.discovery_path,
            viscosity_model_path=args.viscosity_model_path,
            uncertainty_policy=uncertainty_policy,
        )
        print(
            format_report(
                outcome.results,
                annotated_results=outcome.annotated_results,
                candidate_proposals=getattr(outcome, "candidate_proposals", None),
                candidate_reviews=getattr(outcome, "candidate_reviews", None),
                explanation_notes=outcome.explanation_notes,
                critique_notes=outcome.critique_notes,
                brainstorm_candidates=outcome.brainstorm_candidates,
                llm_warnings=outcome.llm_warnings,
                viscosity_predictions=getattr(outcome, "viscosity_predictions", None),
            )
        )
        return

    if not args.metal_ion:
        parser.error("metal-binding workflow requires --metal-ion")
    if not args.ligand_smiles:
        parser.error("metal-binding workflow requires --ligand-smiles")
    outcome = run_metal_binding_workflow(
        metal_ion=args.metal_ion,
        ligand_smiles=args.ligand_smiles,
        model_path=args.stability_constant_model_path,
        allow_fallback=False,
    )
    print(format_metal_binding_report(outcome))


if __name__ == "__main__":
    main()
