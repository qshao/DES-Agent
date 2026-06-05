from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import DEFAULT_CONFIG_PATH
from .llm.config import LLMConfig
from .orchestrator import run_search_report
from .uncertainty import UncertaintyPolicy
from .paths import resolve_existing_path
from .reporting import format_report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-a", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--llm-config", default=None, help="Optional YAML file containing llm settings")
    parser.add_argument("--discovery-path", default=None, help="Optional local discovery directory containing literature.yaml and library.yaml")
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


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint_path = resolve_existing_path(args.checkpoint_path)
    config_path = resolve_existing_path(args.config_path)
    try:
        llm_cfg = load_llm_config(args.llm_config)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        uncertainty_policy = UncertaintyPolicy(
            mode=args.uncertainty_mode,
            min_trust_score=args.min_trust_score,
            soft_penalty_weight=args.soft_penalty_weight,
            std_high_threshold_k=args.std_high_threshold_k,
            std_medium_threshold_k=args.std_medium_threshold_k,
        )
    except ValueError as exc:
        parser.error(str(exc))
    outcome = run_search_report(
        component_a=args.component_a,
        n=args.n,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        llm_cfg=llm_cfg,
        discovery_path=args.discovery_path,
        uncertainty_policy=uncertainty_policy,
    )
    print(
        format_report(
            outcome.results,
            annotated_results=outcome.annotated_results,
            candidate_proposals=getattr(outcome, "candidate_proposals", None),
            explanation_notes=outcome.explanation_notes,
            critique_notes=outcome.critique_notes,
            brainstorm_candidates=outcome.brainstorm_candidates,
            llm_warnings=outcome.llm_warnings,
        )
    )


if __name__ == "__main__":
    main()
