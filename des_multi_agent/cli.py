from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .compare_runs import compare_saved_runs, format_compare_json_text, format_compare_report
from .config import DEFAULT_ABSOLUTE_TM_MAX_K, DEFAULT_CONFIG_PATH, DEFAULT_RELATIVE_DROP_MIN, PROJECT_ROOT
from .doctor import format_doctor_report, run_doctor
from .history import build_history_table, format_history_table
from .label_run import run_label_command
from .leaderboard import build_leaderboard, format_leaderboard
from .llm.config import LLMConfig
from .multi_cycle import run_multi_cycle_search
from .orchestrator import run_search_report
from .paths import resolve_existing_path
from .prediction import discover_ensemble_checkpoints
from . import prediction as _prediction
from .reporting import (
    format_metal_binding_report, format_metal_binding_screen_report, format_metal_selectivity_report,
    format_report, format_report_csv, format_report_json, format_report_prose,
    format_selectivity_des_report,
)
from .workflows.selectivity_des_pipeline import run_selectivity_des_pipeline
from .summary import build_command_summary, render_command_summary
from .task_executor import execute_task_request, execute_task_request_detailed
from .task_router import route_task
from .schemas import DesThresholds
from .uncertainty import UncertaintyPolicy
from .user_config import KNOWN_KEYS, load_user_config, save_user_config
from .workflows.metal_binding import run_metal_binding_workflow
from .workflows.metal_binding_screen import run_metal_binding_screen
from .workflows.metal_binding_selectivity import run_metal_selectivity_screen


THRESHOLD_PRESETS: dict[str, "DesThresholds"] = {}  # populated after DesThresholds import


def _init_presets() -> None:
    THRESHOLD_PRESETS["strict"] = DesThresholds(absolute_tm_max_k=240.0, relative_drop_min=0.15)
    THRESHOLD_PRESETS["standard"] = DesThresholds(
        absolute_tm_max_k=DEFAULT_ABSOLUTE_TM_MAX_K,
        relative_drop_min=DEFAULT_RELATIVE_DROP_MIN,
    )
    THRESHOLD_PRESETS["relaxed"] = DesThresholds(absolute_tm_max_k=280.0, relative_drop_min=0.05)


_init_presets()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=["des", "metal-binding", "metal-selectivity", "selectivity-des"], default="des")
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
    parser.add_argument("--target-metal-ion", default=None, help="Target metal ion for the metal-selectivity workflow (e.g., Cu2+)")
    parser.add_argument("--competitor-metal-ion", default=None, help="Competitor metal ion for the metal-selectivity workflow (e.g., Zn2+)")
    parser.add_argument("--affinity-weight", type=float, default=0.5, dest="affinity_weight",
                        help="Weight for log K(target) in composite selectivity score (default 0.5)")
    parser.add_argument("--selectivity-weight", type=float, default=0.5, dest="selectivity_weight",
                        help="Weight for delta log K in composite selectivity score (default 0.5)")
    parser.add_argument(
        "--n-des-candidates",
        type=int,
        default=20,
        dest="n_des_candidates",
        help="DES candidate search breadth per ligand per cycle (selectivity-des workflow)",
    )
    parser.add_argument(
        "--n-des-cycles",
        type=int,
        default=3,
        dest="n_des_cycles",
        help="DES iteration depth per ligand (selectivity-des workflow)",
    )
    parser.add_argument(
        "--n-outer-cycles",
        type=int,
        default=2,
        dest="n_outer_cycles",
        help="Outer loop iteration cap for selectivity-des workflow",
    )
    parser.add_argument(
        "--min-delta-log-k",
        type=float,
        default=0.0,
        dest="min_delta_log_k",
        help="Minimum delta log K threshold for Phase 1 → Phase 2 bridge filter",
    )
    parser.add_argument(
        "--top-ligands",
        type=int,
        default=3,
        dest="top_ligands",
        help="Maximum ligands passed from Phase 1 to Phase 2 (selectivity-des workflow)",
    )
    parser.add_argument("--save-run-memory", default=None, help="Optional path to write a compact JSON DES run memory file")
    parser.add_argument("--reuse-run", default=None, help="Optional prior DES run folder, run.memory.json file, or history directory of prior DES runs to reuse for ranking")
    parser.add_argument("--output-dir", default=None, help="Optional directory where DES run artifacts are written")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Use all *_best.pt fold checkpoints in ml_des_mp/runs/ for ensemble prediction (mean ± std)",
    )
    parser.add_argument(
        "--candidates-file",
        default=None,
        help="Path to a text file with one candidate SMILES per line; bypasses LLM candidate generation",
    )
    parser.add_argument(
        "--preset",
        choices=["strict", "standard", "relaxed"],
        default=None,
        help="Named threshold preset: strict (Tm≤240 K, drop≥15%%), standard (default), relaxed (Tm≤280 K, drop≥5%%)",
    )
    parser.add_argument(
        "--abs-tm-threshold",
        type=float,
        default=None,
        dest="abs_tm_threshold",
        help="Custom absolute Tm ceiling in K (overrides --preset); e.g. 340 to accept DES-formers up to 340 K",
    )
    parser.add_argument(
        "--rel-drop-min",
        type=float,
        default=None,
        dest="rel_drop_min",
        help="Custom minimum relative Tm drop fraction (overrides --preset); e.g. 0.05 for 5%% drop required",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv", "prose"],
        default="table",
        dest="format",
        help="Output format for the DES report (default: table)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate paths, config, and checkpoint compatibility then exit without running predictions",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=1,
        dest="n_cycles",
        help="Number of screening iterations; the top-K hits from each cycle seed the next (default: 1 = single shot)",
    )
    parser.add_argument(
        "--viscosity-threshold",
        type=float,
        default=None,
        dest="viscosity_threshold",
        help="Maximum acceptable viscosity (cP); DES-formers above this threshold sort below passing candidates",
    )
    parser.add_argument(
        "--viscosity-weight",
        type=float,
        default=0.3,
        dest="viscosity_weight",
        help="Weight [0,1] of the viscosity component in composite ranking (default: 0.3)",
    )
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
    task_execute_parser = subparsers.add_parser("task-execute", help="Translate a plain-language request and execute the matching workflow")
    task_execute_parser.add_argument("request", help="Free-form request to translate and execute")
    task_execute_parser.set_defaults(command="task-execute")
    compare_runs_parser = subparsers.add_parser("compare-runs", help="Compare two saved runs from the same workflow")
    compare_runs_parser.add_argument("left", help="Left run folder or run.memory.json file")
    compare_runs_parser.add_argument("right", help="Right run folder or run.memory.json file")
    compare_runs_parser.add_argument("--json", action="store_true", help="Also print a JSON summary of the comparison")
    compare_runs_parser.set_defaults(command="compare-runs")
    label_run_parser = subparsers.add_parser("label-run", help="Update good/bad labels in a saved DES run memory")
    label_run_parser.add_argument("--run", required=True, help="Prior DES run folder or run.memory.json file")
    label_run_parser.add_argument("--label", action="append", default=[], help="Label spec in the form SMILES=good or SMILES=bad")
    label_run_parser.set_defaults(command="label-run")
    doctor_parser = subparsers.add_parser("doctor", help="Check local repo and example setup")
    doctor_parser.add_argument(
        "--check",
        action="append",
        choices=("checkpoint", "discovery", "artifacts", "llm"),
        default=[],
        help="Run optional local setup checks; may be passed multiple times",
    )
    doctor_parser.set_defaults(command="doctor")
    # G1 — leaderboard
    leaderboard_parser = subparsers.add_parser("leaderboard", help="Show a ranked leaderboard of all compounds across a run history directory")
    leaderboard_parser.add_argument("history_dir", help="Directory containing run subdirectories with run.json files")
    leaderboard_parser.set_defaults(command="leaderboard")
    # E2 — history
    history_parser = subparsers.add_parser("history", help="Show a summary table of all past runs in a history directory")
    history_parser.add_argument("history_dir", help="Directory containing run subdirectories with run.manifest.json files")
    history_parser.set_defaults(command="history")
    # E4 — config
    config_parser = subparsers.add_parser("config", help="Read or write persistent user config")
    config_subparsers = config_parser.add_subparsers(dest="config_subcommand")
    config_set_parser = config_subparsers.add_parser("set", help="Set a config value: KEY=VALUE")
    config_set_parser.add_argument("assignment", help="KEY=VALUE pair, e.g. checkpoint_path=/path/to/ckpt.pt")
    config_parser.set_defaults(command="config")
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


def _discover_checkpoint() -> str | None:
    runs_dir = PROJECT_ROOT / "ml_des_mp" / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = sorted(runs_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def _print_summary(command: str, result, *, machine_readable_stdout: bool = False) -> None:
    summary = build_command_summary(command, result, machine_readable_stdout=machine_readable_stdout)
    stream = sys.stderr if summary.stream == "stderr" else sys.stdout
    print(render_command_summary(summary), file=stream)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Apply user config defaults for flags the user did not provide explicitly
    _user_cfg = load_user_config()
    if args.checkpoint_path is None and "checkpoint_path" in _user_cfg:
        args.checkpoint_path = str(_user_cfg["checkpoint_path"])
    if args.config_path == str(DEFAULT_CONFIG_PATH) and "config_path" in _user_cfg:
        args.config_path = str(_user_cfg["config_path"])
    if args.llm_config is None and "llm_config" in _user_cfg:
        args.llm_config = str(_user_cfg["llm_config"])

    if getattr(args, "command", None) == "task-router":
        try:
            response = route_task(args.request)
        except ValueError as exc:
            parser.error(str(exc))
        print(response.to_json())
        _print_summary("task-router", response)
        return
    if getattr(args, "command", None) == "task-execute":
        try:
            llm_cfg_te = load_llm_config(args.llm_config)
        except ValueError as exc:
            parser.error(str(exc))
        try:
            output = execute_task_request_detailed(args.request, llm_cfg=llm_cfg_te)
        except ValueError as exc:
            parser.error(str(exc))
        print(output.output)
        _print_summary("task-execute", output)
        return
    if getattr(args, "command", None) == "compare-runs":
        try:
            result = compare_saved_runs(args.left, args.right)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        if args.json:
            print(format_compare_report(result), file=sys.stderr)
            print(format_compare_json_text(result))
        else:
            print(format_compare_report(result))
        _print_summary("compare-runs", result, machine_readable_stdout=args.json)
        return
    if getattr(args, "command", None) == "label-run":
        try:
            message = run_label_command(args.run, args.label)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(message)
        _print_summary("label-run", message)
        return
    if getattr(args, "command", None) == "leaderboard":
        try:
            entries = build_leaderboard(args.history_dir)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        print(format_leaderboard(entries))
        return
    if getattr(args, "command", None) == "history":
        try:
            rows = build_history_table(args.history_dir)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        print(format_history_table(rows))
        return
    if getattr(args, "command", None) == "config":
        if getattr(args, "config_subcommand", None) == "set":
            assignment = args.assignment
            if "=" not in assignment:
                parser.error(f"config set requires KEY=VALUE format, got: {assignment!r}")
            key, _, value = assignment.partition("=")
            key = key.strip()
            if key not in KNOWN_KEYS:
                parser.error(f"Unknown config key {key!r}. Valid keys: {', '.join(sorted(KNOWN_KEYS))}")
            save_user_config({key: value})
            print(f"Saved {key} = {value}", file=sys.stderr)
        else:
            parser.error("Usage: des-agent config set KEY=VALUE")
        return
    if getattr(args, "command", None) == "doctor":
        llm_cfg_doctor = None
        if "llm" in (args.check or []):
            try:
                llm_cfg_doctor = load_llm_config(args.llm_config)
            except ValueError as exc:
                parser.error(str(exc))
        try:
            result = run_doctor(PROJECT_ROOT, optional_checks=args.check, llm_cfg=llm_cfg_doctor)
        except ValueError as exc:
            parser.error(str(exc))
        print(format_doctor_report(result))
        _print_summary("doctor", result)
        raise SystemExit(result.exit_code)
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
            discovered = _discover_checkpoint()
            if discovered:
                print(f"[auto] No --checkpoint-path given; using discovered checkpoint: {discovered}", file=sys.stderr)
                args.checkpoint_path = discovered
            else:
                parser.error("DES workflow requires --checkpoint-path (none found in ml_des_mp/runs/)")
        checkpoint_path = resolve_existing_path(args.checkpoint_path)
        config_path = resolve_existing_path(args.config_path)
        ensemble_ckpts: list[str] | None = None
        if getattr(args, "ensemble", False):
            found = discover_ensemble_checkpoints()
            if len(found) < 2:
                parser.error(
                    f"--ensemble requires at least 2 checkpoints in ml_des_mp/runs/; found {len(found)}"
                )
            ensemble_ckpts = [str(p) for p in found]
            print(f"[ensemble] Using {len(ensemble_ckpts)} fold checkpoints: "
                  + ", ".join(Path(p).name for p in ensemble_ckpts), file=sys.stderr)
        # E1 — apply preset thresholds, then override with explicit flags if given
        preset_name = getattr(args, "preset", None)
        thresholds = THRESHOLD_PRESETS[preset_name] if preset_name else None
        abs_tm = getattr(args, "abs_tm_threshold", None)
        rel_drop = getattr(args, "rel_drop_min", None)
        if abs_tm is not None or rel_drop is not None:
            base = thresholds or THRESHOLD_PRESETS["standard"]
            thresholds = DesThresholds(
                absolute_tm_max_k=abs_tm if abs_tm is not None else base.absolute_tm_max_k,
                relative_drop_min=rel_drop if rel_drop is not None else base.relative_drop_min,
            )

        # B7 — dry-run: validate everything then exit without predictions
        if getattr(args, "dry_run", False):
            import yaml as _yaml
            with open(config_path, "r", encoding="utf-8") as _fh:
                _cfg = _yaml.safe_load(_fh)
            compat_warnings = _prediction.check_checkpoint_config_compat(str(checkpoint_path), _cfg or {})
            for w in compat_warnings:
                print(f"[WARNING] {w}", file=sys.stderr)
            print("[dry-run] Paths resolved, config parsed, checkpoint compatible — OK.", file=sys.stderr)
            raise SystemExit(0)

        try:
            if getattr(args, "n_cycles", 1) > 1:
                multi_outcome = run_multi_cycle_search(
                    component_a=args.component_a,
                    n=args.n,
                    checkpoint_path=str(checkpoint_path),
                    config_path=str(config_path),
                    thresholds=thresholds,
                    uncertainty_policy=uncertainty_policy,
                    llm_cfg=llm_cfg,
                    discovery_path=args.discovery_path,
                    viscosity_model_path=args.viscosity_model_path,
                    viscosity_weight=args.viscosity_weight,
                    viscosity_threshold_cp=args.viscosity_threshold,
                    output_dir=args.output_dir,
                    ensemble_checkpoints=ensemble_ckpts,
                    candidates_file=getattr(args, "candidates_file", None),
                    n_cycles=args.n_cycles,
                )
                outcome = multi_outcome.final_outcome
                for delta in multi_outcome.cycle_deltas:
                    new = f"+{len(delta.new_entrants)}" if delta.new_entrants else "0"
                    out = f"-{len(delta.dropouts)}" if delta.dropouts else "0"
                    print(
                        f"[cycle {delta.cycle}/{multi_outcome.total_cycles}] "
                        f"screened={delta.n_screened} des={delta.n_des} "
                        f"top-K changes: {new} new, {out} dropped"
                        + (" — CONVERGED" if delta.converged else ""),
                        file=sys.stderr,
                    )
            else:
                outcome = run_search_report(
                    component_a=args.component_a,
                    n=args.n,
                    checkpoint_path=str(checkpoint_path),
                    config_path=str(config_path),
                    thresholds=thresholds,
                    uncertainty_policy=uncertainty_policy,
                    llm_cfg=llm_cfg,
                    discovery_path=args.discovery_path,
                    viscosity_model_path=args.viscosity_model_path,
                    viscosity_weight=getattr(args, "viscosity_weight", 0.3),
                    viscosity_threshold_cp=getattr(args, "viscosity_threshold", None),
                    output_dir=args.output_dir,
                    ensemble_checkpoints=ensemble_ckpts,
                    candidates_file=getattr(args, "candidates_file", None),
                    save_run_memory_path=getattr(args, "save_run_memory", None),
                    reuse_run_path=getattr(args, "reuse_run", None),
                )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

        # C6 — format selection
        fmt = getattr(args, "format", "table")
        if fmt == "json":
            print(format_report_json(outcome.results, outcome.annotated_results))
        elif fmt == "csv":
            print(format_report_csv(outcome.results, outcome.annotated_results))
        elif fmt == "prose":
            print(format_report_prose(outcome.results, outcome.annotated_results))
        else:
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
                    memory_notes=getattr(outcome, "memory_notes", None),
                    viscosity_predictions=getattr(outcome, "viscosity_predictions", None),
                )
            )
        _print_summary("des", outcome)
        return

    if args.workflow == "selectivity-des":
        if not args.target_metal_ion:
            parser.error("selectivity-des workflow requires --target-metal-ion")
        if not args.competitor_metal_ion:
            parser.error("selectivity-des workflow requires --competitor-metal-ion")
        if not args.checkpoint_path:
            parser.error("selectivity-des workflow requires --checkpoint-path")
        checkpoint_path = resolve_existing_path(args.checkpoint_path)
        config_path = resolve_existing_path(args.config_path)
        try:
            pipeline_outcome = run_selectivity_des_pipeline(
                target_metal=args.target_metal_ion,
                competitor_metal=args.competitor_metal_ion,
                checkpoint_path=str(checkpoint_path),
                config_path=str(config_path),
                n_ligands=args.n,
                n_des_candidates=args.n_des_candidates,
                n_selectivity_cycles=args.n_cycles,
                n_des_cycles=args.n_des_cycles,
                n_outer_cycles=args.n_outer_cycles,
                min_delta_log_k=args.min_delta_log_k,
                top_ligands=args.top_ligands,
                w_affinity=args.affinity_weight,
                w_selectivity=args.selectivity_weight,
                stability_model_path=args.stability_constant_model_path,
                llm_cfg=llm_cfg,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(format_selectivity_des_report(pipeline_outcome))
        _print_summary("selectivity-des", pipeline_outcome)
        return

    if args.workflow == "metal-selectivity":
        if not args.target_metal_ion or not args.competitor_metal_ion:
            parser.error("metal-selectivity workflow requires --target-metal-ion and --competitor-metal-ion")
        from .llm.factory import build_llm_provider as _build_llm_provider
        llm_provider_sel = _build_llm_provider(llm_cfg) if llm_cfg is not None else None
        sel_outcome = run_metal_selectivity_screen(
            target_metal=args.target_metal_ion,
            competitor_metal=args.competitor_metal_ion,
            n=getattr(args, "n", 20),
            model_path=args.stability_constant_model_path,
            llm_provider=llm_provider_sel,
            n_cycles=getattr(args, "n_cycles", 1),
            w_affinity=args.affinity_weight,
            w_selectivity=args.selectivity_weight,
        )
        print(format_metal_selectivity_report(sel_outcome))
        _print_summary("metal-selectivity", sel_outcome)
        return

    if not args.metal_ion:
        parser.error("metal-binding workflow requires --metal-ion")

    # Screening mode: no --ligand-smiles given (search across many candidates)
    use_screen = args.ligand_smiles is None

    if use_screen:
        from .llm.factory import build_llm_provider as _build_llm_provider
        llm_provider_mb = _build_llm_provider(llm_cfg) if llm_cfg is not None else None
        screen_outcome = run_metal_binding_screen(
            metal_ion=args.metal_ion,
            n=getattr(args, "n", 20),
            model_path=args.stability_constant_model_path,
            llm_provider=llm_provider_mb,
            n_cycles=getattr(args, "n_cycles", 1),
        )
        print(format_metal_binding_screen_report(screen_outcome))
        _print_summary("metal-binding", screen_outcome)
        return

    if not args.ligand_smiles:
        parser.error("metal-binding single-pair mode requires --ligand-smiles")
    outcome = run_metal_binding_workflow(
        metal_ion=args.metal_ion,
        ligand_smiles=args.ligand_smiles,
        model_path=args.stability_constant_model_path,
        allow_fallback=False,
    )
    print(format_metal_binding_report(outcome))
    _print_summary("metal-binding", outcome)


if __name__ == "__main__":
    main()
