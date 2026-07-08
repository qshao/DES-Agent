from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from des_multi_agent.cli import load_llm_config
from des_multi_agent.request_normalization import normalize_request_text
from des_multi_agent.llm.factory import build_llm_provider
from des_multi_agent.llm.parser import extract_json_object
from des_multi_agent.orchestrator import run_search_report
from des_multi_agent.proposal_diversity import ProposalDiversityConfig
from des_multi_agent.reporting import format_report
from des_multi_agent.task_router_schema import RouterJob, RouterResponse


REQUEST_FILE = SCRIPT_DIR / "input.txt"
LLM_CONFIG_FILE = SCRIPT_DIR / "llm.gemma4_12b_vllm.yaml"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt"
DEFAULT_CONFIG_PATH = REPO_ROOT / "ml_des_mp/config.yaml"


def _extract_smiles_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    matches = re.findall(r"\(([^()]*)\)", text)
    if matches:
        inside = matches[-1].strip()
        if inside:
            return inside
    return text


def _normalize_router_job(raw_job: dict, request_text: str) -> RouterJob:
    component_a = _extract_smiles_text(raw_job.get("component_a") or raw_job.get("chemical_formula") or raw_job.get("target_compound") or raw_job.get("target_substance") or request_text)
    request_component_a = _extract_smiles_text(request_text)
    if request_component_a and request_component_a != request_text.strip():
        component_a = request_component_a
    n_value = raw_job.get("n") or raw_job.get("max_candidates") or 5
    checkpoint_value = raw_job.get("checkpoint_path") or raw_job.get("checkpoint") or str(DEFAULT_CHECKPOINT_PATH)
    config_value = raw_job.get("config_path") or raw_job.get("config") or str(DEFAULT_CONFIG_PATH)

    if checkpoint_value in {None, "", "default", "ml_des_mp"}:
        checkpoint_path = str(DEFAULT_CHECKPOINT_PATH)
    else:
        checkpoint_path = str(checkpoint_value)
    if config_value in {None, "", "default"}:
        config_path = str(DEFAULT_CONFIG_PATH)
    else:
        config_path = str(config_value)

    return RouterJob(
        component_a=str(component_a).strip() if component_a is not None else None,
        n=int(n_value) if n_value is not None else None,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        llm_config=str(LLM_CONFIG_FILE),
    )


def main() -> int:
    request = REQUEST_FILE.read_text(encoding="utf-8").strip()
    llm_cfg = load_llm_config(LLM_CONFIG_FILE)
    if llm_cfg is None:
        raise RuntimeError(f"Unable to load LLM config from {LLM_CONFIG_FILE}")
    provider = build_llm_provider(llm_cfg)
    if provider is None:
        raise RuntimeError(f"LLM config {LLM_CONFIG_FILE} did not produce an enabled provider")

    normalized = normalize_request_text(request)
    raw_response = provider.route_request(request, normalized=normalized)
    payload = json.loads(extract_json_object(raw_response))
    if not isinstance(payload, dict):
        raise RuntimeError("Gemma router response must be a JSON object")

    raw_job = payload.get("job")
    if not isinstance(raw_job, dict):
        raise RuntimeError("Gemma router response must include a job object")
    job = _normalize_router_job(raw_job, request)
    response = RouterResponse(
        workflow=str(payload.get("workflow", "des")).strip().lower() or "des",
        needs_clarification=bool(payload.get("needs_clarification", False)),
        clarifying_questions=[str(item).strip() for item in payload.get("clarifying_questions", []) if str(item).strip()],
        job=job,
    )

    print("Plain-language request:")
    print(request)
    print()
    print("Raw Gemma router output:")
    print(raw_response.strip())
    print()
    print("Normalized DES job:")
    print(response.to_json())
    print()

    proposal_diversity_cfg = ProposalDiversityConfig(
        diversity_mode="balanced",
        max_similarity=0.82,
        per_family_budget=2,
    )
    print(
        "Proposal diversity settings: "
        f"mode={proposal_diversity_cfg.diversity_mode}, "
        f"max_similarity={proposal_diversity_cfg.max_similarity:.2f}, "
        f"per_family_budget={proposal_diversity_cfg.per_family_budget}"
    )
    outcome = run_search_report(
        component_a=job.component_a or "",
        n=job.n or 5,
        checkpoint_path=job.checkpoint_path or str(DEFAULT_CHECKPOINT_PATH),
        config_path=job.config_path or str(DEFAULT_CONFIG_PATH),
        llm_cfg=llm_cfg,
        discovery_path=job.discovery_path,
        viscosity_model_path=job.viscosity_model_path,
        proposal_diversity_cfg={
            "diversity_mode": proposal_diversity_cfg.diversity_mode,
            "max_similarity": proposal_diversity_cfg.max_similarity,
            "per_family_budget": proposal_diversity_cfg.per_family_budget,
        },
    )

    print("DES report:")
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
            viscosity_predictions=outcome.viscosity_predictions,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
