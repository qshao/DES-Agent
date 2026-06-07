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
from des_multi_agent.llm.factory import build_llm_provider
from des_multi_agent.llm.parser import extract_json_object
from des_multi_agent.reporting import format_metal_binding_report
from des_multi_agent.task_router_schema import RouterJob, RouterResponse
from des_multi_agent.workflows.metal_binding import run_metal_binding_workflow


REQUEST_FILE = SCRIPT_DIR / "input.txt"
LLM_CONFIG_FILE = SCRIPT_DIR / "llm.gemma4_12b.yaml"
DEFAULT_STABILITY_PATH = REPO_ROOT / "artifacts/stability_constants/model.json"


def _extract_field(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


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
    metal_ion = (
        raw_job.get("metal_ion")
        or raw_job.get("target_metal")
        or raw_job.get("metal")
        or raw_job.get("ion")
        or _extract_field(r"metal[_\s-]*ion\s*[:=]?\s*([A-Za-z0-9+\-]+)", request_text)
    )
    ligand_smiles = (
        raw_job.get("ligand_smiles")
        or raw_job.get("target_compound")
        or raw_job.get("target_ligand")
        or raw_job.get("ligand")
        or _extract_field(r"ligand[_\s-]*smiles\s*[:=]?\s*([A-Za-z0-9@+\-#=\[\]\(\)\\/]+)", request_text)
    )
    stability_value = raw_job.get("stability_constant_model_path") or raw_job.get("checkpoint") or str(DEFAULT_STABILITY_PATH)
    if stability_value in {None, "", "default", "artifact"}:
        stability_path = str(DEFAULT_STABILITY_PATH)
    else:
        stability_path = str(stability_value)

    return RouterJob(
        metal_ion=str(metal_ion).strip() if metal_ion is not None else None,
        ligand_smiles=_extract_smiles_text(ligand_smiles),
        stability_constant_model_path=stability_path,
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

    raw_response = provider.route_request(request)
    payload = json.loads(extract_json_object(raw_response))
    if not isinstance(payload, dict):
        raise RuntimeError("Gemma router response must be a JSON object")

    raw_job = payload.get("job")
    if not isinstance(raw_job, dict):
        raise RuntimeError("Gemma router response must include a job object")
    job = _normalize_router_job(raw_job, request)
    response = RouterResponse(
        workflow=str(payload.get("workflow", "metal-binding")).strip().lower() or "metal-binding",
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
    print("Normalized metal-binding job:")
    print(response.to_json())
    print()

    outcome = run_metal_binding_workflow(
        metal_ion=job.metal_ion or "",
        ligand_smiles=job.ligand_smiles or "",
        model_path=job.stability_constant_model_path or str(DEFAULT_STABILITY_PATH),
        allow_fallback=False,
    )

    print("Metal-binding report:")
    print(format_metal_binding_report(outcome))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
