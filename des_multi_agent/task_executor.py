from __future__ import annotations

from .orchestrator import run_search_report
from .reporting import format_metal_binding_report, format_report
from .task_router import route_task
from .workflows.metal_binding import run_metal_binding_workflow


def execute_task_request(request: str, provider=None) -> str:
    response = route_task(request, provider=provider)
    if response.needs_clarification or response.job is None:
        return response.to_json()

    job = response.job
    if response.workflow == "des":
        if job.component_a is None or job.n is None or job.checkpoint_path is None or job.config_path is None:
            raise ValueError("task-execute received an incomplete DES job")
        outcome = run_search_report(
            component_a=job.component_a,
            n=job.n,
            checkpoint_path=job.checkpoint_path,
            config_path=job.config_path,
            discovery_path=job.discovery_path,
            viscosity_model_path=job.viscosity_model_path,
        )
        return format_report(
            outcome.results,
            annotated_results=outcome.annotated_results,
            candidate_proposals=outcome.candidate_proposals,
            candidate_reviews=outcome.candidate_reviews,
            explanation_notes=outcome.explanation_notes,
            critique_notes=outcome.critique_notes,
            brainstorm_candidates=outcome.brainstorm_candidates,
            llm_warnings=outcome.llm_warnings,
            memory_notes=getattr(outcome, "memory_notes", None),
            viscosity_predictions=outcome.viscosity_predictions,
        )

    if response.workflow == "metal-binding":
        if job.metal_ion is None or job.ligand_smiles is None or job.stability_constant_model_path is None:
            raise ValueError("task-execute received an incomplete metal-binding job")
        outcome = run_metal_binding_workflow(
            metal_ion=job.metal_ion,
            ligand_smiles=job.ligand_smiles,
            model_path=job.stability_constant_model_path,
            allow_fallback=False,
        )
        return format_metal_binding_report(outcome)

    raise ValueError(f"Unsupported workflow: {response.workflow}")
