from __future__ import annotations

from dataclasses import dataclass

from .orchestrator import run_search_report
from .reporting import format_metal_binding_report, format_report
from .task_router import route_task
from .workflows.metal_binding import run_metal_binding_workflow


@dataclass(frozen=True)
class TaskExecutionResult:
    needs_clarification: bool
    output: str
    summary_status: str


def execute_task_request_detailed(request: str, provider=None) -> TaskExecutionResult:
    response = route_task(request, provider=provider)
    if response.needs_clarification or response.job is None:
        output = response.to_json()
        return TaskExecutionResult(needs_clarification=True, output=output, summary_status="clarified")

    job = response.job
    if response.workflow == "des":
        outcome = run_search_report(
            component_a=job.component_a,
            n=job.n,
            checkpoint_path=job.checkpoint_path,
            config_path=job.config_path,
            discovery_path=job.discovery_path,
            viscosity_model_path=job.viscosity_model_path,
        )
        report = format_report(
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
        return TaskExecutionResult(needs_clarification=False, output=report, summary_status="executed")

    if response.workflow == "metal-binding":
        outcome = run_metal_binding_workflow(
            metal_ion=job.metal_ion,
            ligand_smiles=job.ligand_smiles,
            model_path=job.stability_constant_model_path,
            allow_fallback=False,
        )
        report = format_metal_binding_report(outcome)
        return TaskExecutionResult(needs_clarification=False, output=report, summary_status="executed")

    raise ValueError(f"Unsupported workflow: {response.workflow}")


def execute_task_request(request: str, provider=None) -> str:
    return execute_task_request_detailed(request, provider=provider).output
