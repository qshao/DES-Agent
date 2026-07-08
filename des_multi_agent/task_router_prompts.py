from __future__ import annotations

from .request_normalization import NormalizedRequest
from .task_router_schema import REQUIRED_FIELDS_BY_WORKFLOW


def _field_name_lines() -> str:
    lines = []
    for workflow, field_names in REQUIRED_FIELDS_BY_WORKFLOW.items():
        lines.append(
            f'For workflow="{workflow}", the job object must use exactly these field names: '
            + ", ".join(field_names) + "."
        )
    return "\n".join(lines)


ROUTER_SYSTEM_PROMPT = (
    "You are a task router. Convert the user's request into strict JSON only.\n"
    "If inputs are missing or ambiguous, return clarification questions.\n"
    "Support workflows: des, metal-binding. If the workflow is unclear, use workflow=\"clarify\" and ask a workflow question. If clarification is needed, set job to null.\n"
    "\n"
    f"{_field_name_lines()}\n"
    "Do not invent other field names.\n"
    "Do not execute anything."
)


def task_router_prompt(request: str, normalized: NormalizedRequest | None = None) -> str:
    prompt = (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        "Return a JSON object with keys workflow, needs_clarification, clarifying_questions, and job.\n"
    )
    if normalized is not None:
        prompt += "\nNormalized request hints:\n"
        prompt += f"- normalized_text: {normalized.normalized_text}\n"
        if normalized.workflow_hint:
            prompt += f"- workflow_hint: {normalized.workflow_hint}\n"
        if normalized.compound_hint:
            prompt += f"- compound_hint: {normalized.compound_hint}\n"
        if normalized.metal_ion_hint:
            prompt += f"- metal_ion_hint: {normalized.metal_ion_hint}\n"
        if normalized.ligand_hint:
            prompt += f"- ligand_hint: {normalized.ligand_hint}\n"
        if normalized.needs_clarification:
            prompt += "- The request may need clarification. Ask before guessing.\n"
            for question in normalized.clarifying_questions:
                prompt += f"- Clarification hint: {question}\n"
    prompt += f"\nUser request:\n{request}\n"
    return prompt
