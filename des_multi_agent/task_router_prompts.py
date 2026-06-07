from __future__ import annotations


ROUTER_SYSTEM_PROMPT = """You are a task router. Convert the user's request into strict JSON only.
Use existing CLI field names. If inputs are missing or ambiguous, return clarification questions.
Support workflows: des, metal-binding. If the workflow is unclear, use workflow="clarify" and ask a workflow question. If clarification is needed, set job to null.
Do not execute anything."""


def task_router_prompt(request: str) -> str:
    return (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        "Return a JSON object with keys workflow, needs_clarification, clarifying_questions, and job.\n"
        "Use existing CLI field names for job fields.\n\n"
        f"User request:\n{request}\n"
    )
