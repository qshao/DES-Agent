from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandSummary:
    stream: str
    lines: list[str]

    def render(self) -> str:
        return "\n".join(self.lines)


def render_command_summary(summary: CommandSummary) -> str:
    return summary.render()


def _summary_header(command: str) -> list[str]:
    return ["summary:", f"- command: {command}"]


def format_des_summary(outcome: Any) -> CommandSummary:
    lines = _summary_header("des")
    lines.append("- workflow: des")
    lines.append(f"- ranked candidates: {len(getattr(outcome, 'results', []))}")
    lines.append(f"- reuse memory: {'yes' if getattr(outcome, 'memory_notes', None) else 'no'}")
    lines.append("- exports written: yes")
    return CommandSummary(stream="stdout", lines=lines)


def format_metal_binding_summary(outcome: Any) -> CommandSummary:
    lines = _summary_header("metal-binding")
    lines.append("- workflow: metal-binding")
    lines.append("- status: ok")
    prediction = getattr(outcome, "prediction", None)
    if prediction is not None:
        lines.append(f"- model: {getattr(prediction, 'model_name', '-')}")
    return CommandSummary(stream="stdout", lines=lines)


def format_doctor_summary(result: Any) -> CommandSummary:
    lines = _summary_header("doctor")
    lines.append(f"- status: {'ok' if not getattr(result, 'errors', []) else 'issues found'}")
    lines.append(f"- errors: {len(getattr(result, 'errors', []))}")
    lines.append(f"- warnings: {len(getattr(result, 'warnings', []))}")
    return CommandSummary(stream="stdout", lines=lines)


def format_compare_runs_summary(result: Any, *, stream: str = "stdout") -> CommandSummary:
    counts = {"new": 0, "removed": 0, "moved": 0, "unchanged": 0}
    for row in getattr(result, "rows", []):
        counts[row.status] += 1
    lines = _summary_header("compare-runs")
    lines.append(f"- workflow: {getattr(result, 'workflow', '-')}")
    lines.append(f"- changed candidates: {counts['new'] + counts['removed'] + counts['moved']}")
    lines.append(
        f"- counts: new={counts['new']}, removed={counts['removed']}, moved={counts['moved']}, unchanged={counts['unchanged']}"
    )
    return CommandSummary(stream=stream, lines=lines)


def format_router_summary(response: Any) -> CommandSummary:
    lines = _summary_header("task-router")
    lines.append(f"- status: {'needs clarification' if getattr(response, 'needs_clarification', False) else 'complete'}")
    questions = getattr(response, "clarifying_questions", [])
    lines.append(f"- clarification questions: {len(questions)}")
    return CommandSummary(stream="stderr", lines=lines)


def format_task_execute_summary(execution: Any) -> CommandSummary:
    lines = _summary_header("task-execute")
    lines.append(f"- status: {getattr(execution, 'summary_status', 'completed')}")
    stream = "stderr" if getattr(execution, "needs_clarification", False) else "stdout"
    return CommandSummary(stream=stream, lines=lines)


def build_command_summary(command: str, result: Any, *, machine_readable_stdout: bool = False) -> CommandSummary:
    if command == "des":
        return format_des_summary(result)
    if command == "metal-binding":
        return format_metal_binding_summary(result)
    if command == "doctor":
        return format_doctor_summary(result)
    if command == "compare-runs":
        stream = "stderr" if machine_readable_stdout else "stdout"
        return format_compare_runs_summary(result, stream=stream)
    if command == "task-router":
        return format_router_summary(result)
    if command == "task-execute":
        return format_task_execute_summary(result)
    return CommandSummary(stream="stdout", lines=["summary:", f"- command: {command}", "- status: completed"])
