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


def format_task_execute_summary(execution: Any) -> CommandSummary:
    lines = _summary_header("task-execute")
    lines.append(f"- status: {getattr(execution, 'summary_status', 'completed')}")
    return CommandSummary(stream="stdout", lines=lines)


def build_command_summary(command: str, result: Any) -> CommandSummary:
    if command == "des":
        return format_des_summary(result)
    if command == "task-execute":
        return format_task_execute_summary(result)
    return CommandSummary(stream="stdout", lines=["summary:", f"- command: {command}", "- status: completed"])
