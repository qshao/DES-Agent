# Run Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact terminal summary block after each command so users can quickly see what happened without reading the full report.

**Architecture:** Add a small presentation layer in `des_multi_agent/summary.py` that formats a short summary from the structured result objects already produced by each command. The CLI prints the main report first, then prints the summary block. For parseable stdout commands like `task-router` and `compare-runs --json`, the summary should go to stderr so stdout stays machine-readable.

**Tech Stack:** Python 3.13, `argparse`, `sys`, `pathlib`, `pytest`, existing `des_multi_agent.cli`, `des_multi_agent.compare_runs`, `des_multi_agent.doctor`, `des_multi_agent.orchestrator`, `des_multi_agent.task_router`, `des_multi_agent.task_executor`, and `des_multi_agent.workflows.metal_binding`.

---

### Task 1: Add the summary formatter module and a structured execution result for task-execute

**Files:**
- Create: `des_multi_agent/summary.py`
- Modify: `des_multi_agent/task_executor.py`
- Test: `tests/test_summary.py`
- Test: `tests/test_task_execute.py`

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import dataclass

from des_multi_agent.summary import build_command_summary, render_command_summary


@dataclass(frozen=True)
class _FakeDesOutcome:
    results: list[object]
    memory_notes: list[str]
    llm_warnings: list[str]
    candidate_reviews: list[object]
    brainstorm_candidates: list[object]
    explanation_notes: list[object]
    critique_notes: list[object]
    viscosity_predictions: list[object]


def test_format_command_summary_for_des_mentions_counts_and_memory():
    outcome = _FakeDesOutcome(
        results=[object(), object()],
        memory_notes=["Loaded reuse memory from runs/run_001/run.memory.json."],
        llm_warnings=[],
        candidate_reviews=[],
        brainstorm_candidates=[],
        explanation_notes=[],
        critique_notes=[],
        viscosity_predictions=[],
    )

    text = render_command_summary(build_command_summary("des", outcome))

    assert "summary:" in text
    assert "workflow: des" in text
    assert "ranked candidates: 2" in text
    assert "reuse memory: yes" in text
```

```python
from des_multi_agent.task_executor import execute_task_request_detailed


def test_execute_task_request_detailed_distinguishes_clarification_and_execution(monkeypatch):
    class _FakeResponse:
        needs_clarification = True
        job = None
        def to_json(self):
            return "{\"workflow\":\"clarify\",\"needs_clarification\":true,\"clarifying_questions\":[\"Which workflow?\"],\"job\":null}"

    monkeypatch.setattr("des_multi_agent.task_executor.route_task", lambda request, provider=None: _FakeResponse())
    clarified = execute_task_request_detailed("find DES partners")
    assert clarified.needs_clarification is True
    assert clarified.summary_status == "clarified"
    assert clarified.output.startswith("{")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_summary.py tests/test_task_execute.py -q
```

Expected: FAIL because `des_multi_agent.summary` does not exist yet and `task_executor` does not yet expose a structured execution result.

- [ ] **Step 3: Write minimal implementation**

```python
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
    lines.append(f"- workflow: des")
    lines.append(f"- ranked candidates: {len(getattr(outcome, 'results', []))}")
    lines.append(f"- reuse memory: {'yes' if getattr(outcome, 'memory_notes', None) else 'no'}")
    lines.append(f"- exports written: yes")
    return CommandSummary(stream="stdout", lines=lines)


def format_metal_binding_summary(outcome: Any) -> CommandSummary:
    lines = _summary_header("metal-binding")
    lines.append(f"- workflow: metal-binding")
    lines.append(f"- status: ok")
    prediction = getattr(outcome, 'prediction', None)
    if prediction is not None:
        lines.append(f"- model: {getattr(prediction, 'model_name', '-')}")
    return CommandSummary(stream="stdout", lines=lines)


def format_doctor_summary(result: Any) -> CommandSummary:
    lines = _summary_header("doctor")
    lines.append(f"- status: {'ok' if not getattr(result, 'errors', []) else 'issues found'}")
    lines.append(f"- errors: {len(getattr(result, 'errors', []))}")
    lines.append(f"- warnings: {len(getattr(result, 'warnings', []))}")
    return CommandSummary(stream="stdout", lines=lines)


def format_compare_runs_summary(result: Any) -> CommandSummary:
    counts = {"new": 0, "removed": 0, "moved": 0, "unchanged": 0}
    for row in getattr(result, 'rows', []):
        counts[row.status] += 1
    lines = _summary_header("compare-runs")
    lines.append(f"- workflow: {getattr(result, 'workflow', '-')}")
    lines.append(f"- changed candidates: {counts['new'] + counts['removed'] + counts['moved']}")
    lines.append(f"- counts: new={counts['new']}, removed={counts['removed']}, moved={counts['moved']}, unchanged={counts['unchanged']}")
    return CommandSummary(stream="stdout", lines=lines)


def format_router_summary(response: Any) -> CommandSummary:
    lines = _summary_header("task-router")
    lines.append(f"- status: {'needs clarification' if getattr(response, 'needs_clarification', False) else 'complete'}")
    questions = getattr(response, 'clarifying_questions', [])
    lines.append(f"- clarification questions: {len(questions)}")
    return CommandSummary(stream="stderr", lines=lines)


def format_task_execute_summary(execution: Any) -> CommandSummary:
    lines = _summary_header("task-execute")
    lines.append(f"- status: {'clarified' if getattr(execution, 'needs_clarification', False) else 'executed'}")
    return CommandSummary(stream="stdout", lines=lines)


def build_command_summary(command: str, result: Any) -> CommandSummary:
    if command == "des":
        return format_des_summary(result)
    if command == "metal-binding":
        return format_metal_binding_summary(result)
    if command == "doctor":
        return format_doctor_summary(result)
    if command == "compare-runs":
        return format_compare_runs_summary(result)
    if command == "task-router":
        return format_router_summary(result)
    if command == "task-execute":
        return format_task_execute_summary(result)
    return CommandSummary(stream="stdout", lines=["summary:", f"- command: {command}", "- status: completed"])
```

```python
from dataclasses import dataclass


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
    output = execute_task_request(request, provider=provider)
    return TaskExecutionResult(needs_clarification=False, output=output, summary_status="executed")
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_summary.py tests/test_task_execute.py -q
```

Expected: PASS once the summary module and structured task-execute result exist.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/summary.py des_multi_agent/task_executor.py tests/test_summary.py tests/test_task_execute.py
git commit -m "feat: add command summary formatting"
```

### Task 2: Route the summary to the right stream from the CLI

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_compare_runs.py`
- Test: `tests/test_doctor.py`

- [ ] **Step 1: Write the failing test**

```python
import des_multi_agent.cli as cli_module
from pathlib import Path


def test_task_router_summary_goes_to_stderr(monkeypatch, capsys):
    class _FakeResponse:
        needs_clarification = True
        clarifying_questions = ["Which workflow?"]
        def to_json(self):
            return '{"workflow":"clarify","needs_clarification":true,"clarifying_questions":["Which workflow?"],"job":null}'

    monkeypatch.setattr(cli_module, "route_task", lambda request, provider=None: _FakeResponse())
    cli_module.main(["task-router", "find DES partners for lidocaine"])
    out = capsys.readouterr()
    assert out.out.strip().startswith("{")
    assert "summary:" in out.err
```

```python
def test_compare_runs_json_keeps_summary_off_stdout(monkeypatch, capsys):
    class _FakeResult:
        workflow = "des"
        rows = []
        left_path = Path("runs/run_001")
        right_path = Path("runs/run_002")
        left_component_a = "CCO"
        right_component_a = "CCO"
        left_n = 5
        right_n = 5

    monkeypatch.setattr(cli_module, "compare_saved_runs", lambda left, right: _FakeResult())
    monkeypatch.setattr(cli_module, "format_compare_report", lambda result: "compare-runs report")
    monkeypatch.setattr(cli_module, "format_compare_json_text", lambda result: "{\"workflow\":\"des\"}")
    cli_module.main(["compare-runs", "runs/run_001", "runs/run_002", "--json"])
    out = capsys.readouterr()
    assert out.out.startswith("compare-runs report")
    assert "{\"workflow\":\"des\"}" in out.out
    assert "summary:" in out.err
```

```python
def test_task_execute_summary_is_printed(monkeypatch, capsys):
    class _FakeExecution:
        needs_clarification = False
        output = "EXECUTED REPORT"
        summary_status = "executed"

    monkeypatch.setattr(cli_module, "execute_task_request_detailed", lambda request, provider=None: _FakeExecution())
    cli_module.main(["task-execute", "find DES partners for lidocaine"])
    out = capsys.readouterr()
    assert out.out.startswith("EXECUTED REPORT")
    assert "summary:" in out.out
```

```python
def test_task_execute_summary_goes_to_stderr_when_clarified(monkeypatch, capsys):
    class _FakeExecution:
        needs_clarification = True
        output = "{\"workflow\":\"clarify\",\"needs_clarification\":true,\"clarifying_questions\":[\"Which workflow?\"],\"job\":null}"
        summary_status = "clarified"

    monkeypatch.setattr(cli_module, "execute_task_request_detailed", lambda request, provider=None: _FakeExecution())
    cli_module.main(["task-execute", "find DES partners for lidocaine"])
    out = capsys.readouterr()
    assert out.out.strip().startswith("{")
    assert "summary:" in out.err
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_cli.py tests/test_compare_runs.py tests/test_doctor.py -q
```

Expected: FAIL because the CLI does not yet print the new summaries and the summary stream routing is not implemented.

- [ ] **Step 3: Write minimal implementation**

```python
from .summary import build_command_summary, render_command_summary
from .task_executor import execute_task_request_detailed
import sys


def _print_summary(command: str, result, *, machine_readable_stdout: bool = False) -> None:
    summary = build_command_summary(command, result)
    stream = sys.stderr if machine_readable_stdout and command in {"task-router", "compare-runs", "task-execute"} else sys.stdout
    print(render_command_summary(summary), file=stream)


# after each command output:
print(report)
_print_summary("compare-runs", result, machine_readable_stdout=args.json)
```

```python
# task-router branch
print(response.to_json())
_print_summary("task-router", response, machine_readable_stdout=True)

# doctor branch
print(format_doctor_report(result))
_print_summary("doctor", result)

# task-execute branch
output = execute_task_request_detailed(args.request)
print(output.output)
_print_summary("task-execute", output, machine_readable_stdout=output.needs_clarification)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_cli.py tests/test_compare_runs.py tests/test_doctor.py -q
```

Expected: PASS after the CLI prints summaries and routes parseable command summaries to stderr.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py tests/test_compare_runs.py tests/test_doctor.py
git commit -m "feat: route command summaries in cli"
```

### Task 3: Document the summary block and add regression coverage for all command families

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Test: `tests/test_demo_des_search.py`
- Test: `tests/test_summary.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_summary_block_is_documented_in_main_docs():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples = Path("examples/README.md").read_text(encoding="utf-8")
    assert "summary:" in readme
    assert "summary:" in tutorial
    assert "summary:" in examples
```

```python
def test_summary_fallback_does_not_crash():
    from des_multi_agent.summary import build_command_summary, render_command_summary

    text = render_command_summary(build_command_summary("unknown-command", object()))
    assert "summary:" in text
    assert "completed" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_demo_des_search.py tests/test_summary.py -q
```

Expected: FAIL until the docs mention the summary block and the fallback formatter exists.

- [ ] **Step 3: Write minimal implementation**

```markdown
Every command prints a compact summary block after the main output. For parseable stdout commands, the summary is written to stderr so the output stream stays machine-readable.
```

```markdown
summary:
- command: compare-runs
- workflow: des
- changed candidates: 3
- counts: new=1, removed=1, moved=1, unchanged=4
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_demo_des_search.py tests/test_summary.py -q
```

Expected: PASS once the docs and fallback summary are in place.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md des_multi_agent/summary.py tests/test_summary.py tests/test_demo_des_search.py
git commit -m "docs: add run summary guidance"
```

