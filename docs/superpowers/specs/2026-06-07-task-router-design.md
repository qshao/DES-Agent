# Task Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a natural-language task-router CLI subcommand that turns plain-English requests into strict JSON jobs for the existing DES and metal-binding workflows, or asks clarifying questions when inputs are incomplete.

**Architecture:** The router will sit in front of the existing CLI and reuse the current workflow names and parameter names as its canonical JSON contract. It will not execute any workflow itself. Instead, it will call the configured LLM, validate the model output, and print JSON only. The router subcommand loads the repository's default LLM config (`llm.example.yaml`) so the user only supplies a request string. If the request is complete, the JSON will contain a fully populated job object. If the request is ambiguous or missing required inputs, the JSON will set `needs_clarification=true` and include explicit follow-up questions.

**Tech Stack:** Python, argparse, JSON validation, existing `des_multi_agent.llm` providers, existing `des_multi_agent.cli` workflow functions, pytest

---

### Task 1: Define the router schema and prompt contract

**Files:**
- Create: `des_multi_agent/task_router_schema.py`
- Create: `des_multi_agent/task_router_prompts.py`
- Modify: `des_multi_agent/llm/schemas.py` if a small shared dataclass is needed for parsed router output
- Test: `tests/test_task_router_schema.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.task_router_schema import RouterJob, RouterResponse


def test_router_response_requires_json_shape():
    response = RouterResponse(
        workflow="des",
        needs_clarification=False,
        clarifying_questions=[],
        job=RouterJob(
            component_a="CCO",
            n=20,
            checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
            config_path="ml_des_mp/config.yaml",
            llm_config=None,
            discovery_path=None,
            viscosity_model_path=None,
            metal_ion=None,
            ligand_smiles=None,
            stability_constant_model_path=None,
        ),
    )
    assert response.workflow == "des"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_task_router_schema.py -v`
Expected: FAIL because the schema module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import asdict, dataclass
import json
from typing import Optional


@dataclass(frozen=True)
class RouterJob:
    component_a: Optional[str] = None
    n: Optional[int] = None
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    llm_config: Optional[str] = None
    discovery_path: Optional[str] = None
    viscosity_model_path: Optional[str] = None
    metal_ion: Optional[str] = None
    ligand_smiles: Optional[str] = None
    stability_constant_model_path: Optional[str] = None


@dataclass(frozen=True)
class RouterResponse:
    workflow: str
    needs_clarification: bool
    clarifying_questions: list[str]
    job: RouterJob | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
```

```python
ROUTER_SYSTEM_PROMPT = """You are a task router. Convert the user's request into strict JSON only.
Use existing CLI field names. If inputs are missing or ambiguous, return clarification questions.
Support workflows: des, metal-binding.
Do not execute anything."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_task_router_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/task_router_schema.py des_multi_agent/task_router_prompts.py tests/test_task_router_schema.py
git commit -m "feat: add task router schema"
```

### Task 2: Implement router parser and validation

**Files:**
- Create: `des_multi_agent/task_router.py`
- Modify: `des_multi_agent/llm/parser.py` if shared JSON extraction helpers are needed
- Test: `tests/test_task_router.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.task_router import parse_router_response


def test_parse_router_response_for_des_job():
    payload = '{"workflow":"des","needs_clarification":false,"clarifying_questions":[],"job":{"component_a":"CCO","n":20,"checkpoint_path":"ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt","config_path":"ml_des_mp/config.yaml","llm_config":"llm.example.yaml","discovery_path":null,"viscosity_model_path":null,"metal_ion":null,"ligand_smiles":null,"stability_constant_model_path":null}}'
    response = parse_router_response(payload)
    assert response.workflow == "des"
    assert response.job.component_a == "CCO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_task_router.py::test_parse_router_response_for_des_job -v`
Expected: FAIL because the parser does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
import json
from des_multi_agent.task_router_schema import RouterJob, RouterResponse


def parse_router_response(payload: str) -> RouterResponse:
    raw = json.loads(payload)
    job = raw.get("job")
    parsed_job = None
    if isinstance(job, dict):
        parsed_job = RouterJob(**job)
    return RouterResponse(
        workflow=raw["workflow"],
        needs_clarification=bool(raw["needs_clarification"]),
        clarifying_questions=list(raw.get("clarifying_questions", [])),
        job=parsed_job,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_task_router.py::test_parse_router_response_for_des_job -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/task_router.py tests/test_task_router.py des_multi_agent/llm/parser.py
git commit -m "feat: add task router parser"
```

### Task 3: Add the `task-router` CLI subcommand

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.cli import build_parser


def test_cli_parser_accepts_task_router_request():
    parser = build_parser()
    args = parser.parse_args(["task-router", "find DES partners for lidocaine"])
    assert args.command == "task-router"
    assert args.request == "find DES partners for lidocaine"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_cli_parser_accepts_task_router_request -v`
Expected: FAIL because the subcommand does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
router_parser = subparsers.add_parser("task-router", help="Translate plain language into a JSON job")
router_parser.add_argument("request", help="Plain-language request to translate")
```

```python
if args.command == "task-router":
    llm_cfg = load_llm_config("llm.example.yaml")
    provider = build_llm_provider(llm_cfg)
    if provider is None:
        parser.error("task-router requires an enabled LLM config")
    response = route_task(args.request, provider=provider)
    print(response.to_json())
    return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py::test_cli_parser_accepts_task_router_request -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: add task router cli subcommand"
```

### Task 4: Wire the router to the existing LLM provider layer

**Files:**
- Modify: `des_multi_agent/task_router.py`
- Modify: `des_multi_agent/task_router_prompts.py`
- Modify: `des_multi_agent/llm/provider.py`
- Modify: `des_multi_agent/llm/base.py`
- Modify: `des_multi_agent/llm/local_provider.py`
- Modify: `des_multi_agent/llm/hosted_provider.py`
- Modify: `des_multi_agent/llm/gemini_provider.py`
- Modify: `des_multi_agent/llm/custom_http_provider.py`
- Modify: `des_multi_agent/llm/nemotron_provider.py`
- Modify: `des_multi_agent/llm/qwen_provider.py`
- Test: `tests/test_task_router.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.task_router import route_task


def test_route_task_returns_clarification_for_ambiguous_request():
    class FakeProvider:
        def route_request(self, request: str):
            return '{"workflow":"des","needs_clarification":true,"clarifying_questions":["Do you mean lidocaine free base or lidocaine HCl?"],"job":null}'

    result = route_task("find good DES partners for lidocaine", provider=FakeProvider())
    assert result.needs_clarification is True
    assert result.clarifying_questions[0].startswith("Do you mean")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_task_router.py::test_route_task_returns_clarification_for_ambiguous_request -v`
Expected: FAIL because the router is not wired yet.

- [ ] **Step 3: Write minimal implementation**

```python
def route_task(request: str, provider):
    payload = provider.route_request(request)
    return parse_router_response(payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_task_router.py::test_route_task_returns_clarification_for_ambiguous_request -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/task_router.py des_multi_agent/task_router_prompts.py tests/test_task_router.py
git commit -m "feat: wire task router to llm"
```

### Task 5: Document the router and add examples

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Create: `examples/task_router/README.md`
- Create: `examples/task_router/input.txt`
- Create: `examples/task_router/output.txt`
- Test: `tests/test_offline_examples.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_task_router_example_exists():
    assert Path("examples/task_router/README.md").exists()
    assert Path("examples/task_router/input.txt").exists()
    assert Path("examples/task_router/output.txt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_offline_examples.py::test_task_router_example_exists -v`
Expected: FAIL because the example folder does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```markdown
# Task Router Example

This example shows how a plain-language request is translated into a JSON job.

## Input
- Request: `find DES partners for lidocaine`

## Output
- The router JSON response, including workflow, job fields, or clarification questions.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_offline_examples.py::test_task_router_example_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md examples/task_router/README.md examples/task_router/input.txt examples/task_router/output.txt tests/test_offline_examples.py
git commit -m "docs: add task router example"
```

### Task 6: Full verification

**Files:**
- No code changes expected

- [ ] **Step 1: Run the targeted task-router suite**

Run: `python -m pytest tests/test_task_router.py tests/test_task_router_schema.py tests/test_cli.py tests/test_offline_examples.py -q`
Expected: PASS

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 3: Commit any final doc/test fixes**

```bash
git add -A
git commit -m "test: verify task router integration"
```
