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
- Test: `tests/test_task_router_schema.py`
- Test: `tests/test_task_router_prompts.py`

- [ ] **Step 1: Write the failing tests**

```python
from des_multi_agent.task_router_schema import RouterJob, RouterResponse


def test_router_response_to_json_contains_expected_fields():
    response = RouterResponse(
        workflow="des",
        needs_clarification=False,
        clarifying_questions=[],
        job=RouterJob(
            component_a="CCO",
            n=20,
            checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
            config_path="ml_des_mp/config.yaml",
            llm_config="llm.example.yaml",
            discovery_path=None,
            viscosity_model_path=None,
            metal_ion=None,
            ligand_smiles=None,
            stability_constant_model_path=None,
        ),
    )
    payload = response.to_json()
    assert '"workflow": "des"' in payload
    assert '"component_a": "CCO"' in payload
    assert '"needs_clarification": false' in payload
```

```python
from des_multi_agent.task_router_prompts import task_router_prompt


def test_task_router_prompt_mentions_json_only_and_workflows():
    prompt = task_router_prompt("find DES partners for lidocaine")
    assert "strict JSON only" in prompt
    assert "des" in prompt
    assert "metal-binding" in prompt
    assert "find DES partners for lidocaine" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`python -m pytest tests/test_task_router_schema.py tests/test_task_router_prompts.py -v`

Expected: FAIL because the schema and prompt modules do not exist yet.

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


def task_router_prompt(request: str) -> str:
    return (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        "Return a JSON object with keys workflow, needs_clarification, clarifying_questions, and job.\n"
        "Use existing CLI field names for job fields.\n\n"
        f"User request:\n{request}\n"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
`python -m pytest tests/test_task_router_schema.py tests/test_task_router_prompts.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/task_router_schema.py des_multi_agent/task_router_prompts.py tests/test_task_router_schema.py tests/test_task_router_prompts.py
git commit -m "feat: add task router schema"
```

### Task 2: Add a router request hook to the LLM provider layer

**Files:**
- Modify: `des_multi_agent/llm/provider.py`
- Modify: `des_multi_agent/llm/base.py`
- Modify: `des_multi_agent/llm/prompts.py`
- Test: `tests/test_llm_provider_router.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.local_provider import OllamaProvider


def test_route_request_uses_task_router_prompt():
    captured = {}

    def fake_request(url, payload, api_key=None, timeout_seconds=None):
        captured["url"] = url
        captured["payload"] = payload
        return '{"message":{"content":"{\\"workflow\\":\\"des\\",\\"needs_clarification\\":true,\\"clarifying_questions\\":[\\"Do you mean lidocaine free base or lidocaine HCl?\\"],\\"job\\":null}"}}'

    provider = OllamaProvider(
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        request_fn=fake_request,
    )
    provider.route_request("find DES partners for lidocaine")

    prompt = captured["payload"]["messages"][0]["content"]
    assert "strict JSON only" in prompt
    assert "find DES partners for lidocaine" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`python -m pytest tests/test_llm_provider_router.py -v`

Expected: FAIL because `route_request()` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def route_request(self, request: str) -> str:
        raise NotImplementedError
```

```python
from .prompts import task_router_prompt


class BaseLLMProvider(LLMProvider):
    ...

    def route_request(self, request: str) -> str:
        return self._request(task_router_prompt(request))
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`python -m pytest tests/test_llm_provider_router.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/provider.py des_multi_agent/llm/base.py des_multi_agent/llm/prompts.py tests/test_llm_provider_router.py
git commit -m "feat: add llm router request hook"
```

### Task 3: Implement the task-router parser and default provider loading

**Files:**
- Create: `des_multi_agent/task_router.py`
- Modify: `des_multi_agent/llm/parser.py`
- Test: `tests/test_task_router.py`

- [ ] **Step 1: Write the failing tests**

```python
from des_multi_agent.task_router import parse_router_response


def test_parse_router_response_accepts_complete_des_job():
    payload = '{"workflow":"des","needs_clarification":false,"clarifying_questions":[],"job":{"component_a":"CCO","n":20,"checkpoint_path":"ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt","config_path":"ml_des_mp/config.yaml","llm_config":"llm.example.yaml","discovery_path":null,"viscosity_model_path":null,"metal_ion":null,"ligand_smiles":null,"stability_constant_model_path":null}}'
    response = parse_router_response(payload)
    assert response.workflow == "des"
    assert response.job.component_a == "CCO"
    assert response.to_json().startswith("{")
```

```python
from des_multi_agent.task_router import parse_router_response
import pytest


def test_parse_router_response_rejects_malformed_json():
    with pytest.raises(ValueError, match="router response"):
        parse_router_response("not json")
```

```python
from des_multi_agent.task_router import route_task


def test_route_task_uses_default_provider_when_not_supplied(monkeypatch):
    class FakeProvider:
        def route_request(self, request: str):
            return '{"workflow":"metal-binding","needs_clarification":false,"clarifying_questions":[],"job":{"component_a":null,"n":null,"checkpoint_path":null,"config_path":null,"llm_config":null,"discovery_path":null,"viscosity_model_path":null,"metal_ion":"Cu2+","ligand_smiles":"NCCN","stability_constant_model_path":"artifacts/stability_constants/model.json"}}'

    monkeypatch.setattr("des_multi_agent.task_router.build_default_router_provider", lambda: FakeProvider())
    response = route_task("predict stability for Cu2+ with NCCN")
    assert response.workflow == "metal-binding"
    assert response.job.metal_ion == "Cu2+"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`python -m pytest tests/test_task_router.py -v`

Expected: FAIL because the router module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path
import json
import yaml

from .config import PROJECT_ROOT
from .llm.config import LLMConfig
from .llm.factory import build_llm_provider
from .llm.parser import extract_json_object
from .task_router_prompts import task_router_prompt
from .task_router_schema import RouterJob, RouterResponse

DEFAULT_ROUTER_LLM_CONFIG = PROJECT_ROOT / "llm.example.yaml"


def build_default_router_provider():
    raw = yaml.safe_load(DEFAULT_ROUTER_LLM_CONFIG.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and "llm" in raw and isinstance(raw["llm"], dict):
        raw = raw["llm"]
    cfg = LLMConfig.from_mapping(raw)
    cfg.validate()
    provider = build_llm_provider(cfg)
    if provider is None:
        raise ValueError("task-router requires an enabled LLM config")
    return provider


def parse_router_response(payload: str) -> RouterResponse:
    raw = json.loads(extract_json_object(payload))
    if not isinstance(raw, dict):
        raise ValueError("router response must be a JSON object")
    workflow = str(raw.get("workflow", "")).strip()
    needs_clarification = bool(raw.get("needs_clarification", False))
    clarifying_questions = [str(item).strip() for item in raw.get("clarifying_questions", []) if str(item).strip()]
    job_data = raw.get("job")
    job = RouterJob(**job_data) if isinstance(job_data, dict) else None
    if needs_clarification and not clarifying_questions:
        raise ValueError("router response requested clarification but provided no questions")
    if not needs_clarification and job is None:
        raise ValueError("router response must include a job when clarification is not needed")
    return RouterResponse(
        workflow=workflow,
        needs_clarification=needs_clarification,
        clarifying_questions=clarifying_questions,
        job=job,
    )


def route_task(request: str, provider=None) -> RouterResponse:
    provider = provider or build_default_router_provider()
    response_text = provider.route_request(request)
    return parse_router_response(response_text)
```

```python
def extract_json_object(raw: str) -> str:
    return _extract_json_block(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
`python -m pytest tests/test_task_router.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/task_router.py des_multi_agent/llm/parser.py tests/test_task_router.py
git commit -m "feat: add task router parsing"
```

### Task 4: Add the `task-router` CLI subcommand

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
from des_multi_agent.cli import build_parser


def test_cli_parser_accepts_task_router_request():
    parser = build_parser()
    args = parser.parse_args(["task-router", "find DES partners for lidocaine"])
    assert args.command == "task-router"
    assert args.request == "find DES partners for lidocaine"
```

```python
from des_multi_agent.cli import build_parser


def test_cli_parser_still_accepts_existing_des_flags():
    parser = build_parser()
    args = parser.parse_args(["--component-a", "CCO", "--n", "5", "--checkpoint-path", "ckpt.pt"])
    assert args.command == "des"
    assert args.component_a == "CCO"
    assert args.n == 5
```

```python
from des_multi_agent.cli import main


def test_task_router_main_prints_json_only(monkeypatch, capsys):
    class FakeResponse:
        def to_json(self):
            return '{"workflow":"des","needs_clarification":false,"clarifying_questions":[],"job":{"component_a":"CCO"}}'

    monkeypatch.setattr("des_multi_agent.cli.route_task", lambda request: FakeResponse())
    main(["task-router", "find DES partners for lidocaine"])
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"workflow":"des","needs_clarification":false,"clarifying_questions":[],"job":{"component_a":"CCO"}}'
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`python -m pytest tests/test_cli.py::test_cli_parser_accepts_task_router_request tests/test_cli.py::test_cli_parser_still_accepts_existing_des_flags tests/test_cli.py::test_task_router_main_prints_json_only -v`

Expected: FAIL because the subcommand does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")
router_parser = subparsers.add_parser("task-router", help="Translate plain language into a JSON job")
router_parser.add_argument("request", help="Plain-language request to translate")
parser.set_defaults(command="des")
```

```python
if args.command == "task-router":
    response = route_task(args.request)
    print(response.to_json())
    return
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
`python -m pytest tests/test_cli.py::test_cli_parser_accepts_task_router_request tests/test_cli.py::test_cli_parser_still_accepts_existing_des_flags tests/test_cli.py::test_task_router_main_prints_json_only -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: add task router cli subcommand"
```

### Task 5: Update docs for the router workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Test: `tests/test_task_router_docs.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_task_router_docs_are_updated():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples = Path("examples/README.md").read_text(encoding="utf-8")
    assert "task-router" in readme
    assert "task-router" in tutorial
    assert "needs_clarification" in tutorial
    assert "task-router" in examples
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`python -m pytest tests/test_task_router_docs.py -v`

Expected: FAIL because the documentation does not mention the router yet.

- [ ] **Step 3: Write minimal implementation**

```markdown
## Task Router

The router converts plain-language requests into JSON jobs and does not execute workflows itself.

```bash
python -m des_multi_agent.cli task-router "find DES partners for lidocaine"
```

If the request is ambiguous, the output JSON will set `needs_clarification=true` and include follow-up questions.
```

```markdown
## Task Router

Use the `task-router` subcommand when you want the system to translate plain language into a job JSON object.
It uses the repository's default `llm.example.yaml` file and prints JSON only.
If the request is incomplete, it asks for clarification instead of guessing.
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`python -m pytest tests/test_task_router_docs.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md tests/test_task_router_docs.py
git commit -m "docs: add task router documentation"
```

### Task 6: Full verification

**Files:**
- No code changes expected

- [ ] **Step 1: Run the targeted router suite**

Run:
`python -m pytest tests/test_task_router_schema.py tests/test_task_router_prompts.py tests/test_llm_provider_router.py tests/test_task_router.py tests/test_cli.py tests/test_task_router_docs.py -q`

Expected: PASS

- [ ] **Step 2: Run the full suite**

Run:
`python -m pytest -q`

Expected: PASS

- [ ] **Step 3: Commit any final doc/test fixes**

```bash
git add -A
git commit -m "test: verify task router integration"
```
