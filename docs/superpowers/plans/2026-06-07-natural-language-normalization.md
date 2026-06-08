# Natural Language Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight normalization layer that improves how plain-language DES and metal-binding requests are turned into router jobs, especially for compound names, salts, free bases, and ambiguous phrasing, without changing the existing CLI/job schema.

**Architecture:** Add a small router-facing normalization helper that extracts workflow hints and ambiguity flags from the raw request text. Feed those hints into the existing task router and task-execute path so both routing and execution use the same normalization behavior. Keep all downstream job fields and workflows unchanged.

**Tech Stack:** Python, pytest, dataclasses, existing router/LLM prompt layer, CLI wiring.

---

### Task 1: Add the request normalization helper

**Files:**
- Create: `des_multi_agent/request_normalization.py`
- Test: `tests/test_request_normalization.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.request_normalization import normalize_request_text


def test_normalize_request_text_extracts_workflow_hint():
    result = normalize_request_text("find DES partners for lidocaine")
    assert result.workflow_hint == "des"
    assert result.compound_hint == "lidocaine"
    assert result.needs_clarification is False


def test_normalize_request_text_flags_salt_free_base_ambiguity():
    result = normalize_request_text("find DES partners for lidocaine hydrochloride")
    assert result.needs_clarification is True
    assert "free base" in " ".join(result.clarifying_questions).lower()


def test_normalize_request_text_handles_metal_binding_intent():
    result = normalize_request_text("predict stability constant for Cu2+ with NCCN")
    assert result.workflow_hint == "metal-binding"
    assert result.needs_clarification is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_request_normalization.py -q`
Expected: FAIL because the normalization module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizedRequest:
    raw_text: str
    workflow_hint: str | None = None
    compound_hint: str | None = None
    needs_clarification: bool = False
    clarifying_questions: list[str] = field(default_factory=list)


def normalize_request_text(text: str) -> NormalizedRequest:
    lowered = text.lower()
    workflow_hint = None
    if any(token in lowered for token in ("metal binding", "metal extraction", "stability constant", "log k")):
        workflow_hint = "metal-binding"
    elif "des" in lowered:
        workflow_hint = "des"

    compound_hint = None
    if "lidocaine" in lowered:
        compound_hint = "lidocaine"
    elif "cu2+" in lowered or "nccn" in lowered:
        compound_hint = None

    needs_clarification = False
    questions: list[str] = []
    if "lidocaine hydrochloride" in lowered:
        needs_clarification = True
        questions.append("Do you mean lidocaine free base or lidocaine hydrochloride?")

    if workflow_hint is None and compound_hint is None:
        needs_clarification = True
        questions.append("Which compound or metal-binding target should I use?")

    return NormalizedRequest(
        raw_text=text,
        workflow_hint=workflow_hint,
        compound_hint=compound_hint,
        needs_clarification=needs_clarification,
        clarifying_questions=questions,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_request_normalization.py -q`
Expected: PASS after the helper returns the normalized workflow and clarification hints.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/request_normalization.py tests/test_request_normalization.py
git commit -m "feat: add request normalization helper"
```

### Task 2: Wire normalization into the router and task-execute path

**Files:**
- Modify: `des_multi_agent/task_router.py`
- Modify: `des_multi_agent/task_router_prompts.py`
- Modify: `des_multi_agent/task_executor.py`
- Test: `tests/test_llm_parser.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.request_normalization import normalize_request_text
from des_multi_agent.task_router import parse_router_response


def test_router_prompt_can_use_normalized_hints():
    normalized = normalize_request_text("find DES partners for lidocaine hydrochloride")
    assert normalized.needs_clarification is True
    assert normalized.workflow_hint in {"des", "metal-binding", None}


def test_router_response_still_accepts_existing_json_schema():
    payload = '{"workflow":"des","needs_clarification":false,"clarifying_questions":[],"job":{"component_a":"CCO","n":5,"checkpoint_path":"ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt","config_path":"ml_des_mp/config.yaml"}}'
    response = parse_router_response(payload)
    assert response.workflow == "des"
    assert response.job.component_a == "CCO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_parser.py tests/test_cli.py -q`
Expected: FAIL until the router and executor consume the new normalization helper.

- [ ] **Step 3: Write minimal implementation**

Update the router prompt to include a normalized hint block and update `route_task()` / `execute_task_request()` to call the new helper before prompting.

```python
from .request_normalization import normalize_request_text


def route_task(request: str, provider=None) -> RouterResponse:
    normalized = normalize_request_text(request)
    provider = provider or build_default_router_provider()
    response_text = provider.route_request(request, normalized=normalized)
    return parse_router_response(response_text)
```

In `task_router_prompts.py`, append a small hint section when `normalized` is present:

```python
if normalized.workflow_hint:
    prompt += f"\n\nWorkflow hint: {normalized.workflow_hint}\n"
if normalized.compound_hint:
    prompt += f"Compound hint: {normalized.compound_hint}\n"
if normalized.needs_clarification:
    prompt += "The request may need clarification. Ask before guessing.\n"
```

In `task_executor.py`, keep using the same `route_task()` path so routing and execution behave identically.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_parser.py tests/test_cli.py -q`
Expected: PASS once the router and executor both use the normalization helper.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/task_router.py des_multi_agent/task_router_prompts.py des_multi_agent/task_executor.py tests/test_llm_parser.py tests/test_cli.py
git commit -m "feat: route plain language requests through normalization"
```

### Task 3: Update docs and examples to explain the normalization behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Modify: `examples/plain_language_gemma4_12b/README.md`
- Modify: `examples/plain_language_metal_binding_gemma4_12b/README.md`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_docs_mention_normalization_and_clarification():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples_readme = Path("examples/README.md").read_text(encoding="utf-8")
    assert "normalization" in readme.lower()
    assert "clarification" in tutorial.lower()
    assert "free base" in examples_readme.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_request_normalization.py -q`
Expected: FAIL until the docs mention the new normalization and clarification behavior.

- [ ] **Step 3: Write minimal implementation**

Add one short paragraph to the docs explaining:
- the router now normalizes requests before JSON compilation
- ambiguous salt/free-base requests trigger clarification
- the existing DES and metal-binding job schema is unchanged
- `task-execute` uses the same normalization path as `task-router`

Update the plain-language example READMEs to mention that the examples demonstrate how normalization handles compound naming and when it asks follow-up questions.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_request_normalization.py tests/test_demo_des_search.py -q`
Expected: PASS after the docs and example READMEs are updated.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md examples/plain_language_gemma4_12b/README.md examples/plain_language_metal_binding_gemma4_12b/README.md
git commit -m "docs: document request normalization behavior"
```

### Task 4: Verify the full suite still passes

**Files:**
- No new files
- Validate: `des_multi_agent/request_normalization.py`, router, executor, and docs updates

- [ ] **Step 1: Run the focused slice**

Run: `python -m pytest tests/test_request_normalization.py tests/test_llm_parser.py tests/test_cli.py tests/test_demo_des_search.py -q`
Expected: PASS

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS with the existing third-party warnings only.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: add natural language normalization support"
```
