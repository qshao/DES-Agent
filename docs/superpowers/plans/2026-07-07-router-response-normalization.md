# Router Response Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two router/JSON-normalization failure modes found in `docs/example-run-report-2026-07-06.md` (Finding 2) and `docs/vllm-example-run-report-2026-07-07.md` (Finding 1) — LLMs inventing non-canonical job field names, and a magic-string allow-list that can't catch every "use the default" paraphrase.

**Architecture:** One new module, `des_multi_agent/router_normalization.py`, providing `apply_field_aliases` (renames known non-canonical keys to `RouterJob` field names) and `resolve_path_or_default` (falls back to a default path unless the given value resolves to a real file). `task-router`/`task-execute` get only the aliasing; the plain-language example scripts get both, replacing their duplicated magic-string fallback logic.

**Tech Stack:** Python 3.11+, pytest, existing `des_multi_agent` package layout.

## Global Constraints

- Aliasing only ever *renames* a key; it must never invent a value for a field that is genuinely absent from the response.
- A canonical field name already present in the response always wins over its alias — aliasing must never overwrite a value the model got right.
- `task-router`/`task-execute` (`des_multi_agent/task_router.py:parse_router_response`) must NOT gain a path-existence fallback — only `apply_field_aliases`. A genuinely missing `config_path`/`checkpoint_path` must keep failing `RouterJob.missing_required_fields`, exactly as today.
- `FIELD_ALIASES` must not include `"model"` as a key — `tests/test_llm_parser.py::test_parse_router_response_ignores_extra_job_fields` asserts it stays dropped.
- The per-workflow field-name list in the router prompt must be generated from `REQUIRED_FIELDS_BY_WORKFLOW` (`des_multi_agent/task_router_schema.py`), not hand-duplicated, so it can't drift out of sync.
- `des_multi_agent/llm/parser.py` (`_extract_json_block` and everything else in that file) is out of scope — do not modify it.
- All 8 existing tests in `tests/test_llm_parser.py` must still pass unmodified after every task.

---

### Task 1: Create the `router_normalization` module

**Files:**
- Create: `des_multi_agent/router_normalization.py`
- Test: `tests/test_router_normalization.py`

**Interfaces:**
- Consumes: `des_multi_agent.paths.resolve_existing_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path` (raises `FileNotFoundError` if the resolved path doesn't exist).
- Produces: `apply_field_aliases(job_data: dict) -> dict` and `resolve_path_or_default(value, default: Path) -> str`, both imported by Task 2 (`task_router.py`) and Task 4 (the 4 example scripts).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_router_normalization.py`:

```python
from pathlib import Path

from des_multi_agent.router_normalization import apply_field_aliases, resolve_path_or_default


def test_apply_field_aliases_renames_known_alias():
    result = apply_field_aliases({"target_molecule": "ethanol"})
    assert result == {"component_a": "ethanol"}


def test_apply_field_aliases_keeps_canonical_when_both_present():
    result = apply_field_aliases({"target_molecule": "ethanol", "component_a": "CCO"})
    assert result == {"component_a": "CCO", "target_molecule": "ethanol"}


def test_apply_field_aliases_leaves_unrelated_keys_untouched():
    result = apply_field_aliases({"component_a": "CCO", "model": "gemma4:12b"})
    assert result == {"component_a": "CCO", "model": "gemma4:12b"}


def test_apply_field_aliases_maps_all_known_aliases():
    result = apply_field_aliases({
        "num_candidates": 20,
        "checkpoint": "ckpt.pt",
        "config": "config.yaml",
        "stability_model": "model.json",
    })
    assert result == {
        "n": 20,
        "checkpoint_path": "ckpt.pt",
        "config_path": "config.yaml",
        "stability_constant_model_path": "model.json",
    }


def test_resolve_path_or_default_returns_existing_path(tmp_path):
    real_file = tmp_path / "config.yaml"
    real_file.write_text("llm: {}")
    default = tmp_path / "default_config.yaml"
    result = resolve_path_or_default(str(real_file), default)
    assert result == str(real_file)


def test_resolve_path_or_default_falls_back_for_missing_path(tmp_path):
    default = tmp_path / "default_config.yaml"
    default.write_text("llm: {}")
    result = resolve_path_or_default("shipped_default", default)
    assert result == str(default)


def test_resolve_path_or_default_falls_back_for_falsy_value(tmp_path):
    default = tmp_path / "default_config.yaml"
    default.write_text("llm: {}")
    assert resolve_path_or_default(None, default) == str(default)
    assert resolve_path_or_default("", default) == str(default)


def test_resolve_path_or_default_falls_back_for_non_string_value(tmp_path):
    default = tmp_path / "default_config.yaml"
    default.write_text("llm: {}")
    assert resolve_path_or_default(True, default) == str(default)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_router_normalization.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'des_multi_agent.router_normalization'`

- [ ] **Step 3: Write the implementation**

Create `des_multi_agent/router_normalization.py`:

```python
from __future__ import annotations

from pathlib import Path

from .paths import resolve_existing_path

FIELD_ALIASES: dict[str, str] = {
    # component_a
    "molecule": "component_a",
    "target_molecule": "component_a",
    "target_compound": "component_a",
    "target_substance": "component_a",
    "chemical_formula": "component_a",
    "smiles": "component_a",
    # n
    "num_candidates": "n",
    "max_candidates": "n",
    "candidate_count": "n",
    "candidates": "n",
    # checkpoint_path
    "checkpoint": "checkpoint_path",
    "use_shipped_checkpoint": "checkpoint_path",
    # config_path
    "config": "config_path",
    "vllm_config": "config_path",
    "use_default_config": "config_path",
    # stability_constant_model_path
    "stability_model": "stability_constant_model_path",
    "stability_constant_model": "stability_constant_model_path",
}


def apply_field_aliases(job_data: dict) -> dict:
    out = dict(job_data)
    for alias, canonical in FIELD_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out.pop(alias)
    return out


def resolve_path_or_default(value, default: Path) -> str:
    if value:
        try:
            resolve_existing_path(str(value))
            return str(value)
        except FileNotFoundError:
            pass
    return str(default)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router_normalization.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/router_normalization.py tests/test_router_normalization.py
git commit -m "feat: add router_normalization module (field aliasing + path fallback)"
```

---

### Task 2: Wire field-name aliasing into `parse_router_response`

**Files:**
- Modify: `des_multi_agent/task_router.py`
- Test: `tests/test_llm_parser.py`

**Interfaces:**
- Consumes: `apply_field_aliases` from Task 1 (`des_multi_agent.router_normalization`).
- Produces: `parse_router_response` now tolerates the field-name aliases listed in `FIELD_ALIASES` in the incoming job JSON — no new exports.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_llm_parser.py` (near the other `parse_router_response` tests):

```python
def test_parse_router_response_applies_field_aliases():
    payload = (
        "{\"workflow\":\"des\",\"needs_clarification\":false,\"clarifying_questions\":[],"
        "\"job\":{\"target_molecule\":\"CCO\",\"num_candidates\":5,\"checkpoint\":\"ckpt.pt\",\"config\":\"ml_des_mp/config.yaml\"}}"
    )
    response = parse_router_response(payload)
    assert response.job is not None
    assert response.job.component_a == "CCO"
    assert response.job.n == 5
    assert response.job.checkpoint_path == "ckpt.pt"
    assert response.job.config_path == "ml_des_mp/config.yaml"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_parser.py::test_parse_router_response_applies_field_aliases -v`
Expected: FAIL with `ValueError: router response job is missing required fields for des: component_a, n, checkpoint_path, config_path` (raised via `response.validate()` inside `parse_router_response`)

- [ ] **Step 3: Write the minimal implementation**

In `des_multi_agent/task_router.py`, add the import:

```python
from .router_normalization import apply_field_aliases
```

Then modify `parse_router_response` — find this block:

```python
    job_data = raw.get("job")
    if isinstance(job_data, dict):
        allowed_fields = {item.name for item in fields(RouterJob)}
        filtered_job_data = {key: value for key, value in job_data.items() if key in allowed_fields}
```

Change it to:

```python
    job_data = raw.get("job")
    if isinstance(job_data, dict):
        job_data = apply_field_aliases(job_data)
        allowed_fields = {item.name for item in fields(RouterJob)}
        filtered_job_data = {key: value for key, value in job_data.items() if key in allowed_fields}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_parser.py -v -k "parse_router_response or task_router_prompt or extract_json_object"`
Expected: all pass, including `test_parse_router_response_applies_field_aliases` and every pre-existing router test (`test_parse_router_response_accepts_complete_job`, `test_parse_router_response_accepts_clarification_state`, `test_parse_router_response_ignores_extra_job_fields`, `test_parse_router_response_rejects_missing_des_required_field`, `test_parse_router_response_rejects_job_when_clarifying`, `test_parse_router_response_rejects_missing_metal_binding_required_field`)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/task_router.py tests/test_llm_parser.py
git commit -m "fix: apply field-name aliasing before parsing router response"
```

---

### Task 3: Enumerate field names in the router prompt

**Files:**
- Modify: `des_multi_agent/task_router_prompts.py`
- Test: `tests/test_llm_parser.py`

**Interfaces:**
- Consumes: `REQUIRED_FIELDS_BY_WORKFLOW` from `des_multi_agent.task_router_schema` (existing, `{"des": ("component_a", "n", "checkpoint_path", "config_path"), "metal-binding": ("metal_ion", "ligand_smiles", "stability_constant_model_path")}`).
- Produces: `ROUTER_SYSTEM_PROMPT` (module-level string, existing name) and `task_router_prompt(request, normalized=None) -> str` (existing signature, unchanged) now include the enumerated field names.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_llm_parser.py`:

```python
def test_task_router_prompt_lists_required_field_names_per_workflow():
    prompt = task_router_prompt("find DES partners for lidocaine")
    assert 'workflow="des"' in prompt
    assert "component_a, n, checkpoint_path, config_path" in prompt
    assert 'workflow="metal-binding"' in prompt
    assert "metal_ion, ligand_smiles, stability_constant_model_path" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_parser.py::test_task_router_prompt_lists_required_field_names_per_workflow -v`
Expected: FAIL — `assert "component_a, n, checkpoint_path, config_path" in prompt` is False (current prompt only says "Use existing CLI field names")

- [ ] **Step 3: Write the implementation**

Replace the full contents of `des_multi_agent/task_router_prompts.py`:

```python
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
```

Note: the old per-call line `"Use existing CLI field names for job fields.\n"` inside `task_router_prompt` is removed since `ROUTER_SYSTEM_PROMPT` now states the exact field names explicitly — keeping both would be redundant and the old vague phrasing is exactly what caused the original bug.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_parser.py -v -k "task_router_prompt"`
Expected: 2 passed (`test_task_router_prompt_mentions_clarify_state`, `test_task_router_prompt_lists_required_field_names_per_workflow`)

Then run the full router/parser test file to confirm nothing else broke:

Run: `pytest tests/test_llm_parser.py -v`
Expected: all passed (no regressions)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/task_router_prompts.py tests/test_llm_parser.py
git commit -m "fix: enumerate RouterJob field names in the router system prompt"
```

---

### Task 4: Consolidate the 4 plain-language example scripts

**Files:**
- Modify: `examples/plain_language_gemma4_12b/run_example.py`
- Modify: `examples/plain_language_gemma4_12b_vllm/run_example.py`
- Modify: `examples/plain_language_metal_binding_gemma4_12b/run_example.py`
- Modify: `examples/plain_language_metal_binding_gemma4_12b_vllm/run_example.py`

**Interfaces:**
- Consumes: `apply_field_aliases` and `resolve_path_or_default` from Task 1 (`des_multi_agent.router_normalization`).
- Produces: nothing new — this task only replaces duplicated logic in `_normalize_router_job` in each of the 4 files with calls into the shared module. No public interface changes.

These 4 files are not covered by any pytest unit test today (verified: no test imports `run_example.py` from any of these folders — they're only exercised live via `./run.sh` against a running Ollama/vLLM server, or via the frozen-baseline `output.txt` comparison in `tests/test_benchmarks_examples.py`, neither of which runs in this task). Verification here is a syntax/import check plus a careful diff review, not a new test suite — inventing pytest coverage for scripts the rest of the codebase intentionally keeps live-only would be against the project's existing convention.

- [ ] **Step 1: Update `examples/plain_language_gemma4_12b/run_example.py`**

Find this block:

```python
def _normalize_router_job(raw_job: dict, request_text: str) -> RouterJob:
    component_a = _extract_smiles_text(raw_job.get("component_a") or raw_job.get("chemical_formula") or raw_job.get("target_compound") or raw_job.get("target_substance") or request_text)
    request_component_a = _extract_smiles_text(request_text)
    if request_component_a and request_component_a != request_text.strip():
        component_a = request_component_a
    n_value = raw_job.get("n") or raw_job.get("max_candidates") or 5
    checkpoint_value = raw_job.get("checkpoint_path") or raw_job.get("checkpoint") or str(DEFAULT_CHECKPOINT_PATH)
    config_value = raw_job.get("config_path") or raw_job.get("config") or str(DEFAULT_CONFIG_PATH)

    if checkpoint_value in {None, "", "default", "ml_des_mp"}:
        checkpoint_path = str(DEFAULT_CHECKPOINT_PATH)
    else:
        checkpoint_path = str(checkpoint_value)
    if config_value in {None, "", "default"}:
        config_path = str(DEFAULT_CONFIG_PATH)
    else:
        config_path = str(config_value)

    return RouterJob(
        component_a=str(component_a).strip() if component_a is not None else None,
        n=int(n_value) if n_value is not None else None,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        llm_config=str(LLM_CONFIG_FILE),
    )
```

Replace it with:

```python
def _normalize_router_job(raw_job: dict, request_text: str) -> RouterJob:
    raw_job = apply_field_aliases(raw_job)
    component_a = _extract_smiles_text(raw_job.get("component_a") or request_text)
    request_component_a = _extract_smiles_text(request_text)
    if request_component_a and request_component_a != request_text.strip():
        component_a = request_component_a
    n_value = raw_job.get("n") or 5
    checkpoint_path = resolve_path_or_default(raw_job.get("checkpoint_path"), DEFAULT_CHECKPOINT_PATH)
    config_path = resolve_path_or_default(raw_job.get("config_path"), DEFAULT_CONFIG_PATH)

    return RouterJob(
        component_a=str(component_a).strip() if component_a is not None else None,
        n=int(n_value) if n_value is not None else None,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        llm_config=str(LLM_CONFIG_FILE),
    )
```

Add the import alongside the existing `from des_multi_agent...` imports near the top of the file:

```python
from des_multi_agent.router_normalization import apply_field_aliases, resolve_path_or_default
```

- [ ] **Step 2: Update `examples/plain_language_gemma4_12b_vllm/run_example.py` identically**

Apply the exact same two edits (the `_normalize_router_job` body replacement and the new import line) to this file. It is otherwise identical to the file in Step 1 except `LLM_CONFIG_FILE = SCRIPT_DIR / "llm.gemma4_12b_vllm.yaml"` — do not touch that line.

- [ ] **Step 3: Update `examples/plain_language_metal_binding_gemma4_12b/run_example.py`**

Find this block:

```python
def _normalize_router_job(raw_job: dict, request_text: str) -> RouterJob:
    metal_ion = (
        raw_job.get("metal_ion")
        or raw_job.get("target_metal")
        or raw_job.get("metal")
        or raw_job.get("ion")
        or _extract_field(r"metal[_\s-]*ion\s*[:=]?\s*([A-Za-z0-9+\-]+)", request_text)
    )
    ligand_smiles = (
        raw_job.get("ligand_smiles")
        or raw_job.get("target_compound")
        or raw_job.get("target_ligand")
        or raw_job.get("ligand")
        or _extract_field(r"ligand[_\s-]*smiles\s*[:=]?\s*([A-Za-z0-9@+\-#=\[\]\(\)\\/]+)", request_text)
    )
    stability_value = raw_job.get("stability_constant_model_path") or raw_job.get("checkpoint") or str(DEFAULT_STABILITY_PATH)
    if stability_value in {None, "", "default", "artifact"}:
        stability_path = str(DEFAULT_STABILITY_PATH)
    else:
        stability_path = str(stability_value)

    return RouterJob(
        metal_ion=str(metal_ion).strip() if metal_ion is not None else None,
        ligand_smiles=_extract_smiles_text(ligand_smiles),
        stability_constant_model_path=stability_path,
        llm_config=str(LLM_CONFIG_FILE),
    )
```

Replace it with:

```python
def _normalize_router_job(raw_job: dict, request_text: str) -> RouterJob:
    raw_job = apply_field_aliases(raw_job)
    metal_ion = (
        raw_job.get("metal_ion")
        or raw_job.get("target_metal")
        or raw_job.get("metal")
        or raw_job.get("ion")
        or _extract_field(r"metal[_\s-]*ion\s*[:=]?\s*([A-Za-z0-9+\-]+)", request_text)
    )
    ligand_smiles = (
        raw_job.get("ligand_smiles")
        or raw_job.get("target_ligand")
        or raw_job.get("ligand")
        or _extract_field(r"ligand[_\s-]*smiles\s*[:=]?\s*([A-Za-z0-9@+\-#=\[\]\(\)\\/]+)", request_text)
    )
    stability_path = resolve_path_or_default(raw_job.get("stability_constant_model_path"), DEFAULT_STABILITY_PATH)

    return RouterJob(
        metal_ion=str(metal_ion).strip() if metal_ion is not None else None,
        ligand_smiles=_extract_smiles_text(ligand_smiles),
        stability_constant_model_path=stability_path,
        llm_config=str(LLM_CONFIG_FILE),
    )
```

Add the same import line as Step 1. Two deliberate fallback removals from the original `_normalize_router_job`, both superseded by the shared module rather than lost:

- `or raw_job.get("checkpoint")` is dropped from the `stability_value` fallback chain: `FIELD_ALIASES` (Task 1) maps `"checkpoint"` to `checkpoint_path`, not `stability_constant_model_path` — `checkpoint_path` isn't a metal-binding field, so a model that says `"checkpoint": "..."` for a metal-binding job now gets that value routed to a field this script ignores. `resolve_path_or_default` already treats *any* missing/unresolvable `stability_constant_model_path` as "use the default," which is strictly more general than the single-key `"checkpoint"` fallback it replaces.
- `or raw_job.get("target_compound")` is dropped from the `ligand_smiles` fallback chain for the identical reason: `FIELD_ALIASES` also maps `"target_compound"` to `component_a` (a DES-only field, and a well-evidenced alias from the DES plain-language script's own original code — see Task 1's `FIELD_ALIASES`). Since `apply_field_aliases` runs first and pops `target_compound` into `component_a`, `raw_job.get("target_compound")` here would always be `None` post-aliasing — leaving it in would be dead code implying a fallback that can never fire. `ligand_smiles` still has 3 remaining fallback keys plus the regex extraction from the raw request text, so no real coverage is lost.

- [ ] **Step 4: Update `examples/plain_language_metal_binding_gemma4_12b_vllm/run_example.py` identically**

Apply the exact same edits from Step 3 to this file. It is otherwise identical except `LLM_CONFIG_FILE = SCRIPT_DIR / "llm.gemma4_12b_vllm.yaml"` — do not touch that line.

- [ ] **Step 5: Verify all 4 files are syntactically valid**

Run:
```bash
python -m py_compile examples/plain_language_gemma4_12b/run_example.py examples/plain_language_gemma4_12b_vllm/run_example.py examples/plain_language_metal_binding_gemma4_12b/run_example.py examples/plain_language_metal_binding_gemma4_12b_vllm/run_example.py
```
Expected: no output, exit code 0.

Then re-run Task 1's tests to confirm the shared module they now depend on is still correct:

Run: `pytest tests/test_router_normalization.py -v`
Expected: 8 passed

- [ ] **Step 6: (Optional, manual) Live smoke test if a local LLM server is available**

If Ollama or a vLLM server is already running locally, re-run one of the previously-failing examples and confirm the `FileNotFoundError` from vLLM report Finding 1 no longer occurs:

```bash
bash examples/plain_language_gemma4_12b_vllm/run.sh
tail -20 examples/plain_language_gemma4_12b_vllm/output.txt
```
Expected: a completed DES report (no traceback), or a graceful `[WARNING]` line — not a `FileNotFoundError: Path does not exist: ...shipped...` traceback. This step requires live infrastructure not guaranteed to be present; skip it if no LLM server is running, the automated verification in Step 5 is sufficient to mark this task done.

- [ ] **Step 7: Commit**

```bash
git add examples/plain_language_gemma4_12b/run_example.py examples/plain_language_gemma4_12b_vllm/run_example.py examples/plain_language_metal_binding_gemma4_12b/run_example.py examples/plain_language_metal_binding_gemma4_12b_vllm/run_example.py
git commit -m "refactor: consolidate plain-language example scripts onto router_normalization"
```

---

### Task 5: Full regression pass

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -q`
Expected: all tests pass, including the new `tests/test_router_normalization.py` (8 tests) and the 2 new/modified tests in `tests/test_llm_parser.py`. No prior test's pass/fail status changes.

- [ ] **Step 2: Run the targeted router/parser tests once more in verbose mode as a final sanity check**

Run: `pytest tests/test_router_normalization.py tests/test_llm_parser.py -v`
Expected: every test name printed with `PASSED`, none `FAILED`/`ERROR`.

- [ ] **Step 3: Confirm no unrelated files changed**

Run: `git status --short`
Expected: clean (everything from Tasks 1–4 already committed); no unstaged changes.

- [ ] **Step 4: Commit (only if Step 1 required any fixes)**

If Step 1 was clean on the first run, there is nothing to commit for this task — it's a verification-only checkpoint. If a regression was found and fixed, commit that fix with a message describing what broke and why:

```bash
git add -A
git commit -m "fix: <describe the regression fixed during final verification>"
```
