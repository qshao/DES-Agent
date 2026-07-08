# Router Response Normalization Design

## Context

Two re-run reports surfaced related, reproducible failures at the boundary between an LLM's raw
JSON response and the code that turns it into a `RouterJob`:

- **`docs/example-run-report-2026-07-06.md`, Finding 2 (+ addendum):** `task-router`/`task-execute`
  fail frequently (4/6 and 5/5 in a repeated-attempt sample) because `ROUTER_SYSTEM_PROMPT`
  (`des_multi_agent/task_router_prompts.py`) only says *"Use existing CLI field names"* without ever
  listing them. Models invent plausible-but-wrong names (e.g. `target_molecule` instead of
  `component_a`); `parse_router_response`'s field filter (`des_multi_agent/task_router.py`) silently
  drops any key that isn't an exact `RouterJob` field name, so the invented key just disappears and
  the subsequent required-field check fails.
- **`docs/vllm-example-run-report-2026-07-07.md`, Finding 1:** `plain_language_gemma4_12b`'s
  `_normalize_router_job` (and its near-duplicate in `plain_language_metal_binding_gemma4_12b`, and
  both `_vllm` twins — 4 files total) treats a job field as "use the default" only for a fixed
  string set (`{None, "", "default", ...}`). The vLLM-served Gemma checkpoint reliably paraphrases
  "shipped ... config" as `"shipped"`/`"shipped_default"`/`"shipped_config"` — none of which match —
  so `resolve_existing_path` (`des_multi_agent/paths.py`) raises `FileNotFoundError` on the literal
  invented string.

**Goal:** fix both failure modes without touching the shared JSON-extraction layer
(`des_multi_agent/llm/parser.py:_extract_json_block`, which is explicitly out of scope for this
plan — it has a separate, unrelated failure mode against "thinking" models, documented as a
follow-up in the vLLM report but not part of this design).

## Architecture

One new module, `des_multi_agent/router_normalization.py`, is the single place this logic lives. It
exposes two independent, composable functions:

- **`apply_field_aliases(job_data: dict) -> dict`** — renames a curated set of observed
  non-canonical key names to their canonical `RouterJob` field names. Never invents a value for a
  field that's genuinely absent; a canonical key already present in `job_data` always wins over its
  alias.
- **`resolve_path_or_default(value, default: Path) -> str`** — returns `str(value)` if it resolves
  to an existing file via `des_multi_agent/paths.py:resolve_existing_path`; otherwise returns
  `str(default)`. Replaces every magic-string allow-list with one existence check, so any future
  paraphrase the model invents is caught the same way, with no new special-casing needed.

**Two call sites use these differently:**

| Call site | `apply_field_aliases` | `resolve_path_or_default` |
|---|---|---|
| `task_router.py:parse_router_response` (used by both `task-router` and `task-execute`) | Yes | **No** |
| 4 plain-language example scripts' router-job normalization | Yes | Yes |

`task-router` is workflow-generic and has no per-workflow "sensible default" path to fall back to —
it only gets aliasing. A genuinely missing `config_path` (not just a wrongly-named one) must keep
failing loudly via the existing `RouterJob.missing_required_fields` check, exactly as
`test_parse_router_response_rejects_missing_des_required_field` already asserts. The example
scripts, written for one specific workflow, already hardcode a real default checkpoint/config path
(`DEFAULT_CHECKPOINT_PATH`, `DEFAULT_CONFIG_PATH`, and the metal-binding script's stability-model
default) — they're the only call sites with a legitimate default to substitute.

## Components

### 1. Prompt fix (`des_multi_agent/task_router_prompts.py`)

`ROUTER_SYSTEM_PROMPT` gains explicit per-workflow field-name enumeration, generated from
`REQUIRED_FIELDS_BY_WORKFLOW` (`des_multi_agent/task_router_schema.py`) rather than hand-typed
separately, so the prompt cannot drift out of sync with the schema:

```python
ROUTER_SYSTEM_PROMPT = """You are a task router. Convert the user's request into strict JSON only.
If inputs are missing or ambiguous, return clarification questions.
Support workflows: des, metal-binding. If the workflow is unclear, use workflow="clarify" and ask a workflow question. If clarification is needed, set job to null.

For workflow="des", the job object must use exactly these field names: component_a, n, checkpoint_path, config_path.
For workflow="metal-binding", the job object must use exactly these field names: metal_ion, ligand_smiles, stability_constant_model_path.
Do not invent other field names.
Do not execute anything."""
```

This is a reduction measure, not a guarantee — models will still occasionally deviate. Its job is
to lower the *rate* of invented field names; `apply_field_aliases` and the existing required-field
validation remain the safety net for when they don't.

### 2. Field-name aliasing (`des_multi_agent/router_normalization.py`)

```python
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
```

The list is drawn only from names actually observed in this session's transcripts and the two
example scripts' own pre-existing ad-hoc `or`-chained alias lookups (e.g.
`raw_job.get("component_a") or raw_job.get("chemical_formula") or raw_job.get("target_compound")
or raw_job.get("target_substance")`) — not a guessed exhaustive list. Deliberately excludes
generic/ambiguous terms like `"model"`, which appears in
`test_parse_router_response_ignores_extra_job_fields` as a field that must stay dropped.

Integration in `task_router.py:parse_router_response`, immediately before the existing
`allowed_fields` filter:

```python
job_data = raw.get("job")
if isinstance(job_data, dict):
    job_data = apply_field_aliases(job_data)
    allowed_fields = {item.name for item in fields(RouterJob)}
    filtered_job_data = {k: v for k, v in job_data.items() if k in allowed_fields}
    ...
```

A canonical key already present in the response always wins — aliasing only fires when the
canonical name is absent, so it can never clobber a value the model got right.

### 3. Path-existence fallback (`des_multi_agent/router_normalization.py`)

```python
from pathlib import Path
from .paths import resolve_existing_path

def resolve_path_or_default(value, default: Path) -> str:
    if value:
        try:
            resolve_existing_path(str(value))
            return str(value)
        except FileNotFoundError:
            pass
    return str(default)
```

Reuses `des_multi_agent/paths.py:resolve_existing_path` — already imported by `orchestrator.py`,
already the exact function whose `FileNotFoundError` produced the traceback in vLLM report Finding
1 — so "does this look like a real path" is defined in exactly one place. A falsy `value` (`None`,
`""`, `False`) or a non-string truthy value (`True`, a number) both fall back to `default` via the
same code path: `str(value)` is passed to `resolve_existing_path`, which raises `FileNotFoundError`
for anything that isn't a real file, caught uniformly.

### 4. Example-script consolidation

The 4 duplicated `_normalize_router_job` implementations (`plain_language_gemma4_12b`,
`plain_language_metal_binding_gemma4_12b`, and their `_vllm` twins) each shrink to their own
field-specific extraction, delegating both the renaming and the fallback decision to the shared
module:

```python
# before (plain_language_gemma4_12b/run_example.py)
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

# after
from des_multi_agent.router_normalization import apply_field_aliases, resolve_path_or_default

raw_job = apply_field_aliases(raw_job)
checkpoint_path = resolve_path_or_default(raw_job.get("checkpoint_path"), DEFAULT_CHECKPOINT_PATH)
config_path = resolve_path_or_default(raw_job.get("config_path"), DEFAULT_CONFIG_PATH)
```

Calling `apply_field_aliases` here too means the example scripts benefit from the same
single-sourced alias list as `task-router`, replacing their previous hand-picked `or`-chains.
`plain_language_metal_binding_gemma4_12b`'s `stability_constant_model_path` field follows the
identical pattern with its own default.

This directly fixes vLLM report Finding 1: `"shipped"`/`"shipped_default"`/`"shipped_config"` all
fail the existence check and fall back to `DEFAULT_CONFIG_PATH`, with no `FileNotFoundError` — and
any future paraphrase is caught the same way, for free.

## Out of scope

- `des_multi_agent/llm/parser.py:_extract_json_block` and the "thinking model" JSON-extraction
  failure (vLLM report Finding 2) — separate root cause, separate follow-up.
- Giving `task-router`/`task-execute` a path-existence fallback for `checkpoint_path`/`config_path`/
  `stability_constant_model_path` — intentionally excluded; see Architecture section for why.
- Any change to `RouterJob`, `RouterResponse.validate()`, or `REQUIRED_FIELDS_BY_WORKFLOW`.

## Testing

- New `tests/test_router_normalization.py`: unit tests for `apply_field_aliases` (each alias
  renames correctly; a present canonical key always wins over its alias; unrelated keys like
  `"model"` pass through untouched) and `resolve_path_or_default` (existing path → unchanged;
  missing/bogus/falsy/non-string value → default). All synthetic dicts/strings, no live LLM calls —
  deterministic and fast, following TDD.
- Extend `tests/test_llm_parser.py`: add a case feeding `parse_router_response` a job using alias
  names instead of canonical ones (e.g. `target_molecule` instead of `component_a`) and assert it
  now succeeds. Confirm all existing router tests in that file still pass unmodified — they should,
  since aliasing never touches an already-correct payload, and `task-router` never gets the
  path-fallback that would change the missing-required-field contract.
- Add an assertion (new or extended test) that `task_router_prompt(...)` contains `component_a`,
  `n`, `checkpoint_path`, `config_path` for the des-workflow section of the prompt.
- Full `pytest tests/ -q` regression pass at the end of implementation.

## Verification

1. `pytest tests/test_router_normalization.py tests/test_llm_parser.py -v` — new and existing unit
   tests green.
2. `pytest tests/ -q` — full suite green, no regressions.
3. Manual smoke test (optional, requires a live LLM): re-run `examples/plain_language_gemma4_12b/`
   and confirm it still succeeds; re-run `examples/plain_language_gemma4_12b_vllm/` (previously 8/8
   failures) and confirm `resolve_path_or_default` now recovers from the `"shipped_default"`-style
   response instead of crashing.
