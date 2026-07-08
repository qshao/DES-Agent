# Thinking-Model JSON Extraction Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `des_multi_agent/llm/parser.py:_extract_json_block` so it correctly handles LLM responses that wrap reasoning in `<think>...</think>` tags containing draft JSON before the real answer (documented in `docs/vllm-example-run-report-2026-07-07.md`, Finding 2), with zero behavior change for responses that don't use this pattern.

**Architecture:** One new private helper, `_strip_think_blocks`, called as the first step inside `_extract_json_block` — before the existing code-fence stripping and brace-matching regex. Because all 10 JSON-parsing call sites in `parser.py` route through `_extract_json_block`, this one change fixes the bug everywhere at once.

**Tech Stack:** Python 3.11+, pytest, existing `des_multi_agent.llm.parser` module.

## Global Constraints

- No `<think>`/`</think>` anywhere in a response → behavior must be byte-for-byte unchanged from today. The existing test `test_extract_json_object_returns_json_payload` (no think-tag, plain "thinking..." prose + fenced JSON) must keep passing unmodified — it is the regression guard for this.
- When `</think>` is present, use the **last** occurrence (handles a model re-reasoning mid-response) and discard everything up to and including it.
- An unclosed `<think>` tag (no `</think>` anywhere in the response) must raise `ValueError` with a message that mentions the response was truncated and suggests raising `max_tokens` — not the generic "not valid JSON" message.
- Balanced-brace / non-greedy JSON scanning for content *after* `</think>` is explicitly out of scope — do not change the existing regex's behavior for the post-think text.
- Do not modify `orchestrator.py`, `task_router.py`, or `router_normalization.py` — the existing `except Exception as exc: llm_warnings.append(...)` pattern in `orchestrator.py` already handles the new `ValueError` correctly with no changes needed.

---

### Task 1: Add `_strip_think_blocks` and wire it into `_extract_json_block`

**Files:**
- Modify: `des_multi_agent/llm/parser.py`
- Test: `tests/test_llm_parser.py`

**Interfaces:**
- Consumes: nothing new — `_strip_think_blocks` is a pure string function with no dependencies beyond the standard library.
- Produces: `_extract_json_block(raw: str) -> str` (existing name and signature, unchanged) now strips `<think>...</think>` content before its existing logic runs. No new public exports — `extract_json_object`, `_coerce_json`, and `parse_candidate_review` (the existing call sites) all benefit automatically since they all call `_extract_json_block` internally.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm_parser.py` (all needed imports — `extract_json_object`, `parse_candidate_brainstorms`, `parse_router_response`, `pytest` — are already present at the top of this file):

```python
def test_extract_json_object_strips_think_block():
    raw = (
        "<think>\n"
        "Let me work out the fields.\n"
        "Draft attempt:\n"
        "```json\n"
        "{\"workflow\": \"draft\", \"job\": {}}\n"
        "```\n"
        "Actually, let me reconsider and use the real field names.\n"
        "</think>\n"
        "\n"
        "{\"workflow\": \"des\", \"job\": {\"component_a\": \"CCO\"}}"
    )
    payload = extract_json_object(raw)
    assert payload == "{\"workflow\": \"des\", \"job\": {\"component_a\": \"CCO\"}}"


def test_extract_json_object_strips_think_block_case_insensitively():
    raw = "<THINK>draft {\"a\": 1}</THINK>\n{\"workflow\": \"des\"}"
    payload = extract_json_object(raw)
    assert payload == "{\"workflow\": \"des\"}"


def test_extract_json_object_raises_clear_error_for_unclosed_think_tag():
    raw = "<think>\nStill reasoning about the fields and never finishing"
    with pytest.raises(ValueError, match="unclosed <think>"):
        extract_json_object(raw)


def test_extract_json_object_uses_last_think_close_with_multiple_blocks():
    raw = (
        "<think>First pass, considering options.</think>"
        "<think>Wait, reconsidering: draft {\"a\": 1}</think>\n"
        "{\"workflow\": \"des\"}"
    )
    payload = extract_json_object(raw)
    assert payload == "{\"workflow\": \"des\"}"


def test_parse_candidate_brainstorms_handles_think_wrapped_response():
    raw = (
        "<think>\n"
        "Draft candidates: [{\"smiles\": \"X\", \"rationale\": \"draft\", \"family\": \"draft\"}]\n"
        "</think>\n"
        "[{\"smiles\": \"OCCO\", \"rationale\": \"diol H-bonding\", \"family\": \"diol\"}]"
    )
    result = parse_candidate_brainstorms(raw)
    assert len(result) == 1
    assert result[0].smiles == "OCCO"
    assert result[0].family == "diol"


def test_parse_router_response_handles_think_wrapped_payload():
    raw = (
        "<think>\n"
        "Draft: {\"workflow\": \"draft\"}\n"
        "Let me finalize.\n"
        "</think>\n"
        "{\"workflow\":\"des\",\"needs_clarification\":false,\"clarifying_questions\":[],"
        "\"job\":{\"component_a\":\"CCO\",\"n\":5,\"checkpoint_path\":\"ckpt.pt\",\"config_path\":\"ml_des_mp/config.yaml\"}}"
    )
    response = parse_router_response(raw)
    assert response.workflow == "des"
    assert response.job is not None
    assert response.job.component_a == "CCO"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_parser.py -v -k "think"`
Expected: all 6 FAIL. `_strip_think_blocks` doesn't exist yet, so the current greedy regex spans from the first `{` inside a draft block to the last `}` in the whole text — producing a mismatched/malformed span for 5 of the 6 tests (assertion failures on wrong content, or a `ValueError`/`JSONDecodeError` from downstream parsing of that malformed span) and "DID NOT RAISE" for the unclosed-tag test (today, no exception is raised for it at all — the regex just finds no braces in that input and returns the text unchanged).

- [ ] **Step 3: Write the implementation**

In `des_multi_agent/llm/parser.py`, find this function:

```python
def _extract_json_block(raw: str) -> str:
    text = _strip_code_fences(raw)
    if text.startswith("[") or text.startswith("{"):
        return text
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if match:
        return match.group(1)
    return text
```

Add a new function above it and change its first line:

```python
def _strip_think_blocks(raw: str) -> str:
    lowered = raw.lower()
    last_close = lowered.rfind("</think>")
    if last_close == -1:
        if "<think>" in lowered:
            raise ValueError(
                "LLM response contains an unclosed <think> tag — the response was likely "
                "truncated before reasoning completed. Consider raising max_tokens."
            )
        return raw
    return raw[last_close + len("</think>"):]


def _extract_json_block(raw: str) -> str:
    text = _strip_think_blocks(raw)
    text = _strip_code_fences(text)
    if text.startswith("[") or text.startswith("{"):
        return text
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if match:
        return match.group(1)
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_parser.py -v -k "think"`
Expected: 6 passed

- [ ] **Step 5: Run the full parser/router test file to confirm no regressions**

Run: `pytest tests/test_llm_parser.py -v`
Expected: every test passes, including `test_extract_json_object_returns_json_payload` (the no-think-tag regression guard) and all pre-existing `parse_router_response`/`parse_candidate_*` tests, unmodified.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/llm/parser.py tests/test_llm_parser.py
git commit -m "fix: strip <think> reasoning blocks before JSON extraction"
```

---

### Task 2: Full regression pass

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -q`
Expected: all tests pass, including the 6 new tests from Task 1. No prior test's pass/fail status changes (baseline before this plan: 987 passed).

- [ ] **Step 2: Confirm no unrelated files changed**

Run: `git status --short`
Expected: clean (Task 1's commit already captured everything); no unstaged changes.

- [ ] **Step 3: (Optional, manual) Live verification if a vLLM server serving Qwen/Qwen3.6-35B-A3B-FP8 is available**

```bash
bash examples/task_router_vllm/run.sh
cat examples/task_router_vllm/output.txt
```
Expected: a valid JSON job object (no `cli.py: error: ...` traceback) — this example failed 100% of attempts before this fix, per `docs/vllm-example-run-report-2026-07-07.md` Finding 2. This step requires live infrastructure not guaranteed to be present; skip it if no vLLM server is running — Task 1's automated tests are sufficient to mark this plan done.

- [ ] **Step 4: Commit (only if Step 1 required a fix)**

If Step 1 was clean, there is nothing to commit for this task. If a regression was found and fixed, commit it with a message describing what broke and why:

```bash
git add -A
git commit -m "fix: <describe the regression fixed during final verification>"
```
