# Thinking-Model JSON Extraction Design

## Context

`docs/vllm-example-run-report-2026-07-07.md`, Finding 2, and `docs/future-improvements.md`'s
follow-up item 1 documented a bug in `des_multi_agent/llm/parser.py:_extract_json_block`, the
single shared JSON-extraction function used by every LLM response parser in the codebase (the
router's `extract_json_object`, and the 8 `_coerce_json`-based parsers plus
`parse_candidate_review` — `parse_candidate_brainstorms`, `parse_explanation_notes`,
`parse_critique_notes`, `parse_contradiction_notes`, `parse_candidate_families`,
`parse_ligand_families`, `parse_chemistry_assessments`, `parse_chemistry_next_steps`).

`Qwen/Qwen3.6-35B-A3B-FP8` (served via vLLM) is a "thinking" model: it reasons inside literal
`<think>...</think>` tags before producing its final answer, and that reasoning text itself
contains draft/example JSON snippets while the model works out field names. Confirmed empirically
in this session by calling the provider directly:

```
Here's a thinking process:
...
4.  **Construct JSON:**
   ```json
   { "workflow": "des", ... "molecule": "ethanol", ... }
   ```
   Wait, should I use exact CLI field names? ...
</think>

{
  "workflow": "des",
  "needs_clarification": false,
  ...
}
```

`_extract_json_block`'s regex (`\{[\s\S]*\}`, greedy, first `{` to the *last* `}` in the whole
text) spans from the first draft block's opening brace to the real answer's closing brace in one
match — producing invalid or wrong-shaped JSON. Symptom varies by exact sampling (observed:
`Expecting value: ...`, `missing required fields for des: ...`, `must be a JSON object`) but the
root cause is single and confirmed, not inferred.

**Goal:** fix `_extract_json_block` so a `<think>...</think>`-wrapped response is parsed correctly,
with zero behavior change for the (current, universal) case where no such tag is present.

## Approach

Add one new private helper, `_strip_think_blocks`, called as the first step inside
`_extract_json_block` — `<think>` is the outermost wrapper, so it must be removed before
`_strip_code_fences` and the existing brace-matching regex run.

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

Because all 10 call sites route through `_extract_json_block`, this single change fixes the bug
everywhere at once — no other function in `parser.py` needs to change.

**Behavior:**

- **No `<think>`/`</think>` anywhere** (every model/provider today) → `_strip_think_blocks` returns
  the input unchanged, byte-for-byte. Zero behavior change for the current, working path. The
  existing `test_extract_json_object_returns_json_payload` test (no think-tag, plain "thinking..."
  prose + fenced JSON) already guards this.
- **`</think>` present** → everything up to and including the *last* occurrence is discarded;
  extraction proceeds on what's left exactly as it does today for a direct-answer response. Using
  the *last* occurrence correctly handles multiple think-blocks (e.g. a model re-reasoning mid
  response: "Wait, let me reconsider...").
- **Case variation** (`<Think>`, `<THINK>`) — handled via `.lower()` for the search only; slicing
  uses positions from the original string, which stay aligned since case-folding doesn't change
  ASCII tag length. Included defensively even though only lowercase has been observed — it's a
  one-line addition, not new surface area.
- **`<think>` present but never closed** (max_tokens cut the response off mid-reasoning) → raises
  `ValueError` with an actionable message instead of silently extracting garbage from inside
  unfinished reasoning. This propagates exactly like today's `json.JSONDecodeError`-derived
  `ValueError` — every orchestrator call site already wraps its parse call in
  `except Exception as exc: llm_warnings.append(f"... failed: {exc}")` (verified directly in
  `des_multi_agent/orchestrator.py`), so the new error surfaces as a warning with a materially more
  useful message than today's generic `"LLM response is not valid JSON. Excerpt: ..."` — with no
  change to the existing graceful-degradation contract.

## Out of scope

- **Balanced-brace / non-greedy JSON scanning.** Content *after* `</think>` that itself contains
  more than one JSON-ish block (e.g. trailing commentary with a stray brace following the real
  answer) is not addressed — the existing greedy regex is unchanged for post-think text. No
  evidence exists for a real failure needing this; only the pre-answer contamination documented in
  Finding 2 is being fixed. (Considered and explicitly rejected in favor of the targeted fix — see
  brainstorming discussion.)
- Any change to `orchestrator.py`'s exception handling — the existing broad `except Exception`
  blocks already handle the new error type correctly; nothing there needs to change.
- Any change to `task_router.py`, `router_normalization.py`, or any other file from the prior
  router-response-normalization branch — this is a separate, independent fix to a different shared
  function.

## Testing

All new tests go in the existing `tests/test_llm_parser.py` (this modifies an already-tested
module, not a new one):

1. `test_extract_json_object_strips_think_block` — a condensed but realistic version of the actual
   raw Qwen3.6 response captured in this session (draft JSON inside `<think>`, real JSON after
   `</think>`) — asserts only the real answer is extracted.
2. `test_extract_json_object_strips_think_block_case_insensitively` — same shape with
   `<THINK>`/`</THINK>`.
3. `test_extract_json_object_raises_clear_error_for_unclosed_think_tag` — `<think>` with no closing
   tag anywhere — asserts `ValueError` mentioning the truncation/`max_tokens` hint.
4. `test_parse_candidate_brainstorms_handles_think_wrapped_response` — proves the fix propagates
   through the `_coerce_json` path used by 8 other parsers, not just the router's direct path.
5. `test_parse_router_response_handles_think_wrapped_payload` — end-to-end through
   `parse_router_response`, mirroring the actual real-world failure from the vLLM report.

## Verification

1. `pytest tests/test_llm_parser.py -v` — new and existing tests green, including the 5 new tests
   above and every pre-existing test in the file (in particular `test_extract_json_object_returns_json_payload`,
   which proves the no-think-tag path is unchanged).
2. `pytest tests/ -q` — full suite green, no regressions.
3. Manual verification (optional, requires a live vLLM server serving `Qwen/Qwen3.6-35B-A3B-FP8`):
   re-run `examples/task_router_vllm/` and `examples/task_execute_vllm/` (both currently fail 100%
   of attempts per the vLLM report's Finding 2) and confirm they now succeed, or at minimum no
   longer fail with a JSON-extraction-shaped error.
