# One-by-One LLM Candidate Review Implementation Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-candidate-at-a-time LLM review layer so Gemma, Nemotron, and Qwen can evaluate up to 20 candidates without large JSON batch failures.

**Architecture:** Candidate discovery and deterministic DES prediction stay unchanged. After filtering, the orchestrator sends the top N candidates to the LLM one by one. Each request returns a strict JSON object for a single candidate, which the parser promotes into an internal review record and uses to adjust ranking or filtering. This keeps the LLM output small and isolates parse failures to a single candidate.

**Tech Stack:** Python, existing DES multi-agent pipeline, Ollama-backed LLM adapters, strict JSON parser, pytest.

---

## One-Candidate Review Record

Each LLM review returns exactly one JSON object with this shape:

```json
{
  "smiles": "OCCO",
  "decision": "keep",
  "confidence": 0.87,
  "rationale": "Short reason why this candidate should stay in the set.",
  "notes": ["Optional short note", "Optional second note"]
}
```

Rules:
- `smiles` must match the reviewed candidate exactly.
- `decision` must be one of `keep`, `reject`, or `deprioritize`.
- `confidence` must be a float between `0.0` and `1.0`.
- `rationale` is required.
- `notes` is optional and may be omitted or empty.
- Any malformed or mismatched record is ignored for that candidate only.

## Behavior

- Discovery still generates up to 20 candidates.
- The orchestrator still computes deterministic DES predictions and uncertainty.
- The LLM only reviews the top N candidates after deterministic filtering.
- Each candidate is reviewed in a separate LLM request.
- `keep` leaves the ranking unchanged.
- `deprioritize` applies a ranking penalty.
- `reject` removes the candidate from the final ranked list by default.
- The report must show the LLM decision, confidence, rationale, and notes when available.
- If the provider fails, the run continues with deterministic results and a warning.
- If the LLM returns the wrong SMILES for a review, the review is dropped and a warning is recorded.

## File Structure

- `des_multi_agent/llm/schemas.py`
  - Add a per-candidate review dataclass.
- `des_multi_agent/llm/prompts.py`
  - Add a single-candidate review prompt.
- `des_multi_agent/llm/parser.py`
  - Add a strict parser for one review object.
- `des_multi_agent/llm/provider.py`
  - Add a provider method for one-candidate review.
- `des_multi_agent/llm/base.py`
  - Implement the shared request flow for per-candidate review.
- `des_multi_agent/orchestrator.py`
  - Call the new review method for the top N candidates and merge results.
- `des_multi_agent/reporting.py`
  - Render one-candidate review decisions in the final report.
- `tests/test_llm_parser.py`
  - Add parsing coverage for single-review JSON.
- `tests/test_llm_orchestrator.py`
  - Add review-merging and ranking behavior tests.
- `tests/test_demo_des_search.py`
  - Add demo coverage for the one-by-one review path.

## Error Handling

- Malformed JSON skips only that candidate’s review.
- Wrong-SMILES reviews are rejected and logged.
- Provider failures become warnings, not hard failures.
- Rejected candidates are removed from the final ranked list unless review retention is explicitly enabled in the future.
- Deprioritized candidates remain visible but rank below kept candidates.

## Testing

- Unit test the new single-candidate review parser.
- Unit test the new single-candidate prompt shape.
- Unit test the provider method that reviews one candidate at a time.
- Unit test orchestrator behavior for `keep`, `reject`, and `deprioritize`.
- Unit test that a malformed review does not break the run.
- Unit test that a wrong-SMILES review is ignored.
- Regression test a small demo run to confirm the report still renders cleanly.
