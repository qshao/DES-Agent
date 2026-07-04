# Concurrent LLM Review Calls — Design Spec

## Goal

Parallelize DES-Agent's four sequential per-candidate/per-ligand LLM review
loops so a `vllm`-backed server can actually exploit continuous batching.
Today every LLM call in these loops is a blocking synchronous `urllib` HTTP
request inside a plain `for` loop — confirmed via a live benchmark where the
vLLM server logs showed `Running: 1 reqs` for the entire run. Ollama and the
hosted API backends (`openai`, `gemini`, `custom_http`) lose nothing from
this today (they have no batching advantage to exploit), but they must keep
working identically once calls become concurrent.

## Context

Four call sites share an identical shape — loop over N items, call one LLM
provider method per item, collect the result, turn any exception into a
warning and move on:

1. `des_multi_agent/orchestrator.py:334-354` (`_review_top_candidates`) —
   loops `candidate_proposals[:top_n]`, calls
   `provider.review_candidate(component_a, proposal.smiles, context)`,
   collects into `review_by_smiles: dict[str, CandidateReview]` and
   `review_notes: list[CandidateReview]`. `top_n` is typically 8-20.
2. `des_multi_agent/orchestrator.py:1074-1078` (chemistry advisor loop) —
   loops `annotated_results[:min(5, len(annotated_results))]`, calls
   `provider.assess_candidate_chemistry(smiles, advisor_context, advisor_memory_notes)`,
   extends a flat `advisor_assessments: list[ChemistryAssessment]`.
3. `des_multi_agent/workflows/metal_binding_screen.py:265-270` — loops
   `cycle_results` (default `n=20`), calls
   `llm_provider.review_ligand(metal_ion, r.ligand_smiles, context)`,
   appends to a flat `all_reviews` list.
4. `des_multi_agent/workflows/metal_binding_selectivity.py:441-446` —
   identical shape, calls
   `llm_provider.review_ligand(target_metal, r.ligand_smiles, context)`.

`des_multi_agent/workflows/selectivity_des_pipeline.py:132` also loops over
ligands, but each iteration runs a whole nested multi-cycle search pipeline,
not a single LLM call — out of scope here; parallelizing that would be a
much larger, riskier change.

**Transport is fully synchronous today**: `RequestTransport.post_json`
(`des_multi_agent/llm/transport.py:20`) wraps a retrying closure around
`request_fn` (default `post_json_chat`), routed through
`LLMCache.get_or_call` (`des_multi_agent/llm/cache.py:35`).
`post_json_chat` (`des_multi_agent/llm/client.py:7`) is plain blocking
`urllib.request.urlopen`. No async/threaded transport exists anywhere in the
codebase; `httpx>=0.24` is declared in `pyproject.toml` but is not imported
anywhere in `des_multi_agent` — a dead dependency, left untouched by this
change.

**Order independence confirmed**: downstream, `_apply_candidate_reviews`
consumes `review_by_smiles` (a dict keyed by SMILES) and final result
ordering happens afterward via `rank_results`/`rank_results_composite`
based on score. The advisor and metal-workflow sites collect into flat
lists that are never order-sensitive either. Completion order of these
per-item LLM calls has zero effect on final output.

**Cache thread-safety**: `LLMCache` is disk-based, one file per
`SHA-256(url+payload)` key, with a non-atomic `path.write_text` write and no
shared in-process state (a fresh `LLMCache` instance is constructed on
every `post_json` call). Different items produce different payloads →
different keys, so ordinary concurrent calls across distinct
candidates/ligands do not collide on the same cache file. A genuine
same-key collision would require two *identical* duplicate items reviewed
concurrently, which upstream deduplication already prevents. This is an
accepted, documented risk — not something this change fixes.

## Decisions (confirmed with user)

- **Concurrency mechanism: `ThreadPoolExecutor`**, not an asyncio/httpx
  rewrite. The existing transport stays exactly as-is (blocking `urllib`);
  threads simply let multiple blocking calls be in flight at once. This is
  a legitimate pattern here because the work is I/O-bound — a blocking
  `urlopen` call releases the GIL while waiting on the network, so other
  threads make progress. No rewrite of `transport.py`/`cache.py`/provider
  classes.
- **Scope: all 4 call sites**, not just the one benchmarked. They share one
  shape, so one shared helper covers all of them with a small, consistent
  diff per site.
- **Concurrency limit: fixed default, `max_workers=8`.** No new config
  field on `LLMConfig`/YAML. 8 concurrent requests is enough to give a
  vLLM server's batcher something to batch, without risking overwhelming a
  rate-limited hosted API backend (`openai`/`gemini`).

## Design

### 1. Shared helper: `des_multi_agent/concurrency.py` (new file)

Placed at the top level (not inside `des_multi_agent/llm/`) because it's
needed by both `orchestrator.py` (top level) and `workflows/*.py`
(subpackage) — the helper itself has no LLM-specific logic, only its
callers do.

```python
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CallResult(Generic[T]):
    value: T | None
    error: Exception | None


def run_concurrent(items: list[Any], call: Callable[[Any], T], max_workers: int = 8) -> list[CallResult[T]]:
    """Run call(item) for every item concurrently; never raises.

    Results are returned in the same order as items (ThreadPoolExecutor.map
    preserves input order), so callers can zip(items, run_concurrent(...))
    to line results up with what produced them. Each call is wrapped so a
    single item's exception is captured as CallResult.error instead of
    aborting the others.
    """
    if not items:
        return []

    def _safe_call(item: Any) -> CallResult[T]:
        try:
            return CallResult(value=call(item), error=None)
        except Exception as exc:  # noqa: BLE001 - intentionally broad, mirrors existing per-item try/except
            return CallResult(value=None, error=exc)

    workers = min(len(items), max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_safe_call, items))
```

### 2. Call-site changes

Each site replaces its sequential `for item in items: try: ... except
Exception as exc: ...` with `for item, res in zip(items, run_concurrent(items,
lambda i: provider.method(...)))`, keeping every existing warning message,
collection variable, and post-processing check byte-identical.

**`orchestrator.py:334-354`** (`_review_top_candidates`):

```python
def _review_top_candidates(
    provider,
    component_a: str,
    candidate_proposals: list[CandidateProposal],
    context: str,
    top_n: int,
    llm_warnings: list[str],
) -> tuple[list[CandidateReview], dict[str, CandidateReview]]:
    review_notes: list[CandidateReview] = []
    review_by_smiles: dict[str, CandidateReview] = {}
    top_proposals = candidate_proposals[: max(0, top_n)]
    results = run_concurrent(top_proposals, lambda p: provider.review_candidate(component_a, p.smiles, context))
    for proposal, res in zip(top_proposals, results):
        if res.error is not None:
            llm_warnings.append(f"LLM candidate review failed for {proposal.smiles}: {res.error}")
            continue
        review = res.value
        if review.smiles != proposal.smiles:
            llm_warnings.append(f"LLM candidate review returned wrong SMILES for {proposal.smiles}")
            continue
        review_notes.append(review)
        review_by_smiles[proposal.smiles] = review
    return review_notes, review_by_smiles
```

**`orchestrator.py:1074-1078`** (chemistry advisor loop):

```python
advisor_items = annotated_results[: min(5, len(annotated_results))]
results = run_concurrent(
    advisor_items,
    lambda item: provider.assess_candidate_chemistry(item.result.curve.smiles_b, advisor_context, advisor_memory_notes),
)
for item, res in zip(advisor_items, results):
    if res.error is not None:
        llm_warnings.append(f"LLM chemistry assessment failed for {item.result.curve.smiles_b}: {res.error}")
        continue
    advisor_assessments.extend(res.value)
```

**`workflows/metal_binding_screen.py:265-270`** and
**`workflows/metal_binding_selectivity.py:441-446`** (identical shape,
`review_ligand` instead of `review_candidate`):

```python
results = run_concurrent(cycle_results, lambda r: llm_provider.review_ligand(metal_ion, r.ligand_smiles, context))
for r, res in zip(cycle_results, results):
    if res.error is not None:
        all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {res.error}")
        continue
    all_reviews.append(res.value)
```

(`metal_binding_selectivity.py` uses `target_metal` in place of `metal_ion`
in the `review_ligand` call — otherwise identical.)

### 3. Testing

- **`tests/test_concurrency.py`** (new): unit tests for `run_concurrent`
  itself — result order matches input order, empty list returns empty
  list, one item's exception is captured in its `CallResult.error` without
  affecting others, `max_workers` never exceeds `len(items)`.
- **One concurrency-proof test** (in the candidate-review site's existing
  test file): a fake provider whose `review_candidate` blocks on a
  `threading.Barrier` until N calls are simultaneously waiting, proving
  calls actually overlap in time rather than serializing — this is the
  test that would have caught today's bug (the benchmark showing `Running:
  1 reqs`).
- **Full existing suite must pass unchanged** for all 4 call sites —
  proves byte-identical external behavior (same warning strings, same
  collected results, same downstream consumption) with only the execution
  strategy changed from sequential to concurrent.

## Out of scope

- No changes to `transport.py`, `cache.py`, or any provider class.
- No new `LLMConfig`/YAML concurrency setting.
- `LLMCache`'s non-atomic write is not hardened — accepted risk, documented
  above.
- `selectivity_des_pipeline.py`'s per-ligand nested-search loop is not
  parallelized.
- The unused `httpx` dependency is not removed or put to use.
