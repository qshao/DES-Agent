from __future__ import annotations

from collections.abc import Sequence

from ..evaluation import DesResult


def _results_summary(results: Sequence[DesResult]) -> str:
    lines = []
    for result in results:
        lines.append(
            f"- {result.curve.smiles_b}: is_des={result.is_des}, min_tm_k={result.min_tm_k:.2f}, rationale={result.rationale}"
        )
    return "\n".join(lines) if lines else "- no ranked results yet"


def candidate_brainstorm_prompt(component_a: str, constraints: dict | None, context: str, max_items: int | None = None) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of candidate partner molecules for DES screening.\n",
        f"Component A: {component_a}\n",
        f"Constraints: {constraints or {}}\n",
        f"Context: {context}\n",
    ]
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append("Each item must contain smiles, rationale, and family.")
    return "".join(parts)


def explanation_prompt(results: Sequence[DesResult], context: str, max_items: int | None = None) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of explanation notes for ranked DES candidates.\n",
        f"Context: {context}\n",
        "Results:\n",
        f"{_results_summary(results)}\n",
    ]
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append("Each item must contain smiles, summary, and evidence.")
    return "".join(parts)


def critique_prompt(results: Sequence[DesResult], context: str, max_items: int | None = None) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of advisory critique notes for ranked DES candidates.\n",
        f"Context: {context}\n",
        "Results:\n",
        f"{_results_summary(results)}\n",
    ]
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append("Each item must contain smiles, assessment, and concerns.")
    return "".join(parts)
