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


def candidate_review_prompt(component_a: str, candidate_smiles: str, context: str) -> str:
    return (
        "Return raw JSON only. Do not use markdown fences or commentary.\n"
        "Return one JSON object for a single candidate review.\n"
        f"Component A: {component_a}\n"
        f"Candidate: {candidate_smiles}\n"
        f"Context: {context}\n"
        "The JSON object must contain smiles, decision, confidence, rationale, and notes.\n"
        "decision must be one of keep, reject, or deprioritize.\n"
        'Example: { "smiles": "OCCO", "decision": "keep", "confidence": 0.87, "rationale": "Good candidate.", "notes": ["short note"] }'
    )


def candidate_brainstorm_prompt(
    component_a: str,
    constraints: dict | None,
    context: str,
    max_items: int | None = None,
    families: list | None = None,
) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of candidate partner molecules for DES screening.\n",
        f"Component A: {component_a}\n",
        f"Constraints: {constraints or {}}\n",
        f"Context: {context}\n",
    ]
    if families:
        parts.append("Distribute candidates across these chemical families:\n")
        for f in families:
            parts.append(f"  - {f.name}: {f.rationale} (role: {f.hbd_hba_role})\n")
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append("Each item must contain smiles, rationale, and family.")
    return "".join(parts)


def family_selection_prompt(
    component_a: str,
    constraints: dict | None,
    context: str,
    max_families: int = 6,
) -> str:
    return "".join([
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of chemical families to explore as DES partner candidates.\n",
        f"Component A: {component_a}\n",
        f"Constraints: {constraints or {}}\n",
        f"Context: {context}\n",
        f"Return at most {max_families} families.\n",
        'Each item must contain name, rationale, and hbd_hba_role ("HBD", "HBA", or "both").',
    ])


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


def contradiction_prompt(results: Sequence[DesResult], context: str, max_items: int | None = None) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array examining whether each ML DES prediction is chemically plausible.\n",
        f"Context: {context}\n",
        "Results:\n",
        f"{_results_summary(results)}\n",
    ]
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append(
        'Each item must contain smiles, agreement ("agree", "conflict", or "uncertain"), and explanation.'
    )
    return "".join(parts)
