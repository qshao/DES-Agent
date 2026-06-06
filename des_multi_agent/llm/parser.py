from __future__ import annotations

import json
import re
from typing import Any

from .schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if not lines:
        return text
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    text = "\n".join(lines).strip()
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    return text


def _extract_json_block(raw: str) -> str:
    text = _strip_code_fences(raw)
    if text.startswith("[") or text.startswith("{"):
        return text
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if match:
        return match.group(1)
    return text


def _coerce_json(raw: str) -> Any:
    data = json.loads(_extract_json_block(raw))
    if isinstance(data, dict):
        for key in ("candidates", "explanations", "critique", "notes", "items"):
            if key in data:
                return data[key]
    return data


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def parse_candidate_brainstorms(raw: str) -> list[CandidateBrainstorm]:
    data = _coerce_json(raw)
    if not isinstance(data, list):
        return []
    out: list[CandidateBrainstorm] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        smiles = str(item.get("smiles", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        family = str(item.get("family", "")).strip()
        if not smiles or not rationale or not family:
            continue
        out.append(CandidateBrainstorm(smiles=smiles, rationale=rationale, family=family))
    return out


def parse_explanation_notes(raw: str) -> list[ExplanationNote]:
    data = _coerce_json(raw)
    if not isinstance(data, list):
        return []
    out: list[ExplanationNote] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        smiles = str(item.get("smiles", "")).strip()
        summary = str(item.get("summary", "")).strip()
        evidence = _normalize_list(item.get("evidence"))
        if not smiles or not summary:
            continue
        out.append(ExplanationNote(smiles=smiles, summary=summary, evidence=evidence))
    return out


def parse_critique_notes(raw: str) -> list[CritiqueNote]:
    data = _coerce_json(raw)
    if not isinstance(data, list):
        return []
    out: list[CritiqueNote] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        smiles = str(item.get("smiles", "")).strip()
        assessment = str(item.get("assessment", "")).strip()
        concerns = _normalize_list(item.get("concerns"))
        if not smiles or not assessment:
            continue
        out.append(CritiqueNote(smiles=smiles, assessment=assessment, concerns=concerns))
    return out
