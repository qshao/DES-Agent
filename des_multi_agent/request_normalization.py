from __future__ import annotations

from dataclasses import dataclass, field
import re

SALT_MARKERS = (
    "hydrochloride",
    "hcl",
    "hydrobromide",
    "sulfate",
    "sulphate",
    "acetate",
    "maleate",
    "mesylate",
    "tosylate",
    "tartrate",
    "citrate",
    "nitrate",
    "phosphate",
)

METAL_ION_PATTERN = re.compile(r"\b([A-Z][a-z]?(?:\d\+|\d-?|\+|\-))\b")
LIGAND_PATTERN = re.compile(r"\b(?:with|ligand|ligand\s+smiles|using)\s+([A-Za-z0-9@+\-#=\[\]\(\)\\/\.]+)\b", re.IGNORECASE)
COMPOUND_AFTER_FOR_PATTERN = re.compile(r"\b(?:for|about|on)\s+([A-Za-z0-9][A-Za-z0-9\s\-]+)$", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizedRequest:
    normalized_text: str
    workflow_hint: str | None = None
    compound_hint: str | None = None
    metal_ion_hint: str | None = None
    ligand_hint: str | None = None
    needs_clarification: bool = False
    clarifying_questions: list[str] = field(default_factory=list)


def _has_salt_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in SALT_MARKERS)


def _extract_compound_hint(text: str) -> str | None:
    lowered = text.lower().strip()
    if "lidocaine" in lowered:
        return "lidocaine"
    match = COMPOUND_AFTER_FOR_PATTERN.search(text)
    if match is not None:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    return None


def _extract_metal_ion_hint(text: str) -> str | None:
    lowered = text.lower()
    if "cu2+" in lowered:
        return "Cu2+"
    if "zn2+" in lowered:
        return "Zn2+"
    if "fe3+" in lowered:
        return "Fe3+"
    if "ni2+" in lowered:
        return "Ni2+"
    match = METAL_ION_PATTERN.search(text)
    if match is not None:
        return match.group(1).strip()
    return None


def _extract_ligand_hint(text: str) -> str | None:
    match = LIGAND_PATTERN.search(text)
    if match is not None:
        value = match.group(1).strip()
        if value:
            return value
    if "nccn" in text.lower():
        return "NCCN"
    return None


def normalize_request_text(text: str) -> NormalizedRequest:
    normalized_text = " ".join(text.strip().split())
    lowered = normalized_text.lower()

    workflow_hint: str | None = None
    if any(token in lowered for token in ("metal binding", "metal extraction", "stability constant", "log k", "ligand selection")):
        workflow_hint = "metal-binding"
    elif "des" in lowered or "deep eutectic" in lowered:
        workflow_hint = "des"

    compound_hint = _extract_compound_hint(normalized_text) if workflow_hint != "metal-binding" else None
    metal_ion_hint = _extract_metal_ion_hint(normalized_text) if workflow_hint == "metal-binding" else None
    ligand_hint = _extract_ligand_hint(normalized_text) if workflow_hint == "metal-binding" else None

    needs_clarification = False
    questions: list[str] = []

    if _has_salt_marker(normalized_text) and compound_hint == "lidocaine":
        needs_clarification = True
        questions.append("Do you mean lidocaine free base or a lidocaine salt form?")

    if workflow_hint == "metal-binding" and metal_ion_hint is None:
        needs_clarification = True
        questions.append("Which metal ion should I use for the metal-binding workflow?")

    if workflow_hint == "metal-binding" and ligand_hint is None:
        needs_clarification = True
        questions.append("Which ligand SMILES should I use for the metal-binding workflow?")

    if workflow_hint is None and compound_hint is None and metal_ion_hint is None and ligand_hint is None:
        needs_clarification = True
        questions.append("Which compound or metal-binding target should I use?")

    return NormalizedRequest(
        normalized_text=normalized_text,
        workflow_hint=workflow_hint,
        compound_hint=compound_hint,
        metal_ion_hint=metal_ion_hint,
        ligand_hint=ligand_hint,
        needs_clarification=needs_clarification,
        clarifying_questions=questions,
    )
