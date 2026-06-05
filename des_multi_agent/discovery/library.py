from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..chemistry_filter import canonicalize_smiles


@dataclass(frozen=True)
class LiteratureRecord:
    component_a: str
    component_b: str
    source: str
    note: str = ""
    reference_id: str = ""


@dataclass(frozen=True)
class LibraryRecord:
    smiles: str
    family: str
    source: str
    note: str = ""


@dataclass(frozen=True)
class DiscoveryLibrary:
    literature: tuple[LiteratureRecord, ...] = ()
    candidate_library: tuple[LibraryRecord, ...] = ()


def _load_yaml_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    if not isinstance(raw, list):
        raise ValueError(f"Discovery file {path} must contain a list of records")
    return raw


def load_discovery_library(path: str | Path) -> DiscoveryLibrary:
    base = Path(path)
    literature_path = base / "literature.yaml"
    library_path = base / "library.yaml"
    literature: list[LiteratureRecord] = []
    candidate_library: list[LibraryRecord] = []

    if literature_path.exists():
        for row in _load_yaml_records(literature_path):
            missing = {key for key in ("component_a", "component_b", "source") if key not in row}
            if missing:
                raise ValueError(f"{literature_path} is missing required keys: {sorted(missing)}")
            literature.append(
                LiteratureRecord(
                    component_a=str(row["component_a"]),
                    component_b=str(row["component_b"]),
                    source=str(row["source"]),
                    note=str(row.get("note", "")),
                    reference_id=str(row.get("reference_id", "")),
                )
            )

    if library_path.exists():
        for row in _load_yaml_records(library_path):
            missing = {key for key in ("smiles", "family", "source") if key not in row}
            if missing:
                raise ValueError(f"{library_path} is missing required keys: {sorted(missing)}")
            candidate_library.append(
                LibraryRecord(
                    smiles=canonicalize_smiles(str(row["smiles"])),
                    family=str(row["family"]),
                    source=str(row["source"]),
                    note=str(row.get("note", "")),
                )
            )

    return DiscoveryLibrary(literature=tuple(literature), candidate_library=tuple(candidate_library))
