"""Tests for offline molecule name → SMILES resolution."""
from __future__ import annotations

import json
import pathlib

import pytest
from rdkit import Chem

from des_multi_agent.chemistry.name_resolution import (
    list_molecules,
    resolve_name,
    resolve_to_smiles,
)

_DICT_PATH = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "molecule_names" / "common_names.json"


# ---------------------------------------------------------------------------
# Group 1: dictionary integrity
# ---------------------------------------------------------------------------

def test_dictionary_all_smiles_parse():
    data = json.loads(_DICT_PATH.read_text())
    for entry in data["entries"]:
        mol = Chem.MolFromSmiles(entry["smiles"])
        assert mol is not None, f"Invalid SMILES in dictionary: {entry['smiles']!r} ({entry['names'][0]})"


def test_dictionary_smiles_are_canonical():
    data = json.loads(_DICT_PATH.read_text())
    for entry in data["entries"]:
        mol = Chem.MolFromSmiles(entry["smiles"])
        assert mol is not None
        canon = Chem.MolToSmiles(mol)
        assert canon == entry["smiles"], (
            f"SMILES for {entry['names'][0]!r} is not canonical: "
            f"{entry['smiles']!r} should be {canon!r}"
        )


def test_dictionary_no_duplicate_names():
    data = json.loads(_DICT_PATH.read_text())
    seen: dict[str, str] = {}  # normalised_name → canonical_name
    for entry in data["entries"]:
        for raw in entry["names"]:
            normalised = " ".join(raw.strip().lower().split())
            assert normalised not in seen, (
                f"Duplicate name {raw!r} appears in both {seen[normalised]!r} and {entry['names'][0]!r}"
            )
            seen[normalised] = entry["names"][0]


def test_dictionary_every_entry_has_name_and_smiles():
    data = json.loads(_DICT_PATH.read_text())
    for entry in data["entries"]:
        assert entry.get("names"), f"Entry missing names: {entry}"
        assert entry.get("smiles"), f"Entry missing smiles: {entry}"


# ---------------------------------------------------------------------------
# Group 2: lookup correctness
# ---------------------------------------------------------------------------

def test_resolve_name_choline_chloride():
    result = resolve_name("choline chloride")
    assert result is not None
    mol = Chem.MolFromSmiles(result)
    assert mol is not None


def test_resolve_name_case_insensitive():
    lower = resolve_name("choline chloride")
    upper = resolve_name("CHOLINE CHLORIDE")
    mixed = resolve_name("Choline Chloride")
    assert lower == upper == mixed
    assert lower is not None


def test_resolve_name_synonym():
    by_full = resolve_name("choline chloride")
    by_abbrev = resolve_name("ChCl")
    assert by_full == by_abbrev
    assert by_full is not None


def test_resolve_name_whitespace_stripped():
    with_spaces = resolve_name("  urea  ")
    without = resolve_name("urea")
    assert with_spaces == without
    assert with_spaces is not None


def test_resolve_name_unknown_returns_none():
    assert resolve_name("not_a_real_molecule_xyz") is None


def test_resolve_to_smiles_known_name():
    smiles = resolve_to_smiles("urea")
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None


def test_resolve_to_smiles_returns_canonical():
    """Both name-lookup and SMILES-pass-through paths return canonical SMILES."""
    by_name = resolve_to_smiles("urea")
    by_smiles = resolve_to_smiles("NC(N)=O")
    assert by_name == by_smiles


# ---------------------------------------------------------------------------
# Group 3: SMILES pass-through
# ---------------------------------------------------------------------------

def test_resolve_to_smiles_valid_smiles_passthrough():
    smiles_in = "C[N+](C)(C)CCO.[Cl-]"
    result = resolve_to_smiles(smiles_in)
    mol = Chem.MolFromSmiles(result)
    assert mol is not None


def test_resolve_to_smiles_canonicalises_non_canonical():
    # "O=C(N)N" is a valid but non-canonical form of urea
    result = resolve_to_smiles("O=C(N)N")
    assert result == "NC(N)=O"


def test_resolve_to_smiles_canonical_smiles_unchanged():
    smiles = "NC(N)=O"
    assert resolve_to_smiles(smiles) == smiles


# ---------------------------------------------------------------------------
# Group 4: error paths
# ---------------------------------------------------------------------------

def test_resolve_to_smiles_empty_string_raises_value_error():
    with pytest.raises(ValueError):
        resolve_to_smiles("")


def test_resolve_to_smiles_whitespace_only_raises_value_error():
    with pytest.raises(ValueError):
        resolve_to_smiles("   ")


def test_resolve_to_smiles_unknown_raises_value_error():
    with pytest.raises(ValueError, match="not_a_molecule_xyz"):
        resolve_to_smiles("not_a_molecule_xyz")


def test_resolve_to_smiles_error_mentions_list_molecules():
    with pytest.raises(ValueError, match="list-molecules"):
        resolve_to_smiles("not_a_molecule_xyz")


def test_resolve_to_smiles_close_match_suggests_correction():
    # "choline chioride" is a typo of "choline chloride"
    with pytest.raises(ValueError, match="choline chloride"):
        resolve_to_smiles("choline chioride")


# ---------------------------------------------------------------------------
# Group 5: list_molecules
# ---------------------------------------------------------------------------

def test_list_molecules_returns_all_entries():
    data = json.loads(_DICT_PATH.read_text())
    result = list_molecules()
    assert len(result) == len(data["entries"])


def test_list_molecules_entry_shape():
    result = list_molecules()
    for entry in result:
        assert "smiles" in entry
        assert "canonical_name" in entry
        assert "synonyms" in entry
        assert "role" in entry


def test_list_molecules_sorted_by_role_then_name():
    result = list_molecules()
    pairs = [(e["role"], e["canonical_name"]) for e in result]
    assert pairs == sorted(pairs)
