# Molecule Name Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to pass common molecule names (e.g. "choline chloride", "urea") anywhere the system expects a SMILES string, with no network dependency.

**Architecture:** A bundled JSON dictionary (`artifacts/molecule_names/common_names.json`) is loaded once at module import by `des_multi_agent/chemistry/name_resolution.py`. The two public functions — `resolve_to_smiles` and `resolve_name` — are called at every user-facing input boundary (CLI argument parser, FastAPI request handler, candidates-file loader). Pipeline internals are untouched and always receive canonical SMILES.

**Tech Stack:** Python stdlib (`difflib`, `json`, `re`), RDKit (`Chem.MolFromSmiles`, `Chem.MolToSmiles`). No new dependencies.

---

## File Map

| File | Action | Notes |
|------|--------|-------|
| `artifacts/molecule_names/common_names.json` | Create | ~80-entry bundled dictionary |
| `des_multi_agent/chemistry/name_resolution.py` | Create | Lookup + pass-through module |
| `tests/test_name_resolution.py` | Create | All resolution unit tests |
| `des_multi_agent/cli.py` | Modify | Replace early SMILES validation with `resolve_to_smiles`; add `list-molecules` subcommand |
| `des_multi_agent/orchestrator.py` | Modify | `_load_candidates_file`: resolve each line |
| `des_multi_agent/server.py` | Modify | Resolve `component_a`/`component_b` before calling orchestrator |
| `tests/test_cli.py` | Modify | Integration tests for name-as-input and unknown-name error |

---

## Task 1: Bundled Dictionary

**Files:**
- Create: `artifacts/molecule_names/common_names.json`

- [ ] **Step 1: Create the artifacts directory and JSON file**

```bash
mkdir -p /home/qshao/DES-Agent/artifacts/molecule_names
```

Then create `artifacts/molecule_names/common_names.json` with the following content (all SMILES are in canonical RDKit form):

```json
{
  "version": "1.0",
  "entries": [
    {
      "smiles": "C[N+](C)(C)CCO.[Cl-]",
      "names": ["choline chloride", "ChCl", "choline Cl", "choline hydrochloride"],
      "role": "HBA"
    },
    {
      "smiles": "C[N+](C)(C)CCO.[Br-]",
      "names": ["choline bromide", "choline hydrobromide"],
      "role": "HBA"
    },
    {
      "smiles": "C[N+](C)(C)CC(=O)[O-]",
      "names": ["betaine", "trimethylglycine", "glycine betaine", "TMG"],
      "role": "HBA"
    },
    {
      "smiles": "CC([O-])=O.C[N+](C)(C)CCO",
      "names": ["choline acetate"],
      "role": "HBA"
    },
    {
      "smiles": "[Br-].CCCC[N+](CCCC)(CCCC)CCCC",
      "names": ["tetrabutylammonium bromide", "TBAB", "Bu4NBr"],
      "role": "HBA"
    },
    {
      "smiles": "[Cl-].CCCC[N+](CCCC)(CCCC)CCCC",
      "names": ["tetrabutylammonium chloride", "TBAC", "Bu4NCl"],
      "role": "HBA"
    },
    {
      "smiles": "[Br-].CCCC[P+](CCCC)(CCCC)CCCC",
      "names": ["tetrabutylphosphonium bromide", "TBPB", "Bu4PBr"],
      "role": "HBA"
    },
    {
      "smiles": "NC(N)=O",
      "names": ["urea", "carbamide"],
      "role": "HBD"
    },
    {
      "smiles": "NC(=S)N",
      "names": ["thiourea", "thiocarbamide"],
      "role": "HBD"
    },
    {
      "smiles": "CNC(N)=O",
      "names": ["N-methylurea", "methylurea", "1-methylurea"],
      "role": "HBD"
    },
    {
      "smiles": "CC(N)=O",
      "names": ["acetamide", "acetic acid amide", "ethanamide"],
      "role": "HBD"
    },
    {
      "smiles": "O=C1CCCCCN1",
      "names": ["caprolactam", "epsilon-caprolactam", "2-azepanone", "6-hexanelactam"],
      "role": "HBD"
    },
    {
      "smiles": "OCC(O)CO",
      "names": ["glycerol", "glycerin", "glycerine", "1,2,3-propanetriol", "propane-1,2,3-triol"],
      "role": "HBD"
    },
    {
      "smiles": "OCCO",
      "names": ["ethylene glycol", "monoethylene glycol", "MEG", "1,2-ethanediol", "ethan-1,2-diol"],
      "role": "HBD"
    },
    {
      "smiles": "CC(O)CO",
      "names": ["propylene glycol", "1,2-propanediol", "propane-1,2-diol", "PG"],
      "role": "HBD"
    },
    {
      "smiles": "OCCCCO",
      "names": ["1,4-butanediol", "butane-1,4-diol", "BDO", "tetramethylene glycol"],
      "role": "HBD"
    },
    {
      "smiles": "CC(O)CCO",
      "names": ["1,3-butanediol", "butane-1,3-diol", "1,3-BD"],
      "role": "HBD"
    },
    {
      "smiles": "OCCOCCO",
      "names": ["diethylene glycol", "DEG", "2,2'-oxydiethanol", "digol"],
      "role": "HBD"
    },
    {
      "smiles": "OCCOCCOCCO",
      "names": ["triethylene glycol", "TEG", "triglycol", "2,2'-(ethane-1,2-diylbis(oxy))diethanol"],
      "role": "HBD"
    },
    {
      "smiles": "OCC(O)C(O)C(O)C(O)CO",
      "names": ["sorbitol", "D-sorbitol", "glucitol", "D-glucitol", "sorbol"],
      "role": "HBD"
    },
    {
      "smiles": "OC(=O)C(=O)O",
      "names": ["oxalic acid", "ethanedioic acid"],
      "role": "HBD"
    },
    {
      "smiles": "OC(=O)CC(=O)O",
      "names": ["malonic acid", "propanedioic acid", "methanedicarboxylic acid"],
      "role": "HBD"
    },
    {
      "smiles": "OC(=O)CC(O)(CC(=O)O)C(=O)O",
      "names": ["citric acid", "2-hydroxypropane-1,2,3-tricarboxylic acid"],
      "role": "HBD"
    },
    {
      "smiles": "OC(=O)C(O)CC(=O)O",
      "names": ["malic acid", "2-hydroxybutanedioic acid", "hydroxybutanedioic acid"],
      "role": "HBD"
    },
    {
      "smiles": "OC(=O)C(O)C(O)C(=O)O",
      "names": ["tartaric acid", "2,3-dihydroxybutanedioic acid", "dihydroxysuccinic acid"],
      "role": "HBD"
    },
    {
      "smiles": "OC(=O)CCC(=O)O",
      "names": ["succinic acid", "butanedioic acid", "amber acid"],
      "role": "HBD"
    },
    {
      "smiles": "OC(=O)CCCC(=O)O",
      "names": ["glutaric acid", "pentanedioic acid"],
      "role": "HBD"
    },
    {
      "smiles": "OC(=O)CCCCC(=O)O",
      "names": ["adipic acid", "hexanedioic acid"],
      "role": "HBD"
    },
    {
      "smiles": "CC(O)C(=O)O",
      "names": ["lactic acid", "2-hydroxypropanoic acid", "milk acid"],
      "role": "HBD"
    },
    {
      "smiles": "CC(=O)CCC(=O)O",
      "names": ["levulinic acid", "4-oxopentanoic acid", "3-acetylpropionic acid"],
      "role": "HBD"
    },
    {
      "smiles": "CC(=O)O",
      "names": ["acetic acid", "ethanoic acid", "glacial acetic acid"],
      "role": "HBD"
    },
    {
      "smiles": "OC=O",
      "names": ["formic acid", "methanoic acid"],
      "role": "HBD"
    },
    {
      "smiles": "CCC(=O)O",
      "names": ["propionic acid", "propanoic acid"],
      "role": "HBD"
    },
    {
      "smiles": "CCCCCCCC(=O)O",
      "names": ["caprylic acid", "octanoic acid", "n-octanoic acid"],
      "role": "HBD"
    },
    {
      "smiles": "CCCCCCCCCC(=O)O",
      "names": ["capric acid", "decanoic acid", "n-decanoic acid"],
      "role": "HBD"
    },
    {
      "smiles": "Oc1ccccc1",
      "names": ["phenol", "hydroxybenzene", "carbolic acid"],
      "role": "HBD"
    },
    {
      "smiles": "CC1CCC(C(C)C)CC1O",
      "names": ["menthol", "l-menthol", "peppermint camphor", "(1R,2S,5R)-menthol"],
      "role": "HBD"
    },
    {
      "smiles": "Cc1ccc(O)c(C(C)C)c1",
      "names": ["thymol", "2-isopropyl-5-methylphenol", "thymic acid"],
      "role": "HBD"
    },
    {
      "smiles": "Cc1ccc(C(C)C)cc1O",
      "names": ["carvacrol", "5-isopropyl-2-methylphenol", "2-methyl-5-isopropylphenol"],
      "role": "HBD"
    },
    {
      "smiles": "OCC1OC(O)(CO)C(O)C1O",
      "names": ["fructose", "D-fructose", "levulose", "fruit sugar"],
      "role": "HBD"
    },
    {
      "smiles": "OCC1OC(O)C(O)C(O)C1O",
      "names": ["glucose", "D-glucose", "dextrose", "grape sugar"],
      "role": "HBD"
    },
    {
      "smiles": "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
      "names": ["sucrose", "table sugar", "saccharose", "cane sugar", "beet sugar"],
      "role": "HBD"
    },
    {
      "smiles": "NCC(=O)O",
      "names": ["glycine", "aminoacetic acid", "2-aminoacetic acid"],
      "role": "amphoteric"
    },
    {
      "smiles": "CC(N)C(=O)O",
      "names": ["alanine", "L-alanine", "2-aminopropanoic acid"],
      "role": "amphoteric"
    },
    {
      "smiles": "OC(=O)C1CCCN1",
      "names": ["proline", "L-proline", "pyrrolidine-2-carboxylic acid"],
      "role": "amphoteric"
    },
    {
      "smiles": "NC(CO)C(=O)O",
      "names": ["serine", "L-serine", "2-amino-3-hydroxypropanoic acid"],
      "role": "amphoteric"
    },
    {
      "smiles": "NC(CS)C(=O)O",
      "names": ["cysteine", "L-cysteine", "2-amino-3-mercaptopropanoic acid"],
      "role": "amphoteric"
    },
    {
      "smiles": "NC(Cc1cnc[nH]1)C(=O)O",
      "names": ["histidine", "L-histidine", "2-amino-3-(1H-imidazol-4-yl)propanoic acid"],
      "role": "amphoteric"
    },
    {
      "smiles": "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
      "names": ["EDTA", "edetic acid", "ethylenediaminetetraacetic acid"],
      "role": "ligand"
    },
    {
      "smiles": "OC(=O)CN(CC(=O)O)CC(=O)O",
      "names": ["NTA", "nitrilotriacetic acid", "2,2',2''-nitrilotriacetic acid"],
      "role": "ligand"
    },
    {
      "smiles": "OC(=O)CN(CCN(CC(=O)O)CCN(CC(=O)O)CC(=O)O)CC(=O)O",
      "names": ["DTPA", "diethylenetriaminepentaacetic acid", "pentetic acid"],
      "role": "ligand"
    },
    {
      "smiles": "Oc1ccc2ncccc2c1",
      "names": ["8-hydroxyquinoline", "oxine", "8-quinolinol"],
      "role": "ligand"
    },
    {
      "smiles": "O",
      "names": ["water", "H2O"],
      "role": "solvent"
    },
    {
      "smiles": "CCO",
      "names": ["ethanol", "ethyl alcohol", "EtOH"],
      "role": "solvent"
    },
    {
      "smiles": "CO",
      "names": ["methanol", "methyl alcohol", "MeOH", "wood alcohol"],
      "role": "solvent"
    },
    {
      "smiles": "CS(C)=O",
      "names": ["DMSO", "dimethyl sulfoxide", "dimethylsulfoxide"],
      "role": "solvent"
    },
    {
      "smiles": "CC#N",
      "names": ["acetonitrile", "MeCN", "methyl cyanide"],
      "role": "solvent"
    }
  ]
}
```

- [ ] **Step 2: Verify the JSON parses**

```bash
python -c "
import json, pathlib
data = json.loads(pathlib.Path('artifacts/molecule_names/common_names.json').read_text())
print(f'version: {data[\"version\"]}')
print(f'entries: {len(data[\"entries\"])}')
"
```

Expected output:
```
version: 1.0
entries: 56
```

- [ ] **Step 3: Verify all SMILES are valid and already canonical**

```bash
python -c "
import json, pathlib
from rdkit import Chem
data = json.loads(pathlib.Path('artifacts/molecule_names/common_names.json').read_text())
bad = []
for e in data['entries']:
    s = e['smiles']
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        bad.append(f'INVALID: {s!r} ({e[\"names\"][0]})')
    else:
        canon = Chem.MolToSmiles(mol)
        if canon != s:
            bad.append(f'NOT CANONICAL: {s!r} -> {canon!r} ({e[\"names\"][0]})')
if bad:
    for b in bad: print(b)
else:
    print('All SMILES valid and canonical.')
"
```

If any SMILES are flagged as not canonical, replace them with the printed canonical form in the JSON file and re-run until output is `All SMILES valid and canonical.`

- [ ] **Step 4: Commit**

```bash
git add artifacts/molecule_names/common_names.json
git commit -m "feat: add molecule name dictionary (56 DES-relevant entries)"
```

---

## Task 2: Name Resolution Module (TDD)

**Files:**
- Create: `tests/test_name_resolution.py`
- Create: `des_multi_agent/chemistry/name_resolution.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_name_resolution.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
python -m pytest tests/test_name_resolution.py -v 2>&1 | head -30
```

Expected: all tests fail with `ModuleNotFoundError` or `ImportError` — the module does not exist yet.

- [ ] **Step 3: Create the resolution module**

Create `des_multi_agent/chemistry/name_resolution.py`:

```python
"""Offline molecule name → SMILES resolution.

Looks up common molecule names and synonyms from a bundled dictionary.
Also accepts and canonicalises valid SMILES strings directly.
No network access required.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from rdkit import Chem

_DICT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "molecule_names" / "common_names.json"


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _build_index() -> tuple[dict[str, str], list[dict]]:
    data = json.loads(_DICT_PATH.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    entries = data["entries"]
    for entry in entries:
        for name in entry["names"]:
            index[_normalise(name)] = entry["smiles"]
    return index, entries


_NAME_INDEX, _ENTRIES = _build_index()


def resolve_name(text: str) -> str | None:
    """Return canonical SMILES if *text* is a known name/synonym, else None.

    Does not check whether *text* is a valid SMILES — call resolve_to_smiles
    for the combined pass-through-or-lookup behaviour.
    """
    return _NAME_INDEX.get(_normalise(text))


def resolve_to_smiles(text: str) -> str:
    """Return canonical SMILES for *text*, which may be a SMILES or a name.

    Resolution order:
      1. If *text* is a valid SMILES, canonicalise and return it.
      2. If *text* matches a known name or synonym, return the dictionary SMILES.
      3. Raise ValueError with a user-friendly message including a 'did you mean'
         suggestion when the input is close to a known name.
    """
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return Chem.MolToSmiles(mol)

    key = _normalise(text)
    if key in _NAME_INDEX:
        return _NAME_INDEX[key]

    suggestion = ""
    close = difflib.get_close_matches(key, _NAME_INDEX.keys(), n=1, cutoff=0.75)
    if close:
        matched_smiles = _NAME_INDEX[close[0]]
        suggestion = f"\n  → Did you mean {close[0]!r}?  (SMILES: {matched_smiles})"

    raise ValueError(
        f"Unknown molecule: {text!r}"
        f"{suggestion}"
        f"\n  → Run 'des-agent list-molecules' to see all supported names."
        f"\n  → If you have a SMILES string, pass that directly instead."
    )


def list_molecules() -> list[dict]:
    """Return all dictionary entries sorted by role then canonical name.

    Each dict has keys: smiles, canonical_name, synonyms, role.
    """
    result = []
    for entry in _ENTRIES:
        result.append({
            "smiles": entry["smiles"],
            "canonical_name": entry["names"][0],
            "synonyms": entry["names"][1:],
            "role": entry["role"],
        })
    return sorted(result, key=lambda e: (e["role"], e["canonical_name"]))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_name_resolution.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemistry/name_resolution.py tests/test_name_resolution.py
git commit -m "feat: add offline molecule name resolution module"
```

---

## Task 3: CLI `--component-a` / `--component-b` / `--ligand-smiles` Resolution (TDD)

**Files:**
- Modify: `des_multi_agent/cli.py` (around lines 539–547)
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write a failing CLI integration test**

Find the existing `tests/test_cli.py` and append these tests:

```python
# --- name resolution integration ---

def test_cli_accepts_molecule_name_for_component_a(monkeypatch, tmp_path):
    """--component-a 'urea' should resolve to NC(N)=O before the pipeline runs."""
    received = {}

    def fake_run_search_report(component_a, **kwargs):
        received["component_a"] = component_a
        from des_multi_agent.orchestrator import SearchOutcome
        from des_multi_agent.reporting import format_report
        return SearchOutcome(
            results=[], annotated_results=[], candidate_proposals=[],
            candidate_reviews=[], brainstorm_candidates=[], explanation_notes=[],
            critique_notes=[], llm_warnings=[], contradiction_notes=[],
            viscosity_predictions=[], chemical_pattern_memory=None,
            chemistry_lesson_summary=None,
        )

    monkeypatch.setattr("des_multi_agent.cli.run_search_report", fake_run_search_report)
    monkeypatch.setattr("des_multi_agent.cli.run_multi_cycle_search", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called")))

    import des_multi_agent.cli as cli_module
    # Patch checkpoint discovery so the test does not need a real file
    monkeypatch.setattr(cli_module, "_discover_checkpoint", lambda: "fake.pt")
    monkeypatch.setattr(cli_module, "resolve_existing_path", lambda p, **kw: p)
    monkeypatch.setattr(cli_module, "check_checkpoint_config_compat", lambda *a, **kw: [])

    import sys
    argv = ["des-agent", "--component-a", "urea", "--checkpoint-path", "fake.pt"]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        cli_module.main()
    except SystemExit:
        pass

    assert received.get("component_a") == "NC(N)=O"


def test_cli_unknown_molecule_name_exits_with_error(monkeypatch, capsys):
    import sys
    import des_multi_agent.cli as cli_module
    monkeypatch.setattr(cli_module, "_discover_checkpoint", lambda: "fake.pt")
    monkeypatch.setattr(cli_module, "resolve_existing_path", lambda p, **kw: p)

    argv = ["des-agent", "--component-a", "not_a_real_molecule_xyz", "--checkpoint-path", "fake.pt"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "not_a_real_molecule_xyz" in (captured.out + captured.err)
```

- [ ] **Step 2: Run to verify the tests fail**

```bash
python -m pytest tests/test_cli.py::test_cli_accepts_molecule_name_for_component_a tests/test_cli.py::test_cli_unknown_molecule_name_exits_with_error -v
```

Expected: both FAIL (the CLI currently only validates SMILES, does not resolve names).

- [ ] **Step 3: Replace the early SMILES validation block in `cli.py`**

In `des_multi_agent/cli.py`, find the block starting at approximately line 538:

```python
        # Validate SMILES early
        try:
            from rdkit import Chem as _Chem
            if _Chem.MolFromSmiles(args.component_a) is None:
                parser.error(
                    f"--component-a {args.component_a!r} is not a valid SMILES string. "
                    "Example: 'CCO' for ethanol, 'c1ccccc1' for benzene."
                )
        except ImportError:
            pass  # rdkit not available; skip early validation
```

Replace it with:

```python
        # Resolve molecule names → canonical SMILES at the input boundary.
        try:
            from .chemistry.name_resolution import resolve_to_smiles as _resolve
            args.component_a = _resolve(args.component_a)
            if getattr(args, "component_b", None):
                args.component_b = _resolve(args.component_b)
        except ImportError:
            pass  # rdkit not available; skip resolution
        except ValueError as exc:
            parser.error(str(exc))
```

Also resolve `--ligand-smiles` in the metal-binding workflow block. Find the block that checks `args.ligand_smiles` (around line 759) and add resolution immediately before the workflow call:

```python
        # Resolve ligand name → SMILES
        try:
            from .chemistry.name_resolution import resolve_to_smiles as _resolve
            args.ligand_smiles = _resolve(args.ligand_smiles)
        except (ImportError, ValueError):
            pass
```

Wait — for `--ligand-smiles`, a ValueError (unknown name) should be a hard error, not silently ignored. Use this instead:

```python
        # Resolve ligand name → SMILES at input boundary
        try:
            from .chemistry.name_resolution import resolve_to_smiles as _resolve
            args.ligand_smiles = _resolve(args.ligand_smiles)
        except ImportError:
            pass
        except ValueError as exc:
            parser.error(str(exc))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cli.py::test_cli_accepts_molecule_name_for_component_a tests/test_cli.py::test_cli_unknown_molecule_name_exits_with_error -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest tests/ -x -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: resolve molecule names at CLI input boundary"
```

---

## Task 4: `list-molecules` Subcommand (TDD)

**Files:**
- Modify: `des_multi_agent/cli.py` (subparsers block around line 296–337, and command dispatch around line 476)

- [ ] **Step 1: Write a failing test**

Add to `tests/test_cli.py`:

```python
def test_list_molecules_subcommand_prints_table(monkeypatch, capsys):
    import sys
    import des_multi_agent.cli as cli_module

    argv = ["des-agent", "list-molecules"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    # Table should contain known entries
    assert "choline chloride" in captured.out
    assert "urea" in captured.out
    assert "HBA" in captured.out
    assert "HBD" in captured.out
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_cli.py::test_list_molecules_subcommand_prints_table -v
```

Expected: FAIL — `list-molecules` subcommand does not exist.

- [ ] **Step 3: Add the subparser**

In `des_multi_agent/cli.py`, after the existing `subparsers.add_parser("supported-metals", ...)` block (around line 321), add:

```python
    subparsers.add_parser(
        "list-molecules",
        help="List all molecule names supported for --component-a / --component-b input",
    )
```

- [ ] **Step 4: Add the command handler**

In the `main()` dispatch block (where `args.command` is checked), add a handler before the workflow dispatch. Find the block that checks `if args.command == "task-router":` and add before it:

```python
    if args.command == "list-molecules":
        from .chemistry.name_resolution import list_molecules
        role_order = {"HBA": 0, "HBD": 1, "amphoteric": 2, "ligand": 3, "solvent": 4}
        entries = sorted(list_molecules(), key=lambda e: (role_order.get(e["role"], 9), e["canonical_name"]))
        current_role = None
        for e in entries:
            if e["role"] != current_role:
                current_role = e["role"]
                print(f"\n[{current_role}]")
            aliases = ", ".join(e["synonyms"]) if e["synonyms"] else ""
            print(f"  {e['canonical_name']:<35}  {e['smiles']}")
            if aliases:
                print(f"    aliases: {aliases}")
        raise SystemExit(0)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_cli.py::test_list_molecules_subcommand_prints_table -v
```

Expected: PASS.

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/ -x -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli.py
git commit -m "feat: add list-molecules subcommand"
```

---

## Task 5: Candidates-File Name Resolution (TDD)

**Files:**
- Modify: `des_multi_agent/orchestrator.py` (`_load_candidates_file`, lines ~414–425)
- Modify: `tests/test_orchestrator.py` (or `tests/test_discovery_orchestrator.py`)

- [ ] **Step 1: Write a failing test**

Add to `tests/test_orchestrator.py`:

```python
def test_candidates_file_accepts_molecule_names(tmp_path):
    """_load_candidates_file should resolve molecule names to SMILES."""
    from des_multi_agent.orchestrator import _load_candidates_file

    f = tmp_path / "candidates.txt"
    f.write_text("urea\ncholine chloride\n# this is a comment\nNC(N)=O\n")
    proposals = _load_candidates_file(str(f))
    smiles_list = [p.smiles for p in proposals]
    # urea → NC(N)=O; choline chloride → its SMILES; comment skipped; NC(N)=O passes through
    assert "NC(N)=O" in smiles_list
    # choline chloride should be resolved — check it's a valid SMILES
    from rdkit import Chem
    for s in smiles_list:
        assert Chem.MolFromSmiles(s) is not None, f"Invalid SMILES in output: {s!r}"


def test_candidates_file_skips_unknown_names_with_warning(tmp_path, capsys):
    """Unknown names in candidates file print a warning and are skipped."""
    from des_multi_agent.orchestrator import _load_candidates_file

    f = tmp_path / "candidates.txt"
    f.write_text("urea\nnot_a_real_molecule_xyz\n")
    proposals = _load_candidates_file(str(f))
    smiles_list = [p.smiles for p in proposals]
    assert len(smiles_list) == 1  # only urea
    captured = capsys.readouterr()
    assert "not_a_real_molecule_xyz" in captured.err
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_orchestrator.py::test_candidates_file_accepts_molecule_names tests/test_orchestrator.py::test_candidates_file_skips_unknown_names_with_warning -v
```

Expected: both FAIL.

- [ ] **Step 3: Modify `_load_candidates_file` in `orchestrator.py`**

Find `_load_candidates_file` (around line 414). Replace:

```python
def _load_candidates_file(path: str) -> list[CandidateProposal]:
    """Read one SMILES per line from a file; skip blanks and # comments."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    proposals: list[CandidateProposal] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        smiles = line.strip()
        if not smiles or smiles.startswith("#"):
            continue
        proposals.append(CandidateProposal(smiles=smiles, rationale="from file", family="unknown", source="file", source_id=""))
    return proposals
```

With:

```python
def _load_candidates_file(path: str) -> list[CandidateProposal]:
    """Read one SMILES or molecule name per line from a file; skip blanks and # comments."""
    from .chemistry.name_resolution import resolve_to_smiles

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    proposals: list[CandidateProposal] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            smiles = resolve_to_smiles(raw)
        except ValueError as exc:
            print(f"[WARNING] candidates file: skipping {raw!r} — {exc}", file=sys.stderr, flush=True)
            continue
        proposals.append(CandidateProposal(smiles=smiles, rationale="from file", family="unknown", source="file", source_id=""))
    return proposals
```

Verify `import sys` is present at the top of `orchestrator.py` (it should already be there).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_orchestrator.py::test_candidates_file_accepts_molecule_names tests/test_orchestrator.py::test_candidates_file_skips_unknown_names_with_warning -v
```

Expected: both PASS.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -x -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: resolve molecule names in candidates file"
```

---

## Task 6: FastAPI Integration (TDD)

**Files:**
- Modify: `des_multi_agent/server.py` (`POST /search` handler, around lines 103–130)
- Modify: `tests/test_server.py` (or create it if absent)

- [ ] **Step 1: Check whether a server test file exists**

```bash
ls tests/test_server.py 2>/dev/null && echo exists || echo missing
```

- [ ] **Step 2: Write failing tests**

If `tests/test_server.py` exists, append these tests. If it does not exist, create it:

```python
"""Tests for FastAPI server name resolution integration."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from des_multi_agent.server import app

client = TestClient(app)


def _minimal_payload(**overrides) -> dict:
    base = {
        "component_a": "NC(N)=O",
        "n": 2,
        "checkpoint_path": "fake.pt",
    }
    base.update(overrides)
    return base


def test_search_resolves_molecule_name(monkeypatch):
    """POST /search with a molecule name for component_a resolves to SMILES."""
    received = {}

    def fake_run_search_report(component_a, **kwargs):
        received["component_a"] = component_a
        from des_multi_agent.orchestrator import SearchOutcome
        return SearchOutcome(
            results=[], annotated_results=[], candidate_proposals=[],
            candidate_reviews=[], brainstorm_candidates=[], explanation_notes=[],
            critique_notes=[], llm_warnings=[], contradiction_notes=[],
            viscosity_predictions=[], chemical_pattern_memory=None,
            chemistry_lesson_summary=None,
        )

    monkeypatch.setattr("des_multi_agent.server.run_search_report", fake_run_search_report)
    # format_report is called after run_search_report; stub it to avoid
    # requiring a fully-populated SearchOutcome in this focused test.
    monkeypatch.setattr("des_multi_agent.server.format_report", lambda outcome, **kw: "")

    resp = client.post("/search", json=_minimal_payload(component_a="urea"))
    assert resp.status_code == 200
    assert received.get("component_a") == "NC(N)=O"


def test_search_returns_422_for_unknown_molecule_name():
    """POST /search with an unresolvable name returns 422."""
    resp = client.post("/search", json=_minimal_payload(component_a="not_a_real_molecule_xyz"))
    assert resp.status_code == 422
    assert "not_a_real_molecule_xyz" in resp.json()["detail"]
```

- [ ] **Step 3: Run to verify they fail**

```bash
python -m pytest tests/test_server.py::test_search_resolves_molecule_name tests/test_server.py::test_search_returns_422_for_unknown_molecule_name -v
```

Expected: both FAIL — server currently passes `component_a` straight through without resolution.

- [ ] **Step 4: Add resolution in `server.py` `POST /search` handler**

In `des_multi_agent/server.py`, find the `search()` function (around line 103). Add resolution at the very start of the function body, before the `policy = UncertaintyPolicy(...)` line:

```python
@app.post("/search", response_model=DESSearchResponse)
def search(req: DESSearchRequest) -> DESSearchResponse:
    """Screen component B candidates against a fixed component A."""
    from .chemistry.name_resolution import resolve_to_smiles
    try:
        component_a = resolve_to_smiles(req.component_a)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        policy = UncertaintyPolicy(
            ...  # existing code unchanged
        )
        ...
        outcome = run_search_report(
            component_a=component_a,   # ← use resolved value
            ...
        )
```

The key change: replace `req.component_a` with `component_a` (the resolved value) in the `run_search_report(...)` call. All other `req.*` fields remain unchanged.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_server.py::test_search_resolves_molecule_name tests/test_server.py::test_search_returns_422_for_unknown_molecule_name -v
```

Expected: both PASS.

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/ -x -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add des_multi_agent/server.py tests/test_server.py
git commit -m "feat: resolve molecule names in FastAPI /search endpoint"
```
