# Protonation-Aware Chemistry (pKa) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the deterministic chemistry layer protonation-aware so that, when a pH is supplied, donor atoms and H-bond counts reflect the dominant ionized species rather than the drawn neutral form.

**Architecture:** A new `protonation.py` module builds the dominant species via direct RDKit RWMol edits from a hand-rolled SMARTS→pKa table. `claim_grounding.structural_facts` and `ground_coordination` gain an optional `pH` argument (default `None` = unchanged, as-drawn); when a pH is passed they profile the species. A one-line fix to `coordination._is_donor` stops a protonated ammonium N⁺ from being miscounted as a metal donor. The metal-binding workflows pass a `binding_pH` (default 7.0); the DES path is untouched (neat).

**Tech Stack:** Python 3.13, RDKit (`Chem`, `RWMol`, SMARTS), pytest. No new dependencies.

## Global Constraints

- **Deterministic + offline:** no network, no LLM, no ML model. Identical results on any backend.
- **Never raises into a prompt/grounding path:** every entry point fails safe (passthrough on error), mirroring the existing `structural_facts` precedent.
- **Backward compatible:** `pH` defaults to `None` and `binding_pH` defaults to `7.0`; with defaults, all existing behavior and the current 684-test baseline are byte-identical.
- **Un-tabulated ionizable groups are left exactly as drawn** — never guessed.
- **TDD:** write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- All commits end with the trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `des_multi_agent/chemistry/protonation.py` (create) | pKa table + `dominant_species()` engine + result dataclasses |
| `des_multi_agent/chemistry/coordination.py` (modify `_is_donor`) | protonated N/O with no lone pair is not a donor |
| `des_multi_agent/chemistry/claim_grounding.py` (modify) | `structural_facts(pH=None)`, `ground_coordination(pH=None)`, species fields + clause |
| `des_multi_agent/workflows/metal_binding_screen.py` (modify) | `binding_pH` param threaded into coordination grounding |
| `des_multi_agent/workflows/metal_binding_selectivity.py` (modify) | `binding_pH` param + a coordination-grounding pass over brainstormed ligands |
| `tests/test_protonation.py` (create) | engine unit tests |
| `tests/test_coordination.py` (modify) | `_is_donor` species cases |
| `tests/test_claim_grounding.py` (modify) | species-aware facts + pH=None regression |
| `tests/test_metal_workflows_grounding.py` (modify) | `binding_pH` wiring |

---

## Task 1: Protonation engine (`protonation.py`)

**Files:**
- Create: `des_multi_agent/chemistry/protonation.py`
- Test: `tests/test_protonation.py`

**Interfaces:**
- Consumes: RDKit only.
- Produces:
  - `IonizedGroup(group_name: str, atom_idx: int, pka: float, state: str, charge: int)` — `state ∈ {"protonated","deprotonated","neutral"}`.
  - `ProtonationResult(input_smiles: str, pH: float, species_smiles: str, mol, groups: list[IonizedGroup], net_charge: int)`.
  - `dominant_species(smiles_or_mol, pH: float = 7.0) -> ProtonationResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_protonation.py`:

```python
"""Tests for the deterministic protonation / dominant-species engine."""
from __future__ import annotations

import pytest
from rdkit import Chem

from des_multi_agent.chemistry.protonation import (
    IonizedGroup,
    ProtonationResult,
    dominant_species,
    _IONIZABLE_SMARTS,
)


def _net(smiles: str, pH: float) -> int:
    return dominant_species(smiles, pH).net_charge


def test_carboxylic_acid_deprotonates_at_ph7():
    res = dominant_species("CC(=O)O", 7.0)
    assert res.net_charge == -1
    assert any(g.state == "deprotonated" and g.group_name == "carboxylic acid" for g in res.groups)


def test_aliphatic_amine_protonates_at_ph7():
    res = dominant_species("CCN", 7.0)
    assert res.net_charge == +1
    assert any(g.state == "protonated" for g in res.groups)


def test_glycine_is_zwitterion_at_ph7():
    res = dominant_species("NCC(=O)O", 7.0)
    assert res.net_charge == 0
    states = sorted(g.state for g in res.groups if g.state != "neutral")
    assert states == ["deprotonated", "protonated"]


def test_glycine_cation_at_ph1():
    assert _net("NCC(=O)O", 1.0) == +1


def test_glycine_anion_at_ph12():
    assert _net("NCC(=O)O", 12.0) == -1


def test_glycerol_unchanged_at_ph7():
    res = dominant_species("OCC(O)CO", 7.0)
    assert res.net_charge == 0
    assert all(g.state == "neutral" for g in res.groups)
    # Canonical species equals canonical input (no edits).
    assert res.species_smiles == Chem.MolToSmiles(Chem.MolFromSmiles("OCC(O)CO"))


def test_imidazole_state_flips_across_its_pka():
    # imidazole basic N, pKa ~7: protonated below, neutral above
    assert _net("c1c[nH]cn1", 6.0) == +1
    assert _net("c1c[nH]cn1", 8.0) == 0


def test_invalid_smiles_passthrough_never_raises():
    res = dominant_species("not_a_smiles", 7.0)
    assert res.species_smiles == "not_a_smiles"
    assert res.groups == []
    assert res.net_charge == 0


def test_untabulated_ionizable_group_left_as_drawn():
    # ethane has no ionizable group → unchanged, net 0
    res = dominant_species("CC", 7.0)
    assert res.net_charge == 0
    assert all(g.state == "neutral" for g in res.groups) or res.groups == []


def test_table_integrity_every_smarts_compiles_and_is_well_formed():
    for smarts, name, pka, kind in _IONIZABLE_SMARTS:
        assert Chem.MolFromSmarts(smarts) is not None, f"bad SMARTS: {smarts!r} ({name})"
        assert isinstance(pka, (int, float))
        assert kind in ("acid", "base")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_protonation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'des_multi_agent.chemistry.protonation'`.

- [ ] **Step 3: Implement `protonation.py`**

Create `des_multi_agent/chemistry/protonation.py`:

```python
"""Deterministic protonation / dominant-species engine.

Given a SMILES and a pH, returns the dominant ionized species by editing the
formal charge and hydrogen count of each tabulated ionizable group. Acids
deprotonate when pH > pKa; bases protonate when pH < pKa. Un-tabulated groups
are left exactly as drawn. Never raises — returns a passthrough result on any
error, mirroring chemistry.claim_grounding.structural_facts.

The pKa table is hand-curated and offline; identical results on any backend.
Each SMARTS is written so the ionizable atom is the FIRST atom of the match
(``match[0]``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

# (smarts, group_name, pka, kind). kind ∈ {"acid","base"}. Most-specific first.
# Ionizable atom is match[0] by construction.
_IONIZABLE_SMARTS: list[tuple[str, str, float, str]] = [
    # --- acids: protonated form is neutral, deprotonated carries -1 ---
    ("[OX2H1]S(=O)=O",           "sulfonic acid",   -1.0, "acid"),
    ("[OX2H1][PX4](=O)",         "phosphonic acid",  2.0, "acid"),
    ("[OX2H1]C=O",               "carboxylic acid",  4.2, "acid"),
    ("[OX2H1]c",                 "phenol",           9.9, "acid"),
    ("[SX2H1]",                  "thiol",           10.5, "acid"),
    # --- bases: protonated form carries +1, neutral form is "deprotonated base" ---
    ("[NX2]=C([NX3])[NX3]",      "guanidine",       12.5, "base"),
    ("[nX2;r5]",                 "imidazole",        7.0, "base"),
    ("[nX2;r6]",                 "pyridine",         5.2, "base"),
    ("[NX3;!$([N+]);!$(NC=O)]c", "aniline",          4.6, "base"),
    ("[NX3;H2,H1,H0;!$([N+]);!$(NC=O);!$(N=*);!$(Nc)]", "amine", 10.6, "base"),
]


@dataclass(frozen=True)
class IonizedGroup:
    group_name: str
    atom_idx: int
    pka: float
    state: str      # "protonated" | "deprotonated" | "neutral"
    charge: int     # formal charge applied to the matched atom in the species


@dataclass(frozen=True)
class ProtonationResult:
    input_smiles: str
    pH: float
    species_smiles: str        # canonical SMILES of the dominant species
    mol: object                # RDKit Mol of the dominant species, or None
    groups: list[IonizedGroup] = field(default_factory=list)
    net_charge: int = 0


def _passthrough(smiles_or_mol, pH: float) -> ProtonationResult:
    """Safe fallback: canonicalize if possible, otherwise echo the raw input."""
    if isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            smi = ""
        net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        return ProtonationResult(smi, pH, smi, mol, [], net)
    raw = smiles_or_mol if isinstance(smiles_or_mol, str) else str(smiles_or_mol)
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return ProtonationResult(raw, pH, raw, None, [], 0)
    smi = Chem.MolToSmiles(mol)
    net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    return ProtonationResult(raw, pH, smi, mol, [], net)


def _deprotonate(atom) -> None:
    atom.SetFormalCharge(-1)
    atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - 1))
    atom.SetNoImplicit(True)


def _protonate(atom) -> None:
    atom.SetFormalCharge(+1)
    atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
    atom.SetNoImplicit(True)


def dominant_species(smiles_or_mol, pH: float = 7.0) -> ProtonationResult:
    """Return the dominant ionized species of *smiles_or_mol* at *pH*.

    Never raises; returns a passthrough ProtonationResult on any failure.
    """
    try:
        base = (
            smiles_or_mol
            if isinstance(smiles_or_mol, Chem.Mol)
            else Chem.MolFromSmiles(smiles_or_mol)
        )
        if base is None:
            return _passthrough(smiles_or_mol, pH)
        input_smiles = Chem.MolToSmiles(base)
        rw = Chem.RWMol(base)

        touched: set[int] = set()
        groups: list[IonizedGroup] = []
        for smarts, name, pka, kind in _IONIZABLE_SMARTS:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            for match in rw.GetSubstructMatches(patt):
                idx = match[0]
                if idx in touched:
                    continue
                touched.add(idx)
                atom = rw.GetAtomWithIdx(idx)
                if kind == "acid" and pH > pka:
                    _deprotonate(atom)
                    groups.append(IonizedGroup(name, idx, pka, "deprotonated", -1))
                elif kind == "base" and pH < pka:
                    _protonate(atom)
                    groups.append(IonizedGroup(name, idx, pka, "protonated", +1))
                else:
                    groups.append(IonizedGroup(name, idx, pka, "neutral", 0))

        species = rw.GetMol()
        Chem.SanitizeMol(species)
        net = sum(a.GetFormalCharge() for a in species.GetAtoms())
        return ProtonationResult(
            input_smiles=input_smiles,
            pH=pH,
            species_smiles=Chem.MolToSmiles(species),
            mol=species,
            groups=groups,
            net_charge=net,
        )
    except Exception:
        return _passthrough(smiles_or_mol, pH)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_protonation.py -q`
Expected: PASS (all tests green). If `test_imidazole_state_flips_across_its_pka` fails on the aromatic-N protonation sanitize, confirm the `[nX2;r5]` atom is the basic nitrogen and that `_protonate` produces `[nH+]`; adjust `_protonate` only if RDKit rejects the aromatic perception, keeping the passthrough safety net intact.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: baseline + new tests pass; 0 failures.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/chemistry/protonation.py tests/test_protonation.py
git commit -m "feat: add deterministic protonation / dominant-species engine"
```

---

## Task 2: `_is_donor` species fix (`coordination.py`)

**Files:**
- Modify: `des_multi_agent/chemistry/coordination.py:32-42`
- Test: `tests/test_coordination.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature change. `coordination_profile` now reports a positively-charged N/O bearing protons (e.g. ammonium N⁺) as a non-donor; deprotonated O⁻ remains a donor.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_coordination.py`:

```python
def test_protonated_ammonium_nitrogen_is_not_a_donor():
    from des_multi_agent.chemistry.coordination import coordination_profile
    # Protonated ethylammonium: N has +1 and three H's, no lone pair to donate.
    prof = coordination_profile("CC[NH3+]")
    assert prof.donor_element_counts.get("N", 0) == 0
    assert prof.n_donor_atoms == 0


def test_deprotonated_carboxylate_oxygens_still_donate():
    from des_multi_agent.chemistry.coordination import coordination_profile
    # Acetate: both carboxylate O's are donors, collapsing to one site.
    prof = coordination_profile("CC(=O)[O-]")
    assert prof.donor_element_counts.get("O", 0) == 2
    assert prof.denticity == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_coordination.py::test_protonated_ammonium_nitrogen_is_not_a_donor -v`
Expected: FAIL — the protonated ammonium N is currently counted as a donor (`donor_element_counts["N"] == 1`).

- [ ] **Step 3: Implement the fix**

In `des_multi_agent/chemistry/coordination.py`, replace the `_is_donor` body (lines 32-42):

```python
def _is_donor(atom) -> bool:
    sym = atom.GetSymbol()
    if sym not in _DONOR_ELEMENTS:
        return False
    # A positively charged atom with a full valence (e.g. quaternary ammonium)
    # has no lone pair available to donate.
    if atom.GetFormalCharge() > 0 and atom.GetTotalNumHs() == 0 and atom.GetDegree() >= 4:
        return False
    if atom.GetFormalCharge() > 0 and sym == "N" and atom.GetDegree() == 4:
        return False
    # A protonated N/O (positive formal charge, lone pair consumed by an N–H/O–H
    # bond) cannot donate to a metal — e.g. an ammonium N⁺ on a deprotonated
    # species. Catches the degree<4 protonated cases the rules above miss.
    if atom.GetFormalCharge() > 0 and sym in ("N", "O") and atom.GetTotalNumHs() > 0:
        return False
    return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_coordination.py -q`
Expected: PASS — both new tests green and all existing coordination tests unchanged.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: 0 failures (neutral-molecule donor counts are unaffected because the new branch only triggers on a positive formal charge).

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/chemistry/coordination.py tests/test_coordination.py
git commit -m "fix: protonated N/O with no lone pair is not a metal donor"
```

---

## Task 3: Species-aware grounding (`claim_grounding.py`)

**Files:**
- Modify: `des_multi_agent/chemistry/claim_grounding.py` (`StructuralFacts`, `structural_facts`, `ground_coordination`)
- Test: `tests/test_claim_grounding.py`

**Interfaces:**
- Consumes: `dominant_species` (Task 1); the `_is_donor` fix (Task 2).
- Produces:
  - `StructuralFacts` gains two fields with defaults: `net_charge: int = 0`, `protonation_summary: str = ""`.
  - `structural_facts(smiles: str, pH: float | None = None) -> StructuralFacts`.
  - `ground_coordination(smiles: str, claim_text: str, pH: float | None = None) -> GroundingVerdict`.
  - `as_prompt_block()` appends a species clause only when `protonation_summary` is non-empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_claim_grounding.py`:

```python
def test_structural_facts_default_ph_none_is_unchanged_for_glycine():
    # Regression guard: pH=None must reproduce the as-drawn Phase 1 result.
    facts = structural_facts("NCC(=O)O")
    assert facts.denticity == 2
    assert facts.n_donor_atoms >= 2
    assert facts.net_charge == 0
    assert facts.protonation_summary == ""


def test_structural_facts_ph7_glycine_is_species_aware():
    # At pH 7 glycine is a zwitterion: N is protonated (not a donor),
    # only the carboxylate site remains → denticity 1.
    facts = structural_facts("NCC(=O)O", pH=7.0)
    assert facts.denticity == 1
    assert facts.donor_element_counts.get("N", 0) == 0
    assert facts.net_charge == 0
    assert facts.protonation_summary != ""


def test_as_prompt_block_species_clause_only_when_ph_applied():
    drawn = structural_facts("NCC(=O)O").as_prompt_block()
    species = structural_facts("NCC(=O)O", pH=7.0).as_prompt_block()
    assert "species @ pH" not in drawn
    assert "species @ pH" in species


def test_ground_coordination_ph_aware_demotes_protonated_amine_claim():
    # "bidentate N,O-chelator" is true as-drawn but not for the pH-7 zwitterion
    # whose amine N is protonated.
    drawn = ground_coordination("NCC(=O)O", "bidentate N,O-chelator")
    species = ground_coordination("NCC(=O)O", "bidentate N,O-chelator", pH=7.0)
    assert drawn.status == "verified"
    assert species.status != "verified"


def test_ground_coordination_default_ph_none_unchanged():
    v = ground_coordination("NCC(=O)O", "bidentate N,O-chelator")
    assert v.status == "verified"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_claim_grounding.py -q -k "species or ph or protonat"`
Expected: FAIL — `structural_facts()` takes no `pH` argument and `StructuralFacts` has no `net_charge`/`protonation_summary`.

- [ ] **Step 3: Implement the changes**

In `des_multi_agent/chemistry/claim_grounding.py`:

(a) Add the import near the other chemistry imports at the top of the file:

```python
from .protonation import dominant_species
```

(b) Add two fields to `StructuralFacts` (after `family_features`) and extend `as_prompt_block`:

```python
@dataclass(frozen=True)
class StructuralFacts:
    """Deterministic structural summary for a single SMILES string."""

    smiles: str                           # canonical SMILES or "unparseable"
    n_hbd: int
    n_hba: int
    hbond_role: str
    n_donor_atoms: int
    denticity: int
    donor_element_counts: dict[str, int]
    mean_donor_softness: float
    family_features: list[str]            # detected family tags, e.g. ["polyol"]
    net_charge: int = 0                   # net formal charge of the profiled species
    protonation_summary: str = ""         # non-empty only when a pH was applied

    def as_prompt_block(self) -> str:
        """Return a compact single-line fact string suitable for LLM injection."""
        donor_str = ", ".join(
            f"{count} {elem}"
            for elem, count in self.donor_element_counts.items()
        ) or "none"
        feat_str = str(self.family_features)
        block = (
            f"computed facts: HBD={self.n_hbd}, HBA={self.n_hba}, "
            f"role={self.hbond_role}, donor atoms={donor_str}, "
            f"denticity={self.denticity}, features={feat_str}"
        )
        if self.protonation_summary:
            block += f"; {self.protonation_summary}"
        return block
```

(c) Replace `structural_facts` so it accepts `pH` and profiles the species when given. Family features stay on the as-drawn molecule:

```python
def structural_facts(smiles: str, pH: float | None = None) -> StructuralFacts:
    """Compute a deterministic structural summary for *smiles*.

    When *pH* is None (default) the molecule is profiled as drawn — byte-identical
    to the original behavior. When *pH* is a float, the dominant ionized species at
    that pH is profiled for H-bond and coordination counts (family features are
    always read from the as-drawn form). Never raises — returns a safe sentinel on
    any error.
    """
    try:
        # H-bond + coordination are computed on the (optionally protonated) species.
        net_charge = 0
        protonation_summary = ""
        if pH is None:
            profiled = smiles
        else:
            res = dominant_species(smiles, pH)
            profiled = res.species_smiles
            net_charge = res.net_charge
            ionized = [g for g in res.groups if g.state != "neutral"]
            if ionized:
                parts = ", ".join(f"{g.group_name} {g.state}" for g in ionized)
                protonation_summary = f"species @ pH{pH:g}: net charge {net_charge:+d} ({parts})"
            else:
                protonation_summary = f"species @ pH{pH:g}: net charge {net_charge:+d} (no ionizable groups)"

        hb = hbond_profile(profiled)
        cp = coordination_profile(profiled)

        # Family features are read from the AS-DRAWN molecule (functional-group
        # identity is named on the neutral form).
        mol = Chem.MolFromSmiles(smiles)
        features: list[str] = []
        if mol is not None:
            for tag, (smarts, min_count) in _FAMILY_SMARTS.items():
                patt = Chem.MolFromSmarts(smarts)
                if patt is not None:
                    matches = mol.GetSubstructMatches(patt)
                    if len(matches) >= min_count:
                        features.append(tag)

        return StructuralFacts(
            smiles=hb.smiles,
            n_hbd=hb.n_hbd,
            n_hba=hb.n_hba,
            hbond_role=hb.role,
            n_donor_atoms=cp.n_donor_atoms,
            denticity=cp.denticity,
            donor_element_counts=dict(cp.donor_element_counts),
            mean_donor_softness=cp.mean_donor_softness,
            family_features=features,
            net_charge=net_charge,
            protonation_summary=protonation_summary,
        )
    except Exception:
        return StructuralFacts(
            smiles="unparseable",
            n_hbd=0,
            n_hba=0,
            hbond_role="none",
            n_donor_atoms=0,
            denticity=0,
            donor_element_counts={},
            mean_donor_softness=0.0,
            family_features=[],
        )
```

(d) Replace `ground_coordination` so it accepts `pH` and verifies against the species:

```python
def ground_coordination(
    smiles: str, claim_text: str, pH: float | None = None
) -> GroundingVerdict:
    """Verify a natural-language coordination claim against structural evidence.

    Uses :func:`verify_coordination_claim` from *claim_verification.py* and maps
    the result to a :class:`GroundingVerdict`. When *pH* is a float, the claim is
    checked against the dominant ionized species at that pH; when None (default),
    against the molecule as drawn.
    """
    target_smiles = smiles
    if pH is not None:
        target_smiles = dominant_species(smiles, pH).species_smiles
    cv = verify_coordination_claim(target_smiles, claim_text)
    status = _COORD_VERDICT_MAP.get(cv.verdict, "unverifiable")
    penalty = 0.25 if status == "contradicted" else 0.0
    detail = "; ".join(cv.notes) if cv.notes else cv.verdict
    return GroundingVerdict(
        claim=claim_text,
        status=status,
        detail=detail,
        penalty=penalty,
    )
```

(Keep the existing `claim=`/`status=`/`detail=`/`penalty=` constructor exactly as it was; only the leading lines and signature change.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_claim_grounding.py -q`
Expected: PASS — new species tests green AND all Phase 1 as-drawn tests unchanged (proving `pH=None` is byte-identical).

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/chemistry/claim_grounding.py tests/test_claim_grounding.py
git commit -m "feat: make structural_facts and ground_coordination protonation-aware (pH opt-in)"
```

---

## Task 4: Metal-binding workflow wiring (`binding_pH`)

**Files:**
- Modify: `des_multi_agent/workflows/metal_binding_screen.py` (signature + `ground_coordination` call)
- Modify: `des_multi_agent/workflows/metal_binding_selectivity.py` (signature + new coordination-grounding pass)
- Test: `tests/test_metal_workflows_grounding.py`

**Interfaces:**
- Consumes: `ground_coordination(smiles, claim, pH=...)` (Task 3).
- Produces:
  - `run_metal_binding_screen(..., binding_pH: float = 7.0)`.
  - `run_metal_selectivity_screen(..., binding_pH: float = 7.0)`, whose `SelectivityScreenOutcome.claim_verdicts` now also contains species-aware coordination verdicts for brainstormed ligands.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metal_workflows_grounding.py`:

```python
def test_metal_binding_screen_accepts_binding_ph():
    import inspect
    from des_multi_agent.workflows.metal_binding_screen import run_metal_binding_screen
    sig = inspect.signature(run_metal_binding_screen)
    assert "binding_pH" in sig.parameters
    assert sig.parameters["binding_pH"].default == 7.0


def test_metal_selectivity_screen_accepts_binding_ph():
    import inspect
    from des_multi_agent.workflows.metal_binding_selectivity import run_metal_selectivity_screen
    sig = inspect.signature(run_metal_selectivity_screen)
    assert "binding_pH" in sig.parameters
    assert sig.parameters["binding_pH"].default == 7.0


def test_selectivity_screen_grounds_coordination_for_brainstormed_ligands():
    """With an LLM that brainstorms a ligand carrying a coordination rationale,
    the selectivity outcome's claim_verdicts include a coordination verdict."""
    from des_multi_agent.llm.schemas import CandidateBrainstorm, CandidateReview
    from des_multi_agent.workflows.metal_binding_selectivity import run_metal_selectivity_screen

    class _FakeLLM:
        def brainstorm_ligands_selectivity(self, target, competitor, constraints, context):
            return [CandidateBrainstorm(smiles="NCC(=O)O", rationale="bidentate N,O-chelator", family="amino acid")]
        def review_ligand(self, metal_ion, ligand_smiles, context):
            return CandidateReview(smiles=ligand_smiles, decision="keep", confidence=0.8, rationale="ok", notes=[])

    outcome = run_metal_selectivity_screen(
        "Cu2+", "Zn2+", n=3, llm_provider=_FakeLLM(), n_cycles=1, binding_pH=7.0,
    )
    assert isinstance(outcome.claim_verdicts, list)
    assert any(getattr(v, "claim", "") == "bidentate N,O-chelator" for v in outcome.claim_verdicts)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_metal_workflows_grounding.py -q -k "binding_ph or grounds_coordination"`
Expected: FAIL — `binding_pH` is not a parameter and the selectivity workflow does not yet ground coordination.

- [ ] **Step 3a: Wire `metal_binding_screen.py`**

Change the signature (currently lines 112-118):

```python
def run_metal_binding_screen(
    metal_ion: str,
    n: int = 20,
    model_path=None,
    llm_provider=None,
    constraints: dict | None = None,
    n_cycles: int = 1,
    binding_pH: float = 7.0,
) -> MetalBindingScreenOutcome:
    from ..chemistry.claim_grounding import ground_coordination as _ground_coord
    seen_smiles: set[str] = set()
```

Change the coordination grounding call (currently line 148) to pass the pH:

```python
                            v = _ground_coord(b.smiles, b.rationale, pH=binding_pH)
```

- [ ] **Step 3b: Wire `metal_binding_selectivity.py`**

Add `binding_pH: float = 7.0` as the final parameter of `run_metal_selectivity_screen` (after `stability_rule_weight: float = 0.5`):

```python
    stability_rule_weight: float = 0.5,
    binding_pH: float = 7.0,
) -> SelectivityScreenOutcome:
    from ..chemistry.claim_grounding import ground_coordination as _ground_coord
    seen_smiles: set[str] = set()
    all_reviews: list[CandidateReview] = []
    all_brainstorm: list[CandidateBrainstorm] = []
    all_warnings: list[str] = []
    all_sel_verdicts: list[object] = []
    all_coord_verdicts: list[object] = []
    cumulative_results: list[SelectivityResult] = []
    prev_cycle_results: list[SelectivityResult] = []
```

Inside the brainstorm `try` block, immediately after
`proposals.extend(_deduplicate_proposals(llm_proposals, seen_smiles))`, add a
species-aware coordination grounding pass (mirroring the screen workflow):

```python
                # Ground coordination claims from LLM rationale (species-aware).
                for b in brainstorms:
                    if b.rationale:
                        try:
                            v = _ground_coord(b.smiles, b.rationale, pH=binding_pH)
                            all_coord_verdicts.append(v)
                            if v.status == "contradicted":
                                all_warnings.append(
                                    f"[GROUNDING] Coordination contradicted for {b.smiles}: {v.detail}"
                                )
                        except Exception:
                            pass
```

Finally, find the `SelectivityScreenOutcome(...)` constructor (it currently passes
`claim_verdicts=all_sel_verdicts`) and change that one argument to include the
coordination verdicts:

```python
        claim_verdicts=all_sel_verdicts + all_coord_verdicts,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_metal_workflows_grounding.py -q`
Expected: PASS — both signatures expose `binding_pH` and the selectivity outcome now carries the coordination verdict.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: 0 failures. (Phase 5 selectivity tests still pass: without an LLM no coordination verdicts are added, so the prior `len(claim_verdicts) >= len(results)` assertion from selectivity grounding still holds.)

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/workflows/metal_binding_screen.py des_multi_agent/workflows/metal_binding_selectivity.py tests/test_metal_workflows_grounding.py
git commit -m "feat: thread binding_pH into species-aware coordination grounding for metal workflows"
```

---

## Task 5: Document the protonation layer (`SKILL.md`)

**Files:**
- Modify: `.claude/skills/des-chemistry/SKILL.md`

**Interfaces:**
- Consumes: the public surface from Tasks 1 and 3.
- Produces: documentation only (no code, no test).

- [ ] **Step 1: Add a `protonation.py` section**

Append a new section to `.claude/skills/des-chemistry/SKILL.md` after the existing
module sections:

```markdown
## `protonation.py` — Dominant-Species Engine (pKa-aware)

**When to use:** any time donor availability or H-bond counts must reflect the
species that actually exists at a given pH rather than the drawn neutral form —
chiefly the metal-binding workflows.

```python
from des_multi_agent.chemistry.protonation import dominant_species

res = dominant_species("NCC(=O)O", pH=7.0)   # glycine
# ProtonationResult(
#   species_smiles="...zwitterion...",
#   net_charge=0,
#   groups=[IonizedGroup("carboxylic acid", ..., "deprotonated", -1),
#           IonizedGroup("amine", ..., "protonated", +1)],
# )
```

- Hand-rolled SMARTS→pKa table; acids deprotonate when pH > pKa, bases protonate
  when pH < pKa. Un-tabulated groups are left exactly as drawn (never guessed).
- **Never raises** — returns a passthrough result on any error.
- Wired into `claim_grounding.structural_facts(smiles, pH=None)` and
  `ground_coordination(smiles, claim, pH=None)`: pass a `pH` to profile the
  species; omit it (default `None`) to keep the as-drawn behavior. The DES partner
  search stays as-drawn (neat); the metal-binding workflows pass `binding_pH`
  (default 7.0).
- **Limitation:** models the free-ligand dominant species, not metal-assisted
  deprotonation on coordination; family classification stays as-drawn.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/des-chemistry/SKILL.md
git commit -m "docs: document protonation/dominant-species layer in des-chemistry skill"
```

---

## Final verification

- [ ] Run the full suite: `python -m pytest tests/ -q` — 0 failures, count ≥ 684 + ~16 new.
- [ ] Confirm `pH=None` / default `binding_pH` paths are byte-identical to pre-feature behavior (the regression tests in Tasks 2 and 3 assert this directly).
- [ ] Confirm the engine is backend-agnostic: `dominant_species` and the grounding paths invoke no LLM and no network.
```
