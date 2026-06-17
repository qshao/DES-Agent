# Reality-Anchored DES Partner Proposals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anchor LLM DES-partner brainstorming to real, known molecules and deterministically flag / demote / drop invented or implausible proposals.

**Architecture:** Two halves mirroring the existing `claim_grounding` layer. Source-side: a `partner_registry` module builds a menu of known real partners (curated registry ∪ auto-role-tagged experimental compounds) injected into the brainstorm prompt. Output-side: `ground_partner_reality` grades each LLM proposal into a `PartnerVerdict` (known/novel_plausible/novel_implausible) that the orchestrator keeps, demotes (−0.25), or drops.

**Tech Stack:** Python, RDKit (existing dep), pytest. No new dependencies.

## Global Constraints

- Offline + deterministic. No network, no embeddings, no new runtime deps. RDKit only.
- Reuse `hbond_profile`, `des_hbond_complementarity`, `canonicalize_smiles` — do not reinvent.
- Backward compatible: new prompt param defaults to "no menu"; existing prompt tests and the disabled-LLM path stay byte-identical; deterministic (no-LLM) example baselines do not move.
- Never raise into the proposal/prompt path — pass through safely on error.
- Role vocabulary: `"HBD" | "HBA" | "amphoteric"` (and `"none"`). Never introduce `"both"`.
- Membership key: canonical InChIKey via `Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))`, recomputed from each source's stored SMILES.
- Artifact paths via `Path(__file__).resolve().parents[2] / "artifacts" / ...` (mirror `chemistry/name_resolution.py`).
- Spec: `docs/superpowers/specs/2026-06-17-reality-anchored-partner-proposals-design.md`.

---

### Task 1: Known-set membership (`partner_registry.py`)

**Files:**
- Create: `des_multi_agent/chemistry/partner_registry.py`
- Test: `tests/test_partner_registry.py`

**Interfaces:**
- Produces: `known_inchikeys() -> frozenset[str]`, `is_known(smiles: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_partner_registry.py
from des_multi_agent.chemistry.partner_registry import is_known, known_inchikeys


def test_known_inchikeys_is_nonempty_frozenset():
    keys = known_inchikeys()
    assert isinstance(keys, frozenset)
    assert len(keys) > 50  # 57 curated + hundreds experimental


def test_is_known_true_for_registry_salt():
    # choline chloride salt is in common_names.json
    assert is_known("C[N+](C)(C)CCO.[Cl-]") is True


def test_is_known_true_for_experimental_compound():
    # glycolic acid O=C(O)CO is in melting_points/experimental.json
    assert is_known("O=C(O)CO") is True


def test_is_known_false_for_invented_molecule():
    # a large fused-ring system not in either dataset
    assert is_known("c1ccc2c(c1)c1ccc3ccccc3c1c1ccccc21") is False


def test_is_known_false_on_unparseable_without_raising():
    assert is_known("not_a_smiles((((") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_partner_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: ... partner_registry`

- [ ] **Step 3: Write minimal implementation**

```python
# des_multi_agent/chemistry/partner_registry.py
"""Reality anchoring for DES partner proposals.

Known-set membership (real, attested compounds), a role-tagged anchor menu,
and a structural-sanity gate. Offline + deterministic. Never raises into the
proposal/prompt path.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

from .hbond import hbond_profile

_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
_COMMON_NAMES_PATH = _ARTIFACTS / "molecule_names" / "common_names.json"
_EXPERIMENTAL_PATH = _ARTIFACTS / "melting_points" / "experimental.json"


def _inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


@lru_cache(maxsize=1)
def known_inchikeys() -> frozenset[str]:
    """Canonical InChIKey set from common_names.json ∪ experimental.json.

    Recomputes the InChIKey from each stored SMILES so membership uses the same
    key form as incoming proposals. Missing/invalid artifacts degrade gracefully
    to whatever loads.
    """
    keys: set[str] = set()
    try:
        data = json.loads(_COMMON_NAMES_PATH.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            k = _inchikey(entry["smiles"])
            if k is not None:
                keys.add(k)
    except Exception:
        pass
    try:
        data = json.loads(_EXPERIMENTAL_PATH.read_text(encoding="utf-8"))
        for record in data.get("entries", {}).values():
            k = _inchikey(record["smiles"])
            if k is not None:
                keys.add(k)
    except Exception:
        pass
    return frozenset(keys)


def is_known(smiles: str) -> bool:
    """True if the canonical InChIKey of `smiles` is in the known set.

    Returns False on unparseable SMILES (never raises).
    """
    k = _inchikey(smiles)
    return k is not None and k in known_inchikeys()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_partner_registry.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemistry/partner_registry.py tests/test_partner_registry.py
git commit -m "feat: known-set membership for partner reality anchoring"
```

---

### Task 2: Structural-sanity gate (`partner_registry.py`)

**Files:**
- Modify: `des_multi_agent/chemistry/partner_registry.py`
- Test: `tests/test_partner_registry.py`

**Interfaces:**
- Produces: `structural_sanity(smiles: str) -> tuple[bool, str]`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_partner_registry.py
from des_multi_agent.chemistry.partner_registry import structural_sanity


def test_structural_sanity_passes_common_des_components():
    for smi in ("CCO", "NC(N)=O", "OCC(O)CO"):  # ethanol, urea, glycerol
        ok, reason = structural_sanity(smi)
        assert ok is True, (smi, reason)
        assert reason == ""


def test_structural_sanity_rejects_disallowed_element():
    ok, reason = structural_sanity("OB(O)O")  # boric acid — boron not allowed
    assert ok is False
    assert "element" in reason.lower()


def test_structural_sanity_rejects_oversized_molecule():
    # long alkane, MW well above 400
    ok, reason = structural_sanity("C" * 40)
    assert ok is False
    assert "weight" in reason.lower()


def test_structural_sanity_rejects_tiny_molecule():
    ok, reason = structural_sanity("C")  # methane, MW ~16 < 40
    assert ok is False
    assert "weight" in reason.lower()


def test_structural_sanity_rejects_radical():
    ok, reason = structural_sanity("[CH3]")  # methyl radical
    assert ok is False
    assert "radical" in reason.lower()


def test_structural_sanity_rejects_invalid_smiles():
    ok, reason = structural_sanity("xyz(((")
    assert ok is False
    assert "invalid" in reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_partner_registry.py -k structural_sanity -v`
Expected: FAIL with `ImportError: cannot import name 'structural_sanity'`

- [ ] **Step 3: Write minimal implementation**

Add to `des_multi_agent/chemistry/partner_registry.py`:

```python
from rdkit.Chem import Descriptors

_ALLOWED_ELEMENTS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def structural_sanity(smiles: str) -> tuple[bool, str]:
    """Deterministic 'is this a sane small molecule' check.

    Fails when: unparseable; any atom outside the allowed-element set; any
    radical electrons; molecular weight outside the open interval (40, 400).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "invalid SMILES"
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in _ALLOWED_ELEMENTS:
            return False, f"disallowed element: {atom.GetSymbol()}"
        if atom.GetNumRadicalElectrons() > 0:
            return False, "radical species"
    mw = Descriptors.MolWt(mol)
    if not (40.0 < mw < 400.0):
        return False, f"molecular weight out of range: {mw:.1f}"
    return True, ""
```

Put the `from rdkit.Chem import Descriptors` import at the top of the module with the other imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_partner_registry.py -k structural_sanity -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemistry/partner_registry.py tests/test_partner_registry.py
git commit -m "feat: structural-sanity gate for partner reality anchoring"
```

---

### Task 3: Anchor menu (`partner_registry.py`)

**Files:**
- Modify: `des_multi_agent/chemistry/partner_registry.py`
- Test: `tests/test_partner_registry.py`

**Interfaces:**
- Consumes: `known_inchikeys` (Task 1), `hbond_profile(smiles).role` from `chemistry/hbond.py`
- Produces: `MenuEntry(smiles: str, display_name: str, role: str)` dataclass, `known_partner_menu(role: str, limit: int = 30) -> list[MenuEntry]`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_partner_registry.py
from des_multi_agent.chemistry.partner_registry import MenuEntry, known_partner_menu


def test_menu_for_hbd_is_nonempty_and_role_serving():
    menu = known_partner_menu("HBD", limit=30)
    assert len(menu) > 0
    assert all(isinstance(e, MenuEntry) for e in menu)
    # every entry must be able to serve an HBD request
    assert all(e.role in ("HBD", "amphoteric") for e in menu)


def test_menu_respects_limit():
    menu = known_partner_menu("HBA", limit=5)
    assert len(menu) <= 5


def test_menu_has_no_duplicate_inchikeys():
    from des_multi_agent.chemistry.partner_registry import _inchikey
    menu = known_partner_menu("amphoteric", limit=100)
    keys = [_inchikey(e.smiles) for e in menu]
    assert len(keys) == len(set(keys))


def test_menu_curated_entries_come_first():
    # curated entries carry human names; auto-tagged ones use the SMILES as name
    menu = known_partner_menu("HBA", limit=100)
    named = [i for i, e in enumerate(menu) if e.display_name != e.smiles]
    auto = [i for i, e in enumerate(menu) if e.display_name == e.smiles]
    if named and auto:
        assert max(named) < min(auto)


def test_menu_amphoteric_request_returns_any_role():
    menu = known_partner_menu("amphoteric", limit=100)
    roles = {e.role for e in menu}
    assert roles  # non-empty; may include HBD, HBA, amphoteric
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_partner_registry.py -k menu -v`
Expected: FAIL with `ImportError: cannot import name 'MenuEntry'`

- [ ] **Step 3: Write minimal implementation**

Add to `des_multi_agent/chemistry/partner_registry.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class MenuEntry:
    smiles: str
    display_name: str   # curated name, or the SMILES for auto-tagged entries
    role: str           # "HBD" | "HBA" | "amphoteric"


def _serves(entry_role: str, wanted: str) -> bool:
    return (
        entry_role == wanted
        or entry_role == "amphoteric"
        or wanted == "amphoteric"
    )


@lru_cache(maxsize=1)
def _all_menu_entries() -> tuple[MenuEntry, ...]:
    """Curated registry entries first, then auto-role-tagged experimental
    compounds, deduped by InChIKey. Built once and cached."""
    entries: list[MenuEntry] = []
    seen: set[str] = set()

    # Curated registry: trust the stored role tag; keep H-bonders only.
    try:
        data = json.loads(_COMMON_NAMES_PATH.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            role = entry.get("role", "")
            if role not in ("HBD", "HBA", "amphoteric"):
                continue
            k = _inchikey(entry["smiles"])
            if k is None or k in seen:
                continue
            seen.add(k)
            name = entry["names"][0] if entry.get("names") else entry["smiles"]
            entries.append(MenuEntry(entry["smiles"], name, role))
    except Exception:
        pass

    # Experimental compounds: derive role from the H-bond profiler.
    try:
        data = json.loads(_EXPERIMENTAL_PATH.read_text(encoding="utf-8"))
        for record in data.get("entries", {}).values():
            smi = record["smiles"]
            k = _inchikey(smi)
            if k is None or k in seen:
                continue
            role = hbond_profile(smi).role
            if role not in ("HBD", "HBA", "amphoteric"):
                continue
            seen.add(k)
            entries.append(MenuEntry(smi, smi, role))
    except Exception:
        pass

    return tuple(entries)


def known_partner_menu(role: str, limit: int = 30) -> list[MenuEntry]:
    """Menu entries that can serve the wanted partner role `role`.

    An entry serves `role` when its role equals `role`, or either side is
    "amphoteric". Curated entries precede auto-tagged ones; capped at `limit`.
    """
    out: list[MenuEntry] = []
    for e in _all_menu_entries():
        if _serves(e.role, role):
            out.append(e)
            if len(out) >= limit:
                break
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_partner_registry.py -v`
Expected: PASS (all tasks 1-3 tests pass)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemistry/partner_registry.py tests/test_partner_registry.py
git commit -m "feat: role-tagged anchor menu for partner reality anchoring"
```

---

### Task 4: `PartnerVerdict` + `ground_partner_reality` (`claim_grounding.py`)

**Files:**
- Modify: `des_multi_agent/chemistry/claim_grounding.py`
- Test: `tests/test_claim_grounding.py`

**Interfaces:**
- Consumes: `is_known`, `structural_sanity` (Tasks 1-2); `des_hbond_complementarity` (already imported in `claim_grounding.py`)
- Produces: `PartnerVerdict(claim, status, detail, penalty, disposition)`, `ground_partner_reality(component_a: str, candidate_smiles: str) -> PartnerVerdict`. `status ∈ {"known","novel_plausible","novel_implausible"}`, `disposition ∈ {"keep","demote","drop"}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_claim_grounding.py
from des_multi_agent.chemistry.claim_grounding import (
    PartnerVerdict,
    ground_partner_reality,
)


def test_partner_reality_known_compound_keeps():
    v = ground_partner_reality("CC(=O)O", "NC(N)=O")  # acetic acid + urea (known)
    assert v.status == "known"
    assert v.disposition == "keep"
    assert v.penalty == 0.0


def test_partner_reality_novel_complementary_keeps_as_plausible():
    # a valid, sane, complementary HBD not in the known set
    v = ground_partner_reality("C[N+](C)(C)CCO.[Cl-]", "OCCCCCCO")  # 1,6-hexanediol
    assert v.status == "novel_plausible"
    assert v.disposition == "keep"
    assert v.penalty == 0.0


def test_partner_reality_noncomplementary_novel_demotes():
    # valid + sane but no H-bond complementarity with a pure alkane component
    v = ground_partner_reality("CCCCCCCC", "CCCCCCCCCC")  # octane + decane
    assert v.status == "novel_implausible"
    assert v.disposition == "demote"
    assert v.penalty == 0.25


def test_partner_reality_bad_element_drops():
    v = ground_partner_reality("CCO", "OB(O)O")  # boron — fails sanity
    assert v.status == "novel_implausible"
    assert v.disposition == "drop"
    assert v.penalty == 0.0


def test_partner_reality_invalid_smiles_drops():
    v = ground_partner_reality("CCO", "garbage(((")
    assert v.status == "novel_implausible"
    assert v.disposition == "drop"


def test_partner_verdict_invariants_enforced():
    import pytest
    with pytest.raises(ValueError):
        PartnerVerdict(claim="x", status="known", detail="", penalty=0.25, disposition="keep")
    with pytest.raises(ValueError):
        PartnerVerdict(claim="x", status="novel_implausible", detail="", penalty=0.0, disposition="demote")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_claim_grounding.py -k partner -v`
Expected: FAIL with `ImportError: cannot import name 'PartnerVerdict'`

- [ ] **Step 3: Write minimal implementation**

Add to `des_multi_agent/chemistry/claim_grounding.py` (after the `GroundingVerdict` class). The module already imports `Chem`, `des_hbond_complementarity`:

```python
from .partner_registry import is_known, structural_sanity


@dataclass(frozen=True)
class PartnerVerdict:
    """Reality grading of one proposed DES partner.

    status:      "known" | "novel_plausible" | "novel_implausible"
    disposition: "keep" | "demote" | "drop"
    Invariants:
      keep   ⟺ penalty == 0.0 and status in {known, novel_plausible}
      demote ⟹ penalty > 0.0 and status == novel_implausible
      drop   ⟹ penalty == 0.0 and status == novel_implausible
    """

    claim: str
    status: str
    detail: str
    penalty: float
    disposition: str

    def __post_init__(self) -> None:
        if self.disposition == "keep":
            if self.penalty != 0.0 or self.status not in ("known", "novel_plausible"):
                raise ValueError("keep requires penalty=0.0 and a non-implausible status")
        elif self.disposition == "demote":
            if self.penalty <= 0.0 or self.status != "novel_implausible":
                raise ValueError("demote requires penalty>0.0 and status=novel_implausible")
        elif self.disposition == "drop":
            if self.penalty != 0.0 or self.status != "novel_implausible":
                raise ValueError("drop requires penalty=0.0 and status=novel_implausible")
        else:
            raise ValueError(f"unknown disposition: {self.disposition!r}")


def ground_partner_reality(component_a: str, candidate_smiles: str) -> PartnerVerdict:
    """Deterministically grade a proposed DES partner against reality.

    Order: invalid → drop; known → keep; bad structure → drop; no H-bond
    complementarity → demote; otherwise novel-plausible → keep. Never raises;
    on internal error returns a neutral keep (we do not punish our own failure).
    """
    claim = f"partner reality: {candidate_smiles}"
    try:
        if Chem.MolFromSmiles(candidate_smiles) is None:
            return PartnerVerdict(claim, "novel_implausible", "invalid SMILES", 0.0, "drop")
        if is_known(candidate_smiles):
            return PartnerVerdict(claim, "known", "known/attested compound", 0.0, "keep")
        ok, reason = structural_sanity(candidate_smiles)
        if not ok:
            return PartnerVerdict(claim, "novel_implausible", reason, 0.0, "drop")
        label = des_hbond_complementarity(component_a, candidate_smiles).label
        if label == "none":
            return PartnerVerdict(
                claim, "novel_implausible",
                "no H-bond complementarity with component A", 0.25, "demote",
            )
        return PartnerVerdict(claim, "novel_plausible", f"novel; complementarity={label}", 0.0, "keep")
    except Exception:
        return PartnerVerdict(claim, "novel_plausible", "reality check skipped (internal error)", 0.0, "keep")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_claim_grounding.py -k partner -v`
Expected: PASS (6 passed)

If `test_partner_reality_novel_complementary_keeps_as_plausible` fails because 1,6-hexanediol turns out to be in `experimental.json` (would return `known`, which is still a `keep`), relax that test to assert `v.disposition == "keep"` only. Verify with `python -c "from des_multi_agent.chemistry.partner_registry import is_known; print(is_known('OCCCCCCO'))"` and adjust the expected status to match.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemistry/claim_grounding.py tests/test_claim_grounding.py
git commit -m "feat: ground_partner_reality verdict for DES partners"
```

---

### Task 5: Prompt anchor block (`prompts.py`)

**Files:**
- Modify: `des_multi_agent/llm/prompts.py:170-201` (`candidate_brainstorm_prompt`)
- Test: `tests/test_prompts.py` (or wherever brainstorm-prompt tests live; create `tests/test_prompts_partner_menu.py` if none)

**Interfaces:**
- Consumes: `MenuEntry` (Task 3)
- Produces: `candidate_brainstorm_prompt(..., known_partner_menu: list | None = None)` — new keyword-only-safe param appended last; when non-empty, renders an anchor block.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts_partner_menu.py
from des_multi_agent.chemistry.partner_registry import MenuEntry
from des_multi_agent.llm.prompts import candidate_brainstorm_prompt


def test_prompt_without_menu_has_no_anchor_block():
    p = candidate_brainstorm_prompt("CCO", None, "ctx")
    assert "known, real molecules" not in p


def test_prompt_with_menu_renders_anchor_block():
    menu = [
        MenuEntry("NC(N)=O", "urea", "HBD"),
        MenuEntry("OCC(O)CO", "glycerol", "amphoteric"),
    ]
    p = candidate_brainstorm_prompt("CCO", None, "ctx", known_partner_menu=menu)
    assert "known, real molecules" in p
    assert "urea [HBD]" in p
    assert "glycerol [amphoteric]" in p


def test_prompt_with_empty_menu_has_no_anchor_block():
    p = candidate_brainstorm_prompt("CCO", None, "ctx", known_partner_menu=[])
    assert "known, real molecules" not in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts_partner_menu.py -v`
Expected: FAIL with `TypeError: candidate_brainstorm_prompt() got an unexpected keyword argument 'known_partner_menu'`

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/llm/prompts.py`, change the `candidate_brainstorm_prompt` signature to append the new param after `facts_block`:

```python
def candidate_brainstorm_prompt(
    component_a: str,
    constraints: dict | None,
    context: str,
    max_items: int | None = None,
    families: list | None = None,
    diversity_mode: str = "balanced",
    family_bias_strength: float = 0.5,
    prior_productive_families: dict[str, int] | None = None,
    facts_block: str = "",
    known_partner_menu: list | None = None,
) -> str:
```

Then, immediately after the existing `if facts_block:` block (before the `parts += [...]` for Component A), add:

```python
    if known_partner_menu:
        menu_lines = ["Prefer partners from this list of known, real molecules. "
                      "You MAY propose others, but only with an explicit "
                      "justification in the rationale:\n"]
        for e in known_partner_menu:
            menu_lines.append(f"  - {e.display_name} [{e.role}]\n")
        parts.append("".join(menu_lines))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompts_partner_menu.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/prompts.py tests/test_prompts_partner_menu.py
git commit -m "feat: render known-partner anchor block in brainstorm prompt"
```

---

### Task 6: Build + inject the menu (`base.py`)

**Files:**
- Modify: `des_multi_agent/llm/base.py:69-112` (`brainstorm_candidates`)
- Test: `tests/test_llm_base_partner_menu.py`

**Interfaces:**
- Consumes: `known_partner_menu` (Task 3), `candidate_brainstorm_prompt(known_partner_menu=...)` (Task 5), `hbond_profile` from `chemistry/hbond.py`
- Produces: `_complementary_role(role: str) -> str` helper in `base.py`; `brainstorm_candidates` now passes a menu into the prompt.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_base_partner_menu.py
from des_multi_agent.llm.base import _complementary_role


def test_complementary_role_mapping():
    assert _complementary_role("HBA") == "HBD"
    assert _complementary_role("HBD") == "HBA"
    assert _complementary_role("amphoteric") == "amphoteric"
    assert _complementary_role("none") == "amphoteric"
    assert _complementary_role("anything else") == "amphoteric"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_base_partner_menu.py -v`
Expected: FAIL with `ImportError: cannot import name '_complementary_role'`

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/llm/base.py`, add imports near the top (the module already imports `structural_facts` from `..chemistry.claim_grounding`):

```python
from ..chemistry.hbond import hbond_profile
from ..chemistry.partner_registry import known_partner_menu
```

Add the helper at module level:

```python
def _complementary_role(role: str) -> str:
    """Partner role that complements a component with the given H-bond role."""
    if role == "HBA":
        return "HBD"
    if role == "HBD":
        return "HBA"
    return "amphoteric"
```

In `brainstorm_candidates`, after `facts_block = structural_facts(component_a).as_prompt_block()` and before the `raw = self._request(...)` call, build the menu defensively:

```python
        try:
            wanted = _complementary_role(hbond_profile(component_a).role)
            partner_menu = known_partner_menu(wanted, limit=30)
        except Exception:
            partner_menu = None
```

Then pass it into the prompt call by adding `known_partner_menu=partner_menu` as the final keyword argument to `candidate_brainstorm_prompt(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_base_partner_menu.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/base.py tests/test_llm_base_partner_menu.py
git commit -m "feat: build complementary-role partner menu for brainstorm"
```

---

### Task 7: Grade / demote / drop in the orchestrator (`orchestrator.py`)

**Files:**
- Modify: `des_multi_agent/orchestrator.py:689-726` (deterministic-grounding block)
- Test: `tests/test_orchestrator_partner_reality.py`

**Interfaces:**
- Consumes: `ground_partner_reality` (Task 4); existing `llm_candidates`, `annotated_results`, `grounding_penalty_by_smiles`, `llm_warnings`, `_apply_review_penalties`.
- Produces: LLM proposals graded; `claim_verdicts` gains `PartnerVerdict`s; demoted candidates penalised; dropped candidates removed from `annotated_results`.

- [ ] **Step 1: Write the failing test**

This is integration-level; drive the grounding logic through a small helper to keep it unit-testable. Add a pure helper `_grade_partner_reality` to `orchestrator.py` and test it directly.

```python
# tests/test_orchestrator_partner_reality.py
from des_multi_agent.orchestrator import _grade_partner_reality


def test_grade_partitions_into_keep_demote_drop():
    # branched alkane: valid + sane (C/H only, MW ~142) but no H-bonding, and
    # very unlikely to appear in the known datasets → exercises the demote branch
    branched_alkane = "CCCC(CC)CCCC"  # 4-ethyloctane
    llm_smiles = {"NC(N)=O", branched_alkane, "OB(O)O", "OCC(O)CO"}
    verdicts, penalties, drops = _grade_partner_reality(
        component_a="OCC(O)CO",          # glycerol — strong H-bonder
        candidate_smiles=["NC(N)=O", branched_alkane, "OB(O)O", "OCC(O)CO"],
        llm_smiles=llm_smiles,
    )
    # urea (known) and glycerol (known) → keep, no penalty, not dropped
    assert "NC(N)=O" not in penalties and "NC(N)=O" not in drops
    # branched alkane → no complementarity with glycerol → demote
    assert penalties.get(branched_alkane, 0.0) == 0.25
    # boron → fails structural sanity → drop
    assert "OB(O)O" in drops
    # one verdict per candidate
    assert len(verdicts) == 4


def test_grade_skips_non_llm_smiles():
    verdicts, penalties, drops = _grade_partner_reality(
        component_a="CCO",
        candidate_smiles=["OB(O)O"],     # would drop if graded
        llm_smiles=set(),                # but it is not LLM-sourced
    )
    assert verdicts == [] and penalties == {} and drops == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator_partner_reality.py -v`
Expected: FAIL with `ImportError: cannot import name '_grade_partner_reality'`

- [ ] **Step 3: Write minimal implementation**

Add the pure helper to `des_multi_agent/orchestrator.py` (top-level, near the other `_`-helpers):

```python
def _grade_partner_reality(
    component_a: str,
    candidate_smiles: list[str],
    llm_smiles: set[str],
) -> tuple[list[object], dict[str, float], set[str]]:
    """Grade LLM-sourced partner proposals against reality.

    Returns (verdicts, penalty_by_smiles, drop_smiles). Non-LLM candidates are
    skipped (real by construction). Never raises.
    """
    from .chemistry.claim_grounding import ground_partner_reality

    verdicts: list[object] = []
    penalties: dict[str, float] = {}
    drops: set[str] = set()
    for smi in candidate_smiles:
        if smi not in llm_smiles:
            continue
        rv = ground_partner_reality(component_a, smi)
        verdicts.append(rv)
        if rv.disposition == "demote":
            penalties[smi] = max(penalties.get(smi, 0.0), rv.penalty)
        elif rv.disposition == "drop":
            drops.add(smi)
    return verdicts, penalties, drops
```

Then wire it into the deterministic-grounding block. Inside the existing `try:` (after the family-grounding loop, before the closing `except Exception`), add:

```python
        llm_smiles = set(family_by_smiles)  # family_by_smiles keyed by LLM smiles
        ordered_smiles = [item.result.curve.smiles_b for item in annotated_results]
        reality_verdicts, reality_penalties, drop_smiles = _grade_partner_reality(
            component_a, ordered_smiles, llm_smiles
        )
        claim_verdicts.extend(reality_verdicts)
        for smi, pen in reality_penalties.items():
            grounding_penalty_by_smiles[smi] = max(grounding_penalty_by_smiles.get(smi, 0.0), pen)
            llm_warnings.append(f"[REALITY] no H-bond complementarity — demoted: {smi}")
        for smi in drop_smiles:
            llm_warnings.append(f"[REALITY] not a real/sane molecule — dropped: {smi}")
```

Initialise `drop_smiles: set[str] = set()` just before the `try:` (next to where `claim_verdicts`/`grounding_penalty_by_smiles` are initialised) so a grounding-block exception cannot leave it undefined. Then, immediately after the `try/except` grounding block and before `if grounding_penalty_by_smiles:`, drop the flagged candidates:

```python
    if drop_smiles:
        annotated_results = [
            it for it in annotated_results
            if it.result.curve.smiles_b not in drop_smiles
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator_partner_reality.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py tests/test_orchestrator_partner_reality.py
git commit -m "feat: grade/demote/drop LLM partner proposals by reality verdict"
```

---

### Task 8: Render `PartnerVerdict` in the report (`reporting.py`)

**Files:**
- Modify: `des_multi_agent/reporting.py:294-302` (claim-verdict render loop)
- Test: `tests/test_reporting_partner_reality.py`

**Interfaces:**
- Consumes: `PartnerVerdict` (Task 4)
- Produces: report lines for `known` / `novel_plausible` / `novel_implausible` statuses, alongside the existing `verified`/`contradicted` rendering.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reporting_partner_reality.py
from des_multi_agent.chemistry.claim_grounding import PartnerVerdict
from des_multi_agent.reporting import format_report


def _verdict(status, detail, penalty, disp):
    return PartnerVerdict(claim="partner reality: X", status=status,
                          detail=detail, penalty=penalty, disposition=disp)


def test_report_renders_partner_statuses():
    verdicts = [
        _verdict("known", "known/attested compound", 0.0, "keep"),
        _verdict("novel_plausible", "novel; complementarity=strong", 0.0, "keep"),
        _verdict("novel_implausible", "no H-bond complementarity with component A", 0.25, "demote"),
    ]
    out = format_report([], claim_verdicts=verdicts)
    assert "✓ known" in out
    assert "◆ novel (plausible)" in out
    assert "✗ implausible — no H-bond complementarity with component A" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporting_partner_reality.py -v`
Expected: FAIL — the new status strings are absent from the report.

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/reporting.py`, extend the verdict-render loop (around line 296). Replace the existing `if/elif` with one that also handles the partner statuses:

```python
        for v in claim_verdicts:
            if v.status == "verified":
                rendered_verdicts.append(f"- ✓ verified | {v.claim}")
            elif v.status == "contradicted":
                rendered_verdicts.append(f"- ✗ contradicted — {v.detail} | {v.claim}")
            elif v.status == "known":
                rendered_verdicts.append(f"- ✓ known | {v.claim}")
            elif v.status == "novel_plausible":
                rendered_verdicts.append(f"- ◆ novel (plausible) | {v.claim}")
            elif v.status == "novel_implausible":
                rendered_verdicts.append(f"- ✗ implausible — {v.detail} | {v.claim}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reporting_partner_reality.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/reporting.py tests/test_reporting_partner_reality.py
git commit -m "feat: render partner reality verdicts in report"
```

---

### Task 9: Skill docs + full regression

**Files:**
- Modify: `.claude/skills/des-chemistry/SKILL.md`

**Interfaces:**
- Consumes: everything above. No new code.

- [ ] **Step 1: Document the layer**

Append a section to `.claude/skills/des-chemistry/SKILL.md`:

```markdown
## Partner reality anchoring (`chemistry/partner_registry.py` + `ground_partner_reality`)

Anchors DES-partner brainstorming to real, attested molecules and grades each
LLM proposal deterministically.

- `partner_registry.known_inchikeys()` / `is_known(smiles)` — membership in the
  curated registry ∪ experimental melting-point dataset (canonical InChIKey).
- `partner_registry.known_partner_menu(role, limit)` — role-tagged menu
  (curated entries first, then experimental compounds auto-tagged via
  `hbond_profile`) injected into the brainstorm prompt for the role
  complementary to component A.
- `partner_registry.structural_sanity(smiles)` — element whitelist
  {H,C,N,O,S,P,F,Cl,Br,I}, MW in (40, 400), no radicals.
- `claim_grounding.ground_partner_reality(component_a, smiles) -> PartnerVerdict`
  — the output-side entry point. Contract:
  - `known` / `novel_plausible` → keep
  - `novel_implausible` + no complementarity → demote (−0.25)
  - `novel_implausible` + bad structure / invalid → drop

Only LLM-sourced proposals are graded; heuristic/discovery candidates are real
by construction. Metal-ligand brainstorming is a planned parallel (not yet wired).
```

- [ ] **Step 2: Run the full suite**

Run: `pytest tests/ -q`
Expected: all green (previous count 706 + the new tests).

- [ ] **Step 3: Confirm deterministic example baselines are unchanged**

Run: `pytest tests/test_benchmarks_examples.py -v`
Expected: PASS — no-LLM example outputs did not move (menu injection only fires with an LLM; grading only touches LLM-sourced candidates).

- [ ] **Step 4: Smoke-check the menu has real content**

Run: `python -c "from des_multi_agent.chemistry.partner_registry import known_partner_menu; m=known_partner_menu('HBD',10); print(len(m)); [print(e.display_name, e.role) for e in m[:5]]"`
Expected: prints a non-zero count and recognisable HBDs (urea, glycerol, a carboxylic acid, etc.).

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/des-chemistry/SKILL.md
git commit -m "docs: document partner reality anchoring in des-chemistry skill"
```

---

## Self-Review Notes

- **Spec coverage:** §5.1 registry→Tasks 1-3; §5.2 verdict→Task 4; §5.3 prompt→Task 5; §5.4 menu build→Task 6; §5.5 orchestrator→Task 7; §5.6 reporting→Task 8; §5.7 docs→Task 9. All covered.
- **Determinism / LLM-agnostic:** every new entry point is pure + offline; Tasks 1-4, 8 tests run with no LLM.
- **Backward compatibility:** prompt param defaults `None` (Task 5 test asserts unchanged output); only LLM-sourced candidates graded (Task 7 test); deterministic baselines checked (Task 9 Step 3).
- **Type consistency:** `MenuEntry(smiles, display_name, role)` and `PartnerVerdict(claim, status, detail, penalty, disposition)` used identically across Tasks 3-8.
- **Known risk:** if a "novel" test molecule is actually present in `experimental.json` it grades `known` (still a keep); Task 4 Step 4 documents the verify-and-adjust path.
