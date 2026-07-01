# gpu4pyscf DFT Selectivity Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `--dft-validate` flag to the metal-selectivity workflow that runs free-ligand DFT (B3LYP-D3(BJ)/def2-SVP via gpu4pyscf) on LLM-nominated top candidates and applies a ±0.05 composite-score refinement to their ranking.

**Architecture:** Five sequential tasks build bottom-up: DFT pipeline module → selectivity adjustment module → LLM nomination → wire into the workflow → CLI + report. Each task is independently testable. Heavy GPU deps (gpu4pyscf, xtb) are only imported when `--dft-validate` is active; all test mocking happens at the internal function boundaries so no GPU is needed to run the test suite.

**Tech Stack:** Python 3.11+, RDKit (already installed), xtb binary (xTB GFN2), gpu4pyscf (B3LYP-D3(BJ)/def2-SVP), PySCF (gpu4pyscf dependency), pytest, unittest.mock.

## Global Constraints

- DFT activates **only** when `--dft-validate` is explicitly on the command line. No auto-trigger.
- Method fixed to B3LYP-D3(BJ)/def2-SVP. No other method/basis is the default.
- Ranking adjustment: ±0.05 maximum per candidate (half the H-bond bias magnitude of ±0.10).
- `SelectivityResult` stays `frozen=True`. DFT results live in `SelectivityScreenOutcome.dft_results: dict[str, DFTResult]`.
- `compute_dft_properties` wraps everything in `try/except Exception` — it never raises into the workflow.
- Optional deps (gpu4pyscf, xtb binary) are checked at CLI startup only when `--dft-validate` is active; failures emit `parser.error()`, not tracebacks.
- Existing 877 tests must pass unchanged after every task.

---

### Task 1: DFT pipeline module

Build `des_multi_agent/chemistry/dft_validator.py` — the only file that touches gpu4pyscf and xtb. Everything downstream uses `compute_dft_properties(smiles) -> DFTResult` and never imports gpu4pyscf directly.

**Files:**
- Create: `des_multi_agent/chemistry/dft_validator.py`
- Create: `tests/test_dft_validator.py`

**Interfaces:**
- Produces:
  - `DFTResult` dataclass with fields: `smiles: str`, `success: bool`, `homo_ev: float | None`, `homo_lumo_gap_ev: float | None`, `donor_charges: list[float]`, `geometry_method: str`, `dft_method: str`, `error: str | None`
  - `compute_dft_properties(smiles: str) -> DFTResult` — full pipeline, never raises
  - `_embed_mmff(smiles: str) -> rdkit.Chem.Mol` — internal, raises ValueError on failure
  - `_xtb_optimize(mol) -> tuple[list[str], np.ndarray]` — internal, raises RuntimeError on failure; returns (atom_symbols, coords_angstrom)
  - `_run_dft(symbols: list[str], coords_angstrom: np.ndarray) -> tuple[float, float, list[int], object]` — returns (homo_ev, homo_lumo_gap_ev, donor_indices, mf); raises RuntimeError if SCF does not converge
  - `_get_donor_charges(mf, donor_indices: list[int]) -> list[float]` — internal

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dft_validator.py
"""Unit tests for dft_validator. Heavy deps (gpu4pyscf, xtb) are mocked."""
from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from des_multi_agent.chemistry.dft_validator import (
    DFTResult, compute_dft_properties, _embed_mmff,
)


class TestDFTResult:
    def test_dataclass_fields(self):
        r = DFTResult(smiles="NCCN", success=True, homo_ev=-8.5, homo_lumo_gap_ev=5.1,
                      donor_charges=[-0.3, -0.3])
        assert r.smiles == "NCCN"
        assert r.success is True
        assert r.dft_method == "B3LYP-D3(BJ)/def2-SVP"

    def test_failure_result(self):
        r = DFTResult(smiles="X", success=False, error="bad")
        assert r.homo_ev is None
        assert r.donor_charges == []


class TestEmbedMMFF:
    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="RDKit cannot parse"):
            _embed_mmff("NOT_A_SMILES!!!")

    def test_valid_smiles_returns_mol(self):
        mol = _embed_mmff("NCCN")
        assert mol is not None
        assert mol.GetNumConformers() == 1


class TestComputeDFTProperties:
    def _make_mock_mf(self, n_electrons=8):
        mf = MagicMock()
        mf.converged = True
        # homo_idx = n_electrons//2 - 1 = 3
        # mo_energy[3] = -8.5/27.2114 Hartree (HOMO), [4] = -5.5/27.2114 (LUMO)
        HARTREE_TO_EV = 27.2114
        mo_energies = [-15.0, -12.0, -10.0, -8.5, -5.5, -3.0]
        mf.mo_energy = [e / HARTREE_TO_EV for e in mo_energies]
        mf.mulliken_pop.return_value = (None, np.array([-0.3, -0.1, -0.1, -0.3]))
        return mf

    def test_invalid_smiles_returns_failure_no_exception(self):
        result = compute_dft_properties("NOT_A_SMILES!!!")
        assert result.success is False
        assert result.error is not None
        assert result.homo_ev is None
        assert result.donor_charges == []

    def test_success_path_fields(self):
        mock_mol = MagicMock()
        mock_mf = self._make_mock_mf(n_electrons=8)
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["N", "C", "C", "N"], np.zeros((4, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   return_value=(-8.5, 5.1, [0, 3], mock_mf)):
            result = compute_dft_properties("NCCN")
        assert result.success is True
        assert abs(result.homo_ev - (-8.5)) < 0.01
        assert abs(result.homo_lumo_gap_ev - 5.1) < 0.01
        assert len(result.donor_charges) == 2   # donor_indices = [0, 3]

    def test_dft_scf_failure_returns_failure(self):
        mock_mol = MagicMock()
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["N", "C", "N"], np.zeros((3, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   side_effect=RuntimeError("SCF did not converge")):
            result = compute_dft_properties("NCN")
        assert result.success is False
        assert "SCF" in result.error

    def test_xtb_failure_returns_failure(self):
        mock_mol = MagicMock()
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   side_effect=RuntimeError("xtb optimization failed")):
            result = compute_dft_properties("NCCN")
        assert result.success is False

    def test_no_donor_atoms_gives_empty_charges(self):
        mock_mol = MagicMock()
        mock_mf = self._make_mock_mf(n_electrons=6)
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["C", "C", "C"], np.zeros((3, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   return_value=(-9.0, 4.5, [], mock_mf)):
            result = compute_dft_properties("CCC")
        assert result.success is True
        assert result.donor_charges == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/qshao/DES-Agent
pytest tests/test_dft_validator.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'des_multi_agent.chemistry.dft_validator'` or `ImportError`.

- [ ] **Step 3: Write `dft_validator.py`**

```python
# des_multi_agent/chemistry/dft_validator.py
"""Free-ligand DFT validation pipeline using gpu4pyscf.

Entry point: compute_dft_properties(smiles) -> DFTResult.
Never raises — all failures return DFTResult(success=False, error=...).
"""
from __future__ import annotations

from dataclasses import dataclass, field


_DONOR_SYMBOLS: frozenset[str] = frozenset({"N", "O", "P", "S"})
_HARTREE_TO_EV: float = 27.2114


@dataclass
class DFTResult:
    smiles: str
    success: bool
    homo_ev: float | None = None
    homo_lumo_gap_ev: float | None = None
    donor_charges: list[float] = field(default_factory=list)
    geometry_method: str = "xtb"
    dft_method: str = "B3LYP-D3(BJ)/def2-SVP"
    error: str | None = None


def _embed_mmff(smiles: str):
    """SMILES → RDKit MMFF94 3D Mol with H. Raises ValueError on failure."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise ValueError(f"3D embedding failed for {smiles!r}")
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    if ff is not None:
        ff.Minimize()
    return mol


def _xtb_optimize(mol) -> tuple[list[str], "np.ndarray"]:
    """RDKit Mol → xTB GFN2-optimized geometry via xtb subprocess.

    Returns (atom_symbols, coords_angstrom). Raises RuntimeError on failure.
    """
    import numpy as np
    import os
    import subprocess
    import tempfile

    conf = mol.GetConformer()
    positions = conf.GetPositions()  # Å
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    xyz_lines = [str(len(symbols)), "xtb input"]
    for sym, pos in zip(symbols, positions):
        xyz_lines.append(f"{sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_in = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_in, "w") as fh:
            fh.write("\n".join(xyz_lines))
        proc = subprocess.run(
            ["xtb", xyz_in, "--opt", "--gfn", "2", "--silent"],
            cwd=tmpdir, capture_output=True, text=True, timeout=120,
        )
        opt_xyz = os.path.join(tmpdir, "xtbopt.xyz")
        if proc.returncode != 0 or not os.path.exists(opt_xyz):
            raise RuntimeError(f"xtb optimization failed: {proc.stderr[:300]}")
        with open(opt_xyz) as fh:
            lines = fh.readlines()

    opt_syms, opt_pos = [], []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            opt_syms.append(parts[0])
            opt_pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return opt_syms, np.array(opt_pos)


def _run_dft(
    symbols: list[str], coords_angstrom: "np.ndarray"
) -> tuple[float, float, list[int], object]:
    """Run B3LYP-D3(BJ)/def2-SVP single-point. Returns (homo_ev, gap_ev, donor_indices, mf).

    Raises RuntimeError if SCF does not converge.
    """
    from pyscf import gto
    from gpu4pyscf import dft as gpu_dft

    atom_list = [(sym, tuple(pos)) for sym, pos in zip(symbols, coords_angstrom)]
    mol = gto.Mole(atom=atom_list, basis="def2-svp", charge=0, spin=0, verbose=0)
    mol.build()

    mf = gpu_dft.RKS(mol)
    mf.xc = "B3LYP"
    mf.disp = "d3bj"
    mf.kernel()

    if not mf.converged:
        raise RuntimeError("SCF did not converge")

    homo_idx = mol.nelectron // 2 - 1
    homo_ev = float(mf.mo_energy[homo_idx]) * _HARTREE_TO_EV
    lumo_ev = float(mf.mo_energy[homo_idx + 1]) * _HARTREE_TO_EV
    gap_ev = lumo_ev - homo_ev

    donor_indices = [i for i, sym in enumerate(symbols) if sym in _DONOR_SYMBOLS]
    return homo_ev, gap_ev, donor_indices, mf


def _get_donor_charges(mf, donor_indices: list[int]) -> list[float]:
    """Extract per-donor-atom charges from a converged RKS object.

    Uses Mulliken population via mf.mulliken_pop() (second return value = atom charges).
    For Löwdin charges (more basis-set-stable) try mf.mulliken_pop_meta_lowdin_ao()
    instead — verify the method name against your installed PySCF version.
    """
    _, atom_charges = mf.mulliken_pop(verbose=0)
    return [float(atom_charges[i]) for i in donor_indices]


def compute_dft_properties(smiles: str) -> DFTResult:
    """Full pipeline: SMILES → DFTResult. Never raises."""
    try:
        mol = _embed_mmff(smiles)
        symbols, coords = _xtb_optimize(mol)
        homo_ev, gap_ev, donor_indices, mf = _run_dft(symbols, coords)
        donor_charges = _get_donor_charges(mf, donor_indices)
        return DFTResult(
            smiles=smiles,
            success=True,
            homo_ev=homo_ev,
            homo_lumo_gap_ev=gap_ev,
            donor_charges=donor_charges,
        )
    except Exception as exc:
        return DFTResult(smiles=smiles, success=False, error=str(exc))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dft_validator.py -v
```
Expected: all 9 tests PASS. (The `test_valid_smiles_returns_mol` test uses real RDKit — no GPU needed since it only calls `_embed_mmff`.)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemistry/dft_validator.py tests/test_dft_validator.py
git commit -m "feat: add DFT pipeline module (dft_validator.py) with gpu4pyscf B3LYP-D3(BJ)/def2-SVP"
```

---

### Task 2: DFT selectivity adjustment

Build `des_multi_agent/chemistry/dft_selectivity.py` — converts a `DFTResult` HOMO energy into a ±0.05 composite-score nudge using existing HSAB metal softness data.

**Files:**
- Create: `des_multi_agent/chemistry/dft_selectivity.py`
- Modify: `tests/test_dft_validator.py` (append new test class)

**Interfaces:**
- Consumes: `DFTResult` from Task 1; `_metal_softness(metal_ion: str) -> float` from `des_multi_agent.chemistry.stability_rules` (already exists — returns 0..1 float where 0=hard, 1=soft, sourced from `_METAL_IDENTITY` table).
- Produces: `dft_selectivity_adjustment(dft_result: DFTResult, target_metal: str, competitor_metal: str) -> float` — returns value in [−0.05, +0.05]. Positive → DFT supports target selectivity; negative → competitor preferred.

- [ ] **Step 1: Append failing tests to `tests/test_dft_validator.py`**

```python
# append to tests/test_dft_validator.py

from des_multi_agent.chemistry.dft_selectivity import dft_selectivity_adjustment


class TestDFTSelectivityAdjustment:
    def _result(self, homo_ev: float) -> DFTResult:
        return DFTResult(smiles="X", success=True, homo_ev=homo_ev, donor_charges=[])

    def test_returns_zero_on_failure(self):
        r = DFTResult(smiles="X", success=False, error="fail")
        assert dft_selectivity_adjustment(r, "Cu2+", "Zn2+") == 0.0

    def test_returns_zero_when_homo_none(self):
        r = DFTResult(smiles="X", success=True, homo_ev=None, donor_charges=[])
        assert dft_selectivity_adjustment(r, "Cu2+", "Zn2+") == 0.0

    def test_adjustment_within_bounds(self):
        # Any HOMO energy must produce adjustment in [-0.05, +0.05]
        for homo in [-12.0, -9.5, -8.5, -7.5, -5.0]:
            adj = dft_selectivity_adjustment(self._result(homo), "Cu2+", "Zn2+")
            assert -0.05 <= adj <= 0.05, f"homo={homo} gave {adj}"

    def test_hard_donor_prefers_hard_metal(self):
        # HOMO ≤ −9.5 eV → hard donor (softness ≈ 0)
        # Hard metal (Mg2+ softness=0) vs soft metal (Cd2+ softness=1)
        # → adjustment should be positive (matches hard target)
        from des_multi_agent.chemistry.stability_rules import _metal_softness
        # Find a metal pair where one is harder than the other
        # Cu2+ softness from _METAL_IDENTITY ≈ 0.5–0.7 (borderline)
        # Use Cu2+ as target, Zn2+ as competitor — both borderline but Cu slightly softer
        hard_donor = self._result(-10.0)   # very hard donor
        soft_donor = self._result(-7.0)    # very soft donor
        adj_hard = dft_selectivity_adjustment(hard_donor, "Cu2+", "Zn2+")
        adj_soft = dft_selectivity_adjustment(soft_donor, "Cu2+", "Zn2+")
        # The two adjustments should have opposite signs or at least differ
        assert adj_hard != adj_soft

    def test_symmetric_metals_gives_near_zero(self):
        # Same metal for target and competitor → softness delta = 0 → adjustment ≈ 0
        adj = dft_selectivity_adjustment(self._result(-8.5), "Cu2+", "Cu2+")
        assert abs(adj) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_dft_validator.py::TestDFTSelectivityAdjustment -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'dft_selectivity_adjustment'`.

- [ ] **Step 3: Write `dft_selectivity.py`**

```python
# des_multi_agent/chemistry/dft_selectivity.py
"""Converts DFT free-ligand HOMO energy to a metal-selectivity ranking adjustment.

Entry point: dft_selectivity_adjustment(dft_result, target_metal, competitor_metal) -> float.
Returns a value in [-0.05, +0.05] to add to composite_score.
"""
from __future__ import annotations

from .dft_validator import DFTResult

# HOMO energy calibration anchors (eV) → donor softness in [0, 1].
# −9.5 eV ≈ hard donors (carboxylate O, amide N);
# −7.5 eV ≈ soft donors (thiolate S, phosphine P).
_HOMO_HARD_EV: float = -9.5   # softness = 0.0
_HOMO_SOFT_EV: float = -7.5   # softness = 1.0

_MAX_ADJ: float = 0.05         # half the H-bond bias magnitude (±0.10)


def _homo_to_softness(homo_ev: float) -> float:
    """Linearly map HOMO energy to donor softness in [0, 1]."""
    t = (homo_ev - _HOMO_HARD_EV) / (_HOMO_SOFT_EV - _HOMO_HARD_EV)
    return max(0.0, min(1.0, t))


def dft_selectivity_adjustment(
    dft_result: DFTResult,
    target_metal: str,
    competitor_metal: str,
) -> float:
    """±0.05 composite-score nudge based on HOMO energy vs HSAB metal softness.

    Positive → ligand HOMO profile matches target better than competitor.
    Returns 0.0 if DFT did not succeed or HOMO is unavailable.
    """
    if not dft_result.success or dft_result.homo_ev is None:
        return 0.0

    from ..chemistry.stability_rules import _metal_softness

    s_target = _metal_softness(target_metal)
    s_comp = _metal_softness(competitor_metal)

    if s_target == s_comp:
        return 0.0

    s_ligand = _homo_to_softness(dft_result.homo_ev)

    # delta > 0 → ligand softness closer to competitor; negate so target-match = positive
    delta = abs(s_ligand - s_target) - abs(s_ligand - s_comp)
    scale = abs(s_target - s_comp)           # normalise by the metal-pair separation
    raw = -delta / scale * _MAX_ADJ if scale > 0 else 0.0
    return max(-_MAX_ADJ, min(_MAX_ADJ, raw))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dft_validator.py -v
```
Expected: all tests (Task 1 + Task 2) PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/chemistry/dft_selectivity.py tests/test_dft_validator.py
git commit -m "feat: add DFT selectivity adjustment (HOMO→HSAB nudge ±0.05)"
```

---

### Task 3: LLM nomination prompt + `nominate_for_dft`

Add `dft_nomination_prompt` to `llm/prompts.py` and `LLMProvider.nominate_for_dft` to `llm/base.py`. When no LLM is configured, a standalone fallback function is also provided so the workflow can call it unconditionally.

**Files:**
- Modify: `des_multi_agent/llm/prompts.py`
- Modify: `des_multi_agent/llm/base.py`
- Create: `tests/test_dft_nomination_prompt.py`

**Interfaces:**
- Consumes: `SelectivityResult` from `metal_binding_selectivity.py` (fields used: `ligand_smiles: str`, `delta_log_k: float`, `composite_score: float`).
- Produces:
  - `dft_nomination_prompt(candidates: list, target_metal: str, competitor_metal: str, top_n: int = 3) -> str` in `prompts.py`
  - `LLMProvider.nominate_for_dft(candidates: list, target_metal: str, competitor_metal: str, top_n: int = 3) -> list[str]` — returns SMILES list, max length `top_n`
  - `nominate_for_dft_fallback(candidates: list, top_n: int) -> list[str]` — standalone (no LLM), returns top-`top_n` SMILES by `composite_score`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dft_nomination_prompt.py
"""Tests for DFT nomination prompt and fallback nomination."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from des_multi_agent.llm.prompts import dft_nomination_prompt
from des_multi_agent.llm.base import nominate_for_dft_fallback


def _make_candidate(smiles: str, delta: float, score: float):
    c = MagicMock()
    c.ligand_smiles = smiles
    c.delta_log_k = delta
    c.composite_score = score
    return c


CANDIDATES = [
    _make_candidate("NCCN", 1.5, 0.90),
    _make_candidate("NCC(=O)O", 1.1, 0.80),
    _make_candidate("c1ccncc1", 0.4, 0.65),
    _make_candidate("NCCCCN", 0.2, 0.55),
]


class TestDFTNominationPrompt:
    def test_prompt_contains_target_metal(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        assert "Cu2+" in p

    def test_prompt_contains_competitor_metal(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        assert "Zn2+" in p

    def test_prompt_contains_all_smiles(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        for c in CANDIDATES:
            assert c.ligand_smiles in p

    def test_prompt_contains_delta_log_k(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        assert "1.50" in p or "ΔlogK" in p

    def test_top_n_in_prompt(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+", top_n=2)
        assert "1–2" in p or "2" in p

    def test_json_instruction_in_prompt(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        assert "JSON" in p


class TestNominateForDFTFallback:
    def test_returns_top_n_by_score(self):
        result = nominate_for_dft_fallback(CANDIDATES, top_n=2)
        assert result == ["NCCN", "NCC(=O)O"]

    def test_respects_top_n_cap(self):
        result = nominate_for_dft_fallback(CANDIDATES, top_n=1)
        assert result == ["NCCN"]

    def test_returns_all_if_fewer_than_n(self):
        result = nominate_for_dft_fallback(CANDIDATES[:2], top_n=5)
        assert len(result) == 2

    def test_empty_candidates_returns_empty(self):
        result = nominate_for_dft_fallback([], top_n=3)
        assert result == []
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_dft_nomination_prompt.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'dft_nomination_prompt'`.

- [ ] **Step 3: Add `dft_nomination_prompt` to `des_multi_agent/llm/prompts.py`**

Append at the end of the file:

```python
def dft_nomination_prompt(
    candidates: list,
    target_metal: str,
    competitor_metal: str,
    top_n: int = 3,
) -> str:
    """Prompt asking the LLM to nominate candidates for DFT validation."""
    rows = []
    for i, r in enumerate(candidates, 1):
        rows.append(
            f"  {i}. {r.ligand_smiles}  ΔlogK={r.delta_log_k:.2f}  score={r.composite_score:.2f}"
        )
    table = "\n".join(rows)
    return (
        f"You are helping prioritize ligands for DFT validation.\n"
        f"Target metal: {target_metal}. Competitor: {competitor_metal}.\n\n"
        f"Top candidates by predicted selectivity (ΔlogK):\n{table}\n\n"
        f"Select 1–{top_n} candidates most worth DFT validation. Prefer:\n"
        f"- Ligands where HSAB ambiguity makes the rule-based prediction uncertain\n"
        f"- Borderline ΔlogK values (small positive) where DFT tiebreaking matters most\n"
        f"- Structurally diverse nominations over similar analogues\n\n"
        f'Return ONLY a JSON list of SMILES strings. Example: ["SMILES1", "SMILES2"]\n'
    )
```

- [ ] **Step 4: Add `nominate_for_dft_fallback` to `des_multi_agent/llm/base.py`**

Append the standalone fallback at module level (outside the `LLMProvider` class), near the top after the existing imports:

```python
def nominate_for_dft_fallback(candidates: list, top_n: int) -> list[str]:
    """Return top-top_n SMILES by composite_score when no LLM is available."""
    sorted_cands = sorted(candidates, key=lambda r: r.composite_score, reverse=True)
    return [r.ligand_smiles for r in sorted_cands[:top_n]]
```

Then add `nominate_for_dft` as a method on `LLMProvider` (add after the `brainstorm_ligands_selectivity` method, around line 235):

```python
    def nominate_for_dft(
        self,
        candidates: list,
        target_metal: str,
        competitor_metal: str,
        top_n: int = 3,
    ) -> list[str]:
        """Return SMILES of candidates nominated for DFT validation.

        Parses the LLM's JSON list response; falls back to top-N by composite_score
        on any parse failure.
        """
        import json
        from .prompts import dft_nomination_prompt

        valid_smiles = {r.ligand_smiles for r in candidates}
        raw = self._request(dft_nomination_prompt(candidates, target_metal, competitor_metal, top_n))
        try:
            nominated = json.loads(raw.strip())
            if not isinstance(nominated, list):
                raise ValueError("response is not a JSON list")
            filtered = [s for s in nominated if isinstance(s, str) and s in valid_smiles]
            if filtered:
                return filtered[:top_n]
        except Exception:
            pass
        return nominate_for_dft_fallback(candidates, top_n)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_dft_nomination_prompt.py -v
```
Expected: all 10 tests PASS.

- [ ] **Step 6: Run full suite to verify no regressions**

```bash
pytest tests/ -q --ignore=tests/test_benchmarks_examples.py 2>&1 | tail -5
```
Expected: same pass count as before (877+), 0 new failures.

- [ ] **Step 7: Commit**

```bash
git add des_multi_agent/llm/prompts.py des_multi_agent/llm/base.py tests/test_dft_nomination_prompt.py
git commit -m "feat: add DFT nomination prompt and nominate_for_dft LLM method"
```

---

### Task 4: Wire DFT stage into metal-selectivity workflow

Modify `SelectivityScreenOutcome` to carry `dft_results`, and add the post-loop DFT stage to `run_metal_selectivity_screen`.

**Files:**
- Modify: `des_multi_agent/workflows/metal_binding_selectivity.py`
- Create: `tests/test_dft_integration.py`

**Interfaces:**
- Consumes:
  - `compute_dft_properties(smiles: str) -> DFTResult` from Task 1
  - `dft_selectivity_adjustment(dft_result, target_metal, competitor_metal) -> float` from Task 2
  - `LLMProvider.nominate_for_dft(...)` / `nominate_for_dft_fallback(...)` from Task 3
- Produces:
  - `SelectivityScreenOutcome.dft_results: dict` — `dict[str, DFTResult]`, default `{}`
  - `run_metal_selectivity_screen(..., dft_validate: bool = False, dft_top_n: int = 3) -> SelectivityScreenOutcome`

- [ ] **Step 1: Write failing integration tests**

```python
# tests/test_dft_integration.py
"""Integration tests for the DFT stage wired into run_metal_selectivity_screen.

All heavy deps are mocked: compute_dft_properties returns a controlled DFTResult.
"""
from __future__ import annotations
import dataclasses
import pytest
from unittest.mock import patch, MagicMock
from des_multi_agent.chemistry.dft_validator import DFTResult
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityScreenOutcome, run_metal_selectivity_screen,
)


def _fake_screen(*args, **kwargs):
    """Patch target: replaces the entire inner screening loop with two fake results."""
    pass   # not used directly — we patch compute_dft_properties instead


FAKE_DFT_SUCCESS = DFTResult(
    smiles="NCCN", success=True, homo_ev=-8.5, homo_lumo_gap_ev=5.1, donor_charges=[-0.3, -0.3]
)
FAKE_DFT_FAIL = DFTResult(smiles="c1ccncc1", success=False, error="no donor atoms")


class TestSelectivityScreenOutcomeHasDFTField:
    def test_dft_results_field_exists_and_defaults_empty(self):
        outcome = SelectivityScreenOutcome(
            target_metal="Cu2+", competitor_metal="Zn2+",
            results=[], n_screened=0, n_cycles=1,
        )
        assert hasattr(outcome, "dft_results")
        assert outcome.dft_results == {}


class TestRunMetalSelectivityScreenDFTParam:
    def test_accepts_dft_validate_param(self):
        """run_metal_selectivity_screen must accept dft_validate and dft_top_n without error."""
        import inspect
        sig = inspect.signature(run_metal_selectivity_screen)
        assert "dft_validate" in sig.parameters
        assert "dft_top_n" in sig.parameters
        assert sig.parameters["dft_validate"].default is False
        assert sig.parameters["dft_top_n"].default == 3


class TestDFTStageWiring:
    def _run_with_mock_dft(self, dft_results_map: dict, dft_top_n: int = 1):
        """Run selectivity screen with mocked DFT results.

        Uses the mock stability model to avoid needing a real checkpoint.
        """
        def fake_compute(smiles):
            return dft_results_map.get(smiles, DFTResult(smiles=smiles, success=False, error="not mocked"))

        with patch("des_multi_agent.chemistry.dft_validator.compute_dft_properties",
                   side_effect=fake_compute), \
             patch("des_multi_agent.workflows.metal_binding_selectivity.predict_log_k",
                   return_value=5.0):
            outcome = run_metal_selectivity_screen(
                target_metal="Cu2+",
                competitor_metal="Zn2+",
                n=3,
                model_path=None,
                llm_provider=None,
                n_cycles=1,
                dft_validate=True,
                dft_top_n=dft_top_n,
            )
        return outcome

    def test_dft_results_populated_on_validate(self):
        outcome = self._run_with_mock_dft({"NCCN": FAKE_DFT_SUCCESS})
        assert isinstance(outcome.dft_results, dict)

    def test_dft_false_leaves_dft_results_empty(self):
        with patch("des_multi_agent.predictors.stability_constants.predict_log_k",
                   return_value=5.0):
            outcome = run_metal_selectivity_screen(
                target_metal="Cu2+", competitor_metal="Zn2+",
                n=3, model_path=None, llm_provider=None,
                n_cycles=1, dft_validate=False,
            )
        assert outcome.dft_results == {}

    def test_dft_failure_adds_warning(self):
        outcome = self._run_with_mock_dft({"NCCN": FAKE_DFT_FAIL})
        dft_warnings = [w for w in outcome.warnings if "[DFT]" in w]
        assert any("Warning" in w for w in dft_warnings)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_dft_integration.py -v 2>&1 | head -30
```
Expected: `TestSelectivityScreenOutcomeHasDFTField` fails (field missing), `TestRunMetalSelectivityScreenDFTParam` fails (param missing).

- [ ] **Step 3: Add `dft_results` field to `SelectivityScreenOutcome`**

In `des_multi_agent/workflows/metal_binding_selectivity.py`, modify the `SelectivityScreenOutcome` dataclass (around line 47–58):

```python
@dataclass
class SelectivityScreenOutcome:
    target_metal: str
    competitor_metal: str
    results: list[SelectivityResult]
    n_screened: int
    n_cycles: int
    llm_brainstorm: list[CandidateBrainstorm] = field(default_factory=list)
    llm_candidate_reviews: list[CandidateReview] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    claim_verdicts: list[object] = field(default_factory=list)
    trajectory: object = None   # SearchTrajectory | None
    dft_results: dict = field(default_factory=dict)   # dict[str, DFTResult]
```

- [ ] **Step 4: Add `dft_validate` and `dft_top_n` params and DFT stage to `run_metal_selectivity_screen`**

**4a.** Update the function signature (line 223):

```python
def run_metal_selectivity_screen(
    target_metal: str,
    competitor_metal: str,
    n: int = 20,
    model_path=None,
    llm_provider=None,
    constraints: dict | None = None,
    n_cycles: int = 1,
    w_affinity: float = 0.5,
    w_selectivity: float = 0.5,
    des_compatible_hints: list[str] | None = None,
    des_incompatible_hints: list[str] | None = None,
    stability_rule_weight: float = 0.0,
    binding_pH: float = 7.0,
    dft_validate: bool = False,
    dft_top_n: int = 3,
) -> SelectivityScreenOutcome:
```

**4b.** Add the DFT stage between the end of the screening loop and the `sel_trajectory = SearchTrajectory(...)` call. The loop ends with `prev_cycle_results = list(cumulative_results)` (inside the for loop). After the loop exits, insert (before `sel_trajectory`):

```python
    # --- Optional DFT validation stage (post-loop) ---
    dft_results_map: dict = {}
    if dft_validate and cumulative_results:
        from ..chemistry.dft_validator import compute_dft_properties as _dft
        from ..chemistry.dft_selectivity import dft_selectivity_adjustment as _dft_adj
        from ..llm.base import nominate_for_dft_fallback as _dft_fallback

        top_k_pool = cumulative_results[: min(dft_top_n * 2, len(cumulative_results))]
        if llm_provider is not None and hasattr(llm_provider, "nominate_for_dft"):
            try:
                nominated_smiles = llm_provider.nominate_for_dft(
                    top_k_pool, target_metal, competitor_metal, dft_top_n
                )
            except Exception as exc:
                all_warnings.append(
                    f"[DFT] LLM nomination failed, using top-{dft_top_n} by score: {exc}"
                )
                nominated_smiles = _dft_fallback(top_k_pool, dft_top_n)
        else:
            nominated_smiles = _dft_fallback(top_k_pool, dft_top_n)

        for smi in nominated_smiles:
            res = _dft(smi)
            dft_results_map[smi] = res
            if not res.success:
                all_warnings.append(f"[DFT] Warning: skipping {smi[:40]!r} — {res.error}")

        successful = [r for r in dft_results_map.values() if r.success]
        if nominated_smiles and not successful:
            all_warnings.append(
                "[DFT] Warning: all DFT computations failed — rule-based ranking used"
            )
        elif successful:
            import dataclasses as _dc
            updated = []
            for r in cumulative_results:
                dft_res = dft_results_map.get(r.ligand_smiles)
                if dft_res and dft_res.success:
                    adj = _dft_adj(dft_res, target_metal, competitor_metal)
                    r = _dc.replace(r, composite_score=r.composite_score + adj)
                updated.append(r)
            cumulative_results = sorted(updated, key=lambda r: r.composite_score, reverse=True)
    # --- End DFT stage ---
```

**4c.** Add `dft_results=dft_results_map` to the `return SelectivityScreenOutcome(...)` call (line 495):

```python
    return SelectivityScreenOutcome(
        target_metal=target_metal,
        competitor_metal=competitor_metal,
        results=cumulative_results,
        n_screened=len(seen_smiles),
        n_cycles=n_cycles,
        llm_brainstorm=all_brainstorm,
        llm_candidate_reviews=all_reviews,
        warnings=all_warnings,
        claim_verdicts=all_sel_verdicts + all_coord_verdicts,
        trajectory=sel_trajectory,
        dft_results=dft_results_map,
    )
```

- [ ] **Step 5: Run integration tests to verify they pass**

```bash
pytest tests/test_dft_integration.py -v
```
Expected: all tests PASS. (The `TestDFTStageWiring` tests mock `compute_dft_properties` and `predict_log_k`, so no real GPU or checkpoint needed.)

- [ ] **Step 6: Run full suite for regressions**

```bash
pytest tests/ -q --ignore=tests/test_benchmarks_examples.py 2>&1 | tail -5
```
Expected: same pass count, 0 new failures.

- [ ] **Step 7: Commit**

```bash
git add des_multi_agent/workflows/metal_binding_selectivity.py tests/test_dft_integration.py
git commit -m "feat: wire DFT validation stage into run_metal_selectivity_screen"
```

---

### Task 5: CLI flags + startup dependency check + report DFT section

Add `--dft-validate` and `--dft-top-n` to the CLI, dep-check at startup, forward to the workflow, and extend `format_metal_selectivity_report` to show DFT columns when present.

**Files:**
- Modify: `des_multi_agent/cli.py`
- Modify: `des_multi_agent/reporting.py`
- Create: `tests/test_dft_cli_report.py`

**Interfaces:**
- Consumes: `run_metal_selectivity_screen(..., dft_validate: bool, dft_top_n: int)` from Task 4; `SelectivityScreenOutcome.dft_results` from Task 4.
- Produces: CLI flags `--dft-validate` (store_true) and `--dft-top-n INT`; `format_metal_selectivity_report` returns DFT columns and DFT summary block when `outcome.dft_results` is non-empty.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dft_cli_report.py
"""Tests for CLI DFT flags and report rendering with DFT results."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from des_multi_agent.chemistry.dft_validator import DFTResult
from des_multi_agent.workflows.metal_binding_selectivity import SelectivityResult, SelectivityScreenOutcome
from des_multi_agent.reporting import format_metal_selectivity_report


def _make_outcome(dft_results: dict | None = None) -> SelectivityScreenOutcome:
    r = SelectivityResult(
        ligand_smiles="NCCN", log_k_target=5.5, log_k_competitor=4.0,
        delta_log_k=1.5, composite_score=0.85, source="test", source_id="", rationale="good",
    )
    return SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+",
        results=[r], n_screened=1, n_cycles=1,
        dft_results=dft_results or {},
    )


class TestReportWithoutDFT:
    def test_no_dft_columns_when_no_dft_results(self):
        report = format_metal_selectivity_report(_make_outcome())
        assert "dft_homo_ev" not in report
        assert "DFT validation" not in report

    def test_report_renders_without_error(self):
        report = format_metal_selectivity_report(_make_outcome())
        assert "Cu2+" in report
        assert "NCCN" in report


class TestReportWithDFT:
    def _outcome_with_dft(self):
        dft = DFTResult(smiles="NCCN", success=True, homo_ev=-8.51,
                        homo_lumo_gap_ev=5.12, donor_charges=[-0.31, -0.29])
        return _make_outcome({"NCCN": dft})

    def test_homo_ev_appears_in_report(self):
        report = format_metal_selectivity_report(self._outcome_with_dft())
        assert "-8.51" in report

    def test_dft_summary_block_present(self):
        report = format_metal_selectivity_report(self._outcome_with_dft())
        assert "DFT validation" in report or "B3LYP" in report

    def test_failed_dft_shows_dash_not_crash(self):
        dft_fail = DFTResult(smiles="NCCN", success=False, error="SCF fail")
        outcome = _make_outcome({"NCCN": dft_fail})
        report = format_metal_selectivity_report(outcome)
        assert "—" in report or "FAILED" in report

    def test_non_nominated_row_shows_dash(self):
        r2 = SelectivityResult(
            ligand_smiles="NCC(=O)O", log_k_target=4.0, log_k_competitor=3.5,
            delta_log_k=0.5, composite_score=0.70, source="test", source_id="", rationale="ok",
        )
        dft = DFTResult(smiles="NCCN", success=True, homo_ev=-8.5,
                        homo_lumo_gap_ev=5.0, donor_charges=[-0.3])
        outcome = _make_outcome({"NCCN": dft})
        outcome = SelectivityScreenOutcome(
            target_metal="Cu2+", competitor_metal="Zn2+",
            results=[outcome.results[0], r2], n_screened=2, n_cycles=1,
            dft_results={"NCCN": dft},
        )
        report = format_metal_selectivity_report(outcome)
        assert "—" in report   # NCC(=O)O has no DFT result


class TestCLIDFTFlags:
    def test_dft_validate_flag_exists(self):
        from des_multi_agent.cli import build_parser
        parser = build_parser()
        # parse a valid metal-selectivity command with --dft-validate
        args = parser.parse_args([
            "--workflow", "metal-selectivity",
            "--target-metal-ion", "Cu2+",
            "--competitor-metal-ion", "Zn2+",
            "--dft-validate",
            "--dft-top-n", "2",
        ])
        assert args.dft_validate is True
        assert args.dft_top_n == 2

    def test_dft_top_n_default_is_3(self):
        from des_multi_agent.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--workflow", "metal-selectivity",
            "--target-metal-ion", "Cu2+",
            "--competitor-metal-ion", "Zn2+",
        ])
        assert args.dft_top_n == 3
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_dft_cli_report.py -v 2>&1 | head -30
```
Expected: `TestCLIDFTFlags` fails (`unrecognized arguments: --dft-validate`); `TestReportWithDFT` may fail or produce wrong output.

- [ ] **Step 3: Add CLI flags to `des_multi_agent/cli.py`**

Find the metal-selectivity argument group in `build_parser()`. The `--affinity-weight` and `--selectivity-weight` flags are around line 180. Add immediately after them:

```python
    parser.add_argument(
        "--dft-validate",
        action="store_true",
        default=False,
        help="Enable DFT validation of top candidates via gpu4pyscf B3LYP-D3(BJ)/def2-SVP "
             "(requires gpu4pyscf and xtb binary; metal-selectivity workflow only)",
    )
    parser.add_argument(
        "--dft-top-n",
        type=int,
        default=3,
        dest="dft_top_n",
        help="Number of candidates to submit for DFT validation (default: 3)",
    )
```

- [ ] **Step 4: Add dep check and forward DFT flags in `cli.py`**

In the `if args.workflow == "metal-selectivity":` block (around line 749), add the dep check before `run_metal_selectivity_screen` is called, and forward the new params:

```python
    if args.workflow == "metal-selectivity":
        if not args.target_metal_ion or not args.competitor_metal_ion:
            parser.error(
                "metal-selectivity workflow requires --target-metal-ion and --competitor-metal-ion. "
                "Example: --target-metal-ion Cu2+ --competitor-metal-ion Zn2+"
            )
        # Startup dep check — only when DFT is requested
        if getattr(args, "dft_validate", False):
            try:
                import gpu4pyscf  # noqa: F401
            except ImportError:
                parser.error(
                    "--dft-validate requires 'gpu4pyscf'. Install with: pip install gpu4pyscf"
                )
            import subprocess as _sp
            try:
                _sp.run(["xtb", "--version"], capture_output=True, timeout=5, check=True)
            except (FileNotFoundError, _sp.CalledProcessError, _sp.TimeoutExpired):
                parser.error(
                    "--dft-validate requires the 'xtb' binary. "
                    "Install from: https://github.com/grimme-lab/xtb/releases"
                )

        from .llm.factory import build_llm_provider as _build_llm_provider
        llm_provider_sel = _build_llm_provider(llm_cfg) if llm_cfg is not None else None
        sel_outcome = run_metal_selectivity_screen(
            target_metal=args.target_metal_ion,
            competitor_metal=args.competitor_metal_ion,
            n=getattr(args, "n", 20),
            model_path=args.stability_constant_model_path,
            llm_provider=llm_provider_sel,
            n_cycles=getattr(args, "n_cycles", 1),
            w_affinity=args.affinity_weight,
            w_selectivity=args.selectivity_weight,
            dft_validate=getattr(args, "dft_validate", False),
            dft_top_n=getattr(args, "dft_top_n", 3),
        )
        print(format_metal_selectivity_report(sel_outcome))
        _print_summary("metal-selectivity", sel_outcome)
        _emit_trajectory(getattr(sel_outcome, "trajectory", None), getattr(args, "output_dir", None))
        return
```

- [ ] **Step 5: Extend `format_metal_selectivity_report` in `des_multi_agent/reporting.py`**

Replace the function body (lines 536–590) with:

```python
def format_metal_selectivity_report(outcome) -> str:
    """Render a ranked-candidate report for a SelectivityScreenOutcome."""
    results = outcome.results
    dft_results = getattr(outcome, "dft_results", {})
    has_dft = bool(dft_results)

    top = results[0] if results else None
    if top:
        top_str = (
            f"{top.ligand_smiles} — score={top.composite_score:.2f} "
            f"(ΔlogK={top.delta_log_k:.2f}, logK({outcome.target_metal})={top.log_k_target:.2f})"
        )
    else:
        top_str = "none"

    col_header = "ligand | log_k_target | log_k_competitor | delta_log_k | score"
    if has_dft:
        col_header += " | dft_homo_ev | dft_donor_chg"
    col_header += " | source | rationale"

    header_lines = [
        f"=== Metal Selectivity Screen: {outcome.target_metal} over {outcome.competitor_metal} ===",
        f"Screened {outcome.n_screened} candidate(s) over {outcome.n_cycles} cycle(s).",
        f"Top ligand: {top_str}",
        "=" * 52,
        "",
        col_header,
    ]

    rows = []
    for r in results:
        src = f"source={r.source}"
        if r.source_id:
            src += f"; id={r.source_id}"
        dft_cols = ""
        if has_dft:
            dr = dft_results.get(r.ligand_smiles)
            if dr and dr.success:
                mean_chg = (
                    sum(dr.donor_charges) / len(dr.donor_charges)
                    if dr.donor_charges else float("nan")
                )
                dft_cols = f" | {dr.homo_ev:.2f} | {mean_chg:.3f}"
            else:
                dft_cols = " | — | —"
        rows.append(
            f"{r.ligand_smiles} | {r.log_k_target:.2f} | {r.log_k_competitor:.2f} | "
            f"{r.delta_log_k:.2f} | {r.composite_score:.2f}{dft_cols} | {src} | {r.rationale}"
        )

    review_lines: list[str] = []
    if outcome.llm_candidate_reviews:
        review_lines.append("")
        review_lines.append("LLM ligand reviews:")
        for rev in outcome.llm_candidate_reviews:
            notes = "; ".join(rev.notes) if rev.notes else "-"
            review_lines.append(
                f"{rev.smiles} | {rev.decision} | confidence={rev.confidence:.2f} | "
                f"{rev.rationale} | {notes}"
            )

    brainstorm_lines: list[str] = []
    if outcome.llm_brainstorm:
        brainstorm_lines.append("")
        brainstorm_lines.append("LLM brainstorm:")
        for b in outcome.llm_brainstorm:
            brainstorm_lines.append(f"{b.smiles} | {b.family} | {b.rationale}")

    dft_lines: list[str] = []
    if has_dft:
        dft_lines.append("")
        dft_lines.append("DFT validation: B3LYP-D3(BJ)/def2-SVP, free ligand, gas phase")
        for smi, dr in dft_results.items():
            if dr.success:
                mean_chg = (
                    sum(dr.donor_charges) / len(dr.donor_charges)
                    if dr.donor_charges else float("nan")
                )
                dft_lines.append(
                    f"  {smi}: HOMO={dr.homo_ev:.2f} eV, "
                    f"gap={dr.homo_lumo_gap_ev:.2f} eV, mean_donor_chg={mean_chg:.3f}"
                )
            else:
                dft_lines.append(f"  {smi}: FAILED — {dr.error}")

    warning_lines: list[str] = []
    if outcome.warnings:
        warning_lines.append("")
        warning_lines.append("Warnings:")
        for w in outcome.warnings:
            warning_lines.append(f"- {w}")

    return "\n".join(header_lines + rows + review_lines + brainstorm_lines + dft_lines + warning_lines)
```

- [ ] **Step 6: Run all new tests**

```bash
pytest tests/test_dft_cli_report.py tests/test_dft_nomination_prompt.py tests/test_dft_validator.py tests/test_dft_integration.py -v
```
Expected: all tests PASS.

- [ ] **Step 7: Run full suite for regressions**

```bash
pytest tests/ -q --ignore=tests/test_benchmarks_examples.py 2>&1 | tail -5
```
Expected: same pass count as before (877+), 0 new failures.

- [ ] **Step 8: Commit**

```bash
git add des_multi_agent/cli.py des_multi_agent/reporting.py tests/test_dft_cli_report.py
git commit -m "feat: add --dft-validate CLI flag, dep check, and DFT report columns"
```
