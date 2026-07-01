# DFT Result Caching + pH-Aware DFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite-backed caching and pH-aware protonation to the existing `--dft-validate` DFT pipeline in the metal-selectivity workflow, without changing any existing behavior when `pH` is not passed or `--dft-validate` is off.

**Architecture:** `dft_validator.py`'s `compute_dft_properties` gains an optional `pH` parameter that, when set, computes the actual protonated species (via the existing `chemistry.protonation.dominant_species`) at its real formal charge instead of always assuming the neutral input. A new `dft_cache.py` module wraps `compute_dft_properties` with a SQLite cache keyed on `(species_smiles, dft_method)`. The metal-selectivity workflow's DFT stage switches to the cached wrapper and passes its existing `binding_pH` parameter through for the first time.

**Tech Stack:** Python 3.11+, RDKit (already a dependency), `sqlite3` (stdlib), `dataclasses`/`json` (stdlib). No new third-party dependencies.

## Global Constraints

- `compute_dft_properties(smiles, pH=None)` — when `pH` is `None` (the default), behavior is byte-for-byte identical to the current implementation. No existing test changes.
- `cached_compute_dft_properties` never raises. Any cache-layer failure (I/O, corruption, serialization) silently falls back to an uncached direct call.
- Only `success=True` DFT results are ever written to the cache.
- Cache key is `(species_smiles, dft_method)` — never the raw input SMILES alone.
- No new CLI flags. `binding_pH` keeps its existing default of `7.0` and its existing Python-API-only exposure (`run_metal_selectivity_screen(..., binding_pH=...)`).
- `_run_dft`'s `spin` parameter stays fixed at `0` — protonation changes never introduce open-shell character for this pipeline's target ligands.
- Full existing suite (914 tests as of commit `f697482`) continues to pass unchanged.

---

## Task 1: pH-aware DFT in `dft_validator.py`

**Files:**
- Modify: `des_multi_agent/chemistry/dft_validator.py`
- Test: `tests/test_dft_validator.py` (append new test classes)

**Interfaces:**
- Consumes: `des_multi_agent.chemistry.protonation.dominant_species(smiles_or_mol, pH: float = 7.0) -> ProtonationResult` (existing, already used by `claim_grounding.py`). `ProtonationResult` has fields `species_smiles: str` and `net_charge: int`.
- Produces:
  - `DEFAULT_DFT_METHOD: str` module constant (`"B3LYP-D3(BJ)/def2-SVP"`) — Task 2 imports this.
  - `DFTResult` gains fields `species_smiles: str | None = None`, `ph: float | None = None`, `from_cache: bool = False` — Task 2 sets `from_cache` on cache hits.
  - `compute_dft_properties(smiles: str, pH: float | None = None) -> DFTResult` — Task 2 calls this on cache misses.
  - `_run_dft(symbols, coords_angstrom, charge: int = 0)` — internal, no other task calls it directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dft_validator.py` (after the existing `TestComputeDFTProperties` class, before the `from des_multi_agent.chemistry.dft_selectivity import ...` line):

```python
class TestPHAwareDFT:
    def _mock_mf(self):
        HARTREE_TO_EV = 27.2114
        mf = MagicMock()
        mf.converged = True
        mo_energies = [-15.0, -12.0, -10.0, -8.5, -5.5, -3.0]
        mf.mo_energy = [e / HARTREE_TO_EV for e in mo_energies]
        mf.mulliken_pop.return_value = (None, np.array([-0.3, -0.1, -0.1, -0.3]))
        return mf

    def test_ph_none_preserves_legacy_behavior(self):
        mock_mol = MagicMock()
        mock_mf = self._mock_mf()
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["N", "C", "C", "N"], np.zeros((4, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   return_value=(-8.5, 5.1, [0, 3], mock_mf)):
            result = compute_dft_properties("NCCN")
        assert result.success is True
        assert result.species_smiles is None
        assert result.ph is None

    def test_ph_aware_deprotonates_carboxylic_acid_at_high_ph(self):
        mock_mol = MagicMock()
        mock_mf = self._mock_mf()
        captured = {}

        def _fake_run_dft(symbols, coords, charge=0):
            captured["charge"] = charge
            return (-9.0, 6.0, [0, 1], mock_mf)

        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["C", "C", "O", "O"], np.zeros((4, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   side_effect=_fake_run_dft):
            result = compute_dft_properties("CC(=O)O", pH=7.4)

        assert result.success is True
        assert result.ph == 7.4
        assert result.species_smiles is not None
        assert result.species_smiles != "CC(=O)O"   # deprotonated -> different canonical SMILES
        assert captured["charge"] == -1

    def test_ph_aware_neutral_carboxylic_acid_at_low_ph(self):
        from rdkit import Chem

        mock_mol = MagicMock()
        mock_mf = self._mock_mf()
        captured = {}

        def _fake_run_dft(symbols, coords, charge=0):
            captured["charge"] = charge
            return (-10.5, 6.5, [0, 1], mock_mf)

        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["C", "C", "O", "O"], np.zeros((4, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   side_effect=_fake_run_dft):
            result = compute_dft_properties("CC(=O)O", pH=2.0)

        assert result.success is True
        assert result.ph == 2.0
        assert captured["charge"] == 0
        # Below the carboxylic acid pKa (~4.2) the species stays neutral —
        # species_smiles should match the canonical form of the input.
        assert result.species_smiles == Chem.MolToSmiles(Chem.MolFromSmiles("CC(=O)O"))

    def test_dft_failure_still_records_ph(self):
        mock_mol = MagicMock()
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   side_effect=RuntimeError("xtb optimization failed")):
            result = compute_dft_properties("CC(=O)O", pH=7.4)
        assert result.success is False
        assert result.ph == 7.4


class TestRunDFTChargeThreading:
    def test_charge_kwarg_passed_to_gto_mole(self):
        from des_multi_agent.chemistry import dft_validator

        mock_gto = MagicMock()
        mock_gto.Mole.return_value = MagicMock()

        mock_mf = MagicMock()
        mock_mf.converged = True
        HARTREE_TO_EV = 27.2114
        mock_mf.mo_energy = [e / HARTREE_TO_EV for e in [-15.0, -12.0, -10.0, -8.5, -5.5]]
        mock_gpu_dft = MagicMock()
        mock_gpu_dft.RKS.return_value = mock_mf

        mock_pyscf = MagicMock(gto=mock_gto)
        mock_gpu4pyscf = MagicMock(dft=mock_gpu_dft)

        with patch.dict("sys.modules", {
            "pyscf": mock_pyscf,
            "gpu4pyscf": mock_gpu4pyscf,
            "gpu4pyscf.dft": mock_gpu_dft,
        }):
            dft_validator._run_dft(["C", "C", "N", "N"], np.zeros((4, 3)), charge=-1)

        _, kwargs = mock_gto.Mole.call_args
        assert kwargs["charge"] == -1
        assert kwargs["spin"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dft_validator.py::TestPHAwareDFT tests/test_dft_validator.py::TestRunDFTChargeThreading -v`
Expected: FAIL — `compute_dft_properties() got an unexpected keyword argument 'pH'` and `_run_dft() got an unexpected keyword argument 'charge'`.

- [ ] **Step 3: Implement pH-awareness in `dft_validator.py`**

Replace the top of the file (constants + `DFTResult`):

```python
_DONOR_SYMBOLS: frozenset[str] = frozenset({"N", "O", "P", "S"})
_HARTREE_TO_EV: float = 27.2114
DEFAULT_DFT_METHOD: str = "B3LYP-D3(BJ)/def2-SVP"


@dataclass
class DFTResult:
    smiles: str
    success: bool
    homo_ev: float | None = None
    homo_lumo_gap_ev: float | None = None
    donor_charges: list[float] = field(default_factory=list)
    geometry_method: str = "xtb"
    dft_method: str = DEFAULT_DFT_METHOD
    error: str | None = None
    species_smiles: str | None = None
    ph: float | None = None
    from_cache: bool = False
```

Replace `_run_dft` with a version accepting `charge`:

```python
def _run_dft(
    symbols: list[str], coords_angstrom: "np.ndarray", charge: int = 0
) -> tuple[float, float, list[int], object]:
    """Run B3LYP-D3(BJ)/def2-SVP single-point. Returns (homo_ev, gap_ev, donor_indices, mf).

    Raises RuntimeError if SCF does not converge.
    """
    from pyscf import gto
    from gpu4pyscf import dft as gpu_dft

    atom_list = [(sym, tuple(pos)) for sym, pos in zip(symbols, coords_angstrom)]
    mol = gto.Mole(atom=atom_list, basis="def2-svp", charge=charge, spin=0, verbose=0)
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
```

Replace `compute_dft_properties`:

```python
def compute_dft_properties(smiles: str, pH: float | None = None) -> DFTResult:
    """Full pipeline: SMILES -> DFTResult. Never raises.

    pH=None (default): computes the neutral input SMILES as-is with charge=0 —
    exact legacy behavior. pH=<float>: computes the dominant protonation state
    at that pH (chemistry.protonation.dominant_species) with its real net
    formal charge.
    """
    species_smiles = smiles
    net_charge = 0
    if pH is not None:
        from .protonation import dominant_species
        protonation = dominant_species(smiles, pH)
        species_smiles = protonation.species_smiles
        net_charge = protonation.net_charge

    result_species_smiles = species_smiles if pH is not None else None

    try:
        mol = _embed_mmff(species_smiles)
        symbols, coords = _xtb_optimize(mol)
        homo_ev, gap_ev, donor_indices, mf = _run_dft(symbols, coords, charge=net_charge)
        donor_charges = _get_donor_charges(mf, donor_indices)
        return DFTResult(
            smiles=smiles,
            success=True,
            homo_ev=homo_ev,
            homo_lumo_gap_ev=gap_ev,
            donor_charges=donor_charges,
            species_smiles=result_species_smiles,
            ph=pH,
        )
    except Exception as exc:
        return DFTResult(
            smiles=smiles, success=False, error=str(exc),
            species_smiles=result_species_smiles, ph=pH,
        )
```

`_embed_mmff`, `_xtb_optimize`, and `_get_donor_charges` are unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dft_validator.py -v`
Expected: all tests PASS (14 existing + 5 new `TestPHAwareDFT` + 1 new `TestRunDFTChargeThreading` = 20 total in this file).

- [ ] **Step 5: Run full suite to check for regressions**

Run: `pytest tests/ -q --ignore=tests/test_benchmarks_examples.py`
Expected: all pass, no regressions (914 + 6 new = 920).

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/chemistry/dft_validator.py tests/test_dft_validator.py
git commit -m "feat: add pH-aware protonation to compute_dft_properties"
```

---

## Task 2: DFT result cache (`dft_cache.py`)

**Files:**
- Create: `des_multi_agent/chemistry/dft_cache.py`
- Test: `tests/test_dft_cache.py`

**Interfaces:**
- Consumes: `DFTResult`, `DEFAULT_DFT_METHOD`, `compute_dft_properties(smiles, pH=None)` from Task 1's `des_multi_agent.chemistry.dft_validator`. `dominant_species(smiles, pH) -> ProtonationResult` (with `.species_smiles`) from `des_multi_agent.chemistry.protonation` (existing, unmodified).
- Produces: `cached_compute_dft_properties(smiles: str, pH: float = 7.0, dft_method: str = DEFAULT_DFT_METHOD, cache_path: str | Path | None = None) -> DFTResult` — Task 3 imports and calls this from the workflow's DFT stage.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dft_cache.py`:

```python
"""Unit tests for dft_cache. compute_dft_properties is mocked; no gpu4pyscf/xtb required."""
from __future__ import annotations

from unittest.mock import patch

from des_multi_agent.chemistry.dft_cache import cached_compute_dft_properties
from des_multi_agent.chemistry.dft_validator import DFTResult


def _success(smiles: str) -> DFTResult:
    return DFTResult(smiles=smiles, success=True, homo_ev=-8.5, homo_lumo_gap_ev=5.1,
                      donor_charges=[-0.3, -0.3])


class TestCacheMissThenHit:
    def test_second_call_is_cache_hit(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("NCCN")) as mock_compute:
            r1 = cached_compute_dft_properties("NCCN", pH=7.0, cache_path=cache_path)
            r2 = cached_compute_dft_properties("NCCN", pH=7.0, cache_path=cache_path)

        assert mock_compute.call_count == 1
        assert r1.from_cache is False
        assert r2.from_cache is True
        assert r2.homo_ev == r1.homo_ev


class TestFailuresNotCached:
    def test_failure_not_cached(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        failure = DFTResult(smiles="X", success=False, error="SCF did not converge")
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=failure) as mock_compute:
            cached_compute_dft_properties("X", pH=7.0, cache_path=cache_path)
            cached_compute_dft_properties("X", pH=7.0, cache_path=cache_path)

        assert mock_compute.call_count == 2   # never cached -> recomputed both times


class TestCacheKeyIsSpeciesSmiles:
    def test_two_spellings_of_same_species_share_cache_entry(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        # "CC(=O)O" and "OC(C)=O" are the same molecule (acetic acid) written
        # differently -> dominant_species canonicalizes both to one species_smiles.
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("CC(=O)O")) as mock_compute:
            cached_compute_dft_properties("CC(=O)O", pH=7.4, cache_path=cache_path)
            cached_compute_dft_properties("OC(C)=O", pH=7.4, cache_path=cache_path)

        assert mock_compute.call_count == 1

    def test_same_input_different_ph_gives_separate_entries(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("CC(=O)O")) as mock_compute:
            cached_compute_dft_properties("CC(=O)O", pH=2.0, cache_path=cache_path)   # neutral
            cached_compute_dft_properties("CC(=O)O", pH=7.4, cache_path=cache_path)   # deprotonated

        assert mock_compute.call_count == 2   # different species_smiles -> no false hit


class TestCacheFailureFallback:
    def test_corrupt_cache_file_falls_back_to_direct_call(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        cache_path.write_text("not a valid sqlite file")
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("NCCN")) as mock_compute:
            result = cached_compute_dft_properties("NCCN", pH=7.0, cache_path=cache_path)

        assert result.success is True
        assert mock_compute.call_count == 1


class TestCacheFileCreation:
    def test_cache_db_created_in_missing_parent_dir(self, tmp_path):
        cache_path = tmp_path / "nested" / "dir" / "dft.sqlite3"
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("NCCN")):
            cached_compute_dft_properties("NCCN", pH=7.0, cache_path=cache_path)
        assert cache_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dft_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'des_multi_agent.chemistry.dft_cache'`.

- [ ] **Step 3: Implement `dft_cache.py`**

Create `des_multi_agent/chemistry/dft_cache.py`:

```python
"""SQLite-backed cache for DFT results, keyed by (species_smiles, dft_method).

Entry point: cached_compute_dft_properties(smiles, pH=7.0, dft_method=..., cache_path=None).
Never raises — any cache-layer failure falls back to an uncached compute_dft_properties call.
Only success=True results are cached.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
from pathlib import Path

from .dft_validator import DFTResult, DEFAULT_DFT_METHOD, compute_dft_properties
from .protonation import dominant_species

DEFAULT_CACHE_PATH: Path = (
    Path(__file__).resolve().parents[2] / "artifacts" / "dft_cache" / "dft_results.sqlite3"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dft_cache (
    species_smiles TEXT NOT NULL,
    dft_method     TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    computed_at    REAL NOT NULL,
    PRIMARY KEY (species_smiles, dft_method)
)
"""


def _resolve_cache_path(cache_path: str | Path | None) -> Path:
    return Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.execute(_SCHEMA)
    return conn


def _load_cached(conn: sqlite3.Connection, species_smiles: str, dft_method: str) -> DFTResult | None:
    row = conn.execute(
        "SELECT result_json FROM dft_cache WHERE species_smiles = ? AND dft_method = ?",
        (species_smiles, dft_method),
    ).fetchone()
    if row is None:
        return None
    return DFTResult(**json.loads(row[0]))


def _store(conn: sqlite3.Connection, species_smiles: str, dft_method: str,
           result: DFTResult, computed_at: float) -> None:
    payload = json.dumps(dataclasses.asdict(result))
    conn.execute(
        "INSERT OR REPLACE INTO dft_cache "
        "(species_smiles, dft_method, result_json, computed_at) VALUES (?, ?, ?, ?)",
        (species_smiles, dft_method, payload, computed_at),
    )
    conn.commit()


def cached_compute_dft_properties(
    smiles: str,
    pH: float = 7.0,
    dft_method: str = DEFAULT_DFT_METHOD,
    cache_path: str | Path | None = None,
) -> DFTResult:
    """Cache-aware wrapper around compute_dft_properties. Never raises."""
    try:
        species_smiles = dominant_species(smiles, pH).species_smiles
    except Exception:
        species_smiles = smiles

    resolved_path = _resolve_cache_path(cache_path)

    conn = None
    try:
        conn = _connect(resolved_path)
        cached = _load_cached(conn, species_smiles, dft_method)
        if cached is not None:
            cached.from_cache = True
            return cached
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()

    result = compute_dft_properties(smiles, pH=pH)

    if result.success:
        conn2 = None
        try:
            conn2 = _connect(resolved_path)
            _store(conn2, species_smiles, dft_method, result, time.time())
        except Exception:
            pass
        finally:
            if conn2 is not None:
                conn2.close()

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dft_cache.py -v`
Expected: 6/6 PASS.

- [ ] **Step 5: Run full suite to check for regressions**

Run: `pytest tests/ -q --ignore=tests/test_benchmarks_examples.py`
Expected: all pass (920 + 6 new = 926).

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/chemistry/dft_cache.py tests/test_dft_cache.py
git commit -m "feat: add SQLite-backed DFT result cache"
```

---

## Task 3: Wire cache + pH into the metal-selectivity workflow

**Files:**
- Modify: `des_multi_agent/workflows/metal_binding_selectivity.py` (DFT stage inside `run_metal_selectivity_screen`)
- Modify: `tests/test_dft_integration.py` (update mock patch target + add pH-forwarding test)

**Interfaces:**
- Consumes: `cached_compute_dft_properties(smiles, pH=7.0, dft_method=..., cache_path=None) -> DFTResult` from Task 2's `des_multi_agent.chemistry.dft_cache`.
- Produces: nothing new — this task only rewires an existing call site.

**Important — why the existing integration test must change:** `tests/test_dft_integration.py`'s `_run` helper currently patches `des_multi_agent.chemistry.dft_validator.compute_dft_properties`. After this task's change, the workflow calls `cached_compute_dft_properties` (imported from `dft_cache`) instead of `compute_dft_properties` directly. Patching the old target would silently stop working (the mock would never be hit, and the real cache/SQLite/protonation path would run instead — masking every assertion in that test class). The patch target must move to `des_multi_agent.chemistry.dft_cache.cached_compute_dft_properties`, and the lambda's signature must accept the new `pH=` keyword.

- [ ] **Step 1: Write the failing test (pH forwarding)**

In `tests/test_dft_integration.py`, add this test inside the existing `TestDFTStageWiring` class (after `test_dft_failure_adds_warning`):

```python
    def test_binding_ph_reaches_dft_stage(self):
        captured_ph = []

        def _capture(smi, pH=None, **kwargs):
            captured_ph.append(pH)
            return FAKE_DFT_SUCCESS

        with (
            patch(
                "des_multi_agent.workflows.metal_binding_selectivity.predict_log_k",
                side_effect=_mock_log_k,
            ),
            patch(
                "des_multi_agent.chemistry.dft_cache.cached_compute_dft_properties",
                side_effect=_capture,
            ),
        ):
            run_metal_selectivity_screen(
                target_metal="Cu2+",
                competitor_metal="Zn2+",
                n=3,
                model_path=None,
                llm_provider=None,
                n_cycles=1,
                dft_validate=True,
                dft_top_n=1,
                binding_pH=5.5,
            )

        assert captured_ph, "expected cached_compute_dft_properties to be called"
        assert all(ph == 5.5 for ph in captured_ph)
```

Also update the existing `_run` helper method (in the same `TestDFTStageWiring` class) to patch the new target — replace:

```python
            patch(
                "des_multi_agent.chemistry.dft_validator.compute_dft_properties",
                side_effect=lambda smi: chosen,
            ),
```

with:

```python
            patch(
                "des_multi_agent.chemistry.dft_cache.cached_compute_dft_properties",
                side_effect=lambda smi, pH=None, **kwargs: chosen,
            ),
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dft_integration.py -v`
Expected: `test_binding_ph_reaches_dft_stage` FAILS (no `pH` kwarg reaches the mock — the workflow still calls the old, un-cached function). The other `TestDFTStageWiring` tests may also fail now that their patch target no longer matches the (not-yet-updated) workflow code.

- [ ] **Step 3: Wire `cached_compute_dft_properties` into the workflow**

In `des_multi_agent/workflows/metal_binding_selectivity.py`, inside `run_metal_selectivity_screen`'s DFT stage (the `if dft_validate and cumulative_results:` block), change the import line:

```python
        from ..chemistry.dft_validator import compute_dft_properties as _dft
```

to:

```python
        from ..chemistry.dft_cache import cached_compute_dft_properties as _dft
```

And change the per-candidate call:

```python
        for smi in nominated_smiles:
            res = _dft(smi)
```

to:

```python
        for smi in nominated_smiles:
            res = _dft(smi, pH=binding_pH)
```

No other lines in the DFT stage change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dft_integration.py -v`
Expected: all PASS, including the new `test_binding_ph_reaches_dft_stage`.

- [ ] **Step 5: Run full suite to check for regressions**

Run: `pytest tests/ -q --ignore=tests/test_benchmarks_examples.py`
Expected: all pass (926 + 1 new = 927).

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/workflows/metal_binding_selectivity.py tests/test_dft_integration.py
git commit -m "feat: wire DFT result cache and binding pH into metal-selectivity DFT stage"
```

---

## Final Verification

After all three tasks:

```bash
pytest tests/ -q --ignore=tests/test_benchmarks_examples.py
```

Expected: all tests pass (927 total, up from 914 before this plan). No new CLI flags exist; `python -m des_multi_agent.cli --workflow metal-selectivity ... --dft-validate` behaves exactly as before except DFT now runs on the pH-7.0-dominant species and repeat candidates across cycles/runs are served from cache.
