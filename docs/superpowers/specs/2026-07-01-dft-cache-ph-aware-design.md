# DFT Result Caching + pH-Aware DFT Design

## Goal

Two targeted efficiency/accuracy improvements to the existing `--dft-validate` metal-selectivity DFT stage (shipped in commits `3f462e7..f697482`):

1. **Caching** — avoid recomputing DFT for a ligand species that has already been computed, across runs and cycles.
2. **pH-awareness** — compute DFT on the actual dominant protonation state at the run's binding pH, instead of always assuming the neutral input SMILES.

Both are backward-compatible extensions of the existing DFT pipeline. Neither changes default behavior when `--dft-validate` is not passed, and neither changes the numeric output of existing `compute_dft_properties(smiles)` calls that don't pass `pH`.

## Scope

Metal-selectivity workflow's DFT stage only (the same scope as the original DFT feature). No CLI surface changes — no new flags. `binding_pH` (already a parameter of `run_metal_selectivity_screen`, default 7.0, currently only used for coordination-claim grounding) is threaded into the DFT stage for the first time.

---

## 1. DFT Result Caching

### Problem

Every `--dft-validate` run recomputes DFT from scratch, even when the exact same ligand species was already computed in an earlier cycle or an earlier run. Multi-cycle screening in particular can re-nominate the same top ligand repeatedly.

### Design

**New file:** `des_multi_agent/chemistry/dft_cache.py`

- SQLite database at `artifacts/dft_cache/dft_results.sqlite3` (path resolved the same way `chemistry/partner_registry.py` resolves `artifacts/`: `Path(__file__).resolve().parents[2] / "artifacts" / "dft_cache" / "dft_results.sqlite3"`). Created on first use if missing.
- Schema:
  ```sql
  CREATE TABLE IF NOT EXISTS dft_cache (
      species_smiles TEXT NOT NULL,
      dft_method     TEXT NOT NULL,
      result_json    TEXT NOT NULL,
      computed_at    REAL NOT NULL,
      PRIMARY KEY (species_smiles, dft_method)
  )
  ```
- Cache key: `(species_smiles, dft_method)` — **not** the original input SMILES. `species_smiles` is the canonical SMILES of the actual protonated species that was computed (see Section 2). This means:
  - The same ligand at two different pH values gets two independent cache entries (their `species_smiles` differ when the pH crosses a pKa).
  - Two different input ligands that happen to protonate to the same dominant species at the same pH correctly share one cache entry.
- `cached_compute_dft_properties(smiles: str, pH: float = 7.0, dft_method: str = "B3LYP-D3(BJ)/def2-SVP", cache_path: str | Path | None = None) -> DFTResult`:
  1. Compute `species_smiles = dominant_species(smiles, pH).species_smiles`. This call is RDKit-only (no DFT) and cheap even on a cache miss.
  2. Look up `(species_smiles, dft_method)` in the cache.
     - **Hit:** deserialize the stored JSON into a `DFTResult`, set `from_cache=True`, return it.
     - **Miss:** call `compute_dft_properties(smiles, pH=pH)`. If the result has `success=True`, serialize and store it (`INSERT OR REPLACE`). Return the result with `from_cache=False`.
  3. Only `success=True` results are cached. A transient xTB timeout or SCF non-convergence is not cached — the next attempt may succeed (e.g. after installing a fix, or on a machine with more memory).
  4. Any exception while touching the cache (missing/corrupt DB file, `sqlite3.OperationalError`, permissions failure, serialization error) is caught internally and the function falls back to calling `compute_dft_properties(smiles, pH=pH)` directly without caching. `cached_compute_dft_properties` never raises — same contract as `compute_dft_properties`.
- Serialization: `DFTResult` is a plain dataclass with only `str`, `bool`, `float | None`, and `list[float]` fields — serialized with `json.dumps(dataclasses.asdict(result))` / `json.loads` + `DFTResult(**data)`.
- No TTL / expiry / invalidation mechanism. The cache is correct-by-construction (same species + same method ⇒ same physics), so entries never go stale on their own. A user who wants to clear it deletes the SQLite file.

### Integration

`des_multi_agent/workflows/metal_binding_selectivity.py`'s DFT stage (currently importing `compute_dft_properties` directly) switches to importing `cached_compute_dft_properties` from `dft_cache.py` instead, and passes `pH=binding_pH` (see Section 2 for why).

---

## 2. pH-Aware DFT

### Problem

`compute_dft_properties` always computes the neutral, as-drawn input SMILES. Real ligands in solution are often ionized — a carboxylic acid ligand at physiological pH 7.4 is predominantly the carboxylate anion, which has a substantially different (higher) HOMO energy than the neutral acid used today. The existing `dominant_species(smiles, pH)` function (already used for coordination-claim grounding in `claim_grounding.py`) computes exactly this dominant ionization state, but the DFT pipeline never calls it.

### Design

**Modify `des_multi_agent/chemistry/dft_validator.py`:**

- `DFTResult` gains three new fields, all with backward-compatible defaults:
  - `species_smiles: str | None = None` — canonical SMILES of the species actually computed (differs from `smiles` only when `pH` triggers a protonation change)
  - `ph: float | None = None` — the pH used, if any
  - `from_cache: bool = False` — set by `dft_cache.py`; always `False` for direct `compute_dft_properties` calls
- `compute_dft_properties(smiles: str, pH: float | None = None) -> DFTResult`:
  - **`pH=None` (default):** exact current behavior, unchanged. Computes the input SMILES as-is with `charge=0`. `species_smiles` and `ph` stay `None`. This preserves all 9 existing Task-1 tests without modification.
  - **`pH=<float>`:** calls `dominant_species(smiles, pH)` to obtain `species_smiles` and `net_charge`. Embeds and computes DFT on `species_smiles` instead of the raw input, passing `charge=net_charge` into PySCF (see below). Sets `result.species_smiles` and `result.ph` on the returned `DFTResult`.
  - Both branches share the same embed → xTB optimize → DFT pipeline; only the input SMILES and the DFT `charge` argument differ.
- `_run_dft(symbols, coords_angstrom, charge: int = 0)`: gains a `charge` parameter, threaded into `gto.Mole(atom=..., basis="def2-svp", charge=charge, spin=0, verbose=0)`. Previously hardcoded to `charge=0` — this was silently wrong for any charged species (PySCF would build the wrong electron count). `spin=0` remains correct regardless of protonation state: protonation adds or removes a bare proton (no electron), so a closed-shell singlet's electron count — and thus its closed-shell character — is unaffected by (de)protonation.
- `compute_dft_properties`'s outer `try/except Exception` already wraps the whole pipeline, so a `dominant_species` failure (which itself never raises — it has its own internal safe fallback) or a charged-species SCF failure is still caught and returns `DFTResult(success=False, ...)` exactly as today.

### Integration

`run_metal_selectivity_screen`'s DFT stage passes `pH=binding_pH` to `cached_compute_dft_properties`. `binding_pH` already flows into the function signature (default `7.0`) and is already used for coordination grounding — this is simply the first time it reaches the DFT stage. No new CLI flag; `binding_pH` remains un-exposed via CLI exactly as it is today (a deliberate scope decision — out of scope for this design).

---

## Data Flow (Updated DFT Stage)

```
nominated SMILES (from LLM or top-N fallback, unchanged from existing DFT stage)
  → cached_compute_dft_properties(smiles, pH=binding_pH)
       → dominant_species(smiles, pH) → species_smiles, net_charge      [cheap, RDKit-only]
       → cache lookup on (species_smiles, dft_method)
            hit  → DFTResult (from_cache=True)                          [no DFT computation]
            miss → compute_dft_properties(smiles, pH=pH)
                     → _embed_mmff(species_smiles)                      [RDKit 3D embed]
                     → _xtb_optimize(mol)                                [xTB GFN2 geometry]
                     → _run_dft(symbols, coords, charge=net_charge)      [gpu4pyscf B3LYP-D3(BJ)/def2-SVP]
                   → cache store (species_smiles, dft_method) → result_json
                   → DFTResult (from_cache=False)
  → composite-score adjustment (unchanged ±0.05 tiebreak logic from the existing feature)
```

---

## Error Handling

- `cached_compute_dft_properties` never raises. Any cache-layer failure (I/O, corruption, serialization) silently degrades to an uncached direct call.
- `compute_dft_properties` never raises, with or without `pH`. A `dominant_species` failure returns a passthrough (unmodified, neutral) result per its own existing contract, so the DFT pipeline proceeds on the neutral species and charge 0 rather than aborting.
- `dominant_species` failing during `cached_compute_dft_properties`'s cache-key computation step (before any DFT work) is likewise non-fatal: its passthrough contract means `species_smiles` just falls back to the canonicalized (or raw, if unparseable) input SMILES, and the cache lookup/store proceeds normally on that key.
- No new failure mode is introduced into the workflow's DFT stage — it already treats any `DFTResult(success=False)` as non-fatal (warning + skip, per the existing feature).

---

## Testing Plan

`tests/test_dft_cache.py` (new):
- Cache miss followed by cache hit on the same `(smiles, pH)` returns an identical `DFTResult` (compare field values, not object identity); the second call's `from_cache` is `True`.
- A `success=False` result from `compute_dft_properties` is never written to the cache (verify by checking a subsequent call still triggers a fresh computation).
- Cache keys on `species_smiles`, not the raw input `smiles` — two different input SMILES that `dominant_species` maps to the same `species_smiles` at the same pH share one cache entry (second lookup is a hit without a second DFT call).
- The same input SMILES at two different pH values (that cross a pKa, producing different `species_smiles`) produces two independent cache entries — no false-positive hit.
- Corrupt/missing/unwritable cache file: `cached_compute_dft_properties` falls back to an uncached `compute_dft_properties` call and still returns a valid result (does not raise).

`tests/test_dft_validator.py` (additions):
- `pH=None` (the default): behavior is byte-for-byte identical to the current implementation — regression coverage for all 9 existing tests.
- A carboxylic acid SMILES at `pH=7.4` (above its pKa ~4.2) is embedded/computed as the deprotonated, `charge=-1` species; `result.species_smiles` differs from `result.smiles`; `result.ph == 7.4`.
- The same carboxylic acid at `pH=2.0` (below its pKa) computes the neutral, `charge=0` species — `result.species_smiles == result.smiles` (mod canonicalization).
- `_run_dft` with a nonzero `charge` argument is threaded correctly into `gto.Mole(...)` (verify via a mocked `gto.Mole` call's kwargs).

**Regression:** full existing suite (914 tests as of `f697482`) continues to pass unchanged.

---

## Global Constraints

- `pH=None` default on `compute_dft_properties` preserves exact current behavior — no existing test or caller is affected unless it explicitly opts into `pH=<float>`.
- Cache is best-effort and silent on failure — never raises, never blocks a DFT computation from proceeding.
- Only successful DFT results are cached.
- Cache key is `(species_smiles, dft_method)` — never the raw input SMILES alone.
- No new CLI flags. `binding_pH` keeps its existing default of `7.0` and its existing (Python-API-only) exposure.
- `_run_dft`'s `spin` stays fixed at `0` — protonation state changes never introduce open-shell character for the ligands this pipeline targets.
