# gpu4pyscf DFT Selectivity Validation Design

## Goal

Add an optional, flag-gated DFT validation stage to the metal-selectivity workflow. When `--dft-validate` is passed, the LLM nominates 1–3 top candidates from the rule-based shortlist, those candidates are run through a free-ligand DFT pipeline (SMILES → RDKit MMFF94 → xTB GFN2 → gpu4pyscf B3LYP-D3(BJ)/def2-SVP single-point), and the resulting HOMO energy and donor-atom Löwdin charges feed a small ranking-score adjustment (±0.05) that refines — but never overrides — the rule-based ΔlogK ranking.

DFT is **never triggered automatically**. The only activation path is an explicit `--dft-validate` flag on the command line. No LLM, no heuristic, and no other flag can start DFT computation.

## Scope

Metal-selectivity workflow only (`--workflow metal-selectivity`). DES screening and plain metal-binding are out of scope.

---

## Architecture

### New files

| File | Responsibility |
|---|---|
| `des_multi_agent/chemistry/dft_validator.py` | Full DFT pipeline: `DFTResult` dataclass + `compute_dft_properties(smiles) -> DFTResult`. Never raises — any failure returns `DFTResult(success=False, error=...)`. |
| `des_multi_agent/chemistry/dft_selectivity.py` | `dft_selectivity_adjustment(dft_result, target_metal, competitor_metal) -> float`. Converts HOMO energy into a ±0.05 composite-score nudge using existing HSAB softness data from `stability_rules._metal_softness`. |
| `tests/test_dft_validator.py` | Unit tests for the DFT pipeline and selectivity adjustment. |
| `tests/test_dft_nomination_prompt.py` | Unit tests for the LLM nomination prompt. |

### Modified files

| File | Change |
|---|---|
| `des_multi_agent/workflows/metal_binding_selectivity.py` | Accept `dft_validate: bool = False`, `dft_top_n: int = 3`. After rule-based ranking: call LLM nomination, run DFT, call `_apply_dft_selectivity_bias`, store results in `SelectivityScreenOutcome.dft_results`. |
| `des_multi_agent/workflows/metal_binding_selectivity.py` | Add `dft_results: dict[str, DFTResult]` field to `SelectivityScreenOutcome` (new optional field, default empty dict). |
| `des_multi_agent/llm/prompts.py` | New `dft_nomination_prompt(candidates, target_metal, competitor_metal) -> str`. |
| `des_multi_agent/llm/base.py` | New `nominate_for_dft(candidates, target_metal, competitor_metal, llm_config) -> list[str]` — returns SMILES of nominated candidates. Falls back to top-`dft_top_n` by `composite_score` when `llm_config is None`. |
| `des_multi_agent/cli.py` | Add `--dft-validate` (store_true, default False) and `--dft-top-n INT` (default 3) to the metal-selectivity argument group. Forward to `run_metal_selectivity_screen`. |
| Metal-selectivity report builder | When `outcome.dft_results` is non-empty, add a `DFT` column to the results table showing `homo_ev` and `donor_chg` for nominated candidates; non-nominated rows show `—`. |

---

## Data Flow

```
--workflow metal-selectivity (existing rule-based ranking)
  └─ SelectivityScreenOutcome.results: list[SelectivityResult]
       sorted by composite_score (ΔlogK + HSAB + chelate)

  [only when --dft-validate is present]
  ├─ LLM nomination
  │    input:  top-K results (K = min(dft_top_n * 2, len(results)))
  │    prompt: dft_nomination_prompt(candidates, target_metal, competitor_metal)
  │    output: list of 1–dft_top_n SMILES to validate
  │    fallback (no LLM): take top-dft_top_n by composite_score directly
  │
  ├─ DFT pipeline (per nominee, sequential)
  │    compute_dft_properties(smiles)
  │    ├─ RDKit AllChem.EmbedMolecule + MMFFOptimizeMolecule  → 3D geometry
  │    ├─ xTB GFN2 optimize (xtb-python bindings; subprocess fallback) → better geometry
  │    └─ gpu4pyscf RKS B3LYP-D3(BJ)/def2-SVP single-point
  │         → homo_ev (Hartree → eV)
  │         → homo_lumo_gap_ev
  │         → donor_charges (Löwdin, at donor atom indices from coordination_profile)
  │
  ├─ _apply_dft_selectivity_bias(results, dft_results, target, competitor)
  │    for each result with dft_result.success == True:
  │      adjustment = dft_selectivity_adjustment(dft_result, target_metal, competitor_metal)
  │      new_result = dataclasses.replace(result, composite_score=result.composite_score + adjustment)
  │    re-sort by composite_score, return updated list
  │
  └─ SelectivityScreenOutcome
       .results        ← updated (DFT-adjusted composite_score for nominees)
       .dft_results    ← dict[smiles, DFTResult] for all attempted nominations
       .warnings       ← [DFT] lines for any per-candidate failures
```

---

## DFT Method

**B3LYP-D3(BJ)/def2-SVP** (free ligand, gas phase, restricted closed-shell RKS).

| Choice | Justification |
|---|---|
| B3LYP (hybrid GGA) | Required for reliable HOMO eigenvalues. Pure GGA functionals (PBE, BLYP) systematically mis-order HOMO energies between hard and soft donors — the exact signal used for HSAB selectivity. Hybrid B3LYP corrects this via 20% exact exchange. |
| D3(BJ) dispersion | One-line addition (`mf.disp = 'd3bj'` in gpu4pyscf). Improves donor-atom charge distributions and conformational stability for chelate-forming ligands. Negligible GPU cost overhead. |
| def2-SVP | Ahlrichs split-valence + polarization on heavy atoms. Adequate for relative HOMO ranking on small organic ligands (< ~50 atoms). def2-TZVP would improve absolute energies but adds 3-4× wall-clock with no benefit for relative comparisons where systematic errors cancel. Overkill avoided. |
| Gas phase | Free-ligand-only calculation avoids metal spin-state combinatorics (Cu²⁺ Jahn-Teller, high/low spin Fe²⁺, etc.) while still providing HSAB-relevant orbital information. |
| Löwdin charges | Basis-set-stable; available directly from PySCF `mf.mulliken_pop()` which returns both Mulliken and Löwdin. Mulliken charges blow up with larger bases; Löwdin does not. |

**gpu4pyscf API sketch:**
```python
from pyscf import gto
from gpu4pyscf import dft as gpu_dft

mol = gto.Mole(atom=xyzblock, basis="def2-svp", charge=0, spin=0, verbose=0)
mol.build()
mf = gpu_dft.RKS(mol)
mf.xc = "B3LYP"
mf.disp = "d3bj"
mf.kernel()

homo_idx = mol.nelectron // 2 - 1
homo_ev = float(mf.mo_energy[homo_idx]) * 27.2114   # Hartree → eV
lumo_ev = float(mf.mo_energy[homo_idx + 1]) * 27.2114
gap_ev  = lumo_ev - homo_ev

_, lowdin = mf.mulliken_pop()                        # returns (mulliken, lowdin)
# donor_indices: RDKit substructure query for heteroatoms ([N,O,S,P]) on the
# optimized geometry, or from coordination_profile if it exposes atom indices.
# Implementation should confirm which field CoordinationProfile provides.
donor_charges = [float(lowdin[i]) for i in donor_indices]
```

---

## DFT Properties → Ranking Signal

`dft_selectivity_adjustment(dft_result, target_metal, competitor_metal) -> float`

1. Retrieve softness scores: `s_target = _metal_softness(target_metal)` and `s_comp = _metal_softness(competitor_metal)` from the existing `stability_rules` table.
2. Map ligand HOMO energy to a donor softness proxy. Reference anchors (calibrated against known compounds):
   - HOMO ≤ −9.5 eV → donor softness ≈ 0.0 (hard: oxalate, glycine carboxylate O)
   - HOMO ≥ −7.5 eV → donor softness ≈ 1.0 (soft: thiol S, phosphine P)
   - Linear interpolation between anchors.
3. Compute affinity difference: `delta = abs(ligand_softness - s_target) - abs(ligand_softness - s_comp)`. Negative delta = ligand matches target better; positive = matches competitor better.
4. Map to adjustment: `clamp(−delta × 0.10, −0.05, +0.05)`. Maximum magnitude ±0.05 — DFT is a tiebreaker, not an override. The H-bond bias pass uses ±0.10; this is intentionally half that.

---

## LLM Nomination Prompt

`dft_nomination_prompt(candidates, target_metal, competitor_metal) -> str`

Rendered block presented to the LLM:
```
You are helping prioritize ligands for DFT validation.
Target metal: {target_metal}. Competitor: {competitor_metal}.

Top candidates by predicted selectivity (ΔlogK):
{table: rank | SMILES | ΔlogK | donor atoms | denticity}

Select 1–{dft_top_n} candidates most worth DFT validation. Prefer:
- Ligands where HSAB ambiguity makes the rule-based prediction uncertain
- Borderline ΔlogK values (small positive) where DFT tiebreaking matters most
- Structurally diverse nominations over similar analogues

Return: a JSON list of SMILES strings only. Example: ["SMILES1", "SMILES2"]
```

`nominate_for_dft` parses the JSON list, validates each SMILES is in the candidate set, and falls back to top-`dft_top_n` by `composite_score` on any parse error or when no LLM is configured.

---

## Error Handling and Optional Dependencies

**Startup check (when `--dft-validate` is passed):**
```python
try:
    import gpu4pyscf  # noqa: F401
    import xtb        # noqa: F401
except ImportError as e:
    sys.exit(f"[DFT] --dft-validate requires {e.name}. Install with: pip install {e.name}")
```

**Per-candidate failure (inside `compute_dft_properties`):**
- Wrap the entire pipeline in `try/except Exception as e: return DFTResult(success=False, error=str(e))`.
- Specific expected failures: RDKit embed failure (disconnected or exotic SMILES), xTB convergence failure, gpu4pyscf SCF non-convergence.
- Log `[DFT] Warning: skipping {smiles[:40]!r} — {error}` to `warnings` list.
- No ranking adjustment applied for this candidate.

**All nominees fail:**
- Run completes with rule-based ranking unchanged.
- Report shows: `[DFT] Warning: all DFT computations failed — rule-based ranking used`.
- Exit code is still 0 (DFT failure is non-fatal).

---

## CLI Interface

```bash
# Minimal (no LLM — takes top-3 by composite_score for DFT):
python -m des_multi_agent.cli --workflow metal-selectivity \
  --target-metal-ion Cu2+ --competitor-metal-ion Zn2+ \
  --n 20 --stability-constant-model-path artifacts/stability_constants/model.json \
  --dft-validate

# With LLM nomination + custom top-N:
python -m des_multi_agent.cli --workflow metal-selectivity \
  --target-metal-ion Cu2+ --competitor-metal-ion Zn2+ \
  --n 20 --stability-constant-model-path artifacts/stability_constants/model.json \
  --llm-config llm.example.yaml \
  --dft-validate --dft-top-n 2
```

New flags (both in the metal-selectivity argument group):
- `--dft-validate`: `action="store_true", default=False`. Enables the DFT stage.
- `--dft-top-n INT`: `type=int, default=3`. How many nominees to compute DFT on. Capped internally at `len(results)`.

---

## Report Output

When `outcome.dft_results` is non-empty, the results table gains two columns:

```
Rank  Ligand SMILES                 ΔlogK  Score  DFT HOMO (eV)  DFT donor chg
  1   NCCN                          1.42   0.87   −8.91 ↑         −0.31
  2   NCC(=O)O                      1.18   0.81   −9.14           −0.28
  3   c1ccncc1                      0.93   0.74    —               —
```

`↑` / `↓` indicates that DFT adjusted the composite score up or down from its pre-DFT value. Non-nominated rows show `—`. A `[DFT]` section at the end of the report lists the method used and any per-candidate warnings.

---

## Testing Plan

`tests/test_dft_validator.py`:
- `test_embed_xtb_returns_coordinates` — NCCN produces a non-empty coordinate block.
- `test_dft_result_fields_present` — mock gpu4pyscf kernel; verify `homo_ev` is float, `donor_charges` is list, `success=True`.
- `test_invalid_smiles_returns_failure` — `"NOT_VALID"` → `success=False`, no exception raised.
- `test_no_donor_atoms_still_returns_result` — benzene (no donors) produces `donor_charges=[]`, `success=True`.
- `test_dft_selectivity_adjustment_hard_target` — hard donor ligand (oxalate-like HOMO ≤ −9.5 eV) against hard target vs soft competitor returns positive adjustment.
- `test_dft_selectivity_adjustment_soft_target` — thiol-like HOMO ≥ −7.5 eV against soft target vs hard competitor returns positive adjustment.
- `test_apply_dft_bias_rerankss` — two `SelectivityResult` objects; after bias, the DFT-adjusted one moves up.

`tests/test_dft_nomination_prompt.py`:
- `test_prompt_contains_candidate_smiles` — all input SMILES appear in rendered prompt.
- `test_prompt_contains_metal_names` — target and competitor metal names appear.
- `test_fallback_when_no_llm` — `nominate_for_dft(..., llm_config=None)` returns top-N by composite_score.

**Regression:** existing 877 tests must pass unchanged. The DFT stage is always guarded by `dft_validate=False` (default), so no existing test path is affected.

---

## Global Constraints

- DFT activates **only** when `--dft-validate` is explicitly passed. No auto-trigger.
- Method: B3LYP-D3(BJ)/def2-SVP. No other method/basis is the default.
- Charges: Löwdin only (not Mulliken).
- Ranking adjustment magnitude: ±0.05 maximum per candidate.
- `SelectivityResult` remains `frozen=True`; DFT results stored in `SelectivityScreenOutcome.dft_results: dict[str, DFTResult]`.
- DFT failures are non-fatal; run always completes with rule-based ranking as fallback.
- Optional deps (gpu4pyscf, xtb) checked at startup only when `--dft-validate` is active.
