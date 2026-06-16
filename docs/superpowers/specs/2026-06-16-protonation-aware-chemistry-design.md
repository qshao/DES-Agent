# Protonation-Aware Chemistry (pKa) Design

**Date:** 2026-06-16
**Status:** Design — pending implementation plan
**Scope:** Single subsystem. Tautomer handling is explicitly deferred to a follow-up (1A-bis).

---

## Context

DES-Agent's deterministic chemistry tools (`coordination.py`, `hbond.py`) and the
grounding layer built on top of them (`claim_grounding.py`) currently perceive a
molecule **only in its drawn (usually neutral) form**. HBD/HBA counts and metal-donor
atoms are read off the structure as written. But the species that actually exists
depends on pH: a carboxylic acid (pKa ~4) is a carboxylate at pH 7; an amine
(pKa ~10) is an ammonium at pH 7. This silently mis-grounds coordination claims in
the metal-binding workflow — e.g. glycine drawn as `NCC(=O)O` looks like a clean
bidentate N,O-chelator, but at pH 7 it is a zwitterion whose protonated amine
nitrogen has no lone pair to donate.

This feature makes the deterministic chemistry **protonation-aware** so that, when a
pH is supplied, donor atoms and H-bond counts reflect the dominant species rather
than the drawn form. It is the natural deepening of the grounding layer: it
multiplies the correctness of every verdict that depends on donor availability.

### Locked-in decisions (from brainstorming)

- **pH model:** pH-parameterized. Each ionizable group's state derives from its pKa
  vs. a `pH` argument; default 7.0 where a pH is used.
- **Estimation method:** hand-rolled SMARTS→pKa table. Fully in-repo, transparent,
  auditable, zero new dependencies. Matches the existing `chemistry/` SMARTS style.
  Un-tabulated groups are left exactly as drawn (never guessed).
- **Integration depth:** grounding layer + workflows pass the real pH. The DES
  partner search stays as-drawn (neat eutectic, no aqueous proton equilibrium); the
  metal-binding workflows pass an aqueous binding pH.
- **Tautomers:** deferred. This spec covers protonation/charge state only.
- **Mechanism:** direct RWMol edit (per-atom formal-charge + H-count edits), not
  reaction SMARTS — most auditable and gives exact control over zwitterion formation.

### Robustness requirement (carried from the grounding work)

The layer is **purely deterministic and offline** — identical results regardless of
LLM backend, and it runs when the LLM is disabled. Every entry point fails safe
(passthrough on any error, never raises into a prompt or grounding path), mirroring
the `structural_facts` precedent.

---

## Architecture

One new module plus two small, surgical touch-points. The DES path is unchanged
(as-drawn); the metal-binding workflow is the real beneficiary.

| File | Action | Responsibility |
|------|--------|----------------|
| `des_multi_agent/chemistry/protonation.py` | Create | pKa table + `dominant_species()` engine |
| `des_multi_agent/chemistry/coordination.py` | Modify (internals only) | `_is_donor`: a positively-charged N/O with no lone pair is not a donor |
| `des_multi_agent/chemistry/claim_grounding.py` | Modify | `structural_facts(smiles, pH=None)`, `ground_coordination(smiles, claim, pH=None)`, species clause in `as_prompt_block()` |
| `des_multi_agent/workflows/metal_binding_screen.py` | Modify | `binding_pH: float = 7.0` param threaded into coordination grounding + fact injection |
| `des_multi_agent/workflows/metal_binding_selectivity.py` | Modify | same `binding_pH` param threaded into coordination grounding |
| `tests/test_protonation.py` | Create | Engine unit tests |
| `tests/test_coordination.py` | Modify | `_is_donor` species cases |
| `tests/test_claim_grounding.py` | Modify | Species-aware facts + regression (pH=None unchanged) |
| `tests/test_metal_workflows_grounding.py` | Modify | `binding_pH` wiring |

### New module: `protonation.py`

```python
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
    species_smiles: str         # canonical SMILES of the dominant species
    mol: object                 # RDKit Mol of the dominant species
    groups: list[IonizedGroup]
    net_charge: int

def dominant_species(smiles_or_mol, pH: float = 7.0) -> ProtonationResult: ...
```

`_IONIZABLE_SMARTS`: ordered list of `(smarts, group_name, pka, kind)` with
`kind ∈ {"acid", "base"}`. Initial table (representative pKa values):

| group | SMARTS (intent) | pKa | kind |
|-------|-----------------|-----|------|
| sulfonic acid | `S(=O)(=O)[OH]` | -1.0 | acid |
| phosphonic/phosphate OH | `P(=O)[OH]` | 2.0 | acid |
| carboxylic acid | `C(=O)[OH]` | 4.2 | acid |
| thiol | `[#16X2H]` | 10.5 | acid |
| phenol | `c[OH]` | 9.9 | acid |
| guanidine | `NC(=N)N` | 12.5 | base |
| aliphatic 1°/2°/3° amine | `[NX3;H2,H1,H0;!$(NC=O);!$(Nc)]` | 10.6 | base |
| aniline (aromatic amine) | `[NX3;!$(NC=O)]c` | 4.6 | base |
| imidazole-type N | `c1cnc[nH]1` (basic N) | 7.0 | base |
| pyridine-type N | `[nX2;r6]` | 5.2 | base |

(Exact SMARTS and the full table are finalized during implementation; amides
`NC=O` and aromatic-bound amines are excluded from the basic-amine pattern so they
are not treated as ordinary bases.)

**Algorithm (`dominant_species`):**

```
mol = parse(smiles_or_mol); rw = RWMol(mol)
touched: set[int] = set()
groups: list[IonizedGroup] = []
for (smarts, name, pka, kind) in _IONIZABLE_SMARTS:    # most-specific first
    for match in rw.GetSubstructMatches(smarts):
        atom_idx = ionizable atom in match
        if atom_idx in touched: continue
        touched.add(atom_idx)
        if kind == "acid" and pH > pka:
            set formal charge -1, remove one explicit/implicit H  -> "deprotonated"
        elif kind == "base" and pH < pka:
            set formal charge +1, add one H                       -> "protonated"
        else:
            -> "neutral" (no edit)
        groups.append(IonizedGroup(name, atom_idx, pka, state, charge))
sanitize(rw)
return ProtonationResult(input, pH, canonical(rw), rw, groups, net_charge(rw))
```

On any exception in parse/edit/sanitize, return a **passthrough**: `species_smiles`
== canonical input (or the raw input if even parsing fails), `mol` == the unedited
mol (or `None`), `groups == []`, `net_charge` from the unedited mol (or 0). Never
raises.

### Touch-point 1: `coordination._is_donor`

Today a protonated ammonium N⁺ (degree-1, bearing H's) would still be counted as a
metal donor; only quaternary N⁺ is excluded. Extend the existing charged-atom guard
so that **any N or O with a positive formal charge and no available lone pair** is
not a donor. Neutral as-drawn molecules are unaffected, so existing coordination
tests remain green; the change only manifests on species produced by `dominant_species`.

The public `coordination_profile` signature is unchanged.

### Touch-point 2: `claim_grounding`

- `structural_facts(smiles: str, pH: float | None = None) -> StructuralFacts`
  - `pH is None` (default) → as-drawn; **byte-identical to current behavior**.
  - `pH` is a float → compute `dominant_species(smiles, pH).mol` first, then feed that
    Mol to `hbond_profile` / `coordination_profile` (both already accept a Mol).
- `StructuralFacts` gains protonation context (e.g. `net_charge: int = 0`,
  `protonation_summary: str = ""`) so `as_prompt_block()` can append a species clause
  **only when a pH was applied** — when `pH is None` the block is unchanged.
- `ground_coordination(smiles: str, claim_text: str, pH: float | None = None)`
  - `pH is None` → current behavior.
  - `pH` set → verify the claim against the **species** denticity/donors.
- `ground_family` and `ground_des_plausibility` are **not** made pH-aware. Family
  identity is named on the neutral functional group; DES is neat. Both keep operating
  on the as-drawn form.

### Workflow wiring

- `run_metal_binding_screen(..., binding_pH: float = 7.0)`: pass `pH=binding_pH` into
  the `ground_coordination` call and the `structural_facts(...).as_prompt_block()`
  fact injection for ligands.
- `run_metal_selectivity_screen(..., binding_pH: float = 7.0)`: same threading into
  its coordination grounding.
- `binding_pH` defaults to 7.0 so existing callers are unaffected.
- The DES `run_search_report` path is **not** changed — it continues to call
  `structural_facts` / `ground_*` with `pH=None` (neat).

---

## Data flow

**Engine:**

```
dominant_species(smiles, pH)
  parse -> RWMol -> per-group pKa-vs-pH edit (touched-atom guard) -> sanitize
  -> ProtonationResult{species_smiles, mol, groups, net_charge}
  (any failure -> passthrough)
```

**Metal-binding consumption (payoff path):**

```
run_metal_binding_screen(metal_ion, ..., binding_pH=7.0)
  for each brainstormed ligand b:
    ground_coordination(b.smiles, b.rationale, pH=binding_pH)
        -> dominant_species(b.smiles, binding_pH).mol
        -> coordination_profile(species_mol)   # _is_donor now species-correct
        -> verify_coordination_claim vs species denticity/donors
    structural_facts(b.smiles, pH=binding_pH).as_prompt_block()
        -> ligand prompt gains species clause
           "species @ pH7.0: net -1, donors 2 O (COO-), N protonated (not donating)"
```

**DES path:** unchanged — every `structural_facts` / `ground_*` call uses `pH=None`,
producing byte-identical output to today.

---

## Error handling

Every layer fails safe, matching the `structural_facts` precedent:

- `dominant_species` wraps parse + edit + sanitize in try/except → passthrough on any
  failure; never raises.
- Un-tabulated ionizable group → left exactly as drawn (no guess, no warning).
- A SMARTS that fails to compile is caught at module load by an assertion test (same
  guard style as `claim_grounding._FAMILY_SMARTS`), so a malformed table entry fails
  loudly in CI, not silently at runtime.
- `structural_facts(pH=…)` / `ground_coordination(pH=…)`: if `dominant_species`
  returns a passthrough, downstream behaves exactly like the as-drawn path.

---

## Known limitations (documented, not bugs)

- Models the **free-ligand dominant species**, not metal-assisted deprotonation that
  can occur upon coordination (e.g. amide N–H deprotonation by a bound metal).
- Microspecies approximation: each ionizable group is treated independently; strongly
  coupled equilibria are not solved.
- Coverage is limited to the tabulated groups; molecules with exotic ionizable motifs
  fall through unchanged.

These are acceptable for a first-order, deterministic, offline correctness upgrade and
are surfaced in the module docstring and the `des-chemistry` skill notes.

---

## Testing (TDD, deterministic, no LLM, no network)

**`tests/test_protonation.py`** — engine:
- Carboxylic acid `CC(=O)O` @ pH 7 → carboxylate, net −1, one deprotonated group.
- Aliphatic amine `CCN` @ pH 7 → ammonium, net +1.
- Glycine `NCC(=O)O` @ pH 7 → zwitterion: net 0, one protonated amine + one
  deprotonated acid.
- Glycine @ pH 1 → cation (net +1); @ pH 12 → anion (net −1). Locks pH-dependence.
- Glycerol `OCC(O)CO` @ pH 7 → unchanged (alcohols not ionized).
- Imidazole around pKa 7 → state flips between pH 6 and pH 8.
- Un-tabulated ionizable / exotic group → passthrough, unchanged.
- Invalid SMILES → passthrough, never raises.
- Table integrity: every SMARTS compiles; every entry has a numeric pKa and a valid
  `kind`.

**`tests/test_coordination.py`** (extend) — `_is_donor`:
- Protonated ammonium N⁺ (degree-1, with H's) → not counted as a donor.
- Deprotonated carboxylate O⁻ → still a donor; carboxylate still collapses to one site.
- Regression: existing neutral-molecule expectations unchanged.

**`tests/test_claim_grounding.py`** (extend):
- `structural_facts("NCC(=O)O")` (pH=None) → identical to current Phase 1 assertions
  (N+2O donors, denticity 2). Proves zero regression.
- `structural_facts("NCC(=O)O", pH=7.0)` → denticity 1, donors 2 O (N⁺ excluded).
- `ground_coordination("NCC(=O)O", "bidentate N,O-chelator", pH=7.0)` → not trivially
  verified as N,O at pH 7 (N protonated) — demonstrates the correctness gain.
- `as_prompt_block()` contains the species clause when `pH` is set, and does not when
  `pH is None`.

**`tests/test_metal_workflows_grounding.py`** (extend):
- `run_metal_binding_screen(..., binding_pH=7.0)` runs; `claim_verdicts` reflect
  species-aware coordination.
- `binding_pH` defaults to 7.0; full suite stays green, 0 failures.

Net: ~20 new deterministic tests.

---

## Out of scope

- Tautomer enumeration/canonicalization (follow-up 1A-bis).
- Making `stability_rules` / `selectivity` ΔlogK pH-dependent.
- Making the DES partner search protonation-aware (neat by design).
- User-facing report/CLI `--pH` surface (could be a later, separate increment).
- Any ML pKa model or online lookup (must stay offline + deterministic).

---

## Verification

1. `pytest tests/test_protonation.py -v` — engine on known molecules; zwitterion and
   pH-extremes; passthrough safety.
2. `pytest tests/test_coordination.py tests/test_claim_grounding.py -v` — `_is_donor`
   species cases and the pH=None regression guard.
3. `pytest tests/ -q` — full suite green (current baseline 684), proving the
   `pH=None` default changed nothing.
4. Backend-agnostic spot check: protonation verdicts are identical with the LLM
   disabled and under any provider (deterministic by construction).
