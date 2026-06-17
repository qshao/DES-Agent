# Reality-Anchored DES Partner Proposals — Design Spec

**Date:** 2026-06-17
**Status:** Draft for review
**Scope:** DES partner brainstorming only (component-A → partner). Metal-ligand
brainstorming is an explicit fast-follow, out of scope here.

---

## 1. Problem

The LLM brainstorm step (`brainstorm_candidates` → `candidate_brainstorm_prompt`)
proposes DES partner molecules from parametric memory. Two failure modes today:

1. **Invented molecules** — valid but non-real SMILES the model fabricated.
2. **Implausible pairings** — real molecules that cannot form a DES with
   component A (no H-bond complementarity).

Nothing currently anchors proposals to real compounds or rejects junk. We want
the LLM to **prefer real, attested partners** and to **flag + demote** any
off-registry proposal that fails a deterministic plausibility gate — consistent
with the existing deterministic grounding layer (`chemistry/claim_grounding.py`).

## 2. Goal

Add a **reality-anchoring** layer to DES partner proposals with two halves,
mirroring the existing `claim_grounding` source-side/output-side split:

- **Source-side:** inject a menu of known, real partners (filtered to the role
  complementary to component A) into the brainstorm prompt.
- **Output-side:** grade every LLM proposal with a deterministic
  `PartnerVerdict` and keep / demote / drop accordingly.

Everything is deterministic and offline — identical verdicts on any LLM backend,
and the gate still runs when the LLM is disabled (though menu injection only
fires when an LLM is configured).

## 3. Global Constraints

- **Offline + deterministic.** No network, no embeddings, no new runtime deps.
  RDKit only (already a dependency).
- **Reuse existing chemistry.** `hbond_profile`, `des_hbond_complementarity`,
  `canonicalize_smiles` — do not reinvent.
- **Backward compatible.** New prompt param defaults to "no menu"; existing
  prompt tests and the disabled-LLM path are byte-identical. Deterministic
  example baselines (no LLM) do not move.
- **Never raise into the proposal/prompt path.** All new entry points pass
  through safely on error, matching `claim_grounding`'s `_passthrough` pattern.
- **Role vocabulary** matches the existing code: `"HBD" | "HBA" | "amphoteric"`
  (and `"none"`). Do not introduce `"both"`.

## 4. Data Sources (no new data fetched)

| Use | Source | Notes |
|-----|--------|-------|
| Known-set membership | `artifacts/molecule_names/common_names.json` (57 role-tagged) **∪** `artifacts/melting_points/experimental.json` (hundreds, InChIKey-keyed real compounds) | Both already ship in repo |
| Anchor menu (curated) | `common_names.json` entries with `role ∈ {HBA, HBD, amphoteric}` | High-confidence role tags + display names |
| Anchor menu (auto-tagged) | `experimental.json` compounds, role derived via `hbond_profile(smiles).role` | Free breadth from real data; display name = SMILES |

**Membership key:** canonical InChIKey via `Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))`,
**recomputed from each source's stored SMILES** (do not trust pre-stored keys —
recomputing guarantees both sources and incoming proposals use the same InChIKey
form). Multi-component SMILES (salts, e.g. `C[N+](C)(C)CCO.[Cl-]`) are keyed
as-is so the registry's choline chloride matches a proposal of the same salt.

**Loader shapes (read exactly these):**
- `common_names.json` → `{"version": str, "entries": [ {"smiles": str, "names": [str], "role": str}, ... ]}`. Iterate `data["entries"]`.
- `experimental.json` → `{"entries": { "<INCHIKEY>": {"smiles": str, "tm_k": float, ...}, ... }, ...}`. Iterate `data["entries"].values()` and read each `["smiles"]`.

## 5. Architecture

### 5.1 New module: `des_multi_agent/chemistry/partner_registry.py`

Module-level, lazily-cached loads (compute once on first use):

```python
def known_inchikeys() -> frozenset[str]:
    """Canonical InChIKey set from common_names.json ∪ experimental.json.
    Cached. Missing/invalid artifacts degrade gracefully to whatever loads."""

def is_known(smiles: str) -> bool:
    """True if the canonical InChIKey of `smiles` is in the known set.
    Returns False on unparseable SMILES (never raises)."""

@dataclass(frozen=True)
class MenuEntry:
    smiles: str
    display_name: str   # curated name, or the SMILES for auto-tagged entries
    role: str           # "HBD" | "HBA" | "amphoteric"

def known_partner_menu(role: str, limit: int = 30) -> list[MenuEntry]:
    """Menu entries that can serve the *wanted partner role* `role`.

    An entry serves `role` when:
        entry.role == role            # exact role match
        or entry.role == "amphoteric" # amphoteric can act as either
        or role == "amphoteric"       # caller wants any H-bonder

    Curated registry entries come first (higher confidence), then
    auto-tagged experimental compounds, deduped by InChIKey, capped at `limit`."""

def structural_sanity(smiles: str) -> tuple[bool, str]:
    """Deterministic 'is this a sane small molecule' check.
    Fails (returns (False, reason)) when any holds:
      - unparseable SMILES
      - any atom outside {H, C, N, O, S, P, F, Cl, Br, I}
      - molecular weight outside (40, 400)
      - any radical electrons present
    Returns (True, "") otherwise."""
```

Auto-tagging rule (uses the existing profiler, no new heuristic):
`role = hbond_profile(smiles).role`; entries with `role == "none"` are excluded
from the menu (they serve no H-bond role).

### 5.2 Extend `des_multi_agent/chemistry/claim_grounding.py`

```python
@dataclass(frozen=True)
class PartnerVerdict:
    claim: str          # e.g. "partner reality: <smiles>"
    status: str         # "known" | "novel_plausible" | "novel_implausible"
    detail: str
    penalty: float      # 0.0 for known/novel_plausible; 0.25 for novel_implausible-demote
    disposition: str    # "keep" | "demote" | "drop"

def ground_partner_reality(component_a: str, candidate_smiles: str) -> PartnerVerdict:
    ...
```

Decision logic (in order; the whole body wrapped so it never raises):

1. `Chem.MolFromSmiles(candidate_smiles) is None`
   → `novel_implausible`, disposition `drop`, penalty `0.0`, detail `"invalid SMILES"`.
2. `is_known(candidate_smiles)`
   → `known`, disposition `keep`, penalty `0.0`, detail `"known/attested compound"`.
3. `ok, reason = structural_sanity(candidate_smiles); not ok`
   → `novel_implausible`, disposition `drop`, penalty `0.0`, detail `reason`.
4. `des_hbond_complementarity(component_a, candidate_smiles).label == "none"`
   → `novel_implausible`, disposition `demote`, penalty `0.25`,
     detail `"no H-bond complementarity with component A"`.
5. else
   → `novel_plausible`, disposition `keep`, penalty `0.0`,
     detail `f"novel; complementarity={label}"`.

**Error passthrough:** any internal exception → `novel_plausible`, `keep`,
penalty `0.0`, detail `"reality check skipped (internal error)"`. We never
demote or drop on our own failure (avoid misleading in the punishing direction).

**Invariants (enforced in `__post_init__`):**
- `disposition == "keep"` ⟺ `penalty == 0.0` and `status ∈ {known, novel_plausible}`.
- `disposition == "demote"` ⟹ `penalty > 0.0` and `status == "novel_implausible"`.
- `disposition == "drop"` ⟹ `penalty == 0.0` and `status == "novel_implausible"`.

### 5.3 Prompt injection: `des_multi_agent/llm/prompts.py`

`candidate_brainstorm_prompt` gains `known_partner_menu: list | None = None`
(list of `MenuEntry`). When non-empty, render before the diversity instruction:

```
Prefer partners from this list of known, real molecules. You MAY propose others,
but only with an explicit justification in the rationale:
  - choline chloride [HBA]
  - urea [HBD]
  - glycerol [amphoteric]
  ...
```

Default `None` → block omitted → existing prompt output unchanged (backward
compatible; existing prompt tests untouched).

### 5.4 Menu construction: `des_multi_agent/llm/base.py`

In `brainstorm_candidates`, before building the prompt (wrapped, non-fatal):

```python
wanted = _complementary_role(hbond_profile(component_a).role)
menu = known_partner_menu(wanted, limit=30)
```

`_complementary_role`: `HBA → "HBD"`, `HBD → "HBA"`,
`amphoteric`/`none` → `"amphoteric"` (menu returns any role). Pass `menu` into
`candidate_brainstorm_prompt(...)`. On any failure, fall back to `menu=None`
and append nothing (proposal path must not break).

### 5.5 Output grading: `des_multi_agent/orchestrator.py`

Inside the existing deterministic-grounding block (the `try` around lines
693–721, where `ground_des_plausibility` / `ground_family` already run and
`grounding_penalty_by_smiles` / `llm_warnings` already exist), add partner-reality
grading for **LLM-sourced** candidates only:

```python
llm_smiles = {c.smiles for c in llm_candidates}   # already have llm_candidates
drop_smiles: set[str] = set()
for item in annotated_results:
    smiles_b = item.result.curve.smiles_b
    if smiles_b not in llm_smiles:
        continue   # heuristic/discovery candidates are real by construction
    rv = ground_partner_reality(component_a, smiles_b)
    claim_verdicts.append(rv)
    if rv.disposition == "demote":
        grounding_penalty_by_smiles[smiles_b] = max(
            grounding_penalty_by_smiles.get(smiles_b, 0.0), rv.penalty)
        llm_warnings.append(f"[REALITY] {rv.detail} — demoted: {smiles_b}")
    elif rv.disposition == "drop":
        drop_smiles.add(smiles_b)
        llm_warnings.append(f"[REALITY] {rv.detail} — dropped: {smiles_b}")
```

After the grounding block, before `_apply_review_penalties`:

```python
if drop_smiles:
    annotated_results = [
        it for it in annotated_results
        if it.result.curve.smiles_b not in drop_smiles
    ]
```

Then the existing `grounding_penalty_by_smiles` / `_apply_review_penalties`
path applies demotions unchanged. `claim_verdicts` already flows into
`SearchOutcome` and the report.

### 5.6 Reporting: `des_multi_agent/reporting.py`

Extend the `claim_verdicts` render loop (around line 294) to recognise
`PartnerVerdict` statuses (it currently only handles `verified`/`contradicted`):

- `status == "known"` → `- ✓ known | {claim}`
- `status == "novel_plausible"` → `- ◆ novel (plausible) | {claim}`
- `status == "novel_implausible"` → `- ✗ implausible — {detail} | {claim}`

Existing `GroundingVerdict` rendering (`verified`/`contradicted`) is unchanged;
add the new branches alongside.

### 5.7 Skill docs: `.claude/skills/des-chemistry/SKILL.md`

Add a "Partner reality anchoring" subsection documenting `partner_registry`
(known set, menu, structural sanity) and `ground_partner_reality` as the
output-side entry point, plus the keep/demote/drop contract.

## 6. File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `des_multi_agent/chemistry/partner_registry.py` | Create | Known-set load, menu builder, structural sanity |
| `des_multi_agent/chemistry/claim_grounding.py` | Modify | `PartnerVerdict` + `ground_partner_reality` |
| `des_multi_agent/llm/prompts.py` | Modify | `known_partner_menu` param + anchor block |
| `des_multi_agent/llm/base.py` | Modify | Build complementary-role menu, pass to prompt |
| `des_multi_agent/orchestrator.py` | Modify | Grade LLM proposals; demote/drop; warn |
| `des_multi_agent/reporting.py` | Modify | Render `PartnerVerdict` statuses |
| `.claude/skills/des-chemistry/SKILL.md` | Modify | Document the layer |
| `tests/test_partner_registry.py` | Create | Registry, menu, sanity, safe paths |
| `tests/test_claim_grounding.py` | Modify | `ground_partner_reality` branches |

Reused unchanged: `chemistry/hbond.py` (`hbond_profile`,
`des_hbond_complementarity`), `chemistry_filter.py` (`canonicalize_smiles`).

## 7. Data Flow

```
component_a ──> hbond_profile.role ──> complementary role
                                          │
                          known_partner_menu(role, 30)
                                          │
              candidate_brainstorm_prompt(..., known_partner_menu=menu)
                                          │
                                   LLM proposals
                                          │
        for each LLM-sourced annotated result:
              ground_partner_reality(component_a, smiles)
                 ├─ known / novel_plausible → keep
                 ├─ novel_implausible (no complementarity) → demote −0.25
                 └─ novel_implausible (bad sanity / invalid) → drop
                                          │
        claim_verdicts ─> SearchOutcome ─> report (✓ known / ◆ novel / ✗ implausible)
```

## 8. Testing (TDD, all deterministic)

**`tests/test_partner_registry.py`:**
- `is_known` True for choline chloride salt SMILES from the registry; True for a
  compound present only in `experimental.json`; False for an obviously invented
  SMILES; False (no raise) for an unparseable string.
- `structural_sanity`: passes ethanol/urea/glycerol; fails a boron compound
  (`B`), fails an oversized molecule (MW > 400), fails a radical species.
- `known_partner_menu("HBD", 5)`: non-empty, every entry role serves an HBD
  request, ≤ 5, curated entries precede auto-tagged, no duplicate InChIKeys.

**`tests/test_claim_grounding.py` (additions):**
- known compound → `status="known"`, `disposition="keep"`, penalty 0.
- novel real complementary partner (valid, sane, complementarity ≠ none, not in
  known set) → `novel_plausible`, `keep`.
- valid-but-non-complementary novel molecule → `novel_implausible`, `demote`,
  penalty 0.25.
- bad-element molecule → `novel_implausible`, `drop`, penalty 0.
- invalid SMILES → `novel_implausible`, `drop`.
- `PartnerVerdict` invariant violations raise `ValueError`.

**Regression:** `pytest tests/ -q` green; prompt tests unchanged (menu defaults
None); benchmark baselines for deterministic (no-LLM) examples unchanged.

## 9. Verification

1. `pytest tests/test_partner_registry.py tests/test_claim_grounding.py -v` — all pass with LLM disabled (proves determinism / LLM-agnosticism).
2. Inspect `known_partner_menu("HBD")` interactively — confirm it contains real
   HBDs (urea, glycerol, a carboxylic acid) and no junk.
3. Run a `des` search with an LLM configured and a prompt induced to invent a
   non-real partner; assert a `novel_implausible` verdict appears in
   `claim_verdicts`, the candidate is dropped/demoted, and the report shows
   `✗ implausible`.
4. `pytest tests/ -q` — full suite green; deterministic example benchmark
   baselines unchanged.

## 10. Out of Scope (explicit)

- Metal-ligand brainstorm anchoring (`brainstorm_ligands*`) — identical pattern,
  separate spec/plan (fast-follow).
- Manual expansion of `common_names.json` and external/public DES dataset import
  — optional later quality enhancements; not required for this feature.
- Embedding/vector RAG — rejected; corpus is small and the curated-registry +
  deterministic-gate approach aligns with the existing grounding layer.
- Any change to the ML melting-point/viscosity/stability predictors.
