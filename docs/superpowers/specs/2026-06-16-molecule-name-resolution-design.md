# Molecule Name Resolution Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to supply common molecule names (e.g. "choline chloride", "urea") anywhere the system currently expects a SMILES string, with zero network dependency.

**Architecture:** A single resolution module (`des_multi_agent/chemistry/name_resolution.py`) backed by a bundled JSON dictionary (`artifacts/molecule_names/common_names.json`). Resolution is applied exactly once, at each user-facing input boundary (CLI argument parser, FastAPI request handler, candidates-file loader). The pipeline internals always receive canonical SMILES and are untouched.

**Tech Stack:** Python stdlib (`difflib`, `json`), RDKit (`Chem.MolFromSmiles`, `Chem.MolToSmiles`) for SMILES validation and canonicalisation. No new dependencies.

---

## 1. New Files

| Path | Purpose |
|------|---------|
| `artifacts/molecule_names/common_names.json` | Bundled name dictionary (~80 entries) |
| `des_multi_agent/chemistry/name_resolution.py` | Lookup + pass-through logic |
| `tests/test_name_resolution.py` | Unit + integration tests |

## 2. Modified Files

| Path | Change |
|------|--------|
| `des_multi_agent/cli.py` | Resolve `--component-a`, `--component-b`, and candidates-file entries before invoking any workflow |
| `des_multi_agent/server.py` | Resolve `component_a` and `component_b` in the `POST /search` request handler |

---

## 3. Dictionary Format

File: `artifacts/molecule_names/common_names.json`

```json
{
  "version": "1.0",
  "entries": [
    {
      "smiles": "C[N+](C)(C)CCO.[Cl-]",
      "names": ["choline chloride", "ChCl", "choline Cl"],
      "role": "HBA"
    },
    {
      "smiles": "NC(N)=O",
      "names": ["urea", "carbamide"],
      "role": "HBD"
    }
  ]
}
```

**Fields:**
- `smiles` — RDKit-canonicalized SMILES string (verified at build time by the dictionary integrity test)
- `names` — list of accepted name strings; first entry is the canonical display name used in error messages and `list-molecules` output; remaining entries are synonyms; all matching is case-insensitive and whitespace-normalised
- `role` — one of `"HBA"`, `"HBD"`, `"amphoteric"`, `"ligand"`, `"solvent"`; informational only, not used by the resolver

**Planned scope (~80 entries):**

| Category | Examples |
|----------|---------|
| HBA — quaternary ammonium/phosphonium salts | choline chloride, choline bromide, betaine, TBAB, TBAC, choline acetate |
| HBD — polyols | glycerol, ethylene glycol, propylene glycol, 1,4-butanediol, diethylene glycol, triethylene glycol, sorbitol |
| HBD — organic acids | oxalic acid, malonic acid, citric acid, malic acid, tartaric acid, succinic acid, glutaric acid, adipic acid, lactic acid, levulinic acid, acetic acid, formic acid, caprylic acid |
| HBD — amides / ureas | urea, thiourea, acetamide, caprolactam, N-methylurea |
| HBD — phenolics | phenol, menthol, thymol, carvacrol |
| HBD — sugars | fructose, glucose, sucrose |
| Amino acids | glycine, alanine, proline, serine, cysteine, histidine |
| Ligands (metal workflows) | EDTA, NTA, DTPA, 8-hydroxyquinoline |
| Common solvents / reference | water, ethanol, methanol, DMSO, acetonitrile |

---

## 4. Resolution Module

File: `des_multi_agent/chemistry/name_resolution.py`

### Public API

```python
def resolve_name(text: str) -> str | None:
    """Return canonical SMILES if text matches a known name (case-insensitive),
    else return None. Does not validate whether text is a SMILES."""

def resolve_to_smiles(text: str) -> str:
    """Return a canonical SMILES for text, which may be:
      - an already-valid SMILES string  → returned unchanged (after RDKit canonicalisation)
      - a known molecule name or synonym → looked up and returned as SMILES
    Raises ValueError with a user-facing message if neither applies.
    """

def list_molecules() -> list[dict]:
    """Return all dictionary entries sorted by role then canonical name.
    Each dict has keys: smiles, canonical_name, synonyms, role.
    Used by the 'des-agent list-molecules' subcommand."""
```

### Resolution logic in `resolve_to_smiles`

1. Attempt `Chem.MolFromSmiles(text)` — if successful, return `Chem.MolToSmiles(mol)` (canonical form)
2. Normalise `text`: `text.strip().lower()`, collapse internal whitespace
3. Look up normalised text in the prebuilt `_NAME_INDEX` dict (built at module load from the JSON)
4. If found, return the entry's SMILES
5. If not found, compute `difflib.get_close_matches(normalised, _NAME_INDEX.keys(), n=1, cutoff=0.75)`
6. Raise `ValueError` with:
   - the unrecognised input
   - the closest match suggestion if one exists
   - a pointer to `des-agent list-molecules`
   - a reminder that SMILES strings are also accepted directly

### Error message format

```
Unknown molecule: 'choline chloride hcl'
  → Did you mean 'choline chloride'?  (SMILES: C[N+](C)(C)CCO.[Cl-])
  → Run 'des-agent list-molecules' to see all supported names.
  → If you have a SMILES string, pass that directly instead.
```

### Internal index

At module load time, `common_names.json` is read once and `_NAME_INDEX: dict[str, str]` is built:
`{normalised_name: smiles}` for every name and synonym in every entry. Module-level singleton — no repeated I/O.

---

## 5. CLI Integration

File: `des_multi_agent/cli.py`

### `--component-a` and `--component-b`

After argument parsing, before any workflow is invoked:

```python
args.component_a = resolve_to_smiles(args.component_a)
if args.component_b:
    args.component_b = resolve_to_smiles(args.component_b)
```

A `ValueError` from `resolve_to_smiles` is caught at the top level and printed as a clean error (not a traceback) before exiting with code 1.

### `--candidates-file`

The file-loading helper reads one entry per line. After this change, each line is passed through `resolve_to_smiles`. Valid SMILES lines pass through unchanged; known name lines are resolved to SMILES. Lines that are neither a valid SMILES nor a known name print a `[WARNING]` to stderr (including the "did you mean" suggestion if applicable) and are skipped; the run continues with the remaining candidates. This matches the existing behaviour for chemically invalid SMILES entries.

### `des-agent list-molecules` subcommand

New subcommand (no arguments). Calls `list_molecules()` and prints a table:

```
Molecule Name              SMILES                          Role
─────────────────────────────────────────────────────────────────
choline chloride           C[N+](C)(C)CCO.[Cl-]           HBA
  aliases: ChCl, choline Cl
betaine                    C[N+](C)(C)CC(=O)[O-]          HBA
...
urea                       NC(N)=O                        HBD
  aliases: carbamide
glycerol                   OCC(O)CO                       HBD
  aliases: glycerin, 1,2,3-propanetriol
...
```

Grouped by role, alphabetical within each group.

---

## 6. FastAPI Integration

File: `des_multi_agent/server.py`

In the `POST /search` handler, resolve before passing to the orchestrator:

```python
try:
    component_a = resolve_to_smiles(request.component_a)
    component_b = resolve_to_smiles(request.component_b) if request.component_b else None
except ValueError as exc:
    raise HTTPException(status_code=422, detail=str(exc))
```

Same pattern for any other endpoint that accepts molecule identifiers.

---

## 7. Tests

File: `tests/test_name_resolution.py`

### Group 1: Dictionary integrity
- Load `common_names.json`; verify every SMILES parses with RDKit and round-trips (`smiles → mol → canonical_smiles → mol` succeeds)
- Verify no normalised name appears more than once across all entries (no collision in `_NAME_INDEX`)
- Verify every entry has at least one name and a non-empty SMILES

### Group 2: Lookup correctness
- `resolve_to_smiles("choline chloride")` returns the expected SMILES
- `resolve_to_smiles("CHOLINE CHLORIDE")` returns same SMILES (case-insensitive)
- `resolve_to_smiles("ChCl")` returns same SMILES (synonym)
- `resolve_to_smiles("  urea  ")` returns `"NC(N)=O"` (whitespace stripping)
- `resolve_name("urea")` returns the SMILES; `resolve_name("not_a_molecule")` returns `None`

### Group 3: SMILES pass-through
- `resolve_to_smiles("C[N+](C)(C)CCO.[Cl-]")` returns a valid canonical SMILES (already a SMILES)
- `resolve_to_smiles("NC(N)=O")` returns `"NC(N)=O"` (already canonical)
- Non-canonical SMILES like `"O=C(N)N"` is accepted and returned in canonical form

### Group 4: Error paths
- Unknown name raises `ValueError` with the input in the message
- Close match (e.g. "choline chioride") triggers a "did you mean" suggestion in the error
- Completely unrecognised string raises `ValueError` with pointer to `list-molecules`

### Group 5: CLI integration
- `test_cli.py`: invoke CLI with `--component-a "choline chloride"` (monkeypatched prediction); assert the pipeline is called with the canonical SMILES `"C[N+](C)(C)CCO.[Cl-]"`
- Invoke CLI with an unknown name; assert exit code 1 and error message contains the unknown name

---

## 8. Out of Scope

- CAS number lookup (no offline CAS → SMILES mapping; would require a separate bundled table)
- IUPAC name parsing (requires OPSIN, a Java library; no offline Python equivalent)
- Fuzzy phonetic matching (edit-distance via `difflib` is sufficient; Levenshtein is not in stdlib)
- Network fallback to PubChem (system must be fully offline)
- Auto-updating the dictionary from external sources (manual curation + PR process only)
