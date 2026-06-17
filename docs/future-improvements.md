# Future Improvements Roadmap

This document tracks the next useful extensions for DES-Agent after the current router and example updates.

## Recently Completed

1. Example benchmark suite
   - Curated examples now act as a regression benchmark for routing, DES screening, viscosity, and metal-binding.

2. Machine-readable exports
   - DES runs now emit JSON, CSV, and a run manifest automatically.

3. Stronger natural-language normalization
   - Plain-language requests now get normalized before routing, including salt and free-base clarification.

4. Active-learning feedback loop
   - Labeled run memory can now be reused from a single run or a whole history directory so later DES runs can learn from prior `good` / `bad` feedback more directly.

5. Molecule-name resolution
   - All CLI entry points now accept common molecule names ("ethanol", "betaine", "glycerol") instead of requiring SMILES. `list-molecules` command prints the full dictionary.

6. Deterministic chemistry grounding layer
   - Source-side structural-fact injection and output-side claim verification (family labels, DES H-bond complementarity) run automatically when an LLM is configured. Contradicted claims are flagged and the candidate takes a −0.25 ranking penalty.

7. Protonation-aware metal binding
   - `run_metal_binding_screen` and `run_metal_selectivity_screen` accept `binding_pH` (default 7.0); the dominant protonation state of each ligand is computed via `dominant_species(smiles, pH)` before coordination profiling.

8. Reality-anchored DES partner proposals
   - Before each LLM brainstorm a menu of up to 30 known, real partner molecules (role-matched to component A) is injected into the prompt. After brainstorm every LLM proposal is graded as `known`, `novel_plausible`, or `novel_implausible`; implausible proposals are demoted (−0.25) or dropped.

## Next Up

1. Metal-ligand brainstorm anchoring
   - Same pattern as DES partner reality anchoring, applied to the metal-binding workflow's `brainstorm_ligands` step. Separate spec/plan; explicitly deferred from the DES partner work.

2. Expanded common-names registry
   - Add more entries to `artifacts/molecule_names/common_names.json`, especially pharmaceutical actives used as DES component A candidates.
