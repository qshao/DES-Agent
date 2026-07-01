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

9. Readable iteration trajectories
   - All three iterative workflows (DES multi-cycle, metal-selectivity, selectivity-DES pipeline) now capture per-cycle snapshots (shortlist, entrants/dropouts, family ledger, convergence reason) and attach them as a `SearchTrajectory` to their outcome. The CLI prints a compact per-cycle trace to stderr and writes a full `trajectory.md` Markdown artifact to `--output-dir` when set. Capture is best-effort and never affects search results.

10. Chemical-awareness layer — iterative efficiency and accumulated knowledge
    - **H-bond complementarity ranking**: `rank_by_hbond` applies a ±0.10 post-prediction ranking adjustment based on DES H-bond complementarity with component A. Deterministic and LLM-agnostic.
    - **Near-miss analogue expansion**: structural analogues are generated not only from confirmed hits but also from candidates just below the DES/binding threshold, targeting the productive chemical boundary.
    - **UCB1 family scoring**: replaces binary saturation (fixed hit-rate threshold) with a UCB1 exploration score per family. The LLM context now receives families ranked by exploration value rather than a flat saturation list.
    - **Adaptive transform selection**: per-transform hit/fail counts are tracked cross-cycle; Laplace-smoothed hit rates bias the RDKit reaction transform order so historically productive transforms are applied first.
    - **Functional-group SAR tracking**: `StructuralFacts.family_features` tags are accumulated as hit/fail counters and injected into brainstorm prompts as sub-family SAR ("prefer polyol, avoid ester").
    - **Cross-run persistence**: accumulated family scores, scaffold counts, and FG SAR are serialised into `RunMemory` and surfaced as memory notes at the start of the next run on the same `component_a`.

## Next Up

1. Metal-ligand brainstorm anchoring
   - Same pattern as DES partner reality anchoring, applied to the metal-binding workflow's `brainstorm_ligands` step. Separate spec/plan; explicitly deferred from the DES partner work.

2. Expanded common-names registry
   - Add more entries to `artifacts/molecule_names/common_names.json`, especially pharmaceutical actives used as DES component A candidates.

3. JSON trajectory export
   - A machine-readable sibling of `trajectory.md` (e.g. `trajectory.json`) for dashboard or downstream analysis use. The `SearchTrajectory` dataclass is already serialisable; the writer is the only addition needed.

4. Legacy final-report table cleanup
   - The dense pipe-table blocks in `reporting.py` predate the trajectory feature. Restructuring them to match the cleaner trajectory-style narrative is a separate, higher-churn effort deferred from the trajectory work.

5. `total_cycles` accuracy under best-effort capture failure (metal workflows)
   - In `metal_binding_selectivity.py` and `selectivity_des_pipeline.py`, `SearchTrajectory.total_cycles` counts successful snapshots rather than cycles actually run. If a snapshot build fails (rare), the header line under-reports. Fix: track a separate `cycles_run` counter in those two workflows.

6. Pharmacophore-based candidate clustering
   - Replace Murcko scaffold (topology-only) with pharmacophore feature clustering (HBD/HBA positions, aromaticity, charge). More meaningful for DES since two different scaffolds with matching H-bond geometry can both form eutectics. Requires new 3D-feature infrastructure.

7. Tanimoto diversity penalty for evaluated failures
   - Compute Morgan fingerprint similarity of new proposals to already-evaluated failures and apply a small deprioritization penalty to proposals that are too structurally similar to known-bad candidates. The `prior_evaluated_smiles` set is already passed; fingerprint coverage tracking is the remaining addition.
