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

11. Metal-ligand brainstorm anchoring (source injection + output reality gate)
    - Source-side: `known_ligand_menu(metal_ion, limit=15)` in `partner_registry.py` pulls all registry molecules with ≥1 donor atom, scores each via `rule_based_log_k`, and returns the top-N sorted by predicted log K (cached per metal ion). Injected into `ligand_brainstorm_prompt` and `ligand_selectivity_brainstorm_prompt` as a ranked anchor list with coordination summaries (e.g. "bidentate (N,O)").
    - Output-side: `ground_ligand_reality(metal_ion, smiles)` in `claim_grounding.py` returns a `PartnerVerdict`; drops proposals with invalid SMILES, zero donor atoms, or structural sanity failures. Applied in both `metal_binding_screen.py` and `metal_binding_selectivity.py`; dropped proposals emit `[GROUNDING]` warnings.

12. `total_cycles` accuracy
    - `SearchTrajectory.total_cycles` previously counted successful snapshots rather than cycles actually run. Fixed in `metal_binding_selectivity.py` and `selectivity_des_pipeline.py` by tracking a dedicated iteration counter.

13. JSON trajectory export
    - `write_trajectory_json_artifact` in `trajectory.py` writes `trajectory.json` alongside `trajectory.md` using an atomic `NamedTemporaryFile` + `Path.replace` write. Called automatically by the CLI when `--output-dir` is set.

14. Tanimoto diversity penalty
    - `_apply_tanimoto_diversity_penalty` in `orchestrator.py` computes Morgan fingerprint (radius=2, 2048-bit) similarity of each new proposal to all DES-negative prior evaluations and subtracts a scaled `ranking_score` penalty when max similarity ≥ 0.70. DES-positive prior results are never penalized. Fingerprints are cached in `_FAIL_FP_CACHE` (module-level dict) so per-cycle cost is O(new failures).

15. Code-review bug fixes
    - **Reality gate SMILES mismatch**: `_apply_ligand_reality_gate` now canonicalizes `b.smiles` before building the drop set, so non-canonical LLM output is correctly filtered against the canonical SMILES stored in `proposals` by `_deduplicate_proposals`.
    - **Reality gate extraction**: the 15-line gate block was copy-pasted in both metal workflow files; extracted into a shared `_apply_ligand_reality_gate` helper in `workflows/_metal_helpers.py`.

16. DFT validation stage for metal-selectivity (`--dft-validate`)
    - Optional, flag-gated post-ranking DFT refinement for the metal-selectivity workflow. Activated only by explicit `--dft-validate`; never auto-triggered.
    - Pipeline: SMILES → RDKit MMFF94 embed → xTB GFN2 geometry optimization → gpu4pyscf B3LYP-D3(BJ)/def2-SVP single-point (free ligand, gas phase).
    - HOMO energy used as HSAB donor-softness proxy (calibration: −9.5 eV = hard donor, −7.5 eV = soft donor) to compute a ±0.05 composite-score nudge via `dft_selectivity_adjustment`. DFT tiebreaks the rule-based ranking; never overrides it.
    - LLM nominates 1–`--dft-top-n` candidates for DFT from the top shortlist; fallback to top-N by `composite_score` when no LLM is configured.
    - DFT failures are non-fatal: run completes with rule-based ranking unchanged, per-candidate warnings appear in the report.
    - Report gains `dft_homo_ev` and `dft_donor_chg` columns for nominated candidates; a `DFT validation: B3LYP-D3(BJ)/def2-SVP` block summarises the method and any warnings.
    - Startup dependency check verifies `gpu4pyscf` and `xtb` are available before committing to DFT computation.

## Next Up

1. Expanded common-names registry
   - Add more entries to `artifacts/molecule_names/common_names.json`, especially pharmaceutical actives used as DES component A candidates.

2. Legacy final-report table cleanup
   - The dense pipe-table blocks in `reporting.py` predate the trajectory feature. Restructuring them to match the cleaner trajectory-style narrative is a separate, higher-churn effort.

3. Pharmacophore-based candidate clustering
   - Replace Murcko scaffold (topology-only) with pharmacophore feature clustering (HBD/HBA positions, aromaticity, charge). More meaningful for DES since two different scaffolds with matching H-bond geometry can both form eutectics. Requires new 3D-feature infrastructure.
