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

17. DFT result caching and pH-aware protonation
    - **Caching**: `des_multi_agent/chemistry/dft_cache.py`'s `cached_compute_dft_properties` wraps `compute_dft_properties` with a SQLite cache at `artifacts/dft_cache/dft_results.sqlite3`, keyed on `(species_smiles, dft_method)` — never the raw input SMILES, so different SMILES spellings of the same molecule correctly share one entry. Only successful DFT results are cached; a cache-layer failure (corrupt file, I/O error) silently falls back to an uncached direct call. No TTL — same species + same method always gives the same physics.
    - **pH-aware DFT**: `compute_dft_properties(smiles, pH=None)` gained an optional `pH` parameter. When set, it computes the actual dominant protonation state at that pH (via the existing `chemistry.protonation.dominant_species`) with its real formal charge, instead of always assuming the neutral input — e.g. a carboxylic acid ligand at pH 7.4 is now computed as the deprotonated carboxylate (`charge=-1`), which has a meaningfully different HOMO energy than the neutral acid. `pH=None` (the default for direct calls) preserves the exact prior neutral-species behavior.
    - The metal-selectivity workflow's DFT stage now threads its existing `binding_pH` parameter (default 7.0, previously only used for coordination-claim grounding) into the DFT call for the first time — this is a deliberate accuracy improvement, not a side effect: any ligand that ionizes at pH 7.0 now gets DFT computed on its physiologically-relevant charged form rather than the drawn-neutral form.
    - `DFTResult` gained `species_smiles` (the species actually computed), `ph` (the pH used), and `from_cache` (whether this result came from the cache) fields.

18. Multi-off-target metal selectivity
    - `--competitor-metal-ion` accepts a comma-separated list of off-target metals (`Zn2+,Fe3+,Ni2+`), not just a single competitor. `run_metal_selectivity_screen`'s (and `run_selectivity_des_pipeline`'s) `competitor_metal` parameter widens to `str | list[str]` — never renamed, so every existing single-competitor call site keeps working unchanged.
    - Ranking is worst-case: `delta_log_k = log_k_target − max(log_k of all off-targets)`. A ligand only ranks well if it beats every off-target simultaneously, not just one — the practical question a wet-lab chemist actually asks when several other metals are present in a mixture.
    - `SelectivityResult` gained `log_k_competitors: dict[str, float]` (full per-metal breakdown) and `worst_competitor_metal: str` (which off-target is the limiting one for that candidate). `log_k_competitor`/`delta_log_k` keep their existing names but always hold the worst-case value.
    - DFT tiebreaking and selectivity grounding now compute against each candidate's own `worst_competitor_metal` rather than a single outcome-wide competitor — different candidates can have different limiting off-targets, and the tiebreak/grounding logic follows whichever metal is actually the bottleneck for that specific ligand.
    - A candidate is dropped entirely (not partially scored) if any single off-target's log K prediction fails. Duplicate off-targets in the input are de-duplicated, order-preserving; an empty off-target list raises `ValueError`.
    - The report gains an `off_target_breakdown` column (only when more than one off-target is given) showing every off-target's log K per candidate, e.g. `Zn2+=10.20, Fe3+=11.80*` — the asterisk marks the limiting metal. A single off-target produces byte-identical report output, CLI behavior, and LLM prompt text to before this feature.

19. Code-review fixes for DFT validation/caching
    - **xtb geometry charge**: `_xtb_optimize` ignored the pH-derived net formal charge — xtb defaulted to charge=0 while the subsequent DFT single-point correctly used the pH-aware charge, so charged (ionized) species were geometry-optimized as if neutral before an internally inconsistent charged single-point. `_xtb_optimize(mol, charge=...)` now passes `--chrg` to xtb so geometry and single-point agree.
    - **`dft_method` mislabeling**: `cached_compute_dft_properties` accepted a `dft_method` argument used only as the SQLite cache-key label — `compute_dft_properties` has no method-selection support and always runs B3LYP-D3(BJ)/def2-SVP, so a non-default `dft_method` would have silently cached a B3LYP result under the wrong method's key. It now raises `ValueError` for any `dft_method` other than `DEFAULT_DFT_METHOD` instead of corrupting the cache.
    - **Private cross-module import**: `dft_selectivity.py` reached into `stability_rules.py`'s private `_metal_softness`. Added a public `metal_softness()` accessor (matching the existing `irving_williams_offset` pattern) and switched the import.

20. vLLM LLM backend
    - `--llm-config` now also accepts `provider: vllm`, a new `VLLMProvider` alongside the existing `ollama`/`openai`/`gemini`/`custom_http` backends — none of which are changed or deprecated. Reuses the existing `payload_style="openai"` request/response format since vLLM's OpenAI-compatible server (`vllm serve <model>`) speaks the same wire format as `OpenAIProvider`/`CustomHTTPProvider`.
    - No `model_name` allow-list (unlike `ollama`'s `_SUPPORTED_OLLAMA_MODEL_PREFIXES`) since a vLLM server process commits to exactly one model at launch; the config's `model_name` is just a label for whichever model the operator already started. `api_key_env` is optional, matching how local unauthenticated servers are already handled for `custom_http`.
    - See `llm.vllm_example.yaml` for a ready-to-edit config, and the README's "Optional vLLM run" section for the `vllm serve` launch command and GPU prerequisites.
    - `doctor --check llm`'s connectivity probe was fixed alongside this feature: an HTTP error response (404, 401, 405, ...) is now treated as *reachable* (the server responded, just not to a bare `GET` on its base URL), so OpenAI-compatible backends (`vllm`, `openai`, `custom_http`) no longer always report `not reachable` even when healthy. Only a true connection failure (refused, timed out, DNS error) still warns.

21. vLLM vs Ollama throughput benchmark (same DES example, same checkpoint, `--component-a ethanol --n 10`)
    - **Methodology**: for each backend/model pair, ran `examples.demo_des_search` end to end (family selection, brainstorm, per-candidate review, uncertainty, explanations, critique, contradiction detection) and measured wall-clock time. Two same-model comparisons were done by identifying and downloading the exact Hugging Face checkpoint behind each locally-pulled Ollama model (matched via architecture tag and license): `google/gemma-4-12B-it` (bf16, matches Ollama's `gemma4:12b` Q4_K_M) and `Qwen/Qwen3.6-35B-A3B-FP8` (matches Ollama's `qwen3.6:latest` Q4_K_M). vLLM was run with `--language-model-only` (text-only, no vision/audio encoder) since this workload has no multimodal input.
    - **Results**: Gemma 4 12B — Ollama (Q4_K_M) 41m1s vs vLLM (bf16) 33m52s, vLLM ~17% faster. Qwen 3.6 35B-A3B — Ollama (Q4_K_M) 6m3s vs vLLM (FP8) 11m4s, Ollama ~1.8x faster. **The result is mixed, not a clean win for either backend.**
    - **Why Qwen lost on vLLM**: this ran on an NVIDIA GB10 (Blackwell, compute capability sm_121, aarch64) devkit, whose FP8 kernel support in vLLM 0.24.0 is immature for MoE models. Getting the server to start at all required three targeted fixes for crashes that are specific to this chip/model/vLLM-version combination, not general DES-Agent issues: `--moe-backend triton` and `--linear-backend triton` (the default DeepGEMM/CUTLASS FP8 kernels raised `Assertion error ... Unknown SF transformation` / `Unknown recipe` — this chip's FP8 scaling-factor layout isn't recognized by DeepGEMM's heuristics yet), plus `VLLM_DEEP_GEMM_WARMUP=skip` (the kernel-warmup step unconditionally tries DeepGEMM regardless of the chosen backend). These forced fallback kernels are slower than vLLM's well-optimized path on more mainstream hardware (H100/A100), which is the most likely explanation for the loss on a 35B MoE model.
    - **Confounds that make this not a pure backend comparison**: quantization format differs per pair (Q4_K_M vs bf16 for Gemma; Q4_K_M vs FP8 for Qwen), and the GB10 box was shared with other unrelated processes at high GPU/memory utilization throughout all runs.
    - **Practical takeaway**: vLLM's throughput advantage for DES-Agent's multi-call-per-cycle workload is real but hardware- and architecture-dependent — verify on your own target hardware before assuming a speedup, especially for MoE models on newer/less-common GPU architectures.

## Next Up

1. Expanded common-names registry
   - Add more entries to `artifacts/molecule_names/common_names.json`, especially pharmaceutical actives used as DES component A candidates.

2. Legacy final-report table cleanup
   - The dense pipe-table blocks in `reporting.py` predate the trajectory feature. Restructuring them to match the cleaner trajectory-style narrative is a separate, higher-churn effort.

3. Pharmacophore-based candidate clustering
   - Replace Murcko scaffold (topology-only) with pharmacophore feature clustering (HBD/HBA positions, aromaticity, charge). More meaningful for DES since two different scaffolds with matching H-bond geometry can both form eutectics. Requires new 3D-feature infrastructure.
