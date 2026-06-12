# Model-Specific Examples

Twenty small runnable examples live here:

- [`des_viscosity/`](./des_viscosity) for the offline DES viscosity workflow
- [`viscosity_template/`](./viscosity_template) for a template-style DES viscosity workflow you can adapt
- [`viscosity_composite_ranking/`](./viscosity_composite_ranking) for viscosity-threshold gating and composite ranking with `--viscosity-threshold` and `--viscosity-weight`
- [`multi_cycle_des/`](./multi_cycle_des) for multi-cycle iterative screening with `--n-cycles`
- [`candidates_file/`](./candidates_file) for screening a curated SMILES list with `--candidates-file`
- [`compare_runs/`](./compare_runs) for comparing two saved DES runs with `compare-runs`
- [`leaderboard_history/`](./leaderboard_history) for ranking compounds and reviewing run history with `leaderboard` and `history`
- [`metal_binding/`](./metal_binding) for the offline metal-binding workflow
- [`ligand_binding_template/`](./ligand_binding_template) for a template-style ligand-binding workflow you can adapt
- [`gemma4_12b/`](./gemma4_12b) for Gemma 4-12B
- [`nemotron_3_nano/`](./nemotron_3_nano) for Nemotron 3 Nano
- [`qwen3_6/`](./qwen3_6) for Qwen 3.6
- [`lidocaine_gemma4_12b/`](./lidocaine_gemma4_12b) for a real lidocaine DES run with Gemma 4-12B
- [`plain_language_gemma4_12b/`](./plain_language_gemma4_12b) for a plain-language DES run routed through Gemma 4-12B
- [`plain_language_metal_binding_gemma4_12b/`](./plain_language_metal_binding_gemma4_12b) for a plain-language metal-binding run routed through Gemma 4-12B
- [`task_router/`](./task_router) for translating a plain-language request into a JSON job
- [`task_execute/`](./task_execute) for routing a plain-language request and running the workflow in one step (requires Ollama)
- [`des_run_memory_feedback/`](./des_run_memory_feedback) for the full DES save-label-reuse feedback loop
- [`betaine_des/`](./betaine_des) for a real-target DES search: betaine, Tm ≤ 340 K, viscosity-minimised ranking, 5-cycle iteration
- [`betaine_des_gemma4_12b/`](./betaine_des_gemma4_12b) for the same betaine search with Ollama Gemma 4-12B: LLM enforces organic H-bonding partners, two-stage brainstorm, and contradiction detection
- [`ni2_co2_selectivity/`](./ni2_co2_selectivity) for a Ni2+/Co2+ selectivity-DES example: Phase 1 screens for selective ligands, Phase 2 finds DES partners for the top hits, outer loop converges when the DES-compatible set stabilises
- [`metal_selectivity_standalone/`](./metal_selectivity_standalone) for the metal-selectivity workflow used independently: screens chelating ligands by Cu2+/Zn2+ selectivity score without running the DES phase
- [`preset_thresholds/`](./preset_thresholds) for named DES threshold presets: runs the same query with `--preset strict` (Tm ≤ 240 K, drop ≥ 15%) and `--preset relaxed` (Tm ≤ 280 K, drop ≥ 5%) side by side
- [`ensemble_prediction/`](./ensemble_prediction) for fold-ensemble predictions with `--ensemble`: auto-discovers all `*_best.pt` checkpoints and adds `ens_std` uncertainty estimates per candidate
- [`uncertainty_controls/`](./uncertainty_controls) for the three `--uncertainty-mode` policies: `report_only` annotates trust scores, `penalize` soft-reranks low-trust candidates, `filter` removes them entirely
- [`output_formats/`](./output_formats) for machine-readable output: shows `--format table`, `json`, `csv`, and `prose` on the same query for scripting and downstream processing
- [`dry_run/`](./dry_run) for setup validation with `--dry-run`: checks paths, config, and checkpoint compatibility without running any predictions — useful for CI and first-time environment checks
- [`ni_co_selectivity_des/`](./ni_co_selectivity_des) for a Ni²⁺/Co²⁺ selectivity-DES run with Gemma4-12B: Phase 1 screens for Ni²⁺-selective HBD/HBA ligands, Phase 2 finds low-viscosity (≤200 cP) DES partners with Tm ≤ 350 K, outer loop feeds DES-compatible hits back to Phase 1
- [`ni_co_selectivity_des_nemotron/`](./ni_co_selectivity_des_nemotron) for the same Ni²⁺/Co²⁺ task with Nemotron-3-Nano: shortlists top-5 HBD/HBA ligands, all 5 DES-compatible, best eutectic Tm 179.5 K (ethylene glycol partner)
- [`ni_co_selectivity_des_qwen36/`](./ni_co_selectivity_des_qwen36) for the same task with Qwen 3.6: brainstorms classical chelators (NTA, IDA, salicylate, glycine, catechol), best eutectic Tm 135.2 K (salicylic acid + p-toluidine), hydrophobic aromatic DES partners

Before adapting a folder, run `python -m des_multi_agent.cli doctor` to verify the core repo and checked-in examples are present. If you also want optional local checks, you can run `python -m des_multi_agent.cli doctor --check checkpoint`, `python -m des_multi_agent.cli doctor --check discovery`, `python -m des_multi_agent.cli doctor --check artifacts`, or `python -m des_multi_agent.cli doctor --check config`. Use `doctor --check llm --llm-config llm.example.yaml` only when you want a live local LLM probe.

If you want to compare two saved runs from the same workflow, use `python -m des_multi_agent.cli compare-runs <run-a> <run-b>` or add `--json` for a machine-readable summary.

Every command prints a compact `summary:` block after its main output. For parseable modes like `task-router` and `compare-runs --json`, the summary is written to `stderr` so `stdout` stays machine-readable.

Each folder includes and can be used as a template for your own work:

- a runnable `run.sh`
- a short `README.md`
- a captured `input.txt`
- a captured `output.txt`

The same folders also power the pytest-based example benchmark suite in [`tests/test_benchmarks_examples.py`](/home/qshao/DES-Agent/tests/test_benchmarks_examples.py), which compares captured outputs against frozen baselines under `tests/fixtures/example_benchmark_baselines/`.

> **Note — melting-point provenance:** the captured `output.txt` files predate the
> layered melting-point resolver (see [`artifacts/melting_points/README.md`](/home/qshao/DES-Agent/artifacts/melting_points/README.md)).
> Live DES runs now print a `Melting-point inputs:` section (and a `tm_src` column in the
> selectivity-DES report) and use experimental/QSPR pure-component melting points, which
> shifts the predicted `min_tm_k` values. The captures were not regenerated because the
> QSPR model (`qspr_model.pt`) is not committed and its training is GPU-stochastic, so
> QSPR-active captures would not reproduce exactly. To reproduce the committed numbers
> deterministically, run with `DES_DISABLE_QSPR=1` (experimental lookup + heuristic only,
> both fully deterministic from committed artifacts).

The LLM-backed examples also include a model-specific `llm.*.yaml` file.

The DES examples call the shared demo entrypoint. In LLM-enabled runs, candidates are reviewed one by one and the brainstorm is two-stage: the LLM first selects chemical families (polyols, amides, imidazolium salts, …), then distributes candidates across those families for better chemical diversity. The LLM also examines each ML prediction for chemical plausibility and reports `agree`, `conflict`, or `uncertain` per candidate.

DES runs can also write into a standard flat run directory with `--output-dir runs/run_001`. That folder becomes the canonical home for `report.txt`, `run.json`, `run.csv`, and `run.manifest.json`. With `--n-cycles N`, each cycle gets its own subdirectory (`cycle_01/`, `cycle_02/`, …) inside the output directory. You can also save, label, and reuse DES run memory with `--save-run-memory`, `label-run`, and `--reuse-run` if you want a later run to bias ranking from an earlier one. If you keep several labeled runs under `runs/`, `--reuse-run runs/` will use the whole labeled history in that history directory.

```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --llm-config <folder>/llm.<name>.yaml
```

**Multi-cycle iterative screening** (`--n-cycles N`): top hits from each cycle seed the next cycle's brainstorm; the loop stops early when the top-K candidate set stabilises across two consecutive cycles. Each cycle prints a progress line to stderr (`[cycle N/M] screened=… des=… top-K changes: …`).

**Viscosity-aware composite ranking**: the DES viscosity examples use `--viscosity-model-path artifacts/designsolvents/viscosity/model.json`. Add `--viscosity-threshold CP` to gate candidates above a viscosity limit (cP) to the bottom of the ranking, and `--viscosity-weight W` (0–1, default 0.3) to control how much viscosity blends into the composite score.

The metal-binding examples use `python -m des_multi_agent.cli --workflow metal-binding ...` and the bundled stability-constant artifact.
The task-router example uses `python -m des_multi_agent.cli task-router "..."` and prints JSON only. It also demonstrates the normalization layer, including follow-up questions for ambiguous requests like a free base versus a salt form.
The task-execute command uses `python -m des_multi_agent.cli task-execute "..."` to route and run the matching workflow in one step.

See [`docs/tutorial.md`](/home/qshao/DES-Agent/docs/tutorial.md) for the full walkthrough and output guide.


Useful inspection commands:

```bash
python -m des_multi_agent.cli supported-metals
python -m des_multi_agent.cli view-run runs/run_001
```
