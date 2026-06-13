# Model-Specific Examples

A set of small runnable examples live here.

Start here:

- Plain-language routing: [`plain_language_gemma4_12b/`](./plain_language_gemma4_12b) or [`plain_language_metal_binding_gemma4_12b/`](./plain_language_metal_binding_gemma4_12b)
- Offline DES: [`des_viscosity/`](./des_viscosity) or [`preset_thresholds/`](./preset_thresholds)
- Diversity-aware DES: [`ni_co_selectivity_des/`](./ni_co_selectivity_des), [`ni_co_selectivity_des_qwen36/`](./ni_co_selectivity_des_qwen36), or [`ni_co_selectivity_des_nemotron/`](./ni_co_selectivity_des_nemotron)
- Chemistry-advisor and run memory: [`des_run_memory_feedback/`](./des_run_memory_feedback)
- Metal-binding and selectivity: [`metal_binding/`](./metal_binding), [`metal_selectivity_standalone/`](./metal_selectivity_standalone), or [`ni2_co2_selectivity/`](./ni2_co2_selectivity)

Feature map:

| Example | Plain-language | Proposal diversity | Chemistry advisor | Run memory | Offline-only |
|---------|----------------|--------------------|-------------------|------------|--------------|
| `plain_language_gemma4_12b/` | yes | yes | yes | no | no |
| `plain_language_metal_binding_gemma4_12b/` | yes | no | no | no | yes |
| `des_viscosity/` | no | no | no | no | yes |
| `des_run_memory_feedback/` | no | yes | no | yes | yes |
| `ni_co_selectivity_des/` | no | yes | yes | no | no |
| `ni_co_selectivity_des_qwen36/` | no | yes | yes | no | no |
| `ni_co_selectivity_des_nemotron/` | no | yes | yes | no | no |
| `metal_binding/` | no | no | yes | no | yes |
| `metal_selectivity_standalone/` | no | no | yes | no | yes |

- [`des_viscosity/`](./des_viscosity) for offline DES viscosity
- [`viscosity_template/`](./viscosity_template) for a template-style DES viscosity workflow you can adapt
- [`viscosity_composite_ranking/`](./viscosity_composite_ranking) for viscosity-threshold gating and composite ranking with `--viscosity-threshold` and `--viscosity-weight`
- [`multi_cycle_des/`](./multi_cycle_des) for multi-cycle iterative screening with `--n-cycles`
- [`candidates_file/`](./candidates_file) for screening a curated SMILES list with `--candidates-file`
- [`compare_runs/`](./compare_runs) for comparing two saved DES runs with `compare-runs`
- [`leaderboard_history/`](./leaderboard_history) for ranking compounds and reviewing run history with `leaderboard` and `history`
- [`metal_binding/`](./metal_binding) for the offline metal-binding workflow
- [`ligand_binding_template/`](./ligand_binding_template) for a template-style ligand-binding workflow you can adapt
- [`gemma4_12b/`](./gemma4_12b) for Gemma 4-12B DES
- [`nemotron_3_nano/`](./nemotron_3_nano) for Nemotron 3 Nano DES
- [`qwen3_6/`](./qwen3_6) for Qwen 3.6 DES
- [`lidocaine_gemma4_12b/`](./lidocaine_gemma4_12b) for lidocaine DES
- [`plain_language_gemma4_12b/`](./plain_language_gemma4_12b) for plain-language DES
- [`plain_language_metal_binding_gemma4_12b/`](./plain_language_metal_binding_gemma4_12b) for plain-language metal binding
- [`task_router/`](./task_router) for translating a plain-language request into a JSON job
- [`task_execute/`](./task_execute) for routing a plain-language request and running the workflow in one step (requires Ollama)
- [`des_run_memory_feedback/`](./des_run_memory_feedback) for DES run memory
- [`betaine_des/`](./betaine_des) for a betaine DES search
- [`betaine_des_gemma4_12b/`](./betaine_des_gemma4_12b) for the betaine search with Ollama Gemma 4-12B
- [`ni2_co2_selectivity/`](./ni2_co2_selectivity) for the Ni2+/Co2+ selectivity-DES example
- [`metal_selectivity_standalone/`](./metal_selectivity_standalone) for standalone metal selectivity
- [`preset_thresholds/`](./preset_thresholds) for named DES presets
- [`ensemble_prediction/`](./ensemble_prediction) for fold ensembles
- [`uncertainty_controls/`](./uncertainty_controls) for uncertainty controls
- [`output_formats/`](./output_formats) for machine-readable output
- [`dry_run/`](./dry_run) for setup validation
- [`ni_co_selectivity_des/`](./ni_co_selectivity_des) for Ni²⁺/Co²⁺ selectivity-DES
- [`ni_co_selectivity_des_nemotron/`](./ni_co_selectivity_des_nemotron) for Ni²⁺/Co²⁺ selectivity with Nemotron-3-Nano
- [`ni_co_selectivity_des_qwen36/`](./ni_co_selectivity_des_qwen36) for Ni²⁺/Co²⁺ selectivity with Qwen 3.6

Before adapting a folder, run `python -m des_multi_agent.cli doctor` to verify the core repo and checked-in examples are present. If you also want optional local checks, you can run `python -m des_multi_agent.cli doctor --check checkpoint`, `python -m des_multi_agent.cli doctor --check discovery`, `python -m des_multi_agent.cli doctor --check artifacts`, or `python -m des_multi_agent.cli doctor --check config`. Use `doctor --check llm --llm-config llm.example.yaml` only when you want a live local LLM probe.

If you want to compare two saved runs from the same workflow, use `python -m des_multi_agent.cli compare-runs <run-a> <run-b>` or add `--json` for a machine-readable summary.

Every command prints a compact `summary:` block after its main output. For parseable modes like `task-router` and `compare-runs --json`, the summary is written to `stderr` so `stdout` stays machine-readable.

Each folder includes and can be used as a template for your own work:

- a runnable `run.sh`
- a short `README.md`
- a captured `input.txt`
- a captured `output.txt`

The same folders also power the pytest-based example benchmark suite in [`tests/test_benchmarks_examples.py`](/home/qshao/DES-Agent/tests/test_benchmarks_examples.py), which compares captured outputs against frozen baselines under `tests/fixtures/example_benchmark_baselines/`.

> **Note — melting-point provenance:** DES runs now print a `Melting-point inputs:`
> section (and a `tm_src` column in the selectivity-DES report) showing where each
> pure-component melting point came from — see [`artifacts/melting_points/README.md`](/home/qshao/DES-Agent/artifacts/melting_points/README.md).
> The **deterministic** DES examples set `export DES_DISABLE_QSPR=1` in their `run.sh`,
> so their captured outputs use only the committed experimental lookup + heuristic and
> reproduce byte-for-byte. The QSPR layer is not exercised in these captures because
> `qspr_model.pt` is not committed and its training is GPU-stochastic; it is documented
> and demonstrated separately. The **LLM-backed** captures (e.g. `gemma4_12b`,
> `ni_co_selectivity_des*`) predate this feature and were not regenerated.

The LLM-backed examples also include a model-specific `llm.*.yaml` file. The shared demo entrypoint covers the LLM-enabled DES runs, including the two-stage brainstorm, proposal-diversity controls, and chemistry-advisor notes. DES runs can also write into a standard flat run directory with `--output-dir runs/run_001`, and run memory can be saved, labeled, and reused to bias later ranking.

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
