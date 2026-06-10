# Model-Specific Examples

Twelve small runnable examples live here:

- [`des_viscosity/`](./des_viscosity) for the offline DES viscosity workflow
- [`viscosity_template/`](./viscosity_template) for a template-style DES viscosity workflow you can adapt
- [`metal_binding/`](./metal_binding) for the offline metal-binding workflow
- [`ligand_binding_template/`](./ligand_binding_template) for a template-style ligand-binding workflow you can adapt
- [`gemma4_12b/`](./gemma4_12b) for Gemma 4-12B
- [`nemotron_3_nano/`](./nemotron_3_nano) for Nemotron 3 Nano
- [`qwen3_6/`](./qwen3_6) for Qwen 3.6
- [`lidocaine_gemma4_12b/`](./lidocaine_gemma4_12b) for a real lidocaine DES run with Gemma 4-12B
- [`plain_language_gemma4_12b/`](./plain_language_gemma4_12b) for a plain-language DES run routed through Gemma 4-12B
- [`plain_language_metal_binding_gemma4_12b/`](./plain_language_metal_binding_gemma4_12b) for a plain-language metal-binding run routed through Gemma 4-12B
- [`task_router/`](./task_router) for translating a plain-language request into a JSON job
- [`des_run_memory_feedback/`](./des_run_memory_feedback) for the full DES save-label-reuse feedback loop

Before adapting a folder, run `python -m des_multi_agent.cli doctor` to verify the core repo and checked-in examples are present. If you also want optional local checks, you can run `python -m des_multi_agent.cli doctor --check checkpoint`, `python -m des_multi_agent.cli doctor --check discovery`, or `python -m des_multi_agent.cli doctor --check artifacts`.

If you want to compare two saved runs from the same workflow, use `python -m des_multi_agent.cli compare-runs <run-a> <run-b>` or add `--json` for a machine-readable summary.

Every command prints a compact `summary:` block after its main output. For parseable modes like `task-router` and `compare-runs --json`, the summary is written to `stderr` so `stdout` stays machine-readable.

Each folder includes and can be used as a template for your own work:

- a runnable `run.sh`
- a short `README.md`
- a captured `input.txt`
- a captured `output.txt`

The same folders also power the pytest-based example benchmark suite in [`tests/test_benchmarks_examples.py`](/home/qshao/DES-Agent/tests/test_benchmarks_examples.py), which compares captured outputs against frozen baselines under `tests/fixtures/example_benchmark_baselines/`.

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
