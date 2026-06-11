# DES Multi-Agent Tutorial

This project combines a deterministic DES screening pipeline with optional layers for uncertainty, local discovery, LLM-assisted candidate brainstorming, viscosity-aware composite ranking, multi-cycle iterative screening with convergence detection, and a separate metal-binding workflow for stability-constant prediction. When LLM mode is enabled, candidates are reviewed one by one and the brainstorm uses a two-stage approach: the LLM first selects chemical families (HBD/HBA role), then distributes candidates across those families. The LLM also examines each prediction for chemical plausibility and flags contradictions. The trained `ml_des_mp` model always makes the final prediction for DES melting temperature.

## What You Need

- Python environment with the project dependencies installed
- A trained checkpoint from `ml_des_mp/runs/`
- Optional: a local discovery directory with `literature.yaml` and `library.yaml`
- Optional: an Ollama service with Gemma, Nemotron, or Qwen available locally
- Optional: the bundled offline artifact JSON files under `artifacts/` for viscosity and metal-binding runs

## Doctor First

Run `python -m des_multi_agent.cli doctor` before any demo to check the core repo and the checked-in example folders. If you want extra local setup coverage, you can run `python -m des_multi_agent.cli doctor --check checkpoint`, `python -m des_multi_agent.cli doctor --check discovery`, or `python -m des_multi_agent.cli doctor --check artifacts` to check the default checkpoint, the bundled discovery fixture, and the local artifact files.

## Mock Demo

Run the fully offline mock demo from the repository root:

```bash
./scripts/demo-mock.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --mock --component-a "CCO" --n 5
```

This does not download a checkpoint or call any external LLM service. It prints a realistic report using canned predictions, uncertainty values, and optional LLM notes.

## Real Deterministic Demo

Run the real demo from the repository root:

```bash
./scripts/demo-real.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
```

If you prefer the wrapper-style override used by `scripts/demo-real.sh`, set the environment variable explicitly:

```bash
DES_CHECKPOINT_PATH=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt ./scripts/demo-real.sh
```

If you want to add a local discovery directory, pass it explicitly:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --discovery-path /path/to/discovery
```

You can also save a compact run-memory JSON file after a DES run, label it in place, and reuse it later to bias ranking without changing the predictor. If you keep multiple labeled runs under one parent `runs/` directory, `--reuse-run` can point at that parent folder to reuse the whole labeled history:

```bash
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --save-run-memory runs/run_001/run.memory.json
python -m des_multi_agent.cli label-run --run runs/run_001 --label "O=good" --label "CC(=O)O=bad"
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --reuse-run runs/run_001/run.memory.json
```

Every DES run can also write into a standard flat run directory with `--output-dir runs/run_001`. That folder becomes the canonical home for `report.txt`, `run.json`, `run.csv`, and `run.manifest.json`. If you want run memory in the same folder, point `--save-run-memory` at `runs/run_001/run.memory.json`. If you later want to reuse all labeled runs in a history directory, point `--reuse-run` at the parent `runs/` folder.

You can compare two saved runs from the same workflow with `compare-runs`:

```bash
python -m des_multi_agent.cli compare-runs runs/run_001/run.memory.json runs/run_002/run.memory.json
python -m des_multi_agent.cli compare-runs runs/run_001/run.memory.json runs/run_002/run.memory.json --json
```

Every command prints a compact `summary:` block after its main output. For parseable modes like `task-router` and `compare-runs --json`, the summary is written to `stderr` so `stdout` stays machine-readable.

The command uses the bundled `ml_des_mp/config.yaml` and a local trained checkpoint.

## Example Benchmark

The example folders also double as a pytest-based example benchmark suite. The benchmark lives in [`tests/test_benchmarks_examples.py`](/home/qshao/DES-Agent/tests/test_benchmarks_examples.py) and compares the checked-in example outputs against frozen baselines under `tests/fixtures/example_benchmark_baselines/`.

## Real Lidocaine Example

For a real model-backed example, see [examples/lidocaine_gemma4_12b/](../examples/lidocaine_gemma4_12b/). It records a lidocaine free-base run with Gemma 4-12B and the shipped `ml_des_mp` checkpoint.

## Plain-Language Gemma Example

If you want to see the natural-language router in action, see [examples/plain_language_gemma4_12b/](../examples/plain_language_gemma4_12b/). It takes a plain-language request, turns it into a JSON job, and then runs the DES workflow with Gemma 4-12B.

## Plain-Language Gemma Metal-Binding Example

If you want to see the same idea applied to the metal-binding workflow, see [examples/plain_language_metal_binding_gemma4_12b/](../examples/plain_language_metal_binding_gemma4_12b/). It takes a plain-language request, turns it into a JSON job, and then runs the metal-binding workflow with Gemma 4-12B.

## DES Run Memory Feedback Example

If you want to see the save-label-reuse loop in a single folder, see [examples/des_run_memory_feedback/](../examples/des_run_memory_feedback/). It shows a DES run that saves `run.memory.json`, labels it in place with `label-run`, and then reuses the labeled memory on the next run.

## DES Viscosity Example

Run the offline DES viscosity example from the repository root:

```bash
./examples/des_viscosity/run.sh
```

The captured output includes a `Viscosity predictions:` section after the DES screening table. For a user-editable starting point, see [`examples/viscosity_template/`](../viscosity_template).

## Metal-Binding Example

Run the metal-binding example from the repository root:

```bash
./examples/metal_binding/run.sh
```

This workflow is separate from DES screening and prints `log K` predictions for a metal ion and ligand pair. For a user-editable starting point, see [`examples/ligand_binding_template/`](../ligand_binding_template).

## Metal Ion Selectivity Screening

Use `--workflow metal-selectivity` to screen ligands for **selectivity** toward a target metal over a competitor. The composite score balances binding affinity (`log K`) and selectivity (`Δlog K = log K_target − log K_competitor`):

```bash
python -m des_multi_agent.cli \
  --workflow metal-selectivity \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion Zn2+ \
  --n 20 --n-cycles 3 \
  --affinity-weight 0.5 --selectivity-weight 0.5
```

`--affinity-weight` and `--selectivity-weight` (both default 0.5) control the relative importance of absolute binding strength versus metal discrimination in the ranking score. With `--n-cycles N` and an LLM config, the loop proposes increasingly selective ligands each cycle using HSAB theory guidance.

The predictor gives the most meaningful selectivity signal when both metals are in the supported identity table:

| Category | Ions |
|----------|------|
| **Alkali metals** | Li+, Na+, K+ |
| **Alkaline earth** | Mg2+, Ca2+, Ba2+ |
| **Post-transition** | Al3+, Pb2+ |
| **First-row transition** | Mn2+, Mn3+, Fe2+, Fe3+, Co2+, Co3+, Ni2+, Cu+, Cu2+, Zn2+ |
| **Second-row transition** | Pd2+, Ag+, Cd2+ |
| **Third-row transition** | Pt2+, Hg+, Hg2+ |
| **Lanthanides** | La3+, Gd3+ |

Metal ions not in this table still work but fall back to zeroed identity features, so Δlog K between two unlisted metals will be near zero.

## Optional LLM Mode

If you want candidate brainstorming and explanation generation, pass an LLM config file:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --llm-config llm.example.yaml
```

You can edit `llm.example.yaml` to switch `model_name` between `gemma4:12b`, `nemotron-3-nano:latest`, and `qwen3.6` while keeping `provider: ollama`.

When an LLM is configured, brainstorming is two-stage: the LLM first selects 4–6 chemical families (e.g., polyols, amides, imidazolium salts) with a rationale and HBD/HBA role for each, then distributes candidate SMILES across those families. This improves chemical diversity compared to single-shot brainstorming. After ML predictions, the LLM also examines each result for chemical plausibility and reports `agree`, `conflict`, or `uncertain` per candidate — surfaced as a "LLM contradiction analysis" section in the report.

## Multi-Cycle Iterative Screening

Use `--n-cycles N` to run up to N screening iterations. Top hits from each cycle are passed forward as context to the next cycle's brainstorm, so the LLM focuses on productive chemical families. The loop stops early when the top-K candidate set stabilises (convergence detection):

```bash
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --n-cycles 3 --llm-config llm.example.yaml
```

Each cycle prints a progress line to stderr:

```
[cycle 1/3] screened=20 des=5 top-K changes: +5 new, 0 dropped
[cycle 2/3] screened=20 des=7 top-K changes: +3 new, -1 dropped
[cycle 3/3] screened=20 des=7 top-K changes: 0 new, 0 dropped — CONVERGED
```

The final report is produced from the last cycle. With `--output-dir runs/run_001`, each cycle writes into its own subdirectory (`cycle_01/`, `cycle_02/`, …).

## Viscosity-Aware Composite Ranking

When a viscosity model is available (via `--viscosity-model-path`), use `--viscosity-threshold` and `--viscosity-weight` to blend viscosity into the composite ranking score:

```bash
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json \
  --viscosity-threshold 500 --viscosity-weight 0.4
```

`--viscosity-threshold CP` gates candidates: DES-formers above the threshold (cP) sort after those that pass it, regardless of Tm. `--viscosity-weight W` (0–1, default 0.3) controls how much viscosity contributes to the composite score alongside the Tm-drop score.


## Task Router

Use the task router when you want plain language translated into a JSON job without running a workflow:

```bash
python -m des_multi_agent.cli task-router "find DES partners for lidocaine"
```

Use `task-execute` when you want the router to translate the request and then run the workflow immediately:

```bash
python -m des_multi_agent.cli task-execute "find DES partners for lidocaine"
```

The router loads `llm.example.yaml` by default, supports both `des` and `metal-binding`, and normalizes common names before returning either a complete job or clarification questions with `workflow=clarify`, as JSON only. If a request is ambiguous, it asks for clarification instead of guessing. For a worked example, see [`examples/task_router/`](/home/qshao/DES-Agent/examples/task_router/).

## What the Output Means

- `smiles_b` is the candidate partner selected for screening
- `is_des` reports whether the predicted curve satisfies both DES criteria
- `min_tm_k` is the minimum predicted melting temperature across the ratio grid
- `trust_score` shows the uncertainty trust value in the range `0.0` to `1.0`
- `rationale` summarizes why the candidate was ranked where it was

If the optional LLM is enabled, the report may also include brainstorm, explanation, critique, contradiction analysis, and warning sections. The contradiction analysis section shows one line per candidate with the LLM's verdict (`agree`, `conflict`, or `uncertain`) and an explanation.
If local discovery is enabled, the report may also show provenance fields such as `source` and `source_id`.
If multi-cycle mode is active (`--n-cycles > 1`), cycle-level progress is printed to stderr during the run and the final report reflects the last cycle only.

## Common Issues

- If the checkpoint path is wrong, the demo fails immediately with a file-not-found error.
- If the optional LLM config is invalid, the CLI reports a clear validation error.
- If you use a provider that is not running locally or is missing credentials, the deterministic screening still runs and the LLM section is skipped with a warning.
- If the discovery directory is missing or malformed, the demo falls back to heuristic candidate generation and reports a warning.
- If a request mentions a free base versus a salt form, the router may ask a clarification question before it executes anything.

## Uncertainty Controls

The library CLI [`des_multi_agent.cli`](/home/qshao/DES-Agent/des_multi_agent/cli.py) lets you tune how uncertainty affects filtering and ranking:

```bash
python -m des_multi_agent.cli --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --uncertainty-mode filter --min-trust-score 0.70 --soft-penalty-weight 0.20
```

The default mode is `penalize`. Use `report_only` if you want to inspect the uncertainty columns without changing ranking.
