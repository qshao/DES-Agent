# DES Multi-Agent Tutorial

This project combines a deterministic DES screening pipeline with optional layers for uncertainty, local discovery, LLM-assisted candidate brainstorming, DES viscosity prediction, and a separate metal-binding workflow for stability-constant prediction. When LLM mode is enabled, candidates are reviewed one by one to keep the JSON payloads small. The trained `ml_des_mp` model always makes the final prediction for DES melting temperature.

## What You Need

- Python environment with the project dependencies installed
- A trained checkpoint from `ml_des_mp/runs/`
- Optional: a local discovery directory with `literature.yaml` and `library.yaml`
- Optional: an Ollama service with Gemma, Nemotron, or Qwen available locally
- Optional: the bundled offline artifact JSON files under `artifacts/` for viscosity and metal-binding runs

## Doctor First

Run `python -m des_multi_agent.cli doctor` before any demo to check the core repo and the checked-in example folders.

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

You can also save a compact run-memory JSON file after a DES run, label it in place, and reuse it later to bias ranking without changing the predictor:

```bash
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --save-run-memory runs/run_001/run.memory.json
python -m des_multi_agent.cli label-run --run runs/run_001 --label "O=good" --label "CC(=O)O=bad"
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --reuse-run runs/run_001/run.memory.json
```

Every DES run also writes machine-readable exports (`run.json`, `run.csv`, `run.manifest.json`) next to the run output. If you save run memory into a folder such as `runs/run_001/run.memory.json`, the export bundle is written in that same folder.

You can compare two saved runs from the same workflow with `compare-runs`:

```bash
python -m des_multi_agent.cli compare-runs runs/run_001/run.memory.json runs/run_002/run.memory.json
```

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

## Optional LLM Mode

If you want candidate brainstorming and explanation generation, pass an LLM config file:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --llm-config llm.example.yaml
```

You can edit `llm.example.yaml` to switch `model_name` between `gemma4:12b`, `nemotron-3-nano:latest`, and `qwen3.6` while keeping `provider: ollama`.


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

If the optional LLM is enabled, the report may also include brainstorm, explanation, critique, and warning sections.
If local discovery is enabled, the report may also show provenance fields such as `source` and `source_id`.

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
