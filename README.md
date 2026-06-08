# DES-Agent

This repository contains a deterministic DES screening pipeline plus optional layers for uncertainty, local discovery, LLM-assisted candidate brainstorming, DES viscosity prediction, and a separate metal-binding workflow for stability-constant prediction.

## Quick Start

Run the doctor check first to verify the local repo and example folders:

```bash
python -m des_multi_agent.cli doctor
```

If you want extra local setup checks, add `--check` for the paths you care about most:

```bash
python -m des_multi_agent.cli doctor --check checkpoint --check discovery --check artifacts
```

Start with the short tutorial in [`docs/tutorial.md`](/home/qshao/DES-Agent/docs/tutorial.md).
The quickest launch point is [`examples/README.md`](/home/qshao/DES-Agent/examples/README.md).

Offline mock demo, recommended first:

```bash
./scripts/demo-mock.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --mock --component-a "CCO" --n 5
```

Real deterministic demo against the shipped checkpoint:

```bash
./scripts/demo-real.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
```

Optional local discovery:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --discovery-path /path/to/discovery
```

Save a DES run memory file for later reuse:

```bash
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --save-run-memory runs/run_001/run.memory.json
```

Every DES run can also write into a standard flat run directory with `--output-dir runs/run_001`. That folder becomes the canonical home for `report.txt`, `run.json`, `run.csv`, and `run.manifest.json`. If you want run memory in the same folder, point `--save-run-memory` at `runs/run_001/run.memory.json`.

Label the saved run in place with explicit SMILES and `good` / `bad` labels:

```bash
python -m des_multi_agent.cli label-run --run runs/run_001 --label "O=good" --label "CC(=O)O=bad"
```

Reuse the labeled DES memory file or folder to nudge ranking on a later run:

```bash
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --reuse-run runs/run_001/run.memory.json
```

Compare two saved runs from the same workflow with `compare-runs`:

```bash
python -m des_multi_agent.cli compare-runs runs/run_001/run.memory.json runs/run_002/run.memory.json
python -m des_multi_agent.cli compare-runs runs/run_001/run.memory.json runs/run_002/run.memory.json --json
```

Every command prints a compact `summary:` block after its main output. For parseable modes like `task-router` and `compare-runs --json`, the summary is written to `stderr` so `stdout` stays machine-readable.

Optional Ollama LLM run (Gemma, Nemotron, or Qwen via `model_name`). The LLM now reviews candidates one by one, so `--n 20` is safe even when you want a larger candidate set:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --llm-config llm.example.yaml
```

Plain-language Gemma example that routes a request first and then runs the DES workflow:

```bash
./examples/plain_language_gemma4_12b/run.sh
```

Plain-language Gemma example for the metal-binding workflow:

```bash
./examples/plain_language_metal_binding_gemma4_12b/run.sh
```

DES run-memory feedback example:

```bash
./examples/des_run_memory_feedback/run.sh
```

DES viscosity example:

```bash
./examples/des_viscosity/run.sh
```

Metal-binding example:

```bash
./examples/metal_binding/run.sh
```


## Task Router

Use the task router to turn a plain-language request into a JSON job without running the workflow:

```bash
python -m des_multi_agent.cli task-router "find DES partners for lidocaine"
```

Use `task-execute` when you want the router to translate the request and then run the workflow immediately:

```bash
python -m des_multi_agent.cli task-execute "find DES partners for lidocaine"
```

The router loads `llm.example.yaml` by default, supports both `des` and `metal-binding`, and normalizes common compound names before returning either a complete job or clarification questions with `workflow=clarify`, as JSON only. It will ask follow-up questions when a request is ambiguous, including free base versus salt-form questions. For a worked example, see [`examples/task_router/`](/home/qshao/DES-Agent/examples/task_router/).

## Project Layout

- `des_multi_agent/` contains the screening orchestration code
- `ml_des_mp/` contains the trained model and the underlying predictor
- `docs/tutorial.md` is the short user guide for the demo
- `examples/des_viscosity/` is an offline DES viscosity example
- `examples/viscosity_template/` is a template-style DES viscosity example you can adapt
- `examples/metal_binding/` is an offline metal-binding example for stability constants
- `examples/ligand_binding_template/` is a template-style metal-binding example you can adapt
- `examples/lidocaine_gemma4_12b/` is a real lidocaine DES example with Gemma 4-12B
- `examples/plain_language_gemma4_12b/` is a plain-language DES example routed through Gemma 4-12B
- `examples/plain_language_metal_binding_gemma4_12b/` is a plain-language metal-binding example routed through Gemma 4-12B
- `llm.example.yaml` is a ready-to-edit optional LLM config
- `docs/future-improvements.md` tracks the next planned extensions
- `tests/test_benchmarks_examples.py` is the example benchmark suite that compares captured outputs against frozen baselines

## Uncertainty Controls

The library CLI [`des_multi_agent.cli`](/home/qshao/DES-Agent/des_multi_agent/cli.py) lets you tune how uncertainty affects filtering and ranking:

```bash
python -m des_multi_agent.cli --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --uncertainty-mode filter --min-trust-score 0.70 --soft-penalty-weight 0.20
```

The default mode is `penalize`. Use `report_only` if you want to inspect the uncertainty columns without changing ranking.
