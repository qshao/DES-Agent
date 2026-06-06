# DES-Agent

This repository contains a deterministic DES screening pipeline plus optional layers for uncertainty, local discovery, and LLM-assisted candidate brainstorming.

## Quick Start

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

Optional Ollama LLM run (Gemma, Nemotron, or Qwen via `model_name`). The LLM now reviews candidates one by one, so `--n 20` is safe even when you want a larger candidate set:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --llm-config llm.example.yaml
```

## Project Layout

- `des_multi_agent/` contains the screening orchestration code
- `ml_des_mp/` contains the trained model and the underlying predictor
- `docs/tutorial.md` is the short user guide for the demo
- `examples/lidocaine_gemma4_12b/` is a real lidocaine DES example with Gemma 4-12B
- `llm.example.yaml` is a ready-to-edit optional LLM config

## Uncertainty Controls

The library CLI [`des_multi_agent.cli`](/home/qshao/DES-Agent/des_multi_agent/cli.py) lets you tune how uncertainty affects filtering and ranking:

```bash
python -m des_multi_agent.cli --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --uncertainty-mode filter --min-trust-score 0.70 --soft-penalty-weight 0.20
```

The default mode is `penalize`. Use `report_only` if you want to inspect the uncertainty columns without changing ranking.
