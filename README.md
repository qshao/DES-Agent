# DES-Agent

This repository contains a deterministic DES screening pipeline plus an optional LLM layer for candidate brainstorming and explanation generation.

## Demo

Start with the short tutorial in [`docs/tutorial.md`](/home/qshao/DES-Agent/docs/tutorial.md).
The quickest launch point is [`examples/README.md`](/home/qshao/DES-Agent/examples/README.md).

Quick offline mock run:

```bash
./scripts/demo-mock.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --mock --component-a "CCO" --n 5
```

Quick deterministic run against a local checkpoint:

```bash
./scripts/demo-real.sh
```

Direct command if you prefer:

```bash
DES_CHECKPOINT_PATH=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path "$DES_CHECKPOINT_PATH"
```

Optional LLM run:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --llm-config llm.example.yaml
```

## Project layout

- `des_multi_agent/` contains the screening orchestration code
- `ml_des_mp/` contains the trained model and the underlying predictor
- `docs/tutorial.md` is the short user guide for the demo
- `llm.example.yaml` is a ready-to-edit optional LLM config

