# DES Multi-Agent Tutorial

This project combines a deterministic DES screening pipeline with optional layers for uncertainty, local discovery, and LLM-assisted candidate brainstorming. The trained `ml_des_mp` model always makes the final prediction.

## What You Need

- Python environment with the project dependencies installed
- A trained checkpoint from `ml_des_mp/runs/`
- Optional: a local discovery directory with `literature.yaml` and `library.yaml`
- Optional: an Ollama service with Gemma, Nemotron, or Qwen available locally

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

The command uses the bundled `ml_des_mp/config.yaml` and a local trained checkpoint.

## Optional LLM Mode

If you want candidate brainstorming and explanation generation, pass an LLM config file:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --llm-config llm.example.yaml
```

You can edit `llm.example.yaml` to switch `model_name` between `gemma4:12b`, `nemotron-3-nano:latest`, and `qwen3.6` while keeping `provider: ollama`.

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

## Uncertainty Controls

The library CLI [`des_multi_agent.cli`](/home/qshao/DES-Agent/des_multi_agent/cli.py) lets you tune how uncertainty affects filtering and ranking:

```bash
python -m des_multi_agent.cli --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --uncertainty-mode filter --min-trust-score 0.70 --soft-penalty-weight 0.20
```

The default mode is `penalize`. Use `report_only` if you want to inspect the uncertainty columns without changing ranking.
