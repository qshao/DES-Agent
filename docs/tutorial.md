# DES Multi-Agent Tutorial

This project combines a deterministic DES screening pipeline with an optional LLM layer for candidate brainstorming and explanation generation. The trained `ml_des_mp` model always makes the final prediction.

## What you need

- Python environment with the project dependencies installed
- A trained checkpoint from `ml_des_mp/runs/`
- Optional: an LLM service such as Ollama, OpenAI, Gemini, or an OpenAI-compatible HTTP API

## Mock demo

Run the fully offline mock demo from the repository root:

```bash
./scripts/demo-mock.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --mock --component-a "CCO" --n 5
```

This does not download a checkpoint or call any external LLM service. It prints a realistic report using canned predictions and canned LLM notes.

## Real deterministic demo

Run the real demo from the repository root:

```bash
./scripts/demo-real.sh
```

Direct command if you prefer:

```bash
DES_CHECKPOINT_PATH=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path "$DES_CHECKPOINT_PATH"
```

The command uses the bundled `ml_des_mp/config.yaml` and a local trained checkpoint. If you have the shipped checkpoint available locally, set `DES_CHECKPOINT_PATH` to `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`.

## Optional LLM mode

If you want candidate brainstorming and explanation generation, pass an LLM config file:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --llm-config llm.example.yaml
```

You can edit `llm.example.yaml` to point to your local Ollama server or API key based provider.

## What the output means

- `smiles_b` is the candidate partner selected for screening
- `is_des` reports whether the predicted curve satisfies both DES criteria
- `min_tm_k` is the minimum predicted melting temperature across the ratio grid
- `rationale` summarizes why the candidate was ranked where it was

If the optional LLM is enabled, the report may also include brainstorm, explanation, critique, and warning sections.

## Common issues

- If the checkpoint path is wrong, the demo fails immediately with a file-not-found error.
- If the optional LLM config is invalid, the CLI reports a clear validation error.
- If you use a provider that is not running locally or is missing credentials, the deterministic screening still runs and the LLM section is skipped with a warning.

