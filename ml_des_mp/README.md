# DES melting point pipeline (physics-informed)

This project trains a physics-informed Siamese network to predict DES melting temperature.

## Quick start

```bash
cd des_melt_pipeline
pip install -r requirements.txt
python main_train.py --config config.yaml
```

Outputs (per split method, per run) are written to `runs/`.

## Key scripts

- `main_train.py` : runs 10 splits for `random_row` and `strict_pair`, prints MAE (train/test) per run + averages, saves checkpoints.
- `predict.py` : load a trained checkpoint and run inference on an external CSV (same columns) and optionally export predictions.
- `interpret.py` : plots loss curves, predicted vs actual, residuals, and (for fixed-feature embedders) permutation importance.

## Notes

- The physics module (`models/physics_core.py`) is copied **verbatim** from the provided `ml.py` (no edits).
- GPU is supported (set `device: cuda` in config).

## Multi-agent DES search

The `des_multi_agent` package can drive deterministic DES screening and, optionally, use an LLM for candidate brainstorming and explanation generation. The LLM layer is advisory only: the trained `ml_des_mp` model still makes the final DES classification.

### Deterministic mode

```bash
python -m des_multi_agent.cli   --component-a "CCO"   --n 10   --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
```

### Optional LLM mode

A ready-to-edit template is available at [`llm.example.yaml`](/home/qshao/DES-Agent/llm.example.yaml).

Create a small YAML file, for example `llm.yaml`:

```yaml
llm:
  enabled: true
  provider: ollama
  model_name: llama3.1
  api_base_url: http://localhost:11434
  max_candidates: 8
  max_tokens: 512
  temperature: 0.2
  timeout_seconds: 30.0
```

OpenAI, Gemini, and OpenAI-compatible generic HTTP APIs use the same `llm:` section with a different `provider` value:

```yaml
llm:
  enabled: true
  provider: openai
  model_name: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY
```

```yaml
llm:
  enabled: true
  provider: gemini
  model_name: gemini-2.0-flash
  api_key_env: GEMINI_API_KEY
```

```yaml
llm:
  enabled: true
  provider: custom_http
  model_name: custom-model
  api_base_url: https://api.example.com/v1/chat/completions
  api_key_env: CUSTOM_API_KEY
```

Then run:

```bash
python -m des_multi_agent.cli   --component-a "CCO"   --n 10   --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt   --llm-config llm.yaml
```

The CLI proposes candidate partners, estimates neat-component melting points, predicts melting curves, prints a ranked DES summary, and includes optional LLM brainstorm, explanation, and critique sections when configured.
