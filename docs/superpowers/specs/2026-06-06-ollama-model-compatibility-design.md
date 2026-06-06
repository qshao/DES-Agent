# Ollama Model Compatibility Design

## Goal

Extend the optional LLM layer so the DES multi-agent system can use any of these Ollama-hosted models through one consistent interface:

- `gemma4:12b`
- `nemotron-3-nano:latest`
- `qwen3.6`

The system must keep using Ollama as the only provider type, while normalizing model-specific raw text into the same strict JSON contract the current Python code already understands.

## Scope

This design covers:

- model selection through `provider: ollama` and `model_name`
- a shared compatibility layer for Ollama requests and responses
- strict JSON normalization for candidate brainstorming, explanations, and critiques
- config and demo updates so the supported models are easy to select
- regression tests for all supported models

This design does not change the deterministic DES prediction path. The trained `ml_des_mp` model remains the final scoring and classification engine.

## Architecture

The LLM layer remains a provider-based adapter, but the Ollama path becomes model-aware.

- The CLI and YAML config continue to expose one provider: `ollama`
- `model_name` selects the model variant
- A shared Ollama compatibility layer builds the request payload, applies any model-specific request flags, and submits the request
- A shared response normalizer converts raw model output into strict JSON before any downstream parser sees it
- The downstream code only consumes validated Python dataclasses, not raw model text

The key principle is that the rest of the system should not care whether the raw text came from Gemma, Nemotron, or Qwen. All three must converge to the same internal schema.

## Components

### `des_multi_agent/llm/factory.py`

- Accepts `provider: ollama`
- Selects the correct Ollama-backed adapter based on `model_name`
- Keeps backwards compatibility with the current Gemma and Nemotron names
- Treats `qwen3.6` as a first-class supported model name

### `des_multi_agent/llm/base.py`

- Owns the common request lifecycle
- Builds the shared Ollama payload
- Supports model-specific request flags where needed, while keeping the transport behavior consistent

### `des_multi_agent/llm/parser.py`

- Serves as the strict JSON interlayer
- Strips markdown fences and surrounding prose when possible
- Extracts the first valid JSON object or array
- Converts valid JSON into the existing candidate, explanation, and critique dataclasses
- Rejects malformed output with a provider-specific error that includes a short payload excerpt

### `des_multi_agent/llm/nemotron_provider.py`

- Contains Nemotron-specific request/response quirks if they differ from the shared Ollama path
- Reuses the shared normalization and parser logic

### `des_multi_agent/llm/gemma_provider.py`

- Keeps Gemma compatibility behavior in one place
- Reuses the shared normalization and parser logic

### `des_multi_agent/llm/qwen_provider.py`

- Adds Qwen support under the same Ollama transport path
- Reuses the shared normalization and parser logic

### `des_multi_agent/cli.py` and `llm.example.yaml`

- Expose a simple way to choose the Ollama model
- Document the supported model names
- Provide an example config for each supported model

## Data Flow

1. The user selects `provider: ollama` and sets `model_name` to one of the supported models.
2. The CLI loads the LLM config and validates that the model name is supported.
3. The provider builds the Ollama request payload.
4. The model returns raw text.
5. The compatibility layer normalizes the raw text into strict JSON.
6. The parser converts that JSON into the existing internal schema.
7. The orchestrator merges the LLM output into `SearchOutcome`.
8. The deterministic `ml_des_mp` predictor remains the final decision path.

If the model output is invalid, the system warns and continues with the deterministic path when possible.

## Error Handling

- Unsupported `model_name` values fail fast during config validation.
- Missing Ollama models or unreachable local Ollama services do not crash the deterministic screening path; they produce a warning and skip the LLM contribution.
- Malformed output raises a provider-specific error with a short excerpt of the offending payload.
- JSON that is valid but missing required fields is treated as a partial parse failure: invalid items are skipped, valid items are kept.
- Extra prose around JSON is tolerated if a valid JSON block can be extracted.
- If no valid JSON block can be found, the provider rejects the output.

## Testing

Add or update tests for:

- provider selection for `gemma4:12b`, `nemotron-3-nano:latest`, and `qwen3.6`
- request payloads for all three Ollama models
- JSON normalization for fenced JSON, prose-wrapped JSON, and malformed JSON
- config validation for supported and unsupported model names
- end-to-end demo behavior with mocked responses for each model
- regression coverage that verifies all three models produce the same internal schema after normalization

## Acceptance Criteria

The feature is complete when:

- the system can run through Ollama with `gemma4:12b`, `nemotron-3-nano:latest`, or `qwen3.6`
- all three models are accepted through the same user-facing provider configuration
- raw model text is normalized into strict JSON before parsing
- the CLI, demo, and example config clearly document the supported models
- the full test suite passes
