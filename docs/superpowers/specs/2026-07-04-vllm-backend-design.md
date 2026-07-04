# vLLM LLM Backend — Design Spec

## Goal

Add vLLM as a new selectable LLM backend, coexisting with the existing `ollama` /
`openai` / `gemini` / `custom_http` providers (none are removed or deprecated).
This lets DES-Agent's iterative multi-agent screening loops (many LLM calls per
cycle) run against a locally-hosted open-source model served by vLLM instead of
Ollama, for better throughput via vLLM's continuous batching — while keeping the
model itself open-source and local (Gemma/Nemotron/Qwen family, same as today).

## Context

The LLM layer already fully abstracts request/response shape via
`des_multi_agent/llm/specs.py`'s `RequestProfile` (`path_template`,
`payload_style`, api-key placement) and `des_multi_agent/llm/base.py`'s
`BaseLLMProvider.build_payload`/`request_url`. Every concrete provider
(`OllamaProvider`, `OpenAIProvider`, `GeminiProvider`, `CustomHTTPProvider`,
plus the Ollama-model-specific `NemotronProvider`/`QwenProvider`) is a thin
subclass that only sets a `request_profile` and implements `extract_text`.

vLLM's OpenAI-compatible server (`vllm serve <model>`) speaks the exact same
wire format already implemented as `payload_style="openai"` — flat
`{"model", "messages", "temperature", "max_tokens"}` request,
`choices[0].message.content` response — which `OpenAIProvider` and
`CustomHTTPProvider` already use. No new payload style or response parser is
needed.

`des_multi_agent/llm/factory.py::build_llm_provider` dispatches on the
`provider` string from `LLMConfig` (parsed from a YAML file passed via
`--llm-config`) to instantiate the right provider class.
`des_multi_agent/llm/config.py::LLMConfig.validate` enforces
provider-specific required fields and, for `ollama` only, an allow-list of
supported model names (`_SUPPORTED_OLLAMA_MODEL_PREFIXES`).

`doctor --check llm` (`des_multi_agent/doctor.py::_check_llm_connectivity`)
already does a generic `urlopen(provider.api_base_url)` reachability probe
against whatever provider was built — this works unchanged for any new
provider that follows the existing `BaseLLMProvider` contract.

## Decisions (confirmed with user)

- **Coexist, not replace.** vLLM is one more provider option; Ollama and all
  other backends are untouched.
- **No server-launch tooling.** Docs only — a documented `vllm serve`
  command plus prerequisites. No wrapper script.
- **No model-name allow-list for vLLM.** Unlike Ollama (which can load any of
  several models by name against one long-running daemon), a vLLM server
  process commits to exactly one model at launch time (`vllm serve <model>`).
  The `model_name` in DES-Agent's config is therefore a label matching
  whatever the operator already launched — there is nothing meaningful left
  for config validation to check, so any non-empty `model_name` is accepted.

## Design

### 1. New provider class

`des_multi_agent/llm/vllm_provider.py`:

```python
from __future__ import annotations

from .base import BaseLLMProvider
from .errors import load_json_or_raise, response_error
from .specs import RequestProfile


class VLLMProvider(BaseLLMProvider):
    request_profile = RequestProfile(
        name="vLLM",
        path_template="/chat/completions",
        payload_style="openai",
        api_key_in_header=True,
        api_key_in_query=False,
    )

    def extract_text(self, raw: str) -> str:
        data = load_json_or_raise("vLLM", raw)
        if not isinstance(data, dict):
            raise response_error("vLLM", "must return a JSON object with choices[0].message.content", raw)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise response_error("vLLM", "is missing choices[0].message.content", raw)
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise response_error("vLLM", "is missing choices[0].message.content", raw)
        content = message.get("content")
        if not isinstance(content, str):
            raise response_error("vLLM", "is missing choices[0].message.content", raw)
        return content
```

This is identical in shape to `OpenAIProvider`/`CustomHTTPProvider` (same
existing duplication pattern already present between those two — not
introducing a new inconsistency).

`api_key_in_header=True` is retained for symmetry with `OpenAIProvider`, but
`api_key_env` stays optional in config (see below) — when unset, `base.py`'s
`_request` passes `api_key=None` and no `Authorization` header is sent,
matching how an unauthenticated local server is already handled today for
`custom_http`.

### 2. Factory wiring

`des_multi_agent/llm/factory.py`: import `VLLMProvider`, add a branch:

```python
if provider == "vllm":
    return VLLMProvider(
        model_name=str(llm_cfg.model_name or ""),
        api_base_url=str(llm_cfg.api_base_url or "http://localhost:8000/v1"),
        api_key_env=llm_cfg.api_key_env,
        max_candidates=llm_cfg.max_candidates,
        max_tokens=llm_cfg.max_tokens,
        temperature=llm_cfg.temperature,
        timeout_seconds=llm_cfg.timeout_seconds,
        diversity_mode=llm_cfg.diversity_mode,
        max_families=llm_cfg.max_families,
        family_bias_strength=llm_cfg.family_bias_strength,
        request_fn=request_impl,
    )
```

Default `api_base_url` of `http://localhost:8000/v1` matches vLLM's default
serve port (8000) plus its OpenAI-compatible `/v1` base path — combined with
`path_template="/chat/completions"` this produces
`http://localhost:8000/v1/chat/completions`, mirroring exactly how
`OpenAIProvider` combines `api_base_url="https://api.openai.com/v1"` with the
same `path_template`.

### 3. Config validation

`des_multi_agent/llm/config.py`:

- Add `"vllm"` to `_ALLOWED_PROVIDERS`.
- New validation branch in `LLMConfig.validate`:

```python
if provider == "vllm":
    missing = []
    if not self.model_name:
        missing.append("model_name")
    if not self.api_base_url:
        missing.append("api_base_url")
    if missing:
        raise ValueError("vLLM LLM config requires " + ", ".join(missing))
    return
```

No `api_key_env` requirement (optional, unlike `openai`/`gemini`). No model
name allow-list (unlike `ollama`).

### 4. Doctor connectivity check

No code change. `_check_llm_connectivity` already works against any
provider's `api_base_url` generically.

### 5. Docs

- **New `llm.vllm_example.yaml`** (sibling to `llm.example.yaml`):

```yaml
llm:
  enabled: true
  provider: vllm
  model_name: Qwen/Qwen3-14B-Instruct
  api_base_url: http://localhost:8000/v1
  max_candidates: 20
  max_tokens: 1024
  temperature: 0.2
  timeout_seconds: 120.0
```

- **README.md**: new paragraph immediately after the existing "Optional Ollama
  LLM run" paragraph (around line 115), covering:
  - Prerequisite: `pip install vllm` and a CUDA-capable GPU (vLLM requires GPU
    serving; unlike Ollama it has no built-in CPU fallback path documented
    here).
  - Launch command: `vllm serve <model> --port 8000` (e.g.
    `vllm serve Qwen/Qwen3-14B-Instruct --port 8000`), run as a separate
    long-lived process before starting a DES-Agent run — same relationship
    Ollama already has (external service DES-Agent talks to over HTTP, not
    something DES-Agent launches itself).
  - Usage: `--llm-config llm.vllm_example.yaml`.
  - One line noting `doctor --check llm --llm-config llm.vllm_example.yaml`
    verifies the server is reachable before a real run.

- **`docs/future-improvements.md`**: one new "Recently Completed" numbered
  entry (item 20) documenting the vLLM provider addition, following the exact
  style of existing entries (e.g. item 16's DFT entry).

### 6. Tests (TDD)

Extends existing provider test files — matches how `OpenAIProvider` and
`CustomHTTPProvider` are tested today (a `request_profile` assertion in
`test_llm_profiles.py` plus a factory-dispatch assertion in
`test_llm_factory.py`; neither gets a dedicated `extract_text`-error test
file, so vLLM won't either, for consistency):

- `tests/test_llm_profiles.py`: `test_vllm_provider_exposes_request_profile`
  — asserts `name`, `path_template`, `payload_style`, `api_key_in_header`,
  `api_key_in_query`.
- `tests/test_llm_factory.py`:
  - `test_provider_vllm_returns_vllm_provider` — builds via
    `build_llm_provider` with `provider: vllm`, asserts
    `provider.__class__.__name__ == "VLLMProvider"`.
  - `test_provider_vllm_does_not_require_api_key_env` — builds a vLLM
    provider config with no `api_key_env` and confirms `build_llm_provider`
    does not raise.
- `tests/test_llm_validation.py` (mirrors the existing
  `test_custom_http_config_requires_model_and_base_url` pattern at line 37):
  - `test_vllm_config_requires_model_and_base_url` — `LLMConfig(enabled=True,
    provider="vllm")` raises `ValueError` matching `"model_name|api_base_url"`.
  - `test_vllm_config_accepts_any_model_name` — `LLMConfig(enabled=True,
    provider="vllm", model_name="mistral-7b-instruct", api_base_url="http://localhost:8000/v1")`
    (a name Ollama's allow-list would reject) calls `.validate()` without
    raising, proving there is no allow-list for this provider.

## Out of scope

- No launch/wrapper script for the vLLM server process.
- No CPU-serving path or non-GPU documentation.
- No changes to `ollama`/`openai`/`gemini`/`custom_http` providers.
- No new payload style or response-parsing format (fully reuses `"openai"`).
- No streaming support (matches existing behavior — `stream` is never set
  for the `"openai"` payload style, so vLLM's server defaults to
  non-streaming responses).
