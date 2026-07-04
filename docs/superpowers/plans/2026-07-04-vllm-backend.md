# vLLM LLM Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `vllm` as a new selectable LLM provider in DES-Agent's LLM layer, so an operator can point `--llm-config` at a locally-hosted vLLM server instead of Ollama, without touching any existing provider.

**Architecture:** vLLM's OpenAI-compatible server already speaks the exact wire format DES-Agent's `payload_style="openai"` already implements. A new `VLLMProvider(BaseLLMProvider)` class (near-identical to the existing `OpenAIProvider`/`CustomHTTPProvider`) is wired into `factory.py`'s provider dispatch and `config.py`'s validation, with no allow-list on `model_name` (a vLLM server commits to one model at launch, so there's nothing left to validate) and no required `api_key_env` (vLLM servers are typically unauthenticated locally). The existing generic `doctor --check llm` connectivity check needs no changes since it works against any provider's `api_base_url`.

**Tech Stack:** Python (stdlib `urllib`-based HTTP transport already in place, no new dependencies), pytest.

## Global Constraints

- Full spec: `docs/superpowers/specs/2026-07-04-vllm-backend-design.md` — read it if anything below is ambiguous.
- Coexist, not replace: `ollama`/`openai`/`gemini`/`custom_http` providers are untouched.
- No server-launch script — docs only (a documented `vllm serve` command).
- No model-name allow-list for `vllm` (unlike Ollama's `_SUPPORTED_OLLAMA_MODEL_PREFIXES`) — any non-empty `model_name` is valid.
- No `api_key_env` requirement for `vllm` (unlike `openai`/`gemini`, which both require it).
- Default `api_base_url` for `vllm` is `http://localhost:8000/v1` (vLLM's default port + its OpenAI-compatible `/v1` base path).
- Reuse `payload_style="openai"` exactly — no new payload style, no streaming support.

---

### Task 1: `VLLMProvider` class + request-profile test

**Files:**
- Create: `des_multi_agent/llm/vllm_provider.py`
- Test: `tests/test_llm_profiles.py`

**Interfaces:**
- Consumes: `des_multi_agent.llm.base.BaseLLMProvider` (existing), `des_multi_agent.llm.specs.RequestProfile` (existing dataclass with fields `name`, `path_template`, `payload_style`, `api_key_in_header`, `api_key_in_query`), `des_multi_agent.llm.errors.load_json_or_raise`/`response_error` (existing).
- Produces: `des_multi_agent.llm.vllm_provider.VLLMProvider` — a `BaseLLMProvider` subclass with `request_profile` class attribute and `extract_text(self, raw: str) -> str` method. Task 3 (factory wiring) imports this class.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_profiles.py` (after the existing `test_qwen_provider_payload_disables_thinking` at the end of the file):

```python
from des_multi_agent.llm.vllm_provider import VLLMProvider


def test_vllm_provider_exposes_request_profile():
    assert VLLMProvider.request_profile.name == "vLLM"
    assert VLLMProvider.request_profile.path_template == "/chat/completions"
    assert VLLMProvider.request_profile.payload_style == "openai"
    assert VLLMProvider.request_profile.api_key_in_header is True
    assert VLLMProvider.request_profile.api_key_in_query is False
```

Add the import at the top of the file alongside the other provider imports (do not leave it inline mid-file — move it up next to `from des_multi_agent.llm.qwen_provider import QwenProvider`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_profiles.py::test_vllm_provider_exposes_request_profile -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'des_multi_agent.llm.vllm_provider'`

- [ ] **Step 3: Write minimal implementation**

Create `des_multi_agent/llm/vllm_provider.py`:

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_profiles.py -v`
Expected: All tests PASS, including `test_vllm_provider_exposes_request_profile`.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/vllm_provider.py tests/test_llm_profiles.py
git commit -m "feat: add VLLMProvider class for OpenAI-compatible vLLM servers"
```

---

### Task 2: Config validation (`vllm` provider support)

**Files:**
- Modify: `des_multi_agent/llm/config.py:7` (`_ALLOWED_PROVIDERS`), `:109-118` (validate branches)
- Test: `tests/test_llm_validation.py`

**Interfaces:**
- Consumes: `des_multi_agent.llm.config.LLMConfig` (existing dataclass, fields `enabled`, `provider`, `model_name`, `api_base_url`, `api_key_env`, etc.), its existing `validate(self) -> None` method (modified in place, raises `ValueError` on invalid config). Does not depend on Task 1.
- Produces: `LLMConfig.validate()` no longer raises `"Unsupported llm.provider"` for `provider == "vllm"`; instead it requires `model_name` and `api_base_url` to be set (raising `ValueError` with a message containing `"vLLM LLM config requires"` if either is missing), and imposes no restriction on the value of `model_name`. Task 3 (factory wiring) relies on this: `build_llm_provider` calls `llm_cfg.validate()` before dispatching, so the `vllm` branch there is only reachable once this task lands.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_validation.py`:

```python
def test_vllm_config_requires_model_and_base_url():
    cfg = LLMConfig(enabled=True, provider="vllm")
    with pytest.raises(ValueError, match="model_name|api_base_url"):
        cfg.validate()


def test_vllm_config_accepts_any_model_name():
    cfg = LLMConfig(
        enabled=True,
        provider="vllm",
        model_name="mistral-7b-instruct",
        api_base_url="http://localhost:8000/v1",
    )
    cfg.validate()  # must not raise — no allow-list for vllm model names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_validation.py::test_vllm_config_requires_model_and_base_url tests/test_llm_validation.py::test_vllm_config_accepts_any_model_name -v`
Expected: Both FAIL with `ValueError: Unsupported llm.provider: vllm`

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/llm/config.py`, change line 7:

```python
_ALLOWED_PROVIDERS = {"disabled", "none", "off", "ollama", "openai", "gemini", "custom_http"}
```

to:

```python
_ALLOWED_PROVIDERS = {"disabled", "none", "off", "ollama", "openai", "gemini", "custom_http", "vllm"}
```

Then, in `validate(self)`, add a new branch immediately after the existing `custom_http` branch (after line 117's `return` and before line 118's `raise ValueError(f"Unsupported llm.provider: {self.provider}")`):

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_validation.py -v`
Expected: All PASS, including both new `vllm` tests.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/config.py tests/test_llm_validation.py
git commit -m "feat: accept vllm as a valid LLM provider in config validation"
```

---

### Task 3: Factory wiring + dispatch tests

**Files:**
- Modify: `des_multi_agent/llm/factory.py:1-13` (imports), `:80-94` (dispatch branches)
- Test: `tests/test_llm_factory.py`

**Interfaces:**
- Consumes: `VLLMProvider` from Task 1 (`des_multi_agent.llm.vllm_provider.VLLMProvider`), the `vllm`-accepting `LLMConfig.validate()` from Task 2, `des_multi_agent.llm.factory.build_llm_provider(cfg, request_fn=None) -> LLMProvider | None` (existing function, modified in place).
- Produces: `build_llm_provider` now returns a `VLLMProvider` instance when `cfg["provider"] == "vllm"`. Nothing later in this plan depends on this task.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_factory.py` (after `test_provider_custom_http_returns_custom_http_provider`):

```python
def test_provider_vllm_returns_vllm_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "vllm",
            "model_name": "Qwen/Qwen3-14B-Instruct",
            "api_base_url": "http://localhost:8000/v1",
        },
        request_fn=lambda *args, **kwargs: '{"choices":[{"message":{"content":"[]"}}]}',
    )
    assert provider.__class__.__name__ == "VLLMProvider"


def test_provider_vllm_does_not_require_api_key_env():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "vllm",
            "model_name": "Qwen/Qwen3-14B-Instruct",
            "api_base_url": "http://localhost:8000/v1",
        },
        request_fn=lambda *args, **kwargs: '{"choices":[{"message":{"content":"[]"}}]}',
    )
    assert provider.api_key_env is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_factory.py::test_provider_vllm_returns_vllm_provider -v`
Expected: FAIL with `AttributeError: module 'des_multi_agent.llm.factory' has no attribute 'VLLMProvider'`-style error, or more likely a plain `NameError`/import error since the factory branch doesn't exist yet — the test calls `build_llm_provider` with `provider: "vllm"`, which (with Task 2 already landed) now passes `validate()` but falls through every `if provider == ...` branch to `raise ValueError(f"Unknown llm.provider: {llm_cfg.provider}")`. Confirm the failure is this `ValueError: Unknown llm.provider: vllm`, not a config-validation error.

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/llm/factory.py`, add the import alongside the other provider imports (after line 13, `from .qwen_provider import QwenProvider`):

```python
from .vllm_provider import VLLMProvider
```

Then add a new dispatch branch in `build_llm_provider`, immediately after the existing `custom_http` branch (after line 93's closing `)` and before line 94's `raise ValueError(f"Unknown llm.provider: {llm_cfg.provider}")`):

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_factory.py -v`
Expected: All tests PASS, including both new `vllm` tests.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/factory.py tests/test_llm_factory.py
git commit -m "feat: dispatch provider=vllm to VLLMProvider in build_llm_provider"
```

---

### Task 4: Example config + docs

**Files:**
- Create: `llm.vllm_example.yaml`
- Modify: `README.md:115` (insert new paragraph + code block after the existing Ollama paragraph, before line 121's Multi-cycle paragraph)
- Modify: `docs/future-improvements.md` (insert new numbered item before `## Next Up`, currently at line 87)

**Interfaces:**
- Consumes: Nothing from earlier tasks directly — this task is documentation/config only, but it references the now-working `provider: vllm` config path from Tasks 2-3, so it must land after them (running the doctor check below as verification requires Task 3's validation and Task 2's dispatch to both be in place).
- Produces: Nothing consumed by later tasks — this is the final task in the plan.

- [ ] **Step 1: Create `llm.vllm_example.yaml`** at the repo root:

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

- [ ] **Step 2: Verify the example config loads and validates**

Run:

```bash
python3 -c "
from des_multi_agent.cli import load_llm_config
from pathlib import Path
cfg = load_llm_config(Path('llm.vllm_example.yaml'))
cfg.validate()
print(cfg.provider, cfg.model_name, cfg.api_base_url)
"
```

Expected output: `vllm Qwen/Qwen3-14B-Instruct http://localhost:8000/v1` with no exception raised.

- [ ] **Step 3: Add the README paragraph**

In `README.md`, insert the following text immediately after line 119 (the closing ` ``` ` of the existing Ollama code block) and before line 121 (`Multi-cycle iterative screening...`). It is one plain paragraph, one bash code block, and one closing sentence — insert exactly this text (not wrapped in any extra outer fence):

Optional vLLM run — an alternative local backend to Ollama for the same open-source models (Gemma, Nemotron, Qwen), using vLLM's continuous batching for faster throughput on multi-candidate cycles. Requires `pip install vllm` and a CUDA-capable GPU (vLLM has no documented CPU-serving path here). Start the server as its own long-lived process before running DES-Agent, the same way Ollama already runs as an external service:

```bash
vllm serve Qwen/Qwen3-14B-Instruct --port 8000
python -m examples.demo_des_search --component-a "ethanol" --n 20 --llm-config llm.vllm_example.yaml
```

`doctor --check llm --llm-config llm.vllm_example.yaml` verifies the vLLM server is reachable before a real run.

- [ ] **Step 4: Add the future-improvements.md entry**

In `docs/future-improvements.md`, insert the following new item immediately before the `## Next Up` heading (currently line 87), right after item 19's last line (`    - **Private cross-module import**: ...` at line 85, and the blank line 86):

```markdown
20. vLLM LLM backend
    - `--llm-config` now also accepts `provider: vllm`, a new `VLLMProvider` alongside the existing `ollama`/`openai`/`gemini`/`custom_http` backends — none of which are changed or deprecated. Reuses the existing `payload_style="openai"` request/response format since vLLM's OpenAI-compatible server (`vllm serve <model>`) speaks the same wire format as `OpenAIProvider`/`CustomHTTPProvider`.
    - No `model_name` allow-list (unlike `ollama`'s `_SUPPORTED_OLLAMA_MODEL_PREFIXES`) since a vLLM server process commits to exactly one model at launch; the config's `model_name` is just a label for whichever model the operator already started. `api_key_env` is optional, matching how local unauthenticated servers are already handled for `custom_http`.
    - See `llm.vllm_example.yaml` for a ready-to-edit config, and the README's "Optional vLLM run" section for the `vllm serve` launch command and GPU prerequisites.
```

- [ ] **Step 5: Commit**

```bash
git add llm.vllm_example.yaml README.md docs/future-improvements.md
git commit -m "docs: document vLLM as an optional LLM backend"
```

---

## Final Verification

After all four tasks are complete, run the full suite to confirm no regressions:

```bash
pytest tests/ -q --tb=short
```

Expected: all tests pass, including the 6 new tests added across Tasks 1-3 (`test_vllm_provider_exposes_request_profile`, `test_provider_vllm_returns_vllm_provider`, `test_provider_vllm_does_not_require_api_key_env`, `test_vllm_config_requires_model_and_base_url`, `test_vllm_config_accepts_any_model_name`, plus the Step 2 manual config-load check in Task 4 which isn't a pytest test but should be re-run here as a sanity check).
