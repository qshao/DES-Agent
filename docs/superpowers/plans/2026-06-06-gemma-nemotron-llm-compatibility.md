# Gemma and Nemotron LLM Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand DES candidate search to support up to 20 candidates and add explicit Ollama support for `nemotron-3-nano` alongside `gemma4:12b`, with a normalization layer that converts model output into the JSON shape the current Python code already understands.

**Architecture:** Keep the deterministic DES pipeline unchanged. Extend the candidate-discovery and LLM layers so the system can request up to 20 candidates and handle two explicit Ollama models. Add a small response-normalization layer between Ollama and the existing JSON parser so model-specific formatting quirks, especially fenced JSON or thinking traces, are stripped before parsing.

**Tech Stack:** Python 3.13, `pytest`, existing `des_multi_agent` package, Ollama chat API.

---

### Task 1: Expand heuristic candidate generation capacity to 20

**Files:**
- Modify: `des_multi_agent/candidate_generation.py`
- Modify: `tests/test_candidate_generation.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.candidate_generation import generate_candidates


def test_generate_candidates_caps_at_twenty_for_large_requests():
    proposals = generate_candidates("CCO", 20)
    assert len(proposals) == 20
    assert len({proposal.smiles for proposal in proposals}) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_generation.py::test_generate_candidates_caps_at_twenty_for_large_requests -v`
Expected: FAIL because the current family library only contains 10 candidates.

- [ ] **Step 3: Write minimal implementation**

```python
_FAMILY_LIBRARY: Sequence[tuple[str, str, str]] = (
    # existing entries ...
    ("short diol", "hydrogen-bond donor", "OCCCO"),
    ("triol", "hydrogen-bond donor", "OCC(O)COO"),
    ("ether alcohol", "hydrogen-bond donor", "COCCO"),
    ("glycol ether", "hydrogen-bond donor", "COCCOC"),
    ("lactam", "hydrogen-bond acceptor", "O=C1CCCN1"),
    ("cyclic urea", "hydrogen-bond donor", "O=C1NC(=O)NC1"),
    ("sulfoxide", "hydrogen-bond acceptor", "CS(=O)C"),
    ("sulfone", "hydrogen-bond acceptor", "CS(=O)(=O)C"),
    ("hydroxypyridine", "hydrogen-bond donor", "OC1=CC=CC=N1"),
    ("dimethylformamide-like", "polar aprotic partner", "CN(C)C=O"),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_generation.py::test_generate_candidates_caps_at_twenty_for_large_requests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/candidate_generation.py tests/test_candidate_generation.py
git commit -m "feat: expand heuristic candidate search to twenty"
```

### Task 2: Add explicit Nemotron Ollama provider support

**Files:**
- Modify: `des_multi_agent/llm/config.py`
- Modify: `des_multi_agent/llm/factory.py`
- Create: `des_multi_agent/llm/nemotron_provider.py`
- Modify: `des_multi_agent/llm/base.py`
- Modify: `des_multi_agent/llm/__init__.py`
- Modify: `tests/test_llm_factory.py`
- Modify: `tests/test_llm_profiles.py`
- Modify: `tests/test_llm_parser.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.factory import build_llm_provider


def test_provider_nemotron_returns_nemotron_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "nemotron",
            "model_name": "nemotron-3-nano:latest",
            "api_base_url": "http://localhost:11434",
        },
        request_fn=lambda *args, **kwargs: '{"message":{"content":"[]"}}',
    )
    assert provider.__class__.__name__ == "NemotronProvider"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_factory.py::test_provider_nemotron_returns_nemotron_provider -v`
Expected: FAIL because `nemotron` is not yet a supported provider.

- [ ] **Step 3: Write minimal implementation**

```python
# des_multi_agent/llm/nemotron_provider.py
from __future__ import annotations

from .base import BaseLLMProvider
from .errors import load_json_or_raise, response_error
from .specs import RequestProfile


class NemotronProvider(BaseLLMProvider):
    request_profile = RequestProfile(
        name="Nemotron",
        path_template="/api/chat",
        payload_style="ollama",
        api_key_in_header=True,
        api_key_in_query=False,
    )

    def extract_text(self, raw: str) -> str:
        data = load_json_or_raise("Nemotron", raw)
        if not isinstance(data, dict):
            raise response_error("Nemotron", "must return a JSON object with message.content", raw)
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        raise response_error("Nemotron", "is missing message.content", raw)
```

```python
# des_multi_agent/llm/factory.py
_PROVIDER_ALIASES = {
    "local": "ollama",
    "hosted": "openai",
    "openai_chat": "openai",
    "nemotron-3-nano": "nemotron",
    "nemotron3nano": "nemotron",
}

if provider == "nemotron":
    return NemotronProvider(
        model_name=str(llm_cfg.model_name or "nemotron-3-nano:latest"),
        api_base_url=str(llm_cfg.api_base_url or "http://localhost:11434"),
        api_key_env=llm_cfg.api_key_env,
        max_candidates=llm_cfg.max_candidates,
        max_tokens=llm_cfg.max_tokens,
        temperature=llm_cfg.temperature,
        timeout_seconds=llm_cfg.timeout_seconds,
        request_fn=request_impl,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_factory.py::test_provider_nemotron_returns_nemotron_provider -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/nemotron_provider.py des_multi_agent/llm/base.py des_multi_agent/llm/config.py des_multi_agent/llm/factory.py des_multi_agent/llm/__init__.py tests/test_llm_factory.py tests/test_llm_profiles.py tests/test_llm_parser.py
git commit -m "feat: add nemotron ollama provider"
```

### Task 3: Add a normalization interlayer for Ollama text output

**Files:**
- Modify: `des_multi_agent/llm/parser.py`
- Modify: `des_multi_agent/llm/base.py`
- Modify: `tests/test_llm_parser.py`
- Modify: `tests/test_llm_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.parser import parse_candidate_brainstorms


def test_parser_strips_thinking_trace_and_fenced_json():
    raw = """Thinking...
```json
[{"smiles":"OCCO","rationale":"polyol","family":"polyol"}]
```"""
    items = parse_candidate_brainstorms(raw)
    assert len(items) == 1
    assert items[0].smiles == "OCCO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_parser.py::test_parser_strips_thinking_trace_and_fenced_json -v`
Expected: FAIL if the parser still depends on raw JSON only.

- [ ] **Step 3: Write minimal implementation**

```python
def _extract_json_block(raw: str) -> str:
    text = _strip_code_fences(raw)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_parser.py::test_parser_strips_thinking_trace_and_fenced_json -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/parser.py tests/test_llm_parser.py tests/test_llm_orchestrator.py
git commit -m "feat: normalize ollama text before parsing"
```

### Task 4: Update docs and demo defaults for the new model option and 20-candidate search

**Files:**
- Modify: `llm.example.yaml`
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Modify: `examples/demo_des_search.py`
- Modify: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_tutorial_mentions_nemotron_and_twenty_candidate_search():
    text = Path("docs/tutorial.md").read_text(encoding="utf-8")
    assert "nemotron-3-nano" in text
    assert "20" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_des_search.py::test_tutorial_mentions_nemotron_and_twenty_candidate_search -v`
Expected: FAIL until the docs are updated.

- [ ] **Step 3: Write minimal implementation**

```yaml
llm:
  enabled: true
  provider: nemotron
  model_name: nemotron-3-nano:latest
  api_base_url: http://localhost:11434
  max_candidates: 20
  max_tokens: 1024
  temperature: 0.2
  timeout_seconds: 120.0
```

```python
# examples/demo_des_search.py
parser.add_argument("--n", type=int, default=20, help="Number of candidate partners to propose")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo_des_search.py::test_tutorial_mentions_nemotron_and_twenty_candidate_search -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm.example.yaml README.md docs/tutorial.md examples/README.md examples/demo_des_search.py tests/test_demo_des_search.py
git commit -m "docs: reflect nemotron and twenty candidate search"
```

### Task 5: End-to-end verification with Gemma and Nemotron configs

**Files:**
- Modify: none
- Test: `tests/test_llm_orchestrator.py`

- [ ] **Step 1: Run the targeted test suite**

Run: `python -m pytest tests/test_candidate_generation.py tests/test_llm_parser.py tests/test_llm_profiles.py tests/test_llm_factory.py tests/test_llm_orchestrator.py tests/test_demo_des_search.py -q`
Expected: PASS.

- [ ] **Step 2: Run the demo with Nemotron explicitly**

Run:
```bash
python -m examples.demo_des_search --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --llm-config /tmp/des_agent_nemotron.yaml
```
Expected: ranked DES report with up to 20 candidates and LLM sections if Nemotron returns parseable JSON.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: verify nemotron and twenty candidate DES workflow"
```

## Self-Review

- Spec coverage: candidate search expansion is covered in Task 1; explicit Nemotron support is covered in Task 2; output normalization is covered in Task 3; docs/demo updates are covered in Task 4; end-to-end verification is covered in Task 5.
- Placeholder scan: no TBD/TODO placeholders remain in the plan.
- Type consistency: `max_candidates` remains the shared control point across provider config, prompt generation, and parsing. `nemotron` is the explicit provider key, while `nemotron-3-nano` is the user-facing model name.
