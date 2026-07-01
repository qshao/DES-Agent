# DES LLM Diversity Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit DES LLM brainstorming diversity controls with `explore`, `balanced`, and `exploit` modes, plus bounded family-count and prior-family-bias settings.

**Architecture:** Extend the existing LLM config with three DES brainstorming knobs, thread them through the provider layer into the DES family-selection and candidate-brainstorm prompts, and enrich iterative-cycle context so prior productive families are passed as structured prompt guidance. The deterministic DES predictor, ranking stack, and run-memory reuse remain unchanged.

**Tech Stack:** Python, argparse-free config dataclasses, existing LLM prompt/provider layer, pytest, markdown docs.

---

## File Map

- Modify: `des_multi_agent/llm/config.py`
  - add validated diversity config fields
- Modify: `des_multi_agent/llm/base.py`
  - thread diversity parameters into DES brainstorming only
- Modify: `des_multi_agent/llm/prompts.py`
  - add mode-aware helper text and structured prior-family prompt sections
- Modify: `des_multi_agent/orchestrator.py`
  - build structured prior-family guidance for DES iterative runs
- Modify: `docs/tutorial.md`
  - document the new controls and how they affect brainstorming only
- Modify: `examples/README.md`
  - add a short explanation of the new diversity behavior for LLM-backed DES examples
- Modify: `tests/test_config.py`
  - add config parsing/validation coverage
- Modify: `tests/test_llm_parser.py`
  - add prompt-generation coverage
- Modify: `tests/test_llm_provider.py`
  - add provider pass-through coverage

### Task 1: Add Diversity Config Fields

**Files:**
- Modify: `des_multi_agent/llm/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
from des_multi_agent.llm.config import LLMConfig
import pytest


def test_llm_config_defaults_des_diversity_controls():
    cfg = LLMConfig.from_mapping({"enabled": True, "provider": "ollama"})
    assert cfg.diversity_mode == "balanced"
    assert cfg.max_families == 6
    assert cfg.family_bias_strength == 0.5


def test_llm_config_accepts_valid_des_diversity_controls():
    cfg = LLMConfig.from_mapping(
        {
            "enabled": True,
            "provider": "ollama",
            "diversity_mode": "explore",
            "max_families": 4,
            "family_bias_strength": 0.75,
        }
    )
    assert cfg.diversity_mode == "explore"
    assert cfg.max_families == 4
    assert cfg.family_bias_strength == 0.75


def test_llm_config_rejects_invalid_diversity_mode():
    cfg = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        diversity_mode="novelty",
    )
    with pytest.raises(ValueError, match="Unsupported llm.diversity_mode"):
        cfg.validate()


def test_llm_config_rejects_non_positive_max_families():
    cfg = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        max_families=0,
    )
    with pytest.raises(ValueError, match="llm.max_families must be positive"):
        cfg.validate()


def test_llm_config_rejects_bias_strength_out_of_range():
    cfg = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        family_bias_strength=1.5,
    )
    with pytest.raises(ValueError, match="llm.family_bias_strength must be in"):
        cfg.validate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL with missing `diversity_mode`, `max_families`, or `family_bias_strength` on `LLMConfig`.

- [ ] **Step 3: Write the minimal implementation**

```python
from dataclasses import dataclass
from typing import Mapping


_ALLOWED_DIVERSITY_MODES = {"explore", "balanced", "exploit"}


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "disabled"
    model_name: str | None = None
    api_base_url: str | None = None
    api_key_env: str | None = None
    max_candidates: int = 20
    max_tokens: int = 512
    temperature: float = 0.2
    timeout_seconds: float = 30.0
    diversity_mode: str = "balanced"
    max_families: int = 6
    family_bias_strength: float = 0.5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "LLMConfig":
        if mapping is None:
            return cls()
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            provider=str(mapping.get("provider", "disabled")),
            model_name=mapping.get("model_name") or None,
            api_base_url=mapping.get("api_base_url") or None,
            api_key_env=mapping.get("api_key_env") or None,
            max_candidates=int(mapping.get("max_candidates", 20)),
            max_tokens=int(mapping.get("max_tokens", 512)),
            temperature=float(mapping.get("temperature", 0.2)),
            timeout_seconds=float(mapping.get("timeout_seconds", 30.0)),
            diversity_mode=str(mapping.get("diversity_mode", "balanced")),
            max_families=int(mapping.get("max_families", 6)),
            family_bias_strength=float(mapping.get("family_bias_strength", 0.5)),
        )

    def validate(self) -> None:
        provider = self.provider.strip().lower()
        mode = self.diversity_mode.strip().lower()
        if mode not in _ALLOWED_DIVERSITY_MODES:
            raise ValueError(f"Unsupported llm.diversity_mode: {self.diversity_mode}")
        if self.max_families <= 0:
            raise ValueError(f"llm.max_families must be positive, got {self.max_families}")
        if not 0.0 <= self.family_bias_strength <= 1.0:
            raise ValueError(
                f"llm.family_bias_strength must be in [0.0, 1.0], got {self.family_bias_strength}"
            )
        # keep the existing provider validation below this block
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/config.py tests/test_config.py
git commit -m "feat: add des llm diversity config"
```

### Task 2: Add Mode-Aware Prompt Helpers

**Files:**
- Modify: `des_multi_agent/llm/prompts.py`
- Test: `tests/test_llm_parser.py`

- [ ] **Step 1: Write the failing tests**

```python
from des_multi_agent.llm.prompts import candidate_brainstorm_prompt, family_selection_prompt


def test_family_selection_prompt_includes_balanced_diversity_guidance():
    prompt = family_selection_prompt(
        component_a="CCO",
        constraints=None,
        context="demo",
        max_families=4,
        diversity_mode="balanced",
        family_bias_strength=0.5,
        prior_productive_families={"polyol": 3, "amide": 1},
    )
    assert "balanced" in prompt
    assert "mix of productive and novel families" in prompt
    assert "polyol" in prompt
    assert "amide" in prompt
    assert "Return at most 4 families." in prompt


def test_candidate_brainstorm_prompt_includes_explore_guidance():
    prompt = candidate_brainstorm_prompt(
        component_a="CCO",
        constraints=None,
        context="demo",
        max_items=8,
        families=[],
        diversity_mode="explore",
        family_bias_strength=0.2,
        prior_productive_families={"polyol": 2},
    )
    assert "explore" in prompt
    assert "chemically distinct families" in prompt
    assert "polyol" in prompt
    assert "Return at most 8 items." in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_parser.py -q`
Expected: FAIL because the prompt builders do not accept the new arguments or do not emit the expected guidance.

- [ ] **Step 3: Write the minimal implementation**

```python
def _diversity_instruction(diversity_mode: str, family_bias_strength: float) -> str:
    if diversity_mode == "explore":
        return (
            "Diversity mode: explore.\n"
            "Prefer chemically distinct families and limit reuse of prior productive families.\n"
            f"Prior-family bias strength: {family_bias_strength:.2f} on a 0-1 scale.\n"
        )
    if diversity_mode == "exploit":
        return (
            "Diversity mode: exploit.\n"
            "Prefer families close to prior productive families unless chemistry strongly argues otherwise.\n"
            f"Prior-family bias strength: {family_bias_strength:.2f} on a 0-1 scale.\n"
        )
    return (
        "Diversity mode: balanced.\n"
        "Preserve a mix of productive and novel families.\n"
        f"Prior-family bias strength: {family_bias_strength:.2f} on a 0-1 scale.\n"
    )


def _prior_family_block(prior_productive_families: dict[str, int] | None) -> str:
    if not prior_productive_families:
        return ""
    lines = ["Prior productive families:\n"]
    for family, count in sorted(prior_productive_families.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  - {family}: {count} prior DES-positive hit(s)\n")
    return "".join(lines)


def family_selection_prompt(
    component_a: str,
    constraints: dict | None,
    context: str,
    max_families: int = 6,
    diversity_mode: str = "balanced",
    family_bias_strength: float = 0.5,
    prior_productive_families: dict[str, int] | None = None,
) -> str:
    return "".join([
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of chemical families to explore as DES partner candidates.\n",
        f"Component A: {component_a}\n",
        f"Constraints: {constraints or {}}\n",
        f"Context: {context}\n",
        _diversity_instruction(diversity_mode, family_bias_strength),
        _prior_family_block(prior_productive_families),
        f"Return at most {max_families} families.\n",
        'Each item must contain name, rationale, and hbd_hba_role ("HBD", "HBA", or "both").',
    ])


def candidate_brainstorm_prompt(
    component_a: str,
    constraints: dict | None,
    context: str,
    max_items: int | None = None,
    families: list | None = None,
    diversity_mode: str = "balanced",
    family_bias_strength: float = 0.5,
    prior_productive_families: dict[str, int] | None = None,
) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        "Return a JSON array of candidate partner molecules for DES screening.\n",
        f"Component A: {component_a}\n",
        f"Constraints: {constraints or {}}\n",
        f"Context: {context}\n",
        _diversity_instruction(diversity_mode, family_bias_strength),
        _prior_family_block(prior_productive_families),
    ]
    if families:
        parts.append("Distribute candidates across these chemical families:\n")
        for f in families:
            parts.append(f"  - {f.name}: {f.rationale} (role: {f.hbd_hba_role})\n")
    if max_items is not None:
        parts.append(f"Return at most {max_items} items.\n")
    parts.append("Each item must contain smiles, rationale, and family.")
    return "".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_parser.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/prompts.py tests/test_llm_parser.py
git commit -m "feat: add diversity-aware des llm prompts"
```

### Task 3: Thread Diversity Controls Through the Provider Layer

**Files:**
- Modify: `des_multi_agent/llm/base.py`
- Test: `tests/test_llm_provider.py`

- [ ] **Step 1: Write the failing tests**

```python
from des_multi_agent.llm.base import BaseLLMProvider


class DummyProvider(BaseLLMProvider):
    request_profile = type("Profile", (), {
        "path_template": "/v1/chat/completions",
        "api_key_in_header": False,
        "api_key_in_query": False,
        "payload_style": "openai",
    })()

    def extract_text(self, raw: str) -> str:
        return raw


def test_brainstorm_candidates_passes_diversity_controls(monkeypatch):
    captured = {}
    provider = DummyProvider(
        model_name="demo",
        api_base_url="http://example.com",
        max_candidates=7,
        temperature=0.2,
    )

    def fake_select(component_a, constraints, context, max_families, diversity_mode, family_bias_strength, prior_productive_families):
        captured["family_args"] = (
            max_families, diversity_mode, family_bias_strength, prior_productive_families
        )
        return []

    def fake_request(prompt):
        captured["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(provider, "select_candidate_families", fake_select)
    monkeypatch.setattr(provider, "_request", fake_request)

    provider.brainstorm_candidates(
        "CCO",
        None,
        "demo",
        max_families=4,
        diversity_mode="balanced",
        family_bias_strength=0.6,
        prior_productive_families={"polyol": 2},
    )

    assert captured["family_args"] == (4, "balanced", 0.6, {"polyol": 2})
    assert "balanced" in captured["prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_provider.py -q`
Expected: FAIL because the provider methods do not accept the new arguments.

- [ ] **Step 3: Write the minimal implementation**

```python
def brainstorm_candidates(
    self,
    component_a: str,
    constraints: dict | None,
    context: str,
    *,
    max_families: int = 6,
    diversity_mode: str = "balanced",
    family_bias_strength: float = 0.5,
    prior_productive_families: dict[str, int] | None = None,
) -> list[CandidateBrainstorm]:
    families: list[CandidateFamily] = []
    try:
        families = self.select_candidate_families(
            component_a,
            constraints,
            context,
            max_families=max_families,
            diversity_mode=diversity_mode,
            family_bias_strength=family_bias_strength,
            prior_productive_families=prior_productive_families,
        )
    except Exception as exc:
        print(f"family selection failed, falling back to single-stage brainstorm: {exc}", file=sys.stderr)
    raw = self._request(
        candidate_brainstorm_prompt(
            component_a,
            constraints,
            context,
            self.max_candidates,
            families,
            diversity_mode=diversity_mode,
            family_bias_strength=family_bias_strength,
            prior_productive_families=prior_productive_families,
        )
    )
    return parse_candidate_brainstorms(raw)[: self.max_candidates]


def select_candidate_families(
    self,
    component_a: str,
    constraints: dict | None,
    context: str,
    *,
    max_families: int = 6,
    diversity_mode: str = "balanced",
    family_bias_strength: float = 0.5,
    prior_productive_families: dict[str, int] | None = None,
) -> list[CandidateFamily]:
    raw = self._request(
        family_selection_prompt(
            component_a,
            constraints,
            context,
            max_families=max_families,
            diversity_mode=diversity_mode,
            family_bias_strength=family_bias_strength,
            prior_productive_families=prior_productive_families,
        )
    )
    return parse_candidate_families(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_provider.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/base.py tests/test_llm_provider.py
git commit -m "feat: thread des llm diversity controls through provider"
```

### Task 4: Wire Prior-Family Guidance into DES Orchestration

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Test: `tests/test_llm_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
from des_multi_agent.orchestrator import _build_prior_productive_family_summary


def test_build_prior_productive_family_summary_limits_to_top_counts():
    summary = _build_prior_productive_family_summary({"polyol": 4, "amide": 2, "acid": 1, "amine": 1})
    assert summary == {"polyol": 4, "amide": 2, "acid": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_orchestrator.py -q`
Expected: FAIL because `_build_prior_productive_family_summary` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
def _build_prior_productive_family_summary(
    family_ledger: dict[str, int] | None,
    limit: int = 3,
) -> dict[str, int]:
    if not family_ledger:
        return {}
    return dict(sorted(family_ledger.items(), key=lambda item: (-item[1], item[0]))[:limit])
```

Then update the DES LLM call site:

```python
prior_productive_families = _build_prior_productive_family_summary(prior_family_ledger)
llm_candidates = provider.brainstorm_candidates(
    component_a,
    None,
    brainstorm_context,
    max_families=getattr(provider, "max_families", 6),
    diversity_mode=getattr(provider, "diversity_mode", "balanced"),
    family_bias_strength=getattr(provider, "family_bias_strength", 0.5),
    prior_productive_families=prior_productive_families,
)
```

If provider attributes are not currently stored, add them during provider construction in Task 5 instead of using `getattr`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_orchestrator.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py tests/test_llm_orchestrator.py
git commit -m "feat: add prior family guidance for des llm brainstorming"
```

### Task 5: Store Diversity Controls on Provider Instances

**Files:**
- Modify: `des_multi_agent/llm/base.py`
- Modify: `des_multi_agent/llm/factory.py`
- Test: `tests/test_llm_provider.py`

- [ ] **Step 1: Write the failing tests**

```python
from des_multi_agent.llm.config import LLMConfig
from des_multi_agent.llm.factory import build_llm_provider


def test_build_llm_provider_sets_des_diversity_attributes():
    cfg = LLMConfig(
        enabled=True,
        provider="custom_http",
        model_name="demo",
        api_base_url="http://example.com",
        diversity_mode="exploit",
        max_families=3,
        family_bias_strength=0.8,
    )
    provider = build_llm_provider(cfg)
    assert provider.diversity_mode == "exploit"
    assert provider.max_families == 3
    assert provider.family_bias_strength == 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_provider.py -q`
Expected: FAIL because the provider instance does not expose these attributes.

- [ ] **Step 3: Write the minimal implementation**

```python
class BaseLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        model_name: str,
        api_base_url: str,
        api_key_env: str | None = None,
        max_candidates: int = 20,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout_seconds: float = 30.0,
        diversity_mode: str = "balanced",
        max_families: int = 6,
        family_bias_strength: float = 0.5,
        request_fn=post_json_chat,
    ):
        self.model_name = model_name
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.max_candidates = max_candidates
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.diversity_mode = diversity_mode
        self.max_families = max_families
        self.family_bias_strength = family_bias_strength
        self.transport = RequestTransport(request_fn=request_fn, timeout_seconds=timeout_seconds)
```

And pass the values in `des_multi_agent/llm/factory.py`:

```python
diversity_mode=llm_cfg.diversity_mode,
max_families=llm_cfg.max_families,
family_bias_strength=llm_cfg.family_bias_strength,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_provider.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/base.py des_multi_agent/llm/factory.py tests/test_llm_provider.py
git commit -m "feat: store des llm diversity controls on providers"
```

### Task 6: Document the New Controls

**Files:**
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`

- [ ] **Step 1: Add a tutorial section**

```md
### DES Brainstorm Diversity Controls

When LLM brainstorming is enabled for DES screening, you can control how broadly the model explores chemical families:

- `diversity_mode: explore` keeps family spread broad
- `diversity_mode: balanced` mixes prior productive families with novel ones
- `diversity_mode: exploit` concentrates on families that worked well in prior cycles

You can also set:

- `max_families` to cap how many families are explored in the family-selection stage
- `family_bias_strength` in `[0.0, 1.0]` to control how strongly prior productive families influence later iterative cycles

These settings affect LLM brainstorming only. They do not directly change the deterministic predictor, ranking rules, uncertainty logic, or run-memory reuse behavior.
```

- [ ] **Step 2: Add a short note to the examples README**

```md
LLM-backed DES workflows now support explicit brainstorming diversity controls through the LLM config:

- `diversity_mode: balanced` is the default
- `max_families` limits family spread during the two-stage brainstorm
- `family_bias_strength` controls how strongly prior productive families influence later cycles

These settings steer candidate generation only; final DES scoring still comes from the deterministic model.
```

- [ ] **Step 3: Verify the docs render cleanly**

Run: `python -m pytest tests/test_benchmarks_examples.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/tutorial.md examples/README.md
git commit -m "docs: explain des llm diversity controls"
```

### Task 7: Run Focused Verification

**Files:**
- Modify: none
- Test: `tests/test_config.py`
- Test: `tests/test_llm_parser.py`
- Test: `tests/test_llm_provider.py`
- Test: `tests/test_llm_orchestrator.py`

- [ ] **Step 1: Run the focused test suite**

Run: `python -m pytest tests/test_config.py tests/test_llm_parser.py tests/test_llm_provider.py tests/test_llm_orchestrator.py -q`
Expected: PASS

- [ ] **Step 2: Run the broader DES regression slice**

Run: `python -m pytest tests/test_cli.py tests/test_demo_des_search.py tests/test_benchmarks_examples.py -q`
Expected: PASS

- [ ] **Step 3: Inspect the diff**

Run: `git diff --stat`
Expected: only the intended LLM config, prompt, orchestrator, tests, and docs files changed.

- [ ] **Step 4: Commit the final integrated slice**

```bash
git add des_multi_agent/llm/config.py des_multi_agent/llm/base.py des_multi_agent/llm/factory.py des_multi_agent/llm/prompts.py des_multi_agent/orchestrator.py docs/tutorial.md examples/README.md tests/test_config.py tests/test_llm_parser.py tests/test_llm_provider.py tests/test_llm_orchestrator.py
git commit -m "feat: add des llm diversity controls"
```
