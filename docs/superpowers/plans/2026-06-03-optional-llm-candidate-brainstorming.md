# Optional LLM Candidate Brainstorming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional LLM layer to the DES multi-agent system that can brainstorm extra candidate partners, generate explanations, and provide advisory critique while preserving the deterministic `ml_des_mp` scoring path as the final authority.

**Architecture:** Introduce a provider abstraction with disabled, local, and hosted modes. The deterministic candidate generator, chemistry filter, prediction stack, and DES classifier remain unchanged. The LLM layer will only add supplemental candidates and human-readable notes, with strict JSON validation and fallback to deterministic-only execution on errors. The CLI will read the LLM settings from config and run unchanged when the feature is disabled.

**Tech Stack:** Python, `des_multi_agent`, `ml_des_mp`, JSON parsing, optional provider SDKs, and `pytest`.

---

### Task 1: Define LLM data contracts and provider interface

**Files:**
- Create: `des_multi_agent/llm/__init__.py`
- Create: `des_multi_agent/llm/schemas.py`
- Create: `des_multi_agent/llm/provider.py`
- Create: `tests/test_llm_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.schemas import CandidateBrainstorm, ExplanationNote, CritiqueNote


def test_llm_schema_round_trip():
    brainstorm = CandidateBrainstorm(smiles="OCCO", rationale="small polyol", family="polyol")
    explanation = ExplanationNote(smiles="OCCO", summary="ranked highly", key_evidence=["low min Tm"])
    critique = CritiqueNote(smiles="OCCO", assessment="advisory only", concerns=["possible outlier"])

    assert brainstorm.smiles == "OCCO"
    assert explanation.key_evidence == ["low min Tm"]
    assert critique.assessment == "advisory only"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_schemas.py -v`
Expected: FAIL because `des_multi_agent.llm.schemas` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateBrainstorm:
    smiles: str
    rationale: str
    family: str


@dataclass(frozen=True)
class ExplanationNote:
    smiles: str
    summary: str
    key_evidence: list[str]


@dataclass(frozen=True)
class CritiqueNote:
    smiles: str
    assessment: str
    concerns: list[str]
```

```python
from abc import ABC, abstractmethod

from .schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote


class LLMProvider(ABC):
    @abstractmethod
    def brainstorm_candidates(self, component_a: str, constraints: dict | None, context: str) -> list[CandidateBrainstorm]:
        raise NotImplementedError

    @abstractmethod
    def generate_explanations(self, results, context: str) -> list[ExplanationNote]:
        raise NotImplementedError

    @abstractmethod
    def critique_results(self, results, context: str) -> list[CritiqueNote]:
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_schemas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/__init__.py des_multi_agent/llm/schemas.py des_multi_agent/llm/provider.py tests/test_llm_schemas.py
git commit -m "feat: define optional llm contracts"
```

### Task 2: Add provider configuration and selection

**Files:**
- Create: `des_multi_agent/llm/config.py`
- Create: `des_multi_agent/llm/client.py`
- Create: `des_multi_agent/llm/local_provider.py`
- Create: `des_multi_agent/llm/hosted_provider.py`
- Create: `des_multi_agent/llm/factory.py`
- Create: `tests/test_llm_factory.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.factory import build_llm_provider


def test_provider_disabled_returns_none():
    provider = build_llm_provider({"enabled": False})
    assert provider is None


def test_provider_local_returns_local_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "local",
            "model_name": "local-test",
            "api_base_url": "http://localhost:8000",
            "api_key_env": "LOCAL_LLM_API_KEY",
        },
        request_fn=lambda *args, **kwargs: '{"candidates":[]}',
    )
    assert provider.__class__.__name__ == "LocalLLMProvider"


def test_provider_hosted_returns_hosted_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "hosted",
            "model_name": "hosted-test",
            "api_base_url": "https://api.example.com",
            "api_key_env": "HOSTED_LLM_API_KEY",
        },
        request_fn=lambda *args, **kwargs: '{"candidates":[]}',
    )
    assert provider.__class__.__name__ == "HostedLLMProvider"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_factory.py -v`
Expected: FAIL because `des_multi_agent.llm.factory` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "disabled"
    model_name: str | None = None
    api_base_url: str | None = None
    api_key_env: str | None = None
    max_candidates: int = 10
    max_tokens: int = 512
    temperature: float = 0.2
    timeout_seconds: float = 30.0
```

```python
import json
from urllib import request as urllib_request


def post_json_chat(url: str, payload: dict, api_key: str | None = None, timeout_seconds: float = 30.0) -> str:
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=body, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib_request.urlopen(req, timeout=timeout_seconds) as resp:
        return resp.read().decode("utf-8")
```

```python
import os

from .client import post_json_chat
from .parser import parse_candidate_brainstorms
from .provider import LLMProvider
from .prompts import candidate_brainstorm_prompt


class LocalLLMProvider(LLMProvider):
    def __init__(self, *, model_name: str, api_base_url: str, api_key_env: str | None, request_fn=post_json_chat):
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.request_fn = request_fn

    def brainstorm_candidates(self, component_a: str, constraints: dict | None, context: str):
        api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        raw = self.request_fn(self.api_base_url, {"model": self.model_name, "prompt": candidate_brainstorm_prompt(component_a, constraints, context)}, api_key=api_key)
        return parse_candidate_brainstorms(raw)

    def generate_explanations(self, results, context: str):
        return []

    def critique_results(self, results, context: str):
        return []
```

```python
import os

from .client import post_json_chat
from .parser import parse_candidate_brainstorms
from .provider import LLMProvider
from .prompts import candidate_brainstorm_prompt


class HostedLLMProvider(LLMProvider):
    def __init__(self, *, model_name: str, api_base_url: str, api_key_env: str | None, request_fn=post_json_chat):
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.request_fn = request_fn

    def brainstorm_candidates(self, component_a: str, constraints: dict | None, context: str):
        api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        raw = self.request_fn(self.api_base_url, {"model": self.model_name, "prompt": candidate_brainstorm_prompt(component_a, constraints, context)}, api_key=api_key)
        return parse_candidate_brainstorms(raw)

    def generate_explanations(self, results, context: str):
        return []

    def critique_results(self, results, context: str):
        return []
```

```python
from .config import LLMConfig
from .hosted_provider import HostedLLMProvider
from .local_provider import LocalLLMProvider
from .provider import LLMProvider


def build_llm_provider(cfg: dict, request_fn=None) -> LLMProvider | None:
    if not cfg.get("enabled", False) or cfg.get("provider", "disabled") == "disabled":
        return None
    llm_cfg = LLMConfig(**cfg)
    if llm_cfg.provider == "local":
        return LocalLLMProvider(
            model_name=str(llm_cfg.model_name or ""),
            api_base_url=str(llm_cfg.api_base_url or ""),
            api_key_env=llm_cfg.api_key_env,
            request_fn=request_fn or post_json_chat,
        )
    if llm_cfg.provider == "hosted":
        return HostedLLMProvider(
            model_name=str(llm_cfg.model_name or ""),
            api_base_url=str(llm_cfg.api_base_url or ""),
            api_key_env=llm_cfg.api_key_env,
            request_fn=request_fn or post_json_chat,
        )
    raise ValueError(f"Unknown llm.provider: {llm_cfg.provider}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/config.py des_multi_agent/llm/client.py des_multi_agent/llm/local_provider.py des_multi_agent/llm/hosted_provider.py des_multi_agent/llm/factory.py tests/test_llm_factory.py
git commit -m "feat: add llm provider selection"
```

### Task 3: Implement structured prompting and output validation

**Files:**
- Create: `des_multi_agent/llm/prompts.py`
- Create: `des_multi_agent/llm/parser.py`
- Create: `tests/test_llm_parser.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.parser import parse_candidate_brainstorms


def test_parser_discards_invalid_entries():
    raw = '[{"smiles":"OCCO","rationale":"polyol","family":"polyol"},{"smiles":"","rationale":"bad","family":"polyol"}]'
    items = parse_candidate_brainstorms(raw)
    assert len(items) == 1
    assert items[0].smiles == "OCCO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_parser.py -v`
Expected: FAIL because parser module is missing.

- [ ] **Step 3: Write minimal implementation**

```python
import json

from .schemas import CandidateBrainstorm


def parse_candidate_brainstorms(raw: str) -> list[CandidateBrainstorm]:
    data = json.loads(raw)
    out = []
    for item in data:
        smiles = str(item.get("smiles", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        family = str(item.get("family", "")).strip()
        if not smiles or not rationale or not family:
            continue
        out.append(CandidateBrainstorm(smiles=smiles, rationale=rationale, family=family))
    return out
```

```python
def candidate_brainstorm_prompt(component_a: str, constraints: dict | None, context: str) -> str:
    return f"""Return JSON list of candidate partner molecules for component A={component_a}.
Constraints: {constraints}
Context: {context}
Each item must include smiles, rationale, family.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_parser.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/prompts.py des_multi_agent/llm/parser.py tests/test_llm_parser.py
git commit -m "feat: add llm prompt and parser utilities"
```

### Task 4: Add optional LLM brainstorming to the orchestrator

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Create: `tests/test_llm_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import CandidateProposal, MeltingPointEstimate


class _FakeLLM:
    def brainstorm_candidates(self, component_a, constraints, context):
        return [CandidateBrainstorm(smiles="OCCO", rationale="polyol", family="polyol")]

    def generate_explanations(self, results, context):
        return []

    def critique_results(self, results, context):
        return []


def test_llm_candidates_are_merged_but_still_filtered(monkeypatch):
    monkeypatch.setattr(orchestrator, "build_llm_provider", lambda cfg, request_fn=None: _FakeLLM())
    monkeypatch.setattr(orchestrator, "generate_candidates", lambda component_a, n, constraints=None: [
        CandidateProposal(smiles="O", rationale="baseline", family="alcohol")
    ])
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(orchestrator, "resolve_melting_point", lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5))
    monkeypatch.setattr(orchestrator, "predict_curve", lambda *args, **kwargs: CurvePrediction(smiles_a="CCO", smiles_b="O", ratios=[0.1], tm_pred_k=[250.0], t1_k=300.0, t2_k=300.0, checkpoint_path="ckpt.pt"))
    monkeypatch.setattr(orchestrator, "classify_des", lambda curve, thresholds: DesResult(curve=curve, absolute_pass=True, relative_pass=True, is_des=True, rationale="ok", min_tm_k=250.0))
    monkeypatch.setattr(orchestrator, "rank_results", lambda results: results)

    results = orchestrator.run_search(
        component_a="CCO",
        n=1,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        llm_cfg={"enabled": True, "provider": "local", "model_name": "local-test", "api_base_url": "http://localhost:8000"},
    )

    assert len(results) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_orchestrator.py -v`
Expected: FAIL because the orchestrator does not yet use the LLM provider.

- [ ] **Step 3: Write minimal implementation**

```python
from .candidate_generation import generate_candidates
from .chemistry_filter import filter_candidates
from .config import DEFAULT_ABSOLUTE_TM_MAX_K, DEFAULT_RELATIVE_DROP_MIN
from .evaluation import classify_des
from .paths import resolve_existing_path
from .prediction import predict_curve
from .property_resolution import resolve_melting_point
from .ranking import rank_results
from .schemas import DesThresholds
from .llm.factory import build_llm_provider


def run_search(component_a: str, n: int, checkpoint_path: str, config_path: str = "ml_des_mp/config.yaml", thresholds: DesThresholds | None = None, llm_cfg: dict | None = None):
    checkpoint_path = resolve_existing_path(checkpoint_path)
    config_path = resolve_existing_path(config_path)
    proposals = generate_candidates(component_a, n=n, constraints=None)
    if llm_cfg:
        provider = build_llm_provider(llm_cfg)
        if provider is not None:
            llm_candidates = provider.brainstorm_candidates(component_a, None, "candidate brainstorming")
            proposals = proposals + [p for p in llm_candidates if p.smiles not in {x.smiles for x in proposals}]
    filtered = filter_candidates(component_a, proposals)
    thresholds = thresholds or DesThresholds(
        absolute_tm_max_k=DEFAULT_ABSOLUTE_TM_MAX_K,
        relative_drop_min=DEFAULT_RELATIVE_DROP_MIN,
    )
    component_a_tp = resolve_melting_point(component_a)
    results = []
    for proposal in filtered:
        component_b_tp = resolve_melting_point(proposal.smiles)
        curve = predict_curve(
            component_a,
            proposal.smiles,
            t1_k=component_a_tp.tm_k,
            t2_k=component_b_tp.tm_k,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        result = classify_des(curve, thresholds)
        results.append(result)
    return rank_results(results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py tests/test_llm_orchestrator.py
git commit -m "feat: merge optional llm brainstorming into orchestrator"
```

### Task 5: Add LLM explanation and critique reporting

**Files:**
- Modify: `des_multi_agent/reporting.py`
- Create: `tests/test_llm_reporting.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.reporting import format_report


def test_report_can_attach_optional_llm_notes():
    report = format_report(
        [],
        explanation_notes=[type("Note", (), {"smiles": "OCCO", "summary": "ranked highly"})()],
        critique_notes=[type("Note", (), {"smiles": "OCCO", "assessment": "advisory only"})()],
    )
    assert "LLM explanations" in report
    assert "LLM critique" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_reporting.py -v`
Expected: FAIL because reporting does not yet accept LLM notes.

- [ ] **Step 3: Write minimal implementation**

```python
def format_report(results, explanation_notes=None, critique_notes=None) -> str:
    lines = ["smiles_b | is_des | min_tm_k | rationale"]
    for r in results:
        lines.append(f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {r.rationale}")
    if explanation_notes:
        lines.append("")
        lines.append("LLM explanations:")
        for note in explanation_notes:
            lines.append(f"{note.smiles} | {note.summary}")
    if critique_notes:
        lines.append("")
        lines.append("LLM critique:")
        for note in critique_notes:
            lines.append(f"{note.smiles} | {note.assessment}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_reporting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/reporting.py tests/test_llm_reporting.py
git commit -m "feat: add optional llm notes to reporting"
```

### Task 6: Add configuration and documentation

**Files:**
- Modify: `ml_des_mp/README.md`
- Modify: `des_multi_agent/config.py`
- Create: `tests/test_llm_config.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.config import LLMConfig


def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.enabled is False
    assert cfg.provider == "disabled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_config.py -v`
Expected: FAIL because config module is missing or incomplete.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "disabled"
    model_name: str | None = None
    api_base_url: str | None = None
    api_key_env: str | None = None
    max_candidates: int = 10
    max_tokens: int = 512
    temperature: float = 0.2
    timeout_seconds: float = 30.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_config.py -v`
Expected: PASS.

- [ ] **Step 5: Update docs and verify the full user flow**

Add an LLM section to `ml_des_mp/README.md` showing config examples for:

```yaml
llm:
  enabled: true
  provider: hosted
  model_name: gpt-4.1-mini
```

Also add the local-provider example and the disabled default.

Run: `pytest`
Expected: all tests pass, including LLM config and reporting tests.

- [ ] **Step 6: Commit**

```bash
git add ml_des_mp/README.md des_multi_agent/llm/config.py tests/test_llm_config.py
git commit -m "docs: document optional llm configuration"
```

## Verification Checklist

Before merging or publishing, verify all of the following:

- `pytest tests/test_llm_schemas.py -v`
- `pytest tests/test_llm_factory.py -v`
- `pytest tests/test_llm_parser.py -v`
- `pytest tests/test_llm_orchestrator.py -v`
- `pytest tests/test_llm_reporting.py -v`
- `pytest tests/test_llm_config.py -v`
- `pytest`

## Notes for the Implementer

- Keep the LLM optional and disabled by default.
- Do not let the LLM change the final DES label or numeric score.
- Validate LLM outputs before merging them into the deterministic pipeline.
- Prefer a shared provider abstraction so local and hosted backends differ only at the adapter layer.

