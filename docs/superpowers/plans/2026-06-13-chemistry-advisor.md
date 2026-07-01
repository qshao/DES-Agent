# Chemistry Advisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable LLM chemistry-advisor layer that improves DES candidate proposal, explanation, warning, and next-step guidance while keeping deterministic prediction and ranking as the source of final scores.

**Architecture:** Add a small advisor interface in the LLM stack, with prompt builders and parsers for chemistry assessments and next-step suggestions. Wire the advisor into the DES workflow at three points: proposal, post-ranking explanation/warnings, and report generation. Use run memory as a soft prior only, so the advisor can reuse successful reasoning patterns without becoming dependent on them.

**Tech Stack:** Python 3.13, dataclasses, existing LLM provider abstraction, JSON prompt parsing, pytest, existing DES reporting and run-memory modules.

---

### Task 1: Add advisor schemas, prompts, and parsers

**Files:**
- Modify: `des_multi_agent/llm/schemas.py`
- Modify: `des_multi_agent/llm/prompts.py`
- Modify: `des_multi_agent/llm/parser.py`
- Test: `tests/test_llm_parser.py`
- Test: `tests/test_llm_candidate_families.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.llm.parser import parse_chemistry_assessments, parse_chemistry_next_steps
from des_multi_agent.llm.prompts import chemistry_assessment_prompt, chemistry_next_step_prompt


def test_chemistry_assessment_prompt_mentions_rationale_and_warnings():
    text = chemistry_assessment_prompt("CCO", "ctx", ["good memory note"])
    assert "rationale" in text
    assert "warnings" in text
    assert "good memory note" in text


def test_parse_chemistry_assessments_round_trips_json():
    raw = (
        '[{"smiles":"OCCO","decision":"keep","confidence":0.91,'
        '"rationale":"Strong H-bonding motif","warnings":["phase separation risk"]}]'
    )
    items = parse_chemistry_assessments(raw)
    assert items[0].smiles == "OCCO"
    assert items[0].decision == "keep"
    assert items[0].warnings == ["phase separation risk"]


def test_parse_chemistry_next_steps_round_trips_json():
    raw = (
        '[{"mode":"conservative","summary":"Tighten family set",'
        '"rationale":"Keep search narrow"},'
        '{"mode":"exploratory","summary":"Shift donor families",'
        '"rationale":"Probe nearby chemistry"}]'
    )
    items = parse_chemistry_next_steps(raw)
    assert [item.mode for item in items] == ["conservative", "exploratory"]
```

- [ ] **Step 2: Run the focused parser tests and confirm they fail first**

Run: `python -m pytest tests/test_llm_parser.py tests/test_llm_candidate_families.py -q`
Expected: FAIL because the new prompt builders and parser functions do not exist yet.

- [ ] **Step 3: Add the minimal schemas and parsing helpers**

```python
@dataclass(frozen=True)
class ChemistryAssessment:
    smiles: str
    decision: str
    confidence: float
    rationale: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChemistryNextStep:
    mode: str
    summary: str
    rationale: str
```

```python
def chemistry_assessment_prompt(
    candidate_smiles: str,
    context: str,
    memory_notes: Sequence[str] | None = None,
) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.
",
        "Return a JSON array of chemistry assessments for a DES candidate.
",
        f"Candidate: {candidate_smiles}
",
        f"Context: {context}
",
    ]
    if memory_notes:
        parts.append("Prior reasoning:
")
        parts.extend(f"- {note}
" for note in memory_notes)
    parts.append("Each item must contain smiles, decision, confidence, rationale, and warnings.")
    return "".join(parts)


def chemistry_next_step_prompt(
    context: str,
    memory_notes: Sequence[str] | None = None,
) -> str:
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.
",
        "Return a JSON array of chemistry next-step suggestions for the DES workflow.
",
        f"Context: {context}
",
    ]
    if memory_notes:
        parts.append("Prior reasoning:
")
        parts.extend(f"- {note}
" for note in memory_notes)
    parts.append("Each item must contain mode, summary, and rationale.")
    return "".join(parts)
```

```python
def parse_chemistry_assessments(raw: str) -> list[ChemistryAssessment]:
    payload = json.loads(raw)
    items: list[ChemistryAssessment] = []
    for item in payload:
        warnings = item.get("warnings", [])
        items.append(
            ChemistryAssessment(
                smiles=str(item["smiles"]),
                decision=str(item["decision"]),
                confidence=float(item["confidence"]),
                rationale=str(item["rationale"]),
                warnings=tuple(str(w) for w in warnings),
            )
        )
    return items


def parse_chemistry_next_steps(raw: str) -> list[ChemistryNextStep]:
    payload = json.loads(raw)
    return [
        ChemistryNextStep(
            mode=str(item["mode"]),
            summary=str(item["summary"]),
            rationale=str(item["rationale"]),
        )
        for item in payload
    ]
```

- [ ] **Step 4: Run the focused parser tests and confirm they pass**

Run: `python -m pytest tests/test_llm_parser.py tests/test_llm_candidate_families.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/schemas.py des_multi_agent/llm/prompts.py des_multi_agent/llm/parser.py tests/test_llm_parser.py tests/test_llm_candidate_families.py
git commit -m "feat: add chemistry advisor prompt and schema plumbing"
```

### Task 2: Extend the provider interface and implement chemistry-advisor calls

**Files:**
- Modify: `des_multi_agent/llm/provider.py`
- Modify: `des_multi_agent/llm/base.py`
- Modify: `des_multi_agent/llm/factory.py`
- Modify: `des_multi_agent/llm/__init__.py`
- Test: `tests/test_llm_factory.py`
- Test: `tests/test_llm_candidate_families.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.config import LLMConfig
from des_multi_agent.llm.factory import build_llm_provider


def test_provider_exposes_chemistry_advisor_methods():
    cfg = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
    )
    provider = build_llm_provider(cfg, request_fn=lambda *args, **kwargs: {})
    assert hasattr(provider, "assess_candidate_chemistry")
    assert hasattr(provider, "suggest_next_steps")
```

- [ ] **Step 2: Run the focused factory test and confirm it fails first**

Run: `python -m pytest tests/test_llm_factory.py -q`
Expected: FAIL because the provider methods are not implemented yet.

- [ ] **Step 3: Add the minimal provider interface and implementation**

```python
class LLMProvider(ABC):
    @abstractmethod
    def assess_candidate_chemistry(
        self,
        candidate_smiles: str,
        context: str,
        memory_notes: list[str] | None = None,
    ) -> list[ChemistryAssessment]:
        raise NotImplementedError

    @abstractmethod
    def suggest_next_steps(
        self,
        context: str,
        memory_notes: list[str] | None = None,
    ) -> list[ChemistryNextStep]:
        raise NotImplementedError
```

```python
class BaseLLMProvider(LLMProvider):
    def assess_candidate_chemistry(
        self,
        candidate_smiles: str,
        context: str,
        memory_notes: list[str] | None = None,
    ) -> list[ChemistryAssessment]:
        raw = self._request(chemistry_assessment_prompt(candidate_smiles, context, memory_notes))
        return parse_chemistry_assessments(raw)

    def suggest_next_steps(
        self,
        context: str,
        memory_notes: list[str] | None = None,
    ) -> list[ChemistryNextStep]:
        raw = self._request(chemistry_next_step_prompt(context, memory_notes))
        return parse_chemistry_next_steps(raw)
```

- [ ] **Step 4: Run the focused factory and candidate tests and confirm they pass**

Run: `python -m pytest tests/test_llm_factory.py tests/test_llm_candidate_families.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/provider.py des_multi_agent/llm/base.py des_multi_agent/llm/factory.py des_multi_agent/llm/__init__.py tests/test_llm_factory.py
git commit -m "feat: add chemistry advisor provider methods"
```

### Task 3: Integrate the advisor into the DES workflow and run-memory flow

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/run_memory.py`
- Modify: `des_multi_agent/reporting.py`
- Test: `tests/test_llm_orchestrator.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_run_memory.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.schemas import ChemistryAssessment, ChemistryNextStep
from des_multi_agent.reporting import format_report


def test_format_report_includes_chemistry_advisor_sections():
    report = format_report(
        [],
        advisor_assessments=[
            ChemistryAssessment(
                smiles="OCCO",
                decision="keep",
                confidence=0.91,
                rationale="Strong H-bonding motif",
                warnings=("phase separation risk",),
            )
        ],
        advisor_next_steps=[
            ChemistryNextStep(
                mode="conservative",
                summary="Tighten family set",
                rationale="Keep search narrow",
            ),
            ChemistryNextStep(
                mode="exploratory",
                summary="Shift donor families",
                rationale="Probe nearby chemistry",
            ),
        ],
    )
    assert "LLM chemistry advisor:" in report
    assert "LLM next steps:" in report
    assert "phase separation risk" in report
```

- [ ] **Step 2: Run the orchestrator/report tests and confirm they fail first**

Run: `python -m pytest tests/test_llm_orchestrator.py tests/test_reporting.py tests/test_run_memory.py -q`
Expected: FAIL because the report signature and workflow wiring do not exist yet.

- [ ] **Step 3: Add the workflow plumbing and memory summary helper**

```python
@dataclass(frozen=True)
class SearchOutcome:
    results: list[DesResult]
    annotated_results: list[AnnotatedResult]
    candidate_proposals: list[CandidateProposal]
    candidate_reviews: list[CandidateReview]
    brainstorm_candidates: list[CandidateBrainstorm]
    explanation_notes: list[ExplanationNote]
    critique_notes: list[CritiqueNote]
    llm_warnings: list[str]
    contradiction_notes: list[ContradictionNote] = field(default_factory=list)
    memory_notes: list[str] = field(default_factory=list)
    viscosity_predictions: list[ViscosityPrediction] = field(default_factory=list)
    advisor_assessments: list[ChemistryAssessment] = field(default_factory=list)
    advisor_next_steps: list[ChemistryNextStep] = field(default_factory=list)
    report_text: str = ""
```

```python
def build_chemistry_advisor_memory_notes(memory: RunMemory | Sequence[RunMemory] | None) -> list[str]:
    notes: list[str] = []
    for item in _iter_run_memories(memory):
        if item.labels:
            good = [label.smiles_b for label in item.labels if label.label == "good"]
            bad = [label.smiles_b for label in item.labels if label.label == "bad"]
            if good:
                notes.append(f"Prior good labels: {', '.join(good[:5])}")
            if bad:
                notes.append(f"Prior bad labels: {', '.join(bad[:5])}")
        if item.ranked_candidates:
            top = ", ".join(candidate.smiles_b for candidate in item.ranked_candidates[:3])
            notes.append(f"Prior top ranked candidates: {top}")
    return notes
```

```python
advisor_assessments = provider.assess_candidate_chemistry(
    proposal.smiles,
    advisor_context,
    memory_notes,
)
advisor_next_steps = provider.suggest_next_steps(advisor_context, memory_notes)
```

- [ ] **Step 4: Update the report formatter to render the new sections**

```python
if advisor_assessments:
    lines.append("")
    lines.append("LLM chemistry advisor:")
    for note in advisor_assessments:
        warnings = "; ".join(note.warnings) if note.warnings else "-"
        lines.append(f"{note.smiles} | {note.decision} | {note.rationale} | {warnings}")

if advisor_next_steps:
    lines.append("")
    lines.append("LLM next steps:")
    for step in advisor_next_steps:
        lines.append(f"{step.mode} | {step.summary} | {step.rationale}")
```

- [ ] **Step 5: Run the orchestrator/report tests and confirm they pass**

Run: `python -m pytest tests/test_llm_orchestrator.py tests/test_reporting.py tests/test_run_memory.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/run_memory.py des_multi_agent/reporting.py tests/test_llm_orchestrator.py tests/test_reporting.py tests/test_run_memory.py
git commit -m "feat: integrate chemistry advisor into DES workflow"
```

### Task 4: Update docs and tighten regression coverage

**Files:**
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Modify: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_tutorial_mentions_chemistry_advisor_modes():
    text = Path("docs/tutorial.md").read_text(encoding="utf-8")
    assert "chemical advisor" in text
    assert "conservative" in text
    assert "exploratory" in text
```

- [ ] **Step 2: Run the doc/regression slice and confirm it fails first**

Run: `python -m pytest tests/test_demo_des_search.py -q`
Expected: PASS for code regressions, while the new doc assertion is not present yet.

- [ ] **Step 3: Update the docs with the new workflow guidance**

```markdown
- The chemistry advisor adds rationale, warnings, and next-step suggestions.
- The advisor is a soft guide, not a replacement for deterministic prediction.
- Memory can bias reasoning, but it does not override current evidence.
```

- [ ] **Step 4: Run the regression slice and confirm it passes**

Run: `python -m pytest tests/test_demo_des_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/tutorial.md examples/README.md tests/test_demo_des_search.py
git commit -m "docs: document chemistry advisor workflow"
```
