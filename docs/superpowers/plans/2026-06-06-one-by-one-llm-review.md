# One-by-One LLM Candidate Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-candidate-at-a-time LLM review layer so the multi-agent system can evaluate up to 20 candidates without large JSON batch failures.

**Architecture:** Candidate discovery and deterministic DES prediction stay unchanged. After filtering, the orchestrator sends the top N candidates to the LLM one by one. Each request returns a strict JSON object for a single candidate, which the parser promotes into an internal review record and uses to adjust ranking or filtering. This keeps the LLM output small and isolates parse failures to a single candidate.

**Tech Stack:** Python, existing DES multi-agent pipeline, Ollama-backed LLM adapters, strict JSON parser, pytest.

---

## Task 1: Add a per-candidate review schema and parser

**Files:**
- Modify: `des_multi_agent/llm/schemas.py`
- Modify: `des_multi_agent/llm/parser.py`
- Modify: `tests/test_llm_parser.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.parser import parse_candidate_review


def test_parse_candidate_review_accepts_strict_json():
    raw = """
    {
      "smiles": "OCCO",
      "decision": "keep",
      "confidence": 0.87,
      "rationale": "Good hydrogen bonding candidate.",
      "notes": ["Stable", "Small polyol"]
    }
    """
    review = parse_candidate_review(raw)
    assert review.smiles == "OCCO"
    assert review.decision == "keep"
    assert review.confidence == 0.87
    assert review.rationale == "Good hydrogen bonding candidate."
    assert review.notes == ["Stable", "Small polyol"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_llm_parser.py::test_parse_candidate_review_accepts_strict_json -v`
Expected: fail because `parse_candidate_review` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
@dataclass(frozen=True)
class CandidateReview:
    smiles: str
    decision: str
    confidence: float
    rationale: str
    notes: list[str] = field(default_factory=list)
```

```python
def parse_candidate_review(raw: str) -> CandidateReview:
    payload = _extract_json_object(raw)
    if not isinstance(payload, dict):
        raise ValueError("Candidate review must be a JSON object")
    notes = payload.get("notes") or []
    if not isinstance(notes, list):
        raise ValueError("Candidate review notes must be a list")
    return CandidateReview(
        smiles=str(payload["smiles"]),
        decision=str(payload["decision"]),
        confidence=float(payload["confidence"]),
        rationale=str(payload["rationale"]),
        notes=[str(item) for item in notes],
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_llm_parser.py::test_parse_candidate_review_accepts_strict_json -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/schemas.py des_multi_agent/llm/parser.py tests/test_llm_parser.py
git commit -m "feat: add per-candidate llm review schema"
```

---

## Task 2: Add a one-candidate review prompt and provider method

**Files:**
- Modify: `des_multi_agent/llm/prompts.py`
- Modify: `des_multi_agent/llm/provider.py`
- Modify: `des_multi_agent/llm/base.py`
- Modify: `tests/test_llm_factory.py`
- Modify: `tests/test_llm_parser.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.prompts import candidate_review_prompt


def test_candidate_review_prompt_requests_one_json_object():
    prompt = candidate_review_prompt(
        component_a="CCN(CC)CC(=O)Nc1c(C)cccc1C",
        candidate_smiles="OCCO",
        context="demo context",
    )
    assert "Return raw JSON only" in prompt
    assert '"smiles": "OCCO"' in prompt
    assert "decision" in prompt
    assert "confidence" in prompt
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_llm_parser.py::test_candidate_review_prompt_requests_one_json_object -v`
Expected: fail because `candidate_review_prompt` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
def candidate_review_prompt(component_a: str, candidate_smiles: str, context: str) -> str:
    return (
        "Return raw JSON only. Do not use markdown fences or commentary.\n"
        "Return one JSON object for a single candidate review.\n"
        f"Component A: {component_a}\n"
        f"Candidate: {candidate_smiles}\n"
        f"Context: {context}\n"
        "The JSON object must contain smiles, decision, confidence, rationale, and notes.\n"
        "decision must be one of keep, reject, or deprioritize."
    )
```

```python
class LLMProvider(ABC):
    @abstractmethod
    def review_candidate(self, component_a: str, candidate_smiles: str, context: str):
        raise NotImplementedError
```

```python
def review_candidate(self, component_a: str, candidate_smiles: str, context: str):
    raw = self._request(candidate_review_prompt(component_a, candidate_smiles, context))
    return parse_candidate_review(raw)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_llm_parser.py tests/test_llm_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/llm/prompts.py des_multi_agent/llm/provider.py des_multi_agent/llm/base.py tests/test_llm_parser.py tests/test_llm_factory.py
git commit -m "feat: add one-by-one candidate review prompt"
```

---

## Task 3: Aggregate per-candidate LLM reviews in the orchestrator

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Modify: `tests/test_llm_orchestrator.py`
- Modify: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.llm.schemas import CandidateReview
from des_multi_agent.schemas import CandidateProposal
from des_multi_agent.orchestrator import _apply_candidate_reviews


def test_apply_candidate_reviews_deprioritizes_candidate():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="demo", family="polyol", source="heuristic", source_id="rule"),
        CandidateProposal(smiles="CC(=O)O", rationale="demo", family="acid", source="heuristic", source_id="rule"),
    ]
    reviews = {
        "OCCO": CandidateReview(
            smiles="OCCO",
            decision="deprioritize",
            confidence=0.25,
            rationale="Looks plausible but less compelling than alternatives.",
            notes=["low confidence"],
        )
    }
    reviewed, penalty_by_smiles = _apply_candidate_reviews(proposals, reviews)
    assert [item.smiles for item in reviewed] == ["OCCO", "CC(=O)O"]
    assert penalty_by_smiles == {"OCCO": 0.25}
```

```python
def test_apply_candidate_reviews_reject_drops_candidate():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="demo", family="polyol", source="heuristic", source_id="rule"),
        CandidateProposal(smiles="CC(=O)O", rationale="demo", family="acid", source="heuristic", source_id="rule"),
    ]
    reviews = {
        "OCCO": CandidateReview(
            smiles="OCCO",
            decision="reject",
            confidence=0.93,
            rationale="Does not look like a useful DES partner.",
            notes=["too similar to the input"],
        )
    }
    reviewed, penalty_by_smiles = _apply_candidate_reviews(proposals, reviews)
    assert [item.smiles for item in reviewed] == ["CC(=O)O"]
    assert penalty_by_smiles == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_llm_orchestrator.py::test_apply_candidate_reviews_deprioritizes_candidate -v`

Run: `python -m pytest tests/test_llm_orchestrator.py::test_apply_candidate_reviews_reject_drops_candidate -v`

Expected: fail because `_apply_candidate_reviews` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
def _apply_candidate_reviews(candidate_proposals, reviews):
    kept = []
    review_penalty_by_smiles = {}
    for proposal in candidate_proposals:
        review = reviews.get(proposal.smiles)
        if review is None:
            kept.append(proposal)
            continue
        if review.decision == "reject":
            continue
        if review.decision == "deprioritize":
            review_penalty_by_smiles[proposal.smiles] = 0.25
        kept.append(proposal)
    return kept, review_penalty_by_smiles
```

```python
def format_report(results, reviewed_candidates=None):
    if reviewed_candidates:
        lines.append("")
        lines.append("LLM candidate reviews:")
        for note in reviewed_candidates:
            lines.append(
                f"{note.smiles} | {note.decision} | confidence={note.confidence:.2f} | {note.rationale}"
            )
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_llm_orchestrator.py tests/test_demo_des_search.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/reporting.py tests/test_llm_orchestrator.py tests/test_demo_des_search.py
git commit -m "feat: review candidates one by one with llm"
```

---

## Task 4: Update docs and example runs to describe the new behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Modify: `examples/lidocaine_gemma4_12b/README.md`
- Modify: `examples/gemma4_12b/README.md`
- Modify: `examples/nemotron_3_nano/README.md`
- Modify: `examples/qwen3_6/README.md`

- [ ] **Step 1: Write the failing test**

```python
def test_tutorial_mentions_one_by_one_review():
    text = Path("docs/tutorial.md").read_text(encoding="utf-8")
    assert "one candidate at a time" in text
    assert "review_candidate" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_demo_des_search.py::test_tutorial_mentions_one_by_one_review -v`
Expected: fail until the docs are updated.

- [ ] **Step 3: Write the minimal implementation**

```markdown
Use `review_candidate` to score each filtered candidate one at a time.
The top N candidates are reviewed individually so large batches do not break Gemma JSON parsing.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_demo_des_search.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md examples/gemma4_12b/README.md examples/nemotron_3_nano/README.md examples/qwen3_6/README.md examples/lidocaine_gemma4_12b/README.md
git commit -m "docs: describe one-by-one llm review"
```

---

## Self-Review Checklist

- Spec coverage:
  - Per-candidate schema and parser: Task 1
  - Prompt and provider method: Task 2
  - Orchestrator merge and ranking: Task 3
  - Documentation updates: Task 4
- Placeholder scan:
  - No TBD/TODO placeholders remain.
  - Each task names exact files and includes commands.
- Type consistency:
  - `CandidateReview` is the new shared record.
  - `review_candidate(component_a, candidate_smiles, context)` is the new provider contract.
  - The orchestrator uses the same `smiles` field for validation and reporting.
