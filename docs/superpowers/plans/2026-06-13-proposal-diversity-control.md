# Proposal Diversity Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable proposal-diversity controller that suppresses exact and near-duplicate chemical proposals before ranking, while preserving a bounded family-aware fallback for DES generation and exposing the controls through the CLI.

**Architecture:** Introduce a standalone controller that accepts `CandidateProposal` objects, canonicalizes SMILES, computes fingerprint similarity, and returns accepted proposals plus suppression notes and fallback-family suggestions. Thread that controller into the DES orchestration path after heuristic, discovery, and LLM candidate collection, then optionally run one bounded replenishment pass using the controller’s suggested families. Expose the policy through CLI flags so users can tune diversity without editing code, while keeping the final deterministic scoring and ranking unchanged.

**Tech Stack:** Python 3.13, dataclasses, RDKit fingerprints/canonical SMILES, argparse CLI wiring, existing `CandidateProposal` schema, pytest, existing DES candidate-generation and orchestrator flow.

---

### Task 1: Add a reusable proposal-diversity controller module

**Files:**
- Create: `des_multi_agent/proposal_diversity.py`
- Modify: `des_multi_agent/schemas.py`
- Test: `tests/test_proposal_diversity.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.proposal_diversity import ProposalDiversityConfig, apply_proposal_diversity
from des_multi_agent.schemas import CandidateProposal


def test_apply_proposal_diversity_removes_exact_duplicates():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="polyol", family="polyol"),
        CandidateProposal(smiles="C(CO)O", rationale="same canonical molecule", family="polyol"),
    ]
    result = apply_proposal_diversity(
        "CCO",
        proposals,
        config=ProposalDiversityConfig(max_similarity=0.85),
    )
    assert [item.smiles for item in result.accepted] == ["OCCO"]
    assert any(item.smiles == "C(CO)O" for item in result.suppressed)


def test_apply_proposal_diversity_suppresses_near_duplicates():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="polyol", family="polyol"),
        CandidateProposal(smiles="OCCCO", rationale="near duplicate", family="polyol"),
    ]
    result = apply_proposal_diversity(
        "CCN",
        proposals,
        config=ProposalDiversityConfig(max_similarity=0.80, deduplicate_exact=True, deduplicate_near=True),
    )
    assert [item.smiles for item in result.accepted] == ["OCCO"]
    assert any(item.smiles == "OCCCO" for item in result.suppressed)


def test_apply_proposal_diversity_suggests_adjacent_families_when_des_budget_collapses():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="polyol", family="polyol"),
        CandidateProposal(smiles="C(CO)O", rationale="duplicate polyol", family="polyol"),
        CandidateProposal(smiles="CC(=O)N", rationale="amide", family="amide"),
    ]
    result = apply_proposal_diversity(
        "CCO",
        proposals,
        config=ProposalDiversityConfig(max_similarity=0.85, family_fallback=True, per_family_budget=1),
    )
    assert result.accepted
    assert result.suggested_families
    assert any(family in {"amide", "urea", "carboxylic acid"} for family in result.suggested_families)
```

- [ ] **Step 2: Run the focused controller tests and confirm they fail first**

Run: `python -m pytest tests/test_proposal_diversity.py -q`
Expected: FAIL because `ProposalDiversityConfig` and `apply_proposal_diversity` do not exist yet.

- [ ] **Step 3: Add the minimal controller implementation**

```python
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from .chemistry_filter import canonicalize_smiles
from .schemas import CandidateProposal


@dataclass(frozen=True)
class ProposalDiversityConfig:
    max_similarity: float = 0.85
    deduplicate_exact: bool = True
    deduplicate_near: bool = True
    family_fallback: bool = True
    per_family_budget: int | None = None

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "ProposalDiversityConfig":
        if mapping is None:
            return cls()
        return cls(
            max_similarity=float(mapping.get("max_similarity", 0.85)),
            deduplicate_exact=bool(mapping.get("deduplicate_exact", True)),
            deduplicate_near=bool(mapping.get("deduplicate_near", True)),
            family_fallback=bool(mapping.get("family_fallback", True)),
            per_family_budget=(int(mapping["per_family_budget"]) if mapping.get("per_family_budget") is not None else None),
        )

    def validate(self) -> None:
        if not 0.0 < self.max_similarity <= 1.0:
            raise ValueError(f"proposal diversity max_similarity must be in (0.0, 1.0], got {self.max_similarity}")
        if self.per_family_budget is not None and self.per_family_budget <= 0:
            raise ValueError(f"proposal diversity per_family_budget must be > 0 when set, got {self.per_family_budget}")


@dataclass(frozen=True)
class ProposalDiversityResult:
    accepted: list[CandidateProposal]
    suppressed: list[CandidateProposal]
    suggested_families: list[str]
    notes: list[str]


def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _similarity(left: str, right: str) -> float:
    return DataStructs.TanimotoSimilarity(_fingerprint(left), _fingerprint(right))


def apply_proposal_diversity(
    component_a: str,
    proposals: Sequence[CandidateProposal],
    *,
    config: ProposalDiversityConfig,
) -> ProposalDiversityResult:
    config.validate()
    seed = canonicalize_smiles(component_a)
    accepted: list[CandidateProposal] = []
    suppressed: list[CandidateProposal] = []
    notes: list[str] = []
    seen_canonical: set[str] = set()
    family_counts: dict[str, int] = defaultdict(int)

    for proposal in proposals:
        canonical = canonicalize_smiles(proposal.smiles)
        if canonical == seed:
            suppressed.append(proposal)
            notes.append(f"Suppressed proposal matching component A: {proposal.smiles}")
            continue
        if config.deduplicate_exact and canonical in seen_canonical:
            suppressed.append(proposal)
            notes.append(f"Suppressed exact duplicate proposal: {proposal.smiles}")
            continue
        if config.deduplicate_near and any(_similarity(canonical, item.smiles) >= config.max_similarity for item in accepted):
            suppressed.append(proposal)
            notes.append(f"Suppressed near-duplicate proposal: {proposal.smiles}")
            continue
        accepted.append(proposal)
        seen_canonical.add(canonical)
        family_counts[proposal.family] += 1

    suggested_families: list[str] = []
    if config.family_fallback:
        overused = [family for family, count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0])) if config.per_family_budget is not None and count >= config.per_family_budget]
        if overused:
            adjacent = ["amide", "urea", "carboxylic acid", "polyol", "diol", "ether alcohol"]
            for family in adjacent:
                if family not in overused and family not in suggested_families:
                    suggested_families.append(family)
                if config.per_family_budget is not None and len(suggested_families) >= config.per_family_budget:
                    break

    return ProposalDiversityResult(
        accepted=accepted,
        suppressed=suppressed,
        suggested_families=suggested_families,
        notes=notes,
    )
```

- [ ] **Step 4: Run the focused controller tests and confirm they pass**

Run: `python -m pytest tests/test_proposal_diversity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/proposal_diversity.py des_multi_agent/schemas.py tests/test_proposal_diversity.py
git commit -m "feat: add proposal diversity controller"
```

### Task 2: Thread the controller through DES orchestration and expose CLI flags

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_llm_orchestrator.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.proposal_diversity import ProposalDiversityConfig
from des_multi_agent.schemas import CandidateProposal


def test_orchestrator_deduplicates_across_heuristic_discovery_and_llm_sources(monkeypatch):
    class _FakeLLM:
        def review_candidate(self, component_a, candidate_smiles, context):
            from des_multi_agent.llm.schemas import CandidateReview
            return CandidateReview(smiles=candidate_smiles, decision="keep", confidence=0.9, rationale="ok", notes=[])

        def brainstorm_candidates(self, component_a, constraints, context, **kwargs):
            from des_multi_agent.llm.schemas import CandidateBrainstorm
            return [
                CandidateBrainstorm(smiles="OCCO", rationale="polyol", family="polyol"),
                CandidateBrainstorm(smiles="C(CO)O", rationale="duplicate polyol", family="polyol"),
            ]

        def generate_explanations(self, results, context):
            return []

        def critique_results(self, results, context):
            return []

        def detect_contradictions(self, results, context):
            return []

        def assess_candidate_chemistry(self, candidate_smiles, context, memory_notes=None):
            return []

        def suggest_next_steps(self, context, memory_notes=None):
            return []

    monkeypatch.setattr("des_multi_agent.orchestrator.build_llm_provider", lambda cfg, request_fn=None: _FakeLLM())
    monkeypatch.setattr(
        "des_multi_agent.orchestrator.generate_candidates",
        lambda component_a, n, constraints=None: [
            CandidateProposal(smiles="CC(=O)N", rationale="amide", family="amide"),
            CandidateProposal(smiles="CC(=O)N", rationale="duplicate amide", family="amide"),
        ],
    )
    monkeypatch.setattr("des_multi_agent.orchestrator.filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr("des_multi_agent.orchestrator.rank_results", lambda results: results)
    monkeypatch.setattr(
        "des_multi_agent.orchestrator.resolve_melting_point",
        lambda component, override_k=None: type("MP", (), {"tm_k": 300.0, "source": "mock", "confidence": 1.0})(),
    )
    monkeypatch.setattr(
        "des_multi_agent.orchestrator.predict_curve",
        lambda *args, **kwargs: type(
            "Curve",
            (),
            {"smiles_a": "CCO", "smiles_b": "OCCO", "ratios": [0.5], "tm_pred_k": [250.0], "t1_k": 300.0, "t2_k": 300.0, "checkpoint_path": "ckpt.pt"},
        )(),
    )
    monkeypatch.setattr(
        "des_multi_agent.orchestrator.classify_des",
        lambda curve, thresholds: type(
            "Res",
            (),
            {"curve": curve, "absolute_pass": True, "relative_pass": True, "is_des": True, "rationale": "ok", "min_tm_k": 250.0},
        )(),
    )

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=2,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        proposal_diversity_cfg={"max_similarity": 0.80, "family_fallback": True, "per_family_budget": 1},
        llm_cfg={"enabled": True, "provider": "ollama", "model_name": "llama3.1", "api_base_url": "http://localhost:11434"},
    )

    assert len(outcome.candidate_proposals) == 2
    assert len({proposal.smiles for proposal in outcome.candidate_proposals}) == 2
    assert any("near-duplicate" in note for note in outcome.memory_notes + outcome.llm_warnings)
```

```python
def test_cli_parser_accepts_proposal_diversity_flags():
    parser = build_parser()
    args = parser.parse_args([
        "--workflow",
        "des",
        "--component-a",
        "CCO",
        "--checkpoint-path",
        "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        "--proposal-max-similarity",
        "0.82",
        "--proposal-family-fallback",
        "--proposal-per-family-budget",
        "2",
    ])
    assert args.proposal_max_similarity == 0.82
    assert args.proposal_family_fallback is True
    assert args.proposal_per_family_budget == 2
```

- [ ] **Step 2: Run the orchestrator and CLI tests and confirm they fail first**

Run: `python -m pytest tests/test_llm_orchestrator.py tests/test_cli.py -q`
Expected: FAIL because `proposal_diversity_cfg` and the CLI flags do not exist yet.

- [ ] **Step 3: Add the minimal orchestration and CLI wiring**

```python
def run_search_report(
    component_a: str,
    n: int,
    checkpoint_path: str,
    config_path: str = "ml_des_mp/config.yaml",
    thresholds: DesThresholds | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
    llm_cfg: Mapping[str, object] | None = None,
    llm_request_fn=None,
    discovery_path: str | None = None,
    viscosity_model_path: str | None = None,
    save_run_memory_path: str | None = None,
    reuse_run_path: str | None = None,
    output_dir: str | None = None,
    ensemble_checkpoints: list[str] | None = None,
    candidates_file: str | None = None,
    viscosity_weight: float = 0.3,
    viscosity_threshold_cp: float | None = None,
    prior_cycle_top_results: list | None = None,
    prior_family_ledger: dict[str, int] | None = None,
    proposal_diversity_cfg: Mapping[str, object] | ProposalDiversityConfig | None = None,
):
    proposal_diversity = (
        proposal_diversity_cfg
        if isinstance(proposal_diversity_cfg, ProposalDiversityConfig)
        else ProposalDiversityConfig.from_mapping(proposal_diversity_cfg)
    )
```

```python
proposal_diversity = (
    proposal_diversity_cfg
    if isinstance(proposal_diversity_cfg, ProposalDiversityConfig)
    else ProposalDiversityConfig.from_mapping(proposal_diversity_cfg)
)
```

```python
merged_candidates, notes = _merge_candidates(discovery_candidates, heuristic_candidates)
diversity_result = apply_proposal_diversity(component_a, merged_candidates, config=proposal_diversity)
candidate_proposals = diversity_result.accepted
llm_warnings.extend(diversity_result.notes)
if diversity_result.suggested_families:
    replenishment = generate_candidates(
        component_a,
        n=max(0, n - len(candidate_proposals)),
        constraints={"allowed_families": diversity_result.suggested_families},
    )
    candidate_proposals, notes = _merge_candidates(candidate_proposals, replenishment)
```

```python
parser.add_argument("--proposal-max-similarity", type=_unit_float, default=0.85, dest="proposal_max_similarity")
parser.add_argument("--proposal-family-fallback", action="store_true", default=True, dest="proposal_family_fallback")
parser.add_argument("--proposal-per-family-budget", type=_positive_int, default=None, dest="proposal_per_family_budget")
```

Implementation details to include:

- apply proposal diversity after merging heuristic, discovery, and LLM candidate lists
- run one bounded replenishment pass if the controller suggests fallback families
- keep the deterministic filter and scoring pipeline unchanged
- pass CLI flags through to `run_search_report` as a proposal-diversity config mapping

- [ ] **Step 4: Run the orchestrator and CLI tests and confirm they pass**

Run: `python -m pytest tests/test_llm_orchestrator.py tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/cli.py tests/test_llm_orchestrator.py tests/test_cli.py
git commit -m "feat: expose proposal diversity controls"
```

### Task 3: Update docs and example coverage for proposal diversity

**Files:**
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Modify: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_docs_mention_proposal_diversity_controls():
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    assert "proposal diversity" in tutorial
    assert "near-duplicates" in tutorial
    examples = Path("examples/README.md").read_text(encoding="utf-8")
    assert "proposal diversity" in examples
    assert "near-duplicate" in examples
```

- [ ] **Step 2: Run the docs/example regression slice and confirm it fails first**

Run: `python -m pytest tests/test_demo_des_search.py -q`
Expected: FAIL because the tutorial and example README do not yet describe the new controller.

- [ ] **Step 3: Update the docs with the new proposal-diversity guidance**

```markdown
- The proposal-diversity controller removes exact and near-duplicate proposals before ranking.
- DES runs can reserve budget for nearby chemical families when a family becomes too repetitive.
- The controller is reusable across workflows and does not depend on live LLM calls.
```

- [ ] **Step 4: Run the docs/example regression slice and confirm it passes**

Run: `python -m pytest tests/test_demo_des_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/tutorial.md examples/README.md tests/test_demo_des_search.py
git commit -m "docs: describe proposal diversity controller"
```
