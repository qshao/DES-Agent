# Phase 2 Candidate Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local discovery layer that surfaces literature-like and similarity-based candidate partners before the existing heuristic generator and prediction pipeline.

**Architecture:** Add a small discovery subsystem with three responsibilities: load local reference data, retrieve candidate partners from literature and similarity sources, and merge those results with the current heuristic generator. Keep the downstream filter, property resolution, prediction, uncertainty, and ranking logic unchanged. Candidate provenance travels with `CandidateProposal`, and the orchestrator/reporting path renders that provenance alongside the DES summary.

**Tech Stack:** Python, RDKit, the existing `des_multi_agent` package, local YAML fixture files, and `pytest`.

---

### Task 1: Extend candidate proposals and add local discovery fixtures

**Files:**
- Modify: `des_multi_agent/schemas.py`
- Modify: `des_multi_agent/candidate_generation.py`
- Create: `des_multi_agent/discovery/library.py`
- Create: `des_multi_agent/discovery/__init__.py`
- Create: `tests/fixtures/discovery/literature.yaml`
- Create: `tests/fixtures/discovery/library.yaml`
- Create: `tests/test_discovery_library.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.discovery import DiscoveryLibrary, load_discovery_library
from des_multi_agent.schemas import CandidateProposal


def test_load_discovery_library_parses_literature_and_library():
    fixtures = Path(__file__).parent / "fixtures" / "discovery"
    library = load_discovery_library(fixtures)

    assert isinstance(library, DiscoveryLibrary)
    assert len(library.literature) == 1
    assert len(library.candidate_library) == 3
    assert library.literature[0].component_b == "OCCO"
    assert library.candidate_library[0].smiles == "O"


def test_candidate_proposal_carries_provenance_fields():
    proposal = CandidateProposal(
        smiles="OCCO",
        rationale="Known local hit",
        family="literature",
        source="literature",
        source_id="LIT-001",
        similarity_score=0.91,
        reference_note="Curated local record",
    )

    assert proposal.source == "literature"
    assert proposal.source_id == "LIT-001"
    assert proposal.similarity_score == 0.91
    assert proposal.reference_note == "Curated local record"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_discovery_library.py -q`
Expected: FAIL because the new discovery package and provenance fields do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# des_multi_agent/schemas.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateProposal:
    smiles: str
    rationale: str
    family: str
    source: str = "heuristic"
    source_id: str = ""
    similarity_score: float | None = None
    reference_note: str = ""


@dataclass(frozen=True)
class MeltingPointEstimate:
    component: str
    tm_k: float
    source: str
    confidence: float


@dataclass(frozen=True)
class DesThresholds:
    absolute_tm_max_k: float
    relative_drop_min: float
```

```python
# des_multi_agent/candidate_generation.py
from __future__ import annotations

from typing import Mapping, Sequence

from rdkit import Chem

from .chemistry_filter import canonicalize_smiles
from .schemas import CandidateProposal


_FAMILY_LIBRARY: Sequence[tuple[str, str, str]] = (
    ("alcohol", "hydrogen-bond donor", "O"),
    ("diol", "hydrogen-bond donor", "OCCO"),
    ("polyol", "hydrogen-bond donor", "OCC(O)CO"),
    ("amide", "hydrogen-bond acceptor", "CC(=O)N"),
    ("carboxylic acid", "hydrogen-bond donor", "CC(=O)O"),
    ("amine", "hydrogen-bond acceptor", "CN"),
    ("quaternary ammonium salt", "ionic partner", "C[N+](C)(C)C.[Cl-]"),
    ("choline-like", "ionic partner", "C[N+](C)(C)CCO.[Cl-]"),
    ("urea", "hydrogen-bond donor", "NC(=O)N"),
    ("sugar-like polyol", "hydrogen-bond donor", "OC[C@H](O)[C@H](O)CO"),
)


def _matches_constraints(family: str, smiles: str, constraints: Mapping[str, object] | None) -> bool:
    if not constraints:
        return True
    allowed_families = constraints.get("allowed_families")
    if allowed_families and family not in set(str(x) for x in allowed_families):
        return False
    excluded_smiles = constraints.get("excluded_smiles")
    if excluded_smiles and smiles in set(str(x) for x in excluded_smiles):
        return False
    return True


def generate_candidates(component_a: str, n: int, constraints=None):
    proposals: list[CandidateProposal] = []
    if n <= 0:
        return proposals

    component_mol = Chem.MolFromSmiles(component_a)
    if component_mol is None:
        raise ValueError("Invalid component A SMILES")

    canonical_component_a = canonicalize_smiles(component_a)
    for family, rationale, smiles in _FAMILY_LIBRARY:
        if canonicalize_smiles(smiles) == canonical_component_a:
            continue
        if not _matches_constraints(family, smiles, constraints):
            continue
        proposals.append(
            CandidateProposal(
                smiles=smiles,
                rationale=f"{rationale} candidate from a small rule-based family",
                family=family,
                source="heuristic",
                source_id="rule-based-family-library",
            )
        )
        if len(proposals) >= n:
            return proposals

    if len(proposals) < n:
        raise ValueError(
            f"Unable to generate {n} unique candidate(s) for component A after applying constraints."
        )
    return proposals
```

```python
# des_multi_agent/discovery/library.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..chemistry_filter import canonicalize_smiles


@dataclass(frozen=True)
class LiteratureRecord:
    component_a: str
    component_b: str
    source: str
    note: str = ""
    reference_id: str = ""


@dataclass(frozen=True)
class LibraryRecord:
    smiles: str
    family: str
    source: str
    note: str = ""


@dataclass(frozen=True)
class DiscoveryLibrary:
    literature: tuple[LiteratureRecord, ...] = ()
    candidate_library: tuple[LibraryRecord, ...] = ()


def _load_yaml_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    if not isinstance(raw, list):
        raise ValueError(f"Discovery file {path} must contain a list of records")
    return raw


def load_discovery_library(path: str | Path) -> DiscoveryLibrary:
    base = Path(path)
    literature_path = base / "literature.yaml"
    library_path = base / "library.yaml"
    literature: list[LiteratureRecord] = []
    candidate_library: list[LibraryRecord] = []

    if literature_path.exists():
        for row in _load_yaml_records(literature_path):
            missing = {key for key in ("component_a", "component_b", "source") if key not in row}
            if missing:
                raise ValueError(f"{literature_path} is missing required keys: {sorted(missing)}")
            literature.append(
                LiteratureRecord(
                    component_a=str(row["component_a"]),
                    component_b=str(row["component_b"]),
                    source=str(row["source"]),
                    note=str(row.get("note", "")),
                    reference_id=str(row.get("reference_id", "")),
                )
            )

    if library_path.exists():
        for row in _load_yaml_records(library_path):
            missing = {key for key in ("smiles", "family", "source") if key not in row}
            if missing:
                raise ValueError(f"{library_path} is missing required keys: {sorted(missing)}")
            candidate_library.append(
                LibraryRecord(
                    smiles=canonicalize_smiles(str(row["smiles"])),
                    family=str(row["family"]),
                    source=str(row["source"]),
                    note=str(row.get("note", "")),
                )
            )

    return DiscoveryLibrary(literature=tuple(literature), candidate_library=tuple(candidate_library))
```

```python
# des_multi_agent/discovery/__init__.py
from .library import DiscoveryLibrary, LibraryRecord, LiteratureRecord, load_discovery_library

__all__ = ["DiscoveryLibrary", "LibraryRecord", "LiteratureRecord", "load_discovery_library"]
```

```yaml
# tests/fixtures/discovery/literature.yaml
- component_a: CCO
  component_b: OCCO
  source: local_literature
  note: Known diol partner for ethanol-like donors.
  reference_id: LIT-001
```

```yaml
# tests/fixtures/discovery/library.yaml
- smiles: O
  family: alcohol
  source: curated_library
  note: Small hydrogen-bond donor.
- smiles: OCCO
  family: diol
  source: curated_library
  note: Local diol candidate.
- smiles: OCC(O)CO
  family: polyol
  source: curated_library
  note: Local polyol candidate.
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_discovery_library.py -q`
Expected: PASS once the provenance fields and discovery loader exist.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/schemas.py des_multi_agent/candidate_generation.py des_multi_agent/discovery/library.py des_multi_agent/discovery/__init__.py tests/fixtures/discovery/literature.yaml tests/fixtures/discovery/library.yaml tests/test_discovery_library.py
git commit -m "feat: add discovery library loader"
```

### Task 2: Implement literature lookup, similarity search, and candidate merging

**Files:**
- Create: `des_multi_agent/discovery/literature.py`
- Create: `des_multi_agent/discovery/similarity.py`
- Create: `des_multi_agent/discovery/merge.py`
- Update: `des_multi_agent/discovery/__init__.py`
- Create: `tests/test_discovery_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent.discovery import literature_lookup, load_discovery_library, merge_discovery_candidates, similarity_search


def test_discovery_returns_literature_and_similarity_hits():
    library = load_discovery_library(Path(__file__).parent / "fixtures" / "discovery")
    literature = literature_lookup("CCO", library)
    similar = similarity_search("CCO", library, limit=2)
    merged = merge_discovery_candidates(literature, similar)

    assert literature
    assert similar
    assert merged
    assert {candidate.source for candidate in merged} <= {"literature", "similarity"}
    assert merged[0].smiles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_discovery_retrieval.py -q`
Expected: FAIL because the retrieval and merge modules do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# des_multi_agent/discovery/literature.py
from __future__ import annotations

from ..chemistry_filter import canonicalize_smiles
from ..schemas import CandidateProposal
from .library import DiscoveryLibrary


def literature_lookup(component_a: str, library: DiscoveryLibrary) -> list[CandidateProposal]:
    canonical_a = canonicalize_smiles(component_a)
    hits: list[CandidateProposal] = []
    for record in library.literature:
        if canonicalize_smiles(record.component_a) != canonical_a:
            continue
        hits.append(
            CandidateProposal(
                smiles=record.component_b,
                rationale=record.note or f"Local literature match from {record.reference_id or record.source}",
                family="literature",
                source="literature",
                source_id=record.reference_id or record.source,
                reference_note=record.note,
            )
        )
    return hits
```

```python
# des_multi_agent/discovery/similarity.py
from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ..schemas import CandidateProposal
from .library import DiscoveryLibrary


def similarity_search(component_a: str, library: DiscoveryLibrary, limit: int) -> list[CandidateProposal]:
    mol_a = Chem.MolFromSmiles(component_a)
    if mol_a is None:
        raise ValueError("Invalid component A SMILES")
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    scored: list[tuple[float, CandidateProposal]] = []
    for record in library.candidate_library:
        mol_b = Chem.MolFromSmiles(record.smiles)
        if mol_b is None:
            continue
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
        score = DataStructs.TanimotoSimilarity(fp_a, fp_b)
        scored.append(
            (
                score,
                CandidateProposal(
                    smiles=record.smiles,
                    rationale=record.note or f"Local similarity hit from {record.source}",
                    family=record.family,
                    source="similarity",
                    source_id=record.source,
                    similarity_score=float(score),
                    reference_note=record.note,
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1].smiles))
    return [candidate for _, candidate in scored[:limit]]
```

```python
# des_multi_agent/discovery/merge.py
from __future__ import annotations

from ..chemistry_filter import canonicalize_smiles
from ..schemas import CandidateProposal


def merge_discovery_candidates(*candidate_groups) -> list[CandidateProposal]:
    merged: list[CandidateProposal] = []
    seen: set[str] = set()
    for group in candidate_groups:
        for candidate in group:
            canonical = canonicalize_smiles(candidate.smiles)
            if canonical in seen:
                continue
            seen.add(canonical)
            merged.append(candidate)
    return merged
```

```python
# des_multi_agent/discovery/__init__.py
from .library import DiscoveryLibrary, LibraryRecord, LiteratureRecord, load_discovery_library
from .literature import literature_lookup
from .merge import merge_discovery_candidates
from .similarity import similarity_search

__all__ = [
    "DiscoveryLibrary",
    "LibraryRecord",
    "LiteratureRecord",
    "load_discovery_library",
    "literature_lookup",
    "merge_discovery_candidates",
    "similarity_search",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_discovery_retrieval.py -q`
Expected: PASS with literature hits, similarity hits, and deduplicated merged output.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/discovery/literature.py des_multi_agent/discovery/similarity.py des_multi_agent/discovery/merge.py des_multi_agent/discovery/__init__.py tests/test_discovery_retrieval.py
git commit -m "feat: add local discovery retrieval"
```

### Task 3: Wire discovery into the orchestrator and report provenance

**Files:**
- Modify: `des_multi_agent/orchestrator.py`
- Modify: `des_multi_agent/reporting.py`
- Modify: `des_multi_agent/cli.py`
- Modify: `des_multi_agent/discovery/__init__.py` if exports need to be surfaced to callers
- Create: `tests/test_discovery_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.reporting import format_report
from des_multi_agent.schemas import DesThresholds
from des_multi_agent.uncertainty import MinimumTmUncertainty, UncertaintyPolicy


def _curve(smiles_a: str, smiles_b: str, min_tm_k: float) -> CurvePrediction:
    return CurvePrediction(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[min_tm_k + 5.0, min_tm_k, min_tm_k + 2.0],
        t1_k=298.15,
        t2_k=300.0,
        checkpoint_path="ckpt.pt",
    )


def _result(smiles_a: str, smiles_b: str, min_tm_k: float) -> DesResult:
    curve = _curve(smiles_a, smiles_b, min_tm_k)
    return DesResult(
        curve=curve,
        absolute_pass=True,
        relative_pass=True,
        is_des=True,
        rationale="ok",
        min_tm_k=min_tm_k,
    )


def _uncertainty(smiles_b: str) -> MinimumTmUncertainty:
    return MinimumTmUncertainty(
        component_a="CCO",
        component_b=smiles_b,
        repeated_values=(238.0, 239.0, 240.0),
        mean_tm_k=239.0,
        std_tm_k=1.0,
        min_tm_k=238.0,
        max_tm_k=240.0,
        trust_score=0.88,
        uncertainty_flag="low",
        explanation="demo",
        checkpoint_path="ckpt.pt",
        config_path="config.yaml",
    )


def test_run_search_report_includes_discovery_provenance(monkeypatch, tmp_path):
    fixture_dir = Path(__file__).parent / "fixtures" / "discovery"
    monkeypatch.setattr(orchestrator, "generate_candidates", lambda component_a, n, constraints=None: [])
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": _curve(component_a, component_b, 230.0 if component_b == "OCCO" else 220.0),
    )
    monkeypatch.setattr(orchestrator, "classify_des", lambda curve, thresholds: _result(curve.smiles_a, curve.smiles_b, min(curve.tm_pred_k)))
    monkeypatch.setattr(orchestrator, "estimate_min_tm_uncertainty", lambda component_a, component_b, checkpoint_path, config_path: _uncertainty(component_b))

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=2,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="report_only"),
        discovery_path=str(fixture_dir),
    )

    report = format_report(
        outcome.results,
        annotated_results=outcome.annotated_results,
        candidate_proposals=outcome.candidate_proposals,
        llm_warnings=outcome.llm_warnings,
    )

    assert any(candidate.source == "literature" for candidate in outcome.candidate_proposals)
    assert "source=literature" in report
    assert "source=similarity" in report or "source=heuristic" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_discovery_orchestrator.py -q`
Expected: FAIL because `run_search_report` and `format_report` do not yet accept or render discovery provenance.

- [ ] **Step 3: Write minimal implementation**

```python
# des_multi_agent/orchestrator.py
from .discovery import load_discovery_library, literature_lookup, merge_discovery_candidates, similarity_search
from .schemas import CandidateProposal


@dataclass(frozen=True)
class SearchOutcome:
    results: list[DesResult]
    annotated_results: list[AnnotatedResult]
    candidate_proposals: list[CandidateProposal]
    brainstorm_candidates: list[CandidateBrainstorm]
    explanation_notes: list[ExplanationNote]
    critique_notes: list[CritiqueNote]
    llm_warnings: list[str]


def _build_discovery_candidates(component_a: str, n: int, discovery_path: str | None) -> list[CandidateProposal]:
    if not discovery_path:
        return []
    library = load_discovery_library(discovery_path)
    literature = literature_lookup(component_a, library)
    similar = similarity_search(component_a, library, limit=n)
    return merge_discovery_candidates(literature, similar)
```

```python
# des_multi_agent/reporting.py
from collections.abc import Sequence
from .schemas import CandidateProposal


def format_report(
    results,
    annotated_results: Sequence[AnnotatedResult] | None = None,
    candidate_proposals: Sequence[CandidateProposal] | None = None,
    explanation_notes: Sequence[ExplanationNote] | None = None,
    critique_notes: Sequence[CritiqueNote] | None = None,
    brainstorm_candidates: Sequence[CandidateBrainstorm] | None = None,
    llm_warnings: Sequence[str] | None = None,
) -> str:
    proposal_by_smiles = {item.smiles: item for item in candidate_proposals or []}
    annotation_by_smiles = {item.result.curve.smiles_b: item for item in annotated_results or []}
    if annotation_by_smiles:
        lines = ["smiles_b | is_des | min_tm_k | source | trust | mean_tm_k | spread_k | std_k | uncertainty_flag | rationale"]
    else:
        lines = ["smiles_b | is_des | min_tm_k | source | rationale"]
    for r in results:
        proposal = proposal_by_smiles.get(r.curve.smiles_b)
        source_text = "heuristic"
        if proposal is not None:
            parts = [f"source={proposal.source}"]
            if proposal.source_id:
                parts.append(f"id={proposal.source_id}")
            if proposal.similarity_score is not None:
                parts.append(f"sim={proposal.similarity_score:.2f}")
            source_text = "; ".join(parts)
        annotation = annotation_by_smiles.get(r.curve.smiles_b)
        if annotation is None:
            lines.append(f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {source_text} | {r.rationale}")
            continue
        lines.append(
            f"{r.curve.smiles_b} | {r.is_des} | {r.min_tm_k:.2f} | {source_text} | "
            f"trust={annotation.trust_score:.2f} | mean={annotation.uncertainty.mean_tm_k:.2f} K | "
            f"spread={annotation.uncertainty.min_tm_k:.2f}-{annotation.uncertainty.max_tm_k:.2f} K | "
            f"std={annotation.uncertainty.std_tm_k:.2f} K | flag={annotation.uncertainty.uncertainty_flag} | {r.rationale}"
        )
```

```python
# des_multi_agent/cli.py
outcome = run_search_report(
    component_a=args.component_a,
    n=args.n,
    checkpoint_path=str(checkpoint_path),
    config_path=str(config_path),
    discovery_path=args.discovery_path,
    llm_cfg=llm_cfg,
    uncertainty_policy=uncertainty_policy,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_discovery_orchestrator.py -q`
Expected: PASS once discovery candidates are merged, provenance survives, and the report includes source labels.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/orchestrator.py des_multi_agent/reporting.py des_multi_agent/cli.py tests/test_discovery_orchestrator.py
git commit -m "feat: wire discovery into orchestrator"
```

### Task 4: Add regression coverage for malformed discovery data and empty fallbacks

**Files:**
- Create: `tests/test_discovery_errors.py`
- Modify: `des_multi_agent/discovery/library.py`
- Modify: `des_multi_agent/orchestrator.py` if fallback warnings need to be surfaced

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from des_multi_agent.discovery import load_discovery_library
from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.schemas import DesThresholds
from des_multi_agent.uncertainty import MinimumTmUncertainty, UncertaintyPolicy


def test_load_discovery_library_rejects_malformed_records(tmp_path: Path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "literature.yaml").write_text("- component_a: CCO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="literature.yaml"):
        load_discovery_library(bad_dir)


def test_run_search_report_falls_back_when_discovery_is_empty(monkeypatch, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(orchestrator, "generate_candidates", lambda component_a, n, constraints=None: [])
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(component=component, tm_k=300.0, source="heuristic", confidence=0.5),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml": CurvePrediction(
            smiles_a=component_a,
            smiles_b=component_b,
            ratios=[0.1, 0.5, 0.9],
            tm_pred_k=[230.0, 229.0, 231.0],
            t1_k=t1_k,
            t2_k=t2_k,
            checkpoint_path=checkpoint_path,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "classify_des",
        lambda curve, thresholds: DesResult(
            curve=curve,
            absolute_pass=True,
            relative_pass=True,
            is_des=True,
            rationale="ok",
            min_tm_k=min(curve.tm_pred_k),
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "estimate_min_tm_uncertainty",
        lambda component_a, component_b, checkpoint_path, config_path: MinimumTmUncertainty(
            component_a=component_a,
            component_b=component_b,
            repeated_values=(229.0, 230.0, 231.0),
            mean_tm_k=230.0,
            std_tm_k=1.0,
            min_tm_k=229.0,
            max_tm_k=231.0,
            trust_score=0.9,
            uncertainty_flag="low",
            explanation="demo",
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        ),
    )

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=1,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        thresholds=DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.1),
        uncertainty_policy=UncertaintyPolicy(mode="report_only"),
        discovery_path=str(empty_dir),
    )

    assert outcome.results
    assert outcome.candidate_proposals
    assert outcome.candidate_proposals[0].source == "heuristic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_discovery_errors.py -q`
Expected: FAIL until malformed discovery records raise a clear `ValueError` and empty discovery falls back to the heuristic generator.

- [ ] **Step 3: Write minimal implementation**

```python
# des_multi_agent/discovery/library.py
# Raise ValueError with the file path and missing required fields for malformed records.
# Return an empty DiscoveryLibrary when the directory exists but contains no discovery files.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_discovery_errors.py -q`
Expected: PASS once malformed discovery input is rejected and empty discovery sources fall back cleanly.

- [ ] **Step 5: Commit**

```bash
git add tests/test_discovery_errors.py des_multi_agent/discovery/library.py des_multi_agent/orchestrator.py
git commit -m "test: add discovery error coverage"
```

### Final Verification

- [ ] Run the focused discovery tests:

```bash
python -m pytest tests/test_discovery_library.py tests/test_discovery_retrieval.py tests/test_discovery_orchestrator.py tests/test_discovery_errors.py -q
```

- [ ] Run the full suite:

```bash
python -m pytest -q
```

Expected: all tests pass, and the new discovery layer is visible in the report without changing the existing prediction or uncertainty behavior.
