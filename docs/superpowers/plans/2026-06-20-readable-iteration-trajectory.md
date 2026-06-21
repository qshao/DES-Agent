# Readable Iteration Trajectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture and render the per-cycle trajectory of every iterative DES-Agent search as a durable `trajectory.md` artifact plus a tidy console trace.

**Architecture:** A new workflow-agnostic module `des_multi_agent/trajectory.py` owns the capture model (`TopEntry`/`CycleSnapshot`/`SearchTrajectory`), two pure renderers, and an atomic artifact writer. Each iterative workflow builds snapshots inside its own loop (where the cycle outcome is in hand) and attaches a `SearchTrajectory` to its existing outcome via an additive `None`-default field. The CLI prints the console trajectory and, when `--output-dir` is set, writes `trajectory.md`.

**Tech Stack:** Python 3.13, dataclasses, RDKit-derived display names (`des_multi_agent.smiles_names.display_name`), pytest.

## Global Constraints

- All new outcome fields are additive and default to `None` — no existing construction site or test may change behavior.
- `trajectory.py` MUST NOT import any workflow module (`multi_cycle`, `workflows/*`, `orchestrator`); workflows depend on it, never the reverse. It may import `smiles_names` and `reporting`.
- Capture is best-effort: a snapshot-build failure appends a `"[TRAJECTORY] …"` warning and continues; it never alters search results.
- Renderers are pure (no I/O) and assume a well-formed `SearchTrajectory`.
- The artifact writer writes atomically (temp file + `Path.replace`), mirroring `des_multi_agent/exporting.py`.
- Workflow names are exactly `"des"`, `"metal-selectivity"`, `"selectivity-des"`.
- Metric names are exactly `"min_tm_k"` (DES) and `"composite_score"` (both metal workflows).
- Existing final reports (`format_report*`, `format_metal_*`) are unchanged.

---

### Task 1: Trajectory data model + shortlist-diff helper

**Files:**
- Create: `des_multi_agent/trajectory.py`
- Test: `tests/test_trajectory_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TopEntry(label: str, metric_name: str, metric_value: float, secondary: str)` — frozen dataclass.
  - `CycleSnapshot(cycle: int, n_screened: int, n_hits: int, top_entries: list[TopEntry], new_entrants: list[str] = [], dropouts: list[str] = [], family_ledger: dict[str, int] = {}, converged: bool = False, convergence_reason: str = "", notable_warnings: list[str] = [])` — frozen dataclass (list/dict defaults via `field(default_factory=...)`).
  - `SearchTrajectory(workflow: str, headline: str, metric_label: str, snapshots: list[CycleSnapshot], total_cycles: int, converged: bool, convergence_reason: str, final_summary: list[TopEntry])` — frozen dataclass.
  - `shortlist_delta(prev_labels: list[str], curr_labels: list[str]) -> tuple[list[str], list[str]]` — returns `(new_entrants, dropouts)`, each sorted, computed as set differences of the two label lists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_model.py
from des_multi_agent.trajectory import (
    CycleSnapshot,
    SearchTrajectory,
    TopEntry,
    shortlist_delta,
)


def test_top_entry_fields():
    e = TopEntry(label="ethylene glycol (OCCO)", metric_name="min_tm_k",
                 metric_value=201.8, secondary="Δ22.6%, high confidence")
    assert e.metric_name == "min_tm_k"
    assert e.metric_value == 201.8


def test_cycle_snapshot_defaults():
    s = CycleSnapshot(cycle=1, n_screened=5, n_hits=5, top_entries=[])
    assert s.new_entrants == [] and s.dropouts == []
    assert s.family_ledger == {}
    assert s.converged is False
    assert s.convergence_reason == "" and s.notable_warnings == []
    # frozen-default lists are independent instances
    s2 = CycleSnapshot(cycle=2, n_screened=1, n_hits=0, top_entries=[])
    assert s.new_entrants is not s2.new_entrants


def test_search_trajectory_fields():
    t = SearchTrajectory(workflow="des", headline="DES partners for ethanol (CCO)",
                         metric_label="min Tm (K)", snapshots=[], total_cycles=0,
                         converged=False, convergence_reason="", final_summary=[])
    assert t.workflow == "des"


def test_shortlist_delta_sorted_set_diff():
    new, dropped = shortlist_delta(["water (O)", "urea (NC(N)=O)"],
                                   ["urea (NC(N)=O)", "glycerol (OCC(O)CO)"])
    assert new == ["glycerol (OCC(O)CO)"]
    assert dropped == ["water (O)"]


def test_shortlist_delta_first_cycle_all_new():
    new, dropped = shortlist_delta([], ["a", "b"])
    assert new == ["a", "b"] and dropped == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'des_multi_agent.trajectory'`

- [ ] **Step 3: Write minimal implementation**

```python
# des_multi_agent/trajectory.py
"""Readable iteration-trajectory model, renderers, and artifact writer.

Workflow-agnostic: this module is imported BY workflows, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopEntry:
    """One ranked candidate as it stood in a given cycle."""
    label: str            # display: "acetamide (CC(=O)N)"
    metric_name: str      # "min_tm_k" | "composite_score"
    metric_value: float
    secondary: str        # short context, e.g. "Δ11.9%, high confidence"


@dataclass(frozen=True)
class CycleSnapshot:
    """Render-oriented record of one iteration."""
    cycle: int
    n_screened: int
    n_hits: int
    top_entries: list[TopEntry]
    new_entrants: list[str] = field(default_factory=list)
    dropouts: list[str] = field(default_factory=list)
    family_ledger: dict[str, int] = field(default_factory=dict)
    converged: bool = False
    convergence_reason: str = ""
    notable_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchTrajectory:
    """Full readable record of an iterative run."""
    workflow: str
    headline: str
    metric_label: str
    snapshots: list[CycleSnapshot]
    total_cycles: int
    converged: bool
    convergence_reason: str
    final_summary: list[TopEntry]


def shortlist_delta(
    prev_labels: list[str], curr_labels: list[str]
) -> tuple[list[str], list[str]]:
    """Return (new_entrants, dropouts) between two shortlists of display labels."""
    prev_set, curr_set = set(prev_labels), set(curr_labels)
    return sorted(curr_set - prev_set), sorted(prev_set - curr_set)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trajectory_model.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/trajectory.py tests/test_trajectory_model.py
git commit -m "feat: trajectory capture model + shortlist-diff helper"
```

---

### Task 2: Trajectory renderers (Markdown + console)

**Files:**
- Modify: `des_multi_agent/trajectory.py` (append two functions)
- Test: `tests/test_trajectory_render.py`

**Interfaces:**
- Consumes: `TopEntry`, `CycleSnapshot`, `SearchTrajectory` from Task 1.
- Produces:
  - `format_trajectory_report(traj: SearchTrajectory) -> str` — Markdown.
  - `format_trajectory_console(traj: SearchTrajectory) -> str` — compact plain text.

Rendering rules (from spec):
- Title line `# Search Trajectory — {headline}`, then a meta line `Workflow: {workflow}  ·  Cycles run: {total_cycles}  ·  Converged: {yes (reason)|no}`.
- Per cycle heading `## Cycle {cycle} — {n_screened} screened, {n_hits} hits` with ` ✓ converged` appended when `snapshot.converged`.
- Omit the "Shortlist change" line on cycle 1 (when both `new_entrants` and `dropouts` are computed against an empty prior — represented by cycle 1 always having empty deltas by construction). Render `Shortlist change vs previous cycle: none — shortlist identical` when both lists empty AND cycle > 1; otherwise `+N entered (…), -M left (…)`.
- `Top by {metric_label}:` then a numbered list `{i}. {label} — {metric_value:.1f}  ({secondary})` (omit the `({secondary})` when secondary is empty).
- `Families reinforced: …` only when `family_ledger` is non-empty.
- A `> warnings:` block listing `notable_warnings` when present.
- Final `## Final shortlist` numbered list from `traj.final_summary` using `{label} — {metric_label} {metric_value:.1f}`.
- Empty `snapshots` → `_No cycles recorded._`; a cycle with empty `top_entries` → `_No hits this cycle._` in place of the top list.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_render.py
from des_multi_agent.trajectory import (
    CycleSnapshot,
    SearchTrajectory,
    TopEntry,
    format_trajectory_console,
    format_trajectory_report,
)


def _eg(metric=201.8, sec="Δ22.6%, high confidence"):
    return TopEntry("ethylene glycol (OCCO)", "min_tm_k", metric, sec)


def _traj():
    c1 = CycleSnapshot(
        cycle=1, n_screened=5, n_hits=5, top_entries=[_eg()],
        family_ledger={"diol": 1, "amide": 1},
    )
    c2 = CycleSnapshot(
        cycle=2, n_screened=5, n_hits=5, top_entries=[_eg()],
        new_entrants=["1,2-propanediol (CC(O)CO)"], dropouts=["water (O)"],
        family_ledger={"diol": 2},
    )
    c3 = CycleSnapshot(
        cycle=3, n_screened=5, n_hits=5, top_entries=[_eg()],
        converged=True, convergence_reason="top-5 shortlist identical to previous cycle",
    )
    return SearchTrajectory(
        workflow="des", headline="DES partners for ethanol (CCO)",
        metric_label="min Tm (K)", snapshots=[c1, c2, c3], total_cycles=3,
        converged=True, convergence_reason="top-5 shortlist identical to previous cycle",
        final_summary=[_eg()],
    )


def test_report_has_title_and_cycle_headings():
    md = format_trajectory_report(_traj())
    assert "# Search Trajectory — DES partners for ethanol (CCO)" in md
    assert "## Cycle 1 — 5 screened, 5 hits" in md
    assert "## Cycle 3 — 5 screened, 5 hits  ✓ converged" in md
    assert "Converged: yes" in md


def test_report_cycle1_omits_change_line_and_shows_families():
    md = format_trajectory_report(_traj())
    cycle1 = md.split("## Cycle 1")[1].split("## Cycle 2")[0]
    assert "Shortlist change" not in cycle1
    assert "Families reinforced: diol (2), amide (1)" not in cycle1  # cycle1 ledger order
    assert "Families reinforced:" in cycle1


def test_report_change_line_and_final_shortlist():
    md = format_trajectory_report(_traj())
    assert "+1 entered (1,2-propanediol (CC(O)CO))" in md
    assert "-1 left (water (O))" in md
    assert "## Final shortlist" in md
    assert "ethylene glycol (OCCO) — min Tm (K) 201.8" in md


def test_report_empty_snapshots():
    t = SearchTrajectory("des", "x", "min Tm (K)", [], 0, False, "", [])
    assert "_No cycles recorded._" in format_trajectory_report(t)


def test_console_one_block_per_cycle():
    txt = format_trajectory_console(_traj())
    assert "Trajectory — DES partners for ethanol (CCO)  (3 cycles, converged)" in txt
    lines = [ln for ln in txt.splitlines() if ln.strip().startswith("cycle ")]
    assert len(lines) == 3
    assert "converged" in lines[2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory_render.py -v`
Expected: FAIL with `ImportError: cannot import name 'format_trajectory_report'`

- [ ] **Step 3: Write minimal implementation**

Append to `des_multi_agent/trajectory.py`:

```python
def _converged_phrase(converged: bool, reason: str) -> str:
    if converged:
        return f"yes ({reason})" if reason else "yes"
    return "no"


def _format_top_list(snapshot: CycleSnapshot, metric_label: str) -> list[str]:
    if not snapshot.top_entries:
        return ["_No hits this cycle._"]
    out = [f"Top by {metric_label}:"]
    for i, e in enumerate(snapshot.top_entries, 1):
        tail = f"  ({e.secondary})" if e.secondary else ""
        out.append(f"{i}. {e.label} — {e.metric_value:.1f}{tail}")
    return out


def _format_change_line(snapshot: CycleSnapshot) -> str | None:
    if snapshot.cycle <= 1:
        return None
    if not snapshot.new_entrants and not snapshot.dropouts:
        return "Shortlist change vs previous cycle: none — shortlist identical"
    parts = []
    if snapshot.new_entrants:
        parts.append(f"+{len(snapshot.new_entrants)} entered ({', '.join(snapshot.new_entrants)})")
    if snapshot.dropouts:
        parts.append(f"-{len(snapshot.dropouts)} left ({', '.join(snapshot.dropouts)})")
    return "Shortlist change vs previous cycle: " + ", ".join(parts)


def format_trajectory_report(traj: SearchTrajectory) -> str:
    lines = [f"# Search Trajectory — {traj.headline}", ""]
    lines.append(
        f"Workflow: {traj.workflow}  ·  Cycles run: {traj.total_cycles}  ·  "
        f"Converged: {_converged_phrase(traj.converged, traj.convergence_reason)}"
    )
    if not traj.snapshots:
        lines += ["", "_No cycles recorded._"]
        return "\n".join(lines)
    for s in traj.snapshots:
        lines.append("")
        suffix = "  ✓ converged" if s.converged else ""
        lines.append(f"## Cycle {s.cycle} — {s.n_screened} screened, {s.n_hits} hits{suffix}")
        change = _format_change_line(s)
        if change:
            lines.append(change)
        lines += _format_top_list(s, traj.metric_label)
        if s.family_ledger:
            fam = ", ".join(f"{k} ({v})" for k, v in s.family_ledger.items())
            lines.append(f"Families reinforced: {fam}")
        if s.converged and s.convergence_reason:
            lines.append(f"Converged: {s.convergence_reason}")
        if s.notable_warnings:
            lines.append("> warnings:")
            lines += [f"> - {w}" for w in s.notable_warnings]
    lines += ["", "## Final shortlist"]
    if traj.final_summary:
        for i, e in enumerate(traj.final_summary, 1):
            lines.append(f"{i}. {e.label} — {traj.metric_label} {e.metric_value:.1f}")
    else:
        lines.append("_No final results._")
    return "\n".join(lines)


def format_trajectory_console(traj: SearchTrajectory) -> str:
    conv = "converged" if traj.converged else "ran to budget"
    out = [f"Trajectory — {traj.headline}  ({traj.total_cycles} cycles, {conv})"]
    for s in traj.snapshots:
        top = ""
        if s.top_entries:
            e = s.top_entries[0]
            top = f" · top: {e.label} {e.metric_value:.1f}"
        if s.cycle <= 1:
            change = f"{s.n_screened} screened, {s.n_hits} hits"
        elif not s.new_entrants and not s.dropouts:
            change = "stable"
        else:
            change = f"+{len(s.new_entrants)}/-{len(s.dropouts)} shortlist"
        fam = ""
        if s.family_ledger:
            fam = " · families: " + ", ".join(s.family_ledger.keys())
        conv_tag = " ✓ converged" if s.converged else ""
        out.append(f"  cycle {s.cycle}: {change}{top}{fam}{conv_tag}")
    return "\n".join(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trajectory_render.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/trajectory.py tests/test_trajectory_render.py
git commit -m "feat: trajectory Markdown + console renderers"
```

---

### Task 3: Atomic trajectory artifact writer

**Files:**
- Modify: `des_multi_agent/trajectory.py` (append writer + `os`/`Path` imports)
- Test: `tests/test_trajectory_writer.py`

**Interfaces:**
- Consumes: `SearchTrajectory`, `format_trajectory_report` from Tasks 1–2.
- Produces: `write_trajectory_artifact(output_dir: str | Path, traj: SearchTrajectory) -> Path` — writes `trajectory.md` into `output_dir` atomically; returns its path; creates `output_dir` if missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_writer.py
from pathlib import Path

from des_multi_agent.trajectory import SearchTrajectory, write_trajectory_artifact


def _traj():
    return SearchTrajectory("des", "DES partners for ethanol (CCO)", "min Tm (K)",
                            [], 0, False, "", [])


def test_writes_trajectory_md(tmp_path):
    out = write_trajectory_artifact(tmp_path, _traj())
    assert out == tmp_path / "trajectory.md"
    assert out.exists()
    assert "# Search Trajectory — DES partners for ethanol (CCO)" in out.read_text()


def test_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "run"
    out = write_trajectory_artifact(target, _traj())
    assert out.exists()
    assert out.parent == target


def test_overwrites_existing(tmp_path):
    (tmp_path / "trajectory.md").write_text("stale", encoding="utf-8")
    out = write_trajectory_artifact(tmp_path, _traj())
    assert "stale" not in out.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory_writer.py -v`
Expected: FAIL with `ImportError: cannot import name 'write_trajectory_artifact'`

- [ ] **Step 3: Write minimal implementation**

Add to the top imports of `des_multi_agent/trajectory.py`:

```python
from pathlib import Path
from tempfile import NamedTemporaryFile
```

Append the writer:

```python
def write_trajectory_artifact(output_dir: str | Path, traj: SearchTrajectory) -> Path:
    """Atomically write trajectory.md into output_dir; return its path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "trajectory.md"
    content = format_trajectory_report(traj)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=out_dir, prefix=".trajectory-", suffix=".md", delete=False
    ) as fh:
        fh.write(content)
        staged = Path(fh.name)
    staged.replace(final_path)
    return final_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trajectory_writer.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/trajectory.py tests/test_trajectory_writer.py
git commit -m "feat: atomic trajectory.md artifact writer"
```

---

### Task 4: Capture DES multi-cycle trajectory

**Files:**
- Modify: `des_multi_agent/multi_cycle.py` (imports; add `trajectory` field to `MultiCycleOutcome`; build snapshots in the loop; build `SearchTrajectory` after the loop)
- Test: `tests/test_trajectory_capture_des.py`

**Interfaces:**
- Consumes: `CycleSnapshot`, `SearchTrajectory`, `TopEntry`, `shortlist_delta` from Task 1; `display_name` from `des_multi_agent.smiles_names`; `_confidence_label` from `des_multi_agent.reporting`.
- Produces: `MultiCycleOutcome.trajectory: SearchTrajectory | None = None`, populated for every run.

Capture details (build inside the existing `for cycle in range(1, n_cycles + 1):` loop in `run_multi_cycle_search`, after `family_ledger`, `top_k`, `new_entrants`, `dropouts`, `converged` are computed — currently around lines 105–129):
- Build an ordered list of the cycle's top DES hits: take `outcome.results`, keep `r.is_des`, in existing order, first `top_k_convergence`.
- For each, build a `TopEntry`: `label=display_name(r.curve.smiles_b)`, `metric_name="min_tm_k"`, `metric_value=r.min_tm_k`, `secondary` = relative-drop % + confidence (see helper below).
- Build display-label lists for the current and previous top hits; derive `new_entrants`/`dropouts` via `shortlist_delta` (replacing the SMILES-based deltas for the snapshot only — `CycleDelta` keeps its SMILES deltas).
- `notable_warnings`: first 3 entries of `outcome.llm_warnings` that start with `[GROUNDING]` or `[REALITY]`.
- Wrap the snapshot build in `try/except` per Global Constraints.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_capture_des.py
from dataclasses import dataclass

from des_multi_agent import multi_cycle
from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty


def _curve(smi_b, tm):
    return CurvePrediction(
        smiles_a="CCO", smiles_b=smi_b, ratios=[0.5], tm_pred_k=[tm],
        t1_k=271.0, t2_k=300.0, checkpoint_path="ckpt.pt",
    )


def _result(smi_b, tm):
    return DesResult(curve=_curve(smi_b, tm), absolute_pass=True, relative_pass=True,
                     is_des=True, rationale="ok", min_tm_k=tm)


def _annotated(res):
    unc = MinimumTmUncertainty(
        component_a="CCO", component_b=res.curve.smiles_b, repeated_values=(),
        mean_tm_k=res.min_tm_k, std_tm_k=0.5, min_tm_k=res.min_tm_k, max_tm_k=res.min_tm_k,
        trust_score=0.85, uncertainty_flag="low", explanation="", checkpoint_path="ckpt.pt",
        config_path="x",
    )
    return AnnotatedResult(result=res, uncertainty=unc, trust_score=0.85, ranking_score=1.0)


@dataclass
class _FakeOutcome:
    results: list
    annotated_results: list
    brainstorm_candidates: list
    llm_warnings: list
    chemical_pattern_memory: object = None
    chemistry_lesson_summary: object = None


def _make_fake(cycle_results):
    """Return a fake run_search_report producing canned cycles in sequence."""
    seq = iter(cycle_results)

    def fake(**kwargs):
        results = next(seq)
        brainstorm = [CandidateBrainstorm(smiles=r.curve.smiles_b, rationale="x", family="diol")
                      for r in results]
        return _FakeOutcome(
            results=results,
            annotated_results=[_annotated(r) for r in results],
            brainstorm_candidates=brainstorm,
            llm_warnings=[],
        )

    return fake


def test_des_trajectory_captured(monkeypatch):
    cycle1 = [_result("OCCO", 201.8), _result("O", 225.0)]
    cycle2 = [_result("OCCO", 201.8), _result("OCC(O)CO", 221.0)]
    monkeypatch.setattr(multi_cycle, "run_search_report", _make_fake([cycle1, cycle2]))

    outcome = multi_cycle.run_multi_cycle_search(
        component_a="CCO", n=2, checkpoint_path="ckpt.pt", n_cycles=2, top_k_convergence=5,
    )

    traj = outcome.trajectory
    assert traj is not None
    assert traj.workflow == "des"
    assert traj.metric_label == "min Tm (K)"
    assert len(traj.snapshots) == 2
    # cycle 2 dropped water, gained glycerol (display names resolved)
    s2 = traj.snapshots[1]
    assert any("glycerol" in lbl or "OCC(O)CO" in lbl for lbl in s2.new_entrants)
    assert any("water" in lbl or lbl.endswith("(O)") for lbl in s2.dropouts)
    # final summary present, family ledger non-empty
    assert traj.final_summary
    assert traj.snapshots[0].family_ledger.get("diol") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory_capture_des.py -v`
Expected: FAIL with `AttributeError: 'MultiCycleOutcome' object has no attribute 'trajectory'`

- [ ] **Step 3: Write minimal implementation**

In `des_multi_agent/multi_cycle.py`, add imports near the top (after existing imports):

```python
from .reporting import _confidence_label
from .smiles_names import display_name
from .trajectory import CycleSnapshot, SearchTrajectory, TopEntry, shortlist_delta
```

Add the field to `MultiCycleOutcome` (after `accumulated_family_ledger`):

```python
    trajectory: object = None   # SearchTrajectory | None
```

Add a module-level helper (before `run_multi_cycle_search`):

```python
def _des_top_entries(outcome, top_k: int) -> tuple[list[TopEntry], list[str]]:
    """Build best-first TopEntry list + display labels for a DES cycle outcome."""
    ann_by_smiles = {a.result.curve.smiles_b: a for a in getattr(outcome, "annotated_results", [])}
    entries: list[TopEntry] = []
    labels: list[str] = []
    hits = [r for r in outcome.results if r.is_des][:top_k]
    for r in hits:
        label = display_name(r.curve.smiles_b)
        t1, t2 = getattr(r.curve, "t1_k", None), getattr(r.curve, "t2_k", None)
        secondary = ""
        if t1 is not None and t2 is not None:
            baseline = min(t1, t2)
            if baseline:
                secondary = f"Δ{(baseline - r.min_tm_k) / baseline * 100:.1f}%"
        ann = ann_by_smiles.get(r.curve.smiles_b)
        if ann is not None:
            conf = _confidence_label(ann.trust_score, ann.uncertainty.uncertainty_flag)
            secondary = f"{secondary}, {conf}" if secondary else conf
        entries.append(TopEntry(label, "min_tm_k", r.min_tm_k, secondary))
        labels.append(label)
    return entries, labels
```

Inside `run_multi_cycle_search`, initialize a snapshot list and previous-label tracker before the loop (next to `prev_top: frozenset = frozenset()`):

```python
    snapshots: list[CycleSnapshot] = []
    prev_labels: list[str] = []
```

Inside the loop, after `cycle_deltas.append(CycleDelta(...))` (around line 129), add:

```python
        try:
            top_entries, curr_labels = _des_top_entries(outcome, top_k_convergence)
            snap_new, snap_drop = shortlist_delta(prev_labels, curr_labels)
            notable = [w for w in outcome.llm_warnings
                       if w.startswith("[GROUNDING]") or w.startswith("[REALITY]")][:3]
            reason = "top-{0} shortlist identical to previous cycle".format(top_k_convergence) if converged else ""
            snapshots.append(CycleSnapshot(
                cycle=cycle,
                n_screened=len(outcome.results),
                n_hits=sum(1 for r in outcome.results if r.is_des),
                top_entries=top_entries,
                new_entrants=snap_new if cycle > 1 else [],
                dropouts=snap_drop if cycle > 1 else [],
                family_ledger=dict(family_ledger),
                converged=converged,
                convergence_reason=reason,
                notable_warnings=notable,
            ))
            prev_labels = curr_labels
        except Exception as exc:  # capture is best-effort
            outcome.llm_warnings.append(f"[TRAJECTORY] snapshot capture failed (cycle {cycle}): {exc}")
```

Replace the `return MultiCycleOutcome(...)` at the end with one that also builds and passes `trajectory`:

```python
    final_converged = cycle_deltas[-1].converged if cycle_deltas else False
    trajectory = SearchTrajectory(
        workflow="des",
        headline=f"DES partners for {display_name(component_a)}",
        metric_label="min Tm (K)",
        snapshots=snapshots,
        total_cycles=len(cycle_deltas),
        converged=final_converged,
        convergence_reason=(snapshots[-1].convergence_reason if snapshots else ""),
        final_summary=(snapshots[-1].top_entries if snapshots else []),
    )
    return MultiCycleOutcome(
        final_outcome=last_outcome,
        cycle_deltas=cycle_deltas,
        total_cycles=len(cycle_deltas),
        converged=final_converged,
        accumulated_family_ledger=dict(accumulated_ledger),
        trajectory=trajectory,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trajectory_capture_des.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the multi_cycle regression tests**

Run: `python -m pytest tests/ -q -k "multi_cycle or trajectory"`
Expected: PASS (all existing multi-cycle tests still green)

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/multi_cycle.py tests/test_trajectory_capture_des.py
git commit -m "feat: capture DES multi-cycle trajectory snapshots"
```

---

### Task 5: Capture metal-selectivity trajectory

**Files:**
- Modify: `des_multi_agent/workflows/metal_binding_selectivity.py` (imports; add `trajectory` field to `SelectivityScreenOutcome`; build snapshots in the loop; build `SearchTrajectory` at return)
- Test: `tests/test_trajectory_capture_metal_selectivity.py`

**Interfaces:**
- Consumes: `CycleSnapshot`, `SearchTrajectory`, `TopEntry`, `shortlist_delta` from Task 1.
- Produces: `SelectivityScreenOutcome.trajectory: SearchTrajectory | None = None`.

Capture details (inside `run_metal_selectivity_screen`'s `for cycle in range(1, n_cycles + 1):` loop, after `cumulative_results` is sorted — around line 296, and after the convergence check sets the break):
- `top_entries`: first 5 of `cumulative_results`, `metric_name="composite_score"`, `metric_value=r.composite_score`, `label=r.ligand_smiles`, `secondary=f"ΔlogK={r.delta_log_k:.2f}, logK({target_metal})={r.log_k_target:.2f}"`.
- `n_hits`: count of `cumulative_results` with `delta_log_k > 0`.
- labels for delta: top-5 `ligand_smiles`.
- `converged`: True only on the cycle where `_top_k_stable(prev_cycle_results, cumulative_results)` fires (cycle > 1); reason `"top-5 ligand set stable vs previous cycle"`.
- `family_ledger`: `{}`.
- Wrap in `try/except`, appending to `all_warnings` on failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_capture_metal_selectivity.py
from des_multi_agent.workflows import metal_binding_selectivity as mbs
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    run_metal_selectivity_screen,
)
from des_multi_agent.schemas import CandidateProposal


def _proposal(smi):
    return CandidateProposal(smiles=smi, rationale="x", family="amine", source="heuristic", source_id="")


def test_metal_selectivity_trajectory_captured(monkeypatch):
    # Two cycles, deterministic proposals + scoring, no LLM.
    monkeypatch.setattr(mbs, "generate_ligand_candidates",
                        lambda metal, n, constraints: [_proposal("NCCN"), _proposal("NCCO")])

    scores = {"NCCN": (8.0, 2.0), "NCCO": (6.0, 1.0)}  # (log_k_target, delta)

    def fake_score(target, competitor, proposal, model_path, w_a, w_s, stability_rule_weight=0.0):
        lk, delta = scores[proposal.smiles]
        return SelectivityResult(
            ligand_smiles=proposal.smiles, log_k_target=lk, log_k_competitor=lk - delta,
            delta_log_k=delta, composite_score=lk + delta, source=proposal.source, source_id="",
            rationale="ok",
        ), []

    monkeypatch.setattr(mbs, "_score_proposal_pair", fake_score)

    outcome = run_metal_selectivity_screen(
        target_metal="Cu2+", competitor_metal="Zn2+", n=2, model_path=None,
        llm_provider=None, n_cycles=2,
    )

    traj = outcome.trajectory
    assert traj is not None
    assert traj.workflow == "metal-selectivity"
    assert traj.metric_label == "composite score"
    assert len(traj.snapshots) >= 1
    top = traj.snapshots[0].top_entries[0]
    assert top.metric_name == "composite_score"
    assert top.label == "NCCN"  # highest composite score
```

NOTE: confirm `SelectivityResult`'s constructor field order/names against
`des_multi_agent/workflows/metal_binding_selectivity.py:16-30` before running;
adjust the `SelectivityResult(...)` kwargs in the test if a field name differs.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory_capture_metal_selectivity.py -v`
Expected: FAIL with `AttributeError: 'SelectivityScreenOutcome' object has no attribute 'trajectory'`

- [ ] **Step 3: Write minimal implementation**

Add imports near the top of `metal_binding_selectivity.py`:

```python
from ..trajectory import CycleSnapshot, SearchTrajectory, TopEntry, shortlist_delta
```

Add the field to `SelectivityScreenOutcome` (after `claim_verdicts`):

```python
    trajectory: object = None   # SearchTrajectory | None
```

Before the loop (next to `prev_cycle_results: list[SelectivityResult] = []`):

```python
    sel_snapshots: list[CycleSnapshot] = []
    sel_prev_labels: list[str] = []
    sel_converged = False
```

Inside the loop, replace the existing convergence block:

```python
        if cycle > 1 and _top_k_stable(prev_cycle_results, cumulative_results):
            print(
                f"[cycle {cycle}/{n_cycles}] top-5 stable — converged early",
                file=sys.stderr,
                flush=True,
            )
            break
```

with one that records the snapshot first (so the converged cycle is captured before `break`):

```python
        this_converged = cycle > 1 and _top_k_stable(prev_cycle_results, cumulative_results)
        try:
            top5 = cumulative_results[:5]
            entries = [
                TopEntry(
                    label=r.ligand_smiles, metric_name="composite_score",
                    metric_value=r.composite_score,
                    secondary=f"ΔlogK={r.delta_log_k:.2f}, logK({target_metal})={r.log_k_target:.2f}",
                )
                for r in top5
            ]
            curr_labels = [r.ligand_smiles for r in top5]
            snap_new, snap_drop = shortlist_delta(sel_prev_labels, curr_labels)
            sel_snapshots.append(CycleSnapshot(
                cycle=cycle,
                n_screened=len(proposals),
                n_hits=sum(1 for r in cumulative_results if r.delta_log_k > 0),
                top_entries=entries,
                new_entrants=snap_new if cycle > 1 else [],
                dropouts=snap_drop if cycle > 1 else [],
                converged=this_converged,
                convergence_reason="top-5 ligand set stable vs previous cycle" if this_converged else "",
            ))
            sel_prev_labels = curr_labels
        except Exception as exc:
            all_warnings.append(f"[TRAJECTORY] snapshot capture failed (cycle {cycle}): {exc}")

        if this_converged:
            sel_converged = True
            print(
                f"[cycle {cycle}/{n_cycles}] top-5 stable — converged early",
                file=sys.stderr,
                flush=True,
            )
            break
```

Before `return SelectivityScreenOutcome(...)`, build the trajectory:

```python
    sel_trajectory = SearchTrajectory(
        workflow="metal-selectivity",
        headline=f"{target_metal} over {competitor_metal} selectivity",
        metric_label="composite score",
        snapshots=sel_snapshots,
        total_cycles=len(sel_snapshots),
        converged=sel_converged,
        convergence_reason=(sel_snapshots[-1].convergence_reason if sel_snapshots else ""),
        final_summary=(sel_snapshots[-1].top_entries if sel_snapshots else []),
    )
```

and pass `trajectory=sel_trajectory` in the `SelectivityScreenOutcome(...)` constructor.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trajectory_capture_metal_selectivity.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run metal-selectivity regression tests**

Run: `python -m pytest tests/ -q -k "selectivity"`
Expected: PASS (existing selectivity tests still green)

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/workflows/metal_binding_selectivity.py tests/test_trajectory_capture_metal_selectivity.py
git commit -m "feat: capture metal-selectivity trajectory snapshots"
```

---

### Task 6: Capture selectivity-DES pipeline trajectory

**Files:**
- Modify: `des_multi_agent/workflows/selectivity_des_pipeline.py` (imports; add `trajectory` field to `SelectivityDesPipelineOutcome`; build outer-cycle snapshots; build `SearchTrajectory` at return)
- Test: `tests/test_trajectory_capture_pipeline.py`

**Interfaces:**
- Consumes: `CycleSnapshot`, `SearchTrajectory`, `TopEntry`, `shortlist_delta` from Task 1.
- Produces: `SelectivityDesPipelineOutcome.trajectory: SearchTrajectory | None = None`.

Capture details (one snapshot per **outer** cycle, built after `ligand_des_results` for that outer cycle is complete — around line 173, before the convergence check at line 177):
- `top_entries`: the DES-compatible ligands this outer cycle, best-first by `ldr.ligand.composite_score`, `metric_name="composite_score"`, `metric_value=ldr.ligand.composite_score`, `label=ldr.ligand.ligand_smiles`, `secondary=f"ΔlogK={ldr.ligand.delta_log_k:.2f}, DES-compatible"`.
- `n_hits`: number of DES-compatible ligands (`len(new_compatible)`).
- `n_screened`: `len(shortlisted)`.
- labels for delta: sorted `new_compatible` set.
- `converged`: True only on the outer cycle where `new_compatible == prev_compatible` (outer_cycle > 1); reason `"DES-compatible ligand set stable vs previous outer cycle"`.
- Wrap in `try/except`, appending to `all_warnings`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_capture_pipeline.py
from des_multi_agent.workflows import selectivity_des_pipeline as pipe


def test_pipeline_trajectory_field_present_and_typed(monkeypatch):
    # Stub the two inner phases so the outer loop runs deterministically with no ML/LLM.
    from des_multi_agent.workflows.metal_binding_selectivity import (
        SelectivityResult,
        SelectivityScreenOutcome,
    )

    lig = SelectivityResult(
        ligand_smiles="NCCN", log_k_target=8.0, log_k_competitor=6.0, delta_log_k=2.0,
        composite_score=10.0, source="heuristic", source_id="", rationale="ok",
    )

    monkeypatch.setattr(pipe, "run_metal_selectivity_screen", lambda **kw: SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+", results=[lig], n_screened=1, n_cycles=1,
    ))
    monkeypatch.setattr(pipe, "_bridge_filter", lambda results, mindelta, topn, warnings: [lig])

    class _MCO:
        cycle_deltas = []
        class final_outcome:  # noqa: N801
            results = []

    monkeypatch.setattr(pipe, "run_multi_cycle_search", lambda **kw: _MCO())

    outcome = pipe.run_selectivity_des_pipeline(
        target_metal="Cu2+", competitor_metal="Zn2+", checkpoint_path="ckpt.pt",
        n_ligands=1, n_des_candidates=1, n_outer_cycles=1,
    )

    assert outcome.trajectory is not None
    assert outcome.trajectory.workflow == "selectivity-des"
    assert outcome.trajectory.metric_label == "composite score"
    assert len(outcome.trajectory.snapshots) == 1
```

NOTE: confirm the kwarg names of `run_selectivity_des_pipeline` and the stubbed
helpers against `selectivity_des_pipeline.py` before running; adjust if a name
differs.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory_capture_pipeline.py -v`
Expected: FAIL with `AttributeError: 'SelectivityDesPipelineOutcome' object has no attribute 'trajectory'`

- [ ] **Step 3: Write minimal implementation**

Add imports near the top of `selectivity_des_pipeline.py`:

```python
from ..trajectory import CycleSnapshot, SearchTrajectory, TopEntry, shortlist_delta
```

Add the field to `SelectivityDesPipelineOutcome` (after `warnings`):

```python
    trajectory: object = None   # SearchTrajectory | None
```

Before the outer loop (next to `converged = False`):

```python
    pipe_snapshots: list[CycleSnapshot] = []
    pipe_prev_labels: list[str] = []
```

Inside the outer loop, after `final_ligand_des_results = ligand_des_results` and the
two `des_*_smiles` assignments (around line 175), before the convergence check:

```python
        this_converged = outer_cycle > 1 and new_compatible == prev_compatible
        try:
            compatible_ldrs = sorted(
                (ldr for ldr in ligand_des_results if ldr.des_compatible),
                key=lambda ldr: ldr.ligand.composite_score, reverse=True,
            )
            entries = [
                TopEntry(
                    label=ldr.ligand.ligand_smiles, metric_name="composite_score",
                    metric_value=ldr.ligand.composite_score,
                    secondary=f"ΔlogK={ldr.ligand.delta_log_k:.2f}, DES-compatible",
                )
                for ldr in compatible_ldrs
            ]
            curr_labels = sorted(new_compatible)
            snap_new, snap_drop = shortlist_delta(pipe_prev_labels, curr_labels)
            pipe_snapshots.append(CycleSnapshot(
                cycle=outer_cycle,
                n_screened=len(shortlisted),
                n_hits=len(new_compatible),
                top_entries=entries,
                new_entrants=snap_new if outer_cycle > 1 else [],
                dropouts=snap_drop if outer_cycle > 1 else [],
                converged=this_converged,
                convergence_reason="DES-compatible ligand set stable vs previous outer cycle" if this_converged else "",
            ))
            pipe_prev_labels = curr_labels
        except Exception as exc:
            all_warnings.append(f"[TRAJECTORY] snapshot capture failed (outer {outer_cycle}): {exc}")
```

(The existing `if outer_cycle > 1 and new_compatible == prev_compatible:` block that
sets `converged = True` and breaks stays as-is, immediately after this capture block.)

Before `return SelectivityDesPipelineOutcome(...)`, build the trajectory:

```python
    pipe_trajectory = SearchTrajectory(
        workflow="selectivity-des",
        headline=f"{target_metal} over {competitor_metal} — selectivity-DES",
        metric_label="composite score",
        snapshots=pipe_snapshots,
        total_cycles=len(pipe_snapshots),
        converged=converged,
        convergence_reason=(pipe_snapshots[-1].convergence_reason if pipe_snapshots else ""),
        final_summary=(pipe_snapshots[-1].top_entries if pipe_snapshots else []),
    )
```

and pass `trajectory=pipe_trajectory` in the `SelectivityDesPipelineOutcome(...)` constructor.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trajectory_capture_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run pipeline regression tests**

Run: `python -m pytest tests/ -q -k "pipeline or selectivity_des"`
Expected: PASS (existing pipeline tests still green)

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/workflows/selectivity_des_pipeline.py tests/test_trajectory_capture_pipeline.py
git commit -m "feat: capture selectivity-DES pipeline trajectory snapshots"
```

---

### Task 7: CLI wiring — print console trajectory + write trajectory.md

**Files:**
- Modify: `des_multi_agent/cli.py` (DES multi-cycle branch ~lines 599–630; metal-selectivity branch ~lines 734–753; selectivity-des branch ~lines 704–733)
- Test: `tests/test_cli_trajectory_artifact.py`

**Interfaces:**
- Consumes: `format_trajectory_console`, `write_trajectory_artifact` from Tasks 2–3; `outcome.trajectory` / `multi_outcome.trajectory` / `pipeline_outcome.trajectory` / `sel_outcome.trajectory` from Tasks 4–6.
- Produces: console trajectory on stdout; `trajectory.md` in `--output-dir` when set.

Behavior:
- Add a shared helper in `cli.py`: `_emit_trajectory(traj, output_dir)` that prints `format_trajectory_console(traj)` to **stderr** when `traj is not None` (stderr keeps stdout clean for `--format json`/`csv` machine consumers, matching the existing per-cycle convention), and writes the artifact when `output_dir` is set.
- DES multi-cycle: replace the existing per-cycle stderr loop (lines 621–630) with a single `_emit_trajectory(multi_outcome.trajectory, args.output_dir)` call placed after `outcome = multi_outcome.final_outcome` and after the final report is printed.
- metal-selectivity branch: after `print(format_metal_selectivity_report(sel_outcome))`, call `_emit_trajectory(getattr(sel_outcome, "trajectory", None), getattr(args, "output_dir", None))`.
- selectivity-des branch: after `print(format_selectivity_des_report(pipeline_outcome))`, call `_emit_trajectory(getattr(pipeline_outcome, "trajectory", None), getattr(args, "output_dir", None))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_trajectory_artifact.py
import subprocess
import sys
from pathlib import Path


def test_des_multicycle_writes_trajectory_md(tmp_path):
    out_dir = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "des_multi_agent.cli",
         "--workflow", "des", "--component-a", "ethanol", "--n", "5",
         "--checkpoint-path", "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
         "--config-path", "ml_des_mp/config.yaml",
         "--n-cycles", "2", "--output-dir", str(out_dir)],
        capture_output=True, text=True, env={"DES_DISABLE_QSPR": "1", "PATH": __import__("os").environ["PATH"]},
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    traj = out_dir / "trajectory.md"
    assert traj.exists()
    text = traj.read_text()
    assert "# Search Trajectory — DES partners for ethanol (CCO)" in text
    assert "## Cycle 1" in text
    # console trajectory printed to stderr (keeps stdout clean for json/csv)
    assert "Trajectory — DES partners for ethanol (CCO)" in proc.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_trajectory_artifact.py -v`
Expected: FAIL (no `trajectory.md` written / string absent from stdout)

- [ ] **Step 3: Write minimal implementation**

Add an import near the other reporting imports in `cli.py`:

```python
from .trajectory import format_trajectory_console, write_trajectory_artifact
```

Add the helper (module level, near other private CLI helpers):

```python
def _emit_trajectory(traj, output_dir) -> None:
    if traj is None:
        return
    print(format_trajectory_console(traj), file=sys.stderr)
    if output_dir:
        try:
            write_trajectory_artifact(output_dir, traj)
        except OSError as exc:
            print(f"[WARNING] failed to write trajectory.md: {exc}", file=sys.stderr)
```

In the DES multi-cycle branch, delete the loop at lines 621–630:

```python
                for delta in multi_outcome.cycle_deltas:
                    new = f"+{len(delta.new_entrants)}" if delta.new_entrants else "0"
                    out = f"-{len(delta.dropouts)}" if delta.dropouts else "0"
                    print(
                        f"[cycle {delta.cycle}/{multi_outcome.total_cycles}] "
                        f"screened={delta.n_screened} des={delta.n_des} "
                        f"top-K changes: {new} new, {out} dropped"
                        + (" — CONVERGED" if delta.converged else ""),
                        file=sys.stderr,
                    )
```

Then, after the final report is printed in the `format == "table"`/etc. block
(after the `print(format_report(...))` call near line 665+, at the end of that
format dispatch), add:

```python
        if getattr(args, "n_cycles", 1) > 1:
            _emit_trajectory(getattr(multi_outcome, "trajectory", None), args.output_dir)
```

NOTE: `multi_outcome` is only defined when `n_cycles > 1`; guard exactly as
shown so single-cycle runs never reference it.

In the metal-selectivity branch, after `print(format_metal_selectivity_report(sel_outcome))`:

```python
        _emit_trajectory(getattr(sel_outcome, "trajectory", None), getattr(args, "output_dir", None))
```

In the selectivity-des branch, after `print(format_selectivity_des_report(pipeline_outcome))`:

```python
        _emit_trajectory(getattr(pipeline_outcome, "trajectory", None), getattr(args, "output_dir", None))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_trajectory_artifact.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (all green; no regressions)

- [ ] **Step 6: Commit**

```bash
git add des_multi_agent/cli.py tests/test_cli_trajectory_artifact.py
git commit -m "feat: emit console trajectory + write trajectory.md from CLI"
```

---

### Task 8: Docs + example refresh

**Files:**
- Modify: `docs/tutorial.md` (add a short "Iteration trajectory" subsection under the multi-cycle/iterative section)
- Modify: `examples/README.md` (note the new `trajectory.md` artifact for multi-cycle examples)
- Modify: `examples/multi_cycle_des/run.sh` (add `--output-dir` so the example emits `trajectory.md`) and `examples/multi_cycle_des/README.md` (mention it)
- Create: `examples/multi_cycle_des/trajectory.md` (the captured artifact)

**Interfaces:**
- Consumes: the finished feature (Tasks 1–7).
- Produces: documentation + one captured example artifact — no library code.

Note on channels: the console trajectory prints to **stderr**, and
`examples/multi_cycle_des/run.sh` redirects stdout to `output.txt` while
discarding stderr — so `output.txt` is unchanged by this feature. The durable,
committable evidence is `trajectory.md`, produced by adding `--output-dir`.

- [ ] **Step 1: Point the example at an output dir and re-run**

Edit `examples/multi_cycle_des/run.sh` to add `--output-dir "${SCRIPT_DIR}"` to
the `python -m des_multi_agent.cli` invocation (keeping the existing
`> "${SCRIPT_DIR}/output.txt" 2>/dev/null` redirect). Then:

Run: `bash examples/multi_cycle_des/run.sh`
Then inspect: `git status examples/multi_cycle_des/`
Expected: a new top-level `trajectory.md`, plus per-cycle `cycle_01/`, `cycle_02/`
subdirectories (each holding that cycle's `report.txt`/`run.json`/`run.csv`/
`run.manifest.json` bundle); `output.txt` is unchanged. Confirm `trajectory.md`
contains `# Search Trajectory — DES partners for ethanol (CCO)` and `## Cycle 1`.
If committing the per-cycle bundle dirs is undesirable noise, add them to the
example's `.gitignore` and commit only `trajectory.md` + `run.sh` + `README.md`.

- [ ] **Step 2: Add tutorial subsection**

Add to `docs/tutorial.md`, in the multi-cycle/iterative workflow section, a subsection with this content:

```markdown
### Iteration trajectory

Any run with `--n-cycles > 1` (and the metal-selectivity / selectivity-DES
workflows) now prints a compact per-cycle **trajectory** to stdout: how many
candidates were screened and hit each cycle, which entered or left the
shortlist, which chemical families were reinforced, and whether the search
converged. Pass `--output-dir DIR` to also write a durable, readable
`DIR/trajectory.md` with the full cycle-by-cycle narrative and the final
shortlist.
```

- [ ] **Step 3: Note the artifact in examples/README.md**

Add a line to `examples/README.md` describing that multi-cycle runs emit
`trajectory.md` alongside the existing `report.txt`/`run.json`/`run.csv`
when `--output-dir` is set.

- [ ] **Step 4: Verify docs render and examples are consistent**

Run: `python -m pytest tests/ -q -k "example or baseline"`
Expected: PASS (any example-baseline tests reflect the re-captured output; update
the corresponding baseline fixture if a test compares `output.txt` and now fails,
committing the re-captured file).

- [ ] **Step 5: Commit**

```bash
git add docs/tutorial.md examples/README.md examples/multi_cycle_des/
git commit -m "docs: document iteration trajectory; refresh multi-cycle example"
```

---

## Notes for the implementer

- Before editing each workflow, open the file and confirm the line anchors in
  this plan still match (the repo may have shifted); the surrounding code shown
  in each task is the reliable anchor, not the line numbers.
- `SearchOutcome` (DES per-cycle outcome) has no `trajectory` field and does not
  need one — only the three *multi-cycle* outcome objects gain it.
- Keep `CycleDelta` and `cycle_deltas` exactly as they are; the snapshot is a
  separate, render-oriented record. Do not try to unify them.
- The metric value is always formatted with `:.1f` in the report and console;
  for `composite_score` that is intentional (one decimal is enough for a ranking
  score). Do not change the format string per workflow.
