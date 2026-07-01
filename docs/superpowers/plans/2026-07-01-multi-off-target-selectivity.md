# Multi-Off-Target Metal Selectivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend metal-selectivity screening from "one target vs one competitor" to "one target vs N off-targets," ranking ligands by worst-case selectivity (must beat every off-target, not just one), while keeping the N=1 case byte-identical everywhere.

**Architecture:** `run_metal_selectivity_screen`'s `competitor_metal` parameter widens to accept `str | list[str]` (never renamed — a bare string still means "one off-target," zero existing call sites need to change). `SelectivityResult` gains a full per-metal breakdown and a `worst_competitor_metal` field; `delta_log_k`/`log_k_competitor` keep their existing names but now always hold the worst-case value. Every downstream consumer (DFT tiebreak, selectivity grounding, LLM prompts, CLI, reporting) is updated to either pass through the list or use each candidate's own `worst_competitor_metal` — never a single outcome-wide value.

**Tech Stack:** Python 3.11+, existing DES-Agent workflow/LLM/reporting layers. No new dependencies.

## Global Constraints

- `run_metal_selectivity_screen`'s (and `run_selectivity_des_pipeline`'s) `competitor_metal` parameter is NEVER renamed — only its type widens to `str | list[str]`. A bare string normalizes to a one-element list internally. This is true for every function in the call chain (`_build_selectivity_context`, `ligand_selectivity_brainstorm_prompt`, `dft_nomination_prompt`) — same rule, no exceptions.
- N=1 (today's usage) produces byte-identical report output, CLI behavior, DFT tiebreak behavior, and LLM prompt/context text in every layer.
- Ranking is always worst-case: `delta_log_k = log_k_target − max(log_k of all off-targets)`.
- A candidate is dropped entirely (not partially scored) if any single off-target's `predict_log_k` call fails.
- `competitor_metal=[]` (empty list) raises `ValueError` immediately.
- Duplicate off-targets in the input are de-duplicated, order-preserving, no error.
- `dft_selectivity_adjustment` and `ground_selectivity`/`verify_selectivity_claim` signatures are UNCHANGED — the workflow always passes a single string (each candidate's own `worst_competitor_metal`) into them, exactly as it does today for the sole competitor.
- `SelectivityScreenOutcome.competitor_metal: str` renames to `competitor_metals: list[str]` (this is an Outcome/result object, not a function parameter — always a list, no bare-string backward compat needed at this layer). Same rename applies to `SelectivityDesPipelineOutcome`.
- Full existing suite (927 tests as of commit `6f84e81`) continues to pass; the N=1 code path is exercised by nearly every existing metal-selectivity test today, so it is the primary regression net.

---

## Task 1: Core scoring engine + LLM prompt layer

**Why this is one task, not several:** `run_metal_selectivity_screen`'s local `competitor_metal` variable is read by its LLM-brainstorm call site, its DFT-nomination call site, and its context-builder call sites — all in the same function body. If the LLM prompt functions aren't updated in the same commit, passing a list into them produces broken/ugly rendered text (`"over ['Zn2+']."` instead of `"over Zn2+."`), which would fail the N=1 byte-identical constraint the moment the full test suite runs at the end of this task. These must land together.

**Files:**
- Modify: `des_multi_agent/workflows/metal_binding_selectivity.py`
- Modify: `des_multi_agent/llm/prompts.py` (two functions: `ligand_selectivity_brainstorm_prompt`, `dft_nomination_prompt`)
- Modify: `des_multi_agent/llm/base.py` (two methods: `brainstorm_ligands_selectivity`, `nominate_for_dft` — type-hint only, no body change)
- Modify (mechanical rename, 5 files): `tests/test_dft_integration.py`, `tests/test_metal_selectivity_screen.py`, `tests/test_dft_cli_report.py`, `tests/test_trajectory_capture_pipeline.py`, `tests/test_selectivity_des_pipeline.py`
- Test: `tests/test_metal_selectivity_screen.py` (new tests appended)

**Interfaces:**
- Consumes: `predict_log_k(metal, smiles, model_path=..., allow_fallback=True)` (existing, unchanged), `rule_based_log_k(metal_ion, smiles)` (existing, unchanged), `dft_selectivity_adjustment(dft_result, target_metal, competitor_metal)` (existing, unchanged signature), `ground_selectivity`/`verify_selectivity_claim` via `_ground_sel` (existing, unchanged signature).
- Produces: `SelectivityResult.log_k_competitors: dict[str, float]`, `SelectivityResult.worst_competitor_metal: str`, `SelectivityScreenOutcome.competitor_metals: list[str]` — Task 2 (`selectivity_des_pipeline.py`) and Task 4 (`reporting.py`) consume these.

### Step 1: Write the failing tests

Append to `tests/test_metal_selectivity_screen.py` (add these imports at the top alongside the existing ones: `from unittest.mock import patch` if not already imported, and `_score_proposal_pair`, `_normalize_competitor_metals`, `_build_selectivity_context` to the existing `from des_multi_agent.workflows.metal_binding_selectivity import (...)` block):

```python
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
    _build_selectivity_context,
    _normalize_competitor_metals,
    _score_proposal_pair,
    _top_k_stable,
    run_metal_selectivity_screen,
)
from des_multi_agent.schemas import CandidateProposal
```

```python
# ---------------------------------------------------------------------------
# _normalize_competitor_metals
# ---------------------------------------------------------------------------

def test_normalize_competitor_metals_string_wraps_in_list():
    assert _normalize_competitor_metals("Zn2+") == ["Zn2+"]


def test_normalize_competitor_metals_dedups_preserving_order():
    assert _normalize_competitor_metals(["Zn2+", "Fe3+", "Zn2+"]) == ["Zn2+", "Fe3+"]


def test_normalize_competitor_metals_empty_raises_value_error():
    with pytest.raises(ValueError, match="at least one metal ion"):
        _normalize_competitor_metals([])


# ---------------------------------------------------------------------------
# _score_proposal_pair — multi-off-target scoring
# ---------------------------------------------------------------------------

def _fake_predict(metal_values: dict) -> callable:
    def _predict(metal, smiles, model_path=None, allow_fallback=True):
        return MagicMock(value=metal_values[metal])
    return _predict


def test_score_proposal_pair_single_competitor_matches_old_behavior():
    proposal = CandidateProposal(smiles="NCCN", rationale="r", family="f",
                                  source="heuristic", source_id="s")
    with patch(
        "des_multi_agent.workflows.metal_binding_selectivity.predict_log_k",
        side_effect=_fake_predict({"Cu2+": 10.0, "Zn2+": 6.0}),
    ):
        result, warnings = _score_proposal_pair(
            "Cu2+", ["Zn2+"], proposal, None, w_affinity=0.5, w_selectivity=0.5,
        )
    assert warnings == []
    assert result.log_k_competitors == {"Zn2+": 6.0}
    assert result.worst_competitor_metal == "Zn2+"
    assert result.log_k_competitor == 6.0
    assert abs(result.delta_log_k - 4.0) < 1e-9


def test_score_proposal_pair_worst_case_among_multiple_off_targets():
    proposal = CandidateProposal(smiles="NCCN", rationale="r", family="f",
                                  source="heuristic", source_id="s")
    with patch(
        "des_multi_agent.workflows.metal_binding_selectivity.predict_log_k",
        side_effect=_fake_predict({"Cu2+": 10.0, "Zn2+": 6.0, "Fe3+": 9.0}),
    ):
        result, warnings = _score_proposal_pair(
            "Cu2+", ["Zn2+", "Fe3+"], proposal, None, w_affinity=0.5, w_selectivity=0.5,
        )
    assert result.log_k_competitors == {"Zn2+": 6.0, "Fe3+": 9.0}
    assert result.worst_competitor_metal == "Fe3+"      # Fe3+ has the higher log K -> it's the bottleneck
    assert result.log_k_competitor == 9.0
    assert abs(result.delta_log_k - 1.0) < 1e-9          # 10.0 - 9.0


def test_score_proposal_pair_drops_candidate_on_any_prediction_failure():
    proposal = CandidateProposal(smiles="NCCN", rationale="r", family="f",
                                  source="heuristic", source_id="s")

    def _predict(metal, smiles, model_path=None, allow_fallback=True):
        if metal == "Fe3+":
            raise RuntimeError("model unavailable")
        return MagicMock(value=10.0)

    with patch(
        "des_multi_agent.workflows.metal_binding_selectivity.predict_log_k",
        side_effect=_predict,
    ):
        result, warnings = _score_proposal_pair(
            "Cu2+", ["Zn2+", "Fe3+"], proposal, None, w_affinity=0.5, w_selectivity=0.5,
        )
    assert result is None
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# _build_selectivity_context — text generalization
# ---------------------------------------------------------------------------

def test_build_selectivity_context_single_competitor_byte_identical():
    ctx = _build_selectivity_context("Cu2+", "Zn2+", [], 1, 0.5, 0.5)
    assert "Competitor metal: Zn2+" in ctx
    assert "Off-target" not in ctx


def test_build_selectivity_context_multi_competitor_text():
    ctx = _build_selectivity_context("Cu2+", ["Zn2+", "Fe3+"], [], 1, 0.5, 0.5)
    assert "Off-target metals: Zn2+, Fe3+" in ctx
    assert "Competitor metal:" not in ctx


def test_build_selectivity_context_per_candidate_line_uses_worst_competitor():
    r = SelectivityResult(
        ligand_smiles="NCCN", log_k_target=10.0, log_k_competitor=9.0,
        delta_log_k=1.0, composite_score=5.5, source="heuristic", source_id="s",
        rationale="r", log_k_competitors={"Zn2+": 6.0, "Fe3+": 9.0},
        worst_competitor_metal="Fe3+",
    )
    ctx = _build_selectivity_context("Cu2+", ["Zn2+", "Fe3+"], [r], 1, 0.5, 0.5)
    assert "log_K(Fe3+)=9.00" in ctx


# ---------------------------------------------------------------------------
# run_metal_selectivity_screen — accepts a list end to end
# ---------------------------------------------------------------------------

def test_run_metal_selectivity_screen_accepts_list_competitor():
    with patch(
        "des_multi_agent.workflows.metal_binding_selectivity.predict_log_k",
        side_effect=_fake_predict({"Cu2+": 10.0, "Zn2+": 6.0, "Fe3+": 8.0}),
    ):
        outcome = run_metal_selectivity_screen(
            target_metal="Cu2+", competitor_metal=["Zn2+", "Fe3+"], n=3,
            model_path=None, llm_provider=None, n_cycles=1,
        )
    assert outcome.competitor_metals == ["Zn2+", "Fe3+"]
    assert all(r.log_k_competitors for r in outcome.results)
    assert all(r.worst_competitor_metal == "Fe3+" for r in outcome.results)
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_metal_selectivity_screen.py -v -k "normalize_competitor or score_proposal_pair or build_selectivity_context or accepts_list_competitor"`
Expected: FAIL — `ImportError: cannot import name '_normalize_competitor_metals'` (and similar, since none of this exists yet).

### Step 3: Implement the data model and scoring changes

In `des_multi_agent/workflows/metal_binding_selectivity.py`:

**3a. Add the normalization helper** right after `_safe_canon`:

```python
def _normalize_competitor_metals(competitor_metal: str | list[str]) -> list[str]:
    """Normalize a bare metal string or list of metals into a de-duplicated,
    order-preserving list. Raises ValueError if the result would be empty."""
    raw = [competitor_metal] if isinstance(competitor_metal, str) else list(competitor_metal)
    seen: set[str] = set()
    out: list[str] = []
    for m in raw:
        if m not in seen:
            seen.add(m)
            out.append(m)
    if not out:
        raise ValueError("competitor_metal must contain at least one metal ion")
    return out
```

**3b. `SelectivityResult`** — add two fields with defaults (existing constructions that don't pass them keep working unchanged):

```python
@dataclass(frozen=True)
class SelectivityResult:
    """One ligand screened in a metal selectivity run.

    ``log_k_competitor``/``delta_log_k`` always hold the WORST-CASE off-target's
    value (the one keeping the ligand from being clean) — for a single off-target
    this is that off-target's value, unchanged from prior behavior.
    """

    ligand_smiles: str
    log_k_target: float
    log_k_competitor: float
    delta_log_k: float
    composite_score: float
    source: str
    source_id: str
    rationale: str
    log_k_competitors: dict[str, float] = field(default_factory=dict)
    worst_competitor_metal: str = ""
```

**3c. `SelectivityScreenOutcome`** — rename the field:

```python
@dataclass
class SelectivityScreenOutcome:
    target_metal: str
    competitor_metals: list[str]
    results: list[SelectivityResult]
    n_screened: int
    n_cycles: int
    llm_brainstorm: list[CandidateBrainstorm] = field(default_factory=list)
    llm_candidate_reviews: list[CandidateReview] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    claim_verdicts: list[object] = field(default_factory=list)
    trajectory: object = None   # SearchTrajectory | None
    dft_results: dict = field(default_factory=dict)   # dict[str, DFTResult]
```

**3d. Rewrite `_score_proposal_pair`** (parameter renamed to plural since it's a private helper with a single internal call site — always receives an already-normalized list):

```python
def _score_proposal_pair(
    target_metal: str,
    competitor_metals: list[str],
    proposal: CandidateProposal,
    model_path,
    w_affinity: float,
    w_selectivity: float,
    stability_rule_weight: float = 0.0,
) -> tuple[SelectivityResult | None, list[str]]:
    warnings: list[str] = []
    try:
        pred_target = predict_log_k(
            target_metal, proposal.smiles, model_path=model_path, allow_fallback=True
        )
        pred_competitors = {
            metal: predict_log_k(metal, proposal.smiles, model_path=model_path, allow_fallback=True)
            for metal in competitor_metals
        }
    except Exception as exc:
        warnings.append(f"Prediction failed for {proposal.smiles}: {exc}")
        return None, warnings

    val_target = pred_target.value
    val_competitors = {metal: pred.value for metal, pred in pred_competitors.items()}
    # Blend in the rule-based (Irving-Williams + HSAB + chelate) log K so the
    # metal *difference* reflects real coordination chemistry rather than the
    # heuristic model's near-uniform output.
    if stability_rule_weight > 0.0:
        try:
            from ..chemistry.stability_rules import rule_based_log_k

            rt = rule_based_log_k(target_metal, proposal.smiles)
            w = stability_rule_weight
            val_target = (1.0 - w) * val_target + w * rt
            for metal in list(val_competitors):
                rc = rule_based_log_k(metal, proposal.smiles)
                val_competitors[metal] = (1.0 - w) * val_competitors[metal] + w * rc
        except Exception as exc:
            warnings.append(f"stability-rule blend failed for {proposal.smiles}: {exc}")

    worst_metal = max(val_competitors, key=val_competitors.get)
    worst_val = val_competitors[worst_metal]

    delta_log_k, composite_score = _compute_composite(
        val_target, worst_val, w_affinity, w_selectivity
    )
    return SelectivityResult(
        ligand_smiles=proposal.smiles,
        log_k_target=val_target,
        log_k_competitor=worst_val,
        delta_log_k=delta_log_k,
        composite_score=composite_score,
        source=proposal.source,
        source_id=proposal.source_id,
        rationale=proposal.rationale,
        log_k_competitors=val_competitors,
        worst_competitor_metal=worst_metal,
    ), warnings
```

**3e. Rewrite `_build_selectivity_context`** — parameter keeps its name, widens to `str | list[str]`:

```python
def _build_selectivity_context(
    target_metal: str,
    competitor_metal: str | list[str],
    prev_results: list[SelectivityResult],
    cycle: int,
    w_affinity: float,
    w_selectivity: float,
    des_compatible_hints: list[str] | None = None,
    des_incompatible_hints: list[str] | None = None,
    family_hit_scores: dict[str, list[float]] | None = None,
    saturated_families: set[str] | None = None,
) -> str:
    metals = [competitor_metal] if isinstance(competitor_metal, str) else list(competitor_metal)
    competitor_line = (
        f"Competitor metal: {metals[0]}" if len(metals) == 1
        else f"Off-target metals: {', '.join(metals)}"
    )
    lines = [
        f"Target metal: {target_metal}",
        competitor_line,
        f"Selectivity weight: {w_selectivity} | Affinity weight: {w_affinity}",
        f"Cycle: {cycle}",
    ]
    if prev_results:
        lines.append("Top ligands from previous cycle (highest composite score first):")
        for r in prev_results[:5]:
            lines.append(
                f"  - {r.ligand_smiles}: log_K({target_metal})={r.log_k_target:.2f}, "
                f"log_K({r.worst_competitor_metal})={r.log_k_competitor:.2f}, "
                f"ΔlogK={r.delta_log_k:.2f}, score={r.composite_score:.2f}"
            )
        failed = [r for r in prev_results if r.delta_log_k <= 0][:3]
        if failed:
            lines.append(f"Non-selective ligands (ΔlogK ≤ 0, avoid similar structures):")
            for r in failed:
                lines.append(f"  - {r.ligand_smiles}: ΔlogK={r.delta_log_k:.2f}")
    if des_compatible_hints:
        lines.append("Ligands that formed DES in previous pass (prefer similar scaffolds):")
        for smiles in des_compatible_hints:
            lines.append(f"  - {smiles}")
    if des_incompatible_hints:
        lines.append("Ligands that did NOT form DES (avoid similar scaffolds):")
        for smiles in des_incompatible_hints:
            lines.append(f"  - {smiles}")
    if family_hit_scores:
        items = sorted(family_hit_scores.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        lines.append("Productive selectivity families (avg composite score):")
        for fam, scores in items[:3]:
            avg = sum(scores) / len(scores)
            lines.append(f"  - {fam}: {len(scores)} selective hits, avg score={avg:.2f}")
    if saturated_families:
        lines.append("Exhausted families (diminishing returns, avoid repeating):")
        for fam in sorted(saturated_families)[:4]:
            lines.append(f"  - {fam}")
    return "\n".join(lines)
```

Note: the per-candidate line now uses `r.worst_competitor_metal`/`r.log_k_competitor` instead of the outer `competitor_metal` — for N=1 these are identical values (the sole off-target IS the worst-case by definition), so output is byte-identical.

**3f. `run_metal_selectivity_screen`** — at the very top of the function body (right after the existing two lazy imports), normalize the parameter:

```python
    from ..chemistry.claim_grounding import ground_coordination as _ground_coord
    from ..chemistry_filter import murcko_scaffold_smiles as _scaffold

    competitor_metals = _normalize_competitor_metals(competitor_metal)
```

Then apply these mechanical replacements for the REST of the function body (every remaining reference to the bare `competitor_metal` variable becomes `competitor_metals`):

- Line ~285 `heuristic = generate_ligand_candidates(target_metal, heuristic_n, constraints)` — unchanged (doesn't reference competitor).
- `_build_selectivity_context(target_metal, competitor_metal, ...)` (both call sites, cycle-start and post-scoring) → `_build_selectivity_context(target_metal, competitor_metals, ...)`.
- `llm_provider.brainstorm_ligands_selectivity(target_metal, competitor_metal, constraints, context)` → `llm_provider.brainstorm_ligands_selectivity(target_metal, competitor_metals, constraints, context)`.
- `result, warnings = _score_proposal_pair(target_metal, competitor_metal, proposal, model_path, w_affinity, w_selectivity, stability_rule_weight=stability_rule_weight)` → `_score_proposal_pair(target_metal, competitor_metals, proposal, ...)` (same kwargs otherwise).
- The selectivity-grounding call site:
  ```python
  v = _ground_sel(target_metal, competitor_metal, r.ligand_smiles, "target_selective")
  ```
  becomes:
  ```python
  v = _ground_sel(target_metal, r.worst_competitor_metal, r.ligand_smiles, "target_selective")
  ```
  (`ground_selectivity`'s own signature is unchanged — it always takes one target + one competitor string; we now pass each candidate's own limiting off-target.)
- The DFT-nomination call site:
  ```python
  nominated_smiles = llm_provider.nominate_for_dft(
      top_k_pool, target_metal, competitor_metal, dft_top_n
  )
  ```
  becomes:
  ```python
  nominated_smiles = llm_provider.nominate_for_dft(
      top_k_pool, target_metal, competitor_metals, dft_top_n
  )
  ```
- The DFT-adjustment call site:
  ```python
  adj = _dft_adj(dft_res, target_metal, competitor_metal)
  ```
  becomes:
  ```python
  adj = _dft_adj(dft_res, target_metal, r.worst_competitor_metal)
  ```
  (`dft_selectivity_adjustment`'s own signature is unchanged.)
- The trajectory headline:
  ```python
  headline=f"{target_metal} over {competitor_metal} selectivity",
  ```
  becomes:
  ```python
  headline=f"{target_metal} over {', '.join(competitor_metals)} selectivity",
  ```
  (join of a one-element list equals the bare string — byte-identical for N=1.)
- The final return statement:
  ```python
  return SelectivityScreenOutcome(
      target_metal=target_metal,
      competitor_metal=competitor_metal,
      ...
  )
  ```
  becomes:
  ```python
  return SelectivityScreenOutcome(
      target_metal=target_metal,
      competitor_metals=competitor_metals,
      ...
  )
  ```

Do a final `grep -n "competitor_metal\b" des_multi_agent/workflows/metal_binding_selectivity.py` after editing and confirm every remaining bare (non-plural) occurrence is either the function's own parameter declaration (`competitor_metal: str | list[str],` in the signature — this name is correct and unchanged) or inside `_normalize_competitor_metals`/`_score_proposal_pair`'s own signature (`competitor_metals` — plural, also correct). There should be no leftover bare `competitor_metal` reference inside the function body after the normalization line.

### Step 4: Generalize the LLM prompt layer

In `des_multi_agent/llm/prompts.py`, replace `ligand_selectivity_brainstorm_prompt`'s signature and the one line that renders the competitor:

```python
def ligand_selectivity_brainstorm_prompt(
    target_metal: str,
    competitor_metal: str | list[str],
    constraints: dict | None,
    context: str,
    max_items: int | None = None,
    families: list | None = None,
    facts_block: str = "",
    known_ligand_menu: list | None = None,
) -> str:
    metals = [competitor_metal] if isinstance(competitor_metal, str) else list(competitor_metal)
    parts = [
        "Return raw JSON only. Do not use markdown fences or commentary.\n",
        f"Return a JSON array of candidate ligand SMILES designed for HIGH SELECTIVITY "
        f"for {target_metal} over {', '.join(metals)}.\n",
    ]
```

(the rest of the function body is unchanged — leave everything from `if facts_block:` onward exactly as-is).

Replace `dft_nomination_prompt`'s signature and header line:

```python
def dft_nomination_prompt(
    candidates: list,
    target_metal: str,
    competitor_metal: str | list[str],
    top_n: int = 3,
) -> str:
    """Prompt asking the LLM to nominate candidates for DFT validation."""
    metals = [competitor_metal] if isinstance(competitor_metal, str) else list(competitor_metal)
    competitor_line = (
        f"Competitor: {metals[0]}." if len(metals) == 1
        else f"Off-target metals: {', '.join(metals)}."
    )
    rows = []
    for i, r in enumerate(candidates, 1):
        rows.append(
            f"  {i}. {r.ligand_smiles}  ΔlogK={r.delta_log_k:.2f}  score={r.composite_score:.2f}"
        )
    table = "\n".join(rows)
    return (
        f"You are helping prioritize ligands for DFT validation.\n"
        f"Target metal: {target_metal}. {competitor_line}\n\n"
        f"Top candidates by predicted selectivity (ΔlogK):\n{table}\n\n"
        f"Select 1–{top_n} candidates most worth DFT validation. Prefer:\n"
        f"- Ligands where HSAB ambiguity makes the rule-based prediction uncertain\n"
        f"- Borderline ΔlogK values (small positive) where DFT tiebreaking matters most\n"
        f"- Structurally diverse nominations over similar analogues\n\n"
        f'Return ONLY a JSON list of SMILES strings. Example: ["SMILES1", "SMILES2"]\n'
    )
```

In `des_multi_agent/llm/base.py`, update only the type hints (no body change — both methods already forward `competitor_metal` straight through unchanged):

```python
    def brainstorm_ligands_selectivity(
        self,
        target_metal: str,
        competitor_metal: str | list[str],
        constraints: dict | None,
        context: str,
    ) -> list[CandidateBrainstorm]:
```

```python
    def nominate_for_dft(
        self,
        candidates: list,
        target_metal: str,
        competitor_metal: str | list[str],
        top_n: int = 3,
    ) -> list[str]:
```

### Step 5: Fix the mechanical `SelectivityScreenOutcome(competitor_metal=...)` renames

In each of these 5 files, every direct `SelectivityScreenOutcome(...)` construction that passes `competitor_metal="Zn2+"` must change to `competitor_metals=["Zn2+"]` (wrap the same string in a one-element list). Do **not** touch any `run_metal_selectivity_screen(competitor_metal="Zn2+", ...)` or `run_selectivity_des_pipeline(competitor_metal="Zn2+", ...)` **function calls** — those keep working unchanged since the parameter still accepts a bare string.

- `tests/test_dft_integration.py` — line 33
- `tests/test_metal_selectivity_screen.py` — lines 231, 249, 333
- `tests/test_dft_cli_report.py` — lines 16, 63
- `tests/test_trajectory_capture_pipeline.py` — lines 22, 63, 112 (this file also has `competitor_metal="Zn2+"` at lines 34, 89, 128 — those are `run_selectivity_des_pipeline(...)` calls, leave them alone)
- `tests/test_selectivity_des_pipeline.py` — line 33 only (this is a `SelectivityScreenOutcome(` fixture; the file's many other `competitor_metal="Zn2+"` occurrences are `run_selectivity_des_pipeline(...)`/`run_metal_selectivity_screen(...)` calls and its `SelectivityDesPipelineOutcome(competitor_metal=...)` constructions at lines 243/293 — leave the pipeline-outcome ones for Task 2, they're a different dataclass)

Example fix (identical pattern in each file):

```python
# before
outcome = SelectivityScreenOutcome(
    target_metal="Cu2+", competitor_metal="Zn2+",
    ...
)

# after
outcome = SelectivityScreenOutcome(
    target_metal="Cu2+", competitor_metals=["Zn2+"],
    ...
)
```

### Step 6: Run tests to verify they pass

Run: `pytest tests/test_metal_selectivity_screen.py tests/test_llm_prompts.py tests/test_dft_nomination_prompt.py tests/test_dft_integration.py tests/test_dft_cli_report.py tests/test_trajectory_capture_pipeline.py tests/test_metal_workflows_grounding.py -v`
Expected: all PASS.

### Step 7: Run full suite to check for regressions

Run: `pytest tests/ -q --ignore=tests/test_benchmarks_examples.py`
Expected: all pass, including `tests/test_selectivity_des_pipeline.py` — that file's only `SelectivityScreenOutcome(` construction (in the `_sel_outcome` fixture) is fixed by Step 5 above, and `SelectivityDesPipelineOutcome` itself is untouched by this task (still on its old single-competitor field), so `run_selectivity_des_pipeline`'s existing code path — which never reads `sel_outcome.competitor_metal(s)` from the nested `SelectivityScreenOutcome` it receives — is unaffected. If anything in `tests/test_selectivity_des_pipeline.py` fails at this step, stop and investigate before proceeding — it should not.

### Step 8: Commit

```bash
git add des_multi_agent/workflows/metal_binding_selectivity.py \
        des_multi_agent/llm/prompts.py des_multi_agent/llm/base.py \
        tests/test_dft_integration.py tests/test_metal_selectivity_screen.py \
        tests/test_dft_cli_report.py tests/test_trajectory_capture_pipeline.py \
        tests/test_selectivity_des_pipeline.py
git commit -m "feat: support multiple off-target metals in selectivity scoring engine"
```

---

## Task 2: `selectivity_des_pipeline.py`

**Files:**
- Modify: `des_multi_agent/workflows/selectivity_des_pipeline.py`
- Test: `tests/test_selectivity_des_pipeline.py` (fix the 2 `SelectivityDesPipelineOutcome(competitor_metal=...)` constructions; append new tests)

**Interfaces:**
- Consumes: `run_metal_selectivity_screen(target_metal, competitor_metal: str | list[str], ...)` (Task 1, already accepts a list) and `SelectivityScreenOutcome.competitor_metals: list[str]` (Task 1).
- Produces: `SelectivityDesPipelineOutcome.competitor_metals: list[str]` — Task 4 (`reporting.py`) consumes this.

### Step 1: Write the failing tests

In `tests/test_selectivity_des_pipeline.py`, fix the two existing `SelectivityDesPipelineOutcome(competitor_metal="Zn2+", ...)` constructions (lines 243, 293) to `competitor_metals=["Zn2+"]`, matching the same rename pattern as Task 1.

Append this new test (adjust the exact mocking pattern to match whatever fixture/mocking style the rest of this file already uses for `run_metal_selectivity_screen` — read the file first to match its existing `monkeypatch`/`patch` conventions):

```python
def test_run_selectivity_des_pipeline_accepts_list_competitor(monkeypatch):
    import des_multi_agent.workflows.selectivity_des_pipeline as pipe

    captured = {}

    def _fake_screen(**kwargs):
        captured["competitor_metal"] = kwargs["competitor_metal"]
        return SelectivityScreenOutcome(
            target_metal=kwargs["target_metal"],
            competitor_metals=["Zn2+", "Fe3+"],
            results=[],
            n_screened=0,
            n_cycles=1,
        )

    monkeypatch.setattr(pipe, "run_metal_selectivity_screen", _fake_screen)

    outcome = pipe.run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal=["Zn2+", "Fe3+"],
        checkpoint_path="ckpt.pt",
        n_outer_cycles=1,
    )

    assert captured["competitor_metal"] == ["Zn2+", "Fe3+"]
    assert outcome.competitor_metals == ["Zn2+", "Fe3+"]
```

(Import `SelectivityScreenOutcome` from `des_multi_agent.workflows.metal_binding_selectivity` at the top of the test file if not already imported.)

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_selectivity_des_pipeline.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'competitor_metals'` on `SelectivityDesPipelineOutcome`, since the dataclass field is still named `competitor_metal` (singular) at this point.

### Step 3: Implement

In `des_multi_agent/workflows/selectivity_des_pipeline.py`:

**3a.** Widen the `SelectivityDesPipelineOutcome` dataclass field:

```python
@dataclass
class SelectivityDesPipelineOutcome:
    target_metal: str
    competitor_metals: list[str]
    selectivity_outcome: SelectivityScreenOutcome
    ligand_des_results: list[LigandDesResult]
    n_outer_cycles_run: int
    converged: bool
    warnings: list[str] = field(default_factory=list)
    trajectory: object = None   # SearchTrajectory | None
```

**3b.** Widen `run_selectivity_des_pipeline`'s parameter type hint only (name unchanged):

```python
def run_selectivity_des_pipeline(
    target_metal: str,
    competitor_metal: str | list[str],
    checkpoint_path: str,
    ...
```

**3c.** The pipeline forwards `competitor_metal` straight to `run_metal_selectivity_screen(competitor_metal=competitor_metal, ...)` — leave this call site completely unchanged; `run_metal_selectivity_screen` (Task 1) already normalizes whatever is passed to it.

**3d.** The trajectory headline and the final return's field both need updating:

```python
    pipe_trajectory = SearchTrajectory(
        workflow="selectivity-des",
        headline=f"{target_metal} over {', '.join(_as_list(competitor_metal))} — selectivity-DES",
        ...
```

Add a tiny local helper near the top of the file (this file has no existing normalization helper to reuse, and does not need the full validate+dedup behavior — that already happened inside `run_metal_selectivity_screen`, which this function always calls before reaching the trajectory/return code):

```python
def _as_list(competitor_metal: str | list[str]) -> list[str]:
    return [competitor_metal] if isinstance(competitor_metal, str) else list(competitor_metal)
```

Use `_as_list(competitor_metal)` in both the trajectory headline and the final return:

```python
    return SelectivityDesPipelineOutcome(
        target_metal=target_metal,
        competitor_metals=_as_list(competitor_metal),
        selectivity_outcome=final_selectivity_outcome,
        ligand_des_results=final_ligand_des_results,
        n_outer_cycles_run=outer_cycle_count,
        converged=converged,
        warnings=all_warnings,
        trajectory=pipe_trajectory,
    )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_selectivity_des_pipeline.py -v`
Expected: all PASS.

### Step 5: Run full suite to check for regressions

Run: `pytest tests/ -q --ignore=tests/test_benchmarks_examples.py`
Expected: all pass (the `tests/test_selectivity_des_pipeline.py` failures left over from Task 1 are now resolved).

### Step 6: Commit

```bash
git add des_multi_agent/workflows/selectivity_des_pipeline.py tests/test_selectivity_des_pipeline.py
git commit -m "feat: thread multi-off-target support through selectivity-des pipeline"
```

---

## Task 3: CLI comma-separated parsing

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py` if it exists — otherwise check for the CLI's existing test file (run `grep -rl "competitor_metal_ion\|metal-selectivity" tests/*.py` to find it) and append there.

**Interfaces:**
- Consumes: `run_metal_selectivity_screen(competitor_metal: str | list[str], ...)` (Task 1) and `run_selectivity_des_pipeline(competitor_metal: str | list[str], ...)` (Task 2).
- Produces: nothing new — this task only changes how the CLI parses one flag before forwarding it.

### Step 1: Write the failing test

First run `grep -rn "competitor_metal_ion\|metal-selectivity" tests/*.py | grep -i cli` to find the right existing CLI test file (if none exists, create `tests/test_cli_selectivity_flags.py`). Add:

```python
def test_competitor_metal_ion_comma_separated_splits_to_list(monkeypatch):
    import des_multi_agent.cli as cli_mod

    captured = {}

    def _fake_screen(**kwargs):
        captured["competitor_metal"] = kwargs["competitor_metal"]
        from des_multi_agent.workflows.metal_binding_selectivity import SelectivityScreenOutcome
        return SelectivityScreenOutcome(
            target_metal=kwargs["target_metal"], competitor_metals=["Zn2+", "Fe3+"],
            results=[], n_screened=0, n_cycles=1,
        )

    monkeypatch.setattr(cli_mod, "run_metal_selectivity_screen", _fake_screen)
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", "--workflow", "metal-selectivity",
         "--target-metal-ion", "Cu2+", "--competitor-metal-ion", "Zn2+,Fe3+"],
    )
    cli_mod.main()

    assert captured["competitor_metal"] == ["Zn2+", "Fe3+"]


def test_competitor_metal_ion_single_value_stays_string(monkeypatch):
    import des_multi_agent.cli as cli_mod

    captured = {}

    def _fake_screen(**kwargs):
        captured["competitor_metal"] = kwargs["competitor_metal"]
        from des_multi_agent.workflows.metal_binding_selectivity import SelectivityScreenOutcome
        return SelectivityScreenOutcome(
            target_metal=kwargs["target_metal"], competitor_metals=["Zn2+"],
            results=[], n_screened=0, n_cycles=1,
        )

    monkeypatch.setattr(cli_mod, "run_metal_selectivity_screen", _fake_screen)
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", "--workflow", "metal-selectivity",
         "--target-metal-ion", "Cu2+", "--competitor-metal-ion", "Zn2+"],
    )
    cli_mod.main()

    assert captured["competitor_metal"] == ["Zn2+"]
```

Read `des_multi_agent/cli.py`'s existing `main()` entry point and any existing CLI test file first to confirm the exact invocation pattern used elsewhere (e.g. whether tests call `cli_mod.main()` with `sys.argv` patched, or call an internal `run(args)` function directly) — match that established pattern exactly; adjust the two tests above to fit if the convention differs from what's sketched here.

### Step 2: Run tests to verify they fail

Run: `pytest <the test file>::test_competitor_metal_ion_comma_separated_splits_to_list -v`
Expected: FAIL — `captured["competitor_metal"]` is still `"Zn2+,Fe3+"` (the raw unsplit string).

### Step 3: Implement

In `des_multi_agent/cli.py`, find the `metal-selectivity` block (around where `args.target_metal_ion`/`args.competitor_metal_ion` are validated) and the `selectivity-des` block. In both, right before constructing the call into the workflow function, split the raw CLI string on comma:

```python
        competitor_metals = [m.strip() for m in args.competitor_metal_ion.split(",") if m.strip()]
```

Metal-selectivity block — change:

```python
        sel_outcome = run_metal_selectivity_screen(
            target_metal=args.target_metal_ion,
            competitor_metal=args.competitor_metal_ion,
            ...
```

to:

```python
        competitor_metals = [m.strip() for m in args.competitor_metal_ion.split(",") if m.strip()]
        sel_outcome = run_metal_selectivity_screen(
            target_metal=args.target_metal_ion,
            competitor_metal=competitor_metals,
            ...
```

Selectivity-des block — same pattern, right before the `run_selectivity_des_pipeline(...)` call:

```python
            competitor_metals = [m.strip() for m in args.competitor_metal_ion.split(",") if m.strip()]
            pipeline_outcome = run_selectivity_des_pipeline(
                target_metal=args.target_metal_ion,
                competitor_metal=competitor_metals,
                ...
```

A single value with no comma (`"Zn2+".split(",")` → `["Zn2+"]`) produces a one-element list — identical downstream behavior to today's bare string, since `run_metal_selectivity_screen`/`run_selectivity_des_pipeline` both accept either form.

### Step 4: Run tests to verify they pass

Run: `pytest <the test file> -v`
Expected: both new tests PASS.

### Step 5: Run full suite to check for regressions

Run: `pytest tests/ -q --ignore=tests/test_benchmarks_examples.py`
Expected: all pass.

### Step 6: Commit

```bash
git add des_multi_agent/cli.py <the test file>
git commit -m "feat: accept comma-separated multi-off-target metals on the CLI"
```

---

## Task 4: Report formatting

**Files:**
- Modify: `des_multi_agent/reporting.py` (`format_metal_selectivity_report`, `format_selectivity_des_report`)
- Test: `tests/test_reporting.py` if it exists — otherwise `grep -rl "format_metal_selectivity_report\|format_selectivity_des_report" tests/*.py` to find the right file, or create `tests/test_selectivity_reporting.py`.

**Interfaces:**
- Consumes: `SelectivityResult.log_k_competitors: dict[str, float]`, `SelectivityResult.worst_competitor_metal: str` (Task 1), `SelectivityScreenOutcome.competitor_metals: list[str]` (Task 1), `SelectivityDesPipelineOutcome.competitor_metals: list[str]` (Task 2).
- Produces: nothing new — this is the final consumer-facing task.

### Step 1: Write the failing tests

First run `grep -rl "format_metal_selectivity_report\|format_selectivity_des_report" tests/*.py` to find where these are currently tested (if anywhere) and add to that file, matching its existing style. Otherwise create `tests/test_selectivity_reporting.py`:

```python
"""Tests for multi-off-target report formatting."""
from __future__ import annotations

from des_multi_agent.reporting import format_metal_selectivity_report
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
)


def _result(smiles, log_k_target, log_k_competitors, worst_metal):
    worst_val = log_k_competitors[worst_metal]
    return SelectivityResult(
        ligand_smiles=smiles, log_k_target=log_k_target, log_k_competitor=worst_val,
        delta_log_k=log_k_target - worst_val, composite_score=log_k_target - worst_val,
        source="heuristic", source_id="s", rationale="r",
        log_k_competitors=log_k_competitors, worst_competitor_metal=worst_metal,
    )


def test_report_single_competitor_byte_identical_header():
    outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metals=["Zn2+"],
        results=[_result("NCCN", 10.0, {"Zn2+": 6.0}, "Zn2+")],
        n_screened=1, n_cycles=1,
    )
    report = format_metal_selectivity_report(outcome)
    assert "=== Metal Selectivity Screen: Cu2+ over Zn2+ ===" in report
    assert "off_target_breakdown" not in report


def test_report_multi_competitor_header_lists_all_off_targets():
    outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metals=["Zn2+", "Fe3+"],
        results=[_result("NCCN", 10.0, {"Zn2+": 6.0, "Fe3+": 9.0}, "Fe3+")],
        n_screened=1, n_cycles=1,
    )
    report = format_metal_selectivity_report(outcome)
    assert "=== Metal Selectivity Screen: Cu2+ over Zn2+, Fe3+ ===" in report


def test_report_multi_competitor_breakdown_column_marks_worst_case():
    outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metals=["Zn2+", "Fe3+"],
        results=[_result("NCCN", 10.0, {"Zn2+": 6.0, "Fe3+": 9.0}, "Fe3+")],
        n_screened=1, n_cycles=1,
    )
    report = format_metal_selectivity_report(outcome)
    assert "off_target_breakdown" in report
    assert "Zn2+=6.00" in report
    assert "Fe3+=9.00*" in report
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_selectivity_reporting.py -v` (or wherever you added them)
Expected: `test_report_single_competitor_byte_identical_header` FAILS with `AttributeError: 'SelectivityScreenOutcome' object has no attribute 'competitor_metal'` (the report code still reads the old field name), and the other two fail similarly or on missing breakdown text.

### Step 3: Implement

In `des_multi_agent/reporting.py`, in `format_metal_selectivity_report`:

Replace the header construction:

```python
    header_lines = [
        f"=== Metal Selectivity Screen: {outcome.target_metal} over {outcome.competitor_metal} ===",
        f"Screened {outcome.n_screened} candidate(s) over {outcome.n_cycles} cycle(s).",
        f"Top ligand: {top_str}",
        "=" * 52,
        "",
        col_header,
    ]
```

with:

```python
    header_lines = [
        f"=== Metal Selectivity Screen: {outcome.target_metal} over "
        f"{', '.join(outcome.competitor_metals)} ===",
        f"Screened {outcome.n_screened} candidate(s) over {outcome.n_cycles} cycle(s).",
        f"Top ligand: {top_str}",
        "=" * 52,
        "",
        col_header,
    ]
```

Add the breakdown column, only when there is more than one off-target. Replace:

```python
    col_header = "ligand | log_k_target | log_k_competitor | delta_log_k | score"
    if has_dft:
        col_header += " | dft_homo_ev | dft_donor_chg"
    col_header += " | source | rationale"
```

with:

```python
    has_multi_competitor = len(outcome.competitor_metals) > 1
    col_header = "ligand | log_k_target | log_k_competitor | delta_log_k | score"
    if has_multi_competitor:
        col_header += " | off_target_breakdown"
    if has_dft:
        col_header += " | dft_homo_ev | dft_donor_chg"
    col_header += " | source | rationale"
```

And in the row-building loop, replace:

```python
        rows.append(
            f"{r.ligand_smiles} | {r.log_k_target:.2f} | {r.log_k_competitor:.2f} | "
            f"{r.delta_log_k:.2f} | {r.composite_score:.2f}{dft_cols} | {src} | {r.rationale}"
        )
```

with:

```python
        breakdown_col = ""
        if has_multi_competitor:
            parts = [
                f"{metal}={val:.2f}" + ("*" if metal == r.worst_competitor_metal else "")
                for metal, val in r.log_k_competitors.items()
            ]
            breakdown_col = f" | {', '.join(parts)}"
        rows.append(
            f"{r.ligand_smiles} | {r.log_k_target:.2f} | {r.log_k_competitor:.2f} | "
            f"{r.delta_log_k:.2f} | {r.composite_score:.2f}{breakdown_col}{dft_cols} | "
            f"{src} | {r.rationale}"
        )
```

In `format_selectivity_des_report`, apply the same header change:

```python
        f"=== Selectivity-DES Pipeline: {outcome.target_metal} over {outcome.competitor_metal} ===",
```

becomes:

```python
        f"=== Selectivity-DES Pipeline: {outcome.target_metal} over "
        f"{', '.join(outcome.competitor_metals)} ===",
```

(The `sec1` table in `format_selectivity_des_report` keeps its existing `log_k_competitor`/`delta_log_k` columns unchanged — those already hold the worst-case value automatically via `SelectivityResult`. No breakdown column is added to this pipeline report; it is out of scope for this task since the design does not require it there, and the per-ligand selectivity breakdown is already fully available via the `metal-selectivity` workflow's own report.)

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_selectivity_reporting.py -v` (or wherever you added them)
Expected: all PASS.

### Step 5: Run full suite to check for regressions

Run: `pytest tests/ -q --ignore=tests/test_benchmarks_examples.py`
Expected: all pass.

### Step 6: Commit

```bash
git add des_multi_agent/reporting.py tests/test_selectivity_reporting.py
git commit -m "feat: report multi-off-target breakdown with worst-case marker"
```

---

## Final Verification

After all four tasks:

```bash
pytest tests/ -q --ignore=tests/test_benchmarks_examples.py
```

Expected: all tests pass. Manually sanity-check the new capability:

```bash
python -m des_multi_agent.cli --workflow metal-selectivity \
  --target-metal-ion Cu2+ --competitor-metal-ion "Zn2+,Fe3+,Ni2+" \
  --n 10 --stability-constant-model-path artifacts/stability_constants/model.json
```

Confirm the report header reads `"Cu2+ over Zn2+, Fe3+, Ni2+"` and the `off_target_breakdown` column appears with an asterisk on the limiting metal for each row. Then confirm the existing single-competitor invocation is unchanged:

```bash
python -m des_multi_agent.cli --workflow metal-selectivity \
  --target-metal-ion Cu2+ --competitor-metal-ion Zn2+ \
  --n 10 --stability-constant-model-path artifacts/stability_constants/model.json
```
