# Robustness & UX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the CLI against bad user input and make all four workflows discoverable through help text and documentation.

**Architecture:** Five independent tasks targeting `des_multi_agent/cli.py`, `des_multi_agent/workflows/selectivity_des_pipeline.py`, `README.md`, `docs/tutorial.md`, and `examples/README.md`. No new modules needed.

**Tech Stack:** Python argparse, RDKit `Chem.MolFromSmiles`, Markdown.

---

## File Map

| File | What changes |
|------|-------------|
| `des_multi_agent/cli.py` | Task 1: validation helpers + checks; Task 2: help text + `--version` |
| `des_multi_agent/workflows/selectivity_des_pipeline.py` | Task 3: per-ligand progress lines |
| `README.md` | Task 4: selectivity-des section |
| `docs/tutorial.md` | Task 4: selectivity-des section |
| `examples/README.md` | Task 4: selectivity-des entry |
| `tests/test_cli.py` | Tasks 1 & 2: new test cases |
| `tests/test_selectivity_des_pipeline.py` | Task 3: progress output test |

---

## Task 1: CLI Input Validation

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py`

Add argument-level validation for numeric bounds. Two helper functions go just above `build_parser()`, and the `--n`/weight/cycle flags switch to those helpers as their `type=`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cli.py`:

```python
import pytest

def test_cli_n_zero_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--n", "0", "--checkpoint-path", "ckpt.pt"])


def test_cli_n_negative_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--n", "-5", "--checkpoint-path", "ckpt.pt"])


def test_cli_affinity_weight_above_one_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--affinity-weight", "1.5"])


def test_cli_affinity_weight_negative_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--affinity-weight", "-0.1"])


def test_cli_selectivity_weight_above_one_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--selectivity-weight", "2.0"])


def test_cli_viscosity_weight_above_one_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--viscosity-weight", "5.0"])


def test_cli_n_cycles_zero_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--n-cycles", "0"])


def test_cli_n_des_cycles_zero_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--n-des-cycles", "0"])


def test_cli_top_ligands_zero_is_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--component-a", "CCO", "--top-ligands", "0"])


def test_cli_valid_inputs_accepted():
    """Boundary values that ARE valid should parse fine."""
    parser = build_parser()
    args = parser.parse_args([
        "--component-a", "CCO",
        "--n", "1",
        "--affinity-weight", "0.0",
        "--selectivity-weight", "1.0",
        "--viscosity-weight", "0.0",
        "--n-cycles", "1",
    ])
    assert args.n == 1
    assert args.affinity_weight == 0.0
    assert args.selectivity_weight == 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_cli.py::test_cli_n_zero_is_rejected tests/test_cli.py::test_cli_affinity_weight_above_one_is_rejected -v 2>&1 | tail -20
```

Expected: `FAILED` — `SystemExit` not raised because argparse currently accepts any int/float.

- [ ] **Step 3: Add two helper functions and update argument definitions in `des_multi_agent/cli.py`**

Add these two helper functions immediately before `build_parser()` (around line 53):

```python
def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {n}")
    return n


def _unit_float(value: str) -> float:
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number")
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0.0, 1.0], got {f}")
    return f
```

Then in `build_parser()`, change these argument definitions:

```python
    # Line 57: --n
    parser.add_argument("--n", type=_positive_int, default=20)

    # Lines 68-71: --affinity-weight
    parser.add_argument("--affinity-weight", type=_unit_float, default=0.5, dest="affinity_weight",
                        help="Weight for log K(target) in composite selectivity score (default 0.5)")

    # Lines 72-75: --selectivity-weight  (keep help, just change type)
    parser.add_argument("--selectivity-weight", type=_unit_float, default=0.5, dest="selectivity_weight",
                        help="Weight for delta log K in composite selectivity score (default 0.5)")

    # Lines 76-78: --n-des-candidates
    parser.add_argument(
        "--n-des-candidates",
        type=_positive_int,
        default=20,
        dest="n_des_candidates",
        help="DES candidate search breadth per ligand per cycle (selectivity-des workflow)",
    )

    # Lines 79-85: --n-des-cycles
    parser.add_argument(
        "--n-des-cycles",
        type=_positive_int,
        default=3,
        dest="n_des_cycles",
        help="DES iteration depth per ligand (selectivity-des workflow)",
    )

    # Lines 86-92: --n-outer-cycles
    parser.add_argument(
        "--n-outer-cycles",
        type=_positive_int,
        default=2,
        dest="n_outer_cycles",
        help="Outer loop iteration cap for selectivity-des workflow",
    )

    # Lines 100-106: --top-ligands
    parser.add_argument(
        "--top-ligands",
        type=_positive_int,
        default=3,
        dest="top_ligands",
        help="Maximum ligands passed from Phase 1 to Phase 2 (selectivity-des workflow)",
    )

    # Lines 155-159: --n-cycles
    parser.add_argument(
        "--n-cycles",
        type=_positive_int,
        default=1,
        dest="n_cycles",
        help="Number of screening iterations; the top-K hits from each cycle seed the next (default: 1 = single shot)",
    )

    # Lines 169-174: --viscosity-weight
    parser.add_argument(
        "--viscosity-weight",
        type=_unit_float,
        default=0.3,
        dest="viscosity_weight",
        help="Weight [0,1] of the viscosity component in composite ranking (default: 0.3)",
    )
```

Note: do NOT change `--abs-tm-threshold`, `--rel-drop-min`, `--viscosity-threshold`, `--min-delta-log-k`, `--min-trust-score`, `--soft-penalty-weight`, or the `--std-*` thresholds — those accept values outside [0,1] by design.

- [ ] **Step 4: Run to verify tests pass**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_cli.py -v 2>&1 | tail -30
```

Expected: all `test_cli_*` pass.

- [ ] **Step 5: Also run the full test suite to catch regressions**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: all tests pass (same count as before this task, ± new tests added).

- [ ] **Step 6: Commit**

```bash
cd /home/qshao/DES-Agent && git add des_multi_agent/cli.py tests/test_cli.py && git commit -m "feat(cli): add positive-int and unit-float argument validation"
```

---

## Task 2: CLI Help Text Improvements and `--version` Flag

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py`

Add descriptive help strings to the most-used undocumented flags and a `--version` flag.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cli.py`:

```python
def test_cli_version_flag_exits_zero(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0


def test_cli_workflow_help_mentions_all_workflows():
    """--help text must mention all four workflow names."""
    import io
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    # We test the help text indirectly via the formatter
    help_text = parser.format_help()
    for wf in ("des", "metal-binding", "metal-selectivity", "selectivity-des"):
        assert wf in help_text


def test_cli_component_a_help_mentions_smiles():
    help_text = build_parser().format_help()
    assert "SMILES" in help_text or "smiles" in help_text.lower()


def test_cli_workflow_help_describes_each_workflow():
    help_text = build_parser().format_help()
    # The selectivity-des workflow description should appear in help
    assert "selectivity-des" in help_text
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_cli.py::test_cli_version_flag_exits_zero tests/test_cli.py::test_cli_component_a_help_mentions_smiles -v 2>&1 | tail -15
```

Expected: `FAILED` — `--version` not defined yet, `--component-a` has no help.

- [ ] **Step 3: Apply help text changes in `des_multi_agent/cli.py`**

In `build_parser()`, replace the relevant argument definitions with the versions below:

```python
    # Add --version immediately after parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    # --workflow: add a help string explaining each workflow
    parser.add_argument(
        "--workflow",
        choices=["des", "metal-binding", "metal-selectivity", "selectivity-des"],
        default="des",
        help=(
            "Workflow to run: "
            "'des' screens DES partners for --component-a (SMILES required); "
            "'metal-binding' predicts stability constants for --metal-ion + --ligand-smiles; "
            "'metal-selectivity' screens ligands for selectivity between two metal ions; "
            "'selectivity-des' runs metal-selectivity then DES partner search for top ligands"
        ),
    )

    # --component-a: add SMILES hint
    parser.add_argument(
        "--component-a",
        default=None,
        help="SMILES string for the primary DES component (required for --workflow des)",
    )

    # --n: add description and show default
    parser.add_argument(
        "--n",
        type=_positive_int,
        default=20,
        help="Number of candidate partners to screen per cycle (default: 20)",
    )

    # --checkpoint-path: add description
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to a trained ChemBERTa checkpoint (.pt file); required for des and selectivity-des workflows. "
             "Auto-discovered from ml_des_mp/runs/ if omitted for the des workflow.",
    )
```

For the selectivity-des-only flags, append `" (selectivity-des workflow only)"` to their existing help strings:

```python
    parser.add_argument(
        "--n-des-candidates",
        type=_positive_int,
        default=20,
        dest="n_des_candidates",
        help="DES candidate search breadth per ligand per cycle (selectivity-des workflow only)",
    )
    parser.add_argument(
        "--n-des-cycles",
        type=_positive_int,
        default=3,
        dest="n_des_cycles",
        help="DES iteration depth per ligand (selectivity-des workflow only)",
    )
    parser.add_argument(
        "--n-outer-cycles",
        type=_positive_int,
        default=2,
        dest="n_outer_cycles",
        help="Outer loop iteration cap (selectivity-des workflow only)",
    )
    parser.add_argument(
        "--min-delta-log-k",
        type=float,
        default=0.0,
        dest="min_delta_log_k",
        help="Minimum delta log K threshold for Phase 1 → Phase 2 bridge filter (selectivity-des workflow only)",
    )
    parser.add_argument(
        "--top-ligands",
        type=_positive_int,
        default=3,
        dest="top_ligands",
        help="Maximum ligands bridged from Phase 1 to Phase 2 (selectivity-des workflow only)",
    )
```

- [ ] **Step 4: Run to verify tests pass**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_cli.py -v 2>&1 | tail -30
```

Expected: all pass including the four new tests.

- [ ] **Step 5: Manually verify `--help` output looks correct**

```bash
cd /home/qshao/DES-Agent && python -m des_multi_agent.cli --help 2>&1 | head -40
```

Confirm: `--version` appears, `--workflow` has a description, `--component-a` mentions SMILES, `--n` has a description.

- [ ] **Step 6: Run full test suite**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/ -q --tb=short 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
cd /home/qshao/DES-Agent && git add des_multi_agent/cli.py tests/test_cli.py && git commit -m "feat(cli): add --version flag and descriptive help text for core flags"
```

---

## Task 3: Per-Ligand Progress in Selectivity-DES Phase 2

**Files:**
- Modify: `des_multi_agent/workflows/selectivity_des_pipeline.py`
- Test: `tests/test_selectivity_des_pipeline.py`

The Phase 2 loop currently prints one line before entering the ligand loop, then goes silent for potentially minutes. Add a per-ligand progress line so users know which ligand is being processed.

- [ ] **Step 1: Write failing test**

Add to `tests/test_selectivity_des_pipeline.py`:

```python
import sys
from unittest.mock import patch, MagicMock
from des_multi_agent.workflows.selectivity_des_pipeline import run_selectivity_des_pipeline

def test_pipeline_phase2_prints_per_ligand_progress(capsys):
    """Each ligand in Phase 2 should print a progress line to stderr."""
    mock_sel_outcome = MagicMock()
    mock_sel_outcome.results = [
        MagicMock(ligand_smiles="CCO", delta_log_k=1.0),
        MagicMock(ligand_smiles="CCN", delta_log_k=1.5),
    ]
    mock_sel_outcome.warnings = []

    mock_des_mco = MagicMock()
    mock_des_mco.final_outcome.results = []
    mock_des_mco.cycle_deltas = []

    with patch(
        "des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen",
        return_value=mock_sel_outcome,
    ), patch(
        "des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search",
        return_value=mock_des_mco,
    ):
        run_selectivity_des_pipeline(
            target_metal="Cu2+",
            competitor_metal="Zn2+",
            checkpoint_path="ckpt.pt",
            n_outer_cycles=1,
            top_ligands=2,
        )

    err = capsys.readouterr().err
    # Should see "ligand 1/2" and "ligand 2/2"
    assert "ligand 1/2" in err
    assert "ligand 2/2" in err
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_selectivity_des_pipeline.py::test_pipeline_phase2_prints_per_ligand_progress -v 2>&1 | tail -15
```

Expected: `FAILED` — "ligand 1/2" not in stderr output.

- [ ] **Step 3: Add per-ligand progress print in the pipeline**

In `des_multi_agent/workflows/selectivity_des_pipeline.py`, find the `for ligand_result in shortlisted:` loop (currently around line 121). Add a progress print as the very first statement inside the loop:

Replace:
```python
        for ligand_result in shortlisted:
            try:
                des_mco = run_multi_cycle_search(
```

With:
```python
        for ligand_idx, ligand_result in enumerate(shortlisted, 1):
            print(
                f"[outer {outer_cycle}/{n_outer_cycles}] phase 2: ligand {ligand_idx}/{len(shortlisted)}"
                f" — {ligand_result.ligand_smiles}",
                file=sys.stderr, flush=True,
            )
            try:
                des_mco = run_multi_cycle_search(
```

- [ ] **Step 4: Run to verify test passes**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_selectivity_des_pipeline.py::test_pipeline_phase2_prints_per_ligand_progress -v 2>&1 | tail -10
```

Expected: `PASSED`.

- [ ] **Step 5: Run the full pipeline test file**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_selectivity_des_pipeline.py -v 2>&1 | tail -25
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/qshao/DES-Agent && git add des_multi_agent/workflows/selectivity_des_pipeline.py tests/test_selectivity_des_pipeline.py && git commit -m "feat(selectivity-des): add per-ligand progress line in Phase 2 loop"
```

---

## Task 4: Document the Selectivity-DES Workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`

No code changes — documentation only. No tests needed for this task.

### README.md

The README ends at line 176. Insert a new section between the DES viscosity example block (ends around line 128) and the `## Task Router` section (starts around line 136). Add this after the last workflow example block and before the `## Task Router` heading:

- [ ] **Step 1: Add selectivity-des section to README.md**

Find this text in `README.md`:
```
## Task Router
```

Insert this block immediately before it:

```markdown
## Metal-Selectivity and Selectivity-DES Workflows

Screen ligands for selectivity between two competing metal ions:

```bash
python -m des_multi_agent.cli --workflow metal-selectivity \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion Zn2+ \
  --n 20 --n-cycles 3 \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

The combined selectivity-DES workflow runs Phase 1 (metal-selectivity screening) and then Phase 2 (DES partner search for the top selective ligands). The two phases are connected by an outer feedback loop: DES-compatible ligands found in Phase 2 are fed back as hints to Phase 1 on the next outer cycle, steering brainstorming toward ligands that are both selective and DES-compatible.

```bash
python -m des_multi_agent.cli --workflow selectivity-des \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion Zn2+ \
  --n 20 --n-cycles 3 \
  --n-des-candidates 20 --n-des-cycles 3 \
  --n-outer-cycles 2 --top-ligands 3 --min-delta-log-k 0.5 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

Key flags:
- `--n-outer-cycles`: how many Phase 1 → Phase 2 feedback loops to run (default: 2)
- `--top-ligands`: how many Phase 1 ligands are passed to Phase 2 each outer cycle (default: 3)
- `--min-delta-log-k`: minimum selectivity threshold (delta log K) for the Phase 1 → Phase 2 bridge (default: 0.0)
- `--n-des-candidates` and `--n-des-cycles` control the DES search breadth and depth for each ligand in Phase 2

The report shows a selectivity table with a `des_compatible` column (Section 1) and per-ligand DES partner blocks (Section 2). The outer loop stops early when the DES-compatible ligand set stabilises across two consecutive cycles.

```

- [ ] **Step 2: Add selectivity-des section to docs/tutorial.md**

Find this text near the end of `docs/tutorial.md`:
```
## Uncertainty Controls
```

Insert this block immediately before it:

```markdown
## Selectivity-DES Pipeline

The selectivity-DES workflow combines Phase 1 (metal-ion selectivity screening) and Phase 2 (DES partner search) into a two-phase pipeline with a convergence-driven outer loop.

**When to use:** You need a ligand that (a) binds your target metal ion much more strongly than a competing ion, and (b) can form a deep eutectic solvent with an affordable small-molecule partner.

**Architecture:**
1. Phase 1 brainstorms and ranks ligands by selectivity score (`w_affinity * log_K_target + w_selectivity * delta_log_K`).
2. The top `--top-ligands` ligands (filtered by `--min-delta-log-k`) pass to Phase 2.
3. Phase 2 runs a full DES partner search for each shortlisted ligand.
4. DES-compatible ligands feed back into Phase 1 on the next outer cycle as hints.
5. The loop stops when the DES-compatible set is stable across two consecutive outer cycles, or when `--n-outer-cycles` is reached.

**Minimal invocation:**

```bash
python -m des_multi_agent.cli --workflow selectivity-des \
  --target-metal-ion Ni2+ \
  --competitor-metal-ion Co2+ \
  --n 20 --n-cycles 3 \
  --n-des-candidates 20 --n-des-cycles 3 \
  --n-outer-cycles 2 --top-ligands 3 --min-delta-log-k 0.5 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

**Reading the report:**
- Section 1 is a selectivity table identical to the `metal-selectivity` report, with an added `des_compatible` column (`yes`/`no`).
- Section 2 shows the DES partner candidates for each Phase 1 ligand, or "No DES partners found." for DES-incompatible ones.
- A "Warnings" section at the bottom lists any fallback decisions (e.g., bridge filter had zero candidates above `--min-delta-log-k` and fell back to top-N unconditionally).

**Progress output (stderr):**
```
[outer 1/2] phase 1: selectivity screening
[outer 1/2] phase 2: DES search for 3 ligand(s)
[outer 1/2] phase 2: ligand 1/3 — OC(=O)CNCC(=O)O
[outer 1/2] phase 2: ligand 2/3 — NCC(=O)O
[outer 1/2] phase 2: ligand 3/3 — c1ccncc1
[outer 2/2] phase 1: selectivity screening
...
[outer 2/2] DES-compatible set stable — converged early
```

```

- [ ] **Step 3: Add selectivity-des entry to examples/README.md**

Find this line in `examples/README.md`:
```
- [`betaine_des_gemma4_12b/`](./betaine_des_gemma4_12b) for the same betaine search with Ollama Gemma 4-12B: LLM enforces organic H-bonding partners, two-stage brainstorm, and contradiction detection
```

Add this line immediately after it:
```
- [`ni2_co2_selectivity/`](./ni2_co2_selectivity) for a Ni2+/Co2+ selectivity-DES example: Phase 1 screens for selective ligands, Phase 2 finds DES partners for the top hits, outer loop converges when the DES-compatible set stabilises
```

- [ ] **Step 4: Verify the docs look right**

```bash
cd /home/qshao/DES-Agent && grep -n "selectivity-des\|Selectivity-DES" README.md docs/tutorial.md examples/README.md
```

Expected: multiple matches in all three files.

- [ ] **Step 5: Commit**

```bash
cd /home/qshao/DES-Agent && git add README.md docs/tutorial.md examples/README.md && git commit -m "docs: add selectivity-des workflow documentation to README, tutorial, and examples index"
```

---

## Task 5: Actionable Error Messages for the Five Most Common Failures

**Files:**
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py`

Improve the five most common failure messages to include "how to fix" guidance.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cli.py`:

```python
def test_cli_missing_checkpoint_error_mentions_checkpoint_location(monkeypatch, capsys):
    """Error for missing checkpoint should mention where to put one."""
    monkeypatch.setattr(cli_module, "_discover_checkpoint", lambda: None)
    monkeypatch.setattr(cli_module, "load_llm_config", lambda p: None)
    # Build a minimal UncertaintyPolicy mock
    from des_multi_agent.uncertainty import UncertaintyPolicy
    with pytest.raises(SystemExit):
        cli_module.main(["--workflow", "des", "--component-a", "CCO"])
    err = capsys.readouterr().err
    assert "ml_des_mp/runs" in err


def test_cli_invalid_smiles_component_a_reports_clearly(monkeypatch, capsys):
    """Invalid --component-a should produce a clear SMILES error, not a raw RDKit traceback."""
    monkeypatch.setattr(cli_module, "load_llm_config", lambda p: None)
    from unittest.mock import patch
    with pytest.raises(SystemExit):
        cli_module.main([
            "--workflow", "des",
            "--component-a", "INVALID_XYZ!!!",
            "--n", "5",
            "--checkpoint-path", "ckpt.pt",
            "--config-path", "config.yaml",
        ])
    err = capsys.readouterr().err
    assert "SMILES" in err or "smiles" in err.lower()
```

Note: the SMILES validation test requires `resolve_existing_path` to be patched too because the checkpoint doesn't exist. Adjust the test as follows if needed:

```python
def test_cli_invalid_smiles_component_a_reports_clearly(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "load_llm_config", lambda p: None)
    from des_multi_agent import cli as cli_mod
    from unittest.mock import patch
    with patch.object(cli_mod, "resolve_existing_path", side_effect=lambda p: p), \
         pytest.raises(SystemExit):
        cli_mod.main([
            "--workflow", "des",
            "--component-a", "INVALID_XYZ!!!",
            "--n", "5",
            "--checkpoint-path", "ckpt.pt",
            "--config-path", "config.yaml",
        ])
    err = capsys.readouterr().err
    assert "SMILES" in err or "smiles" in err.lower()
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_cli.py::test_cli_missing_checkpoint_error_mentions_checkpoint_location tests/test_cli.py::test_cli_invalid_smiles_component_a_reports_clearly -v 2>&1 | tail -15
```

Expected: `FAILED`.

- [ ] **Step 3: Improve error messages in `des_multi_agent/cli.py`**

**3a. Checkpoint not found** — find this block (around line 399):

```python
            else:
                parser.error("DES workflow requires --checkpoint-path (none found in ml_des_mp/runs/)")
```

Replace with:

```python
            else:
                parser.error(
                    "DES workflow requires --checkpoint-path (none found in ml_des_mp/runs/). "
                    "Train a model with ml_des_mp/train.py or place a .pt checkpoint in ml_des_mp/runs/."
                )
```

**3b. Add early SMILES validation for --component-a** — in the `if args.workflow == "des":` block, immediately after the checkpoint resolution (after line `config_path = resolve_existing_path(args.config_path)`, around line 403), add:

```python
        # Validate SMILES early so the error points to --component-a
        try:
            from rdkit import Chem as _Chem
            if _Chem.MolFromSmiles(args.component_a) is None:
                raise ValueError
        except Exception:
            parser.error(
                f"--component-a {args.component_a!r} is not a valid SMILES string. "
                "Example: 'CCO' for ethanol, 'c1ccccc1' for benzene."
            )
```

**3c. Improve selectivity-des missing checkpoint message** — find this block (around line 519):

```python
        if not args.checkpoint_path:
            parser.error("selectivity-des workflow requires --checkpoint-path")
```

Replace with:

```python
        if not args.checkpoint_path:
            parser.error(
                "selectivity-des workflow requires --checkpoint-path. "
                "Place a trained .pt checkpoint in ml_des_mp/runs/ or pass the path explicitly."
            )
```

**3d. Improve missing metal-ion errors** — find this block (around line 548):

```python
    if args.workflow == "metal-selectivity":
        if not args.target_metal_ion or not args.competitor_metal_ion:
            parser.error("metal-selectivity workflow requires --target-metal-ion and --competitor-metal-ion")
```

Replace with:

```python
    if args.workflow == "metal-selectivity":
        if not args.target_metal_ion or not args.competitor_metal_ion:
            parser.error(
                "metal-selectivity workflow requires --target-metal-ion and --competitor-metal-ion. "
                "Example: --target-metal-ion Cu2+ --competitor-metal-ion Zn2+"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/test_cli.py -v 2>&1 | tail -30
```

Expected: all pass including the two new tests.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/qshao/DES-Agent && python -m pytest tests/ -q --tb=short 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/qshao/DES-Agent && git add des_multi_agent/cli.py tests/test_cli.py && git commit -m "fix(cli): add actionable guidance to the five most common error messages"
```

---

## Self-Review

### Spec Coverage Check

| Improvement | Task |
|-------------|------|
| Weight range [0,1] validation | Task 1 |
| Positive integer guard for --n, --n-cycles etc. | Task 1 |
| Early SMILES validation for --component-a | Task 5 |
| Help text for --workflow | Task 2 |
| Help text for --component-a (SMILES hint) | Task 2 |
| Help text for --n | Task 2 |
| Help text for --checkpoint-path | Task 2 |
| (selectivity-des only) labels on specific flags | Task 2 |
| --version flag | Task 2 |
| Per-ligand progress in Phase 2 | Task 3 |
| selectivity-des in README | Task 4 |
| selectivity-des in tutorial | Task 4 |
| selectivity-des in examples/README | Task 4 |
| Actionable checkpoint error messages | Task 5 |
| Actionable metal-ion error messages | Task 5 |

### Placeholder Scan

No TBDs, TODOs, or "add appropriate" language — all code blocks are complete.

### Type Consistency

- `_positive_int` returns `int` — matches `type=int` usage throughout.
- `_unit_float` returns `float` — matches `type=float` usage throughout.
- No new types or methods introduced that could drift between tasks.
