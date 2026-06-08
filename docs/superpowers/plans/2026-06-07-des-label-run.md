# DES Label-Run Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DES-only `label-run` command that updates `good` / `bad` labels in an existing `run.memory.json` file in place, so later DES runs can reuse those labels to nudge ranking only.

**Architecture:** Reuse the existing DES run-memory file as the only feedback store. A small helper in `run_memory.py` will validate and merge `SMILES=good|bad` label specs into the saved memory, preserving the existing workflow boundary and in-place file update behavior. The CLI stays thin: it resolves the run path, calls the helper, and prints a short confirmation. The existing reuse path continues to consume the same memory file, so active learning is just a controlled extension of the current offline memory mechanism.

**Tech Stack:** Python, argparse, json, pytest, existing `des_multi_agent` run-memory / ranking / CLI code.

---

### Task 1: Add run-memory label update helpers

**Files:**
- Modify: `des_multi_agent/run_memory.py`
- Test: `tests/test_run_memory.py`

- [ ] **Step 1: Write the failing test**

```python
from des_multi_agent.run_memory import parse_run_memory, update_run_memory_labels


def test_update_run_memory_labels_last_label_wins():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [{"smiles_b": "O", "label": "good"}],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )

    updated = update_run_memory_labels(
        memory,
        [("O", "good"), ("O", "bad"), ("CC(=O)O", "good")],
    )

    assert [label.smiles_b for label in updated.labels] == ["O", "CC(=O)O"]
    assert updated.labels[0].label == "bad"
    assert updated.labels[1].label == "good"


def test_update_run_memory_labels_rejects_unknown_smiles():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )

    with pytest.raises(ValueError, match="not found in the saved DES run"):
        update_run_memory_labels(memory, [("N", "good")])


def test_update_run_memory_labels_rejects_invalid_label():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )

    with pytest.raises(ValueError, match="label must be good or bad"):
        update_run_memory_labels(memory, [("O", "maybe")])


def test_update_run_memory_labels_changes_reuse_bias():
    memory = parse_run_memory(
        {
            "workflow": "des",
            "component_a": "CCO",
            "n": 20,
            "labels": [],
            "ranked_candidates": [
                {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
                {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            ],
        }
    )
    updated = update_run_memory_labels(memory, [("CC(=O)O", "good")])
    adjusted, notes = apply_run_memory_preferences(
        annotated_results=[
            _make_annotated_result("O", 0.60),
            _make_annotated_result("CC(=O)O", 0.70),
        ],
        memory=updated,
        component_a="CCO",
    )
    assert adjusted[0].result.curve.smiles_b == "CC(=O)O"
    assert notes == [
        "Applied reuse memory to 1 preferred candidate and 0 penalized candidates.",
        "Loaded 2 prior ranked candidates for ranking bias.",
    ]
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:
```bash
python -m pytest tests/test_run_memory.py::test_update_run_memory_labels_last_label_wins -q
```

Expected: fail because `update_run_memory_labels` does not exist yet.

- [ ] **Step 3: Implement the helper**

Add a helper in `des_multi_agent/run_memory.py` with this shape:

```python
def update_run_memory_labels(
    memory: RunMemory,
    label_specs: list[tuple[str, str]],
) -> RunMemory:
    ...
```

Implementation requirements:
- accept only `good` and `bad`
- validate each SMILES against `memory.ranked_candidates`
- preserve the existing DES workflow boundary
- last label for the same SMILES wins
- return a new `RunMemory` with updated `labels`

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:
```bash
python -m pytest tests/test_run_memory.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/run_memory.py tests/test_run_memory.py
git commit -m "feat: add run memory label update helper"
```

### Task 2: Add the `label-run` CLI command

**Files:**
- Create: `des_multi_agent/label_run.py`
- Modify: `des_multi_agent/cli.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_label_run.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py` parser coverage:

```python
def test_cli_parser_accepts_label_run_subcommand():
    parser = build_parser()
    args = parser.parse_args([
        "label-run",
        "--run",
        "runs/run_001",
        "--label",
        "O=good",
        "--label",
        "O=bad",
    ])
    assert args.command == "label-run"
    assert args.run == "runs/run_001"
    assert args.label == ["O=good", "O=bad"]
```

`tests/test_label_run.py` command behavior:

```python
from pathlib import Path

import pytest

from des_multi_agent.cli import main
from des_multi_agent.run_memory import load_run_memory


def test_label_run_updates_memory_in_place(tmp_path: Path, capsys):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.memory.json").write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 20,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""},
            {"smiles_b": "CC(=O)O", "rank": 2, "min_tm_k": 236.03, "trust_score": 0.83, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
        encoding="utf-8",
    )

    main(["label-run", "--run", str(run_dir), "--label", "O=good", "--label", "O=bad"])

    updated = load_run_memory(run_dir)
    assert [label.smiles_b for label in updated.labels] == ["O"]
    assert updated.labels[0].label == "bad"
    assert (run_dir / "run.memory.json").exists()
```

Add error-path tests in the same file:

```python
def test_label_run_rejects_unknown_smiles(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.memory.json").write_text(
        """{
          "workflow": "des",
          "component_a": "CCO",
          "n": 20,
          "labels": [],
          "ranked_candidates": [
            {"smiles_b": "O", "rank": 1, "min_tm_k": 208.69, "trust_score": 0.95, "uncertainty_flag": "low", "source": "heuristic", "source_id": ""}
          ]
        }""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        main(["label-run", "--run", str(run_dir), "--label", "N=good"])


def test_label_run_rejects_malformed_memory(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.memory.json").write_text("{not-json}", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["label-run", "--run", str(run_dir), "--label", "O=good"])
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:
```bash
python -m pytest tests/test_cli.py::test_cli_parser_accepts_label_run_subcommand tests/test_label_run.py -q
```

Expected: fail because `label-run` is not wired yet.

- [ ] **Step 3: Implement the command**

Add `des_multi_agent/label_run.py` with helper functions that:
- parse repeated `--label SMILES=good|bad` specs in order
- load the prior DES run memory via `load_run_memory()`
- call `update_run_memory_labels()`
- write the memory back in place with `write_run_memory()`
- return the updated `Path` and a short confirmation string

Update `des_multi_agent/cli.py` to add:

```python
label_run_parser = subparsers.add_parser("label-run", help="Update good/bad labels in a saved DES run memory")
label_run_parser.add_argument("--run", required=True, help="Prior DES run folder or run.memory.json file")
label_run_parser.add_argument("--label", action="append", default=[], help="Label spec in the form SMILES=good or SMILES=bad")
label_run_parser.set_defaults(command="label-run")
```

And in `main()`:

```python
if getattr(args, "command", None) == "label-run":
    try:
        message = run_label_command(args.run, args.label)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(message)
    return
```

Implementation requirements:
- reject malformed `SMILES=label` specs
- reject labels other than `good` and `bad`
- reject non-DES memory files
- keep the update in place
- preserve last-label-wins ordering

- [ ] **Step 4: Run the focused CLI tests and confirm they pass**

Run:
```bash
python -m pytest tests/test_cli.py tests/test_label_run.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add des_multi_agent/label_run.py des_multi_agent/cli.py tests/test_cli.py tests/test_label_run.py
git commit -m "feat: add label-run command for run memory feedback"
```

### Task 3: Update docs for the active-learning label loop

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`
- Modify: `examples/README.md`
- Modify: `examples/plain_language_gemma4_12b/README.md`
- Modify: `examples/lidocaine_gemma4_12b/README.md`
- Modify: `examples/plain_language_metal_binding_gemma4_12b/README.md`

- [ ] **Step 1: Write the documentation updates**

Add a short run-memory feedback section to the top-level docs that shows the full flow:

```bash
# Save a DES run memory
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --save-run-memory runs/run_001/run.memory.json

# Label the saved run in place
python -m des_multi_agent.cli label-run --run runs/run_001 --label "O=good" --label "CC(=O)O=bad"

# Reuse the labeled run on the next DES search
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --reuse-run runs/run_001
```

Also note explicitly:
- `label-run` is DES-only
- it updates `run.memory.json` in place
- labels affect ranking only
- the next run stays offline/local

- [ ] **Step 2: Run the docs-sensitive tests**

Run:
```bash
python -m pytest tests/test_cli.py tests/test_label_run.py tests/test_run_memory.py -q
```

Expected: pass.

- [ ] **Step 3: Refresh the example README files**

Add one short paragraph to the example README files that already mention run memory reuse, so users know the feedback loop is:
- save memory
- label memory
- reuse memory

Keep the existing example commands intact and only add the label-run snippet where it helps a new user copy the workflow.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/tutorial.md examples/README.md examples/plain_language_gemma4_12b/README.md examples/lidocaine_gemma4_12b/README.md examples/plain_language_metal_binding_gemma4_12b/README.md
git commit -m "docs: document DES label-run feedback loop"
```
