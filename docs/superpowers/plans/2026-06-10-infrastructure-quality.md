# DES-Agent Infrastructure & Code Quality Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the DES-Agent project by committing the stale working tree, adding a packaging manifest, wiring up CI, fixing inconsistent CLI error handling, renaming cryptic test files, and narrowing two silent `except Exception` fallbacks.

**Architecture:** Six independent tasks that can be executed in order. Tasks 1–3 are pure additions (no behavior change). Tasks 4–6 are targeted code changes each covered by a TDD cycle. No task depends on an unfinished predecessor once Task 1 (commit) is done.

**Tech Stack:** Python 3.11+, setuptools, GitHub Actions, pytest, argparse, RDKit, torch.

---

## File Map

| File | Change |
|------|--------|
| `pyproject.toml` | **NEW** — package metadata + pinned runtime deps |
| `.github/workflows/ci.yml` | **NEW** — pytest on push/PR |
| `des_multi_agent/cli.py` | Add `resolve_existing_path` + `try/except` to `selectivity-des` branch |
| `tests/test_selectivity_des_pipeline.py` | Add 1 test: FileNotFoundError → SystemExit |
| `des_multi_agent/predictors/stability_constants.py` | Narrow `except Exception: pass` to `(ValueError, TypeError, AttributeError)` |
| `des_multi_agent/predictors/designsolvents.py` | Same narrowing |
| `tests/test_h1_h2_h3_h4_h5.py` → `tests/test_llm_contradiction_detection.py` | `git mv` rename |
| `tests/test_h6.py` → `tests/test_llm_candidate_families.py` | `git mv` rename |
| `tests/test_b4_c6_c3_g4_b7.py` → `tests/test_llm_cache_format_export.py` | Name before commit |
| `tests/test_c1_x4_x3.py` → `tests/test_smiles_names_server_ensemble.py` | Name before commit |
| `tests/test_d1_d3_f1_d4_e1.py` → `tests/test_validation_dedup_batch_presets.py` | Name before commit |
| `tests/test_g1_b2_d2_e2_e4.py` → `tests/test_leaderboard_history_config.py` | Name before commit |

---

## Task 1: Commit the Uncommitted Working Tree

**Files:**
- All 6 untracked source modules and 5 untracked test files
- All 14 modified tracked files

This task has no new code to write. The goal is to verify the full suite passes and land everything in three logical commits.

- [ ] **Step 1: Verify the full suite passes**

```bash
cd /home/qshao/DES-Agent
pytest tests/ -q --tb=short
```

Expected: `441 passed` (or more). If any fail, fix them before proceeding.

- [ ] **Step 2: Rename the 5 untracked test files before staging**

```bash
mv tests/test_b4_c6_c3_g4_b7.py    tests/test_llm_cache_format_export.py
mv tests/test_c1_x4_x3.py           tests/test_smiles_names_server_ensemble.py
mv tests/test_d1_d3_f1_d4_e1.py     tests/test_validation_dedup_batch_presets.py
mv tests/test_g1_b2_d2_e2_e4.py     tests/test_leaderboard_history_config.py
```

(`test_improvements.py` already has a descriptive name — leave it.)

- [ ] **Step 3: Run the suite again to confirm renames didn't break anything**

```bash
pytest tests/ -q --tb=short
```

Expected: same count as Step 1.

- [ ] **Step 4: Commit new source modules**

```bash
git add \
  des_multi_agent/history.py \
  des_multi_agent/leaderboard.py \
  des_multi_agent/llm/cache.py \
  des_multi_agent/server.py \
  des_multi_agent/smiles_names.py \
  des_multi_agent/user_config.py
git commit -m "feat: add cache, leaderboard, history, server, smiles_names, user_config modules"
```

- [ ] **Step 5: Commit new and renamed test files**

```bash
git add \
  tests/test_llm_cache_format_export.py \
  tests/test_smiles_names_server_ensemble.py \
  tests/test_validation_dedup_batch_presets.py \
  tests/test_leaderboard_history_config.py \
  tests/test_improvements.py
git commit -m "test: add test suites for new modules (renamed from task-ID names)"
```

- [ ] **Step 6: Commit all modifications to tracked source files and their test updates**

```bash
git add \
  des_multi_agent/doctor.py \
  des_multi_agent/exporting.py \
  des_multi_agent/llm/errors.py \
  des_multi_agent/llm/transport.py \
  des_multi_agent/prediction.py \
  des_multi_agent/run_memory.py \
  des_multi_agent/task_executor.py \
  tests/test_cli.py \
  tests/test_demo_des_search.py \
  tests/test_exports.py \
  tests/test_task_execute.py \
  tests/test_uncertainty_reporting.py
git commit -m "feat: improvements to doctor, exporting, llm transport, prediction, run_memory, task_executor"
```

- [ ] **Step 7: Commit docs and examples**

```bash
git add \
  docs/improvement-analysis-2026-06-09.md \
  docs/tutorial.md \
  docs/superpowers/plans/2026-06-09-iteration-loop.md \
  examples/ni2_co2_selectivity/
git commit -m "docs: update tutorial, improvement analysis, iteration-loop plan; add ni2_co2 example"
```

---

## Task 2: Add `pyproject.toml`

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Verify the package is not yet installable**

```bash
pip install -e . 2>&1 | head -5
```

Expected: error about missing `pyproject.toml` or `setup.py`.

- [ ] **Step 2: Create `pyproject.toml`**

Create `/home/qshao/DES-Agent/pyproject.toml`:

```toml
[project]
name = "des-agent"
version = "0.1.0"
description = "Multi-agent DES screening and metal-ion binding prediction pipeline"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "rdkit>=2023.9",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "fastapi>=0.100",
    "uvicorn>=0.20",
    "httpx>=0.24",
    "transformers>=4.30",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "httpx2",
]

[project.scripts]
des-agent = "des_multi_agent.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[tool.setuptools.packages.find]
where = ["."]
include = ["des_multi_agent*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Install in editable mode and verify**

```bash
pip install -e .
python -c "from des_multi_agent import CandidateProposal; print('import OK')"
python -m des_multi_agent.cli --help > /dev/null && echo "CLI OK"
```

Expected: both lines print without error.

- [ ] **Step 4: Run the full suite to confirm nothing regressed**

```bash
pytest tests/ -q --tb=short
```

Expected: same pass count as Task 1.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml with runtime deps and dev extras"
```

---

## Task 3: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflows directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install CPU PyTorch
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu

      - name: Install package and dev extras
        run: pip install -e .[dev]

      - name: Run tests
        run: pytest tests/ -q --tb=short
```

Note: `torch` is installed separately with the CPU wheel before `pip install -e .` so that setuptools does not pull the larger default CUDA build.

- [ ] **Step 3: Validate the YAML syntax locally**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML OK"
```

Expected: `YAML OK`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow running pytest on push and PR"
```

---

## Task 4: Fix CLI Error Guarding for `selectivity-des`

**Files:**
- Modify: `des_multi_agent/cli.py` (lines ~514–539, the `selectivity-des` branch)
- Modify: `tests/test_selectivity_des_pipeline.py` (append one test)

The `selectivity-des` branch currently calls `run_selectivity_des_pipeline` without wrapping it in a `try/except` or resolving the checkpoint path. The `des` branch (lines 401–486) resolves paths with `resolve_existing_path` and catches `(FileNotFoundError, ValueError)`, converting them to clean `parser.error` messages. This task makes `selectivity-des` consistent with that pattern.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_selectivity_des_pipeline.py`:

```python
from unittest.mock import patch as _patch


def test_cli_selectivity_des_file_not_found_exits_cleanly(tmp_path):
    """A FileNotFoundError from the pipeline should become a clean SystemExit."""
    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_text("x")
    with _patch(
        "des_multi_agent.cli.run_selectivity_des_pipeline",
        side_effect=FileNotFoundError("checkpoint not found"),
    ):
        with pytest.raises(SystemExit):
            cli_main([
                "--workflow", "selectivity-des",
                "--target-metal-ion", "Cu2+",
                "--competitor-metal-ion", "Zn2+",
                "--checkpoint-path", str(fake_ckpt),
            ])
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_selectivity_des_pipeline.py::test_cli_selectivity_des_file_not_found_exits_cleanly -v
```

Expected: FAIL — the `FileNotFoundError` propagates as a raw exception instead of becoming `SystemExit`.

- [ ] **Step 3: Read the current `selectivity-des` branch in `cli.py`**

Before editing, read lines 514–540 of `des_multi_agent/cli.py` to see the exact current text.

- [ ] **Step 4: Add `resolve_existing_path` calls and `try/except` guard**

In `des_multi_agent/cli.py`, replace the `selectivity-des` routing block (starting at `if args.workflow == "selectivity-des":`) with:

```python
    if args.workflow == "selectivity-des":
        if not args.target_metal_ion:
            parser.error("selectivity-des workflow requires --target-metal-ion")
        if not args.competitor_metal_ion:
            parser.error("selectivity-des workflow requires --competitor-metal-ion")
        if not args.checkpoint_path:
            parser.error("selectivity-des workflow requires --checkpoint-path")
        checkpoint_path = resolve_existing_path(args.checkpoint_path)
        config_path = resolve_existing_path(args.config_path)
        try:
            pipeline_outcome = run_selectivity_des_pipeline(
                target_metal=args.target_metal_ion,
                competitor_metal=args.competitor_metal_ion,
                checkpoint_path=str(checkpoint_path),
                config_path=str(config_path),
                n_ligands=args.n,
                n_des_candidates=args.n_des_candidates,
                n_selectivity_cycles=args.n_cycles,
                n_des_cycles=args.n_des_cycles,
                n_outer_cycles=args.n_outer_cycles,
                min_delta_log_k=args.min_delta_log_k,
                top_ligands=args.top_ligands,
                w_affinity=args.affinity_weight,
                w_selectivity=args.selectivity_weight,
                stability_model_path=args.stability_constant_model_path,
                llm_cfg=llm_cfg,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(format_selectivity_des_report(pipeline_outcome))
        _print_summary("selectivity-des", pipeline_outcome)
        return
```

The only changes from before are:
1. `checkpoint_path = resolve_existing_path(args.checkpoint_path)` added
2. `config_path = resolve_existing_path(args.config_path)` added
3. `checkpoint_path=str(checkpoint_path)` and `config_path=str(config_path)` (now resolved Paths, not raw strings)
4. Entire `run_selectivity_des_pipeline(...)` call wrapped in `try/except (FileNotFoundError, ValueError)`

- [ ] **Step 5: Run the new test to verify it passes**

```bash
pytest tests/test_selectivity_des_pipeline.py::test_cli_selectivity_des_file_not_found_exits_cleanly -v
```

Expected: PASS

- [ ] **Step 6: Run the full pipeline test suite to confirm no regressions**

The existing routing test (`test_cli_selectivity_des_routes_to_pipeline`) passes a real `tmp_path` file, so `resolve_existing_path` will succeed. Verify:

```bash
pytest tests/test_selectivity_des_pipeline.py -v
```

Expected: all pass.

- [ ] **Step 7: Run the full suite**

```bash
pytest tests/ -q --tb=short
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add des_multi_agent/cli.py tests/test_selectivity_des_pipeline.py
git commit -m "fix(cli): resolve checkpoint path and guard errors in selectivity-des branch"
```

---

## Task 5: Rename Committed Cryptic Test Files

**Files:**
- Rename: `tests/test_h1_h2_h3_h4_h5.py` → `tests/test_llm_contradiction_detection.py`
- Rename: `tests/test_h6.py` → `tests/test_llm_candidate_families.py`

These two files are already tracked by git. Use `git mv` so git records them as renames (not delete+add).

- [ ] **Step 1: Rename with `git mv`**

```bash
git mv tests/test_h1_h2_h3_h4_h5.py tests/test_llm_contradiction_detection.py
git mv tests/test_h6.py              tests/test_llm_candidate_families.py
```

- [ ] **Step 2: Verify pytest still collects and runs them**

```bash
pytest tests/test_llm_contradiction_detection.py tests/test_llm_candidate_families.py -v --collect-only 2>&1 | tail -5
pytest tests/test_llm_contradiction_detection.py tests/test_llm_candidate_families.py -q
```

Expected: all tests collected and passing.

- [ ] **Step 3: Run the full suite**

```bash
pytest tests/ -q --tb=short
```

Expected: same pass count as before. pytest discovers files by `test_*.py` glob so no config change is needed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_llm_contradiction_detection.py tests/test_llm_candidate_families.py
git commit -m "refactor(tests): rename h1-h6 test files to subject-based names"
```

---

## Task 6: Narrow Silent `except Exception` Fallbacks

**Files:**
- Modify: `des_multi_agent/predictors/stability_constants.py` (line ~154)
- Modify: `des_multi_agent/predictors/designsolvents.py` (line ~106)

Both files use `except Exception: pass` as a fallback after calling `model.predict()`. The intent is: if the sklearn-style `.predict()` call fails, silently fall through to a linear predictor. The risk is that an unexpected error (e.g., memory error, import error, attribute error on a wrong object) is silently swallowed. Narrowing to the specific sklearn-compatible exception types preserves the fallback intent while letting truly unexpected errors propagate.

- [ ] **Step 1: Write the failing tests**

The silent `except Exception: pass` lives in the private helper `_predict_from_model(model, features)` in both files. Test it directly.

Create `tests/test_predictor_silent_excepts.py`:

```python
"""Verify that unexpected exceptions are NOT silently swallowed in predictors."""
from __future__ import annotations

import pytest

from des_multi_agent.predictors.stability_constants import _predict_from_model as _sc_predict
from des_multi_agent.predictors.designsolvents import _predict_from_model as _ds_predict


class _BadModel:
    """Model whose .predict() raises MemoryError — should NOT be swallowed."""
    def predict(self, X):
        raise MemoryError("out of memory")


# Minimal feature dict — key names don't matter, just needs to be a non-empty dict
# so `ordered = [features[key] for key in sorted(features)]` produces a list.
_FEATURES = {"a": 1.0, "b": 2.0}


def test_stability_constants_propagates_memory_error():
    with pytest.raises(MemoryError):
        _sc_predict(_BadModel(), _FEATURES)


def test_designsolvents_propagates_memory_error():
    with pytest.raises(MemoryError):
        _ds_predict(_BadModel(), _FEATURES)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_predictor_silent_excepts.py -v
```

Expected: FAIL — `MemoryError` is currently swallowed by `except Exception: pass` and silently falls through to `linear_predict(model, features)`, which then raises `TypeError` (because `_BadModel` is not a `dict`). The test expects `MemoryError` but gets `TypeError`, so it fails.

- [ ] **Step 3: Read the current except clauses**

```bash
sed -n '148,162p' des_multi_agent/predictors/stability_constants.py
sed -n '100,114p' des_multi_agent/predictors/designsolvents.py
```

Confirm both look like:
```python
        except Exception:
            pass
```

- [ ] **Step 4: Narrow the except clauses**

In `des_multi_agent/predictors/stability_constants.py`, replace:
```python
        except Exception:
            pass
```
with:
```python
        except (ValueError, TypeError, AttributeError):
            pass
```

In `des_multi_agent/predictors/designsolvents.py`, replace the identical pattern with:
```python
        except (ValueError, TypeError, AttributeError):
            pass
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
pytest tests/test_predictor_silent_excepts.py -v
```

Expected: PASS — `MemoryError` now propagates instead of being swallowed.

- [ ] **Step 6: Run the full suite**

```bash
pytest tests/ -q --tb=short
```

Expected: all pass. The predictor tests that use real sklearn models still pass because those models raise `ValueError`/`TypeError` on bad input, which are still caught.

- [ ] **Step 7: Commit**

```bash
git add \
  des_multi_agent/predictors/stability_constants.py \
  des_multi_agent/predictors/designsolvents.py \
  tests/test_predictor_silent_excepts.py
git commit -m "fix(predictors): narrow silent except to ValueError/TypeError/AttributeError"
```
