# DES Demo and Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a short, practical demo and tutorial that show how to run the DES multi-agent system in deterministic mode and with optional LLM brainstorming.

**Architecture:** Keep the demo thin and reuse the existing CLI as the source of truth. Add one small runnable example entrypoint for copy-paste execution and one short tutorial markdown file that explains the output, the required inputs, and the optional LLM configuration. Update the main README to point users to the tutorial instead of duplicating the walkthrough in multiple places.

**Tech Stack:** Python 3.13, argparse, Markdown, existing `des_multi_agent` CLI, existing `llm.example.yaml`

---

### Task 1: Add a runnable demo entrypoint

**Files:**
- Create: `examples/demo_des_search.py`
- Test: `tests/test_demo_des_search.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from examples.demo_des_search import build_parser, resolve_defaults


def test_demo_parser_accepts_overrides():
    parser = build_parser()
    args = parser.parse_args(["--component-a", "CCO", "--n", "3"])
    assert args.component_a == "CCO"
    assert args.n == 3


def test_demo_resolve_defaults_returns_repo_paths():
    checkpoint_path, config_path, llm_config_path = resolve_defaults()
    assert checkpoint_path.name.endswith(".pt")
    assert config_path.name == "config.yaml"
    assert llm_config_path.name == "llm.example.yaml"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_demo_des_search.py -v`
Expected: FAIL because `examples/demo_des_search.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import argparse
from pathlib import Path

from des_multi_agent.cli import load_llm_config
from des_multi_agent.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from des_multi_agent.orchestrator import run_search_report
from des_multi_agent.paths import resolve_existing_path
from des_multi_agent.reporting import format_report


DEFAULT_CHECKPOINT = PROJECT_ROOT / "ml_des_mp" / "runs" / "chemberta_random_row_fold01of05_best.pt"
DEFAULT_LLM_CONFIG = PROJECT_ROOT / "llm.example.yaml"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-a", default="CCO")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--llm-config", default=str(DEFAULT_LLM_CONFIG))
    return parser


def resolve_defaults():
    return (
        resolve_existing_path(DEFAULT_CHECKPOINT),
        resolve_existing_path(DEFAULT_CONFIG_PATH),
        resolve_existing_path(DEFAULT_LLM_CONFIG),
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint_path = resolve_existing_path(args.checkpoint_path)
    config_path = resolve_existing_path(args.config_path)
    llm_cfg = load_llm_config(args.llm_config) if args.llm_config else None
    outcome = run_search_report(
        component_a=args.component_a,
        n=args.n,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        llm_cfg=llm_cfg,
    )
    print(
        format_report(
            outcome.results,
            explanation_notes=outcome.explanation_notes,
            critique_notes=outcome.critique_notes,
            brainstorm_candidates=outcome.brainstorm_candidates,
            llm_warnings=outcome.llm_warnings,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_demo_des_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/demo_des_search.py tests/test_demo_des_search.py
git commit -m "feat: add runnable DES demo entrypoint"
```

### Task 2: Write a short tutorial

**Files:**
- Create: `docs/tutorial.md`
- Modify: `README.md`

- [ ] **Step 1: Write the tutorial content**

```markdown
# DES Multi-Agent Tutorial

## What this system does

The system takes a proposed component A, generates plausible partner candidates, predicts melting curves with the trained `ml_des_mp` model, and classifies each pair using both an absolute melting-point cutoff and a relative-drop criterion.

## What you need

- A working Python environment
- RDKit and the dependencies required by `ml_des_mp`
- A trained checkpoint from `ml_des_mp/runs/`

## Deterministic demo

Run:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5
```

This uses the shipped checkpoint and the bundled `ml_des_mp/config.yaml`.

## Optional LLM demo

If you want candidate brainstorming and explanation generation, point the demo at `llm.example.yaml` or your own LLM config:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --llm-config llm.example.yaml
```

## How to read the output

- `smiles_b` is the candidate partner
- `is_des` shows whether the pair passed the DES screen
- `min_tm_k` is the minimum predicted melting temperature across the ratio grid
- `rationale` explains why the candidate was classified the way it was

## Troubleshooting

- If the checkpoint path is wrong, the demo stops immediately with a file-not-found error.
- If the LLM config is enabled but invalid, the CLI shows a clear parser error.
- If `transformers` is missing, install the `ml_des_mp` dependencies before running the demo.
```

- [ ] **Step 2: Update the README to link the tutorial**

```markdown
## Demo and tutorial

See [`docs/tutorial.md`](/home/qshao/DES-Agent/docs/tutorial.md) for a short walkthrough and copy-paste demo commands.
```

- [ ] **Step 3: Add a README smoke check test**

```python
from pathlib import Path


def test_tutorial_and_readme_links_exist():
    assert Path("docs/tutorial.md").exists()
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/tutorial.md" in readme
```

- [ ] **Step 4: Run the doc test**

Run: `python -m pytest tests/test_demo_des_search.py -v`
Expected: PASS after the README link is added.

- [ ] **Step 5: Commit**

```bash
git add docs/tutorial.md README.md tests/test_demo_des_search.py
git commit -m "docs: add DES demo tutorial"
```

### Task 3: Verify the demo end to end

**Files:**
- Modify: `examples/demo_des_search.py` if any smoke issues appear
- Modify: `docs/tutorial.md` if usage details need tightening

- [ ] **Step 1: Run the demo with deterministic mode**

Run: `python -m examples.demo_des_search --component-a "CCO" --n 3 --llm-config ""`
Expected: prints a ranked DES summary using the bundled checkpoint and no LLM section.

- [ ] **Step 2: Run the demo with LLM mode**

Run: `python -m examples.demo_des_search --component-a "CCO" --n 3 --llm-config llm.example.yaml`
Expected: prints the ranked DES summary and, if a provider is configured, optional brainstorm/explanation/critique sections.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add examples/demo_des_search.py docs/tutorial.md README.md tests/test_demo_des_search.py
git commit -m "docs: add DES demo and tutorial"
```

