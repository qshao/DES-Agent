# DES-Agent

[![CI](https://github.com/qshao/DES-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/qshao/DES-Agent/actions/workflows/ci.yml)

This repository contains a deterministic DES screening pipeline plus optional layers for uncertainty, local discovery, LLM-assisted candidate brainstorming (two-stage, family-first), viscosity-aware composite ranking, multi-cycle iterative screening with convergence detection, LLM chemical contradiction detection, and a separate metal-binding workflow for stability-constant prediction. Additional capabilities include: molecule-name resolution (pass "ethanol" instead of "CCO" — the system resolves names to SMILES automatically); a deterministic chemistry grounding layer (source-side structural-fact injection into every LLM prompt plus output-side claim verification against RDKit-computed structure); protonation-aware metal-binding that computes dominant species at a user-specified pH before running stability-constant prediction; reality-anchored partner proposals that anchor LLM brainstorm to a registry of known real molecules, demoting or dropping implausible proposals before ranking; readable iteration trajectories that capture per-cycle shortlist changes, family reinforcement, and convergence reason as both a live console trace and a durable `trajectory.md` artifact; and a chemical-awareness layer that accumulates and leverages domain knowledge across cycles and runs — including H-bond complementarity ranking, near-miss analogue expansion, UCB-guided family exploration, adaptive transform selection, functional-group SAR tracking, and cross-run persistence of all learned signals.

## Quick Start

After cloning, install the package (requires Python ≥ 3.11):

```bash
pip install -e .
```

This registers a `des-agent` console script. The short form and the long form are equivalent:

```bash
des-agent --workflow des ...         # short form
python -m des_multi_agent.cli ...    # equivalent
```

Run the doctor check first to verify the local repo and example folders:

```bash
python -m des_multi_agent.cli doctor
```

List the metal ions with explicit stability-model identity features:

```bash
python -m des_multi_agent.cli supported-metals
```

If you want extra setup checks such as `doctor --check config`, add `--check` for the paths you care about most. `doctor --check llm --llm-config llm.example.yaml` also probes the configured local LLM service:

```bash
python -m des_multi_agent.cli doctor --check checkpoint --check discovery --check artifacts --check config
```

Save the three most-repeated flags as persistent defaults so you don't have to type them on every run:

```bash
python -m des_multi_agent.cli config set checkpoint_path=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
python -m des_multi_agent.cli config set config_path=ml_des_mp/config.yaml
python -m des_multi_agent.cli config set llm_config=llm.example.yaml   # optional, for LLM mode
```

**New to DES-Agent?** Start with [`docs/tutorial-wetlab.md`](docs/tutorial-wetlab.md) — a step-by-step guide written for chemists, covering DES screening, metal-binding prediction, and how to feed experimental results back into the next run.

For the full developer reference with all flags and options, see [`docs/tutorial.md`](docs/tutorial.md).
The quickest launch point for examples is [`examples/README.md`](examples/README.md).

Offline mock demo, recommended first:

```bash
./scripts/demo-mock.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --mock --component-a "ethanol" --n 5
```

> **Tip:** You can pass molecule names instead of SMILES. `--component-a "ethanol"` resolves to CCO automatically. `python -m des_multi_agent.cli list-molecules` prints all recognised names.

Real deterministic demo against the shipped checkpoint:

```bash
./scripts/demo-real.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --component-a "ethanol" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
```

Optional local discovery:

```bash
python -m examples.demo_des_search --component-a "ethanol" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --discovery-path /path/to/discovery
```

Save a DES run memory file for later reuse:

```bash
python -m des_multi_agent.cli --workflow des --component-a "ethanol" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --save-run-memory runs/run_001/run.memory.json
python -m des_multi_agent.cli view-run runs/run_001
```

Every DES run can also write into a standard flat run directory with `--output-dir runs/run_001`. That folder becomes the canonical home for `report.txt`, `run.json`, `run.csv`, and `run.manifest.json`. If you want run memory in the same folder, point `--save-run-memory` at `runs/run_001/run.memory.json`. If you later want to reuse all labeled runs in a history directory, point `--reuse-run` at the parent `runs/` folder.

Label the saved run in place with explicit SMILES and `good` / `bad` labels:

```bash
python -m des_multi_agent.cli label-run --run runs/run_001 --label "water=good" --label "acetic acid=bad"
```

Reuse the labeled DES memory file, folder, or a parent history directory of labeled runs to nudge ranking on a later run:

```bash
python -m des_multi_agent.cli --workflow des --component-a "ethanol" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --reuse-run runs/run_001/run.memory.json
```

Compare two saved runs from the same workflow with `compare-runs`:

```bash
python -m des_multi_agent.cli compare-runs runs/run_001/run.memory.json runs/run_002/run.memory.json
python -m des_multi_agent.cli compare-runs runs/run_001/run.memory.json runs/run_002/run.memory.json --json
```

Every command prints a compact `summary:` block after its main output. For parseable modes like `task-router` and `compare-runs --json`, the summary is written to `stderr` so `stdout` stays machine-readable.

Optional Ollama LLM run (Gemma, Nemotron, or Qwen via `model_name`). The LLM reviews candidates one by one and uses a two-stage brainstorm: it first selects chemical families (polyols, amides, etc.) then distributes candidates across them. It also detects chemical contradictions per candidate (`agree`/`conflict`/`uncertain`):

```bash
python -m examples.demo_des_search --component-a "ethanol" --n 20 --llm-config llm.example.yaml
```

Multi-cycle iterative screening — top hits from each cycle seed the next; stops when top-K converges. Pass `--output-dir` to also write a `trajectory.md` with the full cycle-by-cycle narrative:

```bash
python -m des_multi_agent.cli --workflow des --component-a "ethanol" --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml --n-cycles 3 --llm-config llm.example.yaml \
  --output-dir runs/ethanol_multicycle
```

Viscosity-aware composite ranking with threshold gate:

```bash
python -m des_multi_agent.cli --workflow des --component-a "ethanol" --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json \
  --viscosity-threshold 500 --viscosity-weight 0.4
```

Plain-language Gemma example that routes a request first and then runs the DES workflow:

```bash
./examples/plain_language_gemma4_12b/run.sh
```

Plain-language Gemma example for the metal-binding workflow:

```bash
./examples/plain_language_metal_binding_gemma4_12b/run.sh
```

DES run-memory feedback example:

```bash
./examples/des_run_memory_feedback/run.sh
```

DES viscosity example:

```bash
./examples/des_viscosity/run.sh
```

Metal-binding example:

```bash
./examples/metal_binding/run.sh
```

Validate paths and checkpoint before a real run (exits immediately, no predictions):

```bash
python -m des_multi_agent.cli --workflow des --component-a "ethanol" \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml --dry-run
```

Named threshold presets (`strict` / `standard` / `relaxed`) — no arithmetic required:

```bash
python -m des_multi_agent.cli --workflow des --component-a "ethanol" \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml --preset strict
```

Fold-ensemble predictions with per-candidate `ens_std` uncertainty:

```bash
python -m des_multi_agent.cli --workflow des --component-a "ethanol" \
  --ensemble --config-path ml_des_mp/config.yaml
```

Machine-readable output for scripting:

```bash
# JSON
python -m des_multi_agent.cli --workflow des --component-a "ethanol" \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --format json

# CSV
python -m des_multi_agent.cli --workflow des --component-a "ethanol" \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --format csv > results.csv
```


## Metal-Selectivity and Selectivity-DES Workflows

Screen ligands for selectivity between two competing metal ions:

```bash
python -m des_multi_agent.cli --workflow metal-selectivity \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion Zn2+ \
  --n 20 --n-cycles 3 \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

`--competitor-metal-ion` also accepts a comma-separated list of multiple off-target metals (`Zn2+,Fe3+,Ni2+`). Ranking becomes worst-case: a ligand only ranks well if it beats *every* off-target, not just one (`delta_log_k = log_k_target − max(log_k of all off-targets)`). The report gains an `off_target_breakdown` column showing every off-target's predicted log K per candidate, with an asterisk marking the limiting (worst-case) metal for that row. A single metal (no comma) behaves exactly as before:

```bash
python -m des_multi_agent.cli --workflow metal-selectivity \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion "Zn2+,Fe3+,Ni2+" \
  --n 20 --stability-constant-model-path artifacts/stability_constants/model.json
```

Add `--dft-validate` to refine the top candidates with a B3LYP-D3(BJ)/def2-SVP DFT single-point (requires `gpu4pyscf` and `xtb`). The LLM nominates 1–`--dft-top-n` candidates (default 3); the HOMO energy is used as an HSAB donor-softness proxy to apply a ±0.05 composite-score nudge. DFT failures are non-fatal — the run always completes with rule-based ranking as fallback. DFT now computes the dominant protonation state at `binding_pH` (default 7.0) rather than always assuming the neutral ligand, and every result is cached in `artifacts/dft_cache/dft_results.sqlite3` keyed on the computed species and method, so repeat candidates across cycles or runs skip recomputation:

```bash
python -m des_multi_agent.cli --workflow metal-selectivity \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion Zn2+ \
  --n 20 --stability-constant-model-path artifacts/stability_constants/model.json \
  --dft-validate --dft-top-n 3
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

All three iterative workflows (`des` with `--n-cycles > 1`, `metal-selectivity`, `selectivity-des`) print a compact per-cycle trajectory to stderr after the run and write `trajectory.md` into `--output-dir` when set. See [`examples/multi_cycle_des/trajectory.md`](/home/qshao/DES-Agent/examples/multi_cycle_des/trajectory.md) for a captured example.

## Task Router

Use the task router to turn a plain-language request into a JSON job without running the workflow:

```bash
python -m des_multi_agent.cli task-router "find DES partners for lidocaine"
```

Use `task-execute` when you want the router to translate the request and then run the workflow immediately:

```bash
python -m des_multi_agent.cli task-execute "find DES partners for lidocaine"
```

The router loads `llm.example.yaml` by default, supports both `des` and `metal-binding`, and normalizes common compound names before returning either a complete job or clarification questions with `workflow=clarify`, as JSON only. It will ask follow-up questions when a request is ambiguous, including free base versus salt-form questions. For a worked example, see [`examples/task_router/`](/home/qshao/DES-Agent/examples/task_router/).

## REST API Server

A thin FastAPI wrapper ships alongside the CLI for notebook and programmatic consumers:

```bash
python -m des_multi_agent.server                     # http://0.0.0.0:8000
python -m des_multi_agent.server --host 127.0.0.1 --port 9000
```

Endpoints: `GET /health` · `POST /search` (DES screening) · `POST /metal-binding`. Interactive docs at `http://localhost:8000/docs`. See Section 12 of `docs/tutorial.md` for request/response examples.

## Project Layout

- `des_multi_agent/` contains the screening orchestration code
- `ml_des_mp/` contains the trained model and the underlying predictor
- `docs/tutorial.md` is the short user guide for the demo
- `examples/des_viscosity/` is an offline DES viscosity example
- `examples/viscosity_template/` is a template-style DES viscosity example you can adapt
- `examples/metal_binding/` is an offline metal-binding example for stability constants
- `examples/ligand_binding_template/` is a template-style metal-binding example you can adapt
- `examples/lidocaine_gemma4_12b/` is a real lidocaine DES example with Gemma 4-12B
- `examples/plain_language_gemma4_12b/` is a plain-language DES example routed through Gemma 4-12B
- `examples/plain_language_metal_binding_gemma4_12b/` is a plain-language metal-binding example routed through Gemma 4-12B
- `examples/metal_selectivity_standalone/` is the metal-selectivity workflow used independently (no DES phase)
- `examples/preset_thresholds/` demonstrates `--preset strict` vs `relaxed` side by side
- `examples/ensemble_prediction/` demonstrates `--ensemble` fold-ensemble with `ens_std` per candidate
- `examples/uncertainty_controls/` shows all three `--uncertainty-mode` policies compared
- `examples/output_formats/` shows `--format table/json/csv/prose` on the same query
- `examples/dry_run/` shows `--dry-run` for path and checkpoint validation without running predictions
- `llm.example.yaml` is a ready-to-edit optional LLM config
- `docs/future-improvements.md` tracks the next planned extensions
- `tests/test_benchmarks_examples.py` is the example benchmark suite that compares captured outputs against frozen baselines
- `des_multi_agent/trajectory.py` — workflow-agnostic trajectory model (`TopEntry`/`CycleSnapshot`/`SearchTrajectory`), Markdown + console renderers, and atomic `trajectory.md` writer
- `des_multi_agent/server.py` — thin FastAPI wrapper; start with `python -m des_multi_agent.server`; exposes `GET /health`, `POST /search`, `POST /metal-binding`
- `des_multi_agent/chemistry/partner_registry.py` — known-compound registry and anchor menu for reality-anchored partner proposals
- `des_multi_agent/chemistry/claim_grounding.py` — deterministic chemistry grounding (structural facts, claim verdicts, partner reality grading)
- `artifacts/molecule_names/common_names.json` — curated molecule-name → SMILES mapping used by name resolution and the partner menu

## Uncertainty Controls

The CLI lets you tune how uncertainty affects filtering and ranking with `--uncertainty-mode`:

```bash
python -m des_multi_agent.cli --workflow des --component-a "ethanol" --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --uncertainty-mode filter --min-trust-score 0.70 --soft-penalty-weight 0.20
```

The default mode is `penalize`. Use `report_only` to inspect trust columns without changing ranking, or `filter` to remove low-trust candidates entirely. For all three modes compared on the same query, see [`examples/uncertainty_controls/`](/home/qshao/DES-Agent/examples/uncertainty_controls).

## Chemistry Grounding and Reality Anchoring

When an LLM is configured, the pipeline runs two deterministic grounding passes that are LLM-agnostic (identical results regardless of backend):

**Source-side fact injection:** Structural facts (H-bond profile, coordination profile) are computed for component A and injected into every brainstorm and review prompt so the LLM reasons over computed data rather than memory.

**Output-side claim verification:** Every LLM claim is checked against RDKit-computed structure:
- Family labels (polyol, amide, etc.) are verified with SMARTS.
- DES H-bond complementarity is verified with `des_hbond_complementarity`.
- Contradicted claims are flagged in the report (`✗ contradicted — <correction>`); the candidate takes a −0.25 ranking penalty.

**Reality-anchored partner proposals:** Before each brainstorm, a menu of up to 30 known, real partners (filtered to the complementary H-bond role) is injected into the prompt. After brainstorm, every LLM proposal is graded:
- `✓ known` — matched in the real-compound registry → kept
- `◆ novel (plausible)` — structurally sane, H-bond-complementary → kept
- `✗ implausible` — no H-bond fit → demoted (−0.25); structurally invalid → dropped

No extra flags needed — grounding and reality anchoring activate automatically under `--llm-config`.
