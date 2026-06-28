# DES-Agent Comprehensive Tutorial

DES-Agent is a local, scriptable workflow for screening deep eutectic solvent (DES) candidates and related metal-binding ligands. It combines deterministic prediction, optional LLM assistance, offline property models, run-memory feedback, and example-driven regression tests.

Use this tutorial as the main user guide. The top-level `README.md` is the quick-start page; this file explains how the pieces fit together and which command to use for each task.

## 1. What The System Does

The project has four main workflow families:

| Workflow | Purpose | Main command |
|----------|---------|--------------|
| DES screening | Find candidate DES partners for a component A SMILES | `--workflow des` |
| DES viscosity | Add viscosity prediction and viscosity-aware ranking to DES screening | `--viscosity-model-path` |
| Metal binding | Predict or screen ligand-metal stability constants (`log K`) | `--workflow metal-binding` |
| Metal selectivity | Rank ligands by target-metal affinity and selectivity over a competitor | `--workflow metal-selectivity` |
| Selectivity-DES | Screen selective ligands, then search DES partners for the best ligands | `--workflow selectivity-des` |

The core DES melting-temperature prediction is deterministic and uses the trained `ml_des_mp` checkpoint. LLM mode is optional. When enabled, the LLM proposes candidates, reviews candidates one by one, explains outputs, and flags chemical contradictions, but the ML model still makes the final DES melting-temperature prediction.

## 2. Installation

**Requirements:** Python ≥ 3.11, and the standard scientific stack (PyTorch, RDKit, etc. — listed in `pyproject.toml`).

After cloning the repository, install the package and its dependencies:

```bash
pip install -e .
```

The `-e` flag installs in editable mode so local changes take effect immediately without reinstalling. It also registers the `des-agent` console script, so you can use either form interchangeably:

```bash
des-agent --workflow des ...            # short form (after pip install -e .)
python -m des_multi_agent.cli ...       # equivalent long form
```

If you plan to run the test suite, also install the dev extras:

```bash
pip install -e ".[dev]"
```

For Ollama-backed LLM workflows you will also need [Ollama](https://ollama.com) running locally with your chosen model pulled:

```bash
ollama pull gemma4:12b   # or nemotron-3-nano:latest, qwen3.6, etc.
```

Confirm the model name in your `llm.example.yaml` matches what you pulled. OpenAI and Gemini providers are also supported and do not require Ollama — see Section 10 for all provider options.

## 3. Setup And Health Checks

Run all commands from the repository root unless an example README says otherwise.

Start with the default setup check:

```bash
python -m des_multi_agent.cli doctor
```

Add optional checks for files and local services:

```bash
python -m des_multi_agent.cli doctor --check checkpoint
python -m des_multi_agent.cli doctor --check discovery
python -m des_multi_agent.cli doctor --check artifacts
python -m des_multi_agent.cli doctor --check config
python -m des_multi_agent.cli doctor --check llm --llm-config llm.example.yaml
```

The `checkpoint`, `discovery`, `artifacts`, and `config` checks are local file/config checks. The `llm` check probes the configured local LLM service, so use it only when you expect Ollama or another configured provider to be running.

You can also list metal ions with explicit identity features:

```bash
python -m des_multi_agent.cli supported-metals
```

Unsupported ions can still run in the fallback feature path, but selectivity between two unsupported ions is less meaningful.

### Persistent User Config

To avoid repeating `--checkpoint-path`, `--config-path`, and `--llm-config` on every command, save them as persistent defaults:

```bash
python -m des_multi_agent.cli config set checkpoint_path=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
python -m des_multi_agent.cli config set config_path=ml_des_mp/config.yaml
python -m des_multi_agent.cli config set llm_config=llm.example.yaml
```

These three keys (`checkpoint_path`, `config_path`, `llm_config`) are the only accepted keys. Saved values act as defaults for every subsequent run; an explicit flag on the command line always overrides the saved value. The config is stored in `~/.des-agent/config.yaml` by default (override with `DES_AGENT_CONFIG`).

## 4. First Runs

### Offline Mock Demo

Use this first if you only want to confirm the repo works:

```bash
./scripts/demo-mock.sh
```

Equivalent direct command:

```bash
python -m examples.demo_des_search --mock --component-a "CCO" --n 5
```

This path does not require a trained checkpoint or an LLM service.

### Real Deterministic DES Demo

Run a real DES prediction with the shipped checkpoint:

```bash
python -m examples.demo_des_search \
  --component-a "CCO" \
  --n 5 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
```

The wrapper script uses the same idea:

```bash
./scripts/demo-real.sh
```

You can also pass the checkpoint through the wrapper environment variable:

```bash
DES_CHECKPOINT_PATH=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt ./scripts/demo-real.sh
```

### Molecule Names

Pass a common name instead of a SMILES string wherever `--component-a` is accepted:

```bash
# These are equivalent:
python -m des_multi_agent.cli --workflow des --component-a "ethanol" ...
python -m des_multi_agent.cli --workflow des --component-a "CCO" ...
```

List all supported names:

```bash
python -m des_multi_agent.cli list-molecules
```

Name resolution is case-insensitive and supports common aliases (`ChCl`, `betaine`, `urea`, etc.). If a name is not found, the input is treated as a SMILES string as before.

### Standard Run Directory

For real work, write outputs into a dedicated run folder:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --output-dir runs/run_001 \
  --save-run-memory runs/run_001/run.memory.json
```

The flat run directory contains:

| File | Purpose |
|------|---------|
| `report.txt` | Canonical human-readable report |
| `run.json` | Structured run result |
| `run.csv` | Flat ranked table for spreadsheets |
| `run.manifest.json` | Metadata and artifact filenames |
| `run.memory.json` | Optional compact memory for feedback/reuse |

Inspect a saved run directory:

```bash
python -m des_multi_agent.cli view-run runs/run_001
```

## 5. DES Screening Workflow

### Required Inputs

A normal DES run needs:

- `--workflow des`
- `--component-a` as a SMILES string
- `--checkpoint-path` unless checkpoint auto-discovery finds one
- `--config-path`, normally `ml_des_mp/config.yaml`
- `--n`, the number of candidate partners to screen

Add `--ml-device cpu` to force CPU inference on machines without a GPU, or `--ml-device cuda` to make the selection explicit. The default is taken from the model config.

Minimal command:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml
```

### Candidate Sources

By default, the system uses built-in candidate generation. You can inject literature compounds or a curated library with `--discovery-path`:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --discovery-path my_discovery/
```

The discovery directory must contain one or both of these files:

**`literature.yaml`** — known component-A / component-B pairs from the literature:

```yaml
- component_a: CCO
  component_b: OCCO
  source: local_literature
  note: Known diol partner for ethanol-like donors.
  reference_id: LIT-001
```

**`library.yaml`** — a curated candidate library (screened against any component A):

```yaml
- smiles: OCCO
  family: diol
  source: curated_library
  note: Small diol candidate.
- smiles: OCC(O)CO
  family: polyol
  source: curated_library
  note: Local polyol candidate.
```

Both files are optional; the discovery directory may contain either or both. SMILES in both files are validated and skipped if unparseable. See `tests/fixtures/discovery/` for a working example.

You can also supply candidates directly:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --candidates-file examples/candidates_file/candidates.smiles
```

See [examples/candidates_file/](../examples/candidates_file).

### Proposal Diversity Controls

Use these flags when you want the brainstormed proposal set to stay broader or more focused:

```bash
python -m des_multi_agent.cli   --workflow des   --component-a "CCO"   --n 20   --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt   --config-path ml_des_mp/config.yaml   --proposal-diversity-mode balanced   --proposal-max-similarity 0.85   --proposal-per-family-budget 1
```

| Flag | Meaning |
|------|---------|
| `--proposal-diversity-mode` | `explore`, `balanced`, or `exploit`; controls how strongly the search spreads across chemical families |
| `--proposal-max-similarity` | Suppress near-duplicate proposals above this fingerprint similarity cutoff |
| `--proposal-per-family-budget` | Limit how many accepted proposals can come from the same family |

Recommended starting point:

- `balanced` for most screening runs
- `explore` when the search is collapsing onto one chemical family too early
- `exploit` when you already know the family you want and only need close analogs

### Threshold Presets

Use presets when you do not want to tune thresholds manually:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --preset strict
```

| Preset | Tm ceiling | Minimum relative drop |
|--------|------------|-----------------------|
| `strict` | 240 K | 15% |
| `standard` | 260 K | 10% |
| `relaxed` | 280 K | 5% |

Override manually with `--abs-tm-threshold` and `--rel-drop-min`.

See [examples/preset_thresholds/](../examples/preset_thresholds).

### Uncertainty Modes

The uncertainty layer annotates trust and can optionally affect ranking or filtering:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --uncertainty-mode penalize \
  --min-trust-score 0.55 \
  --soft-penalty-weight 0.35
```

Modes:

| Mode | Behavior |
|------|----------|
| `report_only` | Show uncertainty but do not alter ranking |
| `penalize` | Softly lower low-trust candidates |
| `filter` | Remove candidates below the trust threshold |

The trust score is derived from the ensemble's fold disagreement (standard deviation of predicted Tm across folds, in K). Two thresholds divide the range into high/medium/low confidence bands; adjust them when the defaults are too strict or too loose for your data distribution:

| Flag | Default (K) | Meaning |
|------|-------------|---------|
| `--std-medium-threshold-k` | 5.0 | `ens_std` above this → medium confidence |
| `--std-high-threshold-k` | 15.0 | `ens_std` above this → low confidence |

Candidates with `ens_std` above `--std-high-threshold-k` are flagged as low confidence; those between the two thresholds are medium confidence.

See [examples/uncertainty_controls/](../examples/uncertainty_controls).

### Ensemble Prediction

Use all fold checkpoints in `ml_des_mp/runs/`:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --ensemble \
  --config-path ml_des_mp/config.yaml
```

The report adds `ens_std` to show fold disagreement. Higher `ens_std` means the model ensemble disagrees more.

See [examples/ensemble_prediction/](../examples/ensemble_prediction).

### Output Formats

The DES report can be printed as a table, JSON, CSV, or prose:

```bash
python -m des_multi_agent.cli --workflow des --component-a "CCO" \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --format json

python -m des_multi_agent.cli --workflow des --component-a "CCO" \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --format csv > results.csv
```

The `summary:` block goes to stderr for parseable modes so stdout stays machine-readable.

See [examples/output_formats/](../examples/output_formats).

## 6. Viscosity-Aware DES Ranking

Use the bundled DESignSolvents-style artifact to add viscosity predictions:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json
```

Add threshold gating and composite ranking:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json \
  --viscosity-threshold 500 \
  --viscosity-weight 0.4
```

`--viscosity-threshold` is in cP. Candidates above the threshold are ranked below passing candidates. `--viscosity-weight` is in `[0, 1]` and controls how much viscosity contributes to composite ranking.

See [examples/des_viscosity/](../examples/des_viscosity), [examples/viscosity_template/](../examples/viscosity_template), and [examples/viscosity_composite_ranking/](../examples/viscosity_composite_ranking).

## 7. Multi-Cycle Screening

Use `--n-cycles` when you want iterative search. Top hits from each cycle seed the next cycle, and the loop stops early if the top candidates stabilize.

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --n-cycles 3
```

With `--output-dir runs/run_001`, multi-cycle runs write cycle folders such as `cycle_01/`, `cycle_02/`, and `cycle_03/`.

### Iteration trajectory

Any run with `--n-cycles > 1` (and the metal-selectivity / selectivity-DES
workflows) now prints a compact per-cycle **trajectory** to stderr: how many
candidates were screened and hit each cycle, which entered or left the
shortlist, which chemical families were reinforced, and whether the search
converged. Pass `--output-dir DIR` to also write a durable, readable
`DIR/trajectory.md` with the full cycle-by-cycle narrative and the final
shortlist.

See [examples/multi_cycle_des/](../examples/multi_cycle_des).

## 8. Run Memory, Labels, And Reuse

Save a compact run-memory file:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --output-dir runs/run_001 \
  --save-run-memory runs/run_001/run.memory.json
```

Label saved candidates by explicit SMILES:

```bash
python -m des_multi_agent.cli label-run \
  --run runs/run_001 \
  --label "O=good" \
  --label "CC(=O)O=bad"
```

Reuse one run:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --reuse-run runs/run_001
```

Reuse a whole labeled history directory:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --reuse-run runs/
```

Reuse only nudges ranking. It does not change the underlying predictor or filter candidates out automatically.

See [examples/des_run_memory_feedback/](../examples/des_run_memory_feedback).

## 9. Comparing And Reviewing Runs

Compare two saved DES runs:

```bash
python -m des_multi_agent.cli compare-runs runs/run_001 runs/run_002
```

Add JSON output for automation:

```bash
python -m des_multi_agent.cli compare-runs runs/run_001 runs/run_002 --json
```

Summarize a history directory:

```bash
python -m des_multi_agent.cli history runs/
```

Build a cross-run compound leaderboard:

```bash
python -m des_multi_agent.cli leaderboard runs/
```

Inspect one run directory:

```bash
python -m des_multi_agent.cli view-run runs/run_001
```

See [examples/compare_runs/](../examples/compare_runs) and [examples/leaderboard_history/](../examples/leaderboard_history).

## 10. Optional LLM Workflows

LLM mode is optional. Configure it with an LLM YAML file. `llm.example.yaml` is the ready-to-edit starting point; the repo also ships three model-specific configs you can copy directly:

- `llm.ni_co_selectivity.yaml` — Ollama/Qwen for selectivity-DES
- `llm.ni_co_qwen36.yaml` — Qwen 3.6 variant
- `llm.ni_co_nemotron.yaml` — Nemotron 3 Nano variant

Four providers are supported. Choose one and fill in its required fields:

**Ollama (local)** — requires `api_base_url`; supported models: `gemma4:12b`, `nemotron-3-nano:latest`, `qwen3.6`:

```yaml
llm:
  enabled: true
  provider: ollama
  model_name: gemma4:12b
  api_base_url: http://localhost:11434
```

**OpenAI** — requires `api_key_env` (the name of the env var holding your API key):

```yaml
llm:
  enabled: true
  provider: openai
  model_name: gpt-4o
  api_key_env: OPENAI_API_KEY    # set this env var before running
```

**Gemini** — same pattern as OpenAI:

```yaml
llm:
  enabled: true
  provider: gemini
  model_name: gemini-1.5-pro
  api_key_env: GEMINI_API_KEY
```

**Custom HTTP** — any OpenAI-compatible endpoint (LM Studio, vLLM, etc.); requires `api_base_url`:

```yaml
llm:
  enabled: true
  provider: custom_http
  model_name: my-local-model
  api_base_url: http://localhost:8080
```

All optional fields with their defaults:

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `false` | Must be `true` for LLM calls to fire |
| `max_candidates` | `20` | Max candidates the LLM reviews per cycle |
| `max_tokens` | `512` | Token budget per LLM call |
| `temperature` | `0.2` | Generation temperature |
| `timeout_seconds` | `30.0` | Per-call HTTP timeout |
| `diversity_mode` | `balanced` | `explore` / `balanced` / `exploit` |
| `max_families` | `6` | Cap on chemical families in brainstorm |
| `family_bias_strength` | `0.5` | 0–1; weight of prior productive families |

Run a DES search with LLM support:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "CCO" \
  --n 20 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --llm-config llm.example.yaml
```

When enabled, the LLM can:

- choose chemical families before candidate generation
- propose candidates across those families
- review candidates one by one
- generate explanations and critiques
- flag chemical contradictions as `agree`, `conflict`, or `uncertain`
- act as a chemistry advisor that adds short rationales, warnings, and hybrid next-step suggestions

Use `diversity_mode` to control how broad the brainstorm should be:

- `explore` prefers chemically distinct families
- `balanced` keeps a mix of productive and novel families
- `exploit` concentrates on families that worked well in earlier cycles

`max_families` caps how many families the first brainstorm stage should return, and `family_bias_strength` controls how strongly prior productive families influence later cycles. These settings affect brainstorming only; they do not change the deterministic scorer or the final ranking rules.

The chemistry advisor layer reuses the same LLM provider but focuses on candidate rationales, warnings, and next-step suggestions. It uses prior run memory only as a soft prior, so current-run evidence still controls the final decision.

### Chemical Pattern Memory

Chemical pattern memory turns prior predictions into compact chemistry lessons for the next cycle. It uses both current-cycle outcomes and saved run memory to bias proposal generation and ranking without hard-coding a family lock-in.

Use these flags to control it:

| Flag | Meaning |
|------|---------|
| `--chemical-pattern-memory` | `off`, `soft`, or `adaptive`; controls how strongly prior chemical patterns influence the next DES cycle |
| `--pattern-memory-max-examples` | Maximum representative good/bad examples to include in the prompt context |

Recommended starting point:

- `adaptive` if you want the next cycle to learn from prior hits and failures
- `soft` if you want the memory layer to stay present but conservative
- `off` if you want the run to ignore prior chemical lessons entirely

The pattern-memory layer is separate from proposal diversity: proposal diversity controls how broad the brainstorm is, while pattern memory controls how prior chemical lessons bias the next cycle and the report narrative. The same flags also apply to `selectivity-des`, because its DES phase uses the same multi-cycle DES search loop.

### Chemistry Lesson Summary

The chemistry lesson summary is the short chemistry note that appears in the report and is reused by the next cycle. It is built from the current cycle's results plus any saved run memory, and it turns the run into a compact lesson such as:

- which families or motifs looked productive
- which families or motifs should be avoided
- which next step is conservative versus exploratory
- which failure modes to watch for

It is not a separate user-facing control. When there is enough evidence, it is produced automatically and fed back into the next cycle's prompt context.

The supported example model configs include Gemma 4-12B, Nemotron 3 Nano, and Qwen 3.6. The refreshed chemistry-lesson-summary captures are shown in Gemma 4-12B and lidocaine; betaine remains the frozen baseline example.

See:

- [examples/gemma4_12b/](../examples/gemma4_12b)
- [examples/nemotron_3_nano/](../examples/nemotron_3_nano)
- [examples/qwen3_6/](../examples/qwen3_6)
- [examples/lidocaine_gemma4_12b/](../examples/lidocaine_gemma4_12b)
- [examples/betaine_des_gemma4_12b/](../examples/betaine_des_gemma4_12b)

### Chemistry Grounding Layer

When an LLM is enabled, every coordination claim, selectivity direction, family
label, and DES plausibility assertion is automatically verified against the
molecular structure using deterministic chemistry tools. The grounding layer is
LLM-agnostic — identical verdicts regardless of which model is configured.

**Report output:**
- `✓ verified` — the claim is consistent with the computed structure
- `✗ contradicted — <correction>` — the claim conflicts with structure; the
  candidate is demoted in the ranking by a fixed −0.25 penalty
- Unverifiable claims (e.g. unknown family labels) are left unmarked

**Source-side fact injection:** Before each LLM call, computed structural
facts (HBD, HBA, denticity, donor elements, family tags) are injected into the
prompt as a `Computed facts:` block so the model reasons over verified data
rather than recalled chemistry.

No new flags are needed — grounding runs automatically whenever `--llm-config`
is set.

### Reality-Anchored Partner Proposals

When an LLM is enabled, every proposed DES partner is evaluated against a
registry of known, real molecules before it reaches the ranking step.

**Source-side — anchor menu:** Before the brainstorm call, a menu of up to
30 known HBA/HBD partners (matched to the role complementary to component A)
is injected into the prompt. The menu draws from a curated registry of 57
hand-tagged molecules plus hundreds of experimentally-characterised compounds
auto-role-tagged from the shipped melting-point dataset. The LLM is instructed
to prefer these but may propose off-menu candidates with explicit justification.

**Output-side — partner verdict:** Every LLM proposal is graded
deterministically:

| Verdict | Meaning | Action |
|---------|---------|--------|
| `✓ known` | Canonical InChIKey matches the real-compound registry | Keep |
| `◆ novel (plausible)` | Off-registry; structurally sane; H-bond-complements component A | Keep, flagged |
| `✗ implausible` — no complementarity | Real or invented molecule, but no H-bond fit with component A | Demote −0.25 |
| `✗ implausible` — bad structure | Disallowed element / MW out of range / radical | Drop from results |

Verdicts appear in the Grounding section of the report, alongside the
family and plausibility verdicts from the Chemistry Grounding Layer above.
No new flags are needed — the layer runs automatically whenever `--llm-config`
is set.

## 11. Plain-Language Routing

Use `task-router` to convert a plain-language request into JSON without running anything:

```bash
python -m des_multi_agent.cli task-router "find DES partners for lidocaine"
```

Use `task-execute` to route and run in one step:

```bash
python -m des_multi_agent.cli task-execute "find DES partners for lidocaine"
```

The router supports DES and metal-binding workflows. It normalizes common compound names and asks clarification questions when important inputs are ambiguous, including salt-form versus free-base ambiguity. Ambiguous requests return JSON with `workflow=clarify` and clarification questions instead of invented inputs.

See:

- [examples/task_router/](../examples/task_router)
- [examples/task_execute/](../examples/task_execute)
- [examples/plain_language_gemma4_12b/](../examples/plain_language_gemma4_12b)
- [examples/plain_language_metal_binding_gemma4_12b/](../examples/plain_language_metal_binding_gemma4_12b)

## 12. REST API Server

A thin FastAPI wrapper (`des_multi_agent/server.py`) exposes DES screening and metal-binding as HTTP endpoints. Start it with:

```bash
python -m des_multi_agent.server                        # default: http://0.0.0.0:8000
python -m des_multi_agent.server --host 127.0.0.1 --port 9000
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/search` | DES screening (wraps `run_search_report`) |
| `POST` | `/metal-binding` | Metal-ion binding prediction (wraps `run_metal_binding_workflow`) |

Example DES search via curl:

```bash
curl -s http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "component_a": "CCO",
    "n": 10,
    "checkpoint_path": "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
    "config_path": "ml_des_mp/config.yaml"
  }'
```

The server is also importable for embedding in notebooks or other services:

```python
from des_multi_agent.server import app
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8000)
```

Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs` when the server is running.

## 13. Metal-Binding Workflow

Predict a stability constant for one metal-ligand pair:

```bash
python -m des_multi_agent.cli \
  --workflow metal-binding \
  --metal-ion Cu2+ \
  --ligand-smiles NCCN \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

If `--ligand-smiles` is omitted, the metal-binding workflow screens candidate ligands for the given metal:

```bash
python -m des_multi_agent.cli \
  --workflow metal-binding \
  --metal-ion Cu2+ \
  --n 20 \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

The metal-binding workflow is separate from DES screening. DES run memory does not apply to metal-binding runs.

### Protonation-Aware Coordination

The metal-binding and metal-selectivity workflows profile ligand donor atoms at
the aqueous pH of the binding experiment. This matters because a ligand like
glycine (`NCC(=O)O`) is a bidentate N,O-chelator as drawn but a zwitterion at
pH 7 — the amine N is protonated and cannot donate to the metal.

The `binding_pH` defaults to 7.0. This is a Python-level parameter exposed
through the `run_metal_binding_screen` / `run_metal_selectivity_screen` API.
When the LLM is enabled, it also receives a "species @ pH 7.0" fact block in
its prompt, grounding its coordination claims in the actual species.

See [examples/metal_binding/](../examples/metal_binding) and [examples/ligand_binding_template/](../examples/ligand_binding_template).

## 14. Metal Selectivity

Use `metal-selectivity` to rank ligands by target-metal affinity and selectivity over a competitor:

```bash
python -m des_multi_agent.cli \
  --workflow metal-selectivity \
  --target-metal-ion Cu2+ \
  --competitor-metal-ion Zn2+ \
  --n 20 \
  --n-cycles 3 \
  --affinity-weight 0.5 \
  --selectivity-weight 0.5 \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

The composite score balances:

- `log K(target)`
- `delta log K = log K(target) - log K(competitor)`

The predictor gives the best selectivity signal when both ions are in the explicit identity table. Run this to list them:

```bash
python -m des_multi_agent.cli supported-metals
```

See [examples/metal_selectivity_standalone/](../examples/metal_selectivity_standalone) and [examples/ni2_co2_selectivity/](../examples/ni2_co2_selectivity).

## 15. Selectivity-DES Pipeline

Use `selectivity-des` when you want a two-phase loop:

1. Phase 1 screens ligands for metal-ion selectivity.
2. Phase 2 searches DES partners for the top selective ligands.
3. DES-compatible ligands feed back into the next outer cycle.

```bash
python -m des_multi_agent.cli \
  --workflow selectivity-des \
  --target-metal-ion Ni2+ \
  --competitor-metal-ion Co2+ \
  --n 20 \
  --n-cycles 3 \
  --n-des-candidates 20 \
  --n-des-cycles 3 \
  --n-outer-cycles 2 \
  --top-ligands 3 \
  --min-delta-log-k 0.5 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --stability-constant-model-path artifacts/stability_constants/model.json
```

Add viscosity constraints to Phase 2:

```bash
python -m des_multi_agent.cli \
  --workflow selectivity-des \
  --target-metal-ion Ni2+ \
  --competitor-metal-ion Co2+ \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --stability-constant-model-path artifacts/stability_constants/model.json \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json \
  --viscosity-threshold 200 \
  --viscosity-weight 0.4
```

See:

- [examples/ni_co_selectivity_des/](../examples/ni_co_selectivity_des)
- [examples/ni_co_selectivity_des_nemotron/](../examples/ni_co_selectivity_des_nemotron)
- [examples/ni_co_selectivity_des_qwen36/](../examples/ni_co_selectivity_des_qwen36)

## 16. Programmatic Use

All workflows are importable and callable directly from Python — useful for notebooks, scripts, and pipeline integration without going through the CLI or HTTP server.

### DES Screening

```python
from des_multi_agent.orchestrator import run_search_report

outcome = run_search_report(
    component_a="CCO",
    n=20,
    checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
    config_path="ml_des_mp/config.yaml",
    output_dir="runs/run_001",          # optional: writes report.txt / run.json / run.csv
)
# outcome.report     — formatted text report
# outcome.results    — list of ranked candidate dicts
# outcome.warnings   — list of warning strings
```

### Metal-Binding Prediction

```python
from des_multi_agent.workflows.metal_binding import run_metal_binding_workflow

outcome = run_metal_binding_workflow(
    metal_ion="Cu2+",
    ligand_smiles="NCCN",
    model_path="artifacts/stability_constants/model.json",
)
# outcome.prediction.log_k      — predicted log K
# outcome.prediction.confidence — confidence label
```

### Metal-Selectivity Screening

```python
from des_multi_agent.workflows.metal_binding_selectivity import run_metal_selectivity_screen

outcome = run_metal_selectivity_screen(
    target_metal="Cu2+",
    competitor_metal="Zn2+",
    n=20,
    model_path="artifacts/stability_constants/model.json",
    n_cycles=3,
    binding_pH=7.0,
)
# outcome.ranked_ligands  — list of SelectivityResult
# outcome.trajectory      — SearchTrajectory | None
```

### Selectivity-DES Pipeline

```python
from des_multi_agent.workflows.selectivity_des_pipeline import run_selectivity_des_pipeline

outcome = run_selectivity_des_pipeline(
    target_metal="Ni2+",
    competitor_metal="Co2+",
    checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
    n_ligands=20,
    n_des_candidates=20,
    n_selectivity_cycles=3,
    n_des_cycles=3,
    n_outer_cycles=2,
    stability_model_path="artifacts/stability_constants/model.json",
)
# outcome.selectivity_results  — Phase 1 ligand rankings
# outcome.des_results          — Phase 2 DES partner results per ligand
```

All functions return frozen dataclass outcomes. Trajectory data (when present) is in `outcome.trajectory` as a `SearchTrajectory` — pass it to `format_trajectory_report` or `write_trajectory_artifact` from `des_multi_agent.trajectory` to render it.

## 17. Choosing An Example Folder

Use this table when starting new work:

| Goal | Start here |
|------|------------|
| Confirm install works (no checkpoint needed) | [examples/dry_run/](../examples/dry_run) |
| Screen a curated SMILES list | [examples/candidates_file/](../examples/candidates_file) |
| Basic offline DES viscosity | [examples/des_viscosity/](../examples/des_viscosity) |
| Editable viscosity template | [examples/viscosity_template/](../examples/viscosity_template) |
| Viscosity composite ranking | [examples/viscosity_composite_ranking/](../examples/viscosity_composite_ranking) |
| Named threshold presets (`strict`/`relaxed`) | [examples/preset_thresholds/](../examples/preset_thresholds) |
| Fold-ensemble predictions with `ens_std` | [examples/ensemble_prediction/](../examples/ensemble_prediction) |
| Uncertainty modes (`filter`/`penalize`/`report_only`) | [examples/uncertainty_controls/](../examples/uncertainty_controls) |
| Machine-readable output (`--format json/csv`) | [examples/output_formats/](../examples/output_formats) |
| Multi-cycle iterative DES with trajectory | [examples/multi_cycle_des/](../examples/multi_cycle_des) |
| Plain-language DES with Gemma | [examples/plain_language_gemma4_12b/](../examples/plain_language_gemma4_12b) |
| Plain-language metal binding | [examples/plain_language_metal_binding_gemma4_12b/](../examples/plain_language_metal_binding_gemma4_12b) |
| Save-label-reuse feedback | [examples/des_run_memory_feedback/](../examples/des_run_memory_feedback) |
| Compare saved runs | [examples/compare_runs/](../examples/compare_runs) |
| Leaderboard/history | [examples/leaderboard_history/](../examples/leaderboard_history) |
| Metal binding | [examples/metal_binding/](../examples/metal_binding) |
| Metal selectivity only | [examples/metal_selectivity_standalone/](../examples/metal_selectivity_standalone) |
| Ni²⁺/Co²⁺ selectivity (offline) | [examples/ni2_co2_selectivity/](../examples/ni2_co2_selectivity) |
| Selectivity-DES | [examples/ni_co_selectivity_des/](../examples/ni_co_selectivity_des) |
| Real lidocaine DES run | [examples/lidocaine_gemma4_12b/](../examples/lidocaine_gemma4_12b) |
| Real betaine DES run (deterministic) | [examples/betaine_des/](../examples/betaine_des) |
| Real betaine DES run (Gemma 4-12B) | [examples/betaine_des_gemma4_12b/](../examples/betaine_des_gemma4_12b) |

The examples also feed the benchmark tests, so treat them as runnable documentation.

## 18. Troubleshooting

### Missing Checkpoint

Symptom: `DES workflow requires --checkpoint-path`.

Fix: pass the checkpoint explicitly or place a `*_best.pt` checkpoint under `ml_des_mp/runs/`.

```bash
python -m des_multi_agent.cli doctor --check checkpoint
```

### Bad Config

Symptom: config load errors or unexpected defaults.

Fix:

```bash
python -m des_multi_agent.cli doctor --check config
```

Check `DES_AGENT_CONFIG` if you use a custom config path.

### Invalid SMILES

Symptom: `--component-a` or `--ligand-smiles` fails validation.

Fix: use canonical SMILES when possible. For ethanol, use `CCO`; for lidocaine free base, use `CCN(CC)CC(=O)Nc1c(C)cccc1C`.

### Missing Artifacts

Symptom: viscosity or stability-constant model unavailable.

Fix:

```bash
python -m des_multi_agent.cli doctor --check artifacts
```

Expected local artifacts include:

- `artifacts/designsolvents/viscosity/model.json`
- `artifacts/stability_constants/model.json`

### Ollama Or LLM Not Running

Symptom: LLM provider errors, HTTP 404, or connection errors.

Fix:

```bash
python -m des_multi_agent.cli doctor --check llm --llm-config llm.example.yaml
```

Also confirm the requested `model_name` exists in your local Ollama installation.

### Output Directory Confusion

Use `--output-dir runs/run_001` for DES runs, then inspect with:

```bash
python -m des_multi_agent.cli view-run runs/run_001
```

### QSPR Model Missing Or Stochastic Outputs

Symptom: warnings about a missing `qspr_model.pt`, or melting-point predictions that differ between runs.

The QSPR layer provides an additional melting-point lookup source when the experimental table has no entry for a molecule. Its model file (`qspr_model.pt`) is not committed to the repository because its training is GPU-stochastic. To disable it and use only the committed experimental lookup and heuristic, set:

```bash
export DES_DISABLE_QSPR=1
```

All deterministic examples in `examples/` already set this variable in their `run.sh` so their outputs reproduce byte-for-byte. LLM-backed examples (e.g. `gemma4_12b/`) predate this feature and do not set it.

To build `qspr_model.pt` locally (requires GPU; see `artifacts/melting_points/README.md` for details):

```bash
python -m ml_des_mp.build_mp_lookup    # rebuild the experimental lookup table
python -m ml_des_mp.build_mp_dataset   # merge Bradley open MP data + DES components
python -m ml_des_mp.train_mp_qspr      # train the deep ensemble (~1 min on GPU)
```

When QSPR is enabled, set `DES_MP_DEVICE=cuda` to run the QSPR model on GPU (it defaults to `cpu`, independent of `--ml-device` which controls the DES ChemBERTa stage).

### Third-Party Warnings

The test suite may show deprecation warnings from `torch_geometric`, `torch.jit`, or FastAPI/Starlette test utilities. These warnings are external library warnings unless a test fails.

## 19. Testing And Benchmarking

Run the full suite:

```bash
python -m pytest -q
```

Run the example benchmark suite:

```bash
python -m pytest tests/test_benchmarks_examples.py -q
```

The example benchmark compares checked-in example outputs against frozen baselines under `tests/fixtures/example_benchmark_baselines/`.

Useful focused tests:

```bash
python -m pytest tests/test_cli.py -q
python -m pytest tests/test_doctor.py -q
python -m pytest tests/test_exports.py -q
python -m pytest tests/test_run_memory.py -q
python -m pytest tests/test_selectivity_des_pipeline.py -q
```

If you intentionally refresh example output, update the matching frozen baseline and run the benchmark test before committing.

## 20. Recommended Workflow For New Users

1. Install and save persistent defaults so you never repeat the long flags:

   ```bash
   pip install -e .
   des-agent config set checkpoint_path=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
   des-agent config set config_path=ml_des_mp/config.yaml
   ```

2. Run the health check:

   ```bash
   des-agent doctor
   ```

3. Confirm the repo works with the offline mock (no checkpoint needed):

   ```bash
   ./scripts/demo-mock.sh
   ```

4. Run one real DES command writing to a run directory:

   ```bash
   des-agent --workflow des --component-a "ethanol" --n 20 --output-dir runs/run_001
   ```

5. Inspect the result:

   ```bash
   des-agent view-run runs/run_001
   ```

6. Add viscosity or LLM mode only after the deterministic run works.

7. Save `run.memory.json`, label candidates with `label-run`, and reuse the history when you have feedback.

8. Use `compare-runs`, `history`, and `leaderboard` once you have multiple runs.

This keeps the workflow reproducible and makes each run easy to inspect later.
