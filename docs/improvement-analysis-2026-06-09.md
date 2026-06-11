# DES-Agent Improvement Analysis
_Date: 2026-06-09 · Updated: 2026-06-09 (second brainstorm)_

## Context

This analysis was produced through structured brainstorming sessions against the current codebase.
The project serves three overlapping audiences simultaneously:

- **Researchers** running the CLI on workstations/clusters
- **Non-programmer collaborators** who need interpretable outputs
- **Programmatic consumers** integrating with notebooks, APIs, or automation scripts

Five concrete pain points have been identified:

- **A — Setup friction**: users struggle to get checkpoint/config paths right before anything runs
- **B — LLM integration flakiness**: providers time out, return bad JSON, or are hard to configure
- **C — Results interpretation**: numbers and SMILES are hard to act on without chemistry background
- **D — Data quality**: bad or duplicate inputs cause cryptic failures deep in the pipeline
- **E–F — Workflow ergonomics & reliability**: too many flags to remember; one bad candidate aborts the whole run

---

## Completed Items

The following improvements have been implemented and are covered by the test suite:

| ID | Description |
|----|-------------|
| ✅ A1 | Persistent user config (`~/.des-agent/config.yaml`) |
| ✅ A2 | Checkpoint auto-discovery from `ml_des_mp/runs/` |
| ✅ B1 | Retry with exponential backoff in `transport.py` |
| ✅ B3 | Raw LLM response excerpt in parse-error messages |
| ✅ B5 | LLM connectivity check in `doctor` |
| ✅ C1 | SMILES → human-readable names (local dict + optional PubChem) |
| ✅ C2 | Plain-language result summary block |
| ✅ C4 | Uncertainty translated to plain-language confidence labels |
| ✅ X1 | Numbered progress indicators to stderr |
| ✅ X3 | FastAPI thin wrapper (`server.py`) with `/search` and `/metal-binding` |
| ✅ X4 | Ensemble prediction over all fold checkpoints (mean ± std) |

---

## Improvement Catalogue

### A — Setup Friction

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| A1 | **Persistent config file** — `~/.des-agent/config.yaml` storing default `checkpoint_path`, `config_path`, `llm_config`; CLI reads it when flags are absent | High (50 lines, `pathlib` + `yaml`) | ✅ Done |
| A2 | **Checkpoint auto-discovery** — if no `--checkpoint-path` is given, scan `ml_des_mp/runs/` for `*_best.pt` and select the most recent with a printed notice | High (~20 lines in `paths.py`) | ✅ Done |
| A3 | **Richer path-error messages** — when `resolve_existing_path` fails, suggest the top 3 closest matches found nearby | Medium (fuzzy filesystem scan) | **High** |
| A4 | **`init` command** — interactive wizard that detects the checkpoint and writes a local config file | Medium (interactive prompts) | **Medium** — A1+A2 cover 80% of the same need |
| A5 | **`pip`-installable entry point** — add `console_scripts` so users run `des-agent` instead of `python -m des_multi_agent.cli` | High (`pyproject.toml` one-liner) | **Medium** |

---

### B — LLM Integration Robustness

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| B1 | **Retry with exponential backoff** — configurable retry loop (default 3 attempts, 2s/4s/8s + jitter) wrapping every LLM call in `transport.py` | High (`tenacity` or inline, ~30 lines) | ✅ Done |
| B2 | **LLM response schema validation** — validate parsed LLM JSON against the existing `schemas.py` types; raise structured `LLMSchemaError` instead of a buried `KeyError` | High (add `__post_init__` validators) | **High** |
| B3 | **Raw response in error messages** — include first 200 chars of the raw LLM response when JSON parsing fails | High (one-line change in `parser.py`) | ✅ Done |
| B4 | **LLM call caching** — disk cache keyed on prompt hash in `.cache/llm/`; TTL-configurable | Medium (hash + invalidation policy) | **Medium** — High value during iterative research |
| B5 | **LLM connectivity check in `doctor`** — send a minimal ping to the configured provider and report latency/status | High (`doctor.py` already has the extension pattern) | ✅ Done |
| B6 | **Provider fallback chain** — ordered list of providers in config; if provider 1 fails, fall back to provider 2 | Medium (iterate list in `factory.py`) | **Medium** |
| B7 | **`--dry-run` flag** — validates paths, parses LLM config, pings the provider, but skips predictions | High (guard in `cli.py` + ping call) | **Medium** |

---

### C — Results Interpretation

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| C1 | **SMILES → human-readable names** — resolve candidates to common names via a bundled top-100 DES compound dict or optional PubChem lookup | Medium (local dict: High; PubChem: adds network dep) | ✅ Done |
| C2 | **Plain-language result summary** — prepend the report with a 3–5 sentence prose summary ("X candidates screened, Y are DES-formers, top candidate is Z at W K…") | High (string templating in `reporting.py`) | ✅ Done |
| C3 | **ASCII melting curve chart** — inline sparkline of the Tm-vs-ratio curve for the top N candidates | Medium (small ASCII library or manual rendering) | **Medium** — High value for researchers who care about curve shape |
| C4 | **Uncertainty in plain language** — translate `trust_score=0.62, uncertainty_flag=medium` into "moderate confidence — consider experimental verification" | High (lookup table, ~15 lines in `reporting.py`) | ✅ Done |
| C5 | **Color terminal output** — ANSI/`rich` highlighting: DES-positives green, high-uncertainty yellow, negatives grey; opt-in via `--color` or auto-detect TTY | High (`rich` is a common dep) | **Medium** |
| C6 | **`--format` flag** — support `table` (default), `json`, `csv`, `prose` for stdout report | Medium (routing in `cli.py` + format variants in `reporting.py`) | **Medium** — High value for the programmatic audience |

---

### Cross-Cutting (API / Programmatic Audience)

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| X1 | **Progress indicators** — print `[1/7] Generating candidates…`, `[4/7] Running predictions…` to stderr during a run | High (~10 print statements in `orchestrator.py`) | ✅ Done |
| X2 | **Structured logging** — replace `llm_warnings` with proper `logging` module calls at `DEBUG`/`INFO`/`WARNING`; `--verbose` enables DEBUG | Medium (refactor warning propagation) | **Medium** |
| X3 | **FastAPI thin wrapper** — `des_multi_agent/server.py` with `POST /search` (DES) and `POST /metal-binding` endpoints over the existing orchestrators | Medium (~150 lines, no pipeline changes) | ✅ Done |
| X4 | **Ensemble prediction** — run all 5 fold checkpoints and return mean ± std as the primary prediction; all 5 checkpoints are already on disk | Medium (loop over `*fold*_best.pt`, average outputs) | ✅ Done |
| X5 | **`--watch` / incremental mode** — watch a directory for new SMILES files and run the pipeline automatically | Low (`watchdog` dep, niche use case) | **Low** |

---

### D — Data Quality & Scientific Rigor _(new)_

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| D1 | **Input SMILES validation** — run RDKit `MolFromSmiles` on `component_a` and each candidate immediately at pipeline entry; raise a clear `InvalidSMILESError` with the offending string rather than a cryptic model crash | High (~20 lines in `orchestrator.py`) | **Critical** |
| D2 | **Checkpoint–config compatibility check** — at model load time, compare the embedding method and `n_bits` stored in the checkpoint against the active `config.yaml`; warn loudly if they diverge instead of silently producing wrong predictions | Medium (inspect checkpoint metadata) | **High** |
| D3 | **Candidate deduplication** — canonicalize each proposed SMILES with RDKit before ML predictions; drop duplicates and surface a `memory_note` listing collapsed pairs (same compound from two sources) | Low (~15 lines in `orchestrator.py`) | **High** |
| D4 | **Batch screening from file** — `--candidates-file smiles.txt`, one SMILES per line, bypasses LLM candidate generation and screens the list directly; complements `--ensemble` for power users | Low (~30 lines in `cli.py` + `orchestrator.py`) | **High** |

---

### E — Workflow Ergonomics _(new)_

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| E1 | **Threshold presets** — `--preset strict/standard/relaxed` mapping to sensible `absolute_tm_max_k` + `relative_drop_min` combinations; users don't need to understand raw threshold parameters | Low (~20 lines in `cli.py`) | **High** |
| E2 | **Run history viewer** — `des-agent history <dir>` subcommand printing a ranked table of all past runs (date, n_screened, n_des, top candidate, top min_tm_k); one-glance overview of prior work | Low (~50 lines, reads `run.manifest.json` files) | **Medium** |
| E3 | **`--top-k` report filter** — when screening large N, only show the best K DES-formers in the printed report; full results still written to JSON/CSV | Low (~5 lines in `reporting.py` + CLI arg) | **Medium** |
| E4 | **Config save command** — `des-agent config set checkpoint_path=<path>` convenience wrapper around `user_config.save_user_config()`; no need to hand-edit YAML | Low (~20 lines in `cli.py`) | **Medium** |

---

### F — Reliability & Safety _(new)_

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| F1 | **Per-candidate graceful failure** — if ML prediction fails for one SMILES, log a warning and continue screening the rest; currently one bad input aborts the entire run | Low (~10 lines in `orchestrator.py`) | **High** |
| F2 | **Atomic output writes** — write to `<file>.tmp` then rename; prevents partial or corrupt export files if the process is killed mid-write | Low (~10 lines in `exporting.py`) | **Medium** |
| F3 | **PubChem request guard** — the `smiles_names.py` PubChem path has no timeout or rate limit; add a configurable `timeout_seconds` (default 3 s) and suppress lookups on network errors | Low (~5 lines in `smiles_names.py`) | **Medium** |
| F4 | **Run memory corruption recovery** — when `parse_run_memory` encounters malformed JSON, emit a `memory_note` warning and return empty memory rather than silently swallowing the error | Low (~5 lines in `run_memory.py`) | **Low** |

---

### G — Scientific Communication _(new)_

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| G1 | **Cross-run leaderboard** — `des-agent leaderboard <dir>` merges all runs in a history directory into a single ranked table, deduplicates compounds, and shows the best prediction per compound across experiments | Medium (~80 lines, reads `run.json` files) | **High** |
| G2 | **Candidate provenance footnotes** — render each candidate's `source_id` as a footnote (e.g., `† HuangEtAl2022`) rather than inline noise in the results table | Low (~20 lines in `reporting.py`) | **Medium** |
| G3 | **Markdown / LaTeX table export** — add `markdown` and `latex` as `--format` options; Markdown renders directly in Jupyter notebooks, LaTeX drops into papers | Low (~40 lines in `reporting.py`) | **Medium** |
| G4 | **Confidence interval in export** — include `[mean − 2σ, mean + 2σ]` columns in CSV and JSON when ensemble is used; makes the data immediately actionable for downstream statistics | Low (~10 lines in `exporting.py`) | **Medium** |

---

## Recommended Priority Order

_Items marked ✅ are complete. Remaining items ordered by ROI._

| Rank | IDs | Theme | Rationale |
|------|-----|-------|-----------|
| — | B1, B3, A1, A2, C2, C4, X1, C1, X3, X4, B5 | **Completed** | Core reliability and readability tranche — done |
| 1 | D1 | Input validation | One-line fix prevents cryptic model crashes for all audiences |
| 2 | D3 | Candidate deduplication | Prevents wasted predictions; surfaces provenance conflicts |
| 3 | F1 | Per-candidate graceful failure | Stops one bad SMILES from aborting an entire screening run |
| 4 | D4 | Batch file input | High-value for researchers with a custom candidate list |
| 5 | E1 | Threshold presets | Reduces cognitive load for new users; no schema changes |
| 6 | G1 | Cross-run leaderboard | Enables longitudinal comparison across experiments |
| 7 | B2 | LLM schema validation | Structured errors replace buried `KeyError` crashes |
| 8 | D2 | Checkpoint–config check | Prevents silently wrong predictions from config mismatch |
| 9 | E2, E4 | History viewer + config set | Ergonomic polish for regular users |
| 10 | B4 | LLM call caching | High value during iterative research; medium implementation effort |
| 11 | C3, C5, C6, G3 | Output polish | ASCII chart, color, format flags, Markdown/LaTeX export |
| 12 | G2, G4, E3, F2, F3, F4 | Polish & safety | Provenance footnotes, CI export, top-k filter, atomic writes |
| 13 | A3, A4, A5, B6, B7, X2, X5 | Power-user features | Narrower use cases; tackle after core reliability is solid |

---

## Second Tranche Recommendation

The following six items form a coherent **"data quality + workflow ergonomics"** release.
All are low-to-medium effort with no architectural dependencies between them:

**D1** (SMILES validation) · **D3** (deduplication) · **F1** (per-candidate graceful failure) ·
**D4** (batch file input) · **E1** (threshold presets) · **G1** (cross-run leaderboard)
