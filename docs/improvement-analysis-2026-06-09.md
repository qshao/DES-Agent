# DES-Agent Improvement Analysis
_Date: 2026-06-09_

## Context

This analysis was produced through a structured brainstorming session against the current codebase.
The project serves three overlapping audiences simultaneously:

- **Researchers** running the CLI on workstations/clusters
- **Non-programmer collaborators** who need interpretable outputs
- **Programmatic consumers** integrating with notebooks, APIs, or automation scripts

Three concrete pain points were identified as the most common sources of friction:

- **A — Setup friction**: users struggle to get checkpoint/config paths right before anything runs
- **B — LLM integration flakiness**: providers time out, return bad JSON, or are hard to configure
- **C — Results interpretation**: numbers and SMILES are hard to act on without chemistry background

---

## Improvement Catalogue

### A — Setup Friction

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| A1 | **Persistent config file** — `~/.des-agent/config.yaml` storing default `checkpoint_path`, `config_path`, `llm_config`; CLI reads it when flags are absent | High (50 lines, `pathlib` + `yaml`) | **Critical** |
| A2 | **Checkpoint auto-discovery** — if no `--checkpoint-path` is given, scan `ml_des_mp/runs/` for `*_best.pt` and select the most recent with a printed notice | High (~20 lines in `paths.py`) | **High** |
| A3 | **Richer path-error messages** — when `resolve_existing_path` fails, suggest the top 3 closest matches found nearby | Medium (fuzzy filesystem scan) | **High** |
| A4 | **`init` command** — interactive wizard that detects the checkpoint and writes a local config file | Medium (interactive prompts) | **Medium** — A1+A2 cover 80% of the same need |
| A5 | **`pip`-installable entry point** — add `console_scripts` so users run `des-agent` instead of `python -m des_multi_agent.cli` | High (`pyproject.toml` one-liner) | **Medium** |

---

### B — LLM Integration Robustness

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| B1 | **Retry with exponential backoff** — configurable retry loop (default 3 attempts, 2s/4s/8s + jitter) wrapping every LLM call in `transport.py` | High (`tenacity` or inline, ~30 lines) | **Critical** |
| B2 | **LLM response schema validation** — validate parsed LLM JSON against the existing `schemas.py` types; raise structured `LLMSchemaError` instead of a buried `KeyError` | High (add `__post_init__` validators) | **High** |
| B3 | **Raw response in error messages** — include first 200 chars of the raw LLM response when JSON parsing fails | High (one-line change in `parser.py`) | **High** |
| B4 | **LLM call caching** — disk cache keyed on prompt hash in `.cache/llm/`; TTL-configurable | Medium (hash + invalidation policy) | **Medium** — High value during iterative research |
| B5 | **LLM connectivity check in `doctor`** — send a minimal ping to the configured provider and report latency/status | High (`doctor.py` already has the extension pattern) | **High** |
| B6 | **Provider fallback chain** — ordered list of providers in config; if provider 1 fails, fall back to provider 2 | Medium (iterate list in `factory.py`) | **Medium** |
| B7 | **`--dry-run` flag** — validates paths, parses LLM config, pings the provider, but skips predictions | High (guard in `cli.py` + ping call) | **Medium** |

---

### C — Results Interpretation

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| C1 | **SMILES → human-readable names** — resolve candidates to common names via a bundled top-100 DES compound dict or optional PubChem lookup | Medium (local dict: High; PubChem: adds network dep) | **Critical** — Biggest UX gap for non-expert users |
| C2 | **Plain-language result summary** — prepend the report with a 3–5 sentence prose summary ("X candidates screened, Y are DES-formers, top candidate is Z at W K…") | High (string templating in `reporting.py`) | **High** |
| C3 | **ASCII melting curve chart** — inline sparkline of the Tm-vs-ratio curve for the top N candidates | Medium (small ASCII library or manual rendering) | **Medium** — High value for researchers who care about curve shape |
| C4 | **Uncertainty in plain language** — translate `trust_score=0.62, uncertainty_flag=medium` into "moderate confidence — consider experimental verification" | High (lookup table, ~15 lines in `reporting.py`) | **High** |
| C5 | **Color terminal output** — ANSI/`rich` highlighting: DES-positives green, high-uncertainty yellow, negatives grey; opt-in via `--color` or auto-detect TTY | High (`rich` is a common dep) | **Medium** |
| C6 | **`--format` flag** — support `table` (default), `json`, `csv`, `prose` for stdout report | Medium (routing in `cli.py` + format variants in `reporting.py`) | **Medium** — High value for the programmatic audience |

---

### Cross-Cutting (API / Programmatic Audience)

| ID | Idea | Feasibility | Importance |
|----|------|-------------|------------|
| X1 | **Progress indicators** — print `[1/7] Generating candidates…`, `[4/7] Running predictions…` to stderr during a run | High (~10 print statements in `orchestrator.py`) | **High** — Zero-cost; affects all user types |
| X2 | **Structured logging** — replace `llm_warnings` with proper `logging` module calls at `DEBUG`/`INFO`/`WARNING`; `--verbose` enables DEBUG | Medium (refactor warning propagation) | **Medium** |
| X3 | **FastAPI thin wrapper** — `des_multi_agent/server.py` with `POST /search` (DES) and `POST /metal-binding` endpoints over the existing orchestrators | Medium (~150 lines, no pipeline changes) | **High** — Unlocks programmatic audience cleanly |
| X4 | **Ensemble prediction** — run all 5 fold checkpoints and return mean ± std as the primary prediction; all 5 checkpoints are already on disk | Medium (loop over `*fold*_best.pt`, average outputs) | **High** — Improves scientific credibility with no new training |
| X5 | **`--watch` / incremental mode** — watch a directory for new SMILES files and run the pipeline automatically | Low (`watchdog` dep, niche use case) | **Low** |

---

## Recommended Priority Order

| Rank | IDs | Theme | Rationale |
|------|-----|-------|-----------|
| 1 | B1, B3 | LLM reliability — quick wins | Highest ROI for lowest effort; fixes the most complained-about failure mode immediately |
| 2 | A1 | Persistent config | Eliminates setup friction permanently for all audiences |
| 3 | C2, C4 | Readable reports | Makes results usable for non-expert collaborators without touching the ML layer |
| 4 | X1 | Progress indicators | Trivial to implement; prevents "is this hung?" anxiety on long runs |
| 5 | C1 | SMILES → names | Transformative for non-chemists; local dict variant is moderate effort |
| 6 | B5, A2 | Doctor + auto-discovery | Cheap fixes to the two most common first-run failures |
| 7 | X4 | Ensemble prediction | Improves scientific credibility; leverages existing on-disk checkpoints |
| 8 | X3 | FastAPI server | Unlocks the programmatic audience; clean given the existing architecture |
| 9 | B2, B4 | LLM schema + caching | Worth doing once B1 is stable |
| 10 | A3, C5, C6, B6, B7, A4, A5, X2 | Polish & power-user features | Address narrower use cases; tackle after core reliability is solid |

---

## Natural First Tranche

The following six improvements form a coherent "make it reliable and readable" release.
All are high-impact and low-to-medium effort, with no architectural dependencies between them:

**B1** (retry/backoff) · **B3** (raw response in errors) · **A1** (persistent config) ·
**C2** (plain-language summary) · **C4** (uncertainty explanation) · **X1** (progress indicators)
