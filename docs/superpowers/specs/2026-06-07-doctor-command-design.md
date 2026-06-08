# Doctor Command Design

## Goal
Add a read-only `doctor` subcommand to `des_multi_agent.cli` that helps users verify their local DES-Agent setup before they run a workflow. The command should stay fast, offline, and safe: it must not download models, run inference, or modify files.

## Scope
This first version checks the core repo plus the checked-in example folders.

In scope:
- core repository files and paths
- `ml_des_mp/config.yaml`
- the default DES checkpoint path used by the docs and demo wrappers
- `llm.example.yaml`
- the example folders under `examples/`
- the benchmark fixtures and baseline files used by the example regression suite

Out of scope:
- Ollama connectivity checks
- local artifact validation beyond file presence
- model inference or smoke runs
- network access
- automatic fixes or file edits

## User Experience
The command should be invoked as:

```bash
python -m des_multi_agent.cli doctor
```

It should:
- collect all issues in one run
- print a clear grouped report
- distinguish `errors` from `warnings`
- exit with a nonzero status if any `errors` are found

The output should be readable in a terminal and should explain what is missing or inconsistent and where the user should look.

## Architecture
Add a small doctor layer in the CLI path that assembles a list of checks and renders their results.

Suggested modules:
- `des_multi_agent/doctor.py`
  - implements the checks and result aggregation
- `des_multi_agent/cli.py`
  - adds the `doctor` subcommand and prints the report
- `tests/test_doctor.py`
  - covers success and failure cases
- `README.md`, `docs/tutorial.md`, and `examples/README.md`
  - document the new command and where it fits in the setup flow

The checks should stay simple and local:
- path existence and readability
- example folder structure
- benchmark fixture presence
- basic consistency between docs and example directories

## Checks
The first version should check at least:
- `ml_des_mp/config.yaml` exists
- the default checkpoint referenced by the demo docs exists
- `llm.example.yaml` exists and is readable
- each example folder listed in `examples/README.md` exists
- each example folder has `README.md`, `input.txt`, `output.txt`, and `run.sh`
- the benchmark baseline directory exists and contains the example baseline files
- the top-level docs mention the example benchmark suite and the main example folders

The command should not fail just because an optional example output is noisy, as long as the checked-in artifact exists.

## Error Handling
- If a required file is missing, report it as an `error`.
- If a doc link or example reference is stale, report it as a `warning` unless it blocks the command from understanding the repo layout.
- If multiple problems exist, report all of them together.
- If the user runs the command from a different working directory, it should still resolve paths relative to the repository root.

## Testing
Add tests for:
- a fully healthy repository producing zero errors
- missing core files producing errors
- missing example folders producing errors
- stale or missing benchmark fixture files producing errors
- output grouping that distinguishes errors and warnings
- nonzero exit behavior when errors are present

The tests should stay local and deterministic.

## Success Criteria
The feature is complete when:
- the new `doctor` subcommand runs from the repo root
- it reports all issues in one pass
- it clearly separates errors and warnings
- it remains read-only and offline
- the docs show users how to run it before the first demo
