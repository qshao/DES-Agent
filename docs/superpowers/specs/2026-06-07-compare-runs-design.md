# Compare Runs Design

## Goal
Add a read-only `compare-runs` subcommand that compares two saved run artifacts from the same workflow and prints a compact terminal report showing how the top ranked candidates changed.

## Scope
This first version compares saved runs from the same workflow only.

In scope:
- two run folders or two `run.memory.json` files as inputs
- same-workflow validation
- ranked candidate comparison
- top-of-list comparison only
- terminal output with new / removed / moved / unchanged markers
- hard failure on malformed inputs or workflow mismatches

Out of scope:
- cross-workflow comparison
- JSON export
- CSV export
- live reruns or model inference
- partial comparison against malformed inputs
- editing or rewriting the saved runs

## User Experience
The command should be invoked as:

```bash
python -m des_multi_agent.cli compare-runs <run-a> <run-b>
```

Each argument may be either:
- a run folder containing `run.memory.json`
- a direct `run.memory.json` file path

The output should be a readable terminal summary that shows:
- the run identity or input path for each side
- the top ranked candidates for each run
- whether each candidate is `new`, `removed`, `moved`, or `unchanged`
- the rank change when a candidate exists in both runs

The command should fail fast if either input is malformed or if the workflows do not match.

## Architecture
Add a focused comparison module that reuses the existing run-memory loader and produces a terminal summary.

Suggested modules:
- `des_multi_agent/compare_runs.py`
  - loads the two saved runs
  - validates workflow compatibility
  - computes ranked candidate differences
  - formats the terminal report
- `des_multi_agent/run_memory.py`
  - reused for loading saved runs from folders or files
- `des_multi_agent/cli.py`
  - adds the `compare-runs` subcommand and prints the report
- `tests/test_compare_runs.py`
  - covers workflow mismatch, malformed inputs, and rank diff formatting
- `README.md` and `docs/tutorial.md`
  - document the new comparison command and when to use it

The comparison logic should stay small and deterministic:
- load both saved runs
- verify both runs are parseable
- verify both runs have the same workflow
- build a ranked candidate lookup from each run
- compare only the top few candidates from each run
- render a concise diff-style table

## Checks
The command should compare at least:
- candidate presence in top ranks
- rank changes for candidates appearing in both runs
- new candidates in run B
- removed candidates from run A

The report should only show a compact top-of-list view so the terminal output stays easy to scan.

## Error Handling
- If either file is missing, report a hard error and exit nonzero.
- If either file is malformed, report a hard error and exit nonzero.
- If the workflows do not match, report a hard error and exit nonzero.
- If the saved runs do not contain ranked candidates, report a hard error.
- Do not attempt partial comparison when the inputs are invalid.

## Testing
Add tests for:
- comparing two valid DES run memory files
- comparing two valid run folders
- rejecting mismatched workflows
- rejecting malformed run-memory files
- rejecting missing files
- marking candidates as `new`, `removed`, `moved`, and `unchanged`
- CLI parsing for the new `compare-runs` subcommand

The tests should stay local and deterministic.

## Success Criteria
The feature is complete when:
- `compare-runs` accepts two saved run artifacts
- it rejects invalid or mismatched inputs
- it prints a compact terminal comparison of the top ranked candidates
- it remains read-only and offline
- the docs show users when to use it
