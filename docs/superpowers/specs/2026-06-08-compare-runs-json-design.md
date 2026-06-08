# Compare Runs JSON Output Design

## Goal
Extend `compare-runs` so it can emit a compact machine-readable JSON summary in addition to the existing terminal comparison table. This keeps the command easy to read in a terminal while also making it script-friendly for automation.

## Scope
This first version applies to the existing same-workflow `compare-runs` command.

In scope:
- a `--json` flag on `compare-runs`
- JSON summary output on stdout
- the existing terminal diff report
- summary counts and top changed candidates in JSON
- same-workflow validation and hard-error behavior

Out of scope:
- cross-workflow comparison
- JSON files written to disk
- CSV output
- live reruns or prediction changes
- partial comparison for malformed inputs
- changing the terminal diff semantics

## User Experience
The command should still work the same way by default:

```bash
python -m des_multi_agent.cli compare-runs runs/run_001 runs/run_002
```

When the user adds `--json`, the command should emit both outputs to stdout:

```bash
python -m des_multi_agent.cli compare-runs runs/run_001 runs/run_002 --json
```

The terminal report remains the human-facing view, and the JSON summary provides a compact machine-readable version of the same comparison.

The JSON output should summarize:
- workflow
- the two input paths or identities
- counts of `new`, `removed`, `moved`, and `unchanged` candidates
- the top changed candidates only

## Architecture
Add a small JSON summary builder to the existing compare-runs module and expose it through a new CLI flag.

Suggested modules:
- `des_multi_agent/compare_runs.py`
  - builds the JSON summary alongside the terminal diff report
- `des_multi_agent/cli.py`
  - adds `--json` to the `compare-runs` subcommand and prints the summary
- `tests/test_compare_runs.py`
  - covers JSON summary content and same-workflow validation
- `tests/test_cli.py`
  - covers parsing and stdout behavior for the new flag
- `README.md`, `docs/tutorial.md`, and `examples/README.md`
  - document the JSON mode

The JSON summary should stay compact and stable:
- no full candidate dump
- no file output
- no change to comparison semantics

## Data Flow
1. The user runs `compare-runs run_a run_b --json`.
2. The command loads both saved run artifacts.
3. It verifies the workflow matches and the inputs are valid.
4. It computes the same ranked candidate diffs used by the terminal table.
5. It prints the terminal comparison table.
6. It also prints a compact JSON summary to stdout.

The JSON summary should reflect the same comparison result as the terminal report, not a separate or approximate calculation.

## Error Handling
- If either input is missing or malformed, fail with the existing hard error behavior.
- If the workflows do not match, fail before emitting any JSON summary.
- If the JSON summary cannot be built, treat that as a command error.
- If the command fails validation, do not emit partial JSON.
- Keep the existing same-workflow restriction intact.

## Testing
Add tests for:
- `compare-runs --json` parsing
- JSON summary structure with counts and top changed candidates
- same-workflow validation before JSON output
- hard errors for missing or malformed run artifacts
- preserving the existing terminal report

The tests should remain local and deterministic.

## Success Criteria
The feature is complete when:
- `compare-runs --json` prints a compact machine-readable summary
- the terminal report still appears and is unchanged in meaning
- the JSON output stays same-workflow only and read-only
- invalid inputs still fail hard
- the docs explain when to use JSON mode
