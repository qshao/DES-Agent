# Run Summary Design

## Goal
Add a compact terminal summary block that appears after every command finishes, so users get a quick answer to "what happened?" without reading the full report.

The summary should be:
- terminal-only in v1
- read-only
- short enough to scan quickly
- safe to omit if summary formatting fails
- available for all user-facing commands

The summary must not change:
- prediction results
- report formatting
- exports
- run memory
- command exit status

## Scope
This feature applies to:
- DES runs
- metal-binding runs
- `doctor`
- `compare-runs`
- `task-execute`

For machine-readable stdout commands, the summary should still be available on the terminal, but it must be routed to stderr so stdout stays parseable:
- `task-router`
- `compare-runs --json`

This feature does not add:
- a summary file on disk
- a JSON summary mode
- new prediction logic
- new export behavior

## Architecture
Add a small presentation layer that formats a command-specific summary from the structured result that each command already produces.

Recommended shape:
- each command returns or exposes a small result object
- the summary layer reads that result object
- the CLI prints the main report first
- the CLI prints the summary block immediately after the main report, except for machine-readable stdout modes where the summary should go to stderr so it does not break parsing

The summary layer should stay separate from the command implementations so it can be tested independently and changed without touching prediction or routing logic.

## Components
- `des_multi_agent/summary.py`
  - builds the compact summary block
  - exposes one formatter per command family, plus a generic fallback
- `des_multi_agent/cli.py`
  - prints the summary block after the main command output
- `des_multi_agent/orchestrator.py`
  - exposes the fields needed for DES summaries
- `des_multi_agent/workflows/metal_binding.py`
  - exposes the fields needed for metal-binding summaries
- `des_multi_agent/doctor.py`
  - exposes counts and status for doctor summaries
- `des_multi_agent/compare_runs.py`
  - exposes a short comparison outcome summary
- `des_multi_agent/task_router.py`
  - exposes whether the request was clarified or routed cleanly
- `des_multi_agent/task_executor.py`
  - exposes whether execution completed or deferred to clarification
- `tests/test_summary.py`
  - verifies command-specific formatting and fallback behavior
- `tests/test_cli.py`
  - verifies the CLI prints the summary after the main output
- `README.md` and `docs/tutorial.md`
  - document the summary block so users know to expect it

## Data Flow
1. A command runs and produces its normal result.
2. The command prints its primary report as it does today.
3. The summary layer reads the command result and derives a compact status block.
4. The CLI prints that summary block after the main report, or to stderr if stdout must remain machine-readable.
5. If the summary formatter cannot build a specific command summary, the CLI prints a generic fallback summary instead of failing.

## Summary Content
The summary should show only a few high-signal items.

For DES runs:
- workflow name
- number of ranked candidates
- whether run memory reuse was applied
- whether exports were written

For metal-binding runs:
- workflow name
- binding prediction status
- model name or path when available

For `doctor`:
- overall status
- number of errors
- number of warnings

For `compare-runs`:
- compared workflow
- number of changed candidates in the top diff set
- counts of new, removed, moved, and unchanged candidates

For `task-router`:
- whether the request was complete or needs clarification
- number of clarification questions when present

For `task-execute`:
- whether execution completed
- whether clarification was required instead of execution

## Error Handling
- If summary generation fails, the command output should still succeed and print the main report.
- If a command type is unknown to the summary layer, use a generic "completed" summary.
- If a required field is missing from a result object, omit that field instead of crashing.
- The summary must never suppress existing errors or warnings from the underlying command.
- If a summary formatter raises unexpectedly, catch the exception at the CLI boundary and continue.

## Testing
- Add unit tests for the summary formatter across command types.
- Add CLI tests to confirm the summary prints after the main report.
- Add regression tests for:
  - DES runs with memory reuse and exports
  - metal-binding runs
  - `doctor`
  - `compare-runs`
  - `task-router`
  - `task-execute`
- Add a fallback test that summary generation failure does not break the underlying command output.

