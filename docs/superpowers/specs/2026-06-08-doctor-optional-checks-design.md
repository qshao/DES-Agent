# Doctor Optional Checks Design

## Goal
Extend the existing read-only `doctor` command with an opt-in `--check` mode that performs additional local setup checks for common user workflows. The new checks should help users spot missing files and setup gaps before they start a run, without making `doctor` slower or more brittle by default.

## Scope
This is an extension to the current `doctor` command, not a replacement.

In scope:
- a `doctor --check ...` option in `des_multi_agent.cli`
- optional local checks for:
  - `checkpoint`
  - `discovery`
  - `artifacts`
- warning-only behavior for optional check failures
- deduplication of repeated optional checks
- report formatting that keeps core errors and optional warnings grouped clearly
- tests for valid and invalid optional check names

Out of scope:
- live Ollama probes
- model downloads
- inference or smoke runs
- automatic fixes
- cross-workflow validation
- writing or modifying any files

## User Experience
The command should remain:

```bash
python -m des_multi_agent.cli doctor
```

Optional checks are enabled with:

```bash
python -m des_multi_agent.cli doctor --check checkpoint --check discovery --check artifacts
```

The command should:
- keep the default behavior unchanged when `--check` is not used
- run all requested optional checks in one pass
- collect all core errors and optional warnings together
- print a grouped report with `errors` and `warnings`
- exit nonzero only when core errors are present

Optional failures should be visible but non-blocking so users can see what is missing without turning the command into a setup gatekeeper.

## Architecture
Add optional-check handling to the existing doctor pipeline rather than creating a separate command.

Suggested modules:
- `des_multi_agent/doctor.py`
  - add optional check execution and result aggregation
- `des_multi_agent/cli.py`
  - parse repeated `--check` values and pass them to the doctor runner
- `tests/test_doctor.py`
  - cover optional success, optional warnings, and invalid check names
- `README.md`, `docs/tutorial.md`, and `examples/README.md`
  - document the new `doctor --check` usage

The optional checks should stay local and lightweight:
- file existence and readability
- path resolution relative to the repository root
- no network access
- no inference
- no environment mutation

## Checks
The first version should support exactly these optional checks:
- `checkpoint`
  - validate that the default DES checkpoint referenced by the docs/examples exists
- `discovery`
  - validate that the discovery directory or file path expected by the examples exists
- `artifacts`
  - validate that the local optional artifact paths used by the current repo features exist

The command should ignore duplicates silently. If a user asks for the same optional check multiple times, it should behave as though it was requested once.

## Error Handling
- If the user passes an unsupported optional check name, fail fast with a clear usage error.
- If an optional check target is missing or unreadable, report it as a `warning`.
- If multiple optional issues exist, report all of them together instead of stopping early.
- If the core repo checks fail, preserve the optional warnings in the report so the user can see the full setup picture.
- If an optional check path exists but is not readable, report the path name and the check name so the user knows what to fix.

## Testing
Add tests for:
- default `doctor` behavior staying unchanged when `--check` is not used
- requested optional checks producing warnings when local paths are missing
- requested optional checks producing no warnings when local paths are present
- unsupported `--check` names failing clearly
- duplicate `--check` values being deduplicated
- report grouping that keeps errors and warnings separate

The tests should remain deterministic and local.

## Success Criteria
The feature is complete when:
- `doctor --check` accepts the documented optional check names
- optional check failures are warnings only
- the default `doctor` command behavior is unchanged
- the report stays grouped and readable
- the docs show users how to run the optional checks before a workflow
