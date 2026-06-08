# Standardized Run Directories Design

## Goal
Standardize the filesystem layout for DES runs so each run writes a predictable flat directory containing the human-readable report and the machine-readable artifacts. This makes runs easier to find, reuse, compare, and inspect without changing the underlying predictions.

## Scope
This first version applies to DES runs only.

In scope:
- a new `--output-dir` flag for DES runs
- a flat run directory layout
- `report.txt` as the canonical human-readable report file
- `run.json`, `run.csv`, `run.manifest.json`, and `run.memory.json` placement inside the run directory
- reuse of the run directory by `label-run`, `reuse-run`, and `compare-runs`

Out of scope:
- metal-binding run directories
- router or task-execute run directories
- changing the prediction logic
- changing the report content
- nested subfolders such as `inputs/` or `artifacts/`
- automatic cleanup or versioned directory history

## User Experience
The user should be able to run a DES workflow and send the results to a chosen output directory:

```bash
python -m des_multi_agent.cli --workflow des --component-a "CCO" --n 20 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --config-path ml_des_mp/config.yaml --output-dir runs/run_001
```

That directory should contain the standard run artifacts:
- `report.txt`
- `run.json`
- `run.csv`
- `run.manifest.json`
- `run.memory.json` if memory saving is enabled

The directory becomes the stable anchor for later commands:
- `label-run --run runs/run_001`
- `--reuse-run runs/run_001`
- `compare-runs runs/run_001 runs/run_002`

The terminal report can still be printed, but the directory copy is the canonical saved report so users always know where to look.

## Architecture
Add a small output-directory layer to the DES workflow so the orchestrator and exporter write into a user-selected flat run folder.

Suggested modules:
- `des_multi_agent/orchestrator.py`
  - resolves the run output directory and passes it to the exporter and memory writer
- `des_multi_agent/exporting.py`
  - writes `report.txt`, `run.json`, `run.csv`, and `run.manifest.json`
- `des_multi_agent/run_memory.py`
  - continues to load and save `run.memory.json` in the same run directory
- `des_multi_agent/cli.py`
  - adds `--output-dir` to DES runs
- `tests/test_exports.py`
  - verifies the run directory layout and artifact placement
- `tests/test_cli.py`
  - covers `--output-dir` parsing and error cases
- `README.md`, `docs/tutorial.md`, and `examples/README.md`
  - document the new run directory convention

The directory layout should stay flat and obvious:
- no nested report tree
- no hidden subfolders required for the first version
- no change to non-DES workflows

## Data Flow
1. The user supplies `--output-dir /path/to/run`.
2. The CLI resolves the path and passes it to the DES orchestrator.
3. The DES workflow completes normally.
4. The orchestrator writes the human-readable report to `report.txt` in that directory.
5. The exporter writes `run.json`, `run.csv`, and `run.manifest.json` in the same directory.
6. If memory saving is enabled, `run.memory.json` is written there as well.
7. Later `label-run`, `reuse-run`, and `compare-runs` commands can point at that same directory.

The key behavior is that the run directory is the one place users can expect to find the full result of a DES execution.

## Error Handling
- If `--output-dir` cannot be created or written, fail the DES run with a clear error.
- If an export fails, report the failure clearly so the user knows the run directory is incomplete.
- If a required file already exists, overwrite the known run artifacts for the current run.
- If a later command points at a directory missing `run.memory.json`, report that clearly.
- If the user points `label-run`, `reuse-run`, or `compare-runs` at the wrong workflow directory, reject it with a clear error.

## Testing
Add tests for:
- creating a DES run in a user-supplied output directory
- writing `report.txt`, `run.json`, `run.csv`, and `run.manifest.json` into that directory
- writing `run.memory.json` into the same directory when requested
- rejecting unwritable output paths
- reusing the directory with `label-run`, `reuse-run`, and `compare-runs`
- preserving the default DES behavior when `--output-dir` is not used

The tests should remain local and deterministic.

## Success Criteria
The feature is complete when:
- every DES run can write to a stable user-selected output directory
- the directory contains the report and machine-readable artifacts in a flat layout
- `report.txt` is the canonical human-readable artifact
- later run-memory and comparison commands can reuse the directory directly
- the docs explain the directory convention clearly
