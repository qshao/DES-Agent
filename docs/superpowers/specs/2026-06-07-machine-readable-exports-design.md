# Machine-Readable Exports Design

## Goal
Add automatic machine-readable exports for DES runs so every run writes a structured JSON payload, a flat CSV summary, and a manifest file next to the human-readable report.

## Scope
This first version applies to DES runs only.

In scope:
- automatic export for every DES run
- `run.json`
- `run.csv`
- `run.manifest.json`
- export location next to the run output
- structured run data including ranked results, uncertainty, provenance, memory notes, and warnings

Out of scope:
- metal-binding exports
- router exports
- compare-runs exports
- doctor output exports
- opt-in export flags
- alternate export directories
- post-hoc reruns or prediction changes

## User Experience
Every DES run should produce the same three machine-readable files automatically, without requiring a flag.

The files should live next to the human-readable output so users can discover them easily:
- `run.json`
- `run.csv`
- `run.manifest.json`

The terminal report should remain unchanged. The export is an additional artifact bundle, not a replacement for the report.

## Architecture
Add a small export layer that runs after the DES workflow completes and writes the machine-readable bundle from the final run object.

Suggested modules:
- `des_multi_agent/exporting.py`
  - builds and writes the export bundle
- `des_multi_agent/orchestrator.py`
  - invokes the exporter after a successful DES run
- `des_multi_agent/cli.py`
  - passes the resolved output location through to the orchestrator
- `tests/test_exports.py`
  - verifies the bundle contents and failure behavior
- `README.md` and `docs/tutorial.md`
  - document the exported files and where they are written

The exporter should be a pure post-processing step:
- it receives the completed DES run data
- it does not run prediction
- it does not change ranking
- it does not depend on LLM availability

## Export Contents
`run.json` should contain the structured run payload, including:
- input parameters
- ranked DES results
- uncertainty annotations
- candidate provenance
- memory notes
- warnings

`run.csv` should contain a flat table of the ranked DES results. It should include at least:
- `smiles_b`
- `is_des`
- `min_tm_k`
- `rank`
- `source`
- `source_id`
- `trust_score`
- `uncertainty_flag`

`run.manifest.json` should contain metadata about the export, including:
- workflow name
- timestamp or run identifier if available from the run object
- input parameters
- export file names
- output location

## Error Handling
- If the output directory cannot be written, fail the DES run with a clear error.
- If any export format fails, report that failure clearly rather than silently skipping it.
- If a field required for CSV flattening is missing, fail fast rather than inventing a placeholder.
- If the workflow is not DES, do not invoke the exporter.
- Export failures must not alter the underlying predictions or ranking logic.

## Testing
Add tests for:
- automatic export of a DES run bundle
- `run.json` creation with the expected structured fields
- `run.csv` creation with the expected flattened columns
- `run.manifest.json` creation with the expected metadata fields
- failure when the output path is not writable
- failure when required CSV fields are missing
- no export invocation for non-DES workflows

The tests should remain local and deterministic.

## Success Criteria
The feature is complete when:
- every DES run writes `run.json`, `run.csv`, and `run.manifest.json`
- the files are written next to the run output
- the terminal report remains unchanged
- export failures are clear and do not corrupt the run logic
- the docs tell users where to find the exported files
