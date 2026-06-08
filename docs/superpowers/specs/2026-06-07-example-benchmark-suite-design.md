# Example Benchmark Suite Design

## Goal

Turn the existing example folders into a pytest-based regression benchmark suite that scores example outputs for consistency. The benchmark should protect the current router, DES, viscosity, metal-binding, and plain-language workflows from regressions without rerunning external models.

## Scope

This first version is intentionally narrow:

- use the captured `input.txt` and `output.txt` files in the example folders as the source of truth
- run under `tests/` as part of the normal pytest suite
- compare normalized example outputs against their captured baselines
- report an aggregate benchmark score based on pass/fail coverage across the example set

This version does not:

- rerun live workflows
- call external models during the benchmark itself
- attempt scientific validation of model quality

## Architecture

Add a small benchmark harness under `tests/` that treats each example folder as a regression case.

The suite will:

- load the example request from `input.txt`
- load the expected result from `output.txt`
- optionally inspect the example `README.md` to ensure the folder still documents the intended workflow
- normalize both the expected text and the generated comparison text by removing warning noise and trivial formatting differences
- compare the normalized captured output against the expected normalized baseline

The benchmark remains separate from runtime code. It measures whether the example artifacts still describe the same runnable behavior the repository claims to support.

## Components

- `tests/test_benchmarks_examples.py`
  - parametrized pytest cases for each example folder
  - computes per-example pass/fail results
  - emits an aggregate benchmark score

- `tests/fixtures/example_benchmarks.py`
  - helper functions for locating example folders
  - output normalization helpers

- Example folders under `examples/`
  - treated as benchmark cases
  - each folder already includes the artifacts the benchmark needs:
    - `input.txt`
    - `output.txt`
    - `README.md`

## Example Coverage

The first benchmark suite should cover the existing example set:

- `examples/des_viscosity/`
- `examples/viscosity_template/`
- `examples/metal_binding/`
- `examples/ligand_binding_template/`
- `examples/gemma4_12b/`
- `examples/nemotron_3_nano/`
- `examples/qwen3_6/`
- `examples/lidocaine_gemma4_12b/`
- `examples/plain_language_gemma4_12b/`
- `examples/plain_language_metal_binding_gemma4_12b/`
- `examples/task_router/`

These examples represent the core supported workflows:

- router translation
- DES screening
- DES viscosity
- metal-binding / ligand-selection
- plain-language execution examples

## Data Flow

1. The benchmark enumerates the example folders.
2. For each folder, it loads:
   - the user-facing input request
   - the expected captured output
   - the short README text
3. It normalizes the captured output by:
   - trimming leading and trailing whitespace
   - collapsing repeated blank lines
   - stripping known warning-noise lines from model-loading output
4. It compares the normalized output against the normalized captured baseline text for that example.
5. It records the result as pass or fail.
6. The overall benchmark score is the fraction of examples that pass.

## Normalization Rules

The benchmark should be conservative. It should ignore only formatting and warning noise, not semantic output.

Allowed normalization:

- leading/trailing whitespace
- repeated blank lines
- third-party warning lines from model-loading output

Not ignored in v1:

- ranking order
- report fields
- section names
- predicted values
- router JSON fields
- workflow-specific warnings emitted by the repo itself

The intent is to catch real regressions, not to paper over output drift.

## Error Handling

- If an example folder is missing `input.txt`, `output.txt`, or `README.md`, the benchmark should fail that case with a clear message.
- If the normalized output no longer matches the baseline, the test should fail and point to the folder that drifted.
- If a warning line filter is too broad and removes real content, the failure should be visible in the comparison output.
- If a new example folder is added later, the benchmark should include it explicitly so coverage stays intentional.

## Testing

The benchmark itself is the test.

Additional checks should cover:

- helper normalization behavior
- folder discovery and required-file checks
- aggregate score calculation
- a small smoke test proving the benchmark can read at least one example folder

The benchmark should run quickly enough to stay in the normal pytest suite.

## Success Criteria

The benchmark is successful if:

- the example set is covered by pytest
- the suite produces a stable aggregate score
- the current examples remain reproducible as documentation and as regression baselines
- accidental output drift in router, DES, viscosity, metal-binding, or plain-language examples is detected immediately

