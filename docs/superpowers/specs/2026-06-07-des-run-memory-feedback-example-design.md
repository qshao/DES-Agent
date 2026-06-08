# DES Run Memory Feedback Example Design

**Goal:** Add a new example folder that demonstrates the full DES feedback loop: run DES once, save `run.memory.json`, label the saved run in place, and reuse that labeled memory on a later DES run.

**Architecture:** The example should stay offline, small, and copyable. It will use the existing DES CLI commands only: `--save-run-memory`, `label-run`, and `--reuse-run`. The example folder will contain a single human-readable transcript plus a prebuilt `run.memory.json` so users can inspect the saved memory format before adapting it for their own runs. No new runtime code paths are needed; the example is documentation-through-execution.

**Tech Stack:** Python, shell scripts, JSON, pytest, existing `des_multi_agent` CLI and run-memory helpers.

---

## Example Folder

Create a new folder:

- `examples/des_run_memory_feedback/`

Add these files:

- `examples/des_run_memory_feedback/input.txt`
- `examples/des_run_memory_feedback/output.txt`
- `examples/des_run_memory_feedback/run.memory.json`
- `examples/des_run_memory_feedback/run.sh`
- `examples/des_run_memory_feedback/README.md`

## Example Content

The example should use a simple DES target:

- component A: `CCO`

The example should show the full three-step loop:

1. run DES with `--save-run-memory`
2. run `label-run` on the saved memory
3. run DES again with `--reuse-run` pointing at the labeled memory

The checked-in `run.memory.json` should be a small, valid DES memory sample that matches the example flow and makes the label format easy to inspect.

## Required Behaviors

- The example must use the existing CLI commands directly.
- The example must keep the feedback loop DES-only.
- The example must not introduce a second memory format.
- The example transcript should be captured in a single `output.txt`.
- The example README should explain the save-label-reuse flow and tell users how to adapt the example for their own runs.

## Documentation Updates

Update these docs so the new example is easy to find:

- `examples/README.md`
- `README.md`
- `docs/tutorial.md`

The docs should briefly explain:

- what the new example demonstrates
- where the saved `run.memory.json` is
- how `label-run` fits between saving and reuse
- that the feature is DES-only

## Testing

Add or update tests so the example is kept in sync:

- Verify the new folder exists and includes `input.txt`, `output.txt`, `run.memory.json`, `README.md`, and `run.sh`
- Verify the example README mentions the save-label-reuse flow
- Verify the new example is listed in `examples/README.md`
- Add a small benchmark/index test only if the existing example test harness can cover the new folder without rerunning models

## Non-Goals

- No new predictor code
- No new workflow routing
- No metal-binding feedback support
- No cross-workflow reuse
- No machine-learning training loop
