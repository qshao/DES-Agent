# DES Run Memory Feedback Example

This example shows the full offline DES feedback loop in one folder:

1. run DES once and save `run.memory.json` (or keep several labeled runs under `runs/` and reuse the parent history directory later)
2. label the saved memory in place with `label-run`
3. reuse the labeled memory on the next DES run
4. keep the candidate pool consistent with proposal-diversity controls
5. let chemical-pattern memory reuse prior DES lessons when the next cycle starts
6. show the chemistry lesson summary block in the report so the next cycle has a compact chemical note

## Input

- User-facing request: see [`input.txt`](./input.txt)
- Component A: `CCO`
- Memory file: [`run.memory.json`](./run.memory.json)

## Run

The wrapper resolves the repository root first, so you can run it from any working directory, and captures the three-step loop into a single transcript in [`output.txt`](./output.txt).

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains:

- the input request
- the initial DES run that writes `run.memory.json`
- the `label-run` step that updates the memory in place
- the second DES run that reuses the labeled memory
- the proposal-diversity settings threaded through both DES runs
- the chemical-pattern memory layer that reuses earlier chemical lessons
- the chemistry lesson summary block that now appears in the report output

## How to Adapt

Use this folder as a template for your own feedback loop:

- Replace the request in [`input.txt`](./input.txt) with your own DES target.
- Edit [`run.memory.json`](./run.memory.json) if you want to inspect or seed the label format. If you later build a larger labeled history under `runs/`, you can point `--reuse-run` at the parent history directory to reuse all of it.
- Update the labels in `run.sh` to match your own `good` and `bad` preferences.
- If you want broader or narrower candidate families, edit the proposal-diversity flags in `run.sh` too.
- If you want the next cycle to lean more or less on prior chemistry, edit the chemical-pattern-memory flags in `run.sh`.
- If you want the compact chemistry lesson to be shorter or broader, edit the example inputs and labels so the run has less or more evidence to summarize.
- If you want a different checkpoint, set `DES_CHECKPOINT_PATH` before running the wrapper.
- The feedback loop is DES-only; use the separate metal-binding examples for stability-constant work.
