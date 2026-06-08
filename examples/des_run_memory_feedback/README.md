# DES Run Memory Feedback Example

This example shows the full offline DES feedback loop in one folder:

1. run DES once and save `run.memory.json`
2. label the saved memory in place with `label-run`
3. reuse the labeled memory on the next DES run

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

## How to Adapt

Use this folder as a template for your own feedback loop:

- Replace the request in [`input.txt`](./input.txt) with your own DES target.
- Edit [`run.memory.json`](./run.memory.json) if you want to inspect or seed the label format.
- Update the labels in `run.sh` to match your own `good` and `bad` preferences.
- If you want a different checkpoint, set `DES_CHECKPOINT_PATH` before running the wrapper.
- The feedback loop is DES-only; use the separate metal-binding examples for stability-constant work.
