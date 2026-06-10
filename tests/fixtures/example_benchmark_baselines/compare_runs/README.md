# Compare Runs Example

This example shows how to save two DES screening runs and compare them with `compare-runs`. This is useful for seeing which candidates appeared, were removed, or changed rank when you tweak parameters such as `--n`, thresholds, or uncertainty settings.

## Input

- Component A: `CCO` (ethanol)
- Run A: `n=3`, checkpoint `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Run B: `n=5`, same checkpoint
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains three sections:

- **RUN A**: DES screening table for n=3 (3 candidates).
- **RUN B**: DES screening table for n=5 (5 candidates).
- **COMPARE**: a diff table with one row per candidate and these columns:
  - `status`: `new` (only in right), `removed` (only in left), `moved` (rank changed), `unchanged`
  - `left_rank` / `right_rank`: rank in each run (`-` if absent)

In this example, expanding from n=3 to n=5 adds `acetamide` and `acetic acid` as new candidates and shifts the ranks of the three shared candidates.

## How to Adapt

- Compare runs with different `component_a` values to track how the candidate pool changes across targets.
- Compare before and after labeling run memory with `label-run` to see ranking shifts from reuse.
- Pass `--json` to `compare-runs` for a machine-readable diff: `python -m des_multi_agent.cli compare-runs run_a.json run_b.json --json`.
