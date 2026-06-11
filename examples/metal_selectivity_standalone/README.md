# Metal Selectivity Standalone Example

This example demonstrates `--workflow metal-selectivity`, which screens a library of chelating ligands and ranks them by their ability to bind one metal ion preferentially over another — without running the DES partner search.

Use this workflow when you want to:

- Identify selective ligands before committing to a full selectivity-DES pipeline
- Quickly compare selectivity across different metal pairs
- Use the shortlisted ligands as `--component-a` inputs for a separate DES run

The full two-phase pipeline (`--workflow selectivity-des`) chains this screen directly into DES partner search; see [`examples/ni2_co2_selectivity/`](../ni2_co2_selectivity) for that workflow.

## Input

- Target metal: `Cu2+`
- Competitor metal: `Zn2+`
- Candidate count: `5`
- Stability model: `artifacts/stability_constants/model.json`
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the ranked selectivity table. Key columns:

| Column | Meaning |
|--------|---------|
| `log_k_target` | Predicted log stability constant for `Cu2+` |
| `log_k_competitor` | Predicted log stability constant for `Zn2+` |
| `delta_log_k` | `log_k_target − log_k_competitor`; positive = selective for Cu2+ |
| `score` | Composite selectivity score used for ranking |

## How to Adapt

- Swap `--target-metal-ion` and `--competitor-metal-ion` for any supported ion pair (e.g. `Ni2+` / `Co2+`, `Fe3+` / `Fe2+`).
- Increase `--n` to screen more ligand candidates per cycle.
- Add `--n-cycles` for iterative refinement where top ligands from each cycle seed the next brainstorm.
- Pass `--affinity-weight` and `--selectivity-weight` to tune the composite score balance.
