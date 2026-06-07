# DES Viscosity Example

This example runs the DES workflow with the bundled local DESignSolvents viscosity artifact.

## Input

- Component A: `CCO`
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Viscosity model: `artifacts/designsolvents/viscosity/model.json`
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the captured report output from the DES workflow, including the viscosity prediction section. If you want a template for your own DES work, start from [`examples/viscosity_template/`](../viscosity_template).
