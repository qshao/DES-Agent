# Template-Style DES Viscosity Example

This folder is a template-style DES viscosity example you can adapt for your own work.

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

The file [`output.txt`](./output.txt) contains the captured report output from the DES workflow, including the viscosity prediction section.

## How to Adapt

Use this folder as a template for your own viscosity study:

- Replace `component_a` in [`input.txt`](./input.txt) with your own DES component.
- Increase or decrease `n` depending on how many candidate partners you want the router and predictor to inspect.
- If you have a different local checkpoint, update `checkpoint_path`.
- If you have a different local viscosity artifact, update `viscosity_model_path`.
- If you want the workflow to search local literature or candidate libraries, add a `discovery_path` in the command or wrapper.

The workflow remains offline as long as the checkpoint and artifact paths point to local files.
