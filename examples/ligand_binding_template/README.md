# Template-Style Ligand-Binding Example

This folder is a template-style metal-binding example you can adapt for your own work.

## Input

- Metal ion: `Cu2+`
- Ligand SMILES: `NCCN`
- Stability model: `artifacts/stability_constants/model.json`
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the captured report output from the metal-binding workflow, including the predicted `log K` value.

## How to Adapt

Use this folder as a template for your own ligand-binding study:

- Replace `metal_ion` in [`input.txt`](./input.txt) with the target ion you want to study.
- Replace `ligand_smiles` with your own ligand or chelator.
- If you have a different local stability-constant artifact, update `stability_constant_model_path`.
- Keep the workflow on the metal-binding CLI path so the stability model is only used in the correct context.

The workflow remains offline as long as the stability model path points to a local file.
