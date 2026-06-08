# Metal-Binding Example

This example runs the metal-binding workflow with the bundled local stability-constant artifact.

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

The file [`output.txt`](./output.txt) contains the captured report output from the metal-binding workflow, including the predicted `log K` value. If you want a template for your own ligand-binding work, start from [`examples/ligand_binding_template/`](../ligand_binding_template).
