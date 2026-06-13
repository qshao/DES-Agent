# Metal-Binding Example

This example runs the metal-binding workflow with the bundled local stability-constant artifact. It is a useful deterministic target for the chemistry-advisor layer when you want concise rationale or warning text around a predicted stability constant, but the current example itself stays numeric and local.

If you want to see the advisor output in the repo, this is the smallest place to add it because the input is already a single metal/ligand pair and the result is easy to explain.

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
