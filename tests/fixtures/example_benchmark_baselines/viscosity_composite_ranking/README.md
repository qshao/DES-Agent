# Viscosity Composite Ranking Example

This example demonstrates viscosity-aware composite ranking using `--viscosity-threshold` and `--viscosity-weight`. DES-formers whose predicted viscosity exceeds the threshold sort below passing candidates regardless of melting temperature; `--viscosity-weight` controls how much viscosity blends into the composite score.

## Input

- Component A: `CCO` (ethanol)
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Viscosity model: `artifacts/designsolvents/viscosity/model.json`
- Viscosity threshold: `500` cP — candidates above this sort to the bottom
- Viscosity weight: `0.4` — 40% of the composite score comes from viscosity
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains two sections:

- **DES screening table**: candidates ranked by composite score (Tm-drop + viscosity blend).
- **Viscosity predictions**: predicted viscosity for each DES-positive pair in mPa·s.

All five candidates here fall well below the 500 cP threshold, so none are gated to the bottom. With a tighter threshold (e.g. `--viscosity-threshold 13`) glycerol and other high-viscosity candidates would sort after the lower-viscosity ones.

## How to Adapt

- Lower `--viscosity-threshold` to gate out high-viscosity candidates earlier.
- Adjust `--viscosity-weight` between `0` (ignore viscosity, rank by Tm only) and `1` (rank by viscosity only).
- If you want a template for your own viscosity study, start from [`examples/viscosity_template/`](../viscosity_template).
