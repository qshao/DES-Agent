# Preset Thresholds Example

This example demonstrates `--preset`, the easiest way to tune DES acceptance thresholds without arithmetic. Three named presets are available:

| Preset | Tm ceiling | Min relative drop | Use when |
|--------|-----------|-------------------|----------|
| `strict` | 240 K | 15% | Industrial processes needing strongly-melting DES |
| `standard` | 260 K | 10% | Default — balanced selection |
| `relaxed` | 280 K | 5% | Exploratory screening; wider net |

The script runs two searches on ethanol (`CCO`) — first `strict`, then `relaxed` — so you can compare how the `is_des` column changes for the same five candidates.

## Input

- Component A: `CCO` (ethanol)
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Presets compared: `strict` and `relaxed`
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) shows two tables side by side. Under `strict`, water and acetic acid lose their `is_des=True` label because they fall above the 240 K ceiling. Under `relaxed` both are accepted again.

## How to Adapt

- Use `--preset strict` when you need a DES that works well below room temperature.
- Use `--preset relaxed` for initial screening where you want the broadest candidate set.
- Override individual thresholds with `--abs-tm-threshold` and `--rel-drop-min` when presets do not match your exact requirements.
