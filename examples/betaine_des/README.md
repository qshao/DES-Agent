# Betaine DES Screening Example

This example searches for deep eutectic solvent (DES) partners for **betaine** (trimethylglycine, `C[N+](C)(C)CC(=O)[O-]`), a naturally occurring zwitterionic compound found in sugar beets and widely used in pharmaceutical, cosmetic, and green-solvent applications.

Betaine has a high pure-component melting point (~573 K), so a relaxed Tm ceiling of **340 K** is used with `--abs-tm-threshold` to capture partners that produce a large enough depression to be practically useful. Viscosity is a primary ranking criterion: `--viscosity-weight 0.7` gives 70% weight to the predicted viscosity in the composite score, selecting the most fluid DES candidates first.

Up to 5 iterative cycles (`--n-cycles 5`) are requested. Without an LLM, the heuristic generator returns a consistent candidate pool and convergence is typically detected by cycle 2; adding an LLM config (`--llm-config`) would enable family-aware brainstorming across all cycles for broader coverage.

## Input

- Component A: `C[N+](C)(C)CC(=O)[O-]` (betaine, zwitterionic form)
- Candidate search count per cycle: `20`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Tm ceiling: `340 K` (`--abs-tm-threshold 340`)
- Minimum relative Tm drop: `5%` (`--rel-drop-min 0.05`)
- Viscosity model: `artifacts/designsolvents/viscosity/model.json`
- Viscosity weight: `0.7` (viscosity dominates composite ranking)
- Max cycles: `5`
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains:

- **DES screening table**: 18 of 20 candidates predicted as DES-formers (Tm ≤ 340 K, relative drop ≥ 5%). Candidates are ranked by composite score with viscosity weighted at 70%. The lowest-viscosity DES-formers (DMSO at 13.1 mPa·s, DMF at 13.4 mPa·s, water at 12.3 mPa·s) sort to the top.
- **Viscosity predictions**: predicted mixture viscosity for each DES-positive pair in mPa·s.

## How to Adapt

- Change `--abs-tm-threshold` to tighten or relax the Tm acceptance window. For a stricter pharmaceutical target, try `280`; for wider coverage, try `380`.
- Lower `--viscosity-weight` (toward `0.3`) to weight Tm depression more heavily, or raise it toward `1.0` to rank almost entirely by low viscosity.
- Add `--llm-config llm.example.yaml` to enable multi-cycle chemical-family brainstorming; without it, convergence is fast because the heuristic pool is fixed.
- Replace `C[N+](C)(C)CC(=O)[O-]` with any other high-melting zwitterion or salt to screen a different target.
