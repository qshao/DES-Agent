# Ni2+ / Co2+ Selectivity Screening Example

This example screens ligands for **Ni2+ selectivity over Co2+** with an explicit
requirement that candidates carry both hydrogen bond donor (HBD) and hydrogen
bond acceptor (HBA) groups.  The composite score balances binding affinity and
metal discrimination:

```
composite_score = 0.5 × log_K(Ni2+) + 0.5 × ΔlogK
where ΔlogK = log_K(Ni2+) − log_K(Co2+)
```

## Input

- Target metal: `Ni2+`
- Competitor metal: `Co2+`
- Candidate count per cycle: `20`
- Number of cycles: `3`
- Affinity weight: `0.5`  
- Selectivity weight: `0.5`
- Constraints: `min_hbd=1, min_hba=1` (at least one HBD and one HBA group)
- Stability model: `artifacts/stability_constants/model.json`
- Captured parameters: [`input.txt`](./input.txt)

## Run

Without an LLM (heuristic-only, cycle 1 only):

```bash
./examples/ni2_co2_selectivity/run.sh
```

With an LLM for all three brainstorm cycles:

```bash
./examples/ni2_co2_selectivity/run.sh llm.example.yaml
```

You can also invoke the Python script directly:

```bash
python examples/ni2_co2_selectivity/run.py             # heuristic only
python examples/ni2_co2_selectivity/run.py llm.example.yaml  # with LLM
```

## Multi-Cycle Behaviour

| Mode | Cycle 1 | Cycle 2 | Cycle 3 |
|------|---------|---------|---------|
| No LLM | 20 heuristic candidates | exhausted → stops | — |
| With LLM | 10 heuristic + 10 LLM | 20 LLM (HBD/HBA focus) | 20 LLM (converges or new) |

Without an LLM the heuristic library (20 entries) is exhausted after the first
cycle and the loop exits early.  With an LLM, cycles 2 and 3 add brainstormed
candidates guided by the HBD/HBA constraints and the top hits from the
previous cycle.

If you enable the LLM path, the same proposal-diversity controls used by the selectivity-DES examples can be applied to keep the brainstorm broader or more focused.

## Output

The file [`output.txt`](./output.txt) contains the selectivity report from a
heuristic-only run.  The table columns are:

- `log_k_target` — predicted log K for Ni2+
- `log_k_competitor` — predicted log K for Co2+
- `delta_log_k` — selectivity gap (Ni2+ − Co2+)
- `score` — composite score used for ranking
- `source` — `heuristic` (this run) or `llm` (with LLM enabled)
- `rationale` — coordination family and binding mode

## Why Ni2+ vs Co2+

Ni2+ and Co2+ are adjacent first-row transition metals with very similar ionic
radii and charge.  The heuristic predictor distinguishes them via d-electron
count (Ni2+ has 8, Co2+ has 7) and group number, giving a consistent ΔlogK of
≈ 0.08 across the heuristic library.  An LLM brainstorm can propose ligands
that exploit subtle geometric preferences (square-planar Ni2+ vs octahedral
Co2+) for larger selectivity gains.
