# Ni²⁺/Co²⁺ Selectivity-DES Example — Qwen 3.6

This example runs the full **selectivity-DES pipeline** using **Qwen 3.6** (Ollama) to find:

1. Five Ni²⁺-selective ligands that are hydrogen-bond donors or acceptors (Phase 1)
2. Deep eutectic solvents containing the top three of those ligands, with Tm < 350 K, viscosity ≤ 200 cP, and hydrophobic character (Phase 2)

## Scientific goal

Selective extraction of Ni²⁺ from Co²⁺-containing solutions is industrially important (battery recycling, hydrometallurgy). This run extends the [`ni_co_selectivity_des`](../ni_co_selectivity_des) example by shortlisting **five** ligands in Phase 1 instead of three, giving a wider selectivity pool before the DES search, and uses Qwen 3.6 as a lighter LLM alternative to Gemma4-12B.

## Pipeline architecture

```
Outer loop (2 cycles)
├── Phase 1: Ni²⁺/Co²⁺ selectivity screening
│   ├── Qwen 3.6 brainstorms HBD/HBA ligand candidates (20 per cycle × 3 cycles)
│   ├── Heuristic stability-constant model scores each ligand
│   ├── Score = 0.4 × log K(Ni²⁺) + 0.6 × ΔlogK   (selectivity-weighted)
│   └── Top 5 ligands (ΔlogK ≥ 0.3) pass to Phase 2
│
└── Phase 2: DES partner search for top-3 shortlisted ligands
    ├── 20 candidates per cycle × 3 cycles per ligand
    ├── ChemBERTa ML model predicts eutectic melting temperature
    ├── Viscosity model gates candidates above 200 cP
    ├── Acceptance criteria: Tm ≤ 350 K, viscosity ≤ 200 cP
    └── DES-compatible ligands feed back to Phase 1 as hints
```

## Prerequisites

- Ollama running locally with `qwen3.6` loaded:
  ```bash
  ollama pull qwen3.6
  curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin)['models']]"
  ```
- ChemBERTa checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Stability-constant artifact: `artifacts/stability_constants/model.json`
- Viscosity artifact: `artifacts/designsolvents/viscosity/model.json`
- Verify with:
  ```bash
  python -m des_multi_agent.cli doctor --check checkpoint --check artifacts
  ```

## Run

```bash
./run.sh
# or with a custom output directory:
./run.sh runs/my_ni_co_qwen36_run
```

Expected wall time: **8–18 minutes** (Qwen 3.6 is faster than Gemma4-12B for brainstorming).

## Output

See [`output.txt`](./output.txt) for the captured results. The file has two sections:

**Section 1 — Selectivity table** (Phase 1 final results, top 5 ligands)

| Column | Meaning |
|--------|---------|
| `ligand` | Candidate SMILES |
| `log_k_target` | Predicted log K for Ni²⁺ |
| `log_k_competitor` | Predicted log K for Co²⁺ |
| `delta_log_k` | Ni²⁺/Co²⁺ discrimination; positive = Ni²⁺-selective |
| `score` | Composite ranking score |
| `des_compatible` | Whether a DES partner was found in Phase 2 |

**Section 2 — DES partner blocks** (Phase 2 results, top 3 ligands)

Each ligand block lists DES partners with predicted Tm, viscosity threshold compliance, and eutectic molar ratio.

## Key parameters

| Parameter | Flag | Value | Effect |
|-----------|------|-------|--------|
| Ligands shortlisted | `--top-ligands` | 5 | More selective ligands explored than the default 3 |
| Selectivity emphasis | `--selectivity-weight` | 0.6 | ΔlogK weighted more than absolute affinity |
| Selectivity gate | `--min-delta-log-k` | 0.3 | Falls back to top-N if no candidate passes |
| Tm ceiling | `--abs-tm-threshold` | 350 K | Room-temperature DES target |
| Viscosity gate | `--viscosity-threshold` | 200 cP | Flowable solvent target |

## Comparison with Gemma4-12B run

See [`ni_co_selectivity_des/`](../ni_co_selectivity_des) for the same query using Gemma4-12B with `--top-ligands 3`. Comparing the two runs shows how LLM choice and shortlist size affect the diversity and quality of discovered DES partners.

## LLM config

[`llm.ni_co_qwen36.yaml`](../../llm.ni_co_qwen36.yaml) — edit `model_name` to switch models without changing anything else.
