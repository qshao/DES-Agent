# DES-Agent: Introduction for Wet-Lab Researchers

This tutorial is for chemists and biochemists who want to use DES-Agent to prioritise candidates before going into the lab — no programming experience required beyond running commands in a terminal.

## What DES-Agent Does

DES-Agent predicts two things:

**Deep eutectic solvents (DES).** You have a molecule you want to use as component A (a hydrogen-bond donor or acceptor). DES-Agent screens a large library of candidate component B partners and ranks them by how low their predicted eutectic melting temperature is. Candidates with a lower predicted eutectic temperature, and a big drop relative to both pure components, are the ones worth synthesising and testing first.

**Metal–ligand stability constants (log K).** You have a metal ion (Cu²⁺, Ni²⁺, Zn²⁺, etc.) and want to know which chelating ligands bind most tightly, or which ligands are selective for your target metal over a competing ion. DES-Agent predicts log K values from structure alone, using a local model — no internet connection required.

Both predictions are made offline, in seconds, using trained machine-learning models. An optional AI language model can brainstorm additional candidates and explain predictions in plain language, but the predictions themselves are deterministic and reproducible without one.

---

## 1. Installation (5 minutes)

You need Python 3.11 or newer. Check with:

```bash
python --version
```

Clone the repository and install:

```bash
git clone https://github.com/qshao/DES-Agent.git
cd DES-Agent
pip install -e .
```

This registers the `des-agent` command. Verify it works:

```bash
python -m des_multi_agent.cli doctor
```

You should see something like:

```
[doctor] repo root OK
[doctor] checkpoint found: ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
[doctor] all checks passed
```

If the checkpoint is not found, run the doctor with the explicit path:

```bash
python -m des_multi_agent.cli doctor --check checkpoint
```

---

## 2. One-time Setup: Save Your Defaults

These two flags are needed for every DES search. Save them once so you don't have to type them every time:

```bash
python -m des_multi_agent.cli config set checkpoint_path=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
python -m des_multi_agent.cli config set config_path=ml_des_mp/config.yaml
```

From this point on, DES search commands only need `--component-a` and `--n`.

---

## 3. Your First DES Search

**Goal:** Find DES partners for choline chloride (a common DES component A).

Choline chloride is one of the most studied DES components. It forms eutectic mixtures with urea, ethylene glycol, glycerol, oxalic acid, and many others. Let's screen 20 candidates:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "choline chloride" \
  --n 20 \
  --output-dir runs/choline_01
```

> **Tip:** You can pass common molecule names instead of SMILES. DES-Agent recognises hundreds of names including "choline chloride", "urea", "glycerol", "ethanol", "betaine", "oxalic acid", "malonic acid", "ethylene glycol", and many others. Run `python -m des_multi_agent.cli list-molecules` for the full list.

The run takes 5–15 seconds. Progress is printed to the terminal:

```
[cycle 1/1] screened=20 top min_tm_k=210.4
```

Results are saved in `runs/choline_01/`. The main files:

| File | What it contains |
|------|-----------------|
| `report.txt` | Human-readable ranked table — start here |
| `run.csv` | Same table as a spreadsheet-ready CSV |
| `run.json` | Full structured data (for scripting) |

---

## 4. Reading the DES Report

Open `runs/choline_01/report.txt`. The header tells you the overall summary:

```
=== DES Search: choline chloride ([N+](CCO)(C)(C)C.[Cl-]) ===
Screened 20 candidate(s). 14 predicted DES-former(s) (min Tm ≤ 260 K with ≥10% relative drop).
Top candidate: urea (CN(C)=O) — min Tm 212.3 K (Δ24.1%) | high confidence
=================================================================
```

Then the ranked table:

```
compound         | is_des | min_tm_k | eutectic_x_b | source | trust | confidence
urea (CN(C=O)N) | True   | 212.3    | 0.67         | ...    | 0.91  | high confidence
glycerol (...)  | True   | 220.1    | 0.60         | ...    | 0.87  | high confidence
...
```

### What each column means

| Column | Meaning | Lab interpretation |
|--------|---------|-------------------|
| `is_des` | True if predicted DES-former | Use only `True` rows as synthesis candidates |
| `min_tm_k` | Predicted eutectic melting temperature (Kelvin) | Lower = better DES; subtract 273 to convert to °C |
| `eutectic_x_b` | Molar fraction of component B at the eutectic | Your mixing ratio: `x_b = 0.67` means 67 mol% component B |
| `trust` | Model confidence (0–1) | ≥ 0.80 = reliable; 0.60–0.80 = cross-check with literature |
| `confidence` | Plain-language tier | "high confidence" candidates have consistent ensemble predictions |
| `spread_k` | Range of fold predictions in K | Narrow spread (< 5 K) = model is certain; wide spread = verify |

### Converting predicted Tm to degrees Celsius

Subtract 273.15. A predicted `min_tm_k = 212.3 K` means approximately −61 °C, which is an unusually deep eutectic. Predicted `min_tm_k = 290 K` ≈ 17 °C — liquid at room temperature, good for processing.

### The relative drop (Δ%)

The header shows something like `Δ24.1%`. This is:

```
Δ% = (Tm_pure_minimum − min_tm_k) / Tm_pure_minimum × 100
```

It measures how far the eutectic dips below the melting point of the lowest-melting pure component. A Δ% of less than 5% is marginal (small eutectic effect); 10–15% is a genuine eutectic; 20%+ is a strong one worth prioritising.

### Which candidates to synthesise first

Prioritise candidates that have:
1. `is_des = True`
2. `min_tm_k` below 260 K (below −13 °C) for most applications, or below your application-specific ceiling
3. `confidence = high confidence` or `trust ≥ 0.80`
4. A `Δ%` of at least 10%

A typical first batch from a 20-candidate screen is 3–5 compounds.

---

## 5. Tightening or Relaxing the DES Criteria

Use built-in presets instead of remembering threshold numbers:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "choline chloride" \
  --n 20 \
  --preset strict
```

| Preset | Tm ceiling | Minimum Δ% | Use when |
|--------|------------|-----------|---------|
| `strict` | 240 K (−33 °C) | 15% | Deep eutectics only, very conservative |
| `standard` | 260 K (−13 °C) | 10% | Default — good for most applications |
| `relaxed` | 280 K (7 °C) | 5% | Room-temperature liquids, broader search |

---

## 6. Multi-Cycle Screening: Drilling Deeper

With `--n-cycles`, DES-Agent automatically expands the search by generating structural analogues of the best candidates from each cycle, then screening them in the next cycle. This is useful when the initial screen returns only 2–3 hits and you want to explore nearby chemical space.

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "choline chloride" \
  --n 25 \
  --n-cycles 3 \
  --output-dir runs/choline_multicycle
```

The terminal prints one line per cycle:

```
[cycle 1/3] screened=25 top min_tm_k=212.3
[cycle 2/3] screened=22 top min_tm_k=208.9
[cycle 3/3] top-5 shortlist unchanged — converged early
```

The search converges early when the top-5 ranked candidates don't change between cycles. `trajectory.md` (also saved to `--output-dir`) gives a cycle-by-cycle narrative of what changed.

---

## 7. Metal-Binding Prediction: Single Ligand

To predict the stability constant (log K) for one specific metal-ion/ligand pair:

```bash
python -m des_multi_agent.cli \
  --workflow metal-binding \
  --metal-ion "Cu2+" \
  --ligand-smiles "NCCN"
```

Output:

```
metal_ion | ligand_smiles | value | units
Cu2+      | NCC(=O)O      | 11.31 | log K
```

`log K = 11.31` means the stability constant is 10¹¹·³¹. As a rough guide:

| log K | Binding strength |
|-------|-----------------|
| < 4   | Weak — ligand likely to dissociate under mild conditions |
| 4–8   | Moderate — useful for extraction or transport applications |
| 8–14  | Strong — good for chelation therapy, analytical chemistry |
| > 14  | Very strong — approaching irreversible complexation |

### Passing SMILES for uncommon ligands

For molecules not in the name list, pass SMILES directly. If you don't know the SMILES for your ligand, draw it in ChemDraw, MarvinSketch, or the free online tool Ketcher, then copy the SMILES from the export menu. Common SMILES for reference:

| Ligand | SMILES |
|--------|--------|
| Glycine | `NCC(=O)O` |
| Ethylenediamine (en) | `NCCN` |
| EDTA (fully deprotonated) | `OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O` |
| 8-Hydroxyquinoline (8-HQ) | `Oc1cccc2ncccc12` |
| 2,2′-Bipyridyl (bipy) | `c1ccc(-c2ccccn2)nc1` |
| 1,10-Phenanthroline (phen) | `c1cnc2ccc3ncccc3c2c1` |
| Acetylacetone (acac) | `CC(=O)CC(C)=O` |

---

## 8. Metal Selectivity Screening

**Goal:** Find ligands that bind Cu²⁺ selectively over Zn²⁺ (or any other metal pair).

```bash
python -m des_multi_agent.cli \
  --workflow metal-selectivity \
  --target-metal-ion "Cu2+" \
  --competitor-metal-ion "Zn2+" \
  --n 15 \
  --output-dir runs/cu_zn_selectivity
```

Sample output (top rows):

```
ligand                     | log_k_target | log_k_competitor | delta_log_k | score
8-hydroxyquinoline         | 11.19        | 10.91            | +0.28       | 5.73
2,2-bipyridine             | 10.81        | 10.46            | +0.35       | 5.58
catecholate                | 11.12        | 10.98            | +0.14       | 5.63
nitrilotriacetate (NTA)    | 14.94        | 15.02            | −0.08       | 7.43
```

### Reading the selectivity table

| Column | Meaning | Lab interpretation |
|--------|---------|-------------------|
| `log_k_target` | Predicted log K for your target metal (Cu²⁺) | Absolute binding strength |
| `log_k_competitor` | Predicted log K for the competing metal (Zn²⁺) | How well it binds the unwanted ion |
| `delta_log_k` | `log_k_target − log_k_competitor` | **Key number for selectivity** |
| `score` | Composite ranking score (considers both strength and selectivity) | Used for sorting; higher = better overall |

`delta_log_k > 0` means the ligand prefers your target metal. `delta_log_k < 0` means it prefers the competitor. For practical selectivity, aim for `delta_log_k ≥ 1` — a tenfold preference.

In the example above, 2,2-bipyridine (Δlog K = +0.35) shows modest Cu²⁺ preference, while NTA (Δlog K = −0.08) is essentially non-selective despite its very high absolute binding.

### Supported metal ions

Pass metal ions in the format `Cu2+`, `Zn2+`, `Fe3+`, `Ni2+`, `Co2+`, `Mn2+`, `Ca2+`, `Mg2+`, etc. Check which ions have explicit model features:

```bash
python -m des_multi_agent.cli supported-metals
```

Ions not on that list still get a predicted value via the fallback model path, but selectivity predictions between two unsupported ions are less reliable.

### Optional: DFT refinement with `--dft-validate`

The rule-based selectivity ranking uses predicted log K values and HSAB hard/soft classification. For borderline cases — where two candidates have similar ΔlogK or the HSAB assignment is ambiguous — you can add a quantum-chemistry check that looks directly at the ligand's electron structure.

**What it does.** When you pass `--dft-validate`, DES-Agent runs a density functional theory (DFT) calculation on the top candidates (default: top 3). The calculation predicts the energy of the ligand's highest occupied molecular orbital (HOMO), which is a direct quantum-mechanical measure of how "soft" or "hard" the donor atoms are.

- A **high HOMO energy** (less negative, e.g. −7.5 eV) means the ligand is a **soft donor** — its electrons are more loosely held and it prefers soft metal ions (Cu⁺, Hg²⁺, Pd²⁺, Ag⁺).
- A **low HOMO energy** (more negative, e.g. −9.5 eV) means the ligand is a **hard donor** — oxygen-rich, less polarisable, prefers hard metals (Fe³⁺, Ca²⁺, Mg²⁺, Al³⁺).

For a Cu²⁺ vs Zn²⁺ competition: Cu²⁺ sits on the borderline-soft side while Zn²⁺ is borderline-hard. A ligand whose HOMO energy sits closer to the soft end (−7.5 to −8.5 eV) may be predicted as marginally Cu²⁺-selective by DFT even when the log K difference is small. The DFT result nudges the composite score up or down by at most ±0.05, breaking ties without overriding the rule-based ranking.

**When to use it.**
- You have 2–3 top candidates with similar ΔlogK (< 0.5) and want a tiebreaker before committing synthesis time.
- Your ligand contains borderline donors (imidazole N, thioether S, phenolate O) whose hardness/softness the simple HSAB table may not capture correctly.
- You want to record the quantum-chemical donor character of the candidate alongside the stability constant data.

**When not to bother.**
- Your top candidate already has ΔlogK ≥ 1.0 — the rule-based ranking is unambiguous.
- Your ligand is a simple carboxylate or amine where hard/soft assignment is clear.
- You need a fast screen across many metal pairs — DFT adds 1–5 minutes per candidate.

**Prerequisites.** DFT requires two extra packages that are not installed by default:

```bash
pip install gpu4pyscf xtb-python
```

`gpu4pyscf` runs the B3LYP/def2-SVP DFT calculation; `xtb-python` runs a fast pre-optimisation of the 3D geometry. A GPU is not required — the calculation runs on CPU if no GPU is available, but will be slower (2–5 minutes per candidate vs 15–30 seconds on GPU).

**Command.**

```bash
python -m des_multi_agent.cli \
  --workflow metal-selectivity \
  --target-metal-ion "Cu2+" \
  --competitor-metal-ion "Zn2+" \
  --n 20 \
  --stability-constant-model-path artifacts/stability_constants/model.json \
  --dft-validate \
  --dft-top-n 3
```

DES-Agent will first run the standard selectivity screen, then the LLM (or fallback: top 3 by score) nominates candidates for DFT. A startup check confirms the packages are installed before any computation begins — if either is missing you will see a clear error message with the install command.

**DFT now reflects the actual ionization state at pH 7.0, not the drawn neutral structure.** If your ligand has an ionizable group (a carboxylic acid, a phenol, an amine, etc.), DES-Agent computes the DFT properties of the form that actually dominates at pH 7.0 — for example, a carboxylic acid ligand is computed as its deprotonated carboxylate anion, since that's what's really present in solution at that pH. This gives a more physically realistic HOMO energy than assuming every ligand stays neutral. You don't need to do anything differently — this happens automatically whenever `--dft-validate` is on.

**Repeat candidates are cached.** Every DFT result is saved to `artifacts/dft_cache/dft_results.sqlite3`, keyed by the exact ionized structure and method. If the same ligand comes up again — in a later cycle of the same run, or in an entirely separate run — DES-Agent reuses the cached result instead of recomputing it, so multi-cycle screens with overlapping candidates get dramatically faster after the first pass. You never need to manage this cache yourself; if you want to force a clean recomputation (e.g. after updating gpu4pyscf), simply delete the `artifacts/dft_cache/` directory.

**Reading the DFT columns in the report.**

When `--dft-validate` is active, the selectivity table gains two extra columns for nominated candidates:

```
ligand         | delta_log_k | score | dft_homo_ev | dft_donor_chg
NCCN           | +1.40       | 0.89  | -8.12       | -0.30
NCC(=O)O       | +1.20       | 0.81  | -9.21       | -0.35
c1ccncc1       | +0.80       | 0.74  | —           | —
```

| Column | What it means | How to interpret |
|--------|--------------|-----------------|
| `dft_homo_ev` | HOMO energy in electronvolts (eV) | −7.5 to −8.5 eV = soft donor; −9.0 to −10 eV = hard donor |
| `dft_donor_chg` | Average Löwdin charge on donor atoms (N, O, S, P) | More negative = more electron-rich donor sites; correlates with Lewis basicity |
| `—` | DFT was not run or did not converge for this candidate | Rule-based score used unchanged |

A `[DFT]` summary block at the end of the report lists the method (`B3LYP-D3(BJ)/def2-SVP`) and any per-candidate warnings (e.g. geometry failed to converge). If all DFT calculations fail, the rule-based ranking is used unchanged and the run still completes normally.

---

## 9. The Selectivity-DES Pipeline: Two Steps in One

If you want ligands that are selective for a target metal AND can form DES with a partner molecule, you can run both steps together:

```bash
python -m des_multi_agent.cli \
  --workflow selectivity-des \
  --target-metal-ion "Ni2+" \
  --competitor-metal-ion "Co2+" \
  --n 10 \
  --output-dir runs/ni_co_des
```

Phase 1 screens ligands for Ni²⁺/Co²⁺ selectivity. Phase 2 takes the top selective ligands as component A and searches for DES partners. This is the workflow used in nickel/cobalt hydrometallurgical separation research.

---

## 10. Saving Run Results and Giving Feedback

### Saving results

Always use `--output-dir` for real work. It writes five files you can keep for your lab notebook:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "urea" \
  --n 20 \
  --output-dir runs/urea_screen_2026-07-01
```

The date-stamped directory becomes a permanent record of which candidates the model ranked on which day.

### Labelling results after your experiments

After you synthesise and test candidates in the lab, you can tell DES-Agent which ones worked and which didn't. This biases future searches toward productive chemical families.

Use the `label-run` command — pass molecule names or SMILES and a `good`/`bad` label for each:

```bash
python -m des_multi_agent.cli label-run \
  --run runs/urea_screen_2026-07-01 \
  --label "ethylene glycol=good" \
  --label "oxalic acid=bad"
```

Then pass the labelled run directory to your next search with `--reuse-run`:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "urea" \
  --n 20 \
  --reuse-run runs/urea_screen_2026-07-01 \
  --output-dir runs/urea_screen_round2
```

The next run will rank `"good"` families higher and `"bad"` families lower, improving candidate quality over successive rounds.

### Accumulating memory across many runs

`--reuse-run` also accepts a history directory — pass the parent folder containing all your run directories and every labelled result will contribute:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "urea" \
  --n 20 \
  --reuse-run runs/ \
  --output-dir runs/urea_screen_round3
```

---

## 11. Comparing Two Runs

After you have two runs saved, compare their top candidates:

```bash
python -m des_multi_agent.cli compare-runs \
  runs/urea_screen_2026-07-01 \
  runs/urea_screen_round2
```

This shows which candidates entered or left the top-N shortlist between runs, useful for tracking how feedback is changing the search direction.

---

## 12. Quick Reference: Common Molecule Names

DES-Agent accepts these names directly (case-insensitive). Pass them as `--component-a "name"` or `--ligand-smiles "name"`:

**Common DES component A molecules:**
- choline chloride, ChCl
- urea
- glycerol
- ethylene glycol
- malonic acid, oxalic acid, citric acid
- betaine
- lactic acid
- choline acetate

**Common DES component B targets:**
- urea, thiourea
- glycerol, sorbitol
- ethylene glycol, propylene glycol
- acetic acid, propionic acid

**Common chelating ligands (metal binding):**
- glycine, alanine
- ethylenediamine
- EDTA (pass as SMILES: `OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O`)
- 8-hydroxyquinoline (pass as SMILES: `Oc1cccc2ncccc12`)
- 2,2-bipyridine (pass as SMILES: `c1ccc(-c2ccccn2)nc1`)

For all supported names:

```bash
python -m des_multi_agent.cli list-molecules
```

---

## 13. Threshold Summary

### DES thresholds

| `min_tm_k` | °C equivalent | Meaning |
|------------|---------------|---------|
| < 210 K | < −63 °C | Exceptionally deep eutectic — rare, double-check |
| 210–240 K | −63 to −33 °C | Strong DES — high priority for synthesis |
| 240–260 K | −33 to −13 °C | Good DES — standard target range |
| 260–280 K | −13 to 7 °C | Marginal DES — liquid at room temperature, mild eutectic |
| > 280 K | > 7 °C | Predicted non-DES or very weak eutectic |

### Metal-binding thresholds

| `log K` | Interpretation |
|---------|---------------|
| < 4 | Negligible binding |
| 4–8 | Weak-to-moderate (useful in extraction, transport) |
| 8–12 | Strong (analytical reagent-grade) |
| 12–16 | Very strong (chelation therapy range) |
| > 16 | Exceptionally stable complex |

### Confidence levels

| Level | `trust` score | Recommendation |
|-------|--------------|----------------|
| high confidence | ≥ 0.80 | Proceed to synthesis |
| moderate-high | 0.65–0.80 | Synthesise, but confirm melting point experimentally |
| moderate | 0.50–0.65 | Cross-check against literature before committing |
| low confidence | < 0.50 | Treat as a hypothesis; experimental confirmation essential |

---

## 14. Getting Help

If a run fails or gives unexpected output, the `doctor` command re-validates the install:

```bash
python -m des_multi_agent.cli doctor --check checkpoint --check artifacts
```

View the results of a previously saved run directory without re-running:

```bash
python -m des_multi_agent.cli view-run runs/choline_01
```

The comprehensive developer reference with all flags and options is in [`docs/tutorial.md`](tutorial.md). Bug reports go to the project issue tracker.

---

## Summary of Commands

| Task | Command |
|------|---------|
| Health check | `python -m des_multi_agent.cli doctor` |
| Save config defaults | `python -m des_multi_agent.cli config set checkpoint_path=...` |
| List supported names | `python -m des_multi_agent.cli list-molecules` |
| List supported metals | `python -m des_multi_agent.cli supported-metals` |
| DES screening | `python -m des_multi_agent.cli --workflow des --component-a "name" --n 20 --output-dir runs/X` |
| DES (multi-cycle) | add `--n-cycles 3` |
| DES (strict preset) | add `--preset strict` |
| Metal binding (single) | `python -m des_multi_agent.cli --workflow metal-binding --metal-ion "Cu2+" --ligand-smiles "NCCN"` |
| Metal selectivity screen | `python -m des_multi_agent.cli --workflow metal-selectivity --target-metal-ion "Cu2+" --competitor-metal-ion "Zn2+" --n 15` |
| Metal selectivity + DFT refinement | add `--dft-validate --dft-top-n 3` (requires `gpu4pyscf` and `xtb-python`) |
| Selectivity + DES pipeline | `python -m des_multi_agent.cli --workflow selectivity-des --target-metal-ion "Ni2+" --competitor-metal-ion "Co2+" --n 10` |
| Compare two runs | `python -m des_multi_agent.cli compare-runs runs/A runs/B` |
| View a saved run | `python -m des_multi_agent.cli view-run runs/X` |
| Label results | `python -m des_multi_agent.cli label-run --run runs/X --label "name=good"` |
| DES with feedback | add `--reuse-run runs/X` (run dir, file, or history dir) |
