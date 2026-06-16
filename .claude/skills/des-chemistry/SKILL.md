---
name: des-chemistry
description: "Reference for DES-Agent chemistry modules: coordination perception, Irving-Williams/HSAB stability rules, LLM claim verification, H-bond complementarity scoring, and the physical DES eutectic model. Use when reasoning about DES formation, metal-ligand stability, or cross-checking LLM chemistry claims."
---

# DES-Agent Chemistry Reference

This skill is a **reference card** for the chemistry knowledge modules in
`des_multi_agent/chemistry/`. It describes what each module does, its public
API, and when to use it.

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `coordination.py` | Perceive donor atoms and denticity from a SMILES string |
| `stability_rules.py` | Irving-Williams / HSAB rule-based stability constants for metal selectivity |
| `claim_verification.py` | Cross-check LLM coordination and selectivity claims against structure |
| `hbond.py` | DES H-bond complementarity scoring (HBD↔HBA balance) |

---

## `coordination.py` — Coordination Profile

**When to use:** any time you need to know what donor atoms a ligand has,
how many coordinating sites it presents to a metal, or what chelate ring
sizes it would form.

```python
from des_multi_agent.chemistry.coordination import coordination_profile, DONOR_SOFTNESS

prof = coordination_profile("NCC(=O)O")   # glycine
# CoordinationProfile(
#   n_donor_atoms=3, denticity=2,
#   donor_site_elements=("N", "O"),   # carboxylate two O collapse to one site
#   chelate_ring_sizes=(5,),
#   mean_donor_softness=0.17,
#   donor_element_counts={"N": 1, "O": 2},
# )
```

**Key fields:**

| Field | Meaning |
|-------|---------|
| `n_donor_atoms` | Total N/O/S/P atoms with available lone pairs |
| `denticity` | Number of coordinating *sites* (carboxylate O pair → 1 site) |
| `chelate_ring_sizes` | Tuple of ring sizes (metal + path length); 5- and 6-membered are stable |
| `mean_donor_softness` | 0 = hard (O/F), 1 = soft (S/P); 0.5 = borderline (N) |

**HSAB softness table:** `DONOR_SOFTNESS = {"O": 0.0, "F": 0.0, "N": 0.5, "Cl": 0.6, "S": 1.0, "P": 1.0}`

---

## `stability_rules.py` — Irving-Williams + HSAB Scoring

**When to use:** comparing two metals (e.g. Ni²⁺ vs Co²⁺) for a given ligand;
blending rule-based selectivity into the heuristic stability-constant model.

```python
from des_multi_agent.chemistry.stability_rules import (
    irving_williams_offset,
    hsab_match,
    rule_based_log_k,
    selectivity_delta_log_k,
)

# Irving-Williams series offset (Mn < Fe < Co < Ni < Cu > Zn)
irving_williams_offset("Ni2+")   # → 1.5
irving_williams_offset("Co2+")   # → 1.0

# HSAB donor-acceptor match (0=mismatch, 1=perfect match)
hsab_match("Cu2+", "CCS")        # soft metal + soft S-donor → ~0.9

# Full rule-based log K estimate
rule_based_log_k("Ni2+", "NCC(=O)O")   # glycine with Ni²⁺

# ΔlogK > 0 means target is more stable than competitor
selectivity_delta_log_k("Ni2+", "Co2+", "NCC(=O)O")   # → +0.50
```

**Irving-Williams series (relative offsets):**

```
Mn2+  Fe2+  Co2+  Ni2+  Cu2+  Zn2+
 0.0   0.6   1.0   1.5   2.3   0.9
```

**Formula:** `log K ≈ BASE + IW_offset + 1.5·HSAB_match + 0.8·(denticity−1) + 0.5·|charge| + 0.1·n_donors`

---

## `claim_verification.py` — LLM Claim Cross-Checker

**When to use:** after an LLM proposes coordination modes or selectivity claims,
verify them against the actual structure.

```python
from des_multi_agent.chemistry.claim_verification import (
    verify_coordination_claim,
    verify_selectivity_claim,
    batch_verify_coordination,
)

# Coordination claim: parse "bidentate N,O-chelator" and check against SMILES
r = verify_coordination_claim("NCC(=O)O", "bidentate N,O-chelator")
# ClaimVerification(verdict="ok", actual_denticity=2, ...)

r2 = verify_coordination_claim("CCCC", "bidentate N,O-chelator")
# ClaimVerification(verdict="not_a_ligand", ...)

# Selectivity claim: does the rule-based model support "Ni²⁺ favoured over Co²⁺"?
sv = verify_selectivity_claim("Ni2+", "Co2+", "NCC(=O)O")
# SelectivityVerification(delta_log_k=+0.50, verdict="target_selective")
```

**Verdicts:**

| Verdict | Meaning |
|---------|---------|
| `"ok"` | Claim matches actual structure |
| `"denticity_mismatch"` | Wrong number of coordinating sites (outside ±1 tolerance) |
| `"donor_mismatch"` | Claimed donor element absent from structure |
| `"not_a_ligand"` | No donor atoms in molecule at all |
| `"unparseable"` | No parseable coordination information in claim text |

**Claim text parsing:** looks for dental keywords ("bidentate", "tridentate", …)
followed by a comma-separated element list ("N,O", "N,N,N"). Falls back to
comma-separated NOSP anywhere in text. Does NOT extract from general words like
"monodentate" (avoids matching 'N' inside "moNodentate").

---

## `hbond.py` — DES H-Bond Complementarity

**When to use:** quickly screening whether a candidate pair forms a viable DES
based on H-bond donor/acceptor balance, without running the ML model.

```python
from des_multi_agent.chemistry.hbond import (
    hbond_profile,
    des_hbond_complementarity,
    rank_by_hbond,
)

# Single-molecule profile
p = hbond_profile("NC(=O)N")   # urea
# HBondProfile(n_hbd=2, n_hba=1, role="amphoteric", capacity=3)

# Pair complementarity
r = des_hbond_complementarity("C[N+](C)(C)CCO.[Cl-]", "NC(=O)N")
# HBondComplementarity(complementarity_score=0.67, composite_score=0.71, label="strong")

# Rank many candidates against one fixed component
best = rank_by_hbond("C[N+](C)(C)CCO.[Cl-]", candidate_smiles_list)
# Returns [(smiles, HBondComplementarity), ...] sorted best-first
```

**Scoring formula:**
- `complementarity_score` = average of forward (A-donor↔B-acceptor) and reverse ratios, 0–1
- `capacity_score` = `min((HBD_A + HBA_A + HBD_B + HBA_B) / 10, 1)`
- `composite_score` = `0.6 × complementarity + 0.4 × capacity`

**Labels:** `"strong"` (≥0.6), `"moderate"` (≥0.35), `"weak"` (>0), `"none"` (0)

> **Note:** RDKit Lipinski counts `HBD=HBA=0` for bare water ("O") — a known edge
> case. Methanol and all drug-like DES components score correctly.

---

## Physical DES Eutectic Model

The DES eutectic temperature comes from `predict_curve()` in
`des_multi_agent/prediction.py`. It uses:

- **DESPhysicsModel**: `Tm = max(T1/d1(x), T2/d2(x))` with learned d1, d2, W
- **MC-dropout** (p=0.2 on physics nets) with `_MC_SEED=12345` for reproducibility
- **Melting point resolver** (layered): override → experimental (0.95) → QSPR (0.40–0.85) → heuristic (0.35)

See `des_multi_agent/property_resolution.py` and `des_multi_agent/predictors/melting_point.py`.

---

## Integration Points

### Metal selectivity screen
`metal_binding_selectivity.py → _score_proposal_pair()` blends rule-based
`rule_based_log_k()` into heuristic scores via `stability_rule_weight` (default 0.5).

### Claim verification after LLM brainstorm
After `provider.assess_candidate_chemistry()` returns `ChemistryAssessment` objects,
call `verify_coordination_claim(smiles, assessment.rationale)` to flag structural
mismatches before reporting them to the user.

### H-bond pre-filter
Before running the expensive ML eutectic prediction, call `des_hbond_complementarity()`
as a cheap structural pre-filter. Pairs labelled `"none"` are unlikely DES formers
and can be deprioritised.

---

## Quick Reference: Common DES Pairs

| Pair | Roles | Label |
|------|-------|-------|
| Choline Cl + urea | HBA/HBD + amphoteric | strong (reline) |
| Choline Cl + oxalic acid | HBA/HBD + HBD | moderate |
| Betaine + glycerol | HBA + amphoteric | moderate |
| Menthol + thymol | weak HBD + weak HBD | weak |
