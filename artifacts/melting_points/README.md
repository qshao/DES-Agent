# Pure-component melting-point resolution

`des_multi_agent.property_resolution.resolve_melting_point` supplies the
pure-component melting points (`T1`, `T2`) that the eutectic physics model is
anchored on. It resolves each component through layers, most-trusted first, and
records the `source` and a `confidence` on every estimate:

| Layer | Source | Confidence | Notes |
|-------|--------|-----------|-------|
| 1 | `override` | 1.00 | Caller-supplied value |
| 2 | `experimental` | 0.95 | InChIKey lookup in `experimental.json` |
| 3 | `qspr` | 0.40–0.85 | ChemBERTa ensemble; confidence from spread |
| 4 | `heuristic` | 0.35 | RDKit descriptor fallback |

The eutectic trust score in `des_multi_agent/uncertainty/model.py` is bounded by
the weaker of the two input confidences, so a eutectic built on guessed melting
points is reported as less trustworthy than one built on experimental values.

## Files

- `experimental.json` — InChIKey → experimental Tm, built from the project's own
  training set (the values the physics model was trained on, so using them at
  inference removes train/inference skew). **Committed.**
- `qspr_model.pt` — trained ChemBERTa-embedding ensemble. **Not committed**
  (matches the repo's `*.pt` convention for model weights); regenerate locally.

## Rebuilding

```bash
# Layer 2 lookup table (fast, no network)
python -m ml_des_mp.build_mp_lookup

# QSPR training set: Bradley open MP data (downloaded) merged with the in-domain
# DES components, deduplicated by InChIKey (DES values win)
python -m ml_des_mp.build_mp_dataset

# Train the deep ensemble (reuses the ChemBERTa embedder; ~1 min on GPU)
python -m ml_des_mp.train_mp_qspr
```

Held-out accuracy of the bundled QSPR (n=468 test): **RMSE ≈ 44 K, MAE ≈ 32 K** —
roughly 3× tighter than the heuristic on novel molecules. The deployed ensemble
is refit on all data (these metrics come from the held-out split, so they are a
conservative estimate of the deployed model).

**Calibrated uncertainty (split-conformal).** A held-out calibration split fits a
normalized-residual quantile so the reported interval `Tm ± conformal_q·σ`
(σ = ensemble spread) has a coverage guarantee — measured **91.7% empirical
coverage** for the nominal 90% interval. The confidence the resolver attaches to
a `qspr` estimate is derived from σ relative to a **data-calibrated scale**
(90th-percentile σ on the calibration set), not a hand-picked constant.

## Controls

- `DES_DISABLE_QSPR=1` skips the QSPR layer (experimental lookup + heuristic
  only). Use this to reproduce results from a clean checkout, where only
  `experimental.json` is committed and `qspr_model.pt` is not.
- `DES_MP_DEVICE` selects the QSPR device (default `cpu`). The model loads
  lazily and stays on CPU by default — even when the DES eutectic stage runs on
  `cuda` via `--ml-device` — so the GPU is left free for a local LLM.
- `des_multi_agent.property_resolution.clear_resolver_caches()` drops the cached
  experimental table and QSPR model (for long-lived processes that regenerate an
  artifact or change the controls above mid-run).
