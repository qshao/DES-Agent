"""Build the experimental pure-component melting-point lookup table.

Reads the training CSV (which carries experimental pure-component melting
points in the ``Smiles#1/T#1`` and ``Smiles#2/T#2`` columns) and emits an
InChIKey-keyed JSON table consumed by ``des_multi_agent.property_resolution``.

Using these values at inference keeps pure-component Tm consistent with what
the eutectic physics model was trained on, eliminating train/inference skew.

Usage:
    python -m ml_des_mp.build_mp_lookup \
        --csv ml_des_mp/Melting_temperature_appended_35il_03082026.csv \
        --out artifacts/melting_points/experimental.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from des_multi_agent.property_resolution import canonical_inchikey


def _inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Key on the canonical tautomer so the resolver's two-tier lookup can match
    # alternative tautomer inputs.
    return canonical_inchikey(mol)


def build_table(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by_key: dict[str, list[float]] = defaultdict(list)
    canon_smiles: dict[str, str] = {}
    for r in rows:
        for s_col, t_col in (("Smiles#1", "T#1"), ("Smiles#2", "T#2")):
            sm = (r.get(s_col) or "").strip()
            tv = (r.get(t_col) or "").strip()
            if not sm or not tv:
                continue
            try:
                tm = float(tv)
            except ValueError:
                continue
            key = _inchikey(sm)
            if key is None:
                continue
            by_key[key].append(tm)
            mol = Chem.MolFromSmiles(sm)
            canon_smiles[key] = Chem.MolToSmiles(mol)

    entries: dict[str, dict] = {}
    for key, vals in by_key.items():
        entries[key] = {
            "tm_k": round(statistics.median(vals), 3),
            "n": len(vals),
            "spread_k": round(max(vals) - min(vals), 3),
            "smiles": canon_smiles[key],
        }
    return {
        "model_name": "melting-point-experimental-lookup",
        "source": str(csv_path.name),
        "units": "K",
        "key": "inchikey",
        "entries": entries,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ml_des_mp/Melting_temperature_appended_35il_03082026.csv")
    ap.add_argument("--out", default="artifacts/melting_points/experimental.json")
    args = ap.parse_args(argv)

    table = build_table(Path(args.csv))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, sort_keys=True)
    print(f"Wrote {len(table['entries'])} experimental melting points to {out}")


if __name__ == "__main__":
    main()
