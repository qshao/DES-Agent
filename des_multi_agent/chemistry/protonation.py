"""Deterministic protonation / dominant-species engine.

Given a SMILES and a pH, returns the dominant ionized species by editing the
formal charge and hydrogen count of each tabulated ionizable group. Acids
deprotonate when pH > pKa; bases protonate when pH < pKa. Un-tabulated groups
are left exactly as drawn. Never raises — returns a passthrough result on any
error, mirroring chemistry.claim_grounding.structural_facts.

The pKa table is hand-curated and offline; identical results on any backend.
Each SMARTS is written so the ionizable atom is the FIRST atom of the match
(``match[0]``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

# (smarts, group_name, pka, kind). kind ∈ {"acid","base"}. Most-specific first.
# Ionizable atom is match[0] by construction.
_IONIZABLE_SMARTS: list[tuple[str, str, float, str]] = [
    # --- acids: protonated form is neutral, deprotonated carries -1 ---
    ("[OX2H1]S(=O)=O",           "sulfonic acid",   -1.0, "acid"),
    ("[OX2H1][PX4](=O)",         "phosphonic acid",  2.0, "acid"),
    ("[OX2H1]C=O",               "carboxylic acid",  4.2, "acid"),
    ("[OX2H1]c",                 "phenol",           9.9, "acid"),
    ("[SX2H1]",                  "thiol",           10.5, "acid"),
    # --- bases: protonated form carries +1, neutral form is "deprotonated base" ---
    ("[NX2]=C([NX3])[NX3]",      "guanidine",       12.5, "base"),
    ("[nX2;r5]",                 "imidazole",        7.0, "base"),
    ("[nX2;r6]",                 "pyridine",         5.2, "base"),
    ("[NX3;!$([N+]);!$(NC=O)]c", "aniline",          4.6, "base"),
    ("[NX3;H2,H1,H0;!$([N+]);!$(NC=O);!$(N=*);!$(Nc)]", "amine", 10.6, "base"),
]

# Precompile once at module load — avoids 10 SMARTS compilations per dominant_species call.
_IONIZABLE_PATTERNS: list[tuple[object, str, float, str]] = [
    (Chem.MolFromSmarts(smarts), name, pka, kind)
    for smarts, name, pka, kind in _IONIZABLE_SMARTS
]


@dataclass(frozen=True)
class IonizedGroup:
    group_name: str
    atom_idx: int
    pka: float
    state: str      # "protonated" | "deprotonated" | "neutral"
    charge: int     # formal charge applied to the matched atom in the species


@dataclass(frozen=True)
class ProtonationResult:
    input_smiles: str
    pH: float
    species_smiles: str        # canonical SMILES of the dominant species
    mol: object                # Chem.Mol | None  (object keeps frozen dataclass happy)
    groups: list[IonizedGroup] = field(default_factory=list)
    net_charge: int = 0


def _passthrough(smiles_or_mol: object, pH: float) -> ProtonationResult:
    """Safe fallback: canonicalize if possible, otherwise echo the raw input."""
    if isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            smi = ""
        net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        return ProtonationResult(smi, pH, smi, mol, [], net)
    raw = smiles_or_mol if isinstance(smiles_or_mol, str) else str(smiles_or_mol)
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return ProtonationResult(raw, pH, raw, None, [], 0)
    smi = Chem.MolToSmiles(mol)
    net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    return ProtonationResult(raw, pH, smi, mol, [], net)


def _deprotonate(atom) -> None:
    atom.SetFormalCharge(-1)
    atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - 1))
    atom.SetNoImplicit(True)


def _protonate(atom) -> None:
    atom.SetFormalCharge(+1)
    atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
    atom.SetNoImplicit(True)


def dominant_species(smiles_or_mol, pH: float = 7.0) -> ProtonationResult:
    """Return the dominant ionized species of *smiles_or_mol* at *pH*.

    Never raises; returns a passthrough ProtonationResult on any failure.
    """
    try:
        base = (
            smiles_or_mol
            if isinstance(smiles_or_mol, Chem.Mol)
            else Chem.MolFromSmiles(smiles_or_mol)
        )
        if base is None:
            return _passthrough(smiles_or_mol, pH)
        input_smiles = Chem.MolToSmiles(base)
        rw = Chem.RWMol(base)
        # Flush implicit H counts into the RWMol's cache before any edits so
        # that GetTotalNumHs() is accurate throughout the editing loop.
        rw.UpdatePropertyCache(strict=False)

        touched: set[int] = set()
        groups: list[IonizedGroup] = []
        for patt, name, pka, kind in _IONIZABLE_PATTERNS:
            if patt is None:
                continue
            # Match against the original read-only mol so that edits applied
            # to earlier atoms cannot corrupt SMARTS matching for later ones
            # (e.g. [N+] or H-count patterns would see stale state on rw).
            for match in base.GetSubstructMatches(patt):
                idx = match[0]
                if idx in touched:
                    continue
                touched.add(idx)
                atom = rw.GetAtomWithIdx(idx)
                if kind == "acid" and pH > pka:
                    _deprotonate(atom)
                    groups.append(IonizedGroup(name, idx, pka, "deprotonated", -1))
                elif kind == "base" and pH < pka:
                    _protonate(atom)
                    groups.append(IonizedGroup(name, idx, pka, "protonated", +1))
                else:
                    groups.append(IonizedGroup(name, idx, pka, "neutral", 0))

        species = rw.GetMol()
        Chem.SanitizeMol(species)
        net = sum(a.GetFormalCharge() for a in species.GetAtoms())
        return ProtonationResult(
            input_smiles=input_smiles,
            pH=pH,
            species_smiles=Chem.MolToSmiles(species),
            mol=species,
            groups=groups,
            net_charge=net,
        )
    except Exception:
        return _passthrough(smiles_or_mol, pH)
