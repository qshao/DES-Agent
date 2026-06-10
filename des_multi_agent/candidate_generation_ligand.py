from __future__ import annotations

from typing import Sequence

from rdkit import Chem

from .chemistry_filter import canonicalize_smiles
from .schemas import CandidateProposal


# (family_name, coordination_role, SMILES)
_LIGAND_LIBRARY: Sequence[tuple[str, str, str]] = (
    ("aminoacetate", "bidentate N,O-chelator", "NCC(=O)O"),
    ("iminodiacetate", "tridentate N,O,O-chelator", "OC(=O)CNCC(=O)O"),
    ("nitrilotriacetate", "tetradentate N,O,O,O-chelator", "OC(=O)CN(CC(=O)O)CC(=O)O"),
    ("catecholate", "bidentate O,O-chelator", "Oc1ccccc1O"),
    ("8-hydroxyquinoline", "bidentate N,O-chelator", "Oc1cccc2ncccc12"),
    ("acetylacetonate", "bidentate O,O-chelator", "CC(=O)CC(=O)C"),
    ("acetohydroxamate", "bidentate N,O-chelator", "CC(=O)NO"),
    ("2,2-bipyridine", "bidentate N,N-chelator", "c1ccnc(-c2ccccn2)c1"),
    ("1,10-phenanthroline", "bidentate N,N-chelator", "c1cnc2ccc3ncccc3c2c1"),
    ("salicylate", "bidentate O,O-chelator", "OC(=O)c1ccccc1O"),
    ("cysteine", "tridentate N,O,S-chelator", "NC(CS)C(=O)O"),
    ("2-mercaptoethanol", "bidentate O,S-chelator", "OCCS"),
    ("histidine-like", "bidentate N,N-imidazole donor", "NC(Cc1c[nH]cn1)C(=O)O"),
    ("ethylenediamine", "bidentate N,N-chelator", "NCCN"),
    ("triethylenetetramine", "hexadentate polyamine", "NCCNCCNCCN"),
    ("phosphonate", "bidentate O,O-chelator", "CC(O)(P(=O)(O)O)P(=O)(O)O"),
    ("oxalate", "bidentate O,O-chelator", "OC(=O)C(=O)O"),
    ("citrate", "tridentate O,O,O-chelator", "OC(CC(=O)O)(CC(=O)O)C(=O)O"),
    ("picolinic acid", "bidentate N,O-chelator", "OC(=O)c1ccccn1"),
    ("diethylenetriamine", "tridentate N,N,N-chelator", "NCCNCCN"),
)


def generate_ligand_candidates(metal_ion: str, n: int, constraints=None) -> list[CandidateProposal]:
    proposals: list[CandidateProposal] = []
    seen: set[str] = set()

    for family, role, smiles in _LIGAND_LIBRARY:
        if len(proposals) >= n:
            break
        canon = canonicalize_smiles(smiles)
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        proposals.append(
            CandidateProposal(
                smiles=canon,
                rationale=f"{family} ({role})",
                family=family,
                source="heuristic",
                source_id="rule-based-ligand-library",
            )
        )

    return proposals[:n]
