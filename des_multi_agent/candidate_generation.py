from __future__ import annotations

from typing import Mapping, Sequence

from rdkit import Chem

from .chemistry_filter import canonicalize_smiles
from .schemas import CandidateProposal


_FAMILY_LIBRARY: Sequence[tuple[str, str, str]] = (
    ("alcohol", "hydrogen-bond donor", "O"),
    ("diol", "hydrogen-bond donor", "OCCO"),
    ("polyol", "hydrogen-bond donor", "OCC(O)CO"),
    ("amide", "hydrogen-bond acceptor", "CC(=O)N"),
    ("carboxylic acid", "hydrogen-bond donor", "CC(=O)O"),
    ("amine", "hydrogen-bond acceptor", "CN"),
    ("quaternary ammonium salt", "ionic partner", "C[N+](C)(C)C.[Cl-]"),
    ("choline-like", "ionic partner", "C[N+](C)(C)CCO.[Cl-]"),
    ("urea", "hydrogen-bond donor", "NC(=O)N"),
    ("sugar-like polyol", "hydrogen-bond donor", "OC[C@H](O)[C@H](O)CO"),
    ("short diol", "hydrogen-bond donor", "OCCCO"),
    ("ether alcohol", "hydrogen-bond donor", "COCCO"),
    ("lactam", "hydrogen-bond acceptor", "O=C1CCCN1"),
    ("cyclic urea", "hydrogen-bond donor", "O=C1NC(=O)NC1"),
    ("sulfoxide", "hydrogen-bond acceptor", "CS(=O)C"),
    ("sulfone", "hydrogen-bond acceptor", "CS(=O)(=O)C"),
    ("hydroxypyridine", "hydrogen-bond donor", "OC1=CC=CC=N1"),
    ("dimethylformamide-like", "polar aprotic partner", "CN(C)C=O"),
    ("ether", "polar aprotic partner", "CCOCC"),
    ("hydroxyamide", "hydrogen-bond donor", "CC(=O)NCCO"),
)


def _matches_constraints(family: str, smiles: str, constraints: Mapping[str, object] | None) -> bool:
    if not constraints:
        return True
    allowed_families = constraints.get("allowed_families")
    if allowed_families and family not in set(str(x) for x in allowed_families):
        return False
    excluded_smiles = constraints.get("excluded_smiles")
    if excluded_smiles and smiles in set(str(x) for x in excluded_smiles):
        return False
    return True


def generate_candidates(component_a: str, n: int, constraints=None):
    proposals: list[CandidateProposal] = []
    if n <= 0:
        return proposals

    component_mol = Chem.MolFromSmiles(component_a)
    if component_mol is None:
        raise ValueError("Invalid component A SMILES")

    canonical_component_a = canonicalize_smiles(component_a)
    for family, rationale, smiles in _FAMILY_LIBRARY:
        if canonicalize_smiles(smiles) == canonical_component_a:
            continue
        if not _matches_constraints(family, smiles, constraints):
            continue
        proposals.append(
            CandidateProposal(
                smiles=smiles,
                rationale=f"{rationale} candidate from a small rule-based family",
                family=family,
                source="heuristic",
                source_id="rule-based-family-library",
            )
        )
        if len(proposals) >= n:
            return proposals

    if len(proposals) < n:
        raise ValueError(
            f"Unable to generate {n} unique candidate(s) for component A after applying constraints."
        )
    return proposals
