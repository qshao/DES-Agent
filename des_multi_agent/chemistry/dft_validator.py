"""Free-ligand DFT validation pipeline using gpu4pyscf.

Entry point: compute_dft_properties(smiles) -> DFTResult.
Never raises — all failures return DFTResult(success=False, error=...).
"""
from __future__ import annotations

from dataclasses import dataclass, field


_DONOR_SYMBOLS: frozenset[str] = frozenset({"N", "O", "P", "S"})
_HARTREE_TO_EV: float = 27.2114
DEFAULT_DFT_METHOD: str = "B3LYP-D3(BJ)/def2-SVP"


@dataclass
class DFTResult:
    smiles: str
    success: bool
    homo_ev: float | None = None
    homo_lumo_gap_ev: float | None = None
    donor_charges: list[float] = field(default_factory=list)
    geometry_method: str = "xtb"
    dft_method: str = DEFAULT_DFT_METHOD
    error: str | None = None
    species_smiles: str | None = None
    ph: float | None = None
    from_cache: bool = False


def _embed_mmff(smiles: str):
    """SMILES -> RDKit MMFF94 3D Mol with H. Raises ValueError on failure."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise ValueError(f"3D embedding failed for {smiles!r}")
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    if ff is not None:
        ff.Minimize()
    return mol


def _xtb_optimize(mol, charge: int = 0) -> tuple[list[str], "np.ndarray"]:
    """RDKit Mol -> xTB GFN2-optimized geometry via xtb subprocess.

    ``charge`` is the net formal charge of the species being optimized (0 for
    the neutral legacy path; the pH-derived net charge for the pH-aware path).
    xtb defaults to charge=0 with no ``--chrg`` flag, which would silently
    optimize a charged species as if it were neutral.

    Returns (atom_symbols, coords_angstrom). Raises RuntimeError on failure.
    """
    import numpy as np
    import os
    import subprocess
    import tempfile

    conf = mol.GetConformer()
    positions = conf.GetPositions()  # Angstrom
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    xyz_lines = [str(len(symbols)), "xtb input"]
    for sym, pos in zip(symbols, positions):
        xyz_lines.append(f"{sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_in = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_in, "w") as fh:
            fh.write("\n".join(xyz_lines))
        proc = subprocess.run(
            ["xtb", xyz_in, "--opt", "--gfn", "2", "--chrg", str(charge), "--silent"],
            cwd=tmpdir, capture_output=True, text=True, timeout=120,
        )
        opt_xyz = os.path.join(tmpdir, "xtbopt.xyz")
        if proc.returncode != 0 or not os.path.exists(opt_xyz):
            raise RuntimeError(f"xtb optimization failed: {proc.stderr[:300]}")
        with open(opt_xyz) as fh:
            lines = fh.readlines()

    opt_syms, opt_pos = [], []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            opt_syms.append(parts[0])
            opt_pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return opt_syms, np.array(opt_pos)


def _run_dft(
    symbols: list[str], coords_angstrom: "np.ndarray", charge: int = 0
) -> tuple[float, float, list[int], object]:
    """Run B3LYP-D3(BJ)/def2-SVP single-point. Returns (homo_ev, gap_ev, donor_indices, mf).

    Raises RuntimeError if SCF does not converge.
    """
    from pyscf import gto
    from gpu4pyscf import dft as gpu_dft

    atom_list = [(sym, tuple(pos)) for sym, pos in zip(symbols, coords_angstrom)]
    mol = gto.Mole(atom=atom_list, basis="def2-svp", charge=charge, spin=0, verbose=0)
    mol.build()

    mf = gpu_dft.RKS(mol)
    mf.xc = "B3LYP"
    mf.disp = "d3bj"
    mf.kernel()

    if not mf.converged:
        raise RuntimeError("SCF did not converge")

    homo_idx = mol.nelectron // 2 - 1
    homo_ev = float(mf.mo_energy[homo_idx]) * _HARTREE_TO_EV
    lumo_ev = float(mf.mo_energy[homo_idx + 1]) * _HARTREE_TO_EV
    gap_ev = lumo_ev - homo_ev

    donor_indices = [i for i, sym in enumerate(symbols) if sym in _DONOR_SYMBOLS]
    return homo_ev, gap_ev, donor_indices, mf


def _get_donor_charges(mf, donor_indices: list[int]) -> list[float]:
    """Extract per-donor-atom charges from a converged RKS object.

    Uses Mulliken population via mf.mulliken_pop() (second return value = atom charges).
    For Lowdin charges (more basis-set-stable) try mf.mulliken_pop_meta_lowdin_ao()
    instead — verify the method name against your installed PySCF version.
    """
    _, atom_charges = mf.mulliken_pop(verbose=0)
    return [float(atom_charges[i]) for i in donor_indices]


def compute_dft_properties(smiles: str, pH: float | None = None) -> DFTResult:
    """Full pipeline: SMILES -> DFTResult. Never raises.

    pH=None (default): computes the neutral input SMILES as-is with charge=0 —
    exact legacy behavior. pH=<float>: computes the dominant protonation state
    at that pH (chemistry.protonation.dominant_species) with its real net
    formal charge.
    """
    species_smiles = smiles
    net_charge = 0
    if pH is not None:
        from .protonation import dominant_species
        protonation = dominant_species(smiles, pH)
        species_smiles = protonation.species_smiles
        net_charge = protonation.net_charge

    result_species_smiles = species_smiles if pH is not None else None

    try:
        mol = _embed_mmff(species_smiles)
        symbols, coords = _xtb_optimize(mol, charge=net_charge)
        homo_ev, gap_ev, donor_indices, mf = _run_dft(symbols, coords, charge=net_charge)
        donor_charges = _get_donor_charges(mf, donor_indices)
        return DFTResult(
            smiles=smiles,
            success=True,
            homo_ev=homo_ev,
            homo_lumo_gap_ev=gap_ev,
            donor_charges=donor_charges,
            species_smiles=result_species_smiles,
            ph=pH,
        )
    except Exception as exc:
        return DFTResult(
            smiles=smiles, success=False, error=str(exc),
            species_smiles=result_species_smiles, ph=pH,
        )
