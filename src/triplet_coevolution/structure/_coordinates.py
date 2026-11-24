"""Common chemical, alternate-conformer and assembly-coordinate operations."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from math import isfinite
from urllib.parse import quote

import numpy as np
from Bio.Data.PDBData import nucleic_letters_3to1_extended, protein_letters_3to1_extended

from triplet_coevolution.structure.types import Atom, Chain, Residue, Transform

NUCLEOTIDES = {name.strip(): letter for name, letter in nucleic_letters_3to1_extended.items()}


def clean(value: str) -> str:
    return "" if value.strip() in ("?", ".") else value.strip()


def letter(monomer: str, molecule: str, parents: dict[str, str]) -> str:
    parent = parents.get(monomer, monomer)
    if molecule == "protein":
        return protein_letters_3to1_extended.get(parent, "X")
    return NUCLEOTIDES.get(parent, "N")


def read_atom(name, element, coords, occupancy, altloc) -> Atom | None:
    if not clean(occupancy):
        return None
    occ = float(occupancy)
    if not isfinite(occ) or not 0 <= occ <= 1:
        raise ValueError(f"invalid occupancy {occupancy!r} for atom {name!r}")
    if occ == 0:
        return None
    xyz = tuple(map(float, coords))
    if not name.strip() or not element.strip().isalpha() or not all(map(isfinite, xyz)):
        raise ValueError(f"invalid atom identity or coordinates for {name!r}")
    return Atom(name.strip(), element.strip().upper(), xyz, occ, clean(altloc))


def residue(index, monomers, molecule, parents, author, entries, canonical="") -> Residue:
    # Blank atoms belong to their own chemical component. A disordered point
    # mutation can have two components at the same label_seq_id.
    groups = defaultdict(list)
    for mon, atom in entries:
        groups[mon, atom.altloc].append(atom)
    variants = sorted(groups)
    candidates = [
        key for key in variants if key[1] or not any(m == key[0] and a for m, a in variants)
    ]
    alternatives = tuple(f"{mon}:{alt or '.'}" for mon, alt in candidates)
    selected = []
    alt = ""
    if candidates:

        def occupancy(key):
            atoms = [a for a in groups[key] if a.element not in {"H", "D", "T"}] or groups[key]
            return -sum(a.occupancy for a in atoms) / len(atoms)

        mon, alt = min(candidates, key=lambda key: (occupancy(key), key))
        selected = groups[mon, ""] + (groups[mon, alt] if alt else [])
    else:
        mon = monomers[0] if len(monomers) == 1 else "/".join(monomers)
    atoms = {}
    for atom in selected:
        previous = atoms.get(atom.name)
        if previous is not None and previous.altloc == atom.altloc:
            raise ValueError(f"duplicate atom {atom.name!r} at polymer position {index}")
        atoms[atom.name] = atom
    # The sequence used for mapping comes from the deposited polymer. A change
    # of model or selected point-mutation conformer cannot change that sequence.
    symbols = {letter(m, molecule, parents) for m in monomers}
    unknown = "X" if molecule == "protein" else "N"
    symbol = next(iter(symbols)) if len(symbols) == 1 else unknown
    if symbol == unknown and canonical and len(monomers) == 1:
        symbol = canonical
    return Residue(
        index, mon, symbol, *author, alt, tuple(atoms.values()), alternatives, tuple(monomers)
    )


def transform_copy(chain: Chain, operations: tuple[str, ...], transform: Transform) -> Chain:
    label = quote(chain.label_asym_id, safe="")
    copy = f"{label}@" + (",".join(quote(op, safe="") for op in operations) or "asu")
    residues = chain.residues
    matrix = np.asarray(transform)
    if not np.array_equal(matrix, np.eye(4)):
        coords = np.array([a.xyz for r in residues for a in r.atoms], dtype=float).reshape(-1, 3)
        positions = iter((coords @ matrix[:3, :3].T + matrix[:3, 3]).tolist())
        residues = tuple(
            replace(r, atoms=tuple(replace(a, xyz=tuple(next(positions))) for a in r.atoms))
            for r in residues
        )
    return replace(
        chain, copy_id=copy, operations=operations, transform=transform, residues=residues
    )


def as_transform(rows: Iterable[Iterable[float]]) -> Transform:
    matrix = np.asarray(list(rows), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("assembly transformation must be a finite homogeneous matrix")
    rotation = matrix[:3, :3]
    if not np.array_equal(matrix[3], [0, 0, 0, 1]) or not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-4
    ):
        raise ValueError("biological assembly operation must preserve Cartesian distances")
    return tuple(tuple(map(float, row)) for row in matrix)
