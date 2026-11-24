"""Export assembly-transformed coordinates and author-residue selections for local viewers."""

import json
from collections import defaultdict
from pathlib import Path

from Bio.PDB import MMCIFIO

from triplet_coevolution.structure.types import Chain


def write_chain(path: Path, chain: Chain) -> None:
    """Write the atoms of one selected assembly copy using the deposited residue identities."""
    data = defaultdict(list)
    atom_id = 0
    for residue in chain.residues:
        for atom in residue.atoms:
            atom_id += 1
            atom_fields = {
                "group_PDB": "ATOM",
                "id": atom_id,
                "type_symbol": atom.element,
                "label_atom_id": atom.name,
                "label_alt_id": atom.altloc or ".",
                "label_comp_id": residue.mon_id,
                "label_asym_id": chain.label_asym_id,
                "label_entity_id": chain.entity_id,
                "label_seq_id": residue.seq_index,
                "pdbx_PDB_ins_code": residue.insertion_code or "?",
                "Cartn_x": format(atom.xyz[0], ".12g"),
                "Cartn_y": format(atom.xyz[1], ".12g"),
                "Cartn_z": format(atom.xyz[2], ".12g"),
                "occupancy": atom.occupancy,
                "B_iso_or_equiv": "?",
                "auth_seq_id": residue.auth_seq_id or "?",
                "auth_comp_id": residue.mon_id,
                "auth_asym_id": chain.auth_asym_id or ".",
                "auth_atom_id": atom.name,
                "pdbx_PDB_model_num": "1",
            }
            for name, value in atom_fields.items():
                data["_atom_site." + name].append(str(value))
    if not atom_id:
        return
    data["data_"] = "triplet_view"
    data["_entry.id"] = "triplet_view"
    data["_entity.id"] = chain.entity_id
    data["_entity.type"] = "polymer"
    data["_entity_poly.entity_id"] = chain.entity_id
    data["_entity_poly.type"] = {
        "protein": "polypeptide(L)",
        "dna": "polydeoxyribonucleotide",
        "rna": "polyribonucleotide",
    }[chain.molecule]
    data["_struct_asym.id"] = chain.label_asym_id
    data["_struct_asym.entity_id"] = chain.entity_id
    for res in chain.residues:
        for key, value in (
            ("entity_id", chain.entity_id),
            ("num", str(res.seq_index)),
            ("mon_id", res.mon_id),
        ):
            data["_entity_poly_seq." + key].append(value)
    writer = MMCIFIO()
    writer.set_dict(dict(data))
    writer.save(str(path))


# Read identifiers from JSON as data. No user-supplied chain, insertion code,
# filename or author number is ever interpolated into a viewer command.
PYMOL = """set cif_use_auth, on
python
from pathlib import Path
import json
from pymol import cmd
root = Path.cwd()
index = json.loads((root / "viewer.json").read_text(encoding="utf-8"))
if any(name.startswith("triplet_") for name in cmd.get_names("all")):
    raise ValueError("Open the generated selections in a fresh PyMOL session.")
parts = {}
for component in index["components"]:
    obj = "triplet_" + component
    cmd.load(str(root / (component + ".cif")), obj)
    atoms = cmd.get_model(obj).atom
    parts[component] = {}
    for atom in atoms:
        parts[component].setdefault((atom.chain, atom.resi), []).append(atom.index)
for number, pair in enumerate(index["pairs"], 1):
    names = []
    for key in ("x", "y"):
        component = pair[key]["component"]
        target = pair[key]
        indices = parts[component].get((target["auth_chain"], target["auth_number"] + target["insertion"]), [])
        if not indices:
            raise ValueError("Viewer could not resolve mapped author residue: " + str(target))
        name = "triplet_site_" + str(number) + "_" + key
        cmd.select_list(name, "triplet_" + component, indices, mode="index")
        names.append(name)
    selection = "triplet_pair_" + str(number)
    cmd.select(selection, " or ".join(names))
    cmd.show("sticks", selection)
cmd.orient("triplet_*")
python end
"""

CHIMERAX = '''"""Open with ChimeraX from the generated viewer directory."""
import json
from pathlib import Path
root = Path(__file__).resolve().parent
index = json.loads((root / "viewer.json").read_text(encoding="utf-8"))
parts = {}
for component in index["components"]:
    models, status = session.open_command.open_data(str(root / (component + ".cif")))
    session.models.add(models)
    residues = {}
    for model in models:
        for residue in model.residues:
            residues[(residue.chain_id, str(residue.number), residue.insertion_code.strip())] = residue
    parts[component] = residues
for pair in index["pairs"]:
    for side in ("x", "y"):
        site = pair[side]
        key = (site["auth_chain"], site["auth_number"], site["insertion"])
        residue = parts[site["component"]].get(key)
        if residue is None:
            raise ValueError("Viewer could not resolve mapped author residue: " + str(site))
        residue.atoms.selected = True
        residue.atoms.displays = True
'''


def write_viewers(directory: Path, mapping, pairs: list[dict], *, statistic: str | None) -> dict:
    """Build one file per actual copy so identical author chain IDs stay distinct."""
    directory.mkdir()
    components = []
    for component, role in mapping.roles.items():
        if role.chain:
            path = directory / f"{component}.cif"
            write_chain(path, role.chain)
            if path.exists():
                components.append(component)
    grouped = []
    for row in pairs:
        if any(not row[f"{side}_auth_seq_id"] for side in ("x", "y")):
            continue
        item = {
            "pair": row["pair"],
            "distance_angstrom": float(row["distance_angstrom"]),
            "score": float(row[statistic]) if statistic else None,
        }
        for side, component in zip(("x", "y"), row["pair"], strict=True):
            item[side] = {
                "component": component,
                "reference_id": row[side + "_reference_id"],
                "reference_pos": int(row[side + "_reference_pos"]),
                "copy": row[side + "_copy_id"],
                "auth_chain": row[side + "_auth_asym_id"],
                "auth_number": row[side + "_auth_seq_id"],
                "insertion": row[side + "_insertion_code"],
            }
        grouped.append(item)
    index = {
        "components": components,
        "pairs": grouped,
        "statistic": statistic,
        "selection": "highest finite mapped score per interface"
        if statistic
        else "closest observable pairs per interface",
    }
    (directory / "viewer.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "view.pml").write_text(PYMOL, encoding="utf-8")
    (directory / "view_chimerax.py").write_text(CHIMERAX, encoding="utf-8")
    return {
        "index": "viewer/viewer.json",
        "pymol": "viewer/view.pml",
        "chimerax": "viewer/view_chimerax.py",
        "pairs": len(grouped),
        "unselectable_author_ids": len(pairs) - len(grouped),
        "components": components,
    }
