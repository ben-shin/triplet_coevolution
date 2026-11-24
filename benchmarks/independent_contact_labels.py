"""Map the preregistered chains and compute full reference-axis heavy-atom distances."""

from collections import Counter
from dataclasses import replace

import numpy as np
from Bio import SeqIO
from cohort_sources import STAGE, atomic_json, digest, request_file
from independent_pair_scores import OUT, POLICY, POLICY_SHA

from triplet_coevolution.structure.distances import _atoms, _mapping_table, _row_distances, _targets
from triplet_coevolution.structure.mapping import ReferenceSequence, map_components
from triplet_coevolution.structure.parser import read_structure

STRUCTURE_SHA = {
    "4YG2": "6613a2f02c646ae2f3fffa2067f986ee5fd261837659149f27da23f6188a5d0a",
    "4YBB": "9ca44cd8449fb34b29cf3a84e58c798faee5102a8eefe1a09dca144915b72265",
}


def labels(name):
    rule = POLICY["labels"][name]
    entry = rule["entry"]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{entry}.cif.gz"
    path = request_file(url, OUT / "sources" / (entry + ".cif.gz"))
    if digest(path) != STRUCTURE_SHA[entry]:
        raise ValueError(
            f"Deposited mmCIF version differs from the experimental source freeze: {entry}"
        )
    structure = read_structure(path, assembly=rule["assembly"], model=rule["model"])
    if structure.entry_id.upper() != entry or "DIFFRACTION" not in structure.method.upper():
        raise ValueError(f"Unexpected deposition identity/method for {entry}")
    refs = []
    for role, ref_id in zip("ABC", rule["references"], strict=True):
        with (STAGE / name / f"{role}.fasta").open() as h:
            ref = next(SeqIO.parse(h, "fasta"))
        if ref.id != ref_id:
            raise ValueError("Source reference differs from registration")
        refs.append(
            ReferenceSequence(
                role,
                ref.id,
                str(ref.seq),
                "rna" if name == "ribosome" and role == "A" else "protein",
            )
        )
    result = {
        "entry": entry,
        "source_url": url,
        "source_sha256": digest(path),
        "bytes": path.stat().st_size,
        "policy_sha256": POLICY_SHA,
        "method": structure.method,
        "assembly": structure.assembly,
        "model": structure.model,
        "copies": {},
    }
    for labels in rule["copies"]:
        group = []
        for label, entity in zip(labels, rule["entities"], strict=True):
            (chain,) = [
                c
                for c in structure.chains
                if c.label_asym_id == label and c.entity_id == entity and c.operations == ("1",)
            ]
            group.append(chain)
        copy = "-".join(labels)
        mapping = map_components(
            refs,
            replace(structure, chains=tuple(group)),
            chain_ids={r.component: c.copy_id for r, c in zip(refs, group, strict=True)},
            min_identity=0.25,
            min_coverage=0.6,
        )
        audit = {}
        atoms, targets = {}, {}
        for ref, selected in zip(refs, group, strict=True):
            role = mapping.roles[ref.component]
            (cand,) = [c for c in role.candidates if c.copy_id == selected.copy_id]
            audit[ref.component] = {
                "chain": selected.copy_id,
                "entity": selected.entity_id,
                "auth_asym_id": selected.auth_asym_id,
                "axis": ref.reference_id,
                "source_organism": selected.organism,
                "identity": cand.identity,
                "coverage": cand.coverage,
                "alignment_score": cand.score,
                "status": role.status,
                "gate_30_70": cand.score > 0 and cand.identity >= 0.3 and cand.coverage >= 0.7,
                "positions": dict(Counter(s.status for s in role.sites)),
            }
            atoms[ref.component] = [_atoms(s, ref.molecule == "protein") for s in role.sites]
            targets[ref.component] = _targets(atoms[ref.component])
        _mapping_table(out / (copy + "-mapping.tsv"), mapping)
        distances = {}
        for pair in ("AB", "AC", "BC"):
            x, y = pair
            distances[pair] = np.stack(
                [_row_distances(row, targets[y], len(atoms[y]))[0] for row in atoms[x]]
            )
        archive = out / (copy + "-distance.npz")
        np.savez_compressed(archive, **distances)
        result["copies"][copy] = {
            "axes": audit,
            "distance_sha256": digest(archive),
            "mapping_sha256": digest(out / (copy + "-mapping.tsv")),
        }
        print("Distance freeze", name, copy, digest(archive), flush=True)
    atomic_json(out / "structure.json", result)
    return result


if __name__ == "__main__":
    for system in ["rnap", "ribosome"]:
        labels(system)
