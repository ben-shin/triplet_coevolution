"""PDBx/mmCIF polymer and Cartesian biological assembly records."""

import re
from collections import defaultdict
from itertools import product

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from triplet_coevolution.structure._coordinates import as_transform, clean, read_atom, residue
from triplet_coevolution.structure.types import IDENTITY, Chain, resolve_selection


def _records(data, category):
    # The tokenizer also handles quoted values and multiline text fields.
    prefix = "_" + category + "."
    columns = {k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)}
    if columns and len({len(v) for v in columns.values()}) != 1:
        raise ValueError(f"inconsistent loop column lengths in {category}")
    return (dict(zip(columns, row)) for row in zip(*columns.values()))


def _expression(expression, operators):
    """Expand the dictionary's lists/ranges and products from right to left."""
    expr = "".join(expression.split())
    if "(" in expr or ")" in expr:
        if not re.fullmatch(r"(\([^()]+\))+", expr):
            raise ValueError(f"invalid assembly operator expression {expression!r}")
        groups = re.findall(r"\(([^()]*)\)", expr)
    else:
        groups = [expr]
    factors = []
    for group in groups:
        ids = []
        for token in group.split(","):
            if token in operators:
                ids.append(token)
            elif match := re.fullmatch(r"(\d+)-(\d+)", token):
                lo, hi = map(int, match.groups())
                if hi < lo or hi - lo + 1 > len(operators):
                    raise ValueError(f"invalid assembly operator range {token!r}")
                ids.extend(str(i) for i in range(lo, hi + 1))
            else:
                raise ValueError(f"unavailable assembly operator {token!r}")
        if not ids or any(key not in operators for key in ids):
            raise ValueError(f"unavailable assembly operations in {expression!r}")
        factors.append(ids)
    for keys in product(*factors):
        transform = np.eye(4)
        for key in keys:
            transform = transform @ operators[key]
        yield keys, as_transform(transform)


def _assemblies(data, selected):
    generators = list(_records(data, "pdbx_struct_assembly_gen"))
    available = sorted({row["assembly_id"] for row in generators})
    copies = []
    if selected not in available:
        return available, copies
    operators = {}
    for row in _records(data, "pdbx_struct_oper_list"):
        op = row["id"]
        if op in operators:
            raise ValueError(f"duplicate assembly operator {op!r}")
        rows = [
            [float(row[f"matrix[{i}][{j}]"]) for j in range(1, 4)] + [float(row[f"vector[{i}]"])]
            for i in range(1, 4)
        ]
        # Other operators can describe crystallographic basis changes unrelated
        # to the chosen assembly. Check rigidity on the actual assembly product.
        operators[op] = np.asarray([*rows, (0, 0, 0, 1)])
    for row in generators:
        if row["assembly_id"] == selected:
            labels = tuple(s.strip() for s in row["asym_id_list"].split(","))
            for keys, transform in _expression(row["oper_expression"], operators):
                copies.extend((label, keys, transform) for label in labels)
    return available, copies


def _molecule(kind):
    if kind.startswith("polypeptide("):
        return "protein"
    return {"polyribonucleotide": "rna", "polydeoxyribonucleotide": "dna"}.get(kind)


def _organisms(data):
    source = defaultdict(lambda: (set(), set()))
    for table, species, tax in [
        ("entity_src_gen", "pdbx_gene_src_scientific_name", "pdbx_gene_src_ncbi_taxonomy_id"),
        ("entity_src_nat", "pdbx_organism_scientific", "pdbx_ncbi_taxonomy_id"),
        ("pdbx_entity_src_syn", "organism_scientific", "ncbi_taxonomy_id"),
    ]:
        for row in _records(data, table):
            names, ids = source[row["entity_id"]]
            if name := clean(row.get(species, "")):
                names.add(name)
            if taxid := clean(row.get(tax, "")):
                ids.add(taxid)
    return source


def parse_cif(stream, selected, model):
    data = MMCIF2Dict(stream)
    numbers = data.get("_atom_site.pdbx_PDB_model_num")
    models = set(map(int, numbers)) if numbers else ({1} if "_atom_site.Cartn_x" in data else set())
    all_assemblies = set(data.get("_pdbx_struct_assembly_gen.assembly_id", []))
    selected, model = resolve_selection(selected, model, models, all_assemblies)
    warnings = []
    polymers = {}
    canon = {}
    for row in _records(data, "entity_poly"):
        eid = row["entity_id"]
        if mol := _molecule(row["type"]):
            polymers[eid] = mol
            canon[eid] = "".join(row.get("pdbx_seq_one_letter_code_can", "").split()).upper()
        else:
            warnings.append(f"Excluded unsupported polymer entity {eid} ({row['type']}).")
    asym = {r["id"]: r["entity_id"] for r in _records(data, "struct_asym")}
    labels = {label: eid for label, eid in asym.items() if eid in polymers}
    if not labels:
        raise ValueError("mmCIF requires polymer entity types and struct_asym chain assignments")
    monomers = defaultdict(lambda: defaultdict(list))
    for row in _records(data, "entity_poly_seq"):
        ids = monomers[row["entity_id"]][int(row["num"])]
        if row["mon_id"] not in ids:
            ids.append(row["mon_id"])
    for eid in set(labels.values()):
        if monomers[eid]:
            if sorted(monomers[eid]) != list(range(1, len(monomers[eid]) + 1)):
                raise ValueError(f"noncontiguous polymer sequence for entity {eid}")
        else:
            raise ValueError(f"missing full entity_poly_seq for entity {eid}")
    parents = {
        r["id"]: clean(r["mon_nstd_parent_comp_id"])
        for r in _records(data, "chem_comp")
        if clean(r.get("mon_nstd_parent_comp_id", ""))
    }
    mod = {}
    for row in _records(data, "pdbx_struct_mod_residue"):
        if clean(row.get("label_seq_id", "")) and clean(row.get("parent_comp_id", "")):
            key = (row["label_asym_id"], int(row["label_seq_id"]), row["label_comp_id"])
            mod[key] = row["parent_comp_id"]
    author_chains = defaultdict(set)
    authors = {}
    ambiguous_ids = set()
    for r in _records(data, "pdbx_poly_seq_scheme"):
        label = r["asym_id"]
        if label in labels:
            if strand := clean(r.get("pdb_strand_id", "")):
                author_chains[label].add(strand)
            i = int(r["seq_id"])
            if i not in monomers[labels[label]] or labels[label] != r["entity_id"]:
                raise ValueError(f"polymer sequence scheme mismatch at {label}/{i}")
            if auth := clean(r.get("auth_seq_num", "")):
                position = (auth, clean(r.get("pdb_ins_code", "")))
                if authors.setdefault((label, i), position) != position:
                    ambiguous_ids.add((label, i))
    for site in ambiguous_ids:
        del authors[site]
    sites = defaultdict(list)
    deposited = {}
    for r in _records(data, "atom_site"):
        number = int(r.get("pdbx_PDB_model_num", "1"))
        label = r["label_asym_id"]
        if number != model or label not in labels:
            continue
        eid = labels[label]
        if clean(r.get("label_entity_id", "")) and r["label_entity_id"] != eid:
            raise ValueError(f"atom label_asym_id and label_entity_id disagree for {label}")
        # A ligand with a reused author chain has its own label asym ID.
        seq_id = clean(r.get("label_seq_id", ""))
        if not seq_id:
            raise ValueError(f"missing label_seq_id on a polymer atom in {label}")
        i = int(seq_id)
        if i not in monomers[eid]:
            raise ValueError(f"atom position outside entity sequence at {label}/{i}")
        mon = r["label_comp_id"]
        if mon not in monomers[eid][i]:
            raise ValueError(f"atom monomer {mon} disagrees with entity sequence at {label}/{i}")
        if auth_chain := clean(r.get("auth_asym_id", "")):
            author_chains[label].add(auth_chain)
        if auth := clean(r.get("auth_seq_id", "")):
            pos = (auth, clean(r.get("pdbx_PDB_ins_code", "")))
            if deposited.setdefault((label, i), pos) != pos:
                raise ValueError(f"conflicting atom author residue IDs at {label}/{i}")
            authors[label, i] = pos
        atom = read_atom(
            r["label_atom_id"],
            r["type_symbol"],
            [r["Cartn_" + k] for k in "xyz"],
            r.get("occupancy", ""),
            r.get("label_alt_id", ""),
        )
        if atom is not None:
            sites[label, i].append((mon, atom))
    uncertain = ambiguous_ids - set(deposited)
    if uncertain:
        warnings.append(
            f"Conflicting scheme author IDs were withheld at {len(uncertain)} positions."
        )
    source = _organisms(data)
    chains = {}
    for label, eid in labels.items():
        if len(author_chains[label]) > 1:
            raise ValueError(f"multiple author chain identifiers for label chain {label}")
        auth = next(iter(author_chains[label]), "")
        names, taxids = source[eid]
        sequence = monomers[eid]
        canonical = canon[eid] if len(canon[eid]) == len(sequence) else ""
        residues = []
        for i in range(1, len(sequence) + 1):
            overrides = {m: mod[label, i, m] for m in sequence[i] if (label, i, m) in mod}
            residues.append(
                residue(
                    i,
                    sequence[i],
                    polymers[eid],
                    parents | overrides if overrides else parents,
                    authors.get((label, i), (None, "")),
                    sites[label, i],
                    canonical[i - 1] if canonical else "",
                )
            )
        chains[label] = Chain(
            "",
            label,
            auth,
            eid,
            polymers[eid],
            " | ".join(sorted(names)),
            " | ".join(sorted(taxids)),
            (),
            IDENTITY,
            tuple(residues),
        )
    available, copies = _assemblies(data, selected)
    if any(label not in asym for label, _, _ in copies):
        raise ValueError("biological assembly references an unknown label asym ID")
    entry = data.get("_entry.id", [data.get("data_", "")])[0]
    method = " | ".join(data.get("_exptl.method", []))
    return entry, method, chains, models, available, copies, warnings
