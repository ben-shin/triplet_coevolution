"""Evaluate frozen GatABC scores on independently deposited homologous structures."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from triplet_coevolution.io.score_store import load_score_store
from triplet_coevolution.structure.distances import _mapping_table, iter_distances
from triplet_coevolution.structure.mapping import ReferenceSequence, map_components
from triplet_coevolution.structure.parser import read_structure
from triplet_coevolution.validation.benchmark import _metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/public-experimental"
PRIVATE = ROOT / "results/private-gatabc-validation"
PLAN = json.loads((HERE / "experimental_protocol.json").read_text())
PAIRS = ("AB", "AC", "BC")
METHODS = PLAN["structural"]["metrics"]


def check(path, expected):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    return path


def read_inputs(private):
    if protocol_digest() != "14d58d27f6b6f4730b493a66ac3c3df1fac25ae01b5c1a4ab733b943d3f42b24":
        raise ValueError("Registration digest changed")
    manifest = json.loads((HERE / "gatabc_manifest.json").read_text())
    prep = private / "data" / manifest["paths"]["prepared_directory"]
    for filename, expected in PLAN["frozen_score_files"].items():
        check(private / "phase1-store" / filename, expected)
    store = load_score_store(private / "phase1-store")
    if set(store.scores) != set(METHODS):
        raise ValueError("score statistic set differs from registration")
    for name in [*"ABC", "matches"]:
        filename = "matches.tsv" if name == "matches" else f"{name}.fasta"
        check(prep / filename, store.provenance["inputs"][name]["sha256"])
    refs = []
    for comp, rid in manifest["references"].items():
        (reference,) = [r for r in SeqIO.parse(prep / f"{comp}.fasta", "fasta") if r.id == rid]
        refs.append(ReferenceSequence(comp, rid, str(reference.seq), "protein"))
    return refs, store, prep


def protocol_digest():
    data = (HERE / "experimental_protocol.json").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def assemble(source, pdb, plan):
    source_hashes = json.loads((HERE / "experimental_sources.json").read_text())
    check(source, source_hashes[source.name]["sha256"])
    structure = read_structure(source, assembly=plan["assembly"], model=plan["model"])
    with gzip.open(source, "rt") as h:
        cif = MMCIF2Dict(h)
    if "DIFFRACTION" not in structure.method.upper() or structure.entry_id.upper() != pdb:
        raise ValueError("unexpected experimental method or PDB entry")
    descriptions = dict(zip(cif["_entity.id"], cif["_entity.pdbx_description"], strict=True))
    refs = dict(zip(cif["_struct_ref.id"], cif["_struct_ref.pdbx_db_accession"], strict=True))
    dbrefs = list(
        zip(
            cif["_struct_ref_seq.ref_id"],
            cif["_struct_ref_seq.pdbx_strand_id"],
            cif["_struct_ref_seq.seq_align_beg"],
            cif["_struct_ref_seq.seq_align_end"],
            strict=True,
        )
    )
    groups, centroids, audit = [], [], []
    for names in plan["copies"][pdb]:
        group, centers = [], []
        for component, asym, entity in zip("ABC", names, plan["entity_ids"][pdb], strict=True):
            (chain,) = [
                c for c in structure.chains if c.label_asym_id == asym and c.entity_id == entity
            ]
            description = descriptions[entity]
            if f"subunit {component}".casefold() not in description.casefold():
                raise ValueError(f"unexpected deposited entity description for {pdb} {asym}")
            start, end = 1, len(chain.residues)
            if pdb == "3AL0" and component == "C":
                # Deposited DBREF 3 maps native GatC Q9WY94, separately from the linker and GluRS.
                (bounds,) = [
                    (int(lo), int(hi))
                    for rid, auth, lo, hi in dbrefs
                    if auth == chain.auth_asym_id and refs[rid] == "Q9WY94"
                ]
                start, end = bounds
                chain = replace(chain, residues=chain.residues[start - 1 : end])
            pts = [
                a.xyz
                for res in chain.residues
                for a in res.atoms
                if a.element not in {"H", "D", "T"} and a.occupancy > 0
            ]
            group.append(chain)
            centers.append(np.asarray(pts, dtype=float).mean(axis=0))
            audit.append(
                {
                    "component": component,
                    "copy": chain.copy_id,
                    "entity": entity,
                    "name": description,
                    "polymer_start": start,
                    "polymer_end": end,
                }
            )
        groups.append(group)
        centroids.append(centers)
    # Compare gross geometry of the registered ordering with every role-preserving assignment.
    n = len(groups)
    permutations = list(itertools.permutations(range(n)))
    costs = []
    for bs, cs in itertools.product(permutations, repeat=2):
        cost = sum(
            sum(
                np.linalg.norm(x - y)
                for x, y in itertools.combinations(
                    [centroids[a][0], centroids[b][1], centroids[c][2]], 2
                )
            )
            for a, b, c in zip(range(n), bs, cs, strict=True)
        )
        costs.append((float(cost), bs, cs))
    costs.sort()
    admitted = costs[0][1:] == (tuple(range(n)),) * 2 and (
        len(costs) == 1 or costs[1][0] > costs[0][0] + 1e-6
    )
    return (
        structure,
        groups,
        {
            "entities": audit,
            "centroid_assignment_costs_angstrom": costs,
            "admitted": admitted,
            "method": structure.method,
            "resolution_angstrom": cif["_refine.ls_d_res_high"][0],
        },
    )


def measure(structure, group, refs, store, out):
    block = replace(structure, chains=tuple(group))
    plan = PLAN["structural"]
    chosen = {comp: chain.copy_id for comp, chain in zip("ABC", group, strict=True)}
    mapping = map_components(
        refs,
        block,
        chain_ids=chosen,
        min_identity=plan["min_unique_identity"],
        min_coverage=plan["min_unique_coverage"],
    )
    # Alternative correspondence positions index the sequence supplied to the aligner.
    # Translate those positions back to deposited indices for the excised fusion domain.
    roles = dict(mapping.roles)
    for name, role in roles.items():
        if role.chain and role.chain.residues[0].seq_index != 1:
            sites = tuple(
                replace(
                    s,
                    alternatives=tuple(
                        role.chain.residues[j - 1].seq_index for j in s.alternatives
                    ),
                )
                for s in role.sites
            )
            roles[name] = replace(role, sites=sites)
    mapping = replace(mapping, roles=roles)
    _mapping_table(out / "mapping.tsv", mapping)
    audit = {}
    for role, report in mapping.roles.items():
        (candidate,) = [c for c in report.candidates if c.copy_id == chosen[role]]
        audit[role] = {
            "copy": candidate.copy_id,
            "status": report.status,
            "identity": candidate.identity,
            "coverage": candidate.coverage,
            "score": candidate.score,
            "strict_gate": bool(
                candidate.score > 0
                and candidate.identity >= plan["strict_identity"]
                and candidate.coverage >= plan["strict_coverage"]
            ),
            "positions": dict(Counter(s.status for s in report.sites)),
        }
    lengths = {r.component: len(r.sequence) for r in refs}
    arrays = {}
    for pair in PAIRS:
        shape = (lengths[pair[0]], lengths[pair[1]])
        arrays[pair] = {key: np.full(shape, np.nan) for key in ["heavy_distance", *METHODS]}
    for row in iter_distances(mapping, metric="heavy", close=8, distant=25, score_store=store):
        pair = row["pair"]
        pos = (row["x_reference_pos"] - 1, row["y_reference_pos"] - 1)
        for field, matrix in arrays[pair].items():
            if row[field] is not None:
                matrix[pos] = row[field]
    np.savez_compressed(
        out / "distances-and-scores.npz",
        **{f"{p}_{f}": m for p, values in arrays.items() for f, m in values.items()},
    )
    return arrays, audit


def statistics(arrays, common=None):
    result = {}
    for pair, fields in arrays.items():
        dist = fields["heavy_distance"].ravel()
        observed = np.isfinite(dist)
        finite = np.logical_and.reduce([np.isfinite(fields[m].ravel()) for m in METHODS])
        usable = observed & finite
        if common is not None:
            usable &= common[pair]
        thresholds = {}
        for cutoff in PLAN["structural"]["sensitivity_angstrom"]:
            labels = dist <= cutoff
            counts = {
                "observable_positives": int(np.sum(labels & observed)),
                "common_positives": int(np.sum(labels & usable)),
                "common_natural_prevalence": float(labels[usable].mean()) if usable.any() else None,
            }
            for method in METHODS:
                counts[method] = _metrics(
                    labels[usable], fields[method].ravel()[usable], curves=False
                )
            thresholds[str(cutoff)] = counts
        top = {}
        for method in METHODS:
            scores = fields[method].ravel()
            idx = np.flatnonzero(np.isfinite(scores))
            # Stable sort preserves ascending C-order reference coordinates at exact ties.
            rank = idx[np.argsort(-scores[idx], kind="stable")]
            top[method] = {}
            for requested in PLAN["structural"]["top_k"]:
                inds = rank[:requested]
                pos = int(np.sum(dist[inds] <= 8))
                unknown = int(np.sum(~observed[inds]))
                n = len(inds)
                top[method][str(requested)] = {
                    "n": n,
                    "positive": pos,
                    "negative": n - pos - unknown,
                    "unknown": unknown,
                    "lower": pos / n if n else None,
                    "upper": (pos + unknown) / n if n else None,
                }
        result[pair] = {
            "reference_pairs": len(dist),
            "observable": int(observed.sum()),
            "jointly_scored": int(finite.sum()),
            "common": int(usable.sum()),
            "cutoffs": thresholds,
            "top_before_mask": top,
        }
    macro = {}
    for m in METHODS:
        macro[m] = {}
        for metric in ["average_precision", "auroc"]:
            vals = [result[p]["cutoffs"]["8"][m][metric] for p in PAIRS]
            vals = [v for v in vals if v is not None]
            macro[m][metric] = {
                "interfaces": len(vals),
                "mean": sum(vals) / len(vals) if vals else None,
            }
    return {"interfaces": result, "macro": macro}


def write_metrics(summary, path):
    rows = []
    for population, copies in (
        ("within_copy", summary["copies"]),
        ("all_copy_intersection", summary.get("same_positions_across_copies", {})),
    ):
        for copy, value in copies.items():
            for pair, info in value["interfaces"].items():
                for cutoff, measures in info["cutoffs"].items():
                    for method in METHODS:
                        metrics = measures[method]
                        row = {
                            "population": population,
                            "copy": copy,
                            "pair": pair,
                            "cutoff_angstrom": cutoff,
                            "method": method,
                            "reference_pairs": info["reference_pairs"],
                            "observable_in_copy": info["observable"],
                            "scored_and_observed": metrics["scored"],
                            "positives": metrics["positives_scored"],
                            "prevalence": measures["common_natural_prevalence"],
                            "AP": metrics["average_precision"],
                            "AUROC": metrics["auroc"],
                        }
                        for k in PLAN["structural"]["top_k"]:
                            for field in ("positive", "negative", "unknown", "lower", "upper"):
                                top = info["top_before_mask"][method][str(k)]
                                row[f"top{k}_{field}"] = (
                                    top[field]
                                    if population == "within_copy" and cutoff == "8"
                                    else None
                                )
                        rows.append(row)
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    refs, store, _ = read_inputs(args.private)
    full, summary = {}, {"protocol_sha256": protocol_digest(), "structures": {}, "copies": {}}
    for pdb in PLAN["structural"]["copies"]:
        structure, groups, audit = assemble(
            args.out / "sources" / f"{pdb}.cif.gz", pdb, PLAN["structural"]
        )
        summary["structures"][pdb] = audit
        if not audit["admitted"]:
            continue
        for group in groups:
            key = pdb + "-" + "".join(c.label_asym_id for c in group)
            path = args.out / key
            path.mkdir(exist_ok=True, parents=True)
            arrays, roles = measure(structure, group, refs, store, path)
            full[key] = arrays
            summary["copies"][key] = {"mapping": roles, **statistics(arrays)}
    if full:
        common = {
            p: np.logical_and.reduce(
                [
                    np.isfinite(a[p][f].ravel())
                    for a in full.values()
                    for f in ["heavy_distance", *METHODS]
                ]
            )
            for p in PAIRS
        }
        summary["same_positions_across_copies"] = {
            k: statistics(a, common) for k, a in full.items()
        }
    path = args.out / "structural-summary.json"
    path.write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")
    write_metrics(summary, args.out / "experimental_structure_metrics.tsv")
    print(path)


if __name__ == "__main__":
    main()
