"""Recalculate the original GatABC proximity and distant controls from local supplied files."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from Bio import SeqIO

from triplet_coevolution.io.score_store import load_score_store
from triplet_coevolution.structure.distances import iter_distances
from triplet_coevolution.structure.mapping import ReferenceSequence, map_components
from triplet_coevolution.structure.parser import read_structure
from triplet_coevolution.validation.benchmark import _metrics


def verify(root: Path, *, scores: Path | None = None) -> dict:
    """Verify source bytes, align reference sequences, and count threshold agreement."""
    manifest = json.loads(
        Path(__file__).with_name("gatabc_manifest.json").read_text(encoding="utf-8")
    )
    prep = manifest["paths"]["prepared_directory"]

    def checked(relative):
        path = root / relative
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest["supplied_sha256"][relative]:
            raise ValueError(f"unexpected SHA-256 for {relative}")
        return path

    structure = read_structure(checked("metadata/structures/2G5H.cif"), assembly="1", model=1)
    if tuple(structure.available_assemblies) != ("1",) or tuple(structure.available_models) != (1,):
        raise ValueError("unexpected 2G5H deposition assembly/model options")
    refs = []
    for comp, rid in manifest["references"].items():
        seqs = [
            str(r.seq) for r in SeqIO.parse(checked(f"{prep}/{comp}.fasta"), "fasta") if r.id == rid
        ]
        if len(seqs) != 1:
            raise ValueError("absent or duplicate reference")
        refs.append(
            ReferenceSequence(comp, rid, seqs[0], "protein", organism="Streptococcus pneumoniae")
        )
    aligned = map_components(refs, structure)
    store = load_score_store(scores) if scores else None
    fields = ("mi_bits", "mi_apc", "dca_frobenius", "dca_apc")
    if store:
        if set(store.scores) != set(fields):
            raise ValueError("supply all four complete GatABC pair statistics")
        for name, filename in (
            ("A", "A.fasta"),
            ("B", "B.fasta"),
            ("C", "C.fasta"),
            ("matches", "matches.tsv"),
        ):
            if (
                store.provenance["inputs"][name]["sha256"]
                != manifest["supplied_sha256"][f"{prep}/{filename}"]
            ):
                raise ValueError(f"score store uses a different prepared input: {name}")
    scored = []
    labels = {}
    with checked(manifest["paths"]["labels"]).open(encoding="utf-8-sig", newline="") as h:
        for row in csv.DictReader(h, delimiter="\t"):
            key = (
                row["x_component"] + row["y_component"],
                int(row["x_reference_pos"]),
                int(row["y_reference_pos"]),
            )
            if key in labels:
                raise ValueError("duplicate historical label")
            labels[key] = int(row["contact"])
    counts = {p: Counter() for p in ("AB", "AC", "BC")}
    archival = {p: Counter() for p in counts}
    named = {(r["pair"], *r["positions"]) for r in manifest["reference_pairs"]}
    examples = []
    worst = 0.0
    for row in iter_distances(
        aligned, metric="heavy", close=8.0, distant_metric="ca", distant=25.0, score_store=store
    ):
        pair = row["pair"]
        k = (pair, row["x_reference_pos"], row["y_reference_pos"])
        stats = counts[pair]
        stats["pairs"] += 1
        if row["observable"]:
            stats["observed"] += 1
            if store and all(row[name] is not None for name in fields):
                scored.append((row["contact"], *(row[name] for name in fields)))
                stats["observable_scoreable"] += 1
                stats["close_scoreable"] += row["contact"]
            for cutoff in (4, 5, 8):
                stats[f"heavy_within_{cutoff}"] += int(row["heavy_distance"] <= cutoff)
        if row["cb_distance"] is not None:
            stats["cb_observed"] += 1
            stats["cb_within_8"] += int(row["cb_distance"] <= 8)
        if k in labels:
            positive = bool(labels[k])
            class_name = "positive" if positive else "distant"
            value = row["distant_control_label"]
            measurable = (
                row["observable"]
                if positive
                else row["observable"] and row["ca_distance"] is not None
            )
            agreement = (
                "agrees" if value == labels[k] else ("disagrees" if measurable else "unresolved")
            )
            archival[pair][class_name + "_" + agreement] += 1
            if row["heavy_distance"] is not None:
                residues = [
                    aligned.roles[c].sites[aligned.references[c].positions.index(pos)].residue
                    for c, pos in zip(pair, k[1:], strict=True)
                ]
                atoms = [
                    [
                        a.xyz
                        for a in res.atoms
                        if a.element not in ("H", "D", "T") and a.occupancy > 0
                    ]
                    for res in residues
                ]
                brute = min(math.dist(a, b) for a in atoms[0] for b in atoms[1])
                worst = max(worst, abs(brute - row["heavy_distance"]))
        if k in named:
            examples.append(
                {
                    "pair": pair,
                    "reference_positions": list(k[1:]),
                    "author": [
                        [
                            row[s + "_auth_asym_id"],
                            row[s + "_mon_id"],
                            row[s + "_auth_seq_id"],
                            row[s + "_insertion_code"],
                        ]
                        for s in "xy"
                    ],
                    "heavy": row["heavy_distance"],
                    "ca": row["ca_distance"],
                    "cb": row["cb_distance"],
                    "original_label": labels.get(k),
                }
            )
    result = {
        "structure_sha256": structure.sha256,
        "assembly": "1",
        "model": 1,
        "roles": {
            c: {
                "chain": r.chain.copy_id if r.chain else None,
                "identity": r.identity,
                "coverage": r.coverage,
                "reference_length": len(aligned.references[c].sequence),
                "site_status": dict(Counter(s.status for s in r.sites)),
            }
            for c, r in aligned.roles.items()
        },
        "historical_comparison": {p: dict(v) for p, v in archival.items()},
        "full_reference_pairs": {p: dict(v) for p, v in counts.items()},
        "max_heavy_error_against_direct_atom_pairs_angstrom": worst,
        "examples": examples,
    }
    if store:
        array = np.asarray(scored).reshape(-1, len(fields) + 1)
        result["full_score_comparison"] = {
            name: _metrics(array[:, 0].astype(int), array[:, i], curves=False)
            for i, name in enumerate(fields, start=1)
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--scores",
        type=Path,
        help="Optional checksummed score artifact for the four legacy pair statistics.",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root, scores=args.scores), indent=2, allow_nan=False))
