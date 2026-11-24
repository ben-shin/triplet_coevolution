"""Stream inter-component distances over all reference pairs in a selected assembly/model."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from triplet_coevolution.io.score_store import PAIRS, ScoreStore
from triplet_coevolution.structure.mapping import MappingResult, comparison_spelling

METRICS = ("heavy", "ca", "cb")
END_FIELDS = (
    "component",
    "reference_id",
    "reference_pos",
    "reference_symbol",
    "copy_id",
    "label_asym_id",
    "auth_asym_id",
    "entity_id",
    "polymer_index",
    "mon_id",
    "auth_seq_id",
    "insertion_code",
    "altloc",
    "mapping_status",
)
DIST_FIELDS = tuple(f"{side}_{name}" for side in "xy" for name in END_FIELDS)
OUTPUT_FIELDS = (
    "heavy_distance",
    "ca_distance",
    "cb_distance",
    "distance_angstrom",
    "distant_metric_distance",
    "category",
    "observable",
    "scoreable",
    "contact",
    "distant_control_label",
)
LABEL_FIELDS = tuple(
    f"{s}_{k}" for s in "xy" for k in ("component", "reference_id", "reference_pos")
) + ("contact", "distance_angstrom", "source", "status")


def _end(mapping, component, site):
    role = mapping.roles[component]
    chain = role.chain
    r = site.residue
    return dict(
        zip(
            END_FIELDS,
            (
                component,
                mapping.references[component].reference_id,
                site.reference_position,
                site.reference_symbol,
                chain.copy_id if chain else None,
                chain.label_asym_id if chain else None,
                chain.auth_asym_id if chain else None,
                chain.entity_id if chain else None,
                r.seq_index if r else None,
                r.mon_id if r else None,
                r.auth_seq_id if r else None,
                r.insertion_code if r else None,
                r.altloc if r else None,
                site.status,
            ),
            strict=True,
        )
    )


def _atoms(site, protein):
    heavy = []
    backbone = {}
    if site.residue:
        for atom in site.residue.atoms:
            if atom.element in {"H", "D", "T"} or atom.occupancy <= 0:
                continue
            heavy.append(atom.xyz)
            if protein and atom.name in {"CA", "CB"} and atom.element == "C":
                backbone[atom.name] = atom.xyz
    ca = backbone.get("CA")
    cb = ca if protein and site.residue and site.residue.letter == "G" else backbone.get("CB")
    return np.array(heavy, dtype=float).reshape(-1, 3), ca, cb


def _targets(atoms):
    lengths = np.asarray([len(h) for h, _, _ in atoms], dtype=int)
    has = np.flatnonzero(lengths)
    all_heavy = np.concatenate([atoms[i][0] for i in has]) if len(has) else np.empty((0, 3))
    starts = np.cumsum(lengths[has]) - lengths[has]
    points = [
        np.asarray([a[k] if a[k] is not None else (np.nan,) * 3 for a in atoms], dtype=float)
        for k in (1, 2)
    ]
    return has, starts, all_heavy, *points


def _row_distances(query, targets, n):
    hits, starts, heavy, ca, cb = targets
    q, qca, qcb = query
    distances = np.full(n, np.nan)
    if len(q) and len(heavy):
        # Bound temporary storage independently of the target chain length.
        minima = np.full(len(heavy), np.inf)
        for atom in q:
            delta = heavy - atom
            np.minimum(minima, np.einsum("ij,ij->i", delta, delta), out=minima)
        distances[hits] = np.sqrt(np.minimum.reduceat(minima, starts))
    return (
        distances,
        np.linalg.norm(ca - qca, axis=1) if qca is not None else np.full(n, np.nan),
        np.linalg.norm(cb - qcb, axis=1) if qcb is not None else np.full(n, np.nan),
    )


def _score_axes(store, mapping):
    if store is None:
        return {}, {}
    axes = {}
    for m in store.maps:
        if m.reference_id != mapping.references[m.component].reference_id:
            raise ValueError(
                f"score artifact reference differs from the structural reference for {m.component}"
            )
        lookup = {p: i for i, p in enumerate(m.reference_positions) if p is not None}
        seq = mapping.references[m.component]
        reference = comparison_spelling(seq.sequence, seq.molecule)
        for p, i in lookup.items():
            if (
                p not in seq.positions
                or comparison_spelling(m.symbols[i].upper(), seq.molecule)
                != reference[seq.positions.index(p)]
            ):
                raise ValueError(
                    f"score artifact reference sequence disagrees at {m.component} {p}"
                )
        axes[m.component] = lookup
    ranks = {}
    for pair in PAIRS:
        keep = store.eligible(pair, reference_only=True)
        for stat, blocks in store.scores.items():
            vals = blocks[pair]
            mask = keep & np.isfinite(vals)
            ordered = np.sort(vals[mask])
            rank = np.zeros(vals.shape, dtype=np.int64)
            rank[mask] = len(ordered) + 1 - np.searchsorted(ordered, vals[mask], side="right")
            ranks[stat, pair] = rank
    return axes, ranks


def iter_distances(
    mapping: MappingResult,
    *,
    metric: str = "heavy",
    close: float = 5.0,
    distant: float = 25.0,
    distant_metric: str | None = None,
    score_store: ScoreStore | None = None,
) -> Iterator[dict]:
    """Yield complete reference pairs with observed atom geometry and conditional labels."""
    far = distant_metric or metric
    if (
        metric not in METRICS
        or far not in METRICS
        or not np.isfinite((close, distant)).all()
        or close <= 0
        or distant <= close
    ):
        raise ValueError(
            "choose heavy, ca or cb distance metrics and finite 0 < close < distant cutoffs in angstroms"
        )
    ends = {c: [_end(mapping, c, s) for s in role.sites] for c, role in mapping.roles.items()}
    atoms = {
        c: [_atoms(s, mapping.references[c].molecule == "protein") for s in role.sites]
        for c, role in mapping.roles.items()
    }
    targets = {c: _targets(a) for c, a in atoms.items()}
    axes, ranks = _score_axes(score_store, mapping)
    for pair in PAIRS:
        a, b = pair
        ys = [{f"y_{k}": v for k, v in endpoint.items()} for endpoint in ends[b]]
        for i, start in enumerate(ends[a]):
            base = {"pair": pair, **{f"x_{k}": v for k, v in start.items()}}
            hd, ad, bd = _row_distances(atoms[a][i], targets[b], len(ys))
            all_dist = dict(zip(METRICS, (hd, ad, bd), strict=True))
            pos_dist = all_dist[metric]
            far_dist = all_dist[far]
            for j, target in enumerate(ys):
                row = {**base, **target}
                d, f = pos_dist[j], far_dist[j]
                if not np.isfinite(d):
                    category = "unresolved"
                    label = control = None
                elif d <= close:
                    category = "close"
                    label = control = 1
                elif np.isfinite(f) and f >= distant:
                    category = "distant"
                    label = control = 0
                else:
                    category = "intermediate" if np.isfinite(f) else "unmeasured_distant_metric"
                    label = 0
                    control = None
                for name, values in all_dist.items():
                    row[name + "_distance"] = float(values[j]) if np.isfinite(values[j]) else None
                row.update(
                    distance_angstrom=row[metric + "_distance"],
                    distant_metric_distance=row[far + "_distance"],
                    category=category,
                    observable=label is not None,
                    scoreable=None,
                    contact=label,
                    distant_control_label=control,
                )
                if score_store is not None:
                    x = axes[a].get(row["x_reference_pos"])
                    y = axes[b].get(row["y_reference_pos"])
                    row["scoreable"] = False
                    for name, blocks in score_store.scores.items():
                        rank = ranks[name, pair][x, y] if x is not None and y is not None else 0
                        row[name] = float(blocks[pair][x, y]) if rank else None
                        row[name + "_rank"] = int(rank) if rank else None
                        if rank:
                            row["scoreable"] = True
                yield row


def _mapping_table(path, mapping):
    header = [
        *END_FIELDS,
        "polymer_symbol",
        "correspondence",
        "possible_polymer_indices",
        "alternate_copy_correspondences",
        "deposited_conformers",
        "correspondence_confidence",
        "gap_possible",
        "explicit_anchor",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        out = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        out.writeheader()
        for c in "ABC":
            for site in mapping.roles[c].sites:
                row = _end(mapping, c, site)
                row.update(
                    polymer_symbol=site.residue.letter if site.residue else None,
                    correspondence=(
                        "identical"
                        if site.reference_symbol == site.residue.letter
                        else "substitution"
                    )
                    if site.residue
                    else None,
                    possible_polymer_indices=",".join(map(str, site.alternatives)),
                    alternate_copy_correspondences=json.dumps(
                        [
                            {"copy_id": name, "polymer_indices": indices, "gap_possible": gap}
                            for name, indices, gap in site.alternative_copies
                        ],
                        separators=(",", ":"),
                    ),
                    deposited_conformers=",".join(site.residue.alternatives)
                    if site.residue
                    else None,
                    correspondence_confidence=(
                        "validated_anchor"
                        if site.explicit and site.residue
                        else "invariant_across_optimal_alignments"
                        if site.residue
                        else site.status
                    ),
                    gap_possible=site.possible_gap,
                    explicit_anchor=site.explicit,
                )
                out.writerow(row)


def ensemble_distances(
    mappings: Sequence[MappingResult],
    *,
    metric: str = "heavy",
    close: float = 5.0,
    distant: float = 25.0,
    distant_metric: str | None = None,
) -> Iterator[dict]:
    """Measure model/copy sensitivity across explicit alternatives, without averaging coordinates."""
    if not mappings:
        raise ValueError("supply at least one explicitly selected structural alternative")
    identities = []
    for m in mappings:
        if m.structure.sha256 != mappings[0].structure.sha256:
            raise ValueError(
                "ensemble alternatives must come from the same deposited coordinate source"
            )
        if m.references != mappings[0].references:
            raise ValueError(
                "ensemble comparisons require identical reference sequences and coordinates"
            )
        ids = (
            m.structure.sha256,
            m.structure.assembly,
            m.structure.model,
            tuple(m.roles[c].chain.copy_id if m.roles[c].chain else None for c in "ABC"),
        )
        if ids in identities:
            raise ValueError("duplicate structural alternative")
        identities.append(ids)
    streams = [
        iter_distances(
            m, metric=metric, close=close, distant=distant, distant_metric=distant_metric
        )
        for m in mappings
    ]
    for rows in zip(*streams, strict=True):
        first = rows[0]
        observed = [r["distance_angstrom"] for r in rows if r["observable"]]
        labels = [r["contact"] for r in rows if r["observable"]]
        agreed = len(labels) == len(rows) and len(set(labels)) == 1
        yield {
            "pair": first["pair"],
            "x_reference_pos": first["x_reference_pos"],
            "y_reference_pos": first["y_reference_pos"],
            "alternatives": len(rows),
            "measured": len(observed),
            "min_angstrom": min(observed, default=None),
            "max_angstrom": max(observed, default=None),
            "observed_close_fraction": sum(labels) / len(labels) if labels else None,
            "consensus_contact": labels[0] if agreed else None,
            "status": "consistent"
            if agreed
            else ("incomplete" if len(labels) < len(rows) else "discordant"),
        }


def write_structure_tables(
    directory: Path,
    mapping: MappingResult,
    *,
    metric: str = "heavy",
    close: float = 5.0,
    distant: float = 25.0,
    distant_metric: str | None = None,
    score_store: ScoreStore | None = None,
    source_kind: str | None = None,
) -> dict:
    """Write mappings, streamed distance and label tables, then publish their verified provenance."""
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    names = [
        "mapping.tsv",
        "distances.tsv",
        "labels.tsv",
        "distant_labels.tsv",
        "structure_summary.json",
    ]
    if any((directory / n).exists() for n in names):
        raise FileExistsError("structure outputs already exist")
    source = f"{mapping.structure.entry_id} assembly={mapping.structure.assembly} model={mapping.structure.model} sha256={mapping.structure.sha256}"
    if source_kind is not None:
        if source_kind not in {"experimental", "predicted", "synthetic", "unspecified"}:
            raise ValueError("invalid structure source kind")
        source = f"{source_kind} {source}"
    counts = {p: Counter() for p in PAIRS}
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        stage = Path(tmp)
        _mapping_table(stage / names[0], mapping)
        scores = [k for s in score_store.scores for k in (s, s + "_rank")] if score_store else []
        with (
            (stage / names[1]).open("w", encoding="utf-8", newline="") as geom,
            (stage / names[2]).open("w", encoding="utf-8", newline="") as full,
            (stage / names[3]).open("w", encoding="utf-8", newline="") as controls,
        ):
            distance_writer = csv.DictWriter(
                geom,
                fieldnames=["pair", *DIST_FIELDS, *OUTPUT_FIELDS, *scores],
                delimiter="\t",
                lineterminator="\n",
            )
            distance_writer.writeheader()
            label_writers = [
                csv.DictWriter(h, fieldnames=LABEL_FIELDS, delimiter="\t", lineterminator="\n")
                for h in (full, controls)
            ]
            for writer in label_writers:
                writer.writeheader()
            for row in iter_distances(
                mapping,
                metric=metric,
                close=close,
                distant=distant,
                distant_metric=distant_metric,
                score_store=score_store,
            ):
                distance_writer.writerow(row)
                stat = counts[row["pair"]]
                stat["reference_pairs"] += 1
                stat[row["category"]] += 1
                stat["observable"] += int(row["observable"])
                stat["scoreable"] += int(bool(row["scoreable"]))
                stat["observable_scoreable"] += int(row["observable"] and bool(row["scoreable"]))
                stat["close_scoreable"] += int(row["contact"] == 1 and bool(row["scoreable"]))
                label = {k: row[k] for k in LABEL_FIELDS[:6]}
                label["source"] = source
                label["distance_angstrom"] = row["distance_angstrom"]
                for writer, key in zip(
                    label_writers, ("contact", "distant_control_label"), strict=True
                ):
                    v = row[key]
                    label["contact"] = v
                    label["status"] = (
                        "resolved"
                        if v is not None
                        else ("unknown" if row["observable"] else "unresolved")
                    )
                    writer.writerow(label)
        interfaces = {}
        for pair, c in counts.items():
            summary = {
                name: c[name]
                for name in (
                    "reference_pairs",
                    "observable",
                    "close",
                    "intermediate",
                    "distant",
                    "unmeasured_distant_metric",
                    "unresolved",
                )
            }
            summary["prevalence"] = c["close"] / c["observable"] if c["observable"] else None
            summary["scoreable"] = c["scoreable"] if score_store is not None else None
            summary["observable_scoreable"] = (
                c["observable_scoreable"] if score_store is not None else None
            )
            summary["scoreable_prevalence"] = (
                c["close_scoreable"] / c["observable_scoreable"]
                if c["observable_scoreable"]
                else None
            )
            interfaces[pair] = summary
        roles = {
            c: {
                "status": v.status,
                "copy_id": v.chain.copy_id if v.chain else None,
                "assembly_operations": list(v.chain.operations) if v.chain else [],
                "applied_transform": v.chain.transform if v.chain else None,
                "reference_id": mapping.references[c].reference_id,
                "reference_organism": mapping.references[c].organism,
                "identity": v.identity,
                "coverage": v.coverage,
                "site_statuses": dict(Counter(s.status for s in v.sites)),
                "candidates": [asdict(x) for x in v.candidates],
            }
            for c, v in mapping.roles.items()
        }
        hashes = {}
        for name in names[:-1]:
            with (stage / name).open("rb") as h:
                hashes[name] = hashlib.file_digest(h, "sha256").hexdigest()
        summary = {
            "schema_version": 1,
            "entry_id": mapping.structure.entry_id,
            "structure_sha256": mapping.structure.sha256,
            "assembly": mapping.structure.assembly,
            "model": mapping.structure.model,
            "method": mapping.structure.method,
            "coordinate_evidence": (
                "experimental"
                if any(
                    word in mapping.structure.method.lower()
                    for word in ("diffraction", "microscopy", "nmr", "scattering")
                )
                else "predicted"
                if any(
                    word in mapping.structure.method.lower()
                    for word in ("theoretical", "prediction", "predicted", "computational")
                )
                else "unspecified"
            ),
            "available_assemblies": mapping.structure.available_assemblies,
            "available_models": mapping.structure.available_models,
            "altloc_policy": mapping.structure.altloc_policy,
            "warnings": list(mapping.structure.warnings),
            "mapping_settings": mapping.settings,
            "roles": roles,
            "interfaces": interfaces,
            "definitions": {
                "unit": "angstrom",
                "metric": metric,
                "close_inclusive": close,
                "distant_metric": distant_metric or metric,
                "distant_inclusive": distant,
                "heavy": "Minimum distance between deposited positive-occupancy nonhydrogen atoms.",
                "ca": "Protein alpha carbon to alpha carbon.",
                "cb": "Protein beta carbon to beta carbon, alpha carbon for glycine.",
                "labels": "One means selected metric <= close; zero means measured above that cutoff. Absence of modeled atoms leaves the label unknown. Distances are conditional on deposited atoms and this model/copy.",
                "distant_labels": "One means close; zero means outside close and measured distant metric >= distant. Other pairs are unknown.",
                "prevalence": "Close / measurable selected-metric pairs in the stated interface. Scoreable prevalence additionally requires at least one finite eligible stored score.",
            },
            "output_sha256": hashes,
        }
        (stage / names[-1]).write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        published = []
        try:
            for name in names:
                os.link(stage / name, directory / name)
                published.append(directory / name)
        except BaseException:
            for p in published:
                p.unlink()
            raise
    return summary
