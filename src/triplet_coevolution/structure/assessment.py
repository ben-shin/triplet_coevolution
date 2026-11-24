"""Summarize measured structural distances, candidate geometry and declared controls."""

import csv
import math
import random
from bisect import insort
from collections import Counter
from pathlib import Path

from triplet_coevolution.io.score_store import PAIRS


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(
            stream, delimiter="," if path.suffix.lower() == ".csv" else "\t", strict=True
        )
        if not rows.fieldnames or len(rows.fieldnames) != len(set(rows.fieldnames)):
            raise ValueError(f"missing or duplicate header in {path.name}")
        for row in rows:
            if None in row or None in row.values():
                raise ValueError(f"wrong field count at {path.name} line {rows.line_num}")
            yield row


def _write(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _triples(path, mapping):
    triples = []
    edges = {}
    if not path:
        return triples, edges
    seen = set()
    for row in _rows(path):
        keys = []
        for component, ref in mapping.references.items():
            prefix = component.lower()
            if (
                row.get(prefix + "_reference_id") != ref.reference_id
                or prefix + "_reference_pos" not in row
            ):
                raise ValueError(
                    f"triple table reference identity/columns disagree for {component}"
                )
            position = row[prefix + "_reference_pos"]
            coordinate = int(position) if position else None
            if coordinate is not None and coordinate not in ref.positions:
                raise ValueError(f"triple table position outside {component} reference")
            keys.append(coordinate)
        # Several non-reference insert columns can legitimately lack a numbered coordinate.
        if None not in keys:
            if tuple(keys) in seen:
                raise ValueError("duplicate numbered candidate in triple table")
            seen.add(tuple(keys))
        edge_keys = [
            (pair, keys["ABC".index(pair[0])], keys["ABC".index(pair[1])]) for pair in PAIRS
        ]
        for key in edge_keys:
            edges[key] = None
        triples.append((row, edge_keys))
    if not triples:
        raise ValueError("triple table has no candidate records")
    return triples, edges


def summarize_geometry(directory: Path, config, mapping, *, statistic: str | None):
    """Scan the full distance file once, retaining the requested viewer and triple rows."""
    triples, edges = _triples(config.triples, mapping)
    closest = {p: [] for p in PAIRS}
    counts = {p: [0] * len(config.thresholds) for p in PAIRS}
    measured = dict.fromkeys(PAIRS, 0)
    for serial, row in enumerate(_rows(directory / "distances.tsv")):
        if not row["distance_angstrom"]:
            continue
        pair = row["pair"]
        x, y = int(row["x_reference_pos"]), int(row["y_reference_pos"])
        d = float(row["distance_angstrom"])
        measured[pair] += 1
        for i, cutoff in enumerate(config.thresholds):
            counts[pair][i] += int(d <= cutoff)
        if (pair, x, y) in edges:
            edges[pair, x, y] = d
        if config.viewer_limit and (not statistic or row[statistic]):
            # All stored pair scores rank high-first. Ties follow physical coordinate order.
            key = -float(row[statistic]) if statistic else d
            best = closest[pair]
            item = (key, x, y, serial, row)
            if len(best) < config.viewer_limit or item[:4] < best[-1][:4]:
                insort(best, item)
                if len(best) > config.viewer_limit:
                    best.pop()
    fields = ("interface", "metric", "cutoff_angstrom", "close", "observable", "prevalence")
    entries = [
        dict(
            zip(
                fields,
                (p, config.metric, t, k, measured[p], k / measured[p] if measured[p] else None),
                strict=True,
            )
        )
        for p in PAIRS
        for t, k in zip(config.thresholds, counts[p], strict=True)
    ]
    _write(directory / "cutoff_sensitivity.tsv", fields, entries)
    extra = {"cutoffs": "cutoff_sensitivity.tsv", "thresholds_angstrom": config.thresholds}
    if config.triples:
        fields = (
            "ab_distance",
            "ac_distance",
            "bc_distance",
            "observable_edges",
            "close_edges",
            "geometry",
            "min_edge",
            "span",
        )
        reserved = set(fields).intersection(triples[0][0])
        if reserved:
            raise ValueError(f"triple input uses reserved geometry columns {sorted(reserved)}")

        def triple_rows():
            for row, keys in triples:
                distances = [edges[k] for k in keys]
                observed = [d for d in distances if d is not None]
                close = sum(d <= config.close for d in observed)
                complete = len(observed) == 3
                geometry = (
                    ("separated", "one_close_edge", "wedge", "triangle")[close]
                    if complete
                    else "unresolved"
                )
                yield {
                    **row,
                    **dict(
                        zip(
                            fields,
                            (
                                *distances,
                                len(observed),
                                close,
                                geometry,
                                min(observed) if complete else None,
                                max(observed) if complete else None,
                            ),
                            strict=True,
                        )
                    ),
                }

        _write(directory / "triple_geometry.tsv", (*triples[0][0], *fields), triple_rows())
        extra["triples"] = {
            "path": "triple_geometry.tsv",
            "count": len(triples),
            "span": "maximum of three observed edge distances",
        }
    return [item[-1] for p in PAIRS for item in closest[p]], extra


def sample_distant(directory: Path, ratio: float, seed: int, interfaces: dict):
    """Sample without replacement within each interface; preserve every close pair."""
    label_path = directory / "distant_labels.tsv"
    targets = {
        p: min(interfaces[p]["distant"], math.floor(interfaces[p]["close"] * ratio)) for p in PAIRS
    }
    samples = {p: [] for p in PAIRS}
    seen = dict.fromkeys(PAIRS, 0)
    rng = {p: random.Random(f"{seed}:{p}") for p in PAIRS}
    for row in _rows(label_path):
        if row["contact"] != "0":
            continue
        p = row["x_component"] + row["y_component"]
        seen[p] += 1
        key = (row["x_reference_pos"], row["y_reference_pos"])
        reservoir = samples[p]
        if len(reservoir) < targets[p]:
            reservoir.append(key)
        elif targets[p]:
            slot = rng[p].randrange(seen[p])
            if slot < targets[p]:
                reservoir[slot] = key
    chosen = {p: set(keys) for p, keys in samples.items()}
    with label_path.open(encoding="utf-8", newline="") as handle:
        fields = next(csv.reader(handle, delimiter="\t"))
    rows = (
        row
        for row in _rows(label_path)
        if row["contact"] == "1"
        or (row["x_reference_pos"], row["y_reference_pos"])
        in chosen[row["x_component"] + row["y_component"]]
    )
    _write(directory / "sampled_labels.tsv", fields, rows)
    return {
        "path": "sampled_labels.tsv",
        "seed": seed,
        "ratio": ratio,
        "available_distant": seen,
        "sampled_distant": {p: len(s) for p, s in samples.items()},
        "method": "uniform reservoir without replacement within each interface",
    }


def annotate_functional(directory, path, mapping):
    """Map reported assay observations without recomputing or combining their measurements."""
    fields = (
        "assay_id",
        "source",
        "evidence_type",
        "assay_context",
        "units",
        "value",
        "uncertainty",
    )
    site_fields = tuple(f"{c}_{key}" for c in "abc" for key in ("reference_id", "reference_pos"))
    added = tuple(
        f"{c}_{key}"
        for c in "abc"
        for key in ("mapping_status", "copy_id", "auth_asym_id", "auth_seq_id", "insertion_code")
    )
    rows = []
    sites = {
        c: {site.reference_position: site for site in role.sites}
        for c, role in mapping.roles.items()
    }
    for row in _rows(path):
        if not {*fields, *site_fields}.issubset(row) or set(added).intersection(row):
            raise ValueError(
                "functional labels require assay metadata and all three reference column pairs"
            )
        if any(not row[k].strip() for k in fields[:6]) or not math.isfinite(float(row["value"])):
            raise ValueError(
                "functional labels require a finite reported value, units, assay context and source"
            )
        if row["uncertainty"] and (
            not math.isfinite(float(row["uncertainty"])) or float(row["uncertainty"]) < 0
        ):
            raise ValueError(
                "functional measurement uncertainty must be blank or finite and nonnegative"
            )
        extra = dict.fromkeys(added, "")
        present = 0
        for c in "ABC":
            prefix = c.lower() + "_"
            rid, pos = row[prefix + "reference_id"], row[prefix + "reference_pos"]
            if not rid and not pos:
                continue
            if rid != mapping.references[c].reference_id or not pos or int(pos) not in sites[c]:
                raise ValueError(f"functional label coordinate disagrees for {c}")
            present += 1
            site = sites[c][int(pos)]
            chain = mapping.roles[c].chain
            extra[prefix + "mapping_status"] = site.status
            if chain and site.residue:
                extra[prefix + "copy_id"] = chain.copy_id
                extra[prefix + "auth_asym_id"] = chain.auth_asym_id
                extra[prefix + "auth_seq_id"] = site.residue.auth_seq_id
                extra[prefix + "insertion_code"] = site.residue.insertion_code
        if not present:
            raise ValueError("functional assay record requires at least one reference coordinate")
        rows.append({**row, **extra})
    if not rows:
        raise ValueError("functional label table is empty")
    _write(directory / "functional_evidence.tsv", rows[0].keys(), rows)
    return {
        "path": "functional_evidence.tsv",
        "observations": len(rows),
        "interpretation": "Supplied measurements retain their assay, units and independent source.",
    }


def matched_hard_controls(directory, config, mapping):
    """Match above-cutoff neighbors by interface and caller-measured ordered exposure bins."""
    bins = {}
    for row in _rows(config.exposure_bins):
        if set(row) != {"component", "reference_id", "reference_pos", "exposure_bin"}:
            raise ValueError(
                "exposure TSV needs component, reference_id, reference_pos and exposure_bin"
            )
        c = row["component"]
        if c not in mapping.references or row["reference_id"] != mapping.references[c].reference_id:
            raise ValueError("exposure table reference mismatch")
        position = int(row["reference_pos"])
        if (
            position not in mapping.references[c].positions
            or (c, position) in bins
            or not row["exposure_bin"].strip()
        ):
            raise ValueError("invalid, duplicate or unbinned exposure coordinate")
        bins[c, position] = row["exposure_bin"]

    def stratum(row):
        x, y = row["x_component"], row["y_component"]
        bx = bins.get((x, int(row["x_reference_pos"])))
        by = bins.get((y, int(row["y_reference_pos"])))
        return (x + y, bx, by) if bx is not None and by is not None else None

    path = directory / "labels.tsv"
    positives = Counter()
    excluded = 0
    for row in _rows(path):
        if row["contact"] == "1":
            group = stratum(row)
            if group:
                positives[group] += 1
            else:
                excluded += 1
    # Per-stratum reservoirs cap memory by the number of requested controls.
    samples = {g: [] for g in positives}
    available = Counter()
    ratio = 1.0 if config.negative_ratio is None else config.negative_ratio
    rng = random.Random(config.seed)
    for row in _rows(path):
        if row["contact"] != "0" or float(row["distance_angstrom"]) > config.hard_negative_upper:
            continue
        group = stratum(row)
        if group not in positives:
            continue
        available[group] += 1
        target = math.floor(positives[group] * ratio)
        key = (
            row["x_component"],
            row["x_reference_pos"],
            row["y_component"],
            row["y_reference_pos"],
        )
        reservoir = samples[group]
        if len(reservoir) < target:
            reservoir.append(key)
        elif target:
            slot = rng.randrange(available[group])
            if slot < target:
                reservoir[slot] = key
    # A stratum without either class supplies no matched comparison.
    realized = {g for g, vals in samples.items() if vals}
    controls = {k for g in realized for k in samples[g]}
    with path.open(encoding="utf-8", newline="") as stream:
        fields = next(csv.reader(stream, delimiter="\t"))
    rows = (
        r
        for r in _rows(path)
        if (r["contact"] == "1" and stratum(r) in realized)
        or (r["x_component"], r["x_reference_pos"], r["y_component"], r["y_reference_pos"])
        in controls
    )
    _write(directory / "hard_labels.tsv", fields, rows)
    return {
        "path": "hard_labels.tsv",
        "seed": config.seed,
        "ratio": ratio,
        "distance_upper_inclusive": config.hard_negative_upper,
        "positive_missing_exposure": excluded,
        "strata": [
            {
                "interface": g[0],
                "x_bin": g[1],
                "y_bin": g[2],
                "positive": n,
                "positive_included": n if g in realized else 0,
                "available_controls": available[g],
                "sampled_controls": len(samples[g]),
            }
            for g, n in sorted(positives.items())
        ],
    }
