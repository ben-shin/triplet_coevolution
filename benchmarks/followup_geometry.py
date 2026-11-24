"""Compare connected-information candidates with matched controls on deposited ABC geometry."""

import argparse
import csv
import json
from collections import Counter
from math import ceil, floor
from pathlib import Path

import numpy as np
from cohort_sources import atomic_json, digest
from followup_plm import DEST, PAIRS, REG_SHA, raw_geometry
from independent_pair_scores import OUT, prepare

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.maxent import MaxEntOptions, fit_joint
from triplet_coevolution.io.score_store import load_score_store
from triplet_coevolution.validation.benchmark import _metrics

FEATURES = (
    "support",
    "entropy_A",
    "entropy_B",
    "entropy_C",
    "AB_percentile",
    "AC_percentile",
    "BC_percentile",
)
CALIPERS = np.array([0.08, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])


def percentiles(a):
    valid = np.isfinite(a)
    result = np.full(a.shape, np.nan)
    _, inv, n = np.unique(a[valid], return_inverse=True, return_counts=True)
    result[valid] = ((np.cumsum(n) - n / 2) / n.sum())[inv]
    return result


def select_matched(features, scores, ids, ncase):
    """Choose cases and matched controls from sequence-derived features with deterministic tie handling."""
    features = np.asarray(features, dtype=float)
    scores = np.asarray(scores)
    ids = np.asarray(ids)
    if (
        features.shape != (len(ids), 7)
        or len(ids) != len(scores)
        or len(set(ids)) != len(ids)
        or not np.all(np.isfinite(features))
        or not np.all((features >= 0) & (features <= 1))
        or not np.all(np.isfinite(scores))
        or not 0 <= ncase <= len(ids)
    ):
        raise ValueError("expected finite seven-feature rows and unique eligible candidate IDs")
    order = np.lexsort((ids, -scores))
    cases, pool = order[:ncase], order[ncase:]
    # Flat ID is the final key after equal distances; a stable order thus resolves ties.
    pool = pool[np.argsort(ids[pool])]
    available = np.ones(len(pool), dtype=bool)
    matches = []
    for c in cases:
        delta = np.abs(features[pool] - features[c])
        mask = available & np.all(delta <= CALIPERS, axis=1)
        if np.any(mask):
            admissible = np.flatnonzero(mask)
            best = admissible[np.argmin(np.sum((delta[admissible] / CALIPERS) ** 2, axis=1))]
            matches.append((int(c), int(pool[best])))
            available[best] = False
    return cases, pool, np.asarray(matches, dtype=int).reshape(-1, 2)


def balance(x, y):
    out = {}
    for j, key in enumerate(FEATURES):
        a, b = x[:, j], y[:, j]
        v = (a.var(ddof=1) + b.var(ddof=1)) / 2 if min(len(a), len(b)) > 1 else 0
        out[key] = {
            "case_mean": float(a.mean()) if len(a) else None,
            "control_mean": float(b.mean()) if len(b) else None,
            "smd": float((a.mean() - b.mean()) / np.sqrt(v)) if v > 0 else None,
        }
    return out


def make_selection(system):
    data, maps, _, _ = prepare(system)
    historical = load_score_store(OUT / system / "uniform")
    if any(
        tuple(m.reference_positions) != tuple(h.reference_positions)
        for m, h in zip(maps, historical.maps, strict=True)
    ):
        raise ValueError("score axes differ")
    edge_ranks = {p: percentiles(historical.scores["dca_apc"][p]) for p in PAIRS}
    encoded = []
    entropies = []
    for component in data.components:
        alpha = CANONICAL[component.molecule]
        q = len(alpha)
        values = component.matrix
        counts = np.array([(values == k).sum(axis=0) for k in range(q)])
        frequencies = counts / counts.sum(axis=0)
        entropies.append(
            -np.sum(frequencies * np.log2(np.where(frequencies > 0, frequencies, 1)), axis=0)
            / np.log2(q)
        )
        # Both canonical alphabets are in lexical order.
        assert alpha == "".join(sorted(alpha))
        codes = (values != np.argmax(counts, axis=0)).astype(np.int8)
        codes[(values < 0) | (values >= q)] = -1
        encoded.append(codes)
    dims = tuple(c.n_sites for c in data.components)
    rng = np.random.Generator(np.random.PCG64(638291))
    flat = np.sort(rng.choice(int(np.prod(dims)), size=8192, replace=False))
    opts = MaxEntOptions(prior_mode="total", prior_mass=1, tolerance=1e-8, max_iterations=1000)
    features = []
    stats = []
    ids = []
    edges = []
    positions = []
    failures = Counter()
    max_residual = 0
    for fid, a, b, c in zip(flat, *np.unravel_index(flat, dims), strict=True):
        xyz = [x[:, j] for x, j in zip(encoded, (a, b, c), strict=True)]
        ok = (xyz[0] >= 0) & (xyz[1] >= 0) & (xyz[2] >= 0)
        n = int(ok.sum())
        if n < ceil(data.n_observations / 2):
            failures["joint_support"] += 1
            continue
        if any(np.ptp(x[ok]) == 0 for x in xyz):
            failures["constant_binary_axis"] += 1
            continue
        ranks = [edge_ranks["AB"][a, b], edge_ranks["AC"][a, c], edge_ranks["BC"][b, c]]
        if not np.all(np.isfinite(ranks)):
            failures["unavailable_lower_order_score"] += 1
            continue
        counts = np.bincount(xyz[0][ok] * 4 + xyz[1][ok] * 2 + xyz[2][ok], minlength=8).reshape(
            2, 2, 2
        )
        score = fit_joint(counts, n_valid=n, options=opts).score
        if not np.isfinite(score.connected_information_bits) or score.status != "ok":
            failures["IPF_" + score.status] += 1
            continue
        max_residual = max(max_residual, float(score.ipf_residual))
        features.append(
            [
                n / data.n_observations,
                *[e[j] for e, j in zip(entropies, (a, b, c), strict=True)],
                *ranks,
            ]
        )
        stats.append(float(score.connected_information_bits))
        edges.append(float(np.mean(ranks)))
        ids.append(int(fid))
        positions.append([m.reference_positions[j] for m, j in zip(maps, (a, b, c), strict=True)])
    x = np.array(features)
    cases, pool, matched = select_matched(x, stats, ids, floor(len(stats) * 0.10))
    out = {
        "registration": REG_SHA,
        "ids": ids,
        "positions": positions,
        "features": features,
        "ci": stats,
        "pair_only": edges,
        "cases": cases.tolist(),
        "controls": pool.tolist(),
        "matched": matched.tolist(),
    }
    report = {
        "drawn": len(flat),
        "eligible": len(ids),
        "excluded": dict(failures),
        "cases": len(cases),
        "control_pool": len(pool),
        "matched": len(matched),
        "unmatched": len(cases) - len(matched),
        "max_ipf_residual": max_residual,
        "balance_all": balance(x[cases], x[pool]),
        "balance_matched": balance(x[matched[:, 0]], x[matched[:, 1]]),
        "maximum_matched_differences": dict(
            zip(
                FEATURES,
                np.abs(x[matched[:, 0]] - x[matched[:, 1]]).max(axis=0).tolist(),
                strict=True,
            )
        )
        if len(matched)
        else {},
    }
    for label, a in (("connected_information_bits", stats), ("pair_only_percentile", edges)):
        v = np.asarray(a)
        report[label] = {
            name: [float(z) for z in np.quantile(v[matched[:, j]], [0, 0.25, 0.5, 0.75, 1])]
            if len(matched)
            else []
            for j, name in enumerate(("candidate", "control"))
        }
    return out, report


def paired_geometry(ds, matched, cutoff):
    ds = np.asarray(ds, dtype=float)
    complete = np.isfinite(ds).all(axis=1)
    nclose = (ds <= cutoff).sum(axis=1)
    low = nclose >= 2
    high = nclose + np.isnan(ds).sum(axis=1) >= 2
    groups = {}
    for j, name in enumerate(("candidate", "control")):
        arr = matched[:, j]
        seen = complete[arr]
        groups[name] = {
            "zero_one_wedge_triangle": np.bincount(nclose[arr[seen]], minlength=4).tolist(),
            "unknown": int((~seen).sum()),
            "event_lower": float(low[arr].mean()) if len(arr) else None,
            "event_upper": float(high[arr].mean()) if len(arr) else None,
        }
    both = matched[complete[matched].all(axis=1)]
    y = low[both]
    delta = ds[both[:, 0]].max(axis=1) - ds[both[:, 1]].max(axis=1) if len(both) else np.array([])
    return {
        "groups": groups,
        "complete_pairs": len(both),
        "unresolved_pairs": len(matched) - len(both),
        "candidate_only_wedge_or_triangle": int((y[:, 0] & ~y[:, 1]).sum()),
        "control_only_wedge_or_triangle": int((~y[:, 0] & y[:, 1]).sum()),
        "both_wedge_or_triangle": int(y.all(axis=1).sum()),
        "neither": int((~y).all(axis=1).sum()),
        "paired_rate_difference": float((y[:, 0].astype(float) - y[:, 1]).mean())
        if len(both)
        else None,
        "paired_mean_span_difference_angstrom": float(delta.mean()) if len(delta) else None,
        "complete_pair_median_span_angstrom": [
            float(np.median(ds[both[:, j]].max(axis=1))) for j in (0, 1)
        ]
        if len(both)
        else [],
    }


def run(system):
    out = DEST / system
    out.mkdir(parents=True, exist_ok=True)
    selection, summary = make_selection(system)
    path = out / "geometry-blind-selection.json"
    atomic_json(path, selection)
    summary.update(system=system, registration=REG_SHA, selection_sha256=digest(path), copies={})
    print("Selection fixed", system, summary["selection_sha256"], summary["matched"], flush=True)
    geometry = raw_geometry(system)
    a, b, c = np.asarray(selection["positions"]).T - 1
    for copy, p in geometry.items():
        d = np.column_stack([p["AB"][a, b], p["AC"][a, c], p["BC"][b, c]])
        full = np.isfinite(d).all(axis=1)
        t = (d[full] <= 8).sum(axis=1) >= 2
        results = {
            "sample_eligible": len(d),
            "sample_observable": int(full.sum()),
            "sample_positive": int(t.sum()),
            "sample_unknown": int((~full).sum()),
            "ranks": {},
            "matched_geometry": {},
        }
        for name in ("ci", "pair_only"):
            metrics = _metrics(t, np.array(selection[name])[full], curves=False)
            results["ranks"][name] = {
                "ap": metrics["average_precision"]
                if metrics["class_support"] == "both_classes"
                else None,
                "auroc": metrics["auroc"],
            }
        for k in (5, 8, 10):
            results["matched_geometry"][str(k)] = paired_geometry(
                d, np.asarray(selection["matched"], dtype=int).reshape(-1, 2), k
            )
        summary["copies"][copy] = results
    atomic_json(out / "geometry.json", summary)
    return summary


def table(path, fields):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != list(fields):
            raise ValueError(f"Expected columns {list(fields)} in {path}")
        return list(reader)


def match_table(source, target):
    rows = table(source, ("id", "connected_information_bits", *FEATURES))
    ids = [int(r["id"]) for r in rows]
    scores = [float(r["connected_information_bits"]) for r in rows]
    features = np.array([[float(r[k]) for k in FEATURES] for r in rows]).reshape(-1, 7)
    cases, pool, matched = select_matched(features, scores, ids, len(ids) // 10)
    atomic_json(
        target,
        {
            "inputs_sha256": digest(source),
            "registration": REG_SHA,
            "ids": ids,
            "cases": cases.tolist(),
            "control_pool": pool.tolist(),
            "matched": matched.tolist(),
            "balance_all": balance(features[cases], features[pool]),
            "balance_matched": balance(features[matched[:, 0]], features[matched[:, 1]]),
        },
    )


def evaluate_table(selection, geometry, target):
    reg = json.loads(selection.read_text(encoding="utf-8"))
    if reg.get("registration") != REG_SHA:
        raise ValueError("Matching rule differs")
    ds = np.full((len(reg["ids"]), 3), np.nan)
    lookup = {v: i for i, v in enumerate(reg["ids"])}
    seen = set()
    for row in table(geometry, ("id", "AB", "AC", "BC")):
        key = int(row["id"])
        if key not in lookup or key in seen:
            raise ValueError("Unknown or duplicate geometry ID")
        seen.add(key)
        vals = [float(row[p]) if row[p].strip() else np.nan for p in PAIRS]
        if any(v < 0 or np.isinf(v) for v in vals):
            raise ValueError("Distances must be finite nonnegative numbers or absent")
        ds[lookup[key]] = vals
    matched = np.array(reg["matched"], dtype=int).reshape(-1, 2)
    atomic_json(
        target,
        {
            "selection_sha256": digest(selection),
            "distance_source_sha256": digest(geometry),
            "matched_geometry": {str(k): paired_geometry(ds, matched, k) for k in (5, 8, 10)},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    study = subs.add_parser("study")
    study.add_argument("system", choices=["rnap", "ribosome"])
    select = subs.add_parser("match")
    select.add_argument("candidate_table", type=Path)
    select.add_argument("selection_json", type=Path)
    measure = subs.add_parser("measure")
    measure.add_argument("selection_json", type=Path)
    measure.add_argument("distance_table", type=Path)
    measure.add_argument("output_json", type=Path)
    args = parser.parse_args()
    if args.cmd == "study":
        print("Finished", run(args.system)["system"])
    elif args.cmd == "match":
        match_table(args.candidate_table, args.selection_json)
    else:
        evaluate_table(args.selection_json, args.distance_table, args.output_json)
