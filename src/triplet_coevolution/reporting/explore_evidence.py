"""Render saved structural, benchmarking and QC evidence beside the score maps."""

import json
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import numpy as np

from triplet_coevolution.reporting.explore_data import (
    PAIRS,
    artifact,
    axis_bins,
    finite,
    records,
    saved_benchmark,
)
from triplet_coevolution.reporting.explore_figures import val, xy_svg
from triplet_coevolution.reporting.html import _link, _table, _text


def sibling(path, name):
    return (PurePosixPath(path).parent / name).as_posix()


def available_link(directory, name, label):
    return _link(name, label) if name and artifact(directory, name).is_file() else ""


def structure_view(directory, outputs, store, *, network_evidence=None):
    path = outputs.get("structure")
    if not path:
        return {}, "", "<p>No coordinate-mapped structure was attached to this report.</p>"
    meta = json.loads(artifact(directory, path).read_text(encoding="utf-8"))
    kind = meta.get("source_kind", "unspecified")
    defs = meta.get("definitions", {})
    caption = f"{kind} {defs.get('metric', 'unspecified')} distance ≤ {val(defs.get('close_inclusive'))} Å"
    detail = f"<p>{_text(kind)} coordinates · {_text(meta.get('entry_id', ''))} · assembly {_text(meta.get('assembly', 'unspecified'))} · model {_text(meta.get('model', 'unspecified'))}.</p>"
    if meta.get("method"):
        detail += f"<p>Deposited method · {_text(meta['method'])}. Method-derived classification · {_text(meta.get('coordinate_evidence', 'unspecified'))}.</p>"
    relative = sibling(path, "distances.tsv")
    links = [
        available_link(directory, path, "Structure provenance and thresholds"),
        available_link(directory, relative, "All reference-pair distances"),
    ]
    for label, name in meta.get("viewer", {}).items():
        if isinstance(name, str) and label in ("pymol", "chimerax", "index"):
            links.append(
                available_link(
                    directory,
                    sibling(path, name),
                    {
                        "pymol": "PyMOL selections",
                        "chimerax": "ChimeraX selections",
                        "index": "Viewer residue index",
                    }[label],
                )
            )
    detail += '<div class="filelinks">' + " ".join(links) + "</div>"
    for name, bm in meta.get("benchmarks", {}).items():
        target = sibling(path, bm["path"])
        if artifact(directory, target).is_file():
            detail += "<p>" + _link(target, f"Structural benchmark · {name}") + "</p>"
            if name == "all_measurable":
                detail += (
                    "<details><summary>Discrimination on all measurable structure pairs</summary>"
                    + benchmark_view(directory, {"benchmark": target})
                    + "</details>"
                )
    if store is None or not artifact(directory, relative).is_file():
        return {}, caption, detail
    bins = {}
    for axis in store.maps:
        ids, _ = axis_bins(axis)
        bins[axis.component] = {
            (axis.reference_id, p): int(i)
            for p, i in zip(axis.reference_positions, ids, strict=True)
            if p is not None
        }
    hits = {
        pair: np.zeros(tuple(len(axis_bins(store.maps["ABC".index(c)])[1]) for c in pair), bool)
        for pair in PAIRS
    }
    counts = defaultdict(Counter)
    for row in records(directory, relative):
        pair = row["pair"]
        if pair not in hits:
            continue
        key = (
            pair,
            row["x_reference_id"],
            row["x_reference_pos"],
            row["y_reference_id"],
            row["y_reference_pos"],
        )
        if network_evidence is not None and key in network_evidence:
            network_evidence[key] = {
                "structure_source_kind": kind,
                "distance_angstrom": row["distance_angstrom"],
                "structure_category": row["category"],
            }
        x, y = (
            bins[c].get((row[f"{side}_reference_id"], int(row[f"{side}_reference_pos"])))
            for c, side in zip(pair, "xy", strict=True)
        )
        if x is None or y is None:
            continue
        counter = counts[pair]
        counter["mapped"] += 1
        distance = finite(row["distance_angstrom"])
        if distance is not None:
            counter["observed"] += 1
            counter["sum"] += distance
        if row["contact"] == "1" and distance is not None:
            hits[pair][x, y] = True
            counter["close"] += 1
    detail += _table(
        [
            "Retained interface",
            "Mapped to structure table",
            "Distance measured",
            "Close",
            "Mean measured distance (Å)",
        ],
        [
            (
                p,
                counts[p]["mapped"],
                counts[p]["observed"],
                counts[p]["close"],
                val(counts[p]["sum"] / counts[p]["observed"] if counts[p]["observed"] else None),
            )
            for p in PAIRS
        ],
    )
    return hits, caption, detail


def benchmark_view(directory, outputs):
    result = saved_benchmark(directory, outputs)
    link = available_link(
        directory,
        outputs.get("benchmark"),
        "Complete benchmark with all thresholds and denominators",
    )
    parts = ["<p>" + link + "</p>"] if link else []
    if not result["methods"]:
        return (
            "".join(parts)
            + "<p>"
            + _text(result.get("reason", "No benchmark curves available."))
            + "</p>"
        )
    parts.append(
        "<p>Labels apply to measured coordinates. Scores at tied thresholds enter together. Lines connect up to 128 saved thresholds for display. AUROC and average precision use the full benchmark.</p>"
    )
    parts.append(
        _table(
            [
                "Statistic",
                "Interface",
                "Positive",
                "Negative",
                "Class support",
                "AUROC",
                "Average precision",
            ],
            [
                (
                    m["statistic"],
                    m["scope"],
                    m["positive"],
                    m["negative"],
                    m["status"] or "Unavailable",
                    val(m["auroc"]),
                    val(m["ap"]),
                )
                for m in result["methods"]
            ],
        )
    )
    for m in result["methods"]:
        title = f"{m['statistic']} · {m['scope']}"
        parts.append(f"<details><summary>{_text(title)} · curves, top-K and distances</summary>")
        plots = []
        for key, x, y, label in (
            ("roc", "False-positive rate", "True-positive rate", "ROC"),
            ("pr", "Recall", "Precision", "Precision–recall"),
        ):
            points = m[key]
            if points:
                if key == "roc" and points[0] != [0.0, 0.0]:
                    points = [[0.0, 0.0], *points]
                plots.append(
                    "<figure><h4>"
                    + label
                    + "</h4>"
                    + xy_svg(
                        points,
                        title=title + " " + label,
                        x_label=x,
                        y_label=y,
                        line=True,
                        diagonal=key == "roc",
                    )
                    + "</figure>"
                )
        parts.append('<div class="grid">' + "".join(plots) + "</div>")
        top = m["top_k"]
        if top:
            parts.append(
                _table(
                    [
                        "K",
                        "Returned",
                        "Positive",
                        "Negative",
                        "Unknown",
                        "Precision (all)",
                        "Precision among labeled",
                        "Enrichment",
                        "Median distance (Å)",
                    ],
                    [
                        (
                            k,
                            r.get("returned"),
                            r.get("positive"),
                            r.get("negative"),
                            r.get("unknown"),
                            val(r.get("precision")),
                            val(r.get("precision_among_labeled")),
                            val(r.get("enrichment")),
                            val((r.get("distance", {}).get("min_median_max") or [None] * 3)[1]),
                        )
                        for k, r in top.items()
                    ],
                )
            )
        d = m["distance"]
        mm = d.get("min_median_max")
        if mm:
            parts.append(
                "<p>Measured distances (Å), minimum / median / maximum: "
                + _text(" / ".join(val(x) for x in mm))
                + f" across {_text(d['n'])} labeled scores.</p>"
            )
        u = m["uncertainty"]
        if u:
            state = u.get("reason") or u.get("status") or "See interval estimates"
            parts.append(f"<p>Uncertainty · {_text(u.get('method', ''))} · {_text(state)}.</p>")
            for key, r in u.items():
                if isinstance(r, dict) and r.get("interval") is not None:
                    parts.append(
                        f"<p>{_text(key)} interval · {_text(' to '.join(val(x) for x in r['interval']))}.</p>"
                    )
        parts.append("</details>")
    return "".join(parts)


def support_view(directory, summary):
    outputs = summary["outputs"]
    qc = summary.get("qc_diagnostics", {})
    stats = defaultdict(Counter)
    sites = outputs.get("diagnostics", {}).get("sites")
    if sites:
        for row in records(directory, sites):
            k = (row["component"], row["retained"])
            count = stats[k]
            count["sites"] += 1
            for field in ("entropy_bits", "state_count", "coverage"):
                value = finite(row[field])
                if value is not None:
                    count[field] += value
                    count[field + "_n"] += 1
    parts = []
    if stats:
        parts.append(
            "<h3>Site entropy and support</h3>"
            + _table(
                [
                    "Component",
                    "Retained",
                    "Sites",
                    "Mean entropy (bits)",
                    "Mean canonical states",
                    "Mean canonical coverage",
                ],
                [
                    (
                        c,
                        kept,
                        s["sites"],
                        *(
                            val(s[f] / s[f + "_n"] if s[f + "_n"] else None)
                            for f in ("entropy_bits", "state_count", "coverage")
                        ),
                    )
                    for (c, kept), s in sorted(stats.items())
                ],
            )
        )
    if dup := qc.get("exact_encoded_duplicates"):
        parts.append(
            "<h3>Sequence redundancy</h3>"
            + _table(
                ["Component", "Distinct encoded rows", "Duplicate rows", "Largest group"],
                [
                    (
                        c,
                        r.get("groups", "—"),
                        r.get("duplicate_rows", "—"),
                        r.get("largest_group", "—"),
                    )
                    for c, r in dup.items()
                ],
            )
        )
    if meta := qc.get("metadata"):
        parts.append(
            _table(
                [
                    "Metadata",
                    "Coverage status",
                    "Source column",
                    "Input groups",
                    "Retained groups",
                    "Retained without label",
                ],
                [
                    (
                        k,
                        m.get("status", "unavailable"),
                        m.get("column") or "—",
                        m.get("input", {}).get("distinct", "—"),
                        m.get("retained", {}).get("distinct", "—"),
                        m.get("retained", {}).get("missing", "—"),
                    )
                    for k, m in meta.items()
                ],
            )
        )
    adv = summary.get("advanced", {})
    if c := adv.get("candidates"):
        parts.append(
            f"<h3>Triple candidates</h3><p>{_text(c.get('selection', 'unspecified'))} selection · {_text(c.get('tested', 0))} tested of {_text(c.get('eligible', 'unknown'))} eligible ABC coordinates · {_text(c.get('finite', 'unknown'))} finite primary scores.</p>"
        )
        if source := adv.get("candidate_manifest"):
            m = json.loads(artifact(directory, source).read_text(encoding="utf-8"))
            parts.append(
                f"<p>{'Exhaustive' if m.get('exhaustive') else 'Sampled or restricted'} candidate set · {_text(m.get('scope', 'unspecified'))}. {_text(m.get('interpretation', ''))} {_link(source, 'Selection and screening provenance')}.</p>"
            )
    fit = (adv.get("local_fit") or {}).get("diagnostic_table")
    if fit:
        groups = defaultdict(Counter)
        for row in records(directory, fit):
            g = groups[(row["setting"], row["prior_mode"], row["prior_mass"])]
            g["tested"] += 1
            g["ok"] += row["status"] == "ok"
            ratio = finite(row.get("possible_cells"))
            obs = finite(row.get("observed_cells"))
            values = [
                ("occupancy", obs / ratio if obs is not None and ratio else None),
                *(
                    (f, finite(row.get(f)))
                    for f in ("prior_fraction", "co_information_bits", "connected_information_bits")
                ),
            ]
            for f, v in values:
                if v is not None:
                    g[f] += v
                    g[f + "_n"] += 1
        parts.append(
            "<h3>Prior sensitivity and occupied states</h3><p>Each mean uses candidates with a finite value for that measure. Signed co-information can be negative. Connected information compares the joint target with its pairwise maximum-entropy fit.</p>"
            + _table(
                [
                    "Setting",
                    "Prior mode",
                    "Prior mass",
                    "Fits",
                    "Successful",
                    "Mean occupied fraction",
                    "Mean prior fraction",
                    "Mean signed co-info (bits)",
                    "Mean connected information (bits)",
                ],
                [
                    (
                        *key,
                        s["tested"],
                        s["ok"],
                        *(
                            val(s[f] / s[f + "_n"] if s[f + "_n"] else None)
                            for f in (
                                "occupancy",
                                "prior_fraction",
                                "co_information_bits",
                                "connected_information_bits",
                            )
                        ),
                    )
                    for key, s in groups.items()
                ],
            )
        )
    for key, out, label in (
        ("calibration", "calibration_report", "Null calibration and uncertainty"),
        ("clade_stability", "stability_report", "Held-out clade stability"),
        ("conditional", "conditional_report", "Nested held-out prediction"),
    ):
        title = available_link(directory, outputs.get(out), label)
        if title:
            parts.append("<h3>" + title + "</h3>")
            s = summary.get(key, {})
            if key == "conditional" and s.get("grid"):
                parts.append(
                    _table(
                        ["L2", "Evaluated folds", "Clade mean gain (nats/site)"],
                        [
                            (
                                r["l2"],
                                r["evaluated_folds"],
                                val(r["clade_macro_gain_nats_per_site"]),
                            )
                            for r in s["grid"]
                        ],
                    )
                )
            elif key == "calibration":
                parts.append(
                    f"<p>{_text(s.get('null', ''))} · {_text(s.get('status', ''))} · {_text(s.get('draws', 'unknown'))} draws. The calibration report lists the tail and correction family for each hypothesis.</p>"
                )
            elif key == "clade_stability":
                parts.append(
                    f"<p>{_text(s.get('groups', 'unknown'))} groups · {_text(s.get('folds', 'unknown'))} training/held-out splits.</p>"
                )
    if withheld := adv.get("triple_calibration"):
        parts.append(
            "<p>Triple calibration · "
            + _text(withheld.get("status", "unavailable"))
            + " · "
            + _text(withheld.get("reason", ""))
            + "</p>"
        )
    if model := adv.get("dca_diagnostics"):
        parts.append(
            "<p>Coupling fit · "
            + _text(model.get("status", "See model diagnostics"))
            + ". The analysis manifest records the estimator, solver and failure diagnostics.</p>"
        )
    if not parts:
        parts.append(
            "<p>The supplied artifact has no accompanying observation or candidate diagnostics.</p>"
        )
    return "".join(parts)
