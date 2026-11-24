"""Generate an offline explorer from verified score arrays and saved scientific outputs."""

import csv
import json
from importlib.resources import files
from itertools import islice
from pathlib import Path

from triplet_coevolution.inference.scores import get_score
from triplet_coevolution.io.score_store import PAYLOAD_NAME, load_score_store
from triplet_coevolution.reporting.explore_data import (
    PAIRS,
    artifact,
    common_ranks,
    score_grid,
    write_label_scores,
)
from triplet_coevolution.reporting.explore_evidence import (
    available_link,
    benchmark_view,
    sibling,
    structure_view,
    support_view,
)
from triplet_coevolution.reporting.explore_figures import (
    PALETTES,
    colors,
    export_maps,
    grid_svg,
    xy_svg,
)
from triplet_coevolution.reporting.html import _link, _table, _text
from triplet_coevolution.reporting.network import write_network

TITLES = {
    "mi_bits": "Raw mutual information",
    "mi_apc": "APC-corrected mutual information",
    "dca_frobenius": "Raw coupling norm",
    "dca_apc": "APC-corrected coupling norm",
}
PREVIEW_ROWS = 40
PREFERRED = (
    "rank",
    "competition_rank",
    "pair",
    "statistic",
    "direction",
    "score",
    "x_reference_id",
    "x_reference_pos",
    "x_original_col",
    "y_reference_id",
    "y_reference_pos",
    "y_original_col",
    "a_original_col",
    "b_original_col",
    "c_original_col",
    "mi_bits",
    "mi_apc",
    "dca_frobenius",
    "dca_apc",
    "threebody_rms",
    "co_information_bits",
    "connected_information_bits",
    "n_valid",
    "valid_weight",
    "contact",
    "category",
    "observable",
    "distance_angstrom",
    "source",
    "status",
    "setting",
    "prior_mass",
    "prior_fraction",
    "observed_cells",
    "possible_cells",
    "permutation_p",
    "bh_q",
    "null_draws",
    "component",
    "source_column",
    "reference_position",
    "retained",
    "entropy_bits",
    "state_count",
    "coverage",
    "mask_reasons",
)


def _sources(directory, summary, tables):
    outputs = summary["outputs"]
    paths = {
        t.path: f"{t.label} · {t.method} ranked by {t.score} ({t.rows:,} exported)" for t in tables
    }
    for key, meta in summary.get("advanced", {}).get("outputs", {}).items():
        p = meta.get("path")
        if p and p.endswith((".csv", ".tsv")):
            paths.setdefault(p, key.replace("_", " "))
    for k, p in outputs.get("diagnostics", {}).items():
        if p.endswith((".csv", ".tsv")):
            paths.setdefault(p, "QC · " + k.replace("_", " "))
    if p := outputs.get("weights"):
        paths[p] = "Observation weights"
    for key, tabs in (
        ("structure", ("distances.tsv", "labels.tsv", "mapping.tsv")),
        ("calibration", ("calibration.tsv", "tree_sensitivity.tsv")),
        ("stability", ("stability_folds.tsv", "stability_coordinates.tsv")),
    ):
        if p := outputs.get(key):
            for name in tabs:
                relative = sibling(p, name)
                if artifact(directory, relative).is_file():
                    paths[relative] = key.capitalize() + " · " + name
    return paths


def _table_views(directory, sources):
    data = []
    details = []
    for path, title in sources.items():
        source = artifact(directory, path)
        with source.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="," if source.suffix.lower() == ".csv" else "\t")
            cols = reader.fieldnames or []
            keep = list(dict.fromkeys([*(c for c in PREFERRED if c in cols), *cols]))[:30]
            rows = [
                [(row.get(c) or "")[:200] for c in keep] for row in islice(reader, PREVIEW_ROWS)
            ]
        data.append({"title": title, "fields": keep, "rows": rows})
        details.append(
            "<details><summary>"
            + _text(title)
            + "</summary><p>"
            + _link(path, "Download the complete exported table")
            + "</p>"
            + _table(keep[:12], [r[:12] for r in rows[:4]])
            + "</details>"
        )
    return data, "".join(details)


def _maps(directory, store, palette, formats, overlays, threshold):
    if store is None:
        return (
            "<p>The full matrices are unavailable in this run. Ranked exports appear under Data tables.</p>",
            "<p>Common-axis comparisons require full MI and DCA matrices.</p>",
            [],
            [],
        )
    maps = []
    rank = []
    exports = []
    names = []
    for name, title in TITLES.items():
        if name not in store.scores:
            continue
        names.append(name)
        metadata = get_score(name).metadata(
            apc_universe=store.provenance.get("apc_universe", "retained"),
            dca_design=store.provenance.get("methods", {}).get("dca_design"),
        )
        units = metadata["units"]
        signed = name.endswith("apc")
        grids = {p: score_grid(store, p, name) for p in PAIRS}
        maximum = max(g["max_abs"] for g in grids.values())
        maps.append(
            f'<article data-score-map="{name}"><h3>{_text(title)}</h3><p><code>{name}</code> · {_text(units)} · {_text(metadata["direction"])} scores rank first · {_text(metadata["universe"])}.</p><div class="grid">'
        )
        for p, g in grids.items():
            svg = grid_svg(
                g,
                title=f"{title} · {p}",
                units=units,
                maximum=maximum,
                signed=signed,
                palette=palette,
                identifier=name + "-" + p,
                overlay=overlays.get(p),
            )
            lo, hi = g["support"]
            support = "No eligible values" if lo is None else f"Joint support {lo:,}–{hi:,} rows"
            maps.append(
                f"<figure><h4>{p} interface</h4>{svg}<figcaption>Rows {p[0]}, columns {p[1]} · {g['finite']:,}/{g['total']:,} finite and eligible cells. {support}.</figcaption></figure>"
            )
        maps.append("</div></article>")
        exports += export_maps(
            directory,
            name,
            grids,
            maximum=maximum,
            units=units,
            signed=signed,
            palette=palette,
            formats=formats,
            overlays=overlays,
            threshold=threshold,
        )
    for mi, dc, label in (
        ("mi_bits", "dca_frobenius", "Raw"),
        ("mi_apc", "dca_apc", "APC-corrected"),
    ):
        if not {mi, dc} <= store.scores.keys():
            continue
        rank.append(f'<h3>{label} scores</h3><div class="grid">')
        for p in PAIRS:
            r = common_ranks(store, p, mi, dc)
            xy = [(x, y) for _, _, x, y in r["points"]]
            rank.append(f"<figure><h4>{p} interface</h4>")
            if xy:
                rank.append(
                    xy_svg(
                        xy,
                        title=f"{p} {label} MI vs DCA competition ranks",
                        x_label="MI rank",
                        y_label="DCA rank",
                        xmax=max(2, r["common"]),
                        ymax=max(2, r["common"]),
                        origin=1,
                        diagonal=True,
                    )
                )
            rank.append(
                f"<figcaption>{len(xy):,} displayed of {r['common']:,} common, finite coordinate pairs. Coincident ranks overlap.</figcaption></figure>"
            )
        rank.append("</div>")
    return (
        "".join(maps),
        "".join(rank) or "<p>This run has no joint MI and DCA comparison.</p>",
        exports,
        names,
    )


def _asset(name):
    return files("triplet_coevolution.reporting").joinpath(name).read_text(encoding="utf-8")


def write_explorer(
    directory, summary, tables, *, store=None, palette="accessible", figure_formats=()
):
    """Write a file://-ready bounded explorer. Scientific thresholds come from the saved run."""
    directory = Path(directory)
    outputs = summary["outputs"]
    if palette not in PALETTES:
        raise ValueError(f"unknown report palette {palette!r}")
    formats = tuple(dict.fromkeys(figure_formats))
    if any(x not in ("svg", "pdf", "png") for x in formats):
        raise ValueError("report figure formats are svg, pdf and png")
    if store is None and (p := outputs.get("score_manifest")):
        store = load_score_store(artifact(directory, p))
    evidence = {}
    limit = outputs.get("network", {}).get("limit_per_table", 25)
    if outputs.get("structure"):
        for t in tables:
            if len(t.prefixes) == 2:
                for row in t.first_rows(directory, limit):
                    evidence[
                        t.label,
                        row["x_reference_id"],
                        row["x_reference_pos"],
                        row["y_reference_id"],
                        row["y_reference_pos"],
                    ] = {}
    overlay, threshold, structure = structure_view(
        directory, outputs, store, network_evidence=evidence
    )
    if outputs.get("structure"):
        annotated = write_network(directory, tables, limit, structure_evidence=evidence)
        structure += (
            "<p>"
            + _link(annotated["path"], "Download structure-annotated Cytoscape GraphML")
            + ". Pair edges carry statistical scores and available physical distances. Edges to a three-site factor indicate membership.</p>"
        )
    maps, ranks, exports, names = _maps(directory, store, palette, formats, overlay, threshold)
    sources = _sources(directory, summary, tables)
    if store is not None and (labels := outputs.get("benchmark_labels")):
        labeled = write_label_scores(directory, labels, store)
        sources = {labeled: "Supplied labels with supported scores", **sources}
    tables_data, downloads = _table_views(directory, sources)
    data = json.dumps(
        {
            "tables": tables_data,
            "colors": {p: {"seq": colors(p, False), "div": colors(p, True)} for p in PALETTES},
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    palette_menu = "".join(
        f'<option value="{p}"' + (" selected" if p == palette else "") + f">{label}</option>"
        for p, label in (("accessible", "Teal / blue–orange"), ("amber-blue", "Amber / teal–rose"))
    )
    map_menu = "".join(f'<option value="{p}">{_text(TITLES[p])}</option>' for p in names)
    overlay_key = (
        f'<p><span class="legend-dot"></span> Ring = at least one reference pair in the bin with {_text(threshold)}. Unmarked bins may include unresolved coordinates.</p>'
        if overlay
        else ""
    )
    files_out = [_link(p, p) for p in exports]
    if p := outputs.get("score_manifest"):
        files_out.extend(
            [
                _link(p, "Score manifest"),
                available_link(
                    directory, sibling(p, PAYLOAD_NAME), "Complete pair matrices and support"
                ),
            ]
        )
    ranked = " · ".join(
        f"{t.label} {t.method}, {t.score} ({get_score(t.score).direction} first)" for t in tables
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Score explorer · Triplet coevolution</title><style>{_asset("explorer.css")}</style></head>
<body><a class="skip" href="#maps">Skip to score maps</a><header><a href="report.html">Analysis report</a><h1>Score explorer</h1><p>Pairwise evidence across the three interfaces, with support and calibration from the saved analysis.</p></header>
<nav aria-label="Explorer sections"><a href="#maps">Score maps</a><a href="#ranks">MI versus DCA</a><a href="#geometry">Structure</a><a href="#benchmark">Label benchmarks</a><a href="#support">Model support</a><a href="#tables">Data tables</a></nav><main>
<section id="maps"><div class="toolbar"><h2>Score maps</h2><div class="controls js-only"><label>Statistic<select id="map-selection"><option value="">All statistics</option>{map_menu}</select></label><label>Palette<select id="palette">{palette_menu}</select></label></div></div>
<p>Axes use original alignment columns. Each bin shows the mean of eligible finite scores in at most 28 × 28 bins per interface. Hatching marks bins with no eligible value. Each statistic shares one scale across AB, AC and BC. APC colors are centered at zero.</p>{overlay_key}{maps}<div class="filelinks">{" ".join(files_out)}</div></section>
<section id="ranks"><h2>MI versus DCA ranks</h2><p>Both methods rank exactly the same coordinates within each interface. Rank 1 is strongest. Ties share competition rank. Up to 192 coordinates are spaced through row-major order for display.</p>{ranks}</section>
<section id="geometry"><h2>Structural distances</h2>{structure}</section>
<section id="benchmark"><h2>Label benchmarks</h2>{benchmark_view(directory, outputs)}</section>
<section id="support"><h2>Model support and validation</h2>{support_view(directory, summary)}</section>
<section id="tables"><h2>Data tables</h2><p>Ranked exports use the recorded sort statistic and export limit. {_text(ranked)}</p>
<p>Previews use the first {PREVIEW_ROWS} exported rows, up to 30 columns and 200 characters per cell. Open a local CSV or TSV to filter the complete file. File contents stay in this browser. The reader counts every match and displays up to 100 rows, 100 fields and 400 characters per cell.</p>
<div class="controls js-only"><label>Saved preview<select id="source"></select></label><label>Open full local table<input id="local-file" type="file" accept=".tsv,.csv,.txt"></label></div>
<div class="controls js-only"><label>Search text<input id="search" type="search"></label><label>Exact label or status<input id="label-filter" type="text"></label><label>Numeric field<select id="field"></select></label><label>Minimum<input id="minimum" type="number" step="any"></label><label>Maximum<input id="maximum" type="number" step="any"></label><button id="apply" type="button">Apply filters</button></div>
<p id="table-status" aria-live="polite"></p><div class="scroll" tabindex="0" aria-label="Filtered records"><table class="js-only"><thead><tr id="head"></tr></thead><tbody id="rows"></tbody></table></div>
<h3>Exported tables and static previews</h3>{downloads or "<p>No tabular results were included.</p>"}</section></main>
<footer><a href="report.html">Analysis report and provenance</a></footer><script id="explore-data" type="application/json">{data}</script><script>{_asset("explorer.js")}</script></body></html>"""
    artifact(directory, "explore.html").write_text(page, encoding="utf-8")
    return "explore.html"
