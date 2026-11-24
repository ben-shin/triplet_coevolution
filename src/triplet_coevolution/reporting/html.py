"""Generate a self-contained local HTML report for a completed analysis."""

import base64
import csv
import json
from html import escape
from itertools import groupby
from pathlib import Path
from urllib.parse import quote

from triplet_coevolution.inference.scores import get_score
from triplet_coevolution.reporting.network import RankedTable, ranked_tables

STYLE = """
:root{color-scheme:light;--ink:#182e39;--muted:#50646e;--teal:#086859;--rule:#cdd9d9}
*{box-sizing:border-box}body{margin:0;background:#f7f8f5;color:var(--ink);font:16px/1.52
 'Source Sans 3','Segoe UI',sans-serif}header{background:#173d45;color:white;padding:2.8rem max(5vw,
 calc((100vw - 1080px)/2)) 2.2rem}h1{font:600 clamp(2rem,5vw,3.5rem)/1.1 Georgia,serif;
 margin:0}header p{max-width:68ch;margin:.8rem 0 0;color:#deebe9}nav,main{max-width:1120px;
 margin:auto;padding:0 20px}nav{display:flex;gap:1.3rem;flex-wrap:wrap;padding-top:1rem;
 padding-bottom:1rem;border-bottom:1px solid var(--rule)}a{color:var(--teal);text-underline-offset:
 .18em}nav a{font-weight:600}a:focus-visible,summary:focus-visible{outline:3px solid #c36a12;
 outline-offset:3px}main{padding-bottom:3rem}section{margin:2.4rem 0}h2{font:600 1.85rem Georgia,
 serif;margin:0 0 .6rem}h3{margin:1.1rem 0 .35rem;font-size:1.12rem}p{max-width:84ch;margin:.5rem
 0 1rem}.metrics{display:flex;gap:0;flex-wrap:wrap;background:white;border:1px solid var(--rule);
 border-radius:5px}.metrics div{flex:1;min-width:150px;padding:1rem 1.25rem}.metrics dt{color:
 var(--muted);font-size:.9rem}.metrics dd{margin:0;font-weight:650;font-size:1.65rem}.scroll{
 overflow-x:auto}.scroll table{border-collapse:collapse;width:100%;background:white;margin:.45rem
 0 1rem;font-size:.92rem}.scroll th,.scroll td{text-align:left;white-space:nowrap;padding:.48rem
 .72rem;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}.scroll th{font-weight:
 650;background:#eaf0ee}.small,small{color:var(--muted)}.panel{border-top:1px solid var(--rule);
 padding:.5rem 0}summary{color:var(--teal);cursor:pointer;font-weight:600}.files{padding-left:1.25rem;
 columns:2;column-width:16rem}.files li{break-inside:avoid;margin:.2rem 0}img{width:100%;height:auto;
 border:1px solid var(--rule);background:white}.composition{max-width:640px}.trackrow{display:flex;
 align-items:center;gap:.7rem;margin:.35rem 0}.track{display:flex;flex:1;height:13px;background:
 #e1e6e5}.canonical{background:#137e71}.gap{background:#d59a41}.missing{background:#7c729b}
 .legend{font-size:.87rem;display:flex;gap:1rem;flex-wrap:wrap}.legend i{display:inline-block;width:
 .72rem;height:.72rem;margin-right:.33rem}.audit-links{display:flex;gap:.35rem 1rem;
 flex-wrap:wrap;max-width:none}.scroll .triple th{white-space:normal}.scroll .triple
 td:nth-child(-n+3){white-space:normal;min-width:7.2em}code{font-family:'IBM Plex Mono',Consolas,monospace;
 font-size:.9em;overflow-wrap:anywhere}footer{border-top:1px solid var(--rule);padding-top:1rem;
 color:var(--muted)}@media print{nav,details:not([open]){display:none}body{background:white}
 header{background:none;color:black;padding:1rem}header p{color:black}.scroll{overflow:visible}}
"""


def _text(value) -> str:
    return escape(str(value), quote=True)


def _link(path: str, label: str) -> str:
    return f'<a href="{_text(quote(path, safe="/"))}">{_text(label)}</a>'


def _table(headers, rows, *, triple=False) -> str:
    head = "".join(f'<th scope="col">{_text(label)}</th>' for label in headers)
    body = "".join("<tr>" + "".join(f"<td>{_text(v)}</td>" for v in row) + "</tr>" for row in rows)
    attr = ' class="triple"' if triple else ""
    return f'<div class="scroll"><table{attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _site(row, prefix):
    position = row[f"{prefix}_reference_pos"]
    col = row[f"{prefix}_original_col"]
    return f'{row[f"{prefix}_component"]} {row[f"{prefix}_reference_id"]} · ' + (
        f"{position} (col {col})" if position else f"alignment col {col}"
    )


METHODS = {
    "MI": "Mutual information captures statistical dependence. APC adjusts each component pair for background signal.",
    "DCA": "DCA ranks fitted couplings on the declared conditioning panel. The score definition records the estimator and regularization.",
    "Local maximum entropy": "Connected information measures departure from the fitted pairwise marginal model in bits. Signed co-information and the projected log-linear RMS measure different properties of the same three sites.",
    "Three-body Potts": "Each tested candidate joins one site from each component. Local and conditional models have different interpretations.",
}


def _triple_evidence(directory: Path, triples: list[dict], pair_tables: dict, second_score: str):
    """Join the leading triples to the actual pairwise exports by original MSA columns."""
    evidence = [
        [f"{float(row[second_score]):.5g}" if row.get(second_score) else "Undefined"]
        for row in triples
    ]
    for pair in ("AB", "AC", "BC"):
        keys = [(row[f"{pair[0].lower()}_original_col"], row[f"{pair[1].lower()}_original_col"])
                for row in triples]
        scores = dict.fromkeys(keys, "Outside MI export")
        path = directory / pair_tables[pair].path
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="," if path.suffix == ".csv" else "\t"):
                key = (row["x_original_col"], row["y_original_col"])
                if key in scores:
                    scores[key] = f'{float(row["mi_bits"]):.5g}'
        for result, key in zip(evidence, keys):
            result.append(scores[key])
    return evidence


def _rankings(directory: Path, tables: list[RankedTable]) -> str:
    pair_tables = {t.label: t for t in tables if t.method == "MI"}
    pieces = []
    for method, group in groupby(tables, key=lambda table: table.method):
        pieces.append(f'<div class="panel"><h3>{_text(method)}</h3><p>{METHODS[method]}</p>')
        for table in group:
            top = table.first_rows(directory, 5)
            spec = get_score(table.score)
            title = f"{table.label} · {table.score} · {spec.direction} first · {table.rows:,} exported"
            pieces.append(f"<details open><summary>{_text(title)}</summary>")
            if top:
                headers = [*list(table.label), table.score, "Valid observations"]
                rows = [[*[_site(row, k) for k in table.prefixes],
                         f"{float(row[table.score]):.5g}", row["n_valid"]] for row in top]
                if table.label == "ABC":
                    second = "threebody_rms" if table.score == "co_information_bits" else "co_information_bits"
                    heading = "Interaction RMS" if second == "threebody_rms" else "Co-information (bits)"
                    headers += [heading, "AB MI (bits)", "AC MI (bits)", "BC MI (bits)"]
                    for row, extra in zip(rows, _triple_evidence(directory, top, pair_tables, second)):
                        row.extend(extra)
                pieces.append(_table(headers, rows, triple=table.label == "ABC"))
            else:
                pieces.append("<p>No site combinations passed the selected filters.</p>")
            pieces.append(_link(table.path, f"Open exported {method} {table.label} table") + "</details>")
        pieces.append("</div>")
    return "".join(pieces)


def _definition_rows(data, prefix=""):
    """Flatten JSON objects into readable label–value rows, preserving unavailable values."""
    for key, value in data.items():
        label = (prefix + " / " if prefix else "") + key.replace("_", " ")
        if isinstance(value, dict):
            yield from _definition_rows(value, label)
        elif isinstance(value, list):
            yield label, ", ".join(str(v) for v in value)
        elif key == "path" and value is not None:
            yield label, Path(value).name
        else:
            yield label, "Unavailable" if value is None else value


def _benchmarks(directory: Path, advanced: dict) -> str:
    chunks = []
    for stem, info in advanced.get("outputs", {}).items():
        path = info["path"]
        if stem.startswith("permutation_"):
            delimiter = "," if path.endswith(".csv") else "\t"
            tested = discoveries = 0
            lowest = None
            with (directory / path).open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter=delimiter):
                    if row.get("bh_q") not in (None, ""):
                        value = float(row["bh_q"])
                        tested += 1
                        discoveries += int(value <= 0.05)
                        lowest = value if lowest is None else min(lowest, value)
            minimum = "unavailable" if lowest is None else f"{lowest:.4g}"
            chunks.append(
                f'<h3>{_text(stem.removeprefix("permutation_").upper())} permutation results</h3>'
                f"<p>{tested:,} adjusted results. {discoveries:,} at BH q ≤ 0.05. "
                f"Smallest q {_text(minimum)}. {_link(path, 'Open permutation results')}.</p>"
            )
        elif path == "contacts.json":
            contact_data = json.loads((directory / path).read_text(encoding="utf-8"))
            rows = []
            for method, counts in contact_data["methods"].items():
                metrics = ["Unavailable" if counts[k] is None else f"{counts[k]:.3f}"
                           for k in ("auroc", "average_precision")]
                rows.append((method.upper(), f'{counts["scored"]} / {counts["labels"]}',
                             counts["positives_scored"], counts["negatives_scored"], *metrics))
            chunks.append("<h3>Reference contact labels</h3><p>Performance is evaluated on supplied "
                          "labels with mapped, scored coordinates.</p>")
            chunks.append(_table(["Method", "Scored / labeled", "Positive", "Negative", "AUROC",
                                  "Average precision"], rows))
    if chunks:
        return "".join(chunks)
    return "<p>No benchmarks were requested. Interpret the site rankings as descriptive associations.</p>"


def benchmark_overview(result: dict) -> list[dict]:
    """Retain compact display metrics while the complete curves stay in benchmark.json."""
    rows = []
    for name, methods in result["statistics"].items():
        groups = {**methods["interfaces"], "Pooled": methods["pooled"], "Macro": methods["macro"]}
        for group, metrics in groups.items():
            rows.append({
                "statistic": name, "scope": group, "auroc": metrics["auroc"],
                "average_precision": metrics["average_precision"],
                "common_labeled": metrics.get("coverage", {}).get("common_labeled"),
            })
    return rows


def benchmark_section(rows: list[dict]) -> str:
    """Describe common-label comparisons with their actual interface denominators."""
    if not rows:
        return ""
    displayed = (
        (r["statistic"], r["scope"], r["common_labeled"] if r["common_labeled"] is not None else "—",
         *(f"{r[k]:.5g}" if r[k] is not None else "Unavailable" for k in ("auroc", "average_precision")))
        for r in rows
    )
    return ("<h3>Contact prediction on common labeled pairs</h3>"
            "<p>Macro values average interfaces with defined metrics. Unknown labels remain outside "
            "the classification denominator. The full benchmark includes coverage, top-K, curves "
            "and the status of uncertainty estimates.</p>"
            + _table(["Statistic", "Interface", "Common labels", "AUROC", "Average precision"], displayed))


def _refresh_report_hash(directory: Path, summary: dict) -> None:
    if "artifact_sha256" in summary:
        from triplet_coevolution.io.score_store import file_digest
        summary["artifact_sha256"]["report.html"] = file_digest(directory / "report.html")
        (directory / "run_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )


def _workflow_section(workflow: dict, matching_qc: dict) -> str:
    searches = _table(["Component", "Search method", "Molecule", "Reference", "Retrieved hits"], (
        (name, component["backend"], component["molecule"], component["reference_id"], component["hits"])
        for name, component in workflow["search"].items()
    ))
    excluded = [(row["group_id"], "; ".join(row["reasons"]))
                for row in matching_qc["groups"] if not row["matched"]]
    skipped = _table(["Excluded group", "Reason"], excluded[:10]) if excluded else ""
    paths = [("../workflow_summary.json", "Workflow manifest"),
             ("../" + workflow["matching_qc"], "Full matching audit"),
             ("../" + workflow["prepared"] + "/matches.tsv", "Prepared match table")]
    if workflow.get("analyze_args"):
        paths.append(("../" + workflow["analyze_args"], "Replay analysis arguments"))
    paths.extend(("../" + data["table"], f"{key} search hits")
                 for key, data in workflow["search"].items())
    links = "".join(_link(path, label) for path, label in paths)
    return (f'<section id="workflow"><h2>Search and matching</h2>{searches}'
            f'<p>{workflow["matched_groups"]:,} groups supplied complete matches. '
            f'{workflow["excluded_groups"]:,} groups were excluded under the '
            f'<strong>{_text(workflow["paralogs"])}</strong> policy.</p>{skipped}'
            f'<p class="audit-links">{links}</p></section>')


def attach_workflow_report(directory: Path, workflow_summary: dict, matching_qc: dict) -> None:
    """Include the search and matching audit in the report published by the full workflow."""
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    write_report(directory, summary, ranked_tables(summary),
                 workflow_section=_workflow_section(workflow_summary, matching_qc))
    _refresh_report_hash(directory, summary)


def row_support_section(workflow, prefix=""):
    support = workflow.get("observation_qc")
    if not support:
        return ""
    rows = ((r["observation_id"], r["components"], r["pass"]) for r in support["examples"])
    table = _table(["Excluded observation", "Components with no usable residue", "QC pass"], rows)
    links = " · ".join((
        _link(prefix + support["table"], "All row exclusions"),
        _link(prefix + support["matches"], "Eligible observations"),
        _link(prefix + support["columns"], "Final original columns"),
    ))
    return (
        '<section><h2>Observation quality control</h2>'
        f'<p>{support["eligible_observations"]:,} of {support["input_observations"]:,} candidate '
        f'observations retained a canonical residue in every component. '
        f'{support["excluded_observations"]:,} were excluded. The table shows up to ten exclusions.</p>'
        + table + f'<p>{links}</p></section>'
    )


def attach_evidence_report(directory: Path, workflow: dict) -> None:
    """Connect a scored Stockholm subset to its original row, group and coordinate evidence."""
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    table = _table(["Component", "Folder or name", "Source rows", "Exported rows", "Reference"], (
        (key, comp["name"], comp["source_rows"], comp["exported_rows"], comp["reference_id"])
        for key, comp in workflow["components"].items()
    ))
    links = (("evidence", "Candidate evidence"), ("group_audit", "Group decisions"),
             ("reference_coordinates", "Reference coordinates"), ("matches", "Matched observations"),
             ("analyze_args", "Replay analysis"))
    panel_text = ""
    status_text = ""
    if workflow.get("status"):
        status_text = (f'<p>Workflow status: {_text(workflow["status"])}. '
                       f'{_text(workflow["reason"])} '
                       + _link("../report.html", "Workflow status and evidence") + '</p>')
    if (panel := workflow.get("scoring_panel", {})).get("status") == "selected":
        widths = ", ".join(f"{key} {width}" for key, width in panel["selected_widths"].items())
        panel_text = (f'<p>This bounded panel scored {summary["matched_observations"]:,} observations. '
                      f'Selected site counts before quality filtering: {_text(widths)}. '
                      + _link("../report.html", "Full matching audit and panel selection") + '</p>')
    section = (
        '<section id="workflow"><h2>Stockholm families and matching evidence</h2>' + table
        + f'<p>{workflow["counts"]["resolved_groups"]:,} groups supplied complete matches.</p>'
        + panel_text
        + status_text
        + '<p class="audit-links">' + "".join(_link("../" + workflow[key], label) for key, label in links)
        + _link("../workflow_summary.json", "Workflow manifest") + '</p></section>'
    )
    section += row_support_section(workflow, "../")
    write_report(directory, summary, ranked_tables(summary), workflow_section=section)
    _refresh_report_hash(directory, summary)


def write_report(directory: Path, summary: dict, tables: list[RankedTable], *,
                 workflow_section: str = "") -> None:
    components = summary["components"]
    outputs = summary["outputs"]
    advanced = summary.get("advanced", {})
    metrics = [
        ("Matched observations", f'{summary["matched_observations"]:,}'),
        ("Effective depth", f'{summary["effective_depth"]:.2f}'),
        ("Retained columns", " / ".join(str(c["retained_sites"]) for c in components.values())),
    ]
    support = summary.get("weighting")
    if support is not None:
        metrics.append(("Kish effective count", f"{support['kish_effective_count']:.2f}"))
    metrics_html = "".join(f"<div><dt>{k}</dt><dd>{_text(v)}</dd></div>" for k, v in metrics)
    qc = _table(["Component", "Type", "Reference", "Input sequences", "Retained / original",
                 "Canonical", "Gaps", "Missing"], (
        (name, c["molecule_type"], c["reference_id"], c["input_sequences"],
         f'{c["retained_sites"]} / {c["original_sites"]}', f'{c["mean_coverage"]:.1%}',
         f'{c["mean_gap_fraction"]:.1%}', f'{c["mean_missing_fraction"]:.1%}')
        for name, c in components.items()
    ))
    classes = (("canonical", "mean_coverage"), ("gap", "mean_gap_fraction"),
               ("missing", "mean_missing_fraction"))
    strips = []
    for name, component in components.items():
        bars = "".join(
            f'<span class="{css}" style="width:{100 * component[field]:.3f}%"></span>'
            for css, field in classes
        )
        strips.append(f'<div class="trackrow"><b>{_text(name)}</b><div class="track">{bars}</div></div>')
    legend = "".join(f'<span><i class="{css}"></i>{label}</span>'
                     for css, label in (("canonical", "Canonical"), ("gap", "Gap"),
                                        ("missing", "Ambiguous or missing")))
    files = [t.path for t in tables]
    files.extend(info["path"] for info in advanced.get("outputs", {}).values())
    files.extend([outputs["network"]["path"], "run_summary.json", *outputs["figures"].values()])
    files.extend(
        outputs[key]
        for key in (
            "score_manifest",
            "score_payload",
            "benchmark",
            "calibration",
            "calibration_report",
            "stability",
            "stability_report",
            "conditional",
            "conditional_report",
        )
        if key in outputs
    )
    files.extend(outputs[k] for k in ("weights", "plan", "replay") if k in outputs)
    files.extend(outputs.get("diagnostics", {}).values())
    links = "".join(f"<li>{_link(p, p)}</li>" for p in dict.fromkeys(files))
    audit_block = ""
    if support and "qc_diagnostics" in summary:
        audit = summary["qc_diagnostics"]
        audit_block = (
            f"<p>QC retained {audit['retained_observations']:,} of {audit['input_observations']:,} matched observations. "
            f"Weighting policy <strong>{_text(support['policy'])}</strong>. "
            f"Sum of weights per retained site {support['weight_per_retained_site']:.3g}. "
            f"Valid-pair thresholds are {summary['settings']['min_valid']} observations and {support['min_valid_weight']:g} weight. "
            "The sum of weights and Kish count describe different aspects of redundancy. Neither establishes independent evolutionary sample size.</p>"
            + "<p>"
            + _link(outputs["diagnostics"]["rows"], "Observation audit")
            + " · "
            + _link(outputs["diagnostics"]["sites"], "Site audit")
            + " · "
            + _link(outputs["weights"], "Weights and neighbor counts")
            + " · "
            + _link(outputs["plan"], "Resource estimates")
            + "</p>"
        )
    picture = ""
    if combined := outputs["figures"].get("triplet"):
        payload = base64.b64encode((directory / combined).read_bytes()).decode("ascii")
        picture = ('<h3>Mutual information across component pairs</h3>'
                   f'<img alt="AB, AC and BC score heatmaps with a shared scale" '
                   f'src="data:image/png;base64,{payload}">')
    if combined := outputs["figures"].get("dca"):
        payload = base64.b64encode((directory / combined).read_bytes()).decode("ascii")
        picture += (f'<h3>Direct coupling scores · {_text(summary["ranking"]["dca"])}</h3>'
                    f'<img alt="All three DCA score matrices" src="data:image/png;base64,{payload}">')
    if triple := outputs["figures"].get("threebody"):
        payload = base64.b64encode((directory / triple).read_bytes()).decode("ascii")
        picture += (f'<h3>Leading tested triples · {_text(summary["ranking"]["threebody"])}</h3>'
                    f'<img alt="Leading signed or RMS ABC scores" src="data:image/png;base64,{payload}">')
    settings = _table(["Setting", "Value"], _definition_rows(summary["settings"]))
    if advanced.get("settings"):
        settings += _table(["Model or benchmark setting", "Value"], _definition_rows(advanced["settings"]))
    coverage = ""
    if advanced.get("settings", {}).get("threebody"):
        coverage = "<h3>Candidate coverage</h3>" + _table(
            ["Candidate set", "Value"], _definition_rows(advanced["candidates"])
        )
    if selected := advanced.get("selection_scope"):
        scope = "Scores apply to the tested candidates. The manifest records the eligible grid, selection rules, source coordinates and per-candidate rationale."
        if selected["data_overlap"]:
            scope += (
                " Screening and evaluation use overlapping observations. Rankings are exploratory."
            )
        if selected["external_prior"]:
            scope += " An external prior informed candidate selection. An independent evidence source is required for validation."
        coverage += f"<p>{scope}</p>" + _link(advanced["candidate_manifest"], "Candidate manifest")
    if fit := advanced.get("local_fit"):
        coverage += "<h3>Local three-site information</h3><p>One weighted joint table defines the independent M1, pairwise maximum-entropy M2 and saturated M3 model. Connected information is KL(M3 || M2) in bits. Larger values indicate departure from a local pairwise model. A latent mixture can produce that departure.</p>"
        coverage += _table(["Primary fit status", "Candidates"], fit["primary_fit_status"].items())
        coverage += _table(
            ["Information status", "Candidates"], fit["primary_information_status"].items()
        )
        coverage += (
            "<p>"
            + _link(
                fit["diagnostic_table"], "Every candidate, prior setting and convergence residual"
            )
            + "</p>"
        )
    if dca_fit := advanced.get("dca_diagnostics"):
        design = advanced.get("settings", {}).get("dca_design", {})
        coverage += "<h3>Direct coupling fit</h3>" + _table(
            ["Setting", "Value"],
            [
                ("Backend", design.get("backend", "mf")),
                ("Conditioning", design.get("conditioning", "joint")),
                ("Status", dca_fit.get("status", "fitted")),
            ],
        )
        coverage += (
            "<p>Per-target or covariance diagnostics and the fitting policy appear in the "
            + _link("run_summary.json", "analysis manifest")
            + ".</p>"
        )
    if heldout := summary.get("conditional"):
        coverage += (
            "<h3>Held-out conditional prediction</h3>"
            + _table(
                ["Prespecified L2", "Evaluated folds", "Mean loss gain (nats/site)"],
                [
                    (row["l2"], row["evaluated_folds"], row["clade_macro_gain_nats_per_site"])
                    for row in heldout["grid"]
                ],
            )
            + _link(
                outputs["conditional_report"],
                "Paired test losses, uncertainty and fitted conditionals",
            )
        )
    if stable := summary.get("clade_stability"):
        coverage += "<h3>Held-out clade stability</h3><p>" + _text(stable["groups"]) + " groups in " + _text(stable["folds"]) + " training/held-out splits. " + _link(outputs["stability_report"],"Fold ranks, coordinate support and resampling intervals") + "</p>"
    if named := summary.get("calibration"):
        coverage += "<h3>Named null calibration</h3><p>" + _text(named["null"]) + " · " + _text(named["status"]) + " · " + _text(named["draws"]) + " draws. " + _link(outputs["calibration_report"], "Hypotheses, attainable resolution and correction families") + "</p>"
    if (calibration := advanced.get("triple_calibration")) and calibration["status"] == "withheld":
        coverage += (
            "<p><strong>Triple calibration withheld.</strong> "
            + _text(calibration["reason"])
            + "</p>"
        )
    if advanced.get("global_fit"):
        coverage += "<h3>Global fit diagnostics</h3>" + _table(
            ["Fit diagnostic", "Value"], _definition_rows(advanced["global_fit"])
        )
    input_table = _table(["Input", "File", "SHA-256"], (
        (key, Path(value["path"]).name, value["sha256"]) for key, value in summary["inputs"].items()
    ))
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>Triplet coevolution analysis</title><style>{STYLE}</style></head>
<body><header><h1>Triplet coevolution</h1><p>Associations across three matched sequence alignments</p></header>
<nav aria-label="Sections">{'<a href="#workflow">Search &amp; matching</a>' if workflow_section else ""}
<a href="#quality">Quality control</a><a href="#scores">Site rankings</a>
<a href="#benchmarks">Benchmarks</a><a href="#files">Files &amp; settings</a></nav>
<main>{workflow_section}<section id="quality"><h2>Alignment quality</h2><dl class="metrics">{metrics_html}</dl>
{qc}<div class="composition" aria-hidden="true">{"".join(strips)}<div class="legend">{legend}</div></div>
<p class="small">Coverage, gaps and missing symbols are averaged across retained observations before site filtering.
Effective depth is the sum of observation weights. Reference positions and original alignment columns start at 1.</p>{audit_block}</section>
<section id="scores"><h2>Site rankings</h2><p>Up to five leading results per ranking.
Shared ancestry, limited depth and indirect associations can influence every score.</p>
{_rankings(directory, tables)}{coverage}{picture}</section>
<section id="benchmarks"><h2>Signal benchmarks</h2>{_benchmarks(directory, advanced)}{benchmark_section(summary.get("benchmark_overview", []))}</section>
<section id="files"><h2>Files &amp; settings</h2><ul class="files">{links}</ul>
<p>The GraphML network includes up to {outputs["network"]["limit_per_table"]} entries per ranking.
Three-site interactions have their own nodes linked to each member site.</p>
<details><summary>Analysis settings and input checksums</summary>{settings}{input_table}</details></section>
<footer>Triplet coevolution {_text(summary["software_version"])}</footer></main></body></html>"""
    (directory / "report.html").write_text(doc, encoding="utf-8")
