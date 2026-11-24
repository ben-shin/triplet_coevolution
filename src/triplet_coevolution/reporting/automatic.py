"""Write the local landing page for every terminal family-run outcome."""

from triplet_coevolution.reporting.html import STYLE, _link, _table, _text, row_support_section

GUIDANCE = {
    "no_matches": "Open the group and candidate evidence tables to see missing identities, "
    "ambiguous loci and incomplete groups. A future run can use additional "
    "annotation or cached identifier records.",
    "provider_failure": "Retry the same command after service recovery. Each run has a new output "
    "directory, and completed lookup records can be reused.",
    "insufficient_data": "The matching audit and coordinate table identify the available evidence. "
    "Scoring requires enough matched, jointly observed canonical residues.",
    "resource_limit": "The synchronized subsets are available for replay. A future run can use "
    "target lists or project analysis_limits suited to the available hardware.",
    "invalid_input": "Check the input layout and source identified in the message above.",
    "analysis_error": "The prepared inputs and evidence are available. Check the analysis settings "
    "and the reported cause before a future run.",
}


def write_workflow_report(directory, summary):
    status = summary["status"]
    body = [
        f"<section><h2>{_text(status.replace('_', ' ').capitalize())}</h2>",
        f"<p>{_text(summary['reason'])}</p>",
    ]
    if status in GUIDANCE:
        body.append(f"<p>{GUIDANCE[status]}</p>")
    if summary.get("score_report"):
        body.append(
            f"<p>{_link(summary['score_report'], 'Open site rankings, quality control and benchmarks')}</p>"
        )
    body.append("</section>")
    if "counts" in summary:
        counts = summary["counts"]
        tiers = summary["role_evidence_counts"]
        body.append(
            f"<section><h2>Matching evidence</h2><p>{counts['resolved_groups']:,} "
            f"of {counts['groups']:,} groups resolved. "
            f"{tiers.get('orthogroup_supported', 0):,} had orthogroup support for all three roles. "
            f"{tiers.get('profile_membership_included', 0):,} relied on family membership for "
            "one or more roles.</p>"
        )
        body.append(
            _table(
                [
                    "Role",
                    "Folder",
                    "Molecule",
                    "Type evidence",
                    "Reference row",
                    "Matched / input rows",
                ],
                (
                    (
                        name,
                        c["name"],
                        c["molecule"] or "unknown",
                        summary["molecules"][name]["source"],
                        c["reference_row_id"] or "unavailable",
                        f"{c['exported_rows']} / {c['source_rows']}",
                    )
                    for name, c in summary["components"].items()
                ),
            )
        )
        if statuses := summary.get("group_status_counts"):
            body.append(_table(["Group decision", "Groups"], statuses.items()))
        if issues := summary["providers"].get("lookup_status_counts"):
            body.append(_table(["Lookup or placement issue", "Affected rows"], issues.items()))
        body.append("</section>")
    body.append(row_support_section(summary))
    notes = list(summary.get("analysis_blockers", ()))
    if summary.get("cache_warning"):
        notes.append(f"Saving the reusable cache failed. {summary['cache_warning']}")
    notes += [
        f"{key}: {value}" for key, value in summary.get("providers", {}).get("failures", {}).items()
    ]
    plan = summary.get("analysis_plan", {})
    if plan:
        body.append("<section><h2>Analysis plan</h2>")
        if scope := summary.get("scored_scope"):
            counts = ", ".join(f"{k} {v:,}" for k, v in scope["retained_columns"].items())
            body.append(
                f"<p>Scored {scope['observations']:,} of "
                f"{summary['counts']['resolved_groups']:,} resolved observations. "
                f"Retained sites after quality filtering: {_text(counts)}.</p>"
            )
        if panel := summary.get("scoring_panel"):
            body.append(f"<p>{_text(panel['reason'])}</p>")
            if panel.get("status") == "selected":
                body.append(
                    _table(
                        ["Component", "Selected original columns", "Source columns"],
                        (
                            (k, v, panel["source_widths"][k])
                            for k, v in panel["selected_widths"].items()
                        ),
                    )
                )
        body.append(_table(["Method or setting", "Planned value"], plan["settings"].items()))
        notes += [f"{k}: {v}" for k, v in plan["skipped"].items()]
        body.append("</section>")
    if notes:
        body.append(
            "<section><h2>Run notes</h2><ul>"
            + "".join(f"<li>{_text(n)}</li>" for n in notes)
            + "</ul></section>"
        )
    keys = {
        "evidence": "All candidate evidence",
        "group_audit": "Group decisions",
        "qc_plan": "Blocked analysis plan and QC diagnostics",
        "selected_evidence": "Selected evidence",
        "matches": "Synchronized matches",
        "reference_coordinates": "Reference coordinates",
        "resolver_summary": "Lookup provenance",
        "analyze_args": "Analysis replay arguments",
        "full_analyze_args": "Full matched data replay arguments",
        "analysis_summary": "Analysis manifest",
    }
    links = [_link("workflow_summary.json", "Workflow status and manifest")]
    links.extend(_link(summary[k], label) for k, label in keys.items() if summary.get(k))
    panel = summary.get("scoring_panel", {})
    for key, title in (
        ("matches", "Scored panel observations"),
        ("site_panel", "Scored panel candidates"),
    ):
        if panel.get("status") == "selected" and panel.get(key):
            links.append(_link(panel[key], title))
    body.append(
        '<section><h2>Files</h2><ul class="files">'
        + "".join(f"<li>{link}</li>" for link in links)
        + "</ul></section>"
    )
    version = _text(summary["software_version"])
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-family run</title>
<style>{STYLE}</style></head><body><header><h1>Three-family run</h1>
<p>Triplet coevolution {version}</p></header><main>{"".join(body)}</main></body></html>"""
    (directory / "report.html").write_text(doc, encoding="utf-8")
