"""Terminal outcomes and bounded scoring for a three-folder run."""

import csv
import hashlib
import sqlite3
from contextlib import redirect_stderr
from datetime import UTC, datetime
from importlib import import_module
from io import StringIO
from pathlib import Path
from uuid import uuid4

import typer

from triplet_coevolution import __version__
from triplet_coevolution.inference.advanced import ModelFitError
from triplet_coevolution.msa.support import MissingRowSupport
from triplet_coevolution.reporting.automatic import write_workflow_report
from triplet_coevolution.workflow import _analyze, _write_analysis_args, _write_json

EXIT_CODES = {
    "complete": 0,
    "invalid_input": 1,
    "analysis_error": 1,
    "no_matches": 3,
    "insufficient_data": 3,
    "partial": 3,
    "resource_limit": 3,
    "provider_failure": 4,
}
DEFAULT_LIMITS = {"pair_cells": 1_000_000, "weight_cells": 4_000_000_000}


def _terminal(summary, status, reason):
    summary.update(status=status, exit_code=EXIT_CODES[status], reason=reason, report="report.html")
    if status in {"insufficient_data", "analysis_error", "resource_limit"}:
        summary["ready_for_analysis"] = False
        if reason not in summary["analysis_blockers"]:
            summary["analysis_blockers"].append(reason)


def _plan(components, n, overrides, limits, plots):
    a, b, c = (comp["source_columns"] for comp in components.values())
    pairs = a * b + a * c + b * c
    width = sum(
        comp["source_columns"] * min(n, 20 if comp["molecule"] == "protein" else 4)
        for comp in components.values()
    )
    work = n * n * (a + b + c)
    limits = {**DEFAULT_LIMITS, **limits}
    dca = min(width, n) <= 512 and n * width <= 3_000_000 and pairs <= 80_000
    draws = (
        19
        if pairs <= 12_000 and pairs * n <= 800_000 and work <= 10_000_000
        else 5
        if pairs <= 80_000 and pairs * n <= 4_000_000 and work <= 80_000_000
        else 0
    )
    settings = {"dca": dca, "threebody": True, "permutations": draws, "seed": 0, **overrides}
    if settings["threebody"]:
        settings.setdefault("max_triplets", 64)
    if plots is not None:
        settings["plots"] = plots
    skipped = {}
    if "dca" not in overrides and not dca:
        skipped["dca"] = "Panel exceeds the automatic DCA size budget."
    if "permutations" not in overrides and not draws:
        skipped["permutations"] = "Panel exceeds the automatic permutation refit budget."
    if settings.get("plots"):
        try:
            import_module("triplet_coevolution.reporting.plots")
        except (ImportError, OSError) as exc:
            skipped["plots"] = f"PNG export requires a working plots extra. {exc}"
            settings["plots"] = False
    plan = {
        "pair_cells": pairs,
        "weight_cells": work,
        "limits": limits,
        "overrides": sorted(overrides),
        "skipped": skipped,
    }
    return settings, plan


def _record_settings(plan, settings):
    plan["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}


def _score(stage, out, summary, kwargs, settings, plan):
    from triplet_coevolution.reporting.html import attach_evidence_report

    while True:
        effective = {**kwargs, **settings}
        replay = "panel_analyze_args.json" if "scoring_panel" in summary else "analyze_args.json"
        if "observation_qc" in summary:
            replay = "scored_analyze_args.json"
        _write_analysis_args(stage / "prepared", effective, replay)
        summary["analyze_args"] = f"prepared/{replay}"
        _record_settings(plan, settings)
        try:
            with redirect_stderr(StringIO()):
                analysis = _analyze(stage, out, effective)
            break
        except typer.Exit as exc:
            cause = exc.__cause__
            if isinstance(cause, ModelFitError) and settings.get(cause.method):
                method = cause.method
                plan["skipped"][method] = str(cause)
                settings[method] = False
                dependent = (
                    ("ridge",)
                    if method == "dca"
                    else (
                        "threebody_model",
                        "global_l2",
                        "pseudocount",
                        "max_triplets",
                        "triplet_candidates",
                    )
                )
                for key in dependent:
                    settings.pop(key, None)
                    kwargs.pop(key, None)
                continue
            if isinstance(cause, MissingRowSupport) and "observation_qc" not in summary:
                from triplet_coevolution.automatic_qc import filter_observations

                revised, audit = filter_observations(stage / "prepared", effective)
                summary["observation_qc"] = audit
                eligible = audit["eligible_observations"]
                minimum = settings.get("min_valid", 2)
                if eligible < minimum:
                    summary["analyze_args"] = None
                    _terminal(
                        summary,
                        "insufficient_data",
                        f"After synchronized observation QC, "
                        f"{eligible} observations remain and min_valid requires {minimum}.",
                    )
                    return
                kwargs.update(revised)
                settings.pop("site_panel", None)
                continue
            elif isinstance(cause, ValueError) and str(cause).startswith(
                "the permutation null requires"
            ):
                plan["skipped"]["permutations"] = str(cause)
                settings.update(permutations=0)
                settings.pop("strata_column", None)
                kwargs.pop("strata_column", None)
                continue
            else:
                _terminal(summary, "analysis_error", str(cause or exc))
            return
    summary.update(
        analysis_summary="analysis/run_summary.json", score_report="analysis/report.html"
    )
    summary["scored_scope"] = {
        "selection": "bounded_panel"
        if "scoring_panel" in summary
        else ("qc_supported" if "observation_qc" in summary else "full"),
        "observations": analysis["matched_observations"],
        "retained_columns": {n: c["retained_sites"] for n, c in analysis["components"].items()},
    }
    if not any(t["rows"] for t in analysis["outputs"]["tables"].values()):
        _terminal(summary, "insufficient_data", "No pair passed the configured output thresholds.")
    elif "observation_qc" in summary:
        _terminal(
            summary,
            "partial",
            "Scoring finished on observations with usable residues in all three components.",
        )
    elif "scoring_panel" in summary:
        _terminal(summary, "partial", "Scoring finished on a bounded panel of the resolved data.")
    elif plan["skipped"]:
        _terminal(summary, "partial", "Scoring finished with the method omissions listed below.")
    else:
        _terminal(summary, "complete", "Scoring finished for the resolved observations.")
    attach_evidence_report(stage / "analysis", summary)


def _bounded_score(stage, out, summary, kwargs, plan, plots):
    from triplet_coevolution.automatic_panel import make_scoring_panel

    panel_kwargs, audit = make_scoring_panel(
        stage / "prepared", kwargs, summary["components"], plan["limits"], min_valid=2, seed=0
    )
    summary["scoring_panel"] = audit
    if audit["status"] != "selected":
        _terminal(summary, "resource_limit", audit["reason"])
        return
    components = {
        n: {**c, "source_columns": audit["selected_widths"][n]}
        for n, c in summary["components"].items()
    }
    settings, panel_plan = _plan(components, audit["n_selected"], {}, plan["limits"], plots)
    summary["full_analysis_plan"] = plan
    summary["full_analyze_args"] = summary["analyze_args"]
    summary["analysis_plan"] = panel_plan
    _score(stage, out, summary, {**kwargs, **panel_kwargs}, settings, panel_plan)


def finish_run(stage, out, summary, kwargs, overrides, limits, plots):
    n = summary["counts"]["resolved_groups"]
    settings, plan = _plan(summary["components"], n, overrides, limits, plots)
    summary["analysis_plan"] = plan
    summary["scored_scope"] = None
    _record_settings(plan, settings)
    if summary["preparation_error"]:
        _terminal(summary, "invalid_input", summary["preparation_error"])
    elif summary["providers"]["status"] == "partial":
        summary["ready_for_analysis"] = False
        summary["analysis_blockers"].append(
            "A provider outage left competing identities unchecked."
        )
        _terminal(
            summary,
            "provider_failure",
            "Public metadata lookups failed. Incomplete evidence could hide competing matches.",
        )
    elif not n:
        _terminal(
            summary,
            "no_matches",
            "No group had one supported locus for each of the three families.",
        )
    elif summary["analysis_blockers"]:
        _terminal(
            summary,
            "insufficient_data",
            "Coordinate or molecule evidence was insufficient for scoring.",
        )
    else:
        _write_analysis_args(stage / "prepared", {**kwargs, **settings})
        summary["analyze_args"] = "prepared/analyze_args.json"
        if n < settings.get("min_valid", 2):
            _terminal(
                summary, "insufficient_data", f"{n} matched observations fall below min_valid."
            )
        elif any(plan[k] > v for k, v in plan["limits"].items()):
            if overrides or limits:
                _terminal(
                    summary,
                    "resource_limit",
                    "The full matched panel exceeds the analysis "
                    "budget. The project specifies the analysis or its limits.",
                )
            else:
                _bounded_score(stage, out, summary, kwargs, plan, plots)
        else:
            _score(stage, out, summary, kwargs, settings, plan)
    write_workflow_report(stage, summary)


def run_families(directory, out, online, cache_dir, plots):
    from triplet_coevolution.cli import _check_output, _stage_run
    from triplet_coevolution.family_workflow import _prepare

    directory = directory.expanduser().resolve()
    if out is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path("results") / f"run-{stamp}-{uuid4().hex[:12]}"
    out = out.expanduser().resolve()
    _check_output(out)
    if out.is_relative_to(directory):
        raise ValueError("--out must be outside the input project")
    key = hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    shared = (cache_dir or (out.parent / ".triplet-cache" / key)).expanduser().resolve()
    if shared.is_relative_to(directory) or shared.is_relative_to(out) or out.is_relative_to(shared):
        raise ValueError("keep the lookup cache separate from the input and output directories")
    try:
        summary = _prepare(
            None,
            directory,
            out,
            True,
            online,
            None,
            automatic=True,
            cache_directory=shared,
            plots=plots,
        )
    except (ValueError, TypeError, OSError, csv.Error, sqlite3.Error, EOFError) as exc:
        summary = {
            "schema_version": 1,
            "software_version": __version__,
            "mode": "automatic",
            "input_directory": str(directory),
        }
        _terminal(summary, "invalid_input", str(exc))
        with _stage_run(out) as stage:
            write_workflow_report(stage, summary)
            _write_json(stage / "workflow_summary.json", summary)
    typer.echo(f"{summary['status']} (exit {summary['exit_code']}): {summary['reason']}")
    typer.echo(f"Report: {out / 'report.html'}")
    return summary["exit_code"]
