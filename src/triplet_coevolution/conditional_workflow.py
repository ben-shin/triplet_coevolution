"""Run a prespecified comparison of nested models on held-out clades."""

import csv
import json
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution.analysis_plan import match_metadata
from triplet_coevolution.clade_workflow import CladePolicy, _relocated, _source
from triplet_coevolution.config import _unique
from triplet_coevolution.inference.nested import NestedOptions, compare_nested, write_nested
from triplet_coevolution.io.score_store import file_digest
from triplet_coevolution.msa.support import site_masks
from triplet_coevolution.provenance import file_hashes
from triplet_coevolution.score_views import write_json
from triplet_coevolution.validation.clade_splits import load_split_plan, make_split_plan
from triplet_coevolution.validation.null_archive import read_json
from triplet_coevolution.validation.tree_design import TreeSource


def read_policy(config):
    base = Path.cwd()
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        base = path.parent
        config = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
    if not isinstance(config, dict):
        raise TypeError("model policy must be a JSON object")
    config = dict(config)
    version = config.pop("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("unknown model policy schema version")
    return config, base


def conditional_design(config):
    opts, base = read_policy(config)
    if "clades" not in opts:
        raise ValueError(
            "conditional requires clades with a split_manifest, clade_column, or saved clade policy JSON"
        )
    clades = opts.pop("clades")
    if isinstance(clades, str):
        clades = base / clades
    elif isinstance(clades, dict):
        clades = dict(clades)
        if "split_manifest" in clades:
            clades["split_manifest"] = base / clades["split_manifest"]
        if "clade_tree" in clades:
            tree = TreeSource.parse(clades["clade_tree"], base)
            clades["clade_tree"] = {"newick": tree.newick, "tip_map": tree.tip_map}
    policy = CladePolicy.from_config(clades)
    if policy.options:
        raise ValueError(
            "clades accepts split and weight settings only; set model options in options"
        )
    parameters = NestedOptions(**opts.pop("options", {}))
    grid = opts.pop("l2_grid", [parameters.l2])
    if (
        not isinstance(grid, list)
        or not 1 <= len(grid) <= 12
        or any(type(v) not in (float, int) for v in grid)
        or len(set(grid)) != len(grid)
    ):
        raise ValueError(
            "l2_grid must be a prespecified list of 1 to 12 distinct positive penalties"
        )
    settings = [replace(parameters, l2=v) for v in grid]
    if len(settings) > 1 and all(
        x is not None for x in (parameters.field_l2, parameters.pair_l2, parameters.triple_l2)
    ):
        raise ValueError(
            "l2_grid cannot vary the fit when every order-specific penalty is overridden"
        )
    if "panel_file" in opts:
        if "panel" in opts:
            raise ValueError("provide either panel or panel_file")
        opts["panel_file"] = base / opts["panel_file"]
    if set(opts) - {
        "panel_file",
        "panel",
        "edges",
        "triples",
        "candidate_options",
        "bootstrap_draws",
        "null_draws",
        "seed",
    }:
        raise ValueError("unknown conditional setting")
    if len(grid) > 1 and opts.get("null_draws", 0):
        raise ValueError(
            "prespecify one comparison for a parametric null; run descriptive regularization sensitivity separately"
        )
    return policy, settings, opts


def write_conditional(stage, design, *, msa, maps, table, digests):
    policy, settings, opts = design
    opts = dict(opts)
    panel_hash = None
    if file := opts.pop("panel_file", None):
        panel_hash = file_digest(file)
        masks = site_masks(file, msa.components, maps)
        if file_digest(file) != panel_hash:
            raise ValueError("nested panel file changed while reading")
        opts["panel"] = {
            m.component: [x for x, yes in zip(m.original_positions, mask, strict=True) if yes]
            for m, mask in zip(maps, masks, strict=True)
        }
    stage.mkdir(parents=True, exist_ok=True)
    groups, meta, tests = policy.partition(table, digests["matches"])
    ids = [r.observation_id for r in msa.matches]
    plan = (
        load_split_plan(policy.split_manifest, ids)
        if policy.split_manifest
        else make_split_plan(
            ids, groups, test_groups=tests, source={"group_source": meta, "inputs": digests}
        )
    )
    write_json(stage / "split_manifest.json", plan.metadata())
    weights = {
        "policy": policy.weight_policy,
        "identity_threshold": policy.identity_threshold,
        "min_overlap": policy.min_overlap,
    }
    if policy.weight_component:
        weights["component"] = policy.weight_component
    if policy.weight_column:
        weights["supplied_weights"] = match_metadata(
            table, range(len(ids)), policy.weight_column, numeric=True
        )
    rows = []
    for i, setting in enumerate(settings):
        name = f"fit_{i + 1}"
        fit = compare_nested(msa, maps, plan, options=setting, weight_options=weights, **opts)
        record = write_nested(stage / name, fit)
        rows.append(
            {
                "l2": setting.l2,
                "status": record["status"],
                "evaluated_folds": record["evaluated_folds"],
                "clade_macro_gain_nats_per_site": record["clade_macro_gain_nats_per_site"],
                "heldout_clade_interval": record["heldout_clade_interval"],
                "summary": name + "/nested_summary.json",
            }
        )
    report = "<h1>Held-out conditional prediction</h1><p>Positive gain means lower paired test log loss for the model with three-site terms, in nats per site. Each linked fit includes train/test support, per-site losses, fitted models and interval diagnostics. The grid is descriptive. Predictive gain depends on the selected graph, clade partition and observations.</p>"
    for row in rows:
        interval = row["heldout_clade_interval"]
        ci = (
            f"{interval['lower']:.4g} to {interval['upper']:.4g}"
            if interval.get("lower") is not None
            else str(interval.get("status", "unavailable")).replace("_", " ")
        )
        report += f'<p><a href="{row["summary"]}">L2 {row["l2"]:g}</a> · {escape(row["status"])} · {row["evaluated_folds"]} evaluated folds · mean gain {escape(str(row["clade_macro_gain_nats_per_site"]))} · clade interval {escape(ci)}</p>'
    (stage / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Held-out conditional prediction</title>'
        + report
        + "</html>",
        encoding="utf8",
    )
    result = {
        "schema": "triplet.conditional-workflow.v1",
        "inputs": digests,
        "panel_file_sha256": panel_hash,
        "split_sha256": plan.metadata()["plan_sha256"],
        "source_axis_scope": "all original matched rows and source axes before analysis-level QC; declared panel applied by original columns",
        "selection": "all prespecified penalties reported; no held-out tuning",
        "grid": rows,
        "report": "report.html",
        "artifact_sha256": file_hashes(stage),
    }
    write_json(stage / "conditional_summary.json", result)
    return result


def conditional_run(run, out, config, *, inputs=None):
    """Fit two matched regularized conditionals per training fold and evaluate their test losses."""
    from triplet_coevolution.cli import _stage_run

    design = conditional_design(config)
    source, out = Path(run).resolve(), Path(out).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("conditional output must be separate from source analysis")
    with _stage_run(out) as stage:
        summary = read_json(source / "run_summary.json")
        msa, maps, table, digests = _source(source, summary, _relocated(inputs))
        return write_conditional(stage, design, msa=msa, maps=maps, table=table, digests=digests)


def conditional(
    config: Annotated[Path, typer.Option("--config")],
    run: Annotated[Path, typer.Option("--run")],
    out: Annotated[Path, typer.Option("--out")],
    inputs: Annotated[Path | None, typer.Option("--inputs")] = None,
):
    """Compare held-out conditional loss for shared pair and pair-plus-three-site models."""
    try:
        conditional_run(run, out, config, inputs=inputs)
    except (ValueError, TypeError, KeyError, OSError, ImportError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(str(out / "conditional_summary.json"))
