"""Command-line analysis of three curated, matched alignments."""

import csv
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Annotated, get_args

import typer

from triplet_coevolution import __version__
from triplet_coevolution.analysis_engine import execute
from triplet_coevolution.calibration import calibrate
from triplet_coevolution.clade_workflow import split, stability
from triplet_coevolution.command_progress import configure
from triplet_coevolution.conditional_workflow import (
    conditional,
)
from triplet_coevolution.dca_compare import dca_compare
from triplet_coevolution.family_workflow import evidence
from triplet_coevolution.pipeline_commands import doctor, pipeline, select, validate_inputs
from triplet_coevolution.project_commands import init, score
from triplet_coevolution.publication import (
    SUMMARY_NAME,
    _check_output,
    _file_stamp,
    _input_identity,
    _stage_run,
)
from triplet_coevolution.score_commands import benchmark, export, report
from triplet_coevolution.shard_commands import merge_null, null_shard, shard_plan, shard_status
from triplet_coevolution.structure.workflow import structure_map
from triplet_coevolution.workflow import prepare, run

app = typer.Typer(
    help="Search, match and score three protein or nucleotide components.",
    epilog="Try triplet init --template protein-rna-rna --out demo to create a runnable local project.",
    no_args_is_help=True,
)


__all__ = [
    "SUMMARY_NAME",
    "_check_output",
    "_file_stamp",
    "_input_identity",
    "_stage_run",
    "analyze",
    "app",
]

app.command("init")(init)
app.command("score")(score)
app.command("prepare")(prepare)
app.command("doctor")(doctor)
app.command("validate-inputs")(validate_inputs)
app.command("pipeline")(pipeline)
app.command("select")(select)
app.command("shard-plan")(shard_plan)
app.command("null-shard")(null_shard)
app.command("merge-null")(merge_null)
app.command("shard-status")(shard_status)
app.command("run")(run)
app.command("evidence")(evidence)
app.command("benchmark")(benchmark)
app.command("export")(export)
app.command("report")(report)
app.command("calibrate")(calibrate)
app.command("conditional")(conditional)
app.command("dca-compare")(dca_compare)
app.command("stability")(stability)
app.command("split")(split)
app.command("structure-map")(structure_map)
app.command("contacts")(structure_map)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"triplet {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    quiet: Annotated[
        bool, typer.Option("--quiet", help="Suppress ordinary command output.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Log command and publication progress on stderr.")
    ] = False,
    json_log: Annotated[
        bool,
        typer.Option(
            "--json-log", help="Emit structured command and publication events on stderr."
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the installed version."
        ),
    ] = False,
) -> None:
    """Analyze coevolution in protein, DNA and RNA sequences."""
    configure(ctx, quiet=quiet, verbose=verbose, json_log=json_log)


@wraps(execute)
def analyze(*args, **kwargs):
    """Translate the native Python result into CLI output and process exit status."""
    try:
        result = execute(*args, **kwargs)
    except (ValueError, TypeError, OSError, ImportError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    # Direct Python callers of this legacy command retain their output contract.
    out = Path(signature(execute).bind_partial(*args, **kwargs).arguments.get("out", "results"))
    if kwargs.get("dry_run") or "matched_observations" not in result:
        typer.echo(f"{result['status']}: {out / 'plan.json'}", err=result["status"] == "blocked")
        if result["status"] == "blocked":
            raise typer.Exit(3)
        return result
    typer.echo(f"matched obs: {result['matched_observations']}")
    typer.echo(f"effective depth: {result['effective_depth']:.2f}")
    typer.echo(
        "kept cols: "
        + " ".join(f"{k}={v['retained_sites']}" for k, v in result["components"].items())
    )
    for k, v in result["outputs"]["tables"].items():
        typer.echo(f"{k}: {out / v['path']}")
    if result["outputs"]["figures"]:
        typer.echo(f"plots: {out / 'figures'}")
    typer.echo(f"report: {out / 'report.html'}")
    typer.echo(f"summary: {out / SUMMARY_NAME}")


def _cli_parameters():
    params = []
    for parameter in signature(execute).parameters.values():
        if parameter.name.startswith("_"):
            continue
        kind, option = get_args(parameter.annotation)
        params.append(
            parameter.replace(
                annotation=Annotated[
                    kind, typer.Option(option.flag, help=option.help, min=option.min)
                ]
            )
        )
    return signature(execute).replace(parameters=params)


analyze.__signature__ = _cli_parameters()
app.command("analyze")(analyze)


@wraps(analyze)
def plan(*args, **kwargs):
    kwargs["dry_run"] = True
    return analyze(*args, **kwargs)


app.command(
    "plan",
    help="Validate native analysis inputs and inspect QC and allocation estimates without fitting.",
)(plan)


@wraps(analyze)
def local(*args, **kwargs):
    """Run the local M1/M2/M3 analysis with native input and QC options."""
    if kwargs.get("threebody_model", "maxent") != "maxent":
        typer.echo("error: triplet local requires the maxent model", err=True)
        raise typer.Exit(1)
    kwargs.update(threebody=True, threebody_model="maxent")
    return analyze(*args, **kwargs)


# Keep the full native input/QC interface and display the actual model defaults in help.
_native_parameters = signature(analyze)
local.__signature__ = _native_parameters.replace(
    parameters=[
        parameter.replace(default={"threebody": True, "threebody_model": "maxent"}[name])
        if name in {"threebody", "threebody_model"}
        else parameter
        for name, parameter in _native_parameters.parameters.items()
    ]
)
app.command(
    "local",
    help="Compare independent, pairwise maximum-entropy and saturated three-site distributions.",
)(local)


if __name__ == "__main__":
    app()
