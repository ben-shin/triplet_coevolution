"""Command entry points for planning, inspection and resumable local execution."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution.diagnostics import doctor_report, validate_config, validate_folders
from triplet_coevolution.local_pipeline import run_pipeline
from triplet_coevolution.operations import put_json, read_json

Output = Annotated[Path | None, typer.Option("--out", help="Optional new/empty result directory.")]
Config = Annotated[Path | None, typer.Option("--config")]
Work = Annotated[
    Path,
    typer.Option(
        "--work-dir", help="Persistent workspace for receipts, attempts and verified artifacts."
    ),
]
Workers = Annotated[int, typer.Option("--workers", min=1)]
Threads = Annotated[int, typer.Option("--blas-threads", min=1)]
Memory = Annotated[float | None, typer.Option("--memory-mib", min=0.0001)]
Disk = Annotated[float, typer.Option("--disk-mib", min=0)]
Errors = (OSError, ValueError, TypeError, KeyError, csv.Error, subprocess.SubprocessError)


def _result(result, out, name):
    if out:
        from triplet_coevolution.publication import _stage_run

        with _stage_run(out) as tmp:
            put_json(tmp / name, result)
    typer.echo(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] == "blocked":
        raise typer.Exit(3)


def doctor(
    scratch: Annotated[Path, typer.Option("--scratch")] = Path("."),
    required: Annotated[list[str] | None, typer.Option("--require-tool")] = None,
    out: Output = None,
    workers: Workers = 1,
    blas_threads: Threads = 1,
    memory_mib: Memory = None,
    disk_mib: Disk = 0,
) -> None:
    """Inspect runtime, optional tools and available CPU, memory and disk. Prints JSON."""
    try:
        _result(
            doctor_report(
                scratch,
                required=required or (),
                workers=workers,
                blas_threads=blas_threads,
                memory_mib=memory_mib,
                disk_mib=disk_mib,
            ),
            out,
            "doctor.json",
        )
    except Errors as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


def validate_inputs(
    config: Config = None,
    directory: Annotated[Path | None, typer.Option("--input")] = None,
    prepared: Annotated[Path | None, typer.Option("--prepared")] = None,
    out: Output = None,
) -> None:
    """Validate workflow inputs offline, or reproduce the full native plan on prepared inputs."""
    try:
        if out:
            from triplet_coevolution.publication import _check_output

            _check_output(out)
        if sum(p is not None for p in (config, directory, prepared)) != 1:
            raise ValueError("choose exactly one of --config, --input and --prepared")
        if prepared:
            if not out:
                raise ValueError("--prepared requires --out for its QC tables and plan")
            replay = read_json(prepared / "analyze_args.json")
            if (
                not isinstance(replay, list)
                or not replay
                or replay[0] != "analyze"
                or not all(isinstance(v, str) for v in replay)
                or "--out" in replay
            ):
                raise ValueError("invalid prepared native argument manifest")
            ret = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "triplet_coevolution.cli",
                    "plan",
                    *replay[1:],
                    "--out",
                    str(out.resolve()),
                ],
                cwd=prepared,
                check=False,
            )
            if ret.returncode:
                raise typer.Exit(ret.returncode)
            return
        result = validate_config(config) if config else validate_folders(directory)
        _result(result, out, "validation.json")
    except Errors as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


def pipeline(
    config: Annotated[Path, typer.Option("--config")],
    work_dir: Work,
    until: Annotated[
        str,
        typer.Option(
            "--until", help="prepare, plan, score, select, calibrate, benchmark or report."
        ),
    ] = "report",
    labels: Annotated[Path | None, typer.Option("--labels")] = None,
    null_config: Annotated[Path | None, typer.Option("--null-config")] = None,
    workers: Workers = 1,
    blas_threads: Threads = 1,
    memory_mib: Memory = None,
    disk_mib: Disk = 0,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
    json_log: Annotated[
        bool, typer.Option("--json-log", help="Emit each stage event as JSON on stderr.")
    ] = False,
) -> None:
    """Prepare, plan, score and regenerate views with verified restartable local stages."""
    try:
        if quiet and (verbose or json_log):
            raise ValueError("--quiet conflicts with --verbose/--json-log")

        def emit(event):
            from triplet_coevolution.command_progress import quiet_mode

            if quiet or quiet_mode.get():
                return
            if json_log:
                typer.echo(json.dumps(event), err=True)
            elif verbose or event["event"] in ("stage_finished", "run_finished"):
                typer.echo(
                    " ".join(
                        str(event[k])
                        for k in ("event", "stage", "action", "status", "elapsed_seconds")
                        if k in event
                    ),
                    err=True,
                )

        result = run_pipeline(
            config,
            work_dir,
            until=until,
            labels=labels,
            null_config=null_config,
            workers=workers,
            blas_threads=blas_threads,
            memory_mib=memory_mib,
            disk_mib=disk_mib,
            dry_run=dry_run,
            emit=emit,
        )
        if not quiet:
            typer.echo(json.dumps(result, indent=2) if dry_run else str(result["manifest"]))
        if (
            result["status"] in ("insufficient_data", "blocked")
            or result.get("validation", {}).get("status") == "blocked"
        ):
            raise typer.Exit(3)
    except Errors as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


def select(
    run: Annotated[Path, typer.Option("--run")],
    out: Annotated[Path, typer.Option("--out")],
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Default reuses the fitted screen, otherwise seeded uniform."),
    ] = None,
    maximum: Annotated[int, typer.Option("--maximum", min=1)] = 1000,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 0,
    candidate_file: Annotated[Path | None, typer.Option("--candidate-file")] = None,
    prior_source: Annotated[str | None, typer.Option("--prior-source")] = None,
    statistic: Annotated[str, typer.Option("--pair-statistic")] = "mi_bits",
    screen_budget: Annotated[int, typer.Option("--screen-budget", min=1)] = 100000,
    training_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--training-id",
            help="Restrict an internally refit screen to these source observations.",
        ),
    ] = None,
    file_origin: Annotated[
        str | None,
        typer.Option(
            "--file-origin", help="For explicit files: prespecified, same_data or training."
        ),
    ] = None,
) -> None:
    """Recover a fitted candidate manifest or build a new coordinate selection from archived scores."""
    from triplet_coevolution.api import select as select_from_run

    try:
        result = select_from_run(
            run,
            out,
            mode=mode,
            maximum=maximum,
            seed=seed,
            candidate_file=candidate_file,
            prior_source=prior_source,
            statistic=statistic,
            screen_budget=screen_budget,
            training_ids=training_ids,
            file_origin=file_origin,
        )
        typer.echo(str(result.manifest))
    except Errors as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
