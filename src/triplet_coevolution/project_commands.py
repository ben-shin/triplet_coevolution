"""Small command adapters for versioned native-analysis projects."""

from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution.api import AnalysisRequest, analyze
from triplet_coevolution.templates import TEMPLATES, init_project


def init(
    out: Annotated[Path, typer.Option("--out", help="New example project directory.")],
    template: Annotated[
        str, typer.Option("--template", help=", ".join(TEMPLATES))
    ] = "protein-protein-rna",
):
    """Create a runnable synthetic three-folder project and native analysis configuration."""
    try:
        path = init_project(out, template)
    except (ValueError, OSError, TypeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(str(path))


def score(
    config: Annotated[Path, typer.Option("--config", help="Versioned native analysis JSON.")],
    out: Annotated[Path, typer.Option("--out", help="New or empty result directory.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Write QC and preflight without fitting.")
    ] = False,
):
    """Run or plan the same native analysis available in the Python API."""
    try:
        result = analyze(AnalysisRequest.from_config(config), out, dry_run=dry_run)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(str(result.manifest))
    if result.summary.get("status") == "blocked":
        raise typer.Exit(3)
