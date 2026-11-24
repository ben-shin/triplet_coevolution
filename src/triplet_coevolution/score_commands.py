"""Commands operating on completed scores without invoking analysis."""

import csv
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution.io.score_store import write_score_store
from triplet_coevolution.score_views import (
    read_scores,
    write_json,
    write_ranked_views,
)

Scores = Annotated[
    Path, typer.Option("--scores", help="Native analysis, scores directory or score_manifest.json.")
]
Out = Annotated[Path, typer.Option("--out", help="New or empty output directory.")]
FromV1 = Annotated[
    bool,
    typer.Option("--from-v1", help="Import complete legacy score tables and run_summary.json."),
]


@contextmanager
def _output(directory: Path) -> Iterator[Path]:
    from triplet_coevolution.publication import _check_output, _stage_run

    try:
        _check_output(directory)
        with _stage_run(directory) as staging:
            yield staging
    except (ValueError, KeyError, OSError, TypeError, ImportError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


def benchmark(
    scores: Scores,
    labels: Annotated[Path, typer.Option("--labels", "--contacts")],
    out: Out,
    from_v1: FromV1 = False,
    merged: Annotated[
        bool,
        typer.Option(
            "--merged", help="Also export the full reference-coordinate score/label join."
        ),
    ] = False,
    bootstrap: Annotated[
        int,
        typer.Option(
            "--bootstrap", min=0, help="Cluster bootstrap draws for labeled validation units."
        ),
    ] = 0,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 0,
    top_k: Annotated[list[int] | None, typer.Option("--top-k", min=1)] = None,
) -> None:
    """Evaluate all stored statistics against independently supplied contact labels."""
    from triplet_coevolution.api import benchmark as benchmark_archive

    try:
        result = benchmark_archive(
            scores,
            labels,
            out,
            from_v1=from_v1,
            merged=merged,
            bootstrap=bootstrap,
            seed=seed,
            top_k=tuple(top_k or [10, 25, 50, 100]),
        )
    except (ValueError, KeyError, OSError, TypeError, ImportError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Benchmark: {result.manifest}")


def export(
    scores: Scores,
    out: Out,
    from_v1: FromV1 = False,
    mi_score: Annotated[str | None, typer.Option("--mi-score", "--score")] = None,
    dca_score: Annotated[str | None, typer.Option("--dca-score")] = None,
    max_pairs: Annotated[int, typer.Option("--max-pairs", min=1)] = 1000,
    all_pairs: Annotated[bool, typer.Option("--all-pairs")] = False,
    fmt: Annotated[str, typer.Option("--fmt")] = "tsv",
) -> None:
    """Rerank pairs, or migrate a complete v1 scoring universe into a verified archive."""
    with _output(out) as stage:
        if all_pairs and max_pairs != 1000:
            raise ValueError("choose one of --max-pairs and --all-pairs")
        store = read_scores(scores, from_v1=from_v1)
        tables = write_ranked_views(
            stage,
            store,
            mi_score=mi_score,
            dca_score=dca_score,
            max_pairs=None if all_pairs else max_pairs,
            fmt=fmt,
        )
        summary = {
            "schema_version": 1,
            "apc_universe": store.provenance["apc_universe"],
            "tables": [asdict(table) for table in tables],
        }
        if from_v1:
            manifest = write_score_store(stage / "scores", store)
            summary["score_manifest"] = manifest.relative_to(stage).as_posix()
        write_json(stage / "view_summary.json", summary)
    typer.echo(f"Ranked tables: {out / 'view_summary.json'}")


def report(
    out: Out,
    scores: Annotated[
        Path | None, typer.Option("--scores", help="Render views of a complete pair-score archive.")
    ] = None,
    run: Annotated[
        Path | None,
        typer.Option(
            "--run", help="Regenerate a native analysis report including triples and nulls."
        ),
    ] = None,
    from_v1: FromV1 = False,
    labels: Annotated[Path | None, typer.Option("--labels")] = None,
    mi_score: Annotated[str | None, typer.Option("--mi-score", "--score")] = None,
    dca_score: Annotated[str | None, typer.Option("--dca-score")] = None,
    plots: Annotated[bool, typer.Option("--plots")] = False,
    max_pairs: Annotated[int, typer.Option("--max-pairs", min=1)] = 1000,
    palette: Annotated[
        str, typer.Option("--palette", help="accessible or amber-blue")
    ] = "accessible",
    figure_format: Annotated[
        list[str] | None, typer.Option("--figure-format", help="Repeat for svg, pdf or png")
    ] = None,
    structure: Annotated[
        Path | None,
        typer.Option(
            "--structure",
            help="Link a verified native structure-map result on the same references.",
        ),
    ] = None,
) -> None:
    """Regenerate offline views and structural overlays from completed scientific artifacts."""
    from triplet_coevolution.api import report as report_archive

    try:
        result = report_archive(
            out,
            run=run,
            scores=scores,
            from_v1=from_v1,
            labels=labels,
            mi_score=mi_score,
            dca_score=dca_score,
            plots=plots,
            max_pairs=max_pairs,
            palette=palette,
            figure_formats=tuple(figure_format or ()),
            structure=structure,
        )
    except (ValueError, KeyError, OSError, TypeError, ImportError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Report: {result.report}")
