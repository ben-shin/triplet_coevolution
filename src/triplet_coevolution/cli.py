"""Command-line analysis of three curated, matched alignments."""

import csv
import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution import __version__
from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.io.alignments import read_msa
from triplet_coevolution.io.export import write_triplet_mi_tables
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.msa.masking import apply_triplet_masks, column_mask
from triplet_coevolution.msa.qc import triplet_qc
from triplet_coevolution.msa.synchronize import synchronize

app = typer.Typer(
    help="Measure pairwise mutual information across three matched sequence alignments.",
    no_args_is_help=True,
)

SUMMARY_NAME = "run_summary.json"


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"triplet {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the installed version."
        ),
    ] = False,
) -> None:
    """Analyze curated triplets of protein, DNA, or RNA sequences."""


def _check_output(out: Path) -> None:
    if (out.exists() or out.is_symlink()) and not out.is_dir():
        raise ValueError(f"--out must be a new or empty directory: {out}")
    if out.is_dir() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}. Choose a new --out directory.")


@contextmanager
def _stage_run(out: Path) -> Iterator[Path]:
    """Complete the run in private and publish its manifest last.

    The exclusive staging directory serializes writers targeting the same output.
    Exclusive file creation and rollback also protect pre-existing empty directories.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        out.mkdir()
        created = True
    except FileExistsError:
        _check_output(out)

    stage = out / ".triplet-in-progress"
    try:
        stage.mkdir()
        try:
            if any(path != stage for path in out.iterdir()):
                raise ValueError(
                    f"output directory is not empty: {out}. Choose a new --out directory."
                )
            yield stage
            with ExitStack() as rollback:
                # Directories come before their contents. The manifest marks a completed run.
                artifacts = sorted(stage.rglob("*"), key=lambda p: (p.name == SUMMARY_NAME, p))
                for artifact in artifacts:
                    target = out / artifact.relative_to(stage)
                    if artifact.is_dir():
                        target.mkdir()
                        rollback.callback(target.rmdir)
                    else:
                        with artifact.open("rb") as source, target.open("xb") as dest:
                            rollback.callback(target.unlink)
                            shutil.copyfileobj(source, dest)
                rollback.pop_all()
        finally:
            shutil.rmtree(stage)
    except BaseException:
        if created:
            # A concurrent writer may have claimed this destination.
            with suppress(OSError):
                out.rmdir()
        raise


def _file_stamp(path: Path) -> tuple[int, int, int, int]:
    if not path.is_file():
        raise ValueError(f"input file not found: {path}")
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _input_identity(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.absolute()), "sha256": digest}


def _row_count(path: Path, fmt: str) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t" if fmt == "tsv" else ",")
        next(rows)  # Header is present even when there are no qualifying pairs.
        return sum(1 for _ in rows)


@app.command("analyze")
def analyze(
    a: Annotated[Path, typer.Option("--a", help="Aligned FASTA or Stockholm file for A.")],
    b: Annotated[Path, typer.Option("--b", help="Aligned FASTA or Stockholm file for B.")],
    c: Annotated[Path, typer.Option("--c", help="Aligned FASTA or Stockholm file for C.")],
    matches: Annotated[Path, typer.Option("--matches", help="Curated CSV or TSV triplet table.")],
    a_ref: Annotated[str, typer.Option("--a-ref", help="Reference sequence ID in A.")],
    b_ref: Annotated[str, typer.Option("--b-ref", help="Reference sequence ID in B.")],
    c_ref: Annotated[str, typer.Option("--c-ref", help="Reference sequence ID in C.")],
    out: Annotated[Path, typer.Option("--out", help="New or empty results directory.")] = Path(
        "results"
    ),
    a_type: Annotated[
        str, typer.Option("--a-type", help="Molecule type for A (protein, dna, or rna).")
    ] = "protein",
    b_type: Annotated[
        str, typer.Option("--b-type", help="Molecule type for B (protein, dna, or rna).")
    ] = "protein",
    c_type: Annotated[
        str, typer.Option("--c-type", help="Molecule type for C (protein, dna, or rna).")
    ] = "protein",
    a_start: Annotated[
        int, typer.Option("--a-start", help="First occupied A reference position.")
    ] = 1,
    b_start: Annotated[
        int, typer.Option("--b-start", help="First occupied B reference position.")
    ] = 1,
    c_start: Annotated[
        int, typer.Option("--c-start", help="First occupied C reference position.")
    ] = 1,
    min_coverage: Annotated[
        float,
        typer.Option(
            "--min-coverage", help="Minimum fraction of canonical residues per column (0 to 1)."
        ),
    ] = 0.5,
    max_gap_fraction: Annotated[
        float, typer.Option("--max-gap-fraction", help="Maximum gap fraction per column (0 to 1).")
    ] = 0.5,
    max_missing_fraction: Annotated[
        float,
        typer.Option(
            "--max-missing-fraction",
            help="Maximum ambiguous or missing fraction per column (0 to 1).",
        ),
    ] = 0.5,
    identity_threshold: Annotated[
        float,
        typer.Option(
            "--identity-threshold",
            help="Minimum mean component identity for sequence weighting (0 to 1).",
        ),
    ] = 0.8,
    min_overlap: Annotated[
        float,
        typer.Option(
            "--min-overlap",
            help="Minimum jointly canonical fraction per component for weighting (0 to 1).",
        ),
    ] = 0.5,
    min_valid: Annotated[
        int,
        typer.Option(
            "--min-valid",
            help="Minimum unweighted observations with canonical residues at both sites.",
        ),
    ] = 2,
    max_pairs: Annotated[
        int, typer.Option("--max-pairs", help="Maximum number of ranked pairs per component pair.")
    ] = 1000,
    score: Annotated[str, typer.Option("--score", help="Rank by mi_apc or mi_bits.")] = "mi_apc",
    fmt: Annotated[str, typer.Option("--fmt", help="Ranked table format (tsv or csv).")] = "tsv",
    require_reference: Annotated[
        bool,
        typer.Option(
            "--require-reference",
            help="Only export rows with numbered reference positions at both sites.",
        ),
    ] = False,
    plots: Annotated[
        bool,
        typer.Option(
            "--plots/--no-plots", help="Save PNG heatmaps of the full filtered score matrices."
        ),
    ] = False,
) -> None:
    """Rank AB, AC, and BC site pairs using curated one-to-one sequence matches."""
    try:
        if score not in {"mi_bits", "mi_apc"}:
            raise ValueError("--score must be mi_bits or mi_apc")
        if fmt not in {"csv", "tsv"}:
            raise ValueError("--fmt must be csv or tsv")
        if min_valid < 1 or max_pairs < 1:
            raise ValueError("--min-valid and --max-pairs must be positive integers")
        out = out.expanduser()
        _check_output(out)

        if plots:
            try:
                from triplet_coevolution.reporting.plots import save_mi_plots
            except ModuleNotFoundError as exc:
                if exc.name == "matplotlib":
                    raise ValueError(
                        '--plots requires Matplotlib. Install it with pip install ".[plots]" '
                        "from the project directory."
                    ) from exc
                raise

        paths = {
            key: path.expanduser()
            for key, path in zip(("A", "B", "C", "matches"), (a, b, c, matches))
        }
        stamps = {key: _file_stamp(path) for key, path in paths.items()}
        inputs = {key: _input_identity(path) for key, path in paths.items()}
        original = tuple(
            read_msa(paths[name], name, molecule)
            for name, molecule in zip(("A", "B", "C"), (a_type, b_type, c_type))
        )
        table = read_triplet_table(paths["matches"])
        for key, path in paths.items():
            if _file_stamp(path) != stamps[key]:
                raise ValueError(
                    f"input changed while reading: {path}. Retry with stable input files."
                )

        maps = tuple(
            build_reference_map(component, ref_id, start)
            for component, ref_id, start in zip(
                original, (a_ref, b_ref, c_ref), (a_start, b_start, c_start)
            )
        )
        matched = synchronize(*original, table)
        qc = triplet_qc(matched)
        mask_options = {
            "min_coverage": min_coverage,
            "max_gap_fraction": max_gap_fraction,
            "max_missing_fraction": max_missing_fraction,
        }
        keep = tuple(column_mask(part, **mask_options) for part in qc.components)
        for part, mask in zip(qc.components, keep):
            if not mask.any():
                raise ValueError(
                    f"QC removed every column in component {part.name}. Check the masking thresholds."
                )

        filtered = apply_triplet_masks(matched, *keep)
        maps = tuple(
            subset_reference_map(mapping, cols)
            for mapping, cols in zip(
                maps, (filtered.a_columns, filtered.b_columns, filtered.c_columns)
            )
        )
        weighting_options = {"identity_threshold": identity_threshold, "min_overlap": min_overlap}
        weights = sequence_weights(filtered.msa, **weighting_options)
        result = triplet_mi(filtered.msa, weights=weights, min_valid=min_valid)
        table_options = {
            "score": score,
            "max_pairs": max_pairs,
            "min_valid": min_valid,
            "require_reference": require_reference,
            "fmt": fmt,
        }
        with _stage_run(out) as stage:
            tables = write_triplet_mi_tables(stage, result, *maps, **table_options)
            figures = (
                save_mi_plots(stage / "figures", result, *maps, score=score, min_valid=min_valid)
                if plots
                else {}
            )
            summary = {
                "schema_version": 1,
                "software_version": __version__,
                "inputs": inputs,
                "components": {
                    component.name: {
                        "molecule_type": component.molecule.value,
                        "reference_id": mapping.reference_id,
                        "reference_start": mapping.start,
                        "input_sequences": component.n_sequences,
                        "original_sites": component.n_sites,
                        "retained_sites": mapping.n_sites,
                        "retained_columns": mapping.original_positions,
                        "mean_coverage": stats.mean_coverage,
                        "mean_gap_fraction": stats.mean_gap_fraction,
                        "mean_missing_fraction": stats.mean_missing_fraction,
                    }
                    for component, mapping, stats in zip(original, maps, qc.components)
                },
                "settings": {**mask_options, **weighting_options, **table_options, "plots": plots},
                "matched_observations": matched.n_observations,
                "effective_depth": weights.n_eff,
                "outputs": {
                    "tables": {
                        name.upper(): {"path": path.name, "rows": _row_count(path, fmt)}
                        for name, path in tables.items()
                    },
                    "figures": {
                        ("triplet" if name == "triplet" else name.upper()): path.relative_to(
                            stage
                        ).as_posix()
                        for name, path in figures.items()
                    },
                },
            }
            (stage / SUMMARY_NAME).write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    except (ValueError, TypeError, OSError, ImportError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"matched obs: {matched.n_observations}")
    typer.echo(f"effective depth: {weights.n_eff:.2f}")
    typer.echo("kept cols: " + " ".join(f"{m.component}={m.n_sites}" for m in maps))
    for name, path in tables.items():
        typer.echo(f"{name}: {out / path.name}")
    if plots:
        typer.echo(f"plots: {out / 'figures'}")
    typer.echo(f"summary: {out / SUMMARY_NAME}")


if __name__ == "__main__":
    app()
