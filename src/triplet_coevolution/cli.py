"""Command-line analysis of three curated, matched alignments."""

import csv
import hashlib
import json
import math
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution import __version__
from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.family_workflow import evidence
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.io.alignments import read_msa
from triplet_coevolution.io.export import write_triplet_mi_tables
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.msa.masking import apply_triplet_masks, column_mask
from triplet_coevolution.msa.qc import triplet_qc
from triplet_coevolution.msa.synchronize import synchronize
from triplet_coevolution.reporting.html import write_report
from triplet_coevolution.reporting.network import ranked_tables, write_network
from triplet_coevolution.workflow import prepare, run

app = typer.Typer(
    help="Search, match and score three protein or nucleotide components.",
    no_args_is_help=True,
)

SUMMARY_NAME = "run_summary.json"

app.command("prepare")(prepare)
app.command("run")(run)
app.command("evidence")(evidence)


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
    """Analyze coevolution in protein, DNA and RNA sequences."""


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
                final_manifest = stage / "workflow_summary.json"
                if not final_manifest.exists():
                    final_manifest = stage / SUMMARY_NAME
                artifacts = sorted(
                    stage.rglob("*"), key=lambda p: (p == final_manifest, p.name == SUMMARY_NAME, p)
                )
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
    dca: Annotated[bool, typer.Option("--dca", help="Fit a joint mean-field Potts model.")] = False,
    threebody: Annotated[
        bool, typer.Option("--threebody", help="Estimate a three-site Potts interaction for ABC.")
    ] = False,
    threebody_model: Annotated[
        str, typer.Option("--threebody-model", help="Three-body fit: local or global.")
    ] = "local",
    global_l2: Annotated[
        float, typer.Option("--global-l2", help="Positive regularization for global three-body fit.")
    ] = 0.1,
    ridge: Annotated[float, typer.Option("--ridge", help="Positive DCA ridge penalty.")] = 0.1,
    pseudocount: Annotated[
        float, typer.Option("--pseudocount", help="Three-body smoothing parameter.")
    ] = 0.5,
    max_triplets: Annotated[
        int, typer.Option("--max-triplets", help="Maximum number of ABC candidates to fit.")
    ] = 1000,
    triplet_candidates: Annotated[
        Path | None, typer.Option("--triplet-candidates", help="TSV of candidate ABC site triples.")
    ] = None,
    permutations: Annotated[
        int, typer.Option("--permutations", help="Null replicates for empirical tests (0 disables).")
    ] = 0,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for permutation tests.")] = 0,
    strata_column: Annotated[
        str | None,
        typer.Option("--strata-column", help="Match-table column defining permutation groups."),
    ] = None,
    contacts: Annotated[
        Path | None, typer.Option("--contacts", help="Reference-coordinate contact labels TSV.")
    ] = None,
    a_ref_row: Annotated[
        str | None, typer.Option("--a-ref-row", help="Coordinate row in A if different from --a-ref.")
    ] = None,
    b_ref_row: Annotated[
        str | None, typer.Option("--b-ref-row", help="Coordinate row in B if different from --b-ref.")
    ] = None,
    c_ref_row: Annotated[
        str | None, typer.Option("--c-ref-row", help="Coordinate row in C if different from --c-ref.")
    ] = None,
    a_strand: Annotated[str, typer.Option("--a-strand", help="Reference coordinate direction (+ or -).")] = "+",
    b_strand: Annotated[str, typer.Option("--b-strand", help="Reference coordinate direction (+ or -).")] = "+",
    c_strand: Annotated[str, typer.Option("--c-strand", help="Reference coordinate direction (+ or -).")] = "+",
) -> None:
    """Score curated alignments, benchmark associations and write a local HTML report."""
    try:
        if score not in {"mi_bits", "mi_apc"}:
            raise ValueError("--score must be mi_bits or mi_apc")
        if fmt not in {"csv", "tsv"}:
            raise ValueError("--fmt must be csv or tsv")
        if min_valid < 1 or max_pairs < 1 or max_triplets < 1:
            raise ValueError("--min-valid, --max-pairs and --max-triplets must be positive")
        if permutations < 0 or seed < 0:
            raise ValueError("--permutations and --seed must be nonnegative")
        if strata_column is not None and not permutations:
            raise ValueError("--strata-column requires a positive --permutations value")
        if triplet_candidates is not None and not threebody:
            raise ValueError("--triplet-candidates requires --threebody")
        if threebody_model not in {"local", "global"}:
            raise ValueError("--threebody-model must be local or global")
        if threebody_model == "global" and not threebody:
            raise ValueError("--threebody-model global requires --threebody")
        for name, value in (("--ridge", ridge), ("--global-l2", global_l2),
                            ("--pseudocount", pseudocount)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not dca and ridge != 0.1:
            raise ValueError("--ridge requires --dca")
        if global_l2 != 0.1 and threebody_model != "global":
            raise ValueError("--global-l2 requires --threebody-model global")
        if pseudocount != 0.5 and (not threebody or threebody_model != "local"):
            raise ValueError("--pseudocount requires --threebody with the local model")
        if max_triplets != 1000 and not threebody:
            raise ValueError("--max-triplets requires --threebody")
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
        for key, path in (("triplet_candidates", triplet_candidates), ("contacts", contacts)):
            if path is not None:
                paths[key] = path.expanduser()
        stamps = {key: _file_stamp(path) for key, path in paths.items()}
        inputs = {key: _input_identity(path) for key, path in paths.items()}
        original = tuple(
            read_msa(paths[name], name, molecule)
            for name, molecule in zip(("A", "B", "C"), (a_type, b_type, c_type))
        )
        table = read_triplet_table(paths["matches"])
        strata = None
        if strata_column is not None:
            if strata_column not in table.extra_columns:
                raise ValueError(f"unknown permutation stratum column in match table: {strata_column}")
            strata = tuple(row[strata_column] for row in table.annotations)
            if not all(strata):
                raise ValueError(f"empty permutation group in match table column {strata_column}")
        for key, path in paths.items():
            if _file_stamp(path) != stamps[key]:
                raise ValueError(
                    f"input changed while reading: {path}. Retry with stable input files."
                )

        coordinate_rows = (a_ref_row, b_ref_row, c_ref_row)
        maps = tuple(
            build_reference_map(component, ref_id, start, row_id=row_id, strand=strand)
            for component, ref_id, start, row_id, strand in zip(
                original, (a_ref, b_ref, c_ref), (a_start, b_start, c_start), coordinate_rows,
                (a_strand, b_strand, c_strand),
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
            advanced = {}
            if dca or threebody or permutations or contacts is not None:
                from triplet_coevolution.inference.advanced import write_advanced_results

                advanced = write_advanced_results(
                    stage, filtered.msa, maps, weights, dca=dca, threebody=threebody,
                    ridge=ridge, pseudocount=pseudocount, max_triplets=max_triplets,
                    threebody_model=threebody_model, global_l2=global_l2,
                    triplet_candidates=paths.get("triplet_candidates"), permutations=permutations,
                    seed=seed, strata=strata, contacts=paths.get("contacts"),
                    mi_result=result, **table_options,
                )
                for key in ("triplet_candidates", "contacts"):
                    if key in paths and _file_stamp(paths[key]) != stamps[key]:
                        raise ValueError(f"input changed while reading: {paths[key]}")
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
                        **({"reference_row_id": row_id} if row_id is not None else {}),
                        "reference_start": mapping.start,
                        **({"reference_strand": "-"} if mapping.strand == "-" else {}),
                        "input_sequences": component.n_sequences,
                        "original_sites": component.n_sites,
                        "retained_sites": mapping.n_sites,
                        "retained_columns": mapping.original_positions,
                        "mean_coverage": stats.mean_coverage,
                        "mean_gap_fraction": stats.mean_gap_fraction,
                        "mean_missing_fraction": stats.mean_missing_fraction,
                    }
                    for component, mapping, stats, row_id in zip(original, maps, qc.components, coordinate_rows)
                },
                "settings": {**mask_options, **weighting_options, **table_options, "plots": plots},
                "matched_observations": matched.n_observations,
                "effective_depth": weights.n_eff,
                "advanced": advanced,
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
            if advanced:
                advanced["settings"]["strata_column"] = strata_column
            ranked = ranked_tables(summary)
            summary["outputs"]["network"] = write_network(stage, ranked)
            summary["outputs"]["report"] = "report.html"
            write_report(stage, summary, ranked)
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
    typer.echo(f"report: {out / 'report.html'}")
    typer.echo(f"summary: {out / SUMMARY_NAME}")


if __name__ == "__main__":
    app()
