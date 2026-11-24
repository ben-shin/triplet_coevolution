"""Command-line analysis of three curated, matched alignments."""

import csv
import hashlib
import json
import math
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution import __version__
from triplet_coevolution.analysis_plan import (
    AnalysisBudgetError,
    QCBlocked,
    estimate_model,
    match_metadata,
    publish_plan,
    replay_arguments,
    write_weighted_sites,
    write_weights,
)
from triplet_coevolution.calibration import calibrate
from triplet_coevolution.clade_workflow import CladePolicy, split, stability
from triplet_coevolution.command_progress import configure, event
from triplet_coevolution.conditional_workflow import (
    conditional,
    conditional_design,
    write_conditional,
)
from triplet_coevolution.dca_compare import dca_compare
from triplet_coevolution.family_workflow import evidence
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.inference.scores import get_score, mi_with_apc_policy, pair_coverage
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.io.export import write_triplet_mi_tables
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.io.selected_sources import InputBudgetError, matched_sources
from triplet_coevolution.msa.support import site_masks
from triplet_coevolution.msa.synchronize import synchronize
from triplet_coevolution.pipeline_commands import doctor, pipeline, select, validate_inputs
from triplet_coevolution.provenance import file_hashes
from triplet_coevolution.reporting.html import write_report
from triplet_coevolution.reporting.network import ranked_tables, write_network
from triplet_coevolution.score_commands import benchmark, export, report
from triplet_coevolution.score_views import save_analysis_store, write_json
from triplet_coevolution.shard_commands import merge_null, null_shard, shard_plan, shard_status
from triplet_coevolution.structure.workflow import structure_map
from triplet_coevolution.workflow import prepare, run

app = typer.Typer(
    help="Search, match and score three protein or nucleotide components.",
    no_args_is_help=True,
)

SUMMARY_NAME = "run_summary.json"

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
            event("stage_started", output=str(out))
            yield stage
            with ExitStack() as rollback:
                # Directories come before their contents. The manifest marks a completed run.
                marker_names = (
                    "shard_plan.json",
                    "shard_receipt.json",
                    "selection_receipt.json",
                    "validation.json",
                    "doctor.json",
                    "stability_summary.json",
                    "conditional_summary.json",
                    "dca_comparison.json",
                    "split_manifest.json",
                    "calibration.json",
                    "structure_summary.json",
                    "workflow_summary.json",
                    SUMMARY_NAME,
                    "view_summary.json",
                    "benchmark.json",
                    "score_manifest.json",
                    "plan.json",
                )
                final_manifest = next(
                    (stage / name for name in marker_names if (stage / name).exists()), None
                )
                artifacts = sorted(
                    stage.rglob("*"), key=lambda p: (p == final_manifest, p.name in marker_names, p)
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
                event("stage_published", output=str(out), files=sum(p.is_file() for p in artifacts))
        finally:
            shutil.rmtree(stage)
    except BaseException:
        event("stage_failed", output=str(out))
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
    dca: Annotated[
        bool,
        typer.Option(
            "--dca", help="Fit direct couplings with the selected backend and conditioning panel."
        ),
    ] = False,
    threebody: Annotated[
        bool, typer.Option("--threebody", help="Estimate a three-site Potts interaction for ABC.")
    ] = False,
    threebody_model: Annotated[
        str,
        typer.Option(
            "--threebody-model",
            help="ABC model: local projected log-linear, maxent hierarchy or global pseudolikelihood.",
        ),
    ] = "local",
    global_l2: Annotated[
        float,
        typer.Option("--global-l2", help="Positive regularization for global three-body fit."),
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
    candidate_mode: Annotated[
        str | None,
        typer.Option(
            "--candidate-mode",
            help="uniform, exhaustive, explicit, prior, triangle, wedge, entropy or hybrid.",
        ),
    ] = None,
    candidate_policy: Annotated[
        Path | None,
        typer.Option(
            "--candidate-policy",
            help="JSON screen budgets, training IDs, quotas or external prior provenance.",
        ),
    ] = None,
    local_policy: Annotated[
        Path | None,
        typer.Option(
            "--local-policy",
            help="JSON maximum-entropy support, total or per-cell prior, sensitivity and IPF controls.",
        ),
    ] = None,
    permutations: Annotated[
        int,
        typer.Option("--permutations", help="Null replicates for empirical tests (0 disables)."),
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
        str | None,
        typer.Option("--a-ref-row", help="Coordinate row in A if different from --a-ref."),
    ] = None,
    b_ref_row: Annotated[
        str | None,
        typer.Option("--b-ref-row", help="Coordinate row in B if different from --b-ref."),
    ] = None,
    c_ref_row: Annotated[
        str | None,
        typer.Option("--c-ref-row", help="Coordinate row in C if different from --c-ref."),
    ] = None,
    a_strand: Annotated[
        str, typer.Option("--a-strand", help="Reference coordinate direction (+ or -).")
    ] = "+",
    b_strand: Annotated[
        str, typer.Option("--b-strand", help="Reference coordinate direction (+ or -).")
    ] = "+",
    c_strand: Annotated[
        str, typer.Option("--c-strand", help="Reference coordinate direction (+ or -).")
    ] = "+",
    site_panel: Annotated[
        Path | None,
        typer.Option(
            "--site-panel",
            help="TSV of component and 1-based original_column to intersect with quality masks.",
        ),
    ] = None,
    dca_score: Annotated[
        str | None,
        typer.Option("--dca-score", help="Rank DCA by dca_apc (default) or dca_frobenius."),
    ] = None,
    threebody_score: Annotated[
        str | None,
        typer.Option(
            "--threebody-score",
            help="ABC ranking: connected_information_bits (maxent), threebody_rms, or signed co_information_bits.",
        ),
    ] = None,
    apc_universe: Annotated[
        str | None,
        typer.Option(
            "--apc-universe", help="Compute APC over retained sites or reference-numbered sites."
        ),
    ] = None,
    legacy: Annotated[
        bool,
        typer.Option(
            "--legacy", help="Use v1 raw DCA ranking, retained-site APC and three-body RMS."
        ),
    ] = False,
    all_pairs: Annotated[
        bool,
        typer.Option("--all-pairs", help="Export every eligible finite pair in the ranked tables."),
    ] = False,
    benchmark_bootstrap: Annotated[
        int,
        typer.Option(
            "--benchmark-bootstrap", help="Number of block bootstrap draws for the supplied labels."
        ),
    ] = 0,
    benchmark_k: Annotated[
        list[int] | None,
        typer.Option(
            "--benchmark-k",
            min=1,
            help="Top-K contact-retrieval cutoffs. Repeat for multiple cutoffs.",
        ),
    ] = None,
    qc_policy: Annotated[
        Path | None,
        typer.Option(
            "--qc-policy", help="Versioned JSON row/site filtering and diagnostics policy."
        ),
    ] = None,
    weight_policy: Annotated[
        str,
        typer.Option(
            "--weight-policy",
            help="equal_component, component, concatenated, cluster_representatives, clade_balanced, supplied, or uniform.",
        ),
    ] = "equal_component",
    weight_component: Annotated[
        str | None,
        typer.Option("--weight-component", help="Component A, B or C for component weighting."),
    ] = None,
    clade_column: Annotated[
        str | None,
        typer.Option(
            "--clade-column", help="Match-table clade field for clade-balanced weights or metadata."
        ),
    ] = None,
    weight_column: Annotated[
        str | None,
        typer.Option(
            "--weight-column", help="Match-table field with nonnegative absolute supplied weights."
        ),
    ] = None,
    min_valid_weight: Annotated[
        float,
        typer.Option(
            "--min-valid-weight",
            help="Minimum positive canonical weight at a scored pair/triple, inclusive.",
        ),
    ] = 0.0,
    memory_mib: Annotated[
        float | None,
        typer.Option(
            "--memory-mib",
            help="Numeric-workspace budget in MiB, including score kernels; estimates exclude process overhead.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Inspect inputs, QC and allocation estimates without fitting or permuting.",
        ),
    ] = False,
    null_policy: Annotated[
        Path | None,
        typer.Option(
            "--null-policy", help="Named null-model and registered-statistic calibration JSON."
        ),
    ] = None,
    stability_policy: Annotated[
        Path | None,
        typer.Option(
            "--stability-policy",
            help="Clade hold-out and bootstrap policy JSON applied to original input rows and sites.",
        ),
    ] = None,
    conditional_policy: Annotated[
        Path | None,
        typer.Option(
            "--conditional-policy",
            help="Fit nested pair and three-site models on prespecified held-out clades.",
        ),
    ] = None,
    dca_backend: Annotated[
        str | None,
        typer.Option("--dca-backend", help="mf or plm (regularized asymmetric pseudolikelihood)."),
    ] = None,
    dca_conditioning: Annotated[
        str | None,
        typer.Option("--dca-conditioning", help="joint ABC or pairwise separate interface fits."),
    ] = None,
    dca_policy: Annotated[
        Path | None,
        typer.Option(
            "--dca-policy", help="DCA feature, regularization, solver and budget policy JSON."
        ),
    ] = None,
    a_format: Annotated[
        str | None,
        typer.Option("--a-format", help="Aligned input format override (fasta, stockholm, a3m)."),
    ] = None,
    b_format: Annotated[
        str | None,
        typer.Option("--b-format", help="Aligned input format override (fasta, stockholm, a3m)."),
    ] = None,
    c_format: Annotated[
        str | None,
        typer.Option("--c-format", help="Aligned input format override (fasta, stockholm, a3m)."),
    ] = None,
) -> dict | None:
    """Score curated alignments, benchmark associations and write a local HTML report."""
    options = locals().copy()
    try:
        from triplet_coevolution.inference.dca_models import DCADesign
        from triplet_coevolution.msa.diagnostics import QCPolicy, inspect_qc, write_qc
        from triplet_coevolution.validation.null_policy import NullPolicy, calibration_plan

        nested = conditional_design(conditional_policy) if conditional_policy else None
        design = None
        if any(x is not None for x in (dca_policy, dca_backend, dca_conditioning)):
            if not dca:
                raise ValueError("DCA backend, conditioning and policy require --dca")
            policy_stamp = _file_stamp(dca_policy) if dca_policy else None
            design = DCADesign.parse(
                dca_policy,
                backend=dca_backend,
                conditioning=dca_conditioning,
                ridge=ridge if ridge != 0.1 else None,
            )
            if legacy and design != DCADesign(ridge=ridge):
                raise ValueError("--legacy requires the original joint mean-field DCA settings")
        named_null = NullPolicy.from_config(null_policy) if null_policy else None
        clade_policy = CladePolicy.from_config(stability_policy) if stability_policy else None
        if named_null and permutations:
            raise ValueError("choose --null-policy or legacy --permutations for a native analysis")
        started = datetime.now(UTC)
        if not math.isfinite(min_valid_weight) or min_valid_weight < 0:
            raise ValueError("--min-valid-weight must be finite and nonnegative")
        if memory_mib is not None and (not math.isfinite(memory_mib) or memory_mib <= 0):
            raise ValueError("--memory-mib must be finite and positive")
        budget = None
        if memory_mib is not None:
            numerator, denominator = memory_mib.as_integer_ratio()
            budget = numerator * 1024**2 // denominator
        if weight_policy not in {
            "equal_component",
            "component",
            "concatenated",
            "cluster_representatives",
            "clade_balanced",
            "supplied",
            "uniform",
        }:
            raise ValueError("unknown weight policy")
        if (weight_policy == "component") != (weight_component is not None) or (
            weight_component is not None and weight_component not in ("A", "B", "C")
        ):
            raise ValueError("component weighting requires --weight-component A, B or C")
        if (weight_policy == "supplied") != (weight_column is not None):
            raise ValueError("supplied weighting requires --weight-column")
        if weight_policy == "clade_balanced" and clade_column is None:
            raise ValueError("clade_balanced weighting requires --clade-column")
        if legacy and (
            dca_score not in (None, "dca_frobenius")
            or threebody_score not in (None, "threebody_rms")
            or apc_universe not in (None, "retained")
        ):
            raise ValueError("--legacy requires dca_frobenius, threebody_rms and retained APC")
        if dca_score is not None and not dca:
            raise ValueError("--dca-score requires --dca")
        if threebody_score is not None and not threebody:
            raise ValueError("--threebody-score requires --threebody")
        dca_score = dca_score or ("dca_frobenius" if legacy else "dca_apc")
        if legacy and (
            threebody_model == "maxent" or candidate_mode or candidate_policy or local_policy
        ):
            raise ValueError("--legacy uses the legacy threebody models and candidate modes")
        threebody_score = threebody_score or (
            "connected_information_bits" if threebody_model == "maxent" else "threebody_rms"
        )
        if threebody_score == "connected_information_bits" and threebody_model != "maxent":
            raise ValueError("connected_information_bits requires --threebody-model maxent")
        if (candidate_mode or candidate_policy or local_policy) and not threebody:
            raise ValueError("ABC selection and model options require --threebody")
        if local_policy and threebody_model != "maxent":
            raise ValueError("--local-policy requires --threebody-model maxent")
        apc_universe = apc_universe or "retained"
        if apc_universe not in {"reference", "retained"}:
            raise ValueError("--apc-universe must be retained or reference")
        get_score(dca_score, "dca")
        get_score(threebody_score, "threebody")
        if all_pairs and max_pairs != 1000:
            raise ValueError("choose one of --max-pairs and --all-pairs")
        if benchmark_bootstrap < 0 or (benchmark_bootstrap and contacts is None):
            raise ValueError("--benchmark-bootstrap requires labels and a nonnegative draw count")
        if benchmark_k and (
            contacts is None or any(type(k) is not int or k < 1 for k in benchmark_k)
        ):
            raise ValueError("--benchmark-k requires contact labels and positive integer cutoffs")
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
        if threebody_model not in {"local", "global", "maxent"}:
            raise ValueError("--threebody-model must be local, maxent or global")
        if threebody_model != "local" and not threebody:
            raise ValueError("--threebody-model requires --threebody")
        for name, value in (
            ("--ridge", ridge),
            ("--global-l2", global_l2),
            ("--pseudocount", pseudocount),
        ):
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

        if plots and not dry_run:
            try:
                from triplet_coevolution.reporting.plots import save_dca_plot, save_mi_plots
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
        for key, path in (
            ("candidate_policy", candidate_policy),
            ("local_policy", local_policy),
            ("null_policy", null_policy),
            ("stability_policy", stability_policy),
            ("dca_policy", dca_policy),
            ("conditional_policy", conditional_policy),
            ("triplet_candidates", triplet_candidates),
            ("contacts", contacts),
            ("site_panel", site_panel),
            ("qc_policy", qc_policy),
        ):
            if path is not None:
                paths[key] = path.expanduser()
        stamps = {key: _file_stamp(path) for key, path in paths.items()}
        if dca_policy and stamps["dca_policy"] != policy_stamp:
            raise ValueError("DCA policy changed while reading")
        inputs = {key: _input_identity(path) for key, path in paths.items()}
        table = read_triplet_table(paths["matches"])
        coordinate_rows = (a_ref_row, b_ref_row, c_ref_row)
        try:
            source = matched_sources(
                paths,
                table.matches,
                dict(zip("ABC", (a_type, b_type, c_type), strict=True)),
                dict(zip("ABC", (a_ref, b_ref, c_ref), strict=True)),
                row_references=dict(zip("ABC", coordinate_rows, strict=True)),
                starts=dict(zip("ABC", (a_start, b_start, c_start), strict=True)),
                strands=dict(zip("ABC", (a_strand, b_strand, c_strand), strict=True)),
                formats=dict(zip("ABC", (a_format, b_format, c_format), strict=True)),
                byte_limit=budget,
            )
        except InputBudgetError as exc:
            if not dry_run:
                raise AnalysisBudgetError(str(exc)) from exc
            publish_plan(
                out,
                {
                    "schema_version": 1,
                    "status": "blocked",
                    "blockers": [{"method": "inputs", "reason": str(exc)}],
                    "source_selection": exc.details,
                },
                inputs,
            )
            typer.echo(f"Blocked: {exc}. See {out / 'plan.json'}", err=True)
            raise typer.Exit(3) from exc
        original, maps, source_selection = source.components, source.maps, source.statistics
        for key, path in paths.items():
            if _file_stamp(path) != stamps[key]:
                raise ValueError(
                    f"input changed while reading: {path}. Retry with stable input files."
                )
        original_maps = maps
        matched = synchronize(*original, table)
        widths = [c.n_sites for c in matched.components]
        # Encoded alignments, canonical masks, per-column histograms and cumulative gap runs.
        qc_bytes = matched.n_observations * (4 * sum(widths) + 32 * max(widths)) + 20 * 32 * sum(
            widths
        )
        if budget is not None and qc_bytes > budget:
            why = f"QC array workspace estimate of {qc_bytes} bytes exceeds the --memory-mib budget. Use a smaller input panel."
            if dry_run:
                publish_plan(
                    out,
                    {
                        "schema_version": 1,
                        "status": "blocked",
                        "blockers": [{"method": "qc", "reason": why}],
                        "input_rows": matched.n_observations,
                        "input_widths": widths,
                        "qc_workspace_bytes": qc_bytes,
                        "source_selection": source_selection,
                    },
                    inputs,
                )
                typer.echo(f"Blocked: {why} See {out / 'plan.json'}", err=True)
                raise typer.Exit(3)
            raise AnalysisBudgetError(why)
        mask_options = {
            "min_coverage": min_coverage,
            "max_gap_fraction": max_gap_fraction,
            "max_missing_fraction": max_missing_fraction,
        }
        panel_masks = (
            site_masks(paths["site_panel"], original, maps) if site_panel is not None else None
        )
        policy = QCPolicy.from_config(paths.get("qc_policy"))
        if clade_column and policy.clade_column is None:
            policy = replace(policy, clade_column=clade_column)
        audit = inspect_qc(
            matched,
            maps,
            table,
            column_options=mask_options,
            panel_masks=panel_masks,
            policy=policy,
        )
        if audit.blockers:
            if dry_run:
                publish_plan(
                    out,
                    {
                        "schema_version": 1,
                        "status": "blocked",
                        "blockers": [{"method": "qc", "reason": v} for v in audit.blockers],
                    },
                    inputs,
                    audit,
                )
                typer.echo(f"QC blocked: {out / 'plan.json'}", err=True)
                raise typer.Exit(3)
            raise QCBlocked(audit, inputs)
        filtered = audit.filtered
        maps = audit.maps
        qc = audit.qc
        clades = match_metadata(table, audit.row_indices, clade_column)
        supplied = match_metadata(table, audit.row_indices, weight_column, numeric=True)
        if weight_policy == "clade_balanced" and permutations and strata_column is None:
            strata_column = clade_column
        groups = match_metadata(table, audit.row_indices, strata_column)
        strata = tuple(groups.values()) if groups is not None else None
        if weight_policy == "clade_balanced" and permutations:
            group_clades = {}
            for key, g in groups.items():
                if g in group_clades and group_clades[g] != clades[key]:
                    raise ValueError("permutation strata must be nested within weighting clades")
                group_clades[g] = clades[key]
        plan = estimate_model(
            filtered.msa,
            maps,
            dca=dca,
            threebody=threebody,
            candidate_mode=candidate_mode,
            candidate_policy=paths.get("candidate_policy"),
            local_policy=paths.get("local_policy"),
            dca_design=design,
            threebody_model=threebody_model,
            require_reference=require_reference,
            max_triplets=max_triplets,
            triplet_candidates=paths.get("triplet_candidates"),
            permutations=permutations,
            min_valid=min_valid,
            min_valid_weight=min_valid_weight,
            all_pairs=all_pairs,
            max_pairs=max_pairs,
            weight_policy=weight_policy,
            weight_component=weight_component,
            supplied_weights=supplied,
            memory_budget_bytes=budget,
            seed=seed,
            identity_threshold=identity_threshold,
            min_overlap=min_overlap,
        )
        if named_null:
            plan["calibration"] = calibration_plan(
                named_null,
                plan,
                require_reference=require_reference,
                apc_universe=apc_universe,
                dca=dca,
                threebody=threebody,
                threebody_model=threebody_model,
                weight_policy=weight_policy,
            )
        if named_null and named_null.tree:
            from triplet_coevolution.validation.tree_model import tree_workspace

            ids = tuple(m.observation_id for m in filtered.msa.matches)
            plans = []
            for scenario in named_null.tree.scenarios:
                tree, details = scenario.load(ids)
                nodes = max(
                    len(t.parents) for t in (tree.values() if isinstance(tree, dict) else [tree])
                )
                tree_bytes = (
                    tree_workspace(nodes, len(ids), [m.n_sites for m in maps]) * named_null.workers
                )
                plans.append(
                    {
                        "label": scenario.label,
                        "scope": details["scope"],
                        "origin": scenario.origin,
                        "trees": details["topologies"],
                        "numeric_draw_bytes_all_workers": tree_bytes,
                    }
                )
            plan["calibration"]["tree_scenarios"] = plans
        plan["source_selection"] = source_selection
        if nested:
            group_design, settings_grid, _ = nested
            memberships, _, test_groups = group_design.partition(table, inputs["matches"]["sha256"])
            plan["conditional"] = {
                "source_rows": len(table.matches),
                "source_groups": len(set(memberships.values())),
                "planned_folds": 1 if test_groups else len(set(memberships.values())),
                "prespecified_l2": [v.l2 for v in settings_grid],
                "scope": "training-only feature selection and model-size validation per fold",
            }
        if clade_policy:
            from triplet_coevolution.validation.clade_stability import make_split_plan

            group_ids, src, test_groups = clade_policy.partition(table, inputs["matches"]["sha256"])
            splits = make_split_plan(
                tuple(m.observation_id for m in table.matches),
                group_ids,
                test_groups=test_groups,
                source=src,
            )
            plan["stability"] = {
                "source_observations": len(table.matches),
                "groups": len(set(group_ids.values())),
                "folds": len(splits.folds),
                "bootstrap_draws": clade_policy.options.get("bootstrap_draws", 0),
            }
        plan["inputs"] = {k: {"sha256": v["sha256"]} for k, v in inputs.items()}
        plan["input_alignments"] = {
            m.name: {
                "rows": m.n_sequences,
                "columns": m.n_sites,
                "bytes": paths[m.name].stat().st_size,
                "reference_id": axis.reference_id,
            }
            for m, axis in zip(original, maps, strict=True)
        }
        plan["settings"] = {
            "dca": dca,
            "threebody": threebody,
            "threebody_model": threebody_model if threebody else None,
            "ridge": (design.ridge if design else ridge)
            if dca and (not design or design.backend == "mf")
            else None,
            **({"dca_design": design.metadata()} if design else {}),
            "pseudocount": pseudocount if threebody and threebody_model == "local" else None,
            "global_l2": global_l2 if threebody and threebody_model == "global" else None,
            "require_reference": require_reference,
            "max_pairs": None if all_pairs else max_pairs,
            "max_triplets": max_triplets if threebody else None,
            "apc_universe": apc_universe,
            "mi_score": score,
            "dca_score": dca_score if dca else None,
            "threebody_score": threebody_score if threebody else None,
            "seed": seed,
            "legacy": legacy,
        }
        plan["output_directory"] = "."
        plan["estimates"]["qc_workspace_bytes"] = qc_bytes
        plan["estimates"]["numeric_workspace_bytes"] = max(
            plan["estimates"]["numeric_workspace_bytes"], qc_bytes
        )
        plan["contact_labels"] = None
        if contacts is not None:
            if legacy:
                from triplet_coevolution.validation.structure import read_contacts

                plan["contact_labels"] = len(read_contacts(paths["contacts"], maps))
            else:
                from triplet_coevolution.validation.benchmark import read_label_table

                plan["contact_labels"] = len(
                    read_label_table(paths["contacts"], {m.component: m.reference_id for m in maps})
                )
        if permutations:
            from triplet_coevolution.validation.phylogeny import exchangeability_groups

            try:
                exchange = exchangeability_groups(filtered.msa.n_observations, strata)
                plan["exchangeability"] = {
                    "column": strata_column,
                    "groups": len(exchange),
                    "movable_observations": sum(len(g) for g in exchange if len(g) > 1),
                }
            except ValueError as exc:
                plan["blockers"].append({"method": "permutations", "reason": str(exc)})
                plan["status"] = "blocked"
        for key, path in paths.items():
            if _file_stamp(path) != stamps[key]:
                raise ValueError(f"input changed while planning: {path}")
        plan["qc"] = {
            "input_observations": matched.n_observations,
            "retained_observations": filtered.msa.n_observations,
        }
        if dry_run:
            publish_plan(out, plan, inputs, audit)
            typer.echo(f"{plan['status']}: {out / 'plan.json'}")
            if plan["blockers"]:
                raise typer.Exit(3)
            return plan
        if plan["blockers"]:
            issue = plan["blockers"][0]
            if issue["method"] == "permutations":
                raise ValueError(issue["reason"])
            if issue["method"] in {"dca", "threebody"}:
                from triplet_coevolution.inference.advanced import ModelFitError

                raise ModelFitError(issue["method"], issue["reason"])
            raise AnalysisBudgetError(issue["reason"])
        weighting_options = {"identity_threshold": identity_threshold, "min_overlap": min_overlap}
        weight_settings = {
            "weight_policy": weight_policy,
            "weight_component": weight_component,
            "clade_column": clade_column,
            "weight_column": weight_column,
            "min_valid_weight": min_valid_weight,
            "memory_mib": memory_mib,
        }
        weights = sequence_weights(
            filtered.msa,
            **weighting_options,
            policy=weight_policy,
            component=weight_component,
            clades=clades if weight_policy == "clade_balanced" else None,
            supplied_weights=supplied,
            memory_budget_bytes=budget,
        )
        mi_kernel = "v1_scalar" if legacy else "batched"
        result = mi_with_apc_policy(
            triplet_mi(
                filtered.msa,
                weights=weights,
                min_valid=min_valid,
                min_valid_weight=min_valid_weight,
                kernel=mi_kernel,
            ),
            maps,
            apc_universe,
        )
        table_options = {
            "score": score,
            "max_pairs": None if all_pairs else max_pairs,
            "min_valid": min_valid,
            "require_reference": require_reference,
            "fmt": fmt,
            **({"min_valid_weight": min_valid_weight} if min_valid_weight else {}),
        }
        with _stage_run(out) as stage:
            write_json(stage / "analysis_replay.json", replay_arguments(options, out))
            qc_files = write_qc(stage, audit)
            write_weights(stage / "weights.tsv", weights)
            write_weighted_sites(stage / "site_weight_support.tsv", matched, audit, weights)
            qc_files["weighted_sites"] = "site_weight_support.tsv"
            write_json(stage / "plan.json", plan)
            tables = write_triplet_mi_tables(stage, result, *maps, **table_options)
            advanced = {}
            observed_dca = {}

            def collect_observed(_observed_mi, blocks):
                observed_dca.update(blocks or {})

            if dca or threebody or permutations or contacts is not None:
                from triplet_coevolution.inference.advanced import write_advanced_results

                advanced = write_advanced_results(
                    stage,
                    filtered.msa,
                    maps,
                    weights,
                    dca=dca,
                    threebody=threebody,
                    ridge=ridge,
                    pseudocount=pseudocount,
                    max_triplets=max_triplets,
                    threebody_model=threebody_model,
                    global_l2=global_l2,
                    candidate_mode=candidate_mode,
                    candidate_policy=paths.get("candidate_policy"),
                    local_policy=paths.get("local_policy"),
                    dca_design=design,
                    triplet_candidates=paths.get("triplet_candidates"),
                    permutations=permutations,
                    seed=seed,
                    strata=strata,
                    contacts=paths.get("contacts") if legacy else None,
                    mi_result=result,
                    dca_score=dca_score,
                    threebody_score=threebody_score,
                    apc_universe=apc_universe,
                    score_sink=collect_observed,
                    **table_options,
                )
                for key in (
                    "triplet_candidates",
                    "contacts",
                    "local_policy",
                    "candidate_policy",
                    "dca_policy",
                ):
                    if key in paths and _file_stamp(paths[key]) != stamps[key]:
                        raise ValueError(f"input changed while reading: {paths[key]}")
            figures = (
                save_mi_plots(stage / "figures", result, *maps, score=score, min_valid=min_valid)
                if plots
                else {}
            )
            if plots and observed_dca:
                figures["dca"] = save_dca_plot(
                    stage / "figures",
                    observed_dca,
                    maps,
                    score=dca_score,
                    min_valid=min_valid,
                    dca_design=design.metadata() if design else None,
                )
            summary = {
                "schema_version": 2,
                "software_version": __version__,
                "inputs": inputs,
                "components": {
                    component.name: {
                        "molecule_type": component.molecule.value,
                        "reference_id": mapping.reference_id,
                        **({"reference_row_id": row_id} if row_id is not None else {}),
                        "reference_start": mapping.start,
                        **({"reference_strand": "-"} if mapping.strand == "-" else {}),
                        "input_sequences": source_selection[component.name]["source_rows"],
                        "original_sites": component.n_sites,
                        "retained_sites": mapping.n_sites,
                        "retained_columns": mapping.original_positions,
                        "mean_coverage": stats.mean_coverage,
                        "mean_gap_fraction": stats.mean_gap_fraction,
                        "mean_missing_fraction": stats.mean_missing_fraction,
                    }
                    for component, mapping, stats, row_id in zip(
                        original, maps, qc.components, coordinate_rows
                    )
                },
                "settings": {**mask_options, **weighting_options, **table_options, "plots": plots},
                "matched_observations": filtered.msa.n_observations,
                "input_observations": matched.n_observations,
                "weighting": {
                    **weights.summary(),
                    **weight_settings,
                    "kish_effective_count": weights.kish,
                    "sum_weights": weights.n_eff,
                    "weight_per_retained_site": weights.n_eff / sum(m.n_sites for m in maps),
                },
                "qc_diagnostics": audit.summary,
                "source_selection": source_selection,
                "analysis_plan": "plan.json",
                "effective_depth": weights.n_eff,
                "advanced": advanced,
                "outputs": {
                    "diagnostics": qc_files,
                    "weights": "weights.tsv",
                    "plan": "plan.json",
                    "replay": "analysis_replay.json",
                    "tables": {
                        name.upper(): {"path": path.name, "rows": _row_count(path, fmt)}
                        for name, path in tables.items()
                    },
                    "figures": {
                        (name if name in ("triplet", "dca") else name.upper()): path.relative_to(
                            stage
                        ).as_posix()
                        for name, path in figures.items()
                    },
                },
            }
            if advanced:
                advanced["settings"]["strata_column"] = strata_column
            selected = [
                score,
                *([dca_score] if dca else []),
                *([threebody_score] if threebody else []),
            ]
            summary["ranking"] = {
                "legacy": legacy,
                "apc_universe": apc_universe,
                "mi_kernel": mi_kernel,
                "statistics": [
                    get_score(name).metadata(
                        apc_universe=apc_universe,
                        threebody_model=threebody_model,
                        dca_design=design.metadata() if design else None,
                    )
                    for name in selected
                ],
                "mi": score,
                "dca": dca_score if dca else None,
                "threebody": threebody_score if threebody else None,
                "mi_coverage": {
                    x.component + y.component: pair_coverage(
                        block,
                        x,
                        y,
                        score,
                        require_reference=require_reference,
                        min_valid=min_valid,
                        min_valid_weight=min_valid_weight,
                        max_pairs=table_options["max_pairs"],
                        apc_universe=apc_universe,
                    )
                    for block, x, y in (
                        (result.ab, maps[0], maps[1]),
                        (result.ac, maps[0], maps[2]),
                        (result.bc, maps[1], maps[2]),
                    )
                },
            }
            save_analysis_store(
                stage,
                summary,
                maps,
                result,
                observed_dca,
                observation_ids=[row.observation_id for row in filtered.msa.matches],
                labels=paths.get("contacts"),
                bootstrap=benchmark_bootstrap,
                seed=seed,
                top_k=benchmark_k or [10, 25, 50, 100],
            )
            if named_null:
                from triplet_coevolution.calibration import _check_checkpoint, _write_calibration

                _check_checkpoint(named_null, [out])
                if _file_stamp(paths["null_policy"]) != stamps["null_policy"]:
                    raise ValueError("null policy changed during analysis")
                calibrated = _write_calibration(
                    stage / "calibration",
                    stage,
                    named_null,
                    summary=summary,
                    live=(filtered.msa, weights, table),
                )
                triple_tests = [
                    k for k in named_null.statistics if get_score(k).method == "threebody"
                ]
                if triple_tests:
                    advanced["triple_calibration"] = {
                        "status": calibrated["runner"]["status"],
                        "null": named_null.null,
                        "statistics": triple_tests,
                        "draws": named_null.draws,
                        "path": "calibration/calibration.json",
                    }
                    if advanced.get("local_fit"):
                        advanced["local_fit"]["calibration"] = calibrated["question"]
                summary["calibration"] = {
                    "null": named_null.null,
                    "status": calibrated["runner"]["status"],
                    "draws": named_null.draws,
                    "families": calibrated["families"],
                }
                summary["outputs"]["calibration"] = "calibration/calibration.json"
                summary["outputs"]["calibration_report"] = "calibration/report.html"
            if nested:
                if _file_stamp(paths["conditional_policy"]) != stamps["conditional_policy"]:
                    raise ValueError("conditional policy changed during analysis")
                fitted = write_conditional(
                    stage / "conditional",
                    nested,
                    msa=matched,
                    maps=original_maps,
                    table=table,
                    digests={k: inputs[k]["sha256"] for k in ("A", "B", "C", "matches")},
                )
                summary["conditional"] = {
                    "split_sha256": fitted["split_sha256"],
                    "grid": fitted["grid"],
                }
                summary["outputs"]["conditional"] = "conditional/conditional_summary.json"
                summary["outputs"]["conditional_report"] = "conditional/report.html"
            if clade_policy:
                from triplet_coevolution.clade_workflow import _write_stability

                if _file_stamp(paths["stability_policy"]) != stamps["stability_policy"]:
                    raise ValueError("stability policy changed during analysis")
                _write_stability(
                    stage / "stability",
                    stage,
                    clade_policy,
                    summary=summary,
                    live=(matched, original_maps, table),
                )
                summary["clade_stability"] = plan["stability"]
                summary["outputs"]["stability"] = "stability/stability_summary.json"
                summary["outputs"]["stability_report"] = "stability/report.html"
            ranked = ranked_tables(summary)
            if plots and threebody:
                from triplet_coevolution.reporting.plots import save_triple_plot

                triple_table = next(table for table in ranked if table.label == "ABC")
                triple_plot = save_triple_plot(stage, triple_table)
                summary["outputs"]["figures"]["threebody"] = triple_plot.relative_to(
                    stage
                ).as_posix()
            summary["outputs"]["network"] = write_network(stage, ranked)
            summary["outputs"]["report"] = "report.html"
            write_report(stage, summary, ranked)
            ended = datetime.now(UTC)
            summary["runtime"] = {
                "start_utc": started.isoformat(),
                "end_utc": ended.isoformat(),
                "elapsed_seconds": (ended - started).total_seconds(),
            }
            summary["artifact_sha256"] = file_hashes(stage)
            (stage / SUMMARY_NAME).write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    except (ValueError, TypeError, OSError, ImportError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"matched obs: {filtered.msa.n_observations}")
    typer.echo(f"effective depth: {weights.n_eff:.2f}")
    typer.echo("kept cols: " + " ".join(f"{m.component}={m.n_sites}" for m in maps))
    for name, path in tables.items():
        typer.echo(f"{name}: {out / path.name}")
    if plots:
        typer.echo(f"plots: {out / 'figures'}")
    typer.echo(f"report: {out / 'report.html'}")
    typer.echo(f"summary: {out / SUMMARY_NAME}")


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
