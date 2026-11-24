"""Map and benchmark deposited physical coordinates against verified sequence references."""

import csv
from html import escape
from http.client import HTTPException
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution.io.score_store import file_digest
from triplet_coevolution.provenance import file_hashes, software_identity
from triplet_coevolution.reporting.html import (
    STYLE,
    _link,
    _table,
    benchmark_overview,
    benchmark_section,
)
from triplet_coevolution.score_views import read_scores, write_json
from triplet_coevolution.structure.assessment import (
    annotate_functional,
    matched_hard_controls,
    sample_distant,
    summarize_geometry,
)
from triplet_coevolution.structure.config import StructureConfig, load_structure_config
from triplet_coevolution.structure.distances import write_structure_tables
from triplet_coevolution.structure.mapping import map_components
from triplet_coevolution.structure.parser import ChoiceRequired, read_structure
from triplet_coevolution.structure.source import (
    coordinate_source,
    entry_matches,
    read_anchors,
    read_references,
    validate_ambiguous_source,
)
from triplet_coevolution.structure.viewer import write_viewers
from triplet_coevolution.validation.benchmark import benchmark_scores


def _report(directory, summary):
    bits = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Structure mapping</title><style>"
        + STYLE
        + "</style><main><h1>Structure mapping</h1>",
        f"<p>{escape(summary['source_kind'])} coordinates · {escape(summary['status'])}</p>",
    ]
    if "reason" in summary:
        bits.append("<p>" + escape(summary["reason"]) + "</p>")
    if "roles" in summary:
        bits.append(
            f"<h2>{escape(summary['entry_id'])}</h2><p>Assembly {escape(str(summary['assembly']))}, model {summary['model']}. Metric {escape(summary['definitions']['metric'])}, close at or below {summary['definitions']['close_inclusive']:g} Å.</p>"
        )
        bits.append(
            _table(
                ["Component", "Reference", "Assembly copy", "Mapping", "Identity", "Coverage"],
                [
                    (
                        c,
                        r["reference_id"],
                        r["copy_id"] or "—",
                        r["status"],
                        *("—" if r[k] is None else f"{r[k]:.1%}" for k in ("identity", "coverage")),
                    )
                    for c, r in summary["roles"].items()
                ],
            )
        )
        bits.append(
            _table(
                [
                    "Interface",
                    "Reference pairs",
                    "Observable",
                    "Close",
                    "Intermediate",
                    "Distant",
                    "Unresolved",
                    "Close / observable",
                ],
                [
                    (
                        p,
                        *(
                            r[k]
                            for k in (
                                "reference_pairs",
                                "observable",
                                "close",
                                "intermediate",
                                "distant",
                                "unresolved",
                            )
                        ),
                        "—" if r["prevalence"] is None else f"{r['prevalence']:.2%}",
                    )
                    for p, r in summary["interfaces"].items()
                ],
            )
        )
        bits.append(
            "<p>Distances depend on the chosen assembly, model, atom definition and resolved coordinates. A missing measurement has an unknown label. A three-edge configuration describes geometry and cannot establish energetic coupling.</p>"
        )
        for label, result in summary.get("benchmarks", {}).items():
            bits.append(
                "<h2>"
                + escape(label.replace("_", " ").title())
                + "</h2>"
                + benchmark_section(result["overview"])
            )
    else:
        bits.append(
            _table(
                ["Available deposited models", "Available biological assemblies"],
                [(str(summary["available_models"]), str(summary["available_assemblies"]))],
            )
        )
    files = sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.name != "structure_summary.json"
    )
    bits += [
        "<h2>Files</h2><ul>",
        *("<li>" + _link(name, name) + "</li>" for name in ["structure_summary.json", *files]),
        "</ul></main></html>",
    ]
    (directory / "report.html").write_text("".join(bits), encoding="utf-8")


def _score_input(config, references):
    if config.scores is None:
        if config.viewer_score:
            raise ValueError("viewer_score requires a verified score artifact")
        return None, None, None
    path = config.scores
    if path.is_dir():
        path = path / (
            "scores/score_manifest.json"
            if (path / "scores/score_manifest.json").is_file()
            else "score_manifest.json"
        )
    digest = file_digest(path)
    store = read_scores(path)
    if file_digest(path) != digest:
        raise ValueError("score manifest changed while loading")
    for axis in store.maps:
        ref = references[axis.component]
        kind = store.provenance.get("components", {}).get(axis.component, {}).get("molecule_type")
        if axis.strand != ref.strand or (kind and kind != ref.molecule):
            raise ValueError(f"score artifact molecule/strand disagrees for {axis.component}")
    ranking = store.provenance.get("ranking", {})
    statistic = (
        config.viewer_score or ranking.get("dca") or ranking.get("mi") or next(iter(store.scores))
    )
    if statistic not in store.scores:
        raise ValueError(f"unknown or unavailable viewer_score: {statistic}")
    return (
        store,
        statistic,
        {"sha256": digest, "statistics": sorted(store.scores), "viewer_score": statistic},
    )


def _run(stage, config, download):
    config_digest = file_digest(config.path)
    references, ref_inputs = read_references(config)
    inputs = {
        "config": {"file": config.path.name, "sha256": config_digest},
        "references": ref_inputs,
    }
    for key, path in (
        ("explicit_map", config.explicit_map),
        ("triples", config.triples),
        ("functional_labels", config.functional_labels),
        ("exposure_bins", config.exposure_bins),
    ):
        if path:
            inputs[key] = {"file": path.name, "sha256": file_digest(path)}
    store, statistic, score_input = _score_input(config, references)
    if score_input:
        inputs["scores"] = score_input
    anchors = read_anchors(config.explicit_map)
    copies = {r.component: r.copy_id for r in config.references if r.copy_id}
    with coordinate_source(config, download=download) as (path, info):
        inputs["coordinates"] = info
        try:
            structure = read_structure(path, assembly=config.assembly, model=config.model)
        except ChoiceRequired as exc:
            if file_digest(path) != info["sha256"]:
                raise ValueError("coordinate input changed during parsing") from exc
            if config.pdb_id:
                validate_ambiguous_source(path, config.pdb_id)
            summary = {
                "schema_version": 1,
                "status": "selection_required",
                "reason": str(exc),
                "available_models": exc.available_models,
                "available_assemblies": exc.available_assemblies,
            }
            structure = None
        if structure:
            if structure.sha256 != info["sha256"]:
                raise ValueError("coordinate input changed during parsing")
            if config.pdb_id and not entry_matches(config.pdb_id, structure.entry_id):
                raise ValueError("coordinate entry ID does not match the requested public PDB ID")
    if structure:
        mapping = map_components(
            references,
            structure,
            chain_ids=copies,
            explicit_map=anchors,
            min_identity=config.min_identity,
            min_coverage=config.min_coverage,
        )
        summary = write_structure_tables(
            stage,
            mapping,
            metric=config.metric,
            close=config.close,
            distant=config.distant,
            distant_metric=config.distant_metric,
            score_store=store,
            source_kind=config.source_kind,
        )
        deposited_kind = summary["coordinate_evidence"]
        if (deposited_kind, config.source_kind) in {
            ("predicted", "experimental"),
            ("experimental", "predicted"),
        }:
            raise ValueError(
                f"declared source_kind conflicts with deposited method {structure.method}"
            )
        candidates, summary["sensitivity"] = summarize_geometry(
            stage, config, mapping, statistic=statistic
        )
        summary["viewer"] = write_viewers(
            stage / "viewer", mapping, candidates, statistic=statistic
        )
        if config.functional_labels:
            summary["functional_labels"] = annotate_functional(
                stage, config.functional_labels, mapping
            )
        tests = {"all_measurable": "labels.tsv", "close_vs_distant": "distant_labels.tsv"}
        if config.negative_ratio is not None:
            sampled = sample_distant(
                stage, config.negative_ratio, config.seed, summary["interfaces"]
            )
            summary["sensitivity"]["negative_sample"] = sampled
            tests["sampled_controls"] = sampled["path"]
        if config.exposure_bins:
            hard = matched_hard_controls(stage, config, mapping)
            summary["sensitivity"]["exposure_matched_controls"] = hard
            tests["exposure_matched_controls"] = hard["path"]
        summary["benchmarks"] = {}
        if store:
            for name, label_file in tests.items():
                result = benchmark_scores(store, stage / label_file)
                benchmark_file = "benchmark_" + name + ".json"
                write_json(stage / benchmark_file, result)
                summary["benchmarks"][name] = {
                    "path": benchmark_file,
                    "labels": label_file,
                    "overview": benchmark_overview(result),
                }
                del result
        summary["status"] = (
            "complete"
            if all(r.chain for r in mapping.roles.values())
            and all(v["observable"] for v in summary["interfaces"].values())
            else "partial"
        )
        if summary["status"] == "partial":
            summary["reason"] = (
                "One or more interfaces lack a uniquely mapped copy or a measurable pair for this metric. Inspect the mapping candidates and the coordinate audit."
            )
    if file_digest(config.path) != config_digest:
        raise ValueError("structure configuration changed during the run")
    for key, path in (
        ("explicit_map", config.explicit_map),
        ("triples", config.triples),
        ("functional_labels", config.functional_labels),
        ("exposure_bins", config.exposure_bins),
    ):
        if path and file_digest(path) != inputs[key]["sha256"]:
            raise ValueError(f"{key} changed during the run")
    summary.update(
        source_kind=config.source_kind,
        inputs=inputs,
        software=software_identity(),
        report="report.html",
    )
    _report(stage, summary)
    summary["output_sha256"] = file_hashes(stage, exclude=("structure_summary.json",))
    write_json(stage / "structure_summary.json", summary)
    return summary


def run_structure(config: Path | StructureConfig, out: Path, *, download: bool = False) -> dict:
    """Run an atomic structural mapping and distance evaluation on explicit references."""
    from triplet_coevolution.cli import _stage_run

    if not isinstance(config, StructureConfig):
        config = load_structure_config(config)
    with _stage_run(Path(out)) as stage:
        return _run(stage, config, download)


def structure_map(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Versioned JSON with coordinate source and three reference FASTA identities.",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="New or empty output directory.")],
    download: Annotated[
        bool,
        typer.Option(
            "--download",
            help="Permit retrieval of the explicit public PDB ID into the local cache.",
        ),
    ] = False,
):
    """Map three references to a deposited structure and evaluate inter-component distances."""
    try:
        summary = run_structure(config, out, download=download)
    except (ValueError, TypeError, OSError, HTTPException, csv.Error, EOFError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{summary['status']}: {out / 'report.html'}")
    if summary["status"] != "complete":
        raise typer.Exit(3)
