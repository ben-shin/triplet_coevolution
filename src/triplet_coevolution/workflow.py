"""Prepare matched alignments from evidence, or run a local search-to-report workflow."""

import csv
import hashlib
import json
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Annotated

import Bio
import typer

from triplet_coevolution import __version__
from triplet_coevolution.config import COMPONENTS, load_config
from triplet_coevolution.core.coordinates import build_reference_map
from triplet_coevolution.io.metadata import REQUIRED_COLUMNS
from triplet_coevolution.io.profiles import read_reference
from triplet_coevolution.matching.orthology import read_evidence
from triplet_coevolution.matching.triplets import match_groups
from triplet_coevolution.msa.qc import component_qc

ConfigOption = Annotated[
    Path, typer.Option("--config", help="Workflow JSON. Input paths are relative to its directory.")
]
OutOption = Annotated[Path, typer.Option("--out", help="New or empty output directory.")]


def _write_json(path, value):
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_fasta(path, msa):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for seq_id, seq in zip(msa.ids, msa.sequences):
            handle.write(f">{seq_id}\n")
            for start in range(0, len(seq), 80):
                handle.write(seq[start : start + 80] + "\n")


def _inputs(cfg, config):
    paths = {"config": config.absolute(), "evidence": cfg["evidence"]}
    for name, comp in cfg["components"].items():
        for key in (
            "database",
            "reference",
            "profile",
            "alignment",
            "hit_table",
            "reference_hits",
            "alignment_profile",
        ):
            if key in comp:
                paths[f"{name}.{key}"] = comp[key]
    for key, value in cfg.get("analysis", {}).items():
        if isinstance(value, Path):
            paths[f"analysis.{key}"] = value
    return paths


def _provenance(paths):
    result = {}
    for label, path in paths.items():
        with path.open("rb") as handle:
            result[label] = {
                "path": str(path),
                "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
            }
    return result


def _write_prepared(directory, components, cfg, available, coordinate_rows, *, inventories=None):
    directory.mkdir(exist_ok=True)
    groups, metadata, extra = read_evidence(cfg["evidence"])
    declared = {k: set() for k in COMPONENTS}
    if inventories:
        for rows in groups.values():
            for row in rows:
                declared[row.component].add(row.sequence_id)
        available = {
            k: {i for i in ids if inventories[k].contains(i)} for k, ids in declared.items()
        }
        for k, ref in coordinate_rows.items():
            if ref != cfg["components"][k]["reference_id"]:
                available[k].discard(ref)
    matches, qc = match_groups(available, groups, cfg["paralogs"])
    with (directory / "matches.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, (*REQUIRED_COLUMNS, *extra), delimiter="\t")
        writer.writeheader()
        for row in matches:
            writer.writerow({**asdict(row), **metadata[row.observation_id]})
    qc["ready_for_analysis"] = bool(matches)
    qc["alignments"] = {}
    for name in COMPONENTS:
        comp = cfg["components"][name]
        ref = comp["reference_id"]
        ref_row = coordinate_rows[name]
        ids = [getattr(row, f"{name.lower()}_id") for row in matches]
        chosen = ids if ref_row in ids else [ref_row, *ids]
        if inventories:
            inv = inventories[name]
            if not inv.contains(ref_row):
                raise ValueError(f"{name} reference ID {ref_row!r} not found in alignment")
            filename = name + {"fasta": ".fasta", "stockholm": ".sto", "a3m": ".a3m"}[inv.format]
            result = inv.select(chosen, output=directory / filename)
            result.reference_map(
                ref,
                comp["reference_start"],
                row_id=ref_row,
                strand=comp.get("reference_strand", "+"),
            )
            selected = result.msa
            source_count = inv.n_rows
            source_width = inv.n_columns
            count = 0
            sample = []
            unassigned = directory / f"{name}_unassigned.tsv"
            with unassigned.open("w", encoding="utf8", newline="") as stream:
                writer = csv.writer(stream, delimiter="	", lineterminator="\n")
                writer.writerow(["sequence_id"])
                for seq_id in inv.ids(sorted_names=True):
                    if seq_id in declared[name] or (seq_id == ref_row and ref_row != ref):
                        continue
                    count += 1
                    writer.writerow([seq_id])
                    if len(sample) < 1000:
                        sample.append(seq_id)
            qc["unassigned_sequences"][name] = sample
            qc.setdefault("unassigned_sequence_counts", {})[name] = count
            if count > len(sample):
                qc.setdefault("complete_unassigned_tables", {})[name] = unassigned.name
            else:
                unassigned.unlink()
        else:
            filename = f"{name}.fasta"
            msa = components[name]
            build_reference_map(msa, ref, comp["reference_start"], row_id=ref_row)
            selected = msa.take_ids(chosen)
            source_count, source_width = msa.n_sequences, msa.n_sites
            _write_fasta(directory / filename, selected)
        stats = component_qc(selected.take_ids(ids)) if ids else None
        qc["alignments"][name] = {
            "reference_id": ref,
            "reference_row_id": ref_row,
            "reference_start": comp["reference_start"],
            **({"reference_strand": "-"} if comp.get("reference_strand") == "-" else {}),
            "molecule": comp["molecule"],
            **({"selection": result.provenance} if inventories else {}),
            "file": filename,
            "sites": source_width,
            "input_sequences": source_count,
            "exported_sequences": selected.n_sequences,
            "matched_mean_coverage": stats.mean_coverage if stats else None,
            "matched_mean_gap_fraction": stats.mean_gap_fraction if stats else None,
            "matched_mean_missing_fraction": stats.mean_missing_fraction if stats else None,
        }
    _write_json(directory / "matching_qc.json", qc)
    return qc


def _legacy_discovery(name, comp, reference, directory):
    """Replay v1 source-level scoring and retain the engine's alternative reported domains."""
    from triplet_coevolution.msa.builders import assemble_msa
    from triplet_coevolution.search.discovery import DiscoveryResult
    from triplet_coevolution.search.retrieve import retrieve
    from triplet_coevolution.search.shared import program_versions

    directory.mkdir()
    audit_path = directory / "hit_audit.tsv"
    with audit_path.open("w", encoding="utf8", newline="") as handle:
        fields = (
            "seq_id",
            "first",
            "last",
            "strand",
            "score",
            "evalue",
            "model_first",
            "model_last",
            "selected_legacy",
            "reason",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        hits, _ = retrieve(comp, reference, audit_sink=writer.writerow)
    programs = program_versions(comp)
    msa = assemble_msa(
        name,
        comp["molecule"],
        comp["reference_id"],
        reference,
        {k: h.placement.row(len(reference)) for k, h in hits.items()},
    )
    manifest = {
        "schema": "triplet.discovery.v1.compat",
        "role": name,
        "backend": comp["backend"],
        "selection": "highest score per source ID, then E-value, strand and source interval",
        "alignment_scope": "all selected source IDs, including IDs outside the evidence table",
        "reported_scope": "parsed eligible intervals before source-level selection",
        "accepted_source_ids": len(hits),
        "audit": "hit_audit.tsv",
        "audit_sha256": _provenance({"audit": audit_path})["audit"]["sha256"],
        "reference_row_id": msa.ids[0],
        "programs": programs,
    }
    _write_json(directory / "discovery.json", manifest)
    return DiscoveryResult(hits, msa, msa.ids[0], manifest, programs)


def _run_searches(stage, cfg):
    from triplet_coevolution.search.discovery import discover_component

    groups, _, _ = read_evidence(cfg["evidence"])
    declared = {name: set() for name in COMPONENTS}
    for group in groups.values():
        for row in group:
            declared[row.component].add(row.sequence_id)
    search_dir = stage / "search"
    search_dir.mkdir()
    components = {}
    available = {}
    programs = {}
    discoveries = {}
    coordinate_rows = {}
    for name in COMPONENTS:
        comp = cfg["components"][name]
        reference = read_reference(comp)
        directory = search_dir / f"{name}_discovery"
        if comp["discovery_version"] == 1:
            found = _legacy_discovery(name, comp, reference, directory)
        else:
            found = discover_component(
                name,
                comp,
                reference,
                directory,
                wanted_ids=declared[name],
                cache_dir=comp.get("cache_dir"),
            )
        hits = found.hits
        programs[name] = found.programs
        discoveries[name] = found.manifest
        coordinate_rows[name] = found.reference_row_id
        with (search_dir / f"{name}_hits.tsv").open("w", encoding="utf-8", newline="") as handle:
            source_fields = (
                "seq_id",
                "first",
                "last",
                "strand",
                "score",
                "evalue",
                "identity",
                "coverage",
                "model_strand",
                "model_first",
                "model_last",
                "projection_method",
            )
            fields = (
                "row_id",
                *source_fields,
                "reference_first",
                "reference_last",
                "reference_pos_first",
                "reference_pos_last",
            )
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for key in sorted(hits):
                hit = hits[key]
                placement = (
                    hit.placement
                    if comp["discovery_version"] == 1 or key in declared[name]
                    else None
                )
                offset = comp["reference_start"] - 1
                writer.writerow(
                    {
                        "row_id": key,
                        **{field: getattr(hit, field, None) for field in source_fields},
                        "reference_first": placement.first if placement else None,
                        "reference_last": placement.last if placement else None,
                        "reference_pos_first": placement.first + offset if placement else None,
                        "reference_pos_last": placement.last + offset if placement else None,
                    }
                )
        components[name] = found.msa
        _write_fasta(search_dir / f"{name}.fasta", components[name])
        available[name] = set(hits)
    return components, available, programs, discoveries, coordinate_rows


def _analysis_inputs(prepared, cfg, coordinate_rows, qc):
    kwargs = {"matches": prepared / "matches.tsv"}
    for name in COMPONENTS:
        comp = cfg["components"][name]
        key = name.lower()
        kwargs.update(
            {
                key: prepared / qc["alignments"][name]["file"],
                f"{key}_ref": comp["reference_id"],
                f"{key}_type": comp["molecule"],
                f"{key}_start": comp["reference_start"],
            }
        )
        if comp.get("reference_strand") == "-":
            kwargs[f"{key}_strand"] = "-"
        if coordinate_rows[name] != comp["reference_id"]:
            kwargs[f"{key}_ref_row"] = coordinate_rows[name]
    return {**kwargs, **cfg.get("analysis", {})}


def _write_analysis_args(prepared, kwargs, filename="analyze_args.json"):
    """Record native CLI arguments for replay from the prepared directory."""
    args = ["analyze"]
    for key, value in kwargs.items():
        option = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(option)
        elif isinstance(value, list):
            for item in value:
                args.extend([option, str(item)])
        else:
            arg = str(value)
            if isinstance(value, Path) and value.absolute().is_relative_to(prepared.absolute()):
                arg = value.absolute().relative_to(prepared.absolute()).as_posix()
            args.extend([option, arg])
    _write_json(prepared / filename, args)


def _analyze(stage, out, kwargs):
    from triplet_coevolution.cli import analyze

    with redirect_stdout(StringIO()):
        analyze(**kwargs, out=stage / "analysis")
    # The outer publication moves generated inputs to their final location.
    path = stage / "analysis" / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    for record in summary["inputs"].values():
        source = Path(record["path"])
        if source.is_relative_to(stage.absolute()):
            record["path"] = str(out.absolute() / source.relative_to(stage.absolute()))
    _write_json(path, summary)
    return summary


def _workflow(config, out, mode, plots=None):
    from triplet_coevolution.cli import _check_output, _stage_run

    config = config.expanduser().absolute()
    out = out.expanduser()
    _check_output(out)
    config_id = _provenance({"config": config})["config"]
    cfg = load_config(config, mode)
    if plots is not None:
        cfg["analysis"]["plots"] = plots
    paths = _inputs(cfg, config)
    identities = _provenance(paths)
    if identities["config"] != config_id:
        raise ValueError("configuration changed while reading")
    with _stage_run(out) as stage:
        if mode == "run":
            components, available, programs, discoveries, coordinate_rows = _run_searches(
                stage, cfg
            )
            prepared = stage / "prepared"
            qc = _write_prepared(prepared, components, cfg, available, coordinate_rows)
        else:
            from triplet_coevolution.io.alignments import open_msa

            prepared = stage
            coordinate_rows = {
                n: c.get("reference_row_id", c["reference_id"])
                for n, c in cfg["components"].items()
            }
            with ExitStack() as stack:
                inventories = {
                    n: stack.enter_context(
                        open_msa(c["alignment"], n, c["molecule"], c.get("format"))
                    )
                    for n, c in cfg["components"].items()
                }
                qc = _write_prepared(
                    prepared, None, cfg, None, coordinate_rows, inventories=inventories
                )
        analysis_args = _analysis_inputs(prepared, cfg, coordinate_rows, qc)
        _write_analysis_args(prepared, analysis_args)
        summary = {
            "schema_version": cfg["schema_version"],
            "software_version": __version__,
            "biopython_version": Bio.__version__,
            "mode": mode,
            "inputs": identities,
            "paralogs": cfg["paralogs"],
            "matched_groups": qc["matched_groups"],
            "excluded_groups": qc["excluded_groups"],
            "prepared": prepared.relative_to(stage).as_posix(),
            "matching_qc": (prepared / "matching_qc.json").relative_to(stage).as_posix(),
            "analyze_args": (prepared / "analyze_args.json").relative_to(stage).as_posix(),
        }
        if mode == "run":
            summary["search"] = {
                name: {
                    **{k: v for k, v in cfg["components"][name].items() if not isinstance(v, Path)},
                    "hits": len(available[name]),
                    "executables": programs[name],
                    "discovery": discoveries[name],
                    "manifest": f"search/{name}_discovery/discovery.json",
                    "table": f"search/{name}_hits.tsv",
                }
                for name in COMPONENTS
            }
            if qc["ready_for_analysis"]:
                from triplet_coevolution.reporting.html import attach_workflow_report

                _analyze(stage, out, analysis_args)
                summary.update(
                    status="complete",
                    analysis_summary="analysis/run_summary.json",
                    report="analysis/report.html",
                )
                attach_workflow_report(stage / "analysis", summary, qc)
            else:
                from triplet_coevolution.reporting.html import write_unmatched_report

                detail = "; ".join(
                    f"{row['group_id']}: {', '.join(row['reasons'])}" for row in qc["groups"][:4]
                )
                if cfg["schema_version"] == 1:
                    raise ValueError(
                        f"no complete unambiguous triplets after search and matching ({detail})"
                    )
                summary.update(
                    status="no_matches",
                    report="report.html",
                    reason="No complete unambiguous triplets after search and matching.",
                )
                write_unmatched_report(stage, summary)
        if _provenance(paths) != identities:
            raise ValueError("an input changed during the workflow; rerun with stable files")
        _write_json(stage / "workflow_summary.json", summary)
    typer.echo(f"Matched {qc['matched_groups']} groups; excluded {qc['excluded_groups']}.")
    typer.echo(f"Matching QC: {out / summary['matching_qc']}")
    if mode == "run":
        if summary.get("analysis_summary"):
            typer.echo(f"Analysis: {out / summary['analysis_summary']}")
        typer.echo(f"Report: {out / summary['report']}")
        if summary["status"] == "no_matches":
            raise typer.Exit(3)
    elif not qc["ready_for_analysis"]:
        typer.echo("No complete triplets. Inspect the matching QC for excluded candidates.")


def prepare(config: ConfigOption, out: OutOption = Path("prepared")) -> None:
    """Match three unpaired alignments using explicit genome or orthology evidence."""
    try:
        _workflow(config, out, "prepare")
    except (ValueError, TypeError, OSError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


def run(
    config: Annotated[
        Path | None, typer.Option("--config", help="Raw search workflow JSON.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="New or empty output directory.")
    ] = None,
    plots: Annotated[
        bool | None,
        typer.Option(
            "--plots/--no-plots", help="Override the PNG setting in the analysis configuration."
        ),
    ] = None,
    directory: Annotated[
        Path | None,
        typer.Option(
            "--input", help="Three named family folders under this directory (default: input)."
        ),
    ] = None,
    online: Annotated[
        bool | None,
        typer.Option(
            "--online/--offline",
            help="Allow identifier lookups (default) or use only local evidence and caches.",
        ),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Directory for reusable public metadata snapshots."),
    ] = None,
) -> None:
    """Match three families, score their alignments, benchmark and report without prompts."""
    try:
        if config is not None:
            if directory is not None or online is not None or cache_dir is not None:
                raise ValueError(
                    "--input, --online/--offline and --cache-dir apply to family folders"
                )
            _workflow(config, out or Path("results"), "run", plots)
        else:
            from triplet_coevolution.automatic_run import run_families

            status = run_families(directory or Path("input"), out, online, cache_dir, plots)
            if status:
                raise typer.Exit(status)
    except (ValueError, TypeError, OSError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
