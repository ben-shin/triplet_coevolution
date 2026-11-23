"""Prepare evidence and synchronized, coordinate-preserving subsets of family alignments."""

import csv
import sqlite3
from collections import defaultdict
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Annotated

import Bio
import typer

from triplet_coevolution import __version__
from triplet_coevolution.config import COMPONENTS
from triplet_coevolution.core.alphabets import encode_sequence
from triplet_coevolution.core.coordinates import reference_positions
from triplet_coevolution.family_config import family_config, manifest_targets
from triplet_coevolution.io.metadata import REQUIRED_COLUMNS, TripletTable, write_triplet_table
from triplet_coevolution.matching.orthology import read_evidence
from triplet_coevolution.matching.triplets import match_groups
from triplet_coevolution.workflow import _analyze, _write_analysis_args, _write_json


def _candidates(inventory, comp, manifest_ids):
    from triplet_coevolution.evidence import EvidenceRow

    if comp["scope"] == "family":
        yield from inventory.rows()
        return
    requested = comp["targets"] if comp["scope"] == "targets" else manifest_ids
    selectors = defaultdict(list)
    for token in requested:
        found = False
        for row in inventory.find(token):
            selectors[row.row_id].append(token)
            found = True
        if not found:
            yield EvidenceRow(token, present=False, selectors=(token,))
    for row_id, tokens in selectors.items():
        row = inventory.row(row_id)
        yield EvidenceRow(
            row.row_id,
            row.accession,
            row.start,
            row.end,
            selectors=tuple(tokens),
            annotations=row.annotations,
        )


def _reference(inventory, comp):
    if "reference_id" not in comp:
        return None
    if "reference_row_id" in comp:
        row = inventory.row(comp["reference_row_id"])
    else:
        candidates = iter(inventory.find(comp["reference_id"]))
        row = next(candidates, None)
        if next(candidates, None) is not None:
            raise ValueError(
                f"{comp['name']}: ambiguous reference {comp['reference_id']!r}; "
                "specify its exact domain in reference_row_id"
            )
    if row is None:
        raise ValueError(f"{comp['name']}: reference row not found for {comp['reference_id']!r}")
    if row.start is not None and comp.get("reference_start", row.start) != row.start:
        raise ValueError(
            f"{comp['name']}: reference_start disagrees with the Stockholm domain "
            f"start {row.start} in {row.row_id}"
        )
    return row


def _matches(metadata, prepared, counts):
    path = prepared / "matches.tsv"
    if not counts["selected_rows"]:
        path.write_text("\t".join(REQUIRED_COLUMNS) + "\n", encoding="utf-8")
        return ()
    groups, annotations, extras = read_evidence(metadata / "selected_evidence.tsv")
    available = {
        name: {c.sequence_id for rows in groups.values() for c in rows if c.component == name}
        for name in COMPONENTS
    }
    selected, qc = match_groups(available, groups, "exclude")
    if qc["excluded_groups"] or len(selected) != counts["resolved_groups"]:
        raise ValueError("selected evidence is inconsistent with the resolved group audit")
    table = TripletTable(
        tuple(selected), extras, tuple(annotations[r.observation_id] for r in selected)
    )
    write_triplet_table(path, table)
    return table.matches


def _matched_subset(replay, output, extra_reference, count):
    """Remove the extra coordinate row from the normalized, already-extracted Stockholm."""
    with (
        replay.open(encoding="utf-8") as source,
        output.open("x", encoding="utf-8", newline="\n") as dest,
    ):
        for line in source:
            fields = line.split(maxsplit=2)
            first = fields[0]
            if first == extra_reference or (
                first in ("#=GS", "#=GR") and fields[1] == extra_reference
            ):
                continue
            if first == "#=GF":
                if fields[1] in ("NH", "TN"):
                    continue
                if fields[1] == "SQ":
                    line = f"#=GF SQ {count}\n"
            dest.write(line)


def _sequence_validator(component, molecule):
    """Validate selected fragments in source order with the scoring alphabet."""
    if molecule is None:
        return None
    consumed = defaultdict(int)

    def validate(row_id, fragment):
        try:
            if not fragment.isascii():
                raise ValueError(f"non-ASCII residue in {molecule} sequence")
            encode_sequence(fragment, molecule)
        except ValueError as exc:
            raise ValueError(
                f"component {component}, row {row_id!r}, fragment beginning at alignment column "
                f"{consumed[row_id] + 1}: {exc}"
            ) from exc
        consumed[row_id] += len(fragment)

    return validate


def _extract(inventories, references, cfg, prepared, matches, *, automatic=False):
    components = {}
    kwargs = {"matches": prepared / "matches.tsv"}
    missing = []
    with (prepared / "reference_coordinates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "component",
                "component_name",
                "reference_id",
                "row_id",
                "original_column",
                "reference_position",
                "symbol",
            )
        )
        for name, inventory in inventories.items():
            comp = cfg["components"][name]
            ref = references[name]
            ids = [getattr(row, f"{name.lower()}_id") for row in matches]
            rows = list(dict.fromkeys([*ids, *([ref.row_id] if ref else [])]))
            output = prepared / f"{name}.sto"
            separate_reference = ref is not None and ref.row_id not in ids
            replay = prepared / f"{name}.with_reference.sto" if separate_reference else output
            if rows:
                gapped = inventory.extract(
                    rows,
                    replay,
                    reference_row_id=ref.row_id if ref else None,
                    validate_fragment=_sequence_validator(name, comp["molecule"]),
                    **({"transcribe_t": True} if comp.get("transcribe_t") else {}),
                )
                if separate_reference and ids:
                    _matched_subset(replay, output, ref.row_id, len(ids))
            key = name.lower()
            info = {
                "name": comp["name"],
                "family_id": inventory.family_id,
                "accession": inventory.accession,
                "molecule": comp["molecule"],
                "source_rows": inventory.n_rows,
                "source_columns": inventory.n_columns,
                "selection": comp["scope"],
                "exported_rows": len(ids),
                "alignment": f"prepared/{output.name}" if ids else None,
                "replay_alignment": f"prepared/{replay.name}" if rows else None,
                "reference_id": comp.get("reference_id"),
                "reference_row_id": ref.row_id if ref else None,
                "reference_start": None,
                "reference_strand": None,
            }
            if ref:
                start = comp.get("reference_start", ref.start or 1)
                info["reference_start"] = start
                orientation = "-" if ref.start is not None and ref.start > ref.end else "+"
                strand = comp.get("reference_strand", orientation)
                if ref.start != ref.end and ref.start is not None and strand != orientation:
                    raise ValueError(
                        f"{name}: reference_strand disagrees with interval {ref.row_id}"
                    )
                if strand == "-" and comp["molecule"] == "protein":
                    raise ValueError(
                        f"{name}: reverse reference numbering requires a nucleotide component"
                    )
                info["reference_strand"] = strand
                positions = reference_positions(gapped, start, strand=strand)
                if all(position is None for position in positions):
                    missing.append(f"{name} ({comp['name']}): reference has no occupied positions")
                if (
                    ref.end is not None
                    and next((p for p in reversed(positions) if p is not None), None) != ref.end
                ):
                    raise ValueError(
                        f"{name}: reference span disagrees with occupied residues in {ref.row_id}"
                    )
                writer.writerows(
                    (name, comp["name"], comp["reference_id"], ref.row_id, index, position, symbol)
                    for index, (position, symbol) in enumerate(zip(positions, gapped), start=1)
                )
                kwargs.update(
                    {key: replay, f"{key}_ref": comp["reference_id"], f"{key}_start": start}
                )
                if ref.row_id != comp["reference_id"]:
                    kwargs[f"{key}_ref_row"] = ref.row_id
                if strand == "-":
                    kwargs[f"{key}_strand"] = strand
            else:
                if matches or not automatic:
                    reason = "no usable matched coordinate row" if automatic else "set reference_id"
                    missing.append(f"{name} ({comp['name']}): {reason}")
            if comp["molecule"] is None:
                reason = (
                    "scoring molecule could not be inferred" if automatic else "declare molecule"
                )
                missing.append(f"{name} ({comp['name']}): {reason}")
            else:
                kwargs[f"{key}_type"] = comp["molecule"]
            components[name] = info
    if not matches:
        missing.append("no complete unambiguous groups")
    return components, {**kwargs, **cfg["analysis"]}, missing


def _cache(cfg, metadata):
    path = cfg.get("cache")
    if not cfg["uniprot"]:
        return path
    target = metadata / "uniprot.sqlite"
    if path is not None:
        with (
            closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as src,
            closing(sqlite3.connect(target)) as dest,
        ):
            src.backup(dest)
    return target


def _prepare(
    config,
    directory,
    out,
    score,
    uniprot,
    molecule,
    *,
    automatic=False,
    cache_directory=None,
    plots=None,
):
    from triplet_coevolution.cli import _check_output, _file_stamp, _input_identity, _stage_run
    from triplet_coevolution.evidence import EvidenceRow, generate_evidence
    from triplet_coevolution.io.stockholm import open_stockholm

    out = out.expanduser().absolute()
    _check_output(out)
    options = {"uniprot": uniprot, "default_molecule": molecule, "automatic": automatic}
    cfg, paths, order = family_config(config, directory, **options)
    if directory is not None or config is None:
        folder = (directory or Path("input")).resolve()
        if out.resolve().is_relative_to(folder):
            raise ValueError("choose --out outside the three-folder input project")
    stamps = {path: _file_stamp(path) for path in set(paths.values())}
    identities = {path: _input_identity(path) for path in stamps}
    if (cfg, paths, order) != family_config(config, directory, **options):
        raise ValueError("project settings changed while reading; retry with stable inputs")
    if any(
        Path(str(cache) + suffix).exists() and Path(str(cache) + suffix).stat().st_size
        for key, cache in paths.items()
        if key in ("cache", "nucleotide_cache")
        for suffix in ("-wal", "-journal")
    ):
        raise ValueError("use a closed, checkpointed SQLite cache snapshot as input")
    manifest_ids = (
        manifest_targets(cfg["manifest"])
        if any(c["scope"] == "manifest" for c in cfg["components"].values())
        else {}
    )
    with _stage_run(out) as stage:
        metadata, prepared = stage / "metadata", stage / "prepared"
        metadata.mkdir()
        prepared.mkdir()
        with ExitStack() as stack:
            inventories = {}
            selections = {}
            references = {}
            decisions = {}
            source_choices = {}
            reference_choices = {}
            for name, comp in cfg["components"].items():
                if automatic:
                    from triplet_coevolution.automatic_inputs import choose_source

                    candidates = [p for k, p in paths.items() if k.startswith(f"{name}.candidate.")]
                    if candidates:
                        selected, source_choices[name] = choose_source(
                            candidates, comp.get("family")
                        )
                        paths[f"{name}.alignment"] = comp["alignment"] = selected
                    else:
                        source_choices[name] = {
                            "path": str(comp["alignment"]),
                            "reason": "configured or only alignment",
                        }
                inv = stack.enter_context(
                    open_stockholm(
                        comp["alignment"], family=comp.get("family"), molecule=comp["molecule"]
                    )
                )
                inventories[name] = inv
                if automatic:
                    from triplet_coevolution.auto_family import infer_molecule

                    decisions[name] = infer_molecule(inv, comp["molecule"])
                    comp["molecule"] = decisions[name]["molecule"]
                    comp["transcribe_t"] = decisions[name]["transcribe_t"]
                references[name] = _reference(inv, comp)
                selections[name] = _candidates(inv, comp, manifest_ids.get(name, ()))
            resolver_options = {}
            if automatic:
                from triplet_coevolution.automatic_inputs import lookup_snapshot

                cache = lookup_snapshot(
                    cfg.get("cache"), cache_directory, metadata, "uniprot.sqlite", cfg["uniprot"]
                )
                resolver_options = {
                    "automatic": True,
                    "molecule_types": {n: c["molecule"] for n, c in cfg["components"].items()},
                    "automatic_profiles": {n: d["profile"] for n, d in decisions.items()},
                    "nucleotide_cache": lookup_snapshot(
                        cfg.get("nucleotide_cache"),
                        cache_directory,
                        metadata,
                        "nucleotide.sqlite",
                        cfg["uniprot"],
                    ),
                    "nucleotide_online": cfg["uniprot"],
                }
            else:
                cache = _cache(cfg, metadata)
            result = generate_evidence(
                selections,
                metadata,
                manifest=cfg.get("manifest"),
                uniprot=cfg["uniprot"],
                cache=cache,
                expected_orthogroups=cfg["expected_orthogroups"],
                reference_anchors={
                    n: c["reference_id"]
                    for n, c in cfg["components"].items()
                    if "reference_id" in c
                },
                reference_rows={
                    n: EvidenceRow(
                        ref.row_id,
                        ref.accession,
                        ref.start,
                        ref.end,
                        selectors=(cfg["components"][n]["reference_id"],),
                        annotations=ref.annotations,
                    )
                    for n, ref in references.items()
                    if ref is not None
                },
                orthology_scope=cfg.get("orthology_scope"),
                uniprot_components=[
                    n for n, c in cfg["components"].items() if c["molecule"] == "protein"
                ],
                component_names={n: c["name"] for n, c in cfg["components"].items()},
                **resolver_options,
            )
            counts = result["counts"]
            matches = _matches(metadata, prepared, counts)
            if automatic:
                from triplet_coevolution.auto_family import choose_reference

                for name, inv in inventories.items():
                    comp = cfg["components"][name]
                    if references[name]:
                        reference_choices[name] = {
                            "source": "configured",
                            "reference_row_id": references[name].row_id,
                        }
                        continue
                    ids = (getattr(m, f"{name.lower()}_id") for m in matches)
                    ref, reference_choices[name] = choose_reference(
                        inv, ids, comp["molecule"], transcribe_t=comp["transcribe_t"]
                    )
                    if ref:
                        references[name] = ref
                        comp["reference_id"] = ref.row_id
            extract_error = None
            try:
                comps, kwargs, missing = _extract(
                    inventories, references, cfg, prepared, matches, automatic=automatic
                )
            except ValueError as exc:
                if not automatic:
                    raise
                # The freshly created private directory contains our matches and partial extracts.
                # Keep the evidence and the match table, publish none of the incomplete alignments.
                for path in prepared.iterdir():
                    if path.name != "matches.tsv":
                        path.unlink()
                extract_error = str(exc)
                kwargs, missing = {}, [extract_error]
                comps = {
                    n: {
                        "name": cfg["components"][n]["name"],
                        "molecule": cfg["components"][n]["molecule"],
                        "reference_row_id": ref.row_id if (ref := references[n]) else None,
                        "source_columns": inv.n_columns,
                        "source_rows": inv.n_rows,
                        "exported_rows": 0,
                        "alignment": None,
                    }
                    for n, inv in inventories.items()
                }
        summary = {
            "schema_version": 1,
            "software_version": __version__,
            "biopython_version": Bio.__version__,
            "mode": "evidence",
            "inputs": {k: identities[p] for k, p in paths.items()},
            "folder_assignment": order,
            "components": comps,
            "counts": counts,
            "evidence": "metadata/evidence.tsv",
            "selected_evidence": "metadata/selected_evidence.tsv",
            "group_audit": "metadata/groups.tsv",
            "resolver_summary": "metadata/evidence_summary.json",
            "matches": "prepared/matches.tsv",
            "reference_coordinates": "prepared/reference_coordinates.tsv"
            if not extract_error
            else None,
            "ready_for_analysis": not missing,
            "analysis_requested": score,
            "analysis_blockers": missing,
            "analyze_args": None,
            "analysis_summary": None,
        }
        if automatic:
            from triplet_coevolution.automatic_run import finish_run

            summary.update(
                {
                    "mode": "automatic",
                    "molecules": decisions,
                    "sources": source_choices,
                    "preparation_error": extract_error,
                    "references": reference_choices,
                    "network_enabled": cfg["uniprot"],
                    "providers": result["providers"],
                    "role_evidence_counts": result.get("role_evidence_counts", {}),
                    "group_status_counts": result["group_status_counts"],
                    "cache_directory": str(cache_directory),
                }
            )
            if cfg["uniprot"]:
                from triplet_coevolution.automatic_inputs import save_snapshots

                try:
                    save_snapshots(metadata, cache_directory)
                except (ValueError, OSError, sqlite3.Error) as exc:
                    summary["cache_warning"] = str(exc)
            finish_run(stage, out, summary, kwargs, cfg["analysis"], cfg["analysis_limits"], plots)
        elif not missing:
            _write_analysis_args(prepared, kwargs)
            summary["analyze_args"] = "prepared/analyze_args.json"
            if score:
                _analyze(stage, out, kwargs)
                summary["analysis_summary"] = "analysis/run_summary.json"
                summary["report"] = "analysis/report.html"
                from triplet_coevolution.reporting.html import attach_evidence_report

                attach_evidence_report(stage / "analysis", summary)
        if any(_file_stamp(p) != stamp for p, stamp in stamps.items()):
            raise ValueError("an input changed during preparation; retry with stable inputs")
        _write_json(stage / "workflow_summary.json", summary)
    return summary


def evidence(
    directory: Annotated[
        Path | None,
        typer.Option(
            "--input", help="Project with three named component folders (default: input)."
        ),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Alternative explicit-path project JSON.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="New or empty output directory.")] = Path(
        "results/prepared"
    ),
    uniprot: Annotated[
        bool | None,
        typer.Option(
            "--uniprot/--offline",
            help="Allow public UniProt lookups for declared protein components, or require offline operation.",
        ),
    ] = None,
    default_molecule: Annotated[
        str | None,
        typer.Option(
            "--default-molecule", help="Type for undeclared components: protein, dna or rna."
        ),
    ] = None,
    analyze: Annotated[
        bool,
        typer.Option(
            "--analyze",
            help="Also score groups when all three coordinate references and types are defined.",
        ),
    ] = False,
) -> None:
    """Resolve candidate identities and prepare synchronized rows from three Stockholm families."""
    try:
        summary = _prepare(config, directory, out, analyze, uniprot, default_molecule)
    except (ValueError, TypeError, OSError, csv.Error, sqlite3.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for key, comp in summary["components"].items():
        typer.echo(
            f"{key} = {comp['name']}: {comp['exported_rows']} of {comp['source_rows']} rows exported"
        )
    typer.echo(
        f"Resolved groups: {summary['counts']['resolved_groups']}; audit: {out / summary['evidence']}"
    )
    if summary["analysis_blockers"]:
        typer.echo("Analysis unavailable: " + "; ".join(summary["analysis_blockers"]))
    elif summary["analysis_summary"]:
        typer.echo(f"Report: {out / summary['report']}")
    else:
        typer.echo(f"Replay: {out / summary['analyze_args']}")
