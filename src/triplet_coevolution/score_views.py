"""Rank, benchmark and render existing pair-score artifacts without fitting a model."""

import csv
import json
import shutil
from html import escape
from pathlib import Path

from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.mi import TripletMI
from triplet_coevolution.inference.potts import DCABlock
from triplet_coevolution.inference.scores import get_score
from triplet_coevolution.io.export import write_dca_table, write_mi_table
from triplet_coevolution.io.score_store import (
    PAYLOAD_NAME,
    ScoreStore,
    file_digest,
    from_results,
    import_legacy_scores,
    load_score_store,
    write_score_store,
)
from triplet_coevolution.provenance import file_hashes, software_identity
from triplet_coevolution.report_artifacts import local_artifact
from triplet_coevolution.reporting.html import (
    STYLE,
    _link,
    _rankings,
    benchmark_overview,
    benchmark_section,
    write_report,
)
from triplet_coevolution.reporting.network import RankedTable, ranked_tables, write_network
from triplet_coevolution.validation.benchmark import benchmark_scores


def read_scores(source: Path, *, from_v1: bool = False) -> ScoreStore:
    """Load a verified native score store or explicitly import complete v1 ranked tables."""
    if from_v1:
        return import_legacy_scores(source)
    if (source / "scores/score_manifest.json").is_file():
        source = source / "scores"
    return load_score_store(source)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _save_labels(directory, labels, benchmark):
    name = "benchmark_labels.tsv"
    shutil.copyfile(labels, directory / name)
    if file_digest(directory / name) != benchmark["label_sha256"]:
        raise ValueError("label source changed while archiving")
    return name


def save_analysis_store(
    directory: Path,
    summary: dict,
    maps: tuple[ReferenceCoordinates, ...],
    mi: TripletMI,
    dca: dict[str, DCABlock],
    *,
    observation_ids: list[str],
    labels: Path | None,
    bootstrap: int,
    seed: int,
    top_k: list[int],
) -> ScoreStore:
    """Persist observed scores and attach complete benchmark artifacts to a staged native run."""
    selection = summary["ranking"]
    inputs = summary["inputs"]
    advanced = summary["advanced"]
    outputs = summary["outputs"]
    software = software_identity()
    summary["software"] = software
    provenance = {
        "software_version": summary["software_version"],
        **software,
        "inputs": {name: {"sha256": data["sha256"]} for name, data in inputs.items()},
        "components": summary["components"],
        "source_selection": summary.get("source_selection", {}),
        "settings": summary["settings"],
        "apc_universe": selection["apc_universe"],
        "mi_kernel": mi.kernel,
        "ranking": selection,
        "methods": {
            key: value
            for key, value in advanced.get("settings", {}).items()
            if key not in ("triplet_candidates", "contacts")
        },
        "observation_ids": observation_ids,
        "weighting": summary.get("weighting", {}),
        "qc": summary.get("qc_diagnostics", {}),
        "weight_table_sha256": file_digest(directory / "weights.tsv")
        if (directory / "weights.tsv").is_file()
        else None,
    }
    store = from_results(
        maps,
        mi,
        dca,
        provenance=provenance,
        require_reference=summary["settings"]["require_reference"],
    )
    manifest = write_score_store(directory / "scores", store)
    outputs["score_manifest"] = manifest.relative_to(directory).as_posix()
    outputs["score_payload"] = (manifest.parent / PAYLOAD_NAME).relative_to(directory).as_posix()
    if labels is None:
        return store
    benchmark = benchmark_scores(store, labels, top_k=top_k, bootstrap=bootstrap, seed=seed)
    if benchmark["label_sha256"] != inputs["contacts"]["sha256"]:
        raise ValueError("input changed while benchmarking contact labels")
    write_json(directory / "benchmark.json", benchmark)
    outputs["benchmark"] = "benchmark.json"
    from triplet_coevolution.reporting.explore_data import benchmark_preview

    write_json(directory / "benchmark_display.json", benchmark_preview(benchmark))
    outputs["benchmark_display"] = "benchmark_display.json"
    outputs["benchmark_labels"] = _save_labels(directory, labels, benchmark)
    summary["benchmark_overview"] = benchmark_overview(benchmark)
    if selection["legacy"]:
        return store

    # A compact selected-method view keeps the familiar v1 file available to
    # downstream readers. Every metric comes from the same common-label benchmark.
    contact_view = {
        "methods": {},
        "universe": "Input records; metrics on the common scored test labels.",
    }
    total = benchmark["label_records"]
    for family in ("mi", "dca"):
        name = selection[family]
        if name:
            contact_view[f"{family}_statistic"] = name
            pooled = benchmark["statistics"][name]["pooled"]
            metrics = {
                key: pooled[key]
                for key in (
                    "scored",
                    "positives_scored",
                    "negatives_scored",
                    "auroc",
                    "average_precision",
                )
            }
            contact_view["methods"][family] = {
                **metrics,
                "labels": total,
                "unscored": total - metrics["scored"],
            }
    write_json(directory / "contacts.json", contact_view)
    advanced["settings"]["contacts"] = inputs["contacts"]
    advanced["outputs"]["contacts"] = {"path": "contacts.json", "rows": total}
    return store


def write_ranked_views(
    directory: Path,
    store: ScoreStore,
    *,
    mi_score: str | None = None,
    dca_score: str | None = None,
    max_pairs: int | None = 1000,
    fmt: str = "tsv",
) -> list[RankedTable]:
    """Write ranked pair tables with the same kernels and coordinates as native analysis."""
    if fmt not in {"tsv", "csv"}:
        raise ValueError("format must be tsv or csv")
    original = store.provenance.get("ranking", {})
    mi_score = (
        mi_score
        or original.get("mi")
        or store.provenance.get("settings", {}).get("score", "mi_apc")
    )
    get_score(mi_score, "mi")
    selected_dca = dca_score or original.get("dca") or "dca_apc"
    get_score(selected_dca, "dca")
    if dca_score and dca_score not in store.scores:
        raise ValueError("the requested DCA statistic is absent from this store")
    mi, dca = store.to_results()
    tables = []
    maps = {mapping.component: mapping for mapping in store.maps}
    for method, score in [("MI", mi_score), *([("DCA", selected_dca)] if dca else [])]:
        for pair in ("AB", "AC", "BC"):
            path = directory / f"{method.lower()}_{pair.lower()}.{fmt}"
            axes = (maps[pair[0]], maps[pair[1]])
            options = {
                "score": score,
                "max_pairs": max_pairs,
                "min_valid": store.min_valid,
                "min_valid_weight": store.min_valid_weight,
                "require_reference": store.require_reference,
            }
            if method == "MI":
                write_mi_table(path, getattr(mi, pair.lower()), *axes, **options)
            else:
                write_dca_table(path, pair, dca[pair], *axes, **options)
            with path.open(newline="", encoding="utf-8") as stream:
                count = sum(
                    1 for _ in csv.DictReader(stream, delimiter="," if fmt == "csv" else "\t")
                )
            tables.append(
                RankedTable(
                    pair,
                    method,
                    path.name,
                    score,
                    ("x", "y"),
                    count,
                    store.provenance.get("methods", {}).get("dca_design")
                    if method == "DCA"
                    else None,
                )
            )
    return tables


def render_score_view(
    directory: Path,
    store: ScoreStore,
    tables: list[RankedTable],
    *,
    plots: bool = False,
    labels: Path | None = None,
    palette: str = "accessible",
    figure_formats: tuple[str, ...] = (),
    structure: Path | None = None,
    source_manifest: Path | None = None,
) -> dict:
    """Create a portable pair-only HTML report from the completed archive."""
    from dataclasses import asdict

    from triplet_coevolution.report_artifacts import add_structure
    from triplet_coevolution.reporting.explore import write_explorer
    from triplet_coevolution.reporting.explore_data import benchmark_preview

    archive = directory / "scores"
    if source_manifest:
        archive.mkdir()
        for name in ("score_manifest.json", PAYLOAD_NAME):
            shutil.copyfile(source_manifest.parent / name, archive / name)
        saved = json.loads((archive / "score_manifest.json").read_text(encoding="utf8"))
        if file_digest(archive / PAYLOAD_NAME) != saved["payload_sha256"]:
            raise ValueError("score payload changed while rendering")
    else:
        write_score_store(archive, store)
    outputs = {
        "tables": {t.method + "_" + t.label: asdict(t) for t in tables},
        "score_manifest": "scores/score_manifest.json",
        "score_payload": "scores/" + PAYLOAD_NAME,
    }
    if structure:
        outputs["structure"] = add_structure(
            directory, structure, store, file_digest(archive / "score_manifest.json")
        )
    files = [t.path for t in tables] + [outputs["score_manifest"], outputs["score_payload"]]
    network = write_network(directory, tables)
    outputs["network"] = network
    files.append(network["path"])
    reference_rows = [
        f"<li>{escape(m.component)} · {escape(m.reference_id)} · {m.n_sites:,} retained sites</li>"
        for m in store.maps
    ]
    pictures = []
    if plots:
        from triplet_coevolution.reporting.plots import save_dca_plot, save_mi_plots

        mi, dca = store.to_results()
        figures = save_mi_plots(
            directory / "figures",
            mi,
            *store.maps,
            score=tables[0].score,
            min_valid=store.min_valid,
            palette="classic" if store.provenance.get("ranking", {}).get("legacy") else palette,
        )
        if dca:
            figures["dca"] = save_dca_plot(
                directory / "figures",
                dca,
                store.maps,
                score=tables[-1].score,
                dca_design=store.provenance.get("methods", {}).get("dca_design"),
                min_valid=store.min_valid,
            )
        for name, path in figures.items():
            relative = path.relative_to(directory).as_posix()
            files.append(relative)
            if name in ("triplet", "dca"):
                pictures.append(f'<img alt="{escape(name)} pair scores" src="{escape(relative)}">')
    overview = []
    if labels:
        result = benchmark_scores(store, labels)
        write_json(directory / "benchmark.json", result)
        files.append("benchmark.json")
        outputs["benchmark"] = "benchmark.json"
        outputs["benchmark_display"] = "benchmark_display.json"
        write_json(directory / outputs["benchmark_display"], benchmark_preview(result))
        outputs["benchmark_labels"] = _save_labels(directory, labels, result)
        overview = benchmark_overview(result)
    backend = store.provenance.get("methods", {}).get("dca_design", {})
    backend_text = (
        f"DCA estimator {backend.get('backend', 'mf')} with {backend.get('conditioning', 'joint')} conditioning."
        if "dca_apc" in store.scores
        else ""
    )
    view = {
        "schema_version": 2,
        "software_version": store.provenance.get("software_version"),
        "status": "complete",
        "outputs": outputs,
        "report": "report.html",
        "ranking": store.provenance.get("ranking", {}),
        "settings": store.provenance.get("settings", {}),
        "components": store.provenance.get("components", {}),
        "tables": [asdict(t) for t in tables],
    }
    outputs["explorer"] = write_explorer(
        directory, view, tables, store=store, palette=palette, figure_formats=figure_formats
    )
    files.append(outputs["explorer"])
    pages = "".join(f"<li>{_link(path, path)}</li>" for path in files)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Archived pair scores</title>
<style>{STYLE}</style></head><body><header><h1>Archived pair scores</h1></header><main>
<nav>{_link(outputs["explorer"], "Explore scores and download figures")}</nav><section><h2>Score universe</h2><ul>{"".join(reference_rows)}</ul>
<p>Ranked views use the stored coordinates, minimum support of {store.min_valid} observations and {store.min_valid_weight:g} weight, and
{escape(store.provenance.get("apc_universe", "retained"))} APC axes. Display limits apply separately to each interface. {escape(backend_text)}</p></section>
<section><h2>Pair rankings</h2>{_rankings(directory, tables)}{"".join(pictures)}</section>
{benchmark_section(overview)}<section><h2>Files</h2><ul>{pages}</ul></section></main></body></html>"""
    (directory / "report.html").write_text(html, encoding="utf-8")
    view["artifact_sha256"] = file_hashes(directory)
    return view


def regenerate_report(
    source: Path, directory: Path, *, palette="accessible", figure_formats=(), structure=None
) -> dict:
    """Verify a checksummed analysis inventory and render its report in a fresh directory."""
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    expected = summary.get("artifact_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("report regeneration requires an analysis manifest with artifact hashes")
    root = source.resolve(strict=True)
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError(f"invalid artifact path {name!r}")
        if name in ("report.html", "explore.html") or name.startswith("report_assets/"):
            continue
        path = local_artifact(root, name)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_digest(target) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    # The summary references only payloads from its verified inventory. Never open a
    # source report's unverified path while reconstructing the page.
    output = summary["outputs"]
    paths = [v["path"] for v in output["tables"].values()]
    paths += [v["path"] for v in summary.get("advanced", {}).get("outputs", {}).values()]
    paths += [output["network"]["path"], *output["figures"].values()]
    paths += [
        output[k]
        for k in (
            "score_manifest",
            "score_payload",
            "benchmark",
            "benchmark_display",
            "calibration",
            "calibration_report",
            "conditional",
            "conditional_report",
            "stability",
            "stability_report",
            "benchmark_labels",
            "structure",
        )
        if k in output
    ]
    paths += [output[k] for k in ("weights", "plan", "replay") if k in output]
    paths += list(output.get("diagnostics", {}).values())
    advanced = summary.get("advanced", {})
    if advanced.get("candidate_manifest"):
        paths.append(advanced["candidate_manifest"])
    if fit := advanced.get("local_fit"):
        paths.extend([fit["diagnostic_table"], fit["frozen_support"]])
    if any(path not in expected or path == "report.html" for path in paths):
        raise ValueError("report file was not present in the verified native inventory")
    if structure:
        from triplet_coevolution.report_artifacts import add_structure

        stored = read_scores(directory / output["score_manifest"])
        output["structure"] = add_structure(
            directory, structure, stored, file_digest(directory / output["score_manifest"])
        )
    write_report(
        directory, summary, ranked_tables(summary), palette=palette, figure_formats=figure_formats
    )
    summary["artifact_sha256"] = file_hashes(directory)
    write_json(directory / "run_summary.json", summary)
    return summary
