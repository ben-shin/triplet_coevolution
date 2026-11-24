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
) -> None:
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
        return
    benchmark = benchmark_scores(store, labels, top_k=top_k, bootstrap=bootstrap, seed=seed)
    if benchmark["label_sha256"] != inputs["contacts"]["sha256"]:
        raise ValueError("input changed while benchmarking contact labels")
    write_json(directory / "benchmark.json", benchmark)
    outputs["benchmark"] = "benchmark.json"
    summary["benchmark_overview"] = benchmark_overview(benchmark)
    if selection["legacy"]:
        return

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
            tables.append(RankedTable(pair, method, path.name, score, ("x", "y"), count))
    return tables


def render_score_view(
    directory: Path,
    store: ScoreStore,
    tables: list[RankedTable],
    *,
    plots: bool = False,
    labels: Path | None = None,
) -> None:
    """Create a portable pair-only HTML report from the completed archive."""
    files = [t.path for t in tables]
    network = write_network(directory, tables)
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
            directory / "figures", mi, *store.maps, score=tables[0].score, min_valid=store.min_valid
        )
        if dca:
            figures["dca"] = save_dca_plot(
                directory / "figures",
                dca,
                store.maps,
                score=tables[-1].score,
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
        overview = benchmark_overview(result)
    pages = "".join(f"<li>{_link(path, path)}</li>" for path in files)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Archived pair scores</title>
<style>{STYLE}</style></head><body><header><h1>Archived pair scores</h1></header><main>
<section><h2>Score universe</h2><ul>{"".join(reference_rows)}</ul>
<p>Ranked views use the stored coordinates, minimum support of {store.min_valid} observations and {store.min_valid_weight:g} weight, and
{escape(store.provenance.get("apc_universe", "retained"))} APC axes. Display limits apply separately to each interface.</p></section>
<section><h2>Pair rankings</h2>{_rankings(directory, tables)}{"".join(pictures)}</section>
{benchmark_section(overview)}<section><h2>Files</h2><ul>{pages}</ul></section></main></body></html>"""
    (directory / "report.html").write_text(html, encoding="utf-8")


def regenerate_report(source: Path, directory: Path) -> None:
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
        if name == "report.html":
            continue
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"artifact outside source directory: {name!r}")
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
    paths += [output[k] for k in ("score_manifest", "score_payload", "benchmark") if k in output]
    paths += [output[k] for k in ("weights", "plan", "replay") if k in output]
    paths += list(output.get("diagnostics", {}).values())
    if any(path not in expected or path == "report.html" for path in paths):
        raise ValueError("report file was not present in the verified native inventory")
    write_report(directory, summary, ranked_tables(summary))
    summary["artifact_sha256"] = file_hashes(directory)
    write_json(directory / "run_summary.json", summary)
