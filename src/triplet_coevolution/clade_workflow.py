"""Public stability and independent clade-split workflows for matched observations."""

import csv
import hashlib
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution.analysis_plan import match_metadata
from triplet_coevolution.config import _unique
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.io.score_store import file_digest
from triplet_coevolution.io.selected_sources import matched_sources
from triplet_coevolution.msa.synchronize import synchronize
from triplet_coevolution.provenance import file_hashes
from triplet_coevolution.score_views import write_json
from triplet_coevolution.validation.null_archive import checked_file, read_json
from triplet_coevolution.validation.tree_design import TreeSource

OPTIONS = {
    "statistics",
    "top_k",
    "bootstrap_draws",
    "seed",
    "min_valid",
    "min_site_coverage",
    "min_site_entropy",
    "min_valid_weight",
    "max_pair_cells",
    "max_fits",
    "ridge",
    "include_gaps",
    "require_reference",
    "apc_universe",
    "candidate_options",
    "maxent_options",
    "min_state_count",
}


@dataclass(frozen=True)
class CladePolicy:
    split_manifest: Path | None
    clade_column: str | None
    clade_tree: TreeSource | None
    tree_depth: int | None
    source: str
    test_groups: tuple | None
    weight_policy: str
    weight_component: str | None
    weight_column: str | None
    identity_threshold: float
    min_overlap: float
    options: dict

    @classmethod
    def from_config(cls, config):
        if isinstance(config, cls):
            return config
        base = Path.cwd()
        if isinstance(config, (str, Path)):
            path = Path(config)
            base = path.resolve().parent
            config = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
        if not isinstance(config, dict):
            raise TypeError("clade policy must be a JSON object")
        opts = dict(config)
        schema = opts.pop("schema_version", 1)
        if type(schema) is not int or schema != 1:
            raise ValueError("unknown clade policy schema")
        field = opts.pop("clade_column", None)
        source = opts.pop("source", None)
        tree = opts.pop("clade_tree", None)
        depth = opts.pop("tree_depth", None)
        saved = opts.pop("split_manifest", None)
        if sum(v is not None for v in (field, tree, saved)) != 1:
            raise ValueError("specify exactly one clade_column, clade_tree or saved split_manifest")
        if saved is not None:
            saved = (base / saved).resolve()
        if field is not None and (not isinstance(field, str) or not field.strip()):
            raise ValueError("clade_column must be a nonempty match-table header")
        if tree is not None:
            tree = TreeSource.parse(tree, base)
        if (tree is None) != (depth is None):
            raise ValueError("tree_depth is required only for tree-derived groups")
        if saved and source is None:
            source = "prespecified split manifest"
        if not isinstance(source, str) or not source.strip():
            raise ValueError("describe the independent source of the clade assignments in source")
        test = opts.pop("test_groups", None)
        if saved is not None and test is not None:
            raise ValueError("test_groups are already fixed by the split manifest")
        if test is not None and (
            not isinstance(test, list)
            or not test
            or any(not isinstance(g, str) or not g for g in test)
            or len(set(test)) != len(test)
        ):
            raise ValueError("test_groups must list distinct nonempty held-out clade names")
        policy = opts.pop("weight_policy", "uniform")
        comp = opts.pop("weight_component", None)
        weight = opts.pop("weight_column", None)
        if (policy == "component") != (comp is not None) or (policy == "supplied") != (
            weight is not None
        ):
            raise ValueError(
                "component or supplied weighting requires its respective component or weight column"
            )
        identity = opts.pop("identity_threshold", 0.8)
        overlap = opts.pop("min_overlap", 0.5)
        if unknown := set(opts) - OPTIONS:
            raise ValueError(f"unknown stability options: {sorted(unknown)}")
        return cls(
            saved,
            field,
            tree,
            depth,
            source,
            tuple(test) if test is not None else None,
            policy,
            comp,
            weight,
            identity,
            overlap,
            opts,
        )

    def partition(self, table, matches_sha256):
        ids = [m.observation_id for m in table.matches]
        meta = {"source": self.source}
        test = self.test_groups
        if self.split_manifest:
            from triplet_coevolution.validation.clade_splits import load_split_plan

            plan = load_split_plan(self.split_manifest, ids)
            original = plan.source.get(
                "matches_sha256", plan.source.get("inputs", {}).get("matches")
            )
            if original is not None and matches_sha256 != original:
                raise ValueError("input match table differs from the prespecified split manifest")
            meta.update(
                split_plan_sha256=plan.metadata()["plan_sha256"],
                split_manifest_sha256=file_digest(self.split_manifest),
                split_source=plan.source,
            )
            groups = plan.groups
            test = plan.folds[0].test_groups if plan.mode == "prespecified_holdout" else None
        elif self.clade_tree:
            tree = self.clade_tree.load(ids)
            groups = tree.clades_at_depth(self.tree_depth)
            meta.update(
                tree=tree.metadata(),
                depth=self.tree_depth,
                grouping="topological edge depth from supplied root",
            )
        else:
            groups = match_metadata(table, range(len(ids)), self.clade_column)
            meta.update(clade_column=self.clade_column, grouping="supplied observation metadata")
        return groups, meta, test


def _relocated(inputs):
    if inputs is None:
        return {}
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        src = read_json(path)
        if not isinstance(src, dict):
            raise TypeError("relocated input paths must be a JSON object")
        inputs = {k: path.resolve().parent / v for k, v in src.items()}
    if not isinstance(inputs, dict) or set(inputs) - {"A", "B", "C", "matches"}:
        raise ValueError("only A, B, C or matches may be relocated")
    return inputs


def _source(source, summary, inputs):
    manifest = checked_file(
        source, summary["outputs"]["score_manifest"], summary.get("artifact_sha256", {})
    )
    doc = read_json(manifest)
    tag = doc.pop("manifest_sha256")
    raw = json.dumps(doc, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if hashlib.sha256(raw).hexdigest() != tag:
        raise ValueError("native source manifest checksum mismatch")
    prov = doc["provenance"]
    paths = {
        k: Path(inputs.get(k, summary["inputs"][k]["path"])) for k in ("A", "B", "C", "matches")
    }
    paths = {
        k: source / v if k not in inputs and not v.is_absolute() else v for k, v in paths.items()
    }
    digests = {k: prov["inputs"][k]["sha256"] for k in paths}
    if any(file_digest(path) != digests[k] for k, path in paths.items()):
        raise ValueError("native source input checksum mismatch")
    table = read_triplet_table(paths["matches"])
    comp = prov["components"]
    source = matched_sources(
        paths,
        table.matches,
        {c: v["molecule_type"] for c, v in comp.items()},
        {c: v["reference_id"] for c, v in comp.items()},
        row_references={c: v.get("reference_row_id") for c, v in comp.items()},
        starts={c: v["reference_start"] for c, v in comp.items()},
        strands={c: v.get("reference_strand", "+") for c, v in comp.items()},
        formats={c: v["format"] for c, v in prov.get("source_selection", {}).items()},
    )
    components, maps = source.components, source.maps
    if any(file_digest(path) != digests[k] for k, path in paths.items()):
        raise ValueError("source inputs changed while reading")
    return synchronize(*components, table), tuple(maps), table, digests


def _write_stability(dest, source, policy, *, summary=None, inputs=None, live=None):
    from triplet_coevolution.validation.clade_stability import evaluate_stability, write_stability

    policy = CladePolicy.from_config(policy)
    summary = summary if summary is not None else read_json(source / "run_summary.json")
    if live is None:
        msa, maps, table, digests = _source(source, summary, _relocated(inputs))
    else:
        msa, maps, table = live
        digests = {k: summary["inputs"][k]["sha256"] for k in ("A", "B", "C", "matches")}
    groups, group_source, test = policy.partition(table, digests["matches"])
    meta = {
        "inputs": digests,
        "group_source": group_source,
        "cohort": "all matched rows and original source axes entering this native analysis",
        "selection": "fold-specific training-only filtering and ranking from source axes",
        "scope": "conditional on the entered matched panel; inherited upstream preparation may select observations or sites",
    }
    weights = {
        "policy": policy.weight_policy,
        "identity_threshold": policy.identity_threshold,
        "min_overlap": policy.min_overlap,
    }
    if policy.weight_component:
        weights["component"] = policy.weight_component
    if policy.weight_column:
        weights["supplied_weights"] = match_metadata(
            table, range(len(table.matches)), policy.weight_column, numeric=True
        )
    design = summary.get("settings", {}).get("dca_design")
    if design and "ridge" in policy.options and policy.options["ridge"] != design.get("ridge"):
        raise ValueError("stability ridge disagrees with archived DCA estimator")
    result = evaluate_stability(
        msa,
        maps,
        groups,
        source=meta,
        test_groups=test,
        weight_options=weights,
        dca_design=design,
        **policy.options,
    )
    dest.mkdir(parents=True, exist_ok=True)
    write_stability(dest, result)
    info = read_json(dest / "stability_summary.json")
    files = [p for p in sorted(dest.iterdir()) if p.is_file() and p.suffix in {".tsv", ".json"}]
    links = "".join(f'<li><a href="{escape(p.name)}">{escape(p.name)}</a></li>' for p in files)
    details = f"{len(table.matches)} source observations across {len(set(groups.values()))} groups. {escape(policy.source)}."

    def cell(value):
        return escape(
            format(value, ".3g")
            if isinstance(value, float)
            else str(value)
            if value is not None
            else "—"
        )

    headings = [
        "Fold",
        "Statistic",
        "Interface",
        "Status",
        "Shared finite",
        "Held-out top-K recovery",
        "Rank correlation",
    ]
    keys = [
        "fold_id",
        "statistic",
        "interface",
        "status",
        "common_finite",
        "tie_adjusted_recovery",
        "spearman_common",
    ]
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(row.get(k))}</td>" for k in keys) + "</tr>"
        for row in result.fold_rows[:100]
    )
    table_html = (
        "<table><thead><tr>"
        + "".join(f"<th>{h}</th>" for h in headings)
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table>"
    )
    display = f"{len(result.fold_rows)} fold/interface rows; first {min(100, len(result.fold_rows))} shown. Intervals require the reported number of independent blocks and finite bootstrap draws."
    (dest / "report.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Clade stability</title><h1>Clade stability</h1><p>{details}</p><p>Scores and ranks are re-estimated on held-out groups. Group bootstrap intervals describe resampling stability. All comparisons depend on the supplied grouping and source panel. Bootstrap replicates reuse source blocks and do not increase the independent sample count.</p><h2>Held-out results</h2><p>{display}</p>{table_html}<ul>{links}</ul></html>',
        encoding="utf8",
    )
    info["report"] = "report.html"
    info["artifact_sha256"] = file_hashes(dest, exclude=("stability_summary.json",))
    write_json(dest / "stability_summary.json", info)
    return info


def stability_run(source, out, policy, *, inputs=None):
    from triplet_coevolution.cli import _stage_run

    source, out = Path(source).resolve(), Path(out).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("stability output must be separate from the source analysis")
    with _stage_run(out) as stage:
        return _write_stability(stage, source, policy, inputs=inputs)


def split_matches(matches, out, policy):
    from triplet_coevolution.cli import _stage_run
    from triplet_coevolution.validation.clade_stability import make_split_plan

    policy = CladePolicy.from_config(policy)
    matches = Path(matches)
    before = file_digest(matches)
    table = read_triplet_table(matches)
    if file_digest(matches) != before:
        raise ValueError("match table changed while reading")
    ids = tuple(m.observation_id for m in table.matches)
    groups, meta, test = policy.partition(table, before)
    meta["matches_sha256"] = before
    plan = make_split_plan(ids, groups, test_groups=test, source=meta)
    with _stage_run(Path(out)) as stage:
        write_json(stage / "split_manifest.json", plan.metadata())
        with (stage / "split_members.tsv").open("w", encoding="utf8", newline="") as h:
            w = csv.writer(h, delimiter="\t", lineterminator="\n")
            w.writerow(["observation_id", "group", "heldout_fold"])
            heldout = {k: fold.fold_id for fold in plan.folds for k in fold.test_ids}
            for k in ids:
                w.writerow([k, groups[k], heldout.get(k, "")])
    return plan


Opt = Annotated[
    Path, typer.Option("--config", help="Clade grouping, training and resampling policy JSON.")
]
Out = Annotated[Path, typer.Option("--out", help="New or empty output directory.")]


def stability(
    config: Opt,
    out: Out,
    run: Annotated[
        Path, typer.Option("--run", help="Native analysis with its original matched inputs.")
    ],
    inputs: Annotated[
        Path | None, typer.Option("--inputs", help="Optional relocated native source paths.")
    ] = None,
):
    """Estimate reproducibility on held-out clades and by resampling intact clades."""
    try:
        stability_run(run, out, config, inputs=inputs)
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(str(out / "stability_summary.json"))


def split(
    config: Opt,
    out: Out,
    matches: Annotated[
        Path,
        typer.Option("--matches", help="Original observation and clade table, before fitting."),
    ],
):
    """Assign complete clades to held-out folds before reading sequence outcomes."""
    try:
        split_matches(matches, out, config)
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(str(out / "split_manifest.json"))
