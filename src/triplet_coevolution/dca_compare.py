"""Compare a prespecified grid of DCA conditioning and regularization settings."""

import csv
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Annotated

import numpy as np
import typer

from triplet_coevolution.clade_workflow import _relocated
from triplet_coevolution.conditional_workflow import read_policy
from triplet_coevolution.inference.dca_models import DCADesign
from triplet_coevolution.inference.scores import dca_with_apc_policy
from triplet_coevolution.io.score_store import from_results
from triplet_coevolution.msa.qc import component_qc
from triplet_coevolution.provenance import file_hashes
from triplet_coevolution.score_views import write_json
from triplet_coevolution.validation.benchmark import benchmark_scores
from triplet_coevolution.validation.null_archive import RefitData


def _midranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def _correlation(a, b):
    use = np.isfinite(a) & np.isfinite(b)
    a, b = a[use], b[use]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(_midranks(a), _midranks(b))[0, 1])


def compare_dca(run, out, config, *, labels=None, inputs=None):
    from triplet_coevolution.publication import _stage_run

    policy, _ = read_policy(config)
    models = policy.pop("models", None)
    ridge_grid = policy.pop("ridge_grid", None)
    scoped_grid = "conditioning" in policy
    conditioning = policy.pop("conditioning", ["joint", "pairwise"])
    if models is None:
        if (
            not isinstance(ridge_grid, list)
            or not ridge_grid
            or len(set(ridge_grid)) != len(ridge_grid)
        ):
            raise ValueError("provide either named models or a nonempty unique ridge_grid")
        if (
            not isinstance(conditioning, list)
            or not conditioning
            or len(set(conditioning)) != len(conditioning)
        ):
            raise ValueError("conditioning must list distinct joint or pairwise scopes")
        models = {
            f"mf-{scope}-{v}": {"backend": "mf", "conditioning": scope, "ridge": v}
            for scope in conditioning
            for v in ridge_grid
        }
    elif ridge_grid is not None or scoped_grid:
        raise ValueError("provide either models or a ridge_grid with conditioning")
    limit = policy.pop("max_score_rows", 500000)
    if (
        policy
        or type(limit) is not int
        or not 1 <= limit <= 5_000_000
        or not isinstance(models, dict)
        or not 1 <= len(models) <= 24
    ):
        raise ValueError(
            "DCA sensitivity needs 1 to 24 named models, and max_score_rows between 1 and 5000000"
        )
    if any(not isinstance(k, str) or not k.strip() or any(ord(c) < 32 for c in k) for k in models):
        raise ValueError("each model needs a printable name")
    models = {k: DCADesign.parse(v) for k, v in models.items()}
    source, out = Path(run).resolve(), Path(out).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("DCA comparison output must be separate from the native source")
    with _stage_run(out) as stage:
        data = RefitData.load(source, inputs=_relocated(inputs))
        store, msa, weights = data.store, data.msa, data.weights
        maps = store.maps
        pairs = ("AB", "AC", "BC")
        total = sum(store.n_valid[p].size for p in pairs)
        if total * len(models) > limit:
            raise ValueError(
                "DCA comparison exceeds max_score_rows; specify a smaller source site panel or fewer settings"
            )
        qc = {c.name: component_qc(c) for c in msa.components}
        mi, _ = store.to_results()
        results = []
        baseline = {}
        fields = [
            "model",
            "pair",
            "x_reference_id",
            "x_reference_pos",
            "x_original_col",
            "y_reference_id",
            "y_reference_pos",
            "y_original_col",
            "n_valid",
            "valid_weight",
            "dca_frobenius",
            "dca_apc",
        ]
        fields += [
            side + "_" + key
            for side in ("x", "y")
            for key in ("states", "entropy_bits", "coverage", "gap_fraction")
        ]
        with (stage / "dca_sensitivity.tsv").open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for j, (name, design) in enumerate(models.items(), start=1):
                estimate = design.preflight(msa, weights.weights > 0)
                diag = {}
                started = perf_counter()
                blocks = dca_with_apc_policy(
                    design.fit(
                        msa,
                        weights,
                        min_valid=store.min_valid,
                        min_valid_weight=store.min_valid_weight,
                        diagnostics=diag,
                    ),
                    maps,
                    store.provenance["apc_universe"],
                )
                seconds = perf_counter() - started
                interfaces = {}
                for pair, block in blocks.items():
                    a, b = [maps["ABC".index(c)] for c in pair]
                    if j == 1:
                        baseline[pair] = block
                    current = {}
                    qa, qb = qc[pair[0]], qc[pair[1]]
                    covariates = {
                        "state_count_product": np.outer(
                            (qa.state_counts > 0).sum(0), (qb.state_counts > 0).sum(0)
                        ),
                        "lower_site_entropy": np.minimum.outer(qa.entropy_bits, qb.entropy_bits),
                        "lower_site_coverage": np.minimum.outer(qa.coverage, qb.coverage),
                        "higher_gap_fraction": np.maximum.outer(qa.gap_fraction, qb.gap_fraction),
                        "valid_weight": block.valid_weight,
                    }
                    for field, statistic in (
                        ("frobenius", "dca_frobenius"),
                        ("corrected", "dca_apc"),
                    ):
                        arr = getattr(block, field)
                        ref = getattr(baseline[pair], field)
                        mask = np.isfinite(ref) & np.isfinite(arr)
                        current[statistic] = {
                            "finite": int(np.isfinite(arr).sum()),
                            "finite_in_common": int(mask.sum()),
                            "spearman_vs_first": _correlation(arr[mask], ref[mask]),
                            "score_covariate_spearman": {
                                k: _correlation(arr.ravel(), v.ravel())
                                for k, v in covariates.items()
                            },
                        }
                    interfaces[pair] = current
                    for x, y in np.ndindex(block.n_valid.shape):
                        row = {
                            "model": name,
                            "pair": pair,
                            "n_valid": int(block.n_valid[x, y]),
                            "valid_weight": float(block.valid_weight[x, y]),
                        }
                        for side, m, i in (("x", a, x), ("y", b, y)):
                            q = qc[m.component]
                            row.update(
                                {
                                    side + "_" + key: value
                                    for key, value in (
                                        ("reference_id", m.reference_id),
                                        ("reference_pos", m.reference_positions[i]),
                                        ("original_col", m.original_positions[i]),
                                        ("states", int(np.count_nonzero(q.state_counts[:, i]))),
                                        ("entropy_bits", float(q.entropy_bits[i])),
                                        ("coverage", float(q.coverage[i])),
                                        ("gap_fraction", float(q.gap_fraction[i])),
                                    )
                                }
                            )
                        for key, v in (
                            ("dca_apc", block.corrected[x, y]),
                            ("dca_frobenius", block.frobenius[x, y]),
                        ):
                            row[key] = float(v) if np.isfinite(v) else None
                        writer.writerow(row)
                combined = {
                    "name": name,
                    "design": design.metadata(),
                    "fit_seconds": seconds,
                    "preflight": estimate,
                    "diagnostics": diag,
                    "interfaces": interfaces,
                }
                if labels:
                    provenance = {
                        **store.provenance,
                        "methods": {
                            **store.provenance.get("methods", {}),
                            "dca_design": design.metadata(),
                        },
                    }
                    fit_store = from_results(
                        maps,
                        mi,
                        blocks,
                        provenance=provenance,
                        require_reference=store.require_reference,
                    )
                    benchmark = benchmark_scores(fit_store, Path(labels))
                    path = f"benchmark_{j}.json"
                    write_json(stage / path, benchmark)
                    combined.update(benchmark=path, label_sha256=benchmark["label_sha256"])
                results.append(combined)
        rows = "".join(
            f"<li>{escape(r['name'])} · {escape(r['design']['backend'])} · {escape(r['design']['conditioning'])} · {r['fit_seconds']:.3f} seconds"
            + (f' · <a href="{r["benchmark"]}">contact benchmark</a>' if "benchmark" in r else "")
            + "</li>"
            for r in results
        )
        (stage / "report.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>DCA sensitivity</title><h1>DCA sensitivity</h1><p>All fits use the original scored observations, weights and site coordinates. Conditioning can change both the estimate and complete-case support. Structural labels describe contact recovery on this panel.</p><ul>'
            + rows
            + '</ul><a href="dca_sensitivity.tsv">Coordinates, scores and support</a> · <a href="dca_comparison.json">Rank agreement and diagnostics</a></html>',
            encoding="utf8",
        )
        summary = {
            "schema": "triplet.dca-comparison.v1",
            "inputs": store.provenance["inputs"],
            "source_score_manifest_sha256": data.summary["artifact_sha256"][
                data.summary["outputs"]["score_manifest"]
            ],
            "observations": msa.n_observations,
            "weight_sum": float(weights.weights.sum()),
            "apc_universe": store.provenance["apc_universe"],
            "selection": "Prespecified descriptive grid; first model is the rank comparison baseline. No validation labels tune the fits.",
            "models": results,
            "score_rows": total * len(models),
            "artifact_sha256": file_hashes(stage),
        }
        write_json(stage / "dca_comparison.json", summary)
        return summary


def dca_compare(
    config: Annotated[Path, typer.Option("--config")],
    run: Annotated[Path, typer.Option("--run")],
    out: Annotated[Path, typer.Option("--out")],
    labels: Annotated[Path | None, typer.Option("--labels")] = None,
    inputs: Annotated[Path | None, typer.Option("--inputs")] = None,
):
    """Compare regularization, ABC conditioning and backend on identical input axes."""
    try:
        compare_dca(run, out, config, labels=labels, inputs=inputs)
    except (ValueError, TypeError, KeyError, OSError, ImportError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(str(out / "dca_comparison.json"))
