"""Nested held-out comparison of shared fields and pair terms with selected three-site terms."""

import csv
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from math import prod
from pathlib import Path

import numpy as np

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.core.coordinates import subset_reference_map
from triplet_coevolution.inference.candidates import select_candidates
from triplet_coevolution.inference.maxent_support import freeze_support
from triplet_coevolution.inference.mi import canonical_support_at_cutoff
from triplet_coevolution.inference.nested_model import (
    NestedModel,
    PanelSite,
    encode_panel,
    evaluate_model,
    load_model,
    save_model,
)
from triplet_coevolution.inference.nested_solver import (
    Conditionals,
    Layout,
    NestedOptions,
    _integer,
    optimize,
)
from triplet_coevolution.msa.masking import apply_triplet_masks
from triplet_coevolution.validation._clade_scores import fit_weights
from triplet_coevolution.validation._null_checkpoint import canonical
from triplet_coevolution.validation.clade_splits import (
    SplitPlan,
    identity,
    make_split_plan,
    subset_rows,
)
from triplet_coevolution.validation.clade_stability import _weights
from triplet_coevolution.validation.null_policy import mc_interval
from triplet_coevolution.validation.null_runner import ReplicateStreams, run_null

__all__ = [
    "NestedOptions",
    "compare_nested",
    "evaluate_model",
    "fit_nested",
    "load_model",
    "write_nested",
]


@dataclass(frozen=True)
class TrainingFit:
    models: dict
    design: dict
    rows: np.ndarray
    complete: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class NestedResult:
    summary: dict
    fold_rows: list
    site_rows: list
    clade_rows: list
    models: dict


def _panel(msa, maps, panel):
    maps = tuple(maps)
    if len(maps) != 3 or any(
        m.component != c.name or m.n_sites != c.n_sites for m, c in zip(maps, msa.components)
    ):
        raise ValueError("nested source coordinate maps disagree with A/B/C alignment")
    if panel is not None and (not isinstance(panel, dict) or set(panel) != set("ABC")):
        raise ValueError("nested panel must declare original columns for each component")
    masks = []
    for m in maps:
        requested = tuple(panel[m.component]) if panel is not None else m.original_positions
        if (
            not requested
            or any(type(v) is not int or v < 1 for v in requested)
            or len(set(requested)) != len(requested)
            or not set(requested) <= set(m.original_positions)
        ):
            raise ValueError(
                f"nested panel requires known unique positive original columns in {m.component}"
            )
        keys = set(requested)
        masks.append(np.array([i in keys for i in m.original_positions]))
    if sum(int(x.sum()) for x in masks) > 256:
        raise ValueError("nested panel exceeds 256 sites; specify a sparse source-coordinate panel")
    keep = [tuple(int(i) for i in np.flatnonzero(m)) for m in masks]
    return apply_triplet_masks(msa, *masks).msa, tuple(
        subset_reference_map(m, k) for m, k in zip(maps, keep)
    )


def _ordered_terms(maps, edges, triples, training, weights, candidate_options, seed):
    coords = [(m.component, col) for m in maps for col in m.original_positions]
    lookup = {key: i for i, key in enumerate(coords)}

    def locate(item):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or item[0] not in ("A", "B", "C")
            or type(item[1]) is not int
            or tuple(item) not in lookup
        ):
            raise ValueError("term references an unknown panel role or original column")
        return lookup[tuple(item)]

    selected_edges = list(combinations(range(len(coords)), 2)) if edges is None else []
    if edges is not None:
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("a sparse edge requires two source sites")
            k = tuple(sorted(locate(p) for p in edge))
            if k[0] == k[1]:
                raise ValueError("sparse edge requires two distinct sites")
            selected_edges.append(k)
        if len(set(selected_edges)) != len(selected_edges):
            raise ValueError("duplicate sparse edge")
    pair_set = set(selected_edges)
    if triples is not None:
        if candidate_options:
            raise ValueError("provide explicit triples or train-only candidate options")
        selected = []
        for t in triples:
            if len(t) != 3:
                raise ValueError("each K term requires three original columns in ABC order")
            selected.append(tuple(locate((r, i)) for r, i in zip("ABC", t)))
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate explicit nested triple")
        screen = {
            "scope": "prespecified_original_columns",
            "selected_before_edge_filter": len(selected),
        }
    else:
        opts = {"mode": "uniform", "maximum": 100, "seed": seed, **(candidate_options or {})}
        if set(opts) - {
            "mode",
            "maximum",
            "seed",
            "screen_budget",
            "min_entropy_bits",
            "min_site_coverage",
            "uniform_quota",
        } or opts["mode"] not in (
            "uniform",
            "exhaustive",
            "triangle",
            "wedge",
            "entropy",
            "hybrid",
        ):
            raise ValueError(
                "nested screening requires internal training rows or coordinate-only selection"
            )
        data_dependent = opts["mode"] in ("triangle", "wedge", "entropy", "hybrid")
        if data_dependent and training is None:
            return (
                tuple(sorted(selected_edges)),
                (),
                {
                    "scope": "training_rows_only",
                    "mode": opts["mode"],
                    "status": "no_complete_training_rows",
                    "selected_before_edge_filter": 0,
                    "excluded_missing_pair_faces": 0,
                },
            )
        sel = select_candidates(
            maps,
            [np.arange(m.n_sites) for m in maps],
            **opts,
            msa=training if data_dependent else None,
            weights=weights if data_dependent else None,
        )
        selected = [
            tuple(lookup[(m.component, m.original_positions[i])] for m, i in zip(maps, t))
            for t in sel.triples
        ]
        screen = {
            **sel.summary,
            "scope": "training_rows_only" if data_dependent else "coordinate_only",
            "selected_before_edge_filter": len(selected),
        }
    complete = [t for t in selected if all(e in pair_set for e in combinations(t, 2))]
    if triples is not None and len(complete) != len(selected):
        raise ValueError("explicit K requires each of its three pair faces in the sparse graph")
    screen["excluded_missing_pair_faces"] = len(selected) - len(complete)
    return tuple(sorted(selected_edges)), tuple(sorted(complete)), screen


def fit_nested(
    msa,
    maps,
    *,
    options=None,
    panel=None,
    edges=None,
    triples=None,
    candidate_options=None,
    groups=None,
    weight_options=None,
    seed=0,
):
    """Fit both models on training observations only. The caller owns the split."""
    opt = options or NestedOptions()
    if not isinstance(opt, NestedOptions):
        raise TypeError("options must be NestedOptions")
    seed = _integer(seed, "seed", 0)
    train, cmaps = _panel(msa, maps, panel)
    ids = tuple(m.observation_id for m in train.matches)
    weight_options = _weights(weight_options, ids)
    if weight_options["policy"] == "clade_balanced" and (not groups or set(groups) != set(ids)):
        raise ValueError("training weights require exact training clade assignments")
    weights = fit_weights(train, groups or {}, weight_options)
    w = weights.weights
    complete = w > 0
    for c in train.components:
        q = len(CANONICAL[c.molecule]) + opt.include_gaps
        complete &= ((c.matrix >= 0) & (c.matrix < q)).all(axis=1)
    eligible_ids = tuple(key for key, yes in zip(ids, complete, strict=True) if yes)
    pool_count = opt.min_state_count if eligible_ids else 0
    native = freeze_support(
        train,
        include_gaps=opt.include_gaps,
        min_state_count=pool_count,
        training_observations=eligible_ids if pool_count else None,
    )
    sites = tuple(
        PanelSite(
            m.component,
            col,
            m.reference_id,
            m.reference_positions[i],
            m.symbols[i],
            c.molecule.value,
            native.maps[r][i],
        )
        for r, (m, c) in enumerate(zip(cmaps, train.components))
        for i, col in enumerate(m.original_positions)
    )
    rows = encode_panel(train, cmaps, sites)
    n = int(complete.sum())
    total = canonical_support_at_cutoff(float(w[complete].sum()), complete, w, opt.min_valid_weight)
    dims = tuple(max(s.native_to_bin) + 1 for s in sites)
    screen_rows = (
        train if n == len(ids) else subset_rows(train, eligible_ids) if eligible_ids else None
    )
    graph, terms, screen = _ordered_terms(
        cmaps, edges, triples, screen_rows, w[complete], candidate_options, seed
    )
    design = {
        "schema": "triplet.nested-training.v1",
        "training_ids": ids,
        "training_weights": list(weights.table_rows()),
        "training_sequence_sha256": identity(
            [(c.name, c.ids, c.sequences) for c in train.components]
        ),
        "training_match_sha256": identity([asdict(m) for m in train.matches]),
        "complete_row_mask": complete.tolist(),
        "training_missing_cells_sha256": identity(np.packbits(rows < 0).tolist()),
        "positive_complete_rows": n,
        "complete_weight": total,
        "complete_kish": None,
        "sites": [asdict(s) for s in sites],
        "edges": graph,
        "selected_triples": terms,
        "candidate_screen": screen,
        "options": asdict(opt),
        "weight_policy": {k: v for k, v in weight_options.items() if k != "supplied_weights"},
        "graph_scope": "all pairs on panel" if edges is None else "user-declared sparse graph",
        "row_policy": "positive weighted rows complete on entire fitted panel",
        "state_policy": "canonical native states or train-pooled OTHER with optional literal gap",
        "regularization": "L2/2 times squared zero-sum tensor entries by order; same field and pair penalties in both fits",
        "fields": "one zero-sum field per site; global additive energy intercept fixed at zero and cancels in conditionals",
    }
    if n:
        scaled = w[complete] / w[complete].max()
        design["complete_kish"] = float(scaled.sum() ** 2 / (scaled @ scaled))
    design_hash = identity(design)
    design["training_design_sha256"] = design_hash
    if n < opt.min_valid or total < opt.min_valid_weight:
        design["status"] = "insufficient_complete_training_support"
        return TrainingFit({}, design, rows, complete, w)
    models = {}
    for kind, k_terms in (("pair_only", ()), ("pair_plus_k", terms)):
        layout = Layout(dims, graph, k_terms, opt)
        if kind == "pair_plus_k" and not k_terms:
            theta = models["pair_only"].theta
            diag = dict(models["pair_only"].metadata["diagnostics"])
        else:
            theta, diag = optimize(layout, rows[complete], w[complete])
        meta = {**design, "training_design_sha256": design_hash, "diagnostics": diag}
        models[kind] = NestedModel(
            kind, sites, graph, k_terms, opt, theta, canonical(meta).decode()
        )
    design["status"] = (
        "converged"
        if all(m.metadata["diagnostics"]["status"] == "converged" for m in models.values())
        else "fit_not_converged"
    )
    return TrainingFit(models, design, rows, complete, w)


def _by_ids(options, ids):
    opts = dict(options)
    if opts["policy"] == "supplied":
        opts["supplied_weights"] = {i: options["supplied_weights"][i] for i in ids}
    return opts


def _percentile(clades, draws, seed, key):
    g = len(clades)
    unavailable = (
        "not_requested"
        if not draws
        else "insufficient_test_clades"
        if g < 3
        else "insufficient_bootstrap_draws"
        if draws < 20
        else None
    )
    if unavailable:
        return {
            "status": unavailable,
            "independent_test_clades": g,
            "draws": draws,
            "lower": None,
            "upper": None,
        }
    v = np.asarray(clades)
    rng = ReplicateStreams(seed, "heldout_nested_clade_percentile.v1", 0).generator(key)
    samples = [v[rng.integers(g, size=g)].mean() for _ in range(draws)]
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "status": "available",
        "independent_test_clades": g,
        "draws": draws,
        "lower": float(lo),
        "upper": float(hi),
    }


def _parametric(fit, test_rows, test_weight, observed, opt, draws, seed, fold_id):
    if not draws:
        return {"status": "not_requested", "draws": 0}
    d = fit.design
    if (
        d["weight_policy"]["policy"] not in ("uniform", "supplied", "clade_balanced")
        or d["candidate_screen"]["scope"]
        not in ("prespecified_original_columns", "coordinate_only")
        or opt.min_state_count
    ):
        return {
            "status": "unavailable_data_adaptive_weights_bins_or_screen",
            "draws": 0,
            "requested_draws": draws,
        }
    pair, extra = fit.models["pair_only"], fit.models["pair_plus_k"]
    layouts = pair.layout(), extra.layout()
    if prod(layouts[0].dimensions) > opt.max_joint_states or len(layouts[0].dimensions) > 32:
        return {
            "status": "unavailable_exact_joint_state_budget",
            "draws": 0,
            "requested_draws": draws,
        }
    p, _ = pair.exact_law()

    def simulate(rep, streams):
        rng = streams.generator("train_test_independent_configurations")

        def sample(n):
            return np.stack(
                np.unravel_index(rng.choice(p.size, size=n, p=p.ravel()), p.shape), axis=1
            )

        train = sample(len(fit.rows))
        train[fit.rows < 0] = -1
        test = sample(len(test_rows))
        test[test_rows < 0] = -1
        held = (test >= 0).all(axis=1) & (test_weight > 0)
        fits = [optimize(l, train[fit.complete], fit.weight[fit.complete]) for l in layouts]
        if any(diag["status"] != "converged" for _, diag in fits):
            value = np.nan
        else:
            losses = [
                Conditionals(l, test[held]).losses(t)
                for l, (t, _) in zip(layouts, fits, strict=True)
            ]
            value = np.average((losses[0] - losses[1]).mean(axis=1), weights=test_weight[held])
        return {"gain": [value]}

    run = run_null(
        {"gain": [observed]},
        simulate,
        draws=draws,
        seed=seed,
        null_name="nested_exact_iid_pair." + fold_id,
        identity={
            "training": d["training_design_sha256"],
            "pair_parameters": pair.digest(),
            "test_rows_mask_sha256": identity((test_rows < 0).tolist()),
            "test_weights": test_weight.tolist(),
        },
        directions={"gain": "higher"},
    )
    r = run.families["gain"]
    return {
        "status": "available",
        "draws": draws,
        "valid_draws": int(r.valid_draws[0]),
        "invalid_draws": int(r.invalid_draws[0]),
        "p_conservative": float(r.p[0]),
        "exceedances": int(r.exceedances[0]),
        "configurations": p.size,
        "tail_wilson95": mc_interval(int(r.exceedances[0]), int(r.invalid_draws[0]), draws),
        "p_floor": 1 / (1 + draws),
        "statistic": "held-out row-weighted mean conditional log-loss gain, nats/site",
        "unit": "iid configuration from exact normalized fitted pair model; fixed panel, masks, weights and candidate terms; both models refitted",
        "inference_scope": "single prespecified fold and comparison; fitted parametric approximation",
    }


def compare_nested(
    msa,
    maps,
    plan,
    *,
    options=None,
    panel=None,
    edges=None,
    triples=None,
    candidate_options=None,
    weight_options=None,
    bootstrap_draws=0,
    null_draws=0,
    seed=0,
):
    """Prespecified common-fit comparison, estimating positive loss gain on unobserved clades."""
    opt = options or NestedOptions()
    if not isinstance(opt, NestedOptions) or not isinstance(plan, SplitPlan):
        raise TypeError("nested comparison requires typed options and a validated clade SplitPlan")
    ids = tuple(r.observation_id for r in msa.matches)
    if ids != plan.observation_ids:
        raise ValueError("nested split differs from source observation IDs/order")
    heldout = (
        plan.folds[0].test_groups if plan.mode == "prespecified_holdout" and plan.folds else None
    )
    expected = make_split_plan(ids, plan.groups, test_groups=heldout, source=plan.source)
    if expected.metadata()["plan_sha256"] != plan.metadata()["plan_sha256"]:
        raise ValueError("nested split manifest disagrees with reconstructed clade isolation")
    weights = _weights(weight_options, ids)
    bootstrap_draws = _integer(bootstrap_draws, "bootstrap_draws", 0, 100000)
    null_draws = _integer(null_draws, "null_draws", 0, 100000)
    seed = _integer(seed, "seed", 0)
    if (
        len(plan.folds) * (1 + null_draws) > 10_000
        or bootstrap_draws * len(set(plan.groups.values())) > 5_000_000
    ):
        raise ValueError(
            "nested resampling budget exceeds 10000 paired fits or 5000000 sampled clade labels"
        )
    source, cmaps = _panel(msa, maps, panel)
    if edges is not None:
        edges = tuple(tuple(e) for e in edges)
    if triples is not None:
        triples = tuple(tuple(t) for t in triples)
    source_hash = identity(
        [(c.name, c.molecule.value, c.ids, c.sequences) for c in source.components]
    )
    folds, site_rows, clades, models = [], [], [], {}
    split_id = plan.metadata()["plan_sha256"]
    for fold in plan.folds:
        train = subset_rows(source, fold.train_ids)
        groups = {i: plan.groups[i] for i in fold.train_ids}
        tw = _by_ids(weights, fold.train_ids)
        row = {
            "fold_id": fold.fold_id,
            "training_rows": len(fold.train_ids),
            "test_rows": len(fold.test_ids),
            "train_groups": fold.train_groups,
            "test_groups": fold.test_groups,
            "status": "pending",
        }
        folds.append(row)
        if tw["policy"] == "supplied" and not any(tw["supplied_weights"].values()):
            row["status"] = "zero_training_weights"
            continue
        fit = fit_nested(
            train,
            cmaps,
            options=opt,
            edges=edges,
            triples=triples,
            candidate_options=candidate_options,
            weight_options=tw,
            groups=groups,
            seed=seed,
        )
        row.update(
            training_design_sha256=fit.design["training_design_sha256"],
            complete_training_rows=fit.design["positive_complete_rows"],
            training_weight_sum=fit.design["complete_weight"],
            training_kish=fit.design["complete_kish"],
            candidate_count=len(fit.design["selected_triples"]),
            edge_count=len(fit.design["edges"]),
            status=fit.design["status"],
        )
        if fit.models:
            # Keep test sequences and outcomes outside training artifacts. The split identity binds the held-out IDs.
            for name, model in list(fit.models.items()):
                meta = {**model.metadata, "fold_id": fold.fold_id, "split_sha256": split_id}
                fit.models[name] = NestedModel(
                    model.kind,
                    model.sites,
                    model.edges,
                    model.triples,
                    opt,
                    model.theta,
                    canonical(meta).decode(),
                )
            models[fold.fold_id] = fit.models
            row["diagnostics"] = {n: m.metadata["diagnostics"] for n, m in fit.models.items()}
        if row["status"] != "converged":
            continue
        test = subset_rows(source, fold.test_ids)
        test_opts = _by_ids(weights, fold.test_ids)
        w = fit_weights(test, plan.groups, test_opts)
        if w is None:
            row["status"] = "zero_test_weights"
            continue
        pair, extra = fit.models["pair_only"], fit.models["pair_plus_k"]
        encoded = encode_panel(test, cmaps, pair.sites)
        good = (encoded >= 0).all(axis=1) & (w.weights > 0)
        n = int(good.sum())
        total = canonical_support_at_cutoff(
            float(w.weights[good].sum()), good, w.weights, opt.min_valid_weight
        )
        row.update(
            complete_test_rows=n,
            test_weight_sum=total,
            test_row_mask=good.tolist(),
            test_weights=list(w.table_rows()),
            test_complete_kish=None,
        )
        if n < opt.min_valid or total < opt.min_valid_weight:
            row["status"] = "insufficient_complete_test_support"
            continue
        mass = w.weights[good]
        scaled = mass / mass.max()
        row["test_complete_kish"] = float(scaled.sum() ** 2 / (scaled @ scaled))
        la = Conditionals(pair.layout(), encoded[good]).losses(pair.theta)
        lb = Conditionals(extra.layout(), encoded[good]).losses(extra.theta)
        raw_gain = la - lb
        avg_a, avg_b, avg_gain = (
            np.average(arr, axis=0, weights=mass) for arr in (la, lb, raw_gain)
        )
        row.update(
            status="evaluated",
            pair_loss_nats_per_site=float(avg_a.mean()),
            pair_plus_k_loss_nats_per_site=float(avg_b.mean()),
            gain_nats_per_site=float(avg_gain.mean()),
        )
        for j, s in enumerate(pair.sites):
            site_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "component": s.component,
                    "original_column": s.original_column,
                    "reference_id": s.reference_id,
                    "reference_position": s.reference_position,
                    "pair_loss_nats": float(avg_a[j]),
                    "pair_plus_k_loss_nats": float(avg_b[j]),
                    "gain_nats": float(avg_gain[j]),
                }
            )
        held_ids = [key for key, yes in zip(fold.test_ids, good, strict=True) if yes]
        block_means = []
        for group in fold.test_groups:
            ix = np.array([plan.groups[i] == group for i in held_ids])
            value = (
                float(np.average(raw_gain[ix].mean(axis=1), weights=mass[ix]))
                if np.any(ix)
                else None
            )
            clades.append(
                {
                    "fold_id": fold.fold_id,
                    "clade": group,
                    "scoreable_rows": int(ix.sum()),
                    "scoreable_weight": float(mass[ix].sum()),
                    "mean_gain_nats_per_site": value,
                }
            )
            if value is not None:
                block_means.append(value)
        row["clade_macro_gain"] = float(np.mean(block_means))
        ci = _percentile(block_means, bootstrap_draws, seed, fold.fold_id)
        if len(block_means) != len(fold.test_groups):
            ci.update(status="incomplete_test_clade_support", lower=None, upper=None)
        row["heldout_clade_interval"] = ci
        row["parametric_null"] = _parametric(
            fit, encoded, w.weights, row["gain_nats_per_site"], opt, null_draws, seed, fold.fold_id
        )
    group_values = [
        r["mean_gain_nats_per_site"] for r in clades if r["mean_gain_nats_per_site"] is not None
    ]
    interval = _percentile(group_values, bootstrap_draws, seed, split_id)
    if len(group_values) != sum(len(f.test_groups) for f in plan.folds):
        interval.update(status="incomplete_test_clade_support", lower=None, upper=None)
    summary = {
        "schema": "triplet.nested-comparison.v1",
        "status": "evaluated" if group_values else "no_evaluable_holdout",
        "source": plan.source,
        "source_panel_sha256": source_hash,
        "source_coordinate_sha256": identity([asdict(m) for m in cmaps]),
        "split_sha256": split_id,
        "split_mode": plan.mode,
        "planned_folds": len(plan.folds),
        "evaluated_folds": sum(r["status"] == "evaluated" for r in folds),
        "options": asdict(opt),
        "seed": seed,
        "clade_macro_gain_nats_per_site": float(np.mean(group_values)) if group_values else None,
        "heldout_clade_interval": interval,
        "null_draws_per_fold": null_draws,
        "bootstrap_draws": bootstrap_draws,
        "gain_definition": "pair-only minus pair-plus-K conditional log loss in nats per held-out site",
        "training_objective": "weighted mean over complete rows and sites of negative log conditionals plus L2 penalties",
        "regularization_selection": "fixed before holdout; no tuning on evaluated clades",
        "clade_interval_estimand": "paired percentile interval of equal-weight held-out clade mean gains conditional on fitted folds; overlapping training sets are shared",
        "source_site_order": [[m.component, col] for m in cmaps for col in m.original_positions],
        "folds": folds,
        "artifacts": [
            "nested_folds.tsv",
            "nested_sites.tsv",
            "nested_clades.tsv",
            "nested_models/",
        ],
    }
    return NestedResult(summary, folds, site_rows, clades, models)


def write_nested(out, result):
    out = Path(out)
    model_dir = out / "nested_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for fold, models in result.models.items():
        outputs[fold] = {}
        for name, model in models.items():
            path = model_dir / f"{fold}-{name}.npz"
            save_model(path, model)
            outputs[fold][name] = {
                "path": path.relative_to(out).as_posix(),
                "model_sha256": model.digest(),
            }
    summary = {**result.summary, "model_artifacts": outputs}
    (out / "nested_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, rows, first in (
        ("folds", result.fold_rows, ["fold_id", "status"]),
        ("sites", result.site_rows, ["fold_id", "component", "original_column"]),
        ("clades", result.clade_rows, ["fold_id", "clade", "scoreable_rows"]),
    ):
        columns = dict.fromkeys(first)
        for row in rows:
            columns.update(dict.fromkeys(row))
        with (out / f"nested_{name}.tsv").open("w", encoding="utf-8", newline="") as f:
            table = csv.DictWriter(f, columns, delimiter="\t", lineterminator="\n")
            table.writeheader()
            for row in rows:
                table.writerow(
                    {
                        k: json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (list, tuple, dict))
                        else v
                        for k, v in row.items()
                    }
                )
    return summary
