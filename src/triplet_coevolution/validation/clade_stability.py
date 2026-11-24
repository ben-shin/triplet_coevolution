"""Train-only selection, disjoint-clade score replication and conditional block uncertainty."""

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from triplet_coevolution.core.coordinates import subset_reference_map
from triplet_coevolution.inference.maxent import MaxEntOptions
from triplet_coevolution.inference.mi import check_min_valid_weight
from triplet_coevolution.inference.scores import check_apc_universe, get_score
from triplet_coevolution.inference.weights import POLICIES
from triplet_coevolution.msa.masking import apply_triplet_masks, column_mask
from triplet_coevolution.msa.qc import component_qc
from triplet_coevolution.validation._clade_scores import (
    PAIRS,
    common_spearman,
    fit_weights,
    fold_scorer,
    rank_scores,
)
from triplet_coevolution.validation.clade_splits import (
    bootstrap_sample,
    identity,
    integer,
    make_split_plan,
    subset_rows,
)

__all__ = ["bootstrap_sample", "evaluate_stability", "make_split_plan", "write_stability"]


@dataclass(frozen=True)
class StabilityResult:
    plan: object
    summary: dict
    fold_rows: list
    pair_rows: list
    bootstrap_rows: list


def _fraction(value, name, upper=1.0):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
        raise ValueError(f"invalid {name}")
    return float(value)


def _weights(options, ids):
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("weight_options must be a mapping")
    result = {"policy": "uniform", **(options or {})}
    if set(result) - {
        "policy",
        "identity_threshold",
        "min_overlap",
        "component",
        "supplied_weights",
        "memory_budget_bytes",
    }:
        raise ValueError("unsupported stability weight option; clades come from the split groups")
    if result["policy"] not in POLICIES:
        raise ValueError("unknown stability weight policy")
    if ("supplied_weights" in result) != (result["policy"] == "supplied"):
        raise ValueError("supplied weights require exactly the supplied policy")
    if result["policy"] == "supplied":
        supplied = result["supplied_weights"]
        if not isinstance(supplied, Mapping) or set(supplied) != set(ids):
            raise ValueError("supplied weights must cover exactly the source observation IDs")
        values = [_fraction(v, "supplied weight", math.inf) for v in supplied.values()]
        if not any(values):
            raise ValueError("all supplied source weights are zero")
    return result


def _summary_specs(statistics, topo):
    for name in statistics:
        for interface in ("ABC",) if get_score(name).method == "threebody" else PAIRS:
            yield {
                "fold_id": topo.fold_id,
                "statistic": name,
                "interface": interface,
                "training_blocks": len(topo.train_groups),
                "test_blocks": len(topo.test_groups),
                "training_rows": len(topo.train_ids),
                "test_rows": len(topo.test_ids),
            }


def _digest_msa(msa):
    return identity(
        [(c.name, c.molecule.value, c.ids, c.sequences) for c in msa.components]
        + [tuple(m.observation_id for m in msa.matches)]
    )


def _fit_fold(fold, source, maps, groups, settings, options, candidates, maxent, result):
    meta = list(_summary_specs(settings["statistics"], fold))
    train = subset_rows(source, fold.train_ids)
    masks = tuple(
        column_mask(
            component_qc(c),
            min_coverage=settings["min_site_coverage"],
            max_gap_fraction=1.0,
            max_missing_fraction=1.0,
            min_entropy_bits=settings["min_site_entropy"],
        )
        for c in train.components
    )
    keep = tuple(tuple(int(i) for i in np.flatnonzero(m)) for m in masks)
    design = {
        "train_ids": fold.train_ids,
        "training_msa_sha256": _digest_msa(train),
        "train_groups": {k: groups[k] for k in fold.train_ids},
        "retained_original_columns": {
            m.component: [m.original_columns[i] + 1 for i in kk]
            for m, kk in zip(maps, keep, strict=True)
        },
    }
    result.summary["training_designs"][fold.fold_id] = design
    if any(not len(kk) for kk in keep):
        reason = "no_training_sites_for_component"
        design["status"] = reason
        result.fold_rows.extend(dict(s, status=reason) for s in meta)
        return
    train = apply_triplet_masks(train, *masks).msa
    train_maps = tuple(subset_reference_map(m, kk) for m, kk in zip(maps, keep, strict=True))
    weights = fit_weights(train, groups, options)
    if weights is None:
        reason = "zero_training_weight"
        design["status"] = reason
        result.fold_rows.extend(dict(s, status=reason) for s in meta)
        return
    scorer = fold_scorer(train, train_maps, weights, settings, candidates, maxent)
    train_scores = scorer.fit(train, weights)
    train_ranks = {
        key: rank_scores(v, key[0], settings["top_k"]) for key, v in train_scores.items()
    }
    design.update(
        status="available",
        coordinate_maps=[asdict(m) for m in train_maps],
        training_weights=list(weights.table_rows()),
        train_weight_sum=weights.n_eff,
        train_weight_kish=weights.kish,
        triple_candidates=list(scorer.triples.rows()) if scorer.triples else [],
        triple_selection=scorer.triples.summary if scorer.triples else None,
        state_support=scorer.support.metadata() if scorer.support else None,
        training_top_ids={
            ":".join(key): [scorer.coordinates(key[1], i)["candidate_id"] for i in ranks.top]
            for key, ranks in train_ranks.items()
        },
    )
    design["training_fit_sha256"] = identity(
        [
            design,
            {k: v for k, v in settings.items() if k not in ("bootstrap_draws", "seed")},
            {k: v for k, v in options.items() if k != "supplied_weights"},
            candidates if scorer.triples is not None else None,
            asdict(maxent) if scorer.triples is not None else None,
        ]
    )
    test = apply_triplet_masks(subset_rows(source, fold.test_ids), *masks).msa
    test_weights = fit_weights(test, groups, options)
    test_scores = scorer.fit(test, test_weights)
    boot, credit = _bootstrap_scores(
        train,
        design["train_groups"],
        fold,
        settings,
        options,
        scorer,
        train_ranks,
        result.bootstrap_rows,
    )
    _record_comparisons(
        fold,
        meta,
        settings,
        scorer,
        train_ranks,
        train_scores,
        test_scores,
        test_weights,
        boot,
        credit,
        result,
    )


def _bootstrap_scores(train, train_groups, fold, settings, options, scorer, train_ranks, draws):
    # Vectors for selected hypotheses only, bounded explicitly at preflight.
    bcount = settings["bootstrap_draws"]
    bootstrap_values = {
        key: np.full((bcount, len(r.top)), np.nan) for key, r in train_ranks.items()
    }
    bootstrap_credit = {key: np.zeros(len(r.top)) for key, r in train_ranks.items()}
    for replicate in range(bcount):
        sample = bootstrap_sample(
            train, train_groups, seed=settings["seed"], replicate=replicate, fold_id=fold.fold_id
        )
        bw = fit_weights(sample.msa, sample.groups, options, sample.source_ids)
        score = scorer.fit(sample.msa, bw)
        draws.append(
            {
                "fold_id": fold.fold_id,
                "replicate_id": replicate,
                "sampled_groups": sample.sampled_groups,
                "n_rows": sample.msa.n_observations,
                "distinct_source_blocks": len(set(sample.sampled_groups)),
                "source_ids_sha256": identity(sample.source_ids),
                "weight_sum": bw.n_eff if bw else 0.0,
            }
        )
        for key, vals in score.items():
            idx = train_ranks[key].top
            bootstrap_values[key][replicate] = vals[idx]
            bootstrap_credit[key] += rank_scores(vals, key[0], settings["top_k"]).credit[idx]
    return bootstrap_values, bootstrap_credit


def _record_comparisons(
    fold,
    meta,
    settings,
    scorer,
    train_ranks,
    train_scores,
    test_scores,
    test_weights,
    bootstrap_values,
    bootstrap_credit,
    result,
):
    bcount = settings["bootstrap_draws"]
    for entry in meta:
        key = name, interface = entry["statistic"], entry["interface"]
        base, vals, ranks = train_scores[key], test_scores[key], train_ranks[key]
        test_rank = rank_scores(vals, name, settings["top_k"])
        top = ranks.top
        finite = np.isfinite(vals[top])
        display_hits = np.isin(top, test_rank.top)
        comp = {
            **entry,
            "status": "available"
            if ranks.n_finite and test_rank.n_finite
            else "insufficient_score_support",
            "universe_size": len(vals),
            "train_finite": ranks.n_finite,
            "test_finite": test_rank.n_finite,
            "common_finite": int((np.isfinite(base) & np.isfinite(vals)).sum()),
            "train_top_count": len(top),
            "test_top_count": len(test_rank.top),
            "train_cutoff_tie_size": ranks.cutoff_tie_size,
            "train_cutoff_tie_selected": ranks.cutoff_tie_selected,
            "test_cutoff_tie_size": test_rank.cutoff_tie_size,
            "test_cutoff_tie_selected": test_rank.cutoff_tie_selected,
            "selected_test_coverage": float(finite.mean()) if len(top) else None,
            "display_recovery": float(display_hits.mean()) if len(top) else None,
            "tie_adjusted_recovery": float(test_rank.credit[top].mean()) if len(top) else None,
            "spearman_common": common_spearman(base, vals, name),
            "test_weight_sum": test_weights.n_eff if test_weights else 0.0,
            "test_weight_kish": test_weights.kish if test_weights else None,
        }
        result.fold_rows.append(comp)
        for j, i in enumerate(top):
            boot = bootstrap_values[key][:, j]
            count = int(np.isfinite(boot).sum())
            interval_status = (
                "not_requested"
                if not bcount
                else "insufficient_training_blocks"
                if len(fold.train_groups) < 3
                else "insufficient_bootstrap_draws"
                if bcount < 20
                else "incomplete_score_support"
                if count < bcount
                else "available"
            )
            bounds = (
                np.quantile(boot, [0.025, 0.975]).tolist()
                if interval_status == "available"
                else (None, None)
            )
            result.pair_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "statistic": name,
                    "interface": interface,
                    **scorer.coordinates(interface, i),
                    "training_rank": int(ranks.display[i]),
                    "training_midrank": float(ranks.mid[i]),
                    "training_score": float(base[i]),
                    "test_rank": int(test_rank.display[i]) if finite[j] else None,
                    "test_midrank": float(test_rank.mid[i]) if finite[j] else None,
                    "test_score": float(vals[i]) if finite[j] else None,
                    "test_top_credit": float(test_rank.credit[i]),
                    "bootstrap_finite": count,
                    "bootstrap_draws": bcount,
                    "bootstrap_interval_status": interval_status,
                    "bootstrap_score_lower": bounds[0],
                    "bootstrap_score_upper": bounds[1],
                    "bootstrap_top_credit": float(bootstrap_credit[key][j] / bcount)
                    if bcount
                    else None,
                }
            )


def evaluate_stability(
    msa,
    maps,
    groups,
    *,
    statistics=("mi_bits",),
    top_k=10,
    bootstrap_draws=0,
    seed=0,
    test_groups=None,
    source=None,
    min_valid=2,
    min_site_coverage=0.0,
    min_site_entropy=0.0,
    weight_options=None,
    min_valid_weight=0.0,
    max_pair_cells=1_000_000,
    max_fits=10_000,
    ridge=0.1,
    dca_design=None,
    require_reference=False,
    apc_universe="retained",
    include_gaps=False,
    candidate_options=None,
    maxent_options=None,
    min_state_count=0,
):
    """Fit a training panel per fold and measure independent score replication on held-out clades.

    Caller supplies the source cohort and either full axes or independently prespecified axes.
    A data-selected upstream panel makes the estimates conditional on that selection.
    """
    ids = tuple(r.observation_id for r in msa.matches)
    plan = make_split_plan(ids, groups, test_groups=test_groups, source=source)
    maps = tuple(maps)
    if len(maps) != 3 or any(
        c.name != m.component or c.n_sites != m.n_sites for c, m in zip(msa.components, maps)
    ):
        raise ValueError("stability maps must match all three source axes")
    if isinstance(statistics, str) or not statistics or len(set(statistics)) != len(statistics):
        raise ValueError("statistics requires nonempty distinct score names")
    statistics = tuple(statistics)
    methods = {get_score(v).method for v in statistics}
    check_apc_universe(apc_universe)
    if type(require_reference) is not bool or type(include_gaps) is not bool:
        raise ValueError("reference and gap policy must be booleans")
    if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    top_k, bcount = integer(top_k, "top_k", 1), integer(bootstrap_draws, "bootstrap_draws")
    fits = len(plan.folds) * (2 + bcount)
    if fits > integer(max_fits, "max_fits", 1):
        raise ValueError(f"stability requires {fits} cohort fits; exceeds max_fits")
    pair_cells = sum(maps[a].n_sites * maps[b].n_sites for a, b in PAIRS.values())
    if pair_cells > integer(max_pair_cells, "max_pair_cells", 1):
        raise ValueError(
            "source pair grid exceeds max_pair_cells; supply a smaller independently specified panel"
        )
    if (
        len(plan.folds)
        * (1 + bcount)
        * sum(1 if get_score(s).method == "threebody" else 3 for s in statistics)
        * top_k
        > 2_000_000
    ):
        raise ValueError(
            "selected rank and interval workspace exceeds 2000000 entries; reduce folds, top_k or draws"
        )
    opts = {"mode": "uniform", "maximum": 100, "seed": seed, **(candidate_options or {})}
    if set(opts) - {
        "mode",
        "maximum",
        "seed",
        "screen_budget",
        "min_entropy_bits",
        "min_site_coverage",
        "uniform_quota",
    } or opts["mode"] not in ("uniform", "exhaustive", "triangle", "wedge", "entropy", "hybrid"):
        raise ValueError(
            "stability requires an internal training-cohort or prespecified uniform/exhaustive candidate screen"
        )
    if (candidate_options or maxent_options or min_state_count) and "threebody" not in methods:
        raise ValueError("triple options require a triple statistic")
    options = _weights(weight_options, ids)
    maxent = MaxEntOptions(**(maxent_options or {}))
    settings = {
        "statistics": statistics,
        "top_k": top_k,
        "bootstrap_draws": bcount,
        "seed": integer(seed, "seed"),
        "min_valid": integer(min_valid, "min_valid", 1),
        "min_valid_weight": check_min_valid_weight(min_valid_weight),
        "min_site_coverage": _fraction(min_site_coverage, "minimum site coverage"),
        "min_site_entropy": _fraction(min_site_entropy, "minimum site entropy", math.inf),
        "include_gaps": include_gaps,
        "ridge": ridge,
        "dca_design": dca_design,
        "apc_universe": apc_universe,
        "require_reference": require_reference,
        "min_state_count": integer(min_state_count, "min_state_count"),
    }
    public_weights = {k: v for k, v in options.items() if k != "supplied_weights"}
    if "supplied_weights" in options:
        public_weights["source_weights_sha256"] = identity(options["supplied_weights"])
    summary = {
        "schema": "triplet.clade-stability.v1",
        "status": "available" if plan.folds else "insufficient_independent_blocks",
        "source": plan.source,
        "split_sha256": plan.metadata()["plan_sha256"],
        "source_msa_sha256": _digest_msa(msa),
        "source_coordinates_sha256": identity([asdict(m) for m in maps]),
        "settings": settings,
        "weight_options": public_weights,
        "candidate_options": opts if "threebody" in methods else None,
        "maxent_options": asdict(maxent) if "threebody" in methods else None,
        "planned_cohort_fits": fits,
        "independent_unit": "externally declared clade; independence is an assumption",
        "bootstrap": "sample G entire training blocks uniformly with replacement, carrying every matched row; freeze training masks, state bins and candidates; refit weights and scores",
        "interval": "95% block percentile score interval conditional on the training screen; requires >=3 training blocks, >=20 draws and all finite scores",
        "test_estimand": "score replication on disjoint clades with separately estimated weights and scores on frozen training axes; ranks within each interface; no predictive likelihood or null p-value",
        "selection": "unweighted canonical site QC on training rows; canonical display order at ties; held-out boundary ties receive fractional recovery credit",
        "score_definitions": [
            get_score(n).metadata(
                apc_universe=apc_universe, threebody_model="maxent", dca_design=dca_design
            )
            for n in statistics
        ],
        "training_designs": {},
    }
    result = StabilityResult(plan, summary, [], [], [])
    for fold in plan.folds:
        _fit_fold(fold, msa, maps, plan.groups, settings, options, opts, maxent, result)
    comparisons = result.fold_rows
    summary["fold_status_counts"] = dict(Counter(x["status"] for x in comparisons))
    summary["fold_macro"] = []
    for statistic, interface in dict.fromkeys(
        (r["statistic"], r["interface"]) for r in comparisons
    ):
        subset = [
            r for r in comparisons if (r["statistic"], r["interface"]) == (statistic, interface)
        ]
        entry = {"statistic": statistic, "interface": interface, "total_folds": len(subset)}
        for field in ("selected_test_coverage", "tie_adjusted_recovery", "spearman_common"):
            observed = [r[field] for r in subset if r.get(field) is not None]
            entry[field] = float(np.mean(observed)) if observed else None
            entry[field + "_folds"] = len(observed)
        summary["fold_macro"].append(entry)
    summary["artifacts"] = [
        "split_manifest.json",
        "stability_folds.tsv",
        "stability_pairs.tsv",
        "stability_bootstrap.tsv",
    ]
    return result


def write_stability(out, result):
    """Write portable JSON and TSV tables into the caller's staging directory."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roots = {
        "split_manifest.json": result.plan.metadata(),
        "stability_summary.json": result.summary,
    }
    for file, data in roots.items():
        (out / file).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    for file, rows, leading in (
        ("stability_folds.tsv", result.fold_rows, ["fold_id", "statistic", "interface", "status"]),
        (
            "stability_pairs.tsv",
            result.pair_rows,
            ["fold_id", "statistic", "interface", "candidate_id"],
        ),
        (
            "stability_bootstrap.tsv",
            result.bootstrap_rows,
            ["fold_id", "replicate_id", "sampled_groups"],
        ),
    ):
        fields = dict.fromkeys(leading)
        for row in rows:
            fields.update(dict.fromkeys(row))
        with (out / file).open("w", encoding="utf-8", newline="") as stream:
            w = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for row in rows:
                w.writerow(
                    {
                        k: json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (list, tuple, dict))
                        else v
                        for k, v in row.items()
                    }
                )
    return result.summary
