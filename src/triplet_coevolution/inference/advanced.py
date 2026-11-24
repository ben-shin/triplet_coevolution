"""Advanced score tables and calibration on already synchronized, masked observations."""

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from triplet_coevolution.inference.candidates import (
    candidate_id,
    select_candidates,
    write_candidate_manifest,
)
from triplet_coevolution.inference.global_potts import global_triples
from triplet_coevolution.inference.local_options import (
    PAIR_SCREENS,
    SCREENED,
    LocalSettings,
    candidate_options,
)
from triplet_coevolution.inference.local_outputs import output_value, write_local_fits
from triplet_coevolution.inference.mi import MIKernel, TripletMI, check_min_valid_weight, triplet_mi
from triplet_coevolution.inference.potts import PAIRS, DCABlock, direct_couplings
from triplet_coevolution.inference.scores import (
    APCUniverse,
    check_apc_universe,
    coordinate_axes,
    dca_with_apc_policy,
    get_score,
    mi_with_apc_policy,
    pair_coverage,
    ranked_indices,
)
from triplet_coevolution.inference.threebody import fit_triples
from triplet_coevolution.inference.weights import (
    _WeightComparisons,
    rematch_evaluator,
    verify_weight_alignment,
)
from triplet_coevolution.io.export import write_dca_table
from triplet_coevolution.validation.nulls import permutation_test
from triplet_coevolution.validation.structure import contact_metrics, read_contacts

SITE_FIELDS = ("component", "reference_id", "col", "original_col", "reference_pos", "symbol")
PAIR_FIELDS = ["pair", *(f"{prefix}_{key}" for prefix in "xy" for key in SITE_FIELDS)]
TRIPLE_FIELDS = ["candidate_id"] + [f"{prefix}_{key}" for prefix in "abc" for key in SITE_FIELDS]
NULL_FIELDS = ["statistic", "direction", "score", "permutation_p", "bh_q", "null_draws"]


class ModelFitError(ValueError):
    """One optional estimator could not fit the supplied data."""

    def __init__(self, method, reason):
        super().__init__(str(reason))
        self.method = method


def _fit(method, estimator, *args, **kwargs):
    try:
        return estimator(*args, **kwargs)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ModelFitError(method, exc) from exc


def _site(mapping, index, prefix):
    s = mapping.site(int(index))
    return dict(
        zip(
            (f"{prefix}_{key}" for key in SITE_FIELDS),
            (
                mapping.component,
                mapping.reference_id,
                s.current_column,
                s.original_column,
                s.reference_position,
                s.reference_symbol,
            ),
        )
    )


def _identity(path):
    if path is None:
        return None
    path = Path(path)
    with path.open("rb") as handle:
        return {
            "path": str(path.absolute()),
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }


def _table(path, fields, rows, fmt):
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="," if fmt == "csv" else "\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return {"path": path.name, "rows": count}


def write_advanced_results(
    out_dir,
    msa,
    maps,
    weights,
    *,
    dca=False,
    threebody=False,
    ridge=0.1,
    pseudocount=0.5,
    max_triplets=1000,
    triplet_candidates=None,
    permutations=0,
    seed=0,
    strata=None,
    contacts=None,
    min_valid=2,
    min_valid_weight=0.0,
    max_pairs=1000,
    require_reference=False,
    fmt="tsv",
    mi_result=None,
    mi_kernel: MIKernel | None = None,
    score="mi_apc",
    threebody_model="local",
    global_l2=0.1,
    dca_score: str = "dca_apc",
    threebody_score: str | None = None,
    candidate_mode: str | None = None,
    candidate_policy=None,
    local_policy=None,
    apc_universe: APCUniverse = "retained",
    score_sink: Callable[[TripletMI, dict[str, DCABlock]], None] | None = None,
):
    """Write staged ranked tables and return a JSON-native inventory.

    score_sink receives the complete observed MI and DCA blocks exactly once.
    Arrays use the requested APC domain. There is no second observed fit.
    The callback receives an empty DCA mapping when that method is disabled.
    """
    min_valid_weight = check_min_valid_weight(min_valid_weight)
    verify_weight_alignment(msa,weights)
    if fmt not in {"csv", "tsv"}:
        raise ValueError("advanced results require CSV or TSV")
    threebody_score = threebody_score or (
        "connected_information_bits" if threebody_model == "maxent" else "threebody_rms"
    )
    if threebody_score == "connected_information_bits" and threebody_model != "maxent":
        raise ValueError("connected_information_bits requires the maxent model")
    if not threebody and any(
        x is not None for x in (candidate_mode, candidate_policy, local_policy)
    ):
        raise ValueError("ABC policies require threebody inference")
    if local_policy is not None and threebody_model != "maxent":
        raise ValueError("local_policy requires the maxent model")
    mode, screen = candidate_options(
        candidate_mode, candidate_policy, triplet_candidates, max_triplets
    )
    local = LocalSettings.parse(local_policy) if threebody_model == "maxent" else None
    specs = {
        "mi": get_score(score, "mi"),
        "dca": get_score(dca_score, "dca"),
        "threebody": get_score(threebody_score, "threebody"),
    }
    check_apc_universe(apc_universe)
    if permutations < 0 or seed < 0 or max_triplets < 1 or (
        max_pairs is not None and max_pairs < 1
    ):
        raise ValueError("seed/permutations must be nonnegative and output limits positive")
    if mi_kernel is None:
        mi_kernel = mi_result.kernel if mi_result is not None else "batched"
    if mi_kernel not in ("batched", "v1_scalar"):
        raise ValueError("MI kernel must be batched or v1_scalar")
    if mi_result is not None and mi_result.kernel != mi_kernel:
        raise ValueError("supplied MI and null refits must use the same kernel")
    if mi_result is not None and (
        mi_result.min_valid != min_valid
        or mi_result.min_valid_weight != min_valid_weight
        or mi_result.n_observations != msa.n_observations
    ):
        raise ValueError("supplied MI and refits require matching min_valid and observation count")
    if triplet_candidates is not None and not threebody:
        raise ValueError("triplet candidates require threebody inference")
    if threebody_model not in {"local", "maxent", "global"} or (
        threebody_model != "local" and not threebody
    ):
        raise ValueError(
            "threebody_model must be local, maxent or global, and a selected model requires threebody"
        )
    if strata is not None and not permutations:
        raise ValueError("strata require permutations")
    if len(maps) != 3 or any(
        m.component != c.name or m.n_sites != c.n_sites for m, c in zip(maps, msa.components)
    ):
        raise ValueError("reference maps must match each filtered alignment")
    out = Path(out_dir)
    extra_inputs = {
        "triplet_candidates": _identity(triplet_candidates),
        "contacts": _identity(contacts),
    }
    axes = coordinate_axes(maps, require_reference)
    mapped_axes = coordinate_axes(maps, True)
    pair_axes = {
        method: mapped_axes if spec.correction == "APC" and apc_universe == "reference" else axes
        for method, spec in specs.items() if method != "threebody"
    }
    fixed = ()
    eligible = 0
    selection = None
    proposal = None
    manifest = None
    observed_dca_for_screen = None
    if threebody:
        screen_scores = None
        if mode in PAIR_SCREENS and screen.get("training_ids") is None:
            screen_spec = get_score(screen.get("pair_statistic", "mi_bits"))
            if screen_spec.method == "dca":
                if not dca:
                    raise ValueError("DCA candidate screening requires --dca")
                observed_dca_for_screen = dca_with_apc_policy(
                    _fit(
                        "dca",
                        direct_couplings,
                        msa,
                        weights,
                        ridge=ridge,
                        min_valid=min_valid,
                        min_valid_weight=min_valid_weight,
                    ),
                    maps,
                    apc_universe,
                )
                screen_scores = {
                    pair: screen_spec.values(block)
                    for pair, block in observed_dca_for_screen.items()
                }
            else:
                if mi_result is None:
                    mi_result = triplet_mi(
                        msa,
                        weights,
                        min_valid=min_valid,
                        min_valid_weight=min_valid_weight,
                        kernel=mi_kernel,
                    )
                mi_result = mi_with_apc_policy(mi_result, maps, apc_universe)
                screen_scores = {
                    pair: screen_spec.values(getattr(mi_result, pair.lower()))
                    for pair in ("AB", "AC", "BC")
                }
        proposal = select_candidates(
            maps,
            axes,
            mode=mode,
            maximum=max_triplets,
            seed=seed,
            candidate_file=triplet_candidates,
            msa=msa,
            weights=weights,
            pair_scores=screen_scores,
            **screen,
        )
        fixed = proposal.triples
        eligible = proposal.summary["eligible_triples"]
        selection = (
            "file"
            if mode == "explicit"
            else "exhaustive"
            if mode == "uniform" and len(fixed) == eligible
            else mode
        )
    withheld = None
    if threebody_model == "maxent":
        withheld = "Whole-component shuffling breaks pair marginals. No pair-preserving M2 test was performed."
    elif mode in SCREENED or screen.get("file_origin") in {"same_data", "training"}:
        withheld = "Candidate screening uses observations overlapping the scored cohort. A selection-aware triple test was not performed."
    labels = read_contacts(contacts, maps) if contacts is not None else {}
    if extra_inputs != {
        "triplet_candidates": _identity(triplet_candidates),
        "contacts": _identity(contacts),
    }:
        raise ValueError("advanced input changed while reading; retry with stable input files")
    refit_weights = None
    null_weight_policy = None
    if permutations:
        refit_weights = rematch_evaluator(msa, weights, strata, _WeightComparisons)
        null_weight_policy = {
            "supplied": "fixed observation design weights keyed to anchored A rows",
            "clade_balanced": "fixed clade weights within nested permutation strata",
        }.get(weights.policy, "recompute declared identity policy on rematched components")
    outputs = {}
    settings = {
        "dca": dca,
        "threebody": threebody,
        "ridge": ridge,
        "pseudocount": pseudocount,
        "max_triplets": max_triplets,
        "seed": seed,
        "permutations": permutations,
        "stratified": strata is not None,
        "min_valid": min_valid,
        "min_valid_weight": min_valid_weight,
        "null_weight_policy": null_weight_policy,
        "max_pairs": max_pairs,
        "threebody_model": threebody_model,
        "global_l2": global_l2,
        "require_reference": require_reference,
        "fmt": fmt,
        "mi_statistic": score,
        "mi_kernel": mi_kernel,
        "dca_statistic": dca_score if dca else None,
        "threebody_statistic": threebody_score if threebody else None,
        "apc_universe": apc_universe,
        "score_definitions": {
            key: spec.metadata(apc_universe=apc_universe, threebody_model=threebody_model)
            for key, spec in specs.items()
            if key == "mi" or (key == "dca" and dca) or (key == "threebody" and threebody)
        },
        **extra_inputs,
    }

    local_fit = None
    out.mkdir(parents=True, exist_ok=True)
    if proposal is not None:
        manifest = write_candidate_manifest(out, proposal, fmt)
        outputs["candidate_manifest"] = {"path": "candidate_manifest.json", "rows": len(fixed)}
        outputs["candidate_selection"] = {"path": manifest["coordinate_table"], "rows": len(fixed)}

    def score_data(data, row_weights, mi=None, *, collect=False, draw=False):
        nonlocal local_fit
        evaluate_triples = threebody and (not draw or withheld is None)
        pair_scores = {}
        if permutations or labels or collect:
            if mi is None:
                mi = triplet_mi(
                    data, row_weights, min_valid=min_valid, min_valid_weight=min_valid_weight, kernel=mi_kernel,
                    include_gaps=mi_result.include_gaps if mi_result is not None else False,
                )
            mi = mi_with_apc_policy(mi, maps, apc_universe)
            pair_scores["mi"] = {
                pair: specs["mi"].values(getattr(mi, pair.lower())) for pair in ("AB", "AC", "BC")
            }
        dca_blocks = {}
        if dca:
            if not draw and observed_dca_for_screen is not None:
                dca_blocks = observed_dca_for_screen
            else:
                fitted = _fit(
                    "dca",
                    direct_couplings,
                    data,
                    row_weights,
                    ridge=ridge,
                    min_valid=min_valid,
                    min_valid_weight=min_valid_weight,
                )
                dca_blocks = dca_with_apc_policy(fitted, maps, apc_universe)
            pair_scores["dca"] = {pair: specs["dca"].values(b) for pair, b in dca_blocks.items()}
        diagnostics = {}
        triples = []
        if evaluate_triples and local is not None:
            triples, local_fit = write_local_fits(
                out, data, row_weights, proposal, axes, local, fmt, min_valid, min_valid_weight
            )
        elif evaluate_triples and not fixed:
            diagnostics = {"status": "no eligible triples"}
        elif evaluate_triples and threebody_model == "global":
            triples, diagnostics = _fit(
                "threebody",
                global_triples,
                data,
                fixed,
                row_weights,
                l2=global_l2,
                min_valid=min_valid, min_valid_weight=min_valid_weight,
            )
        elif evaluate_triples:
            triples = _fit(
                "threebody",
                fit_triples,
                data,
                fixed,
                row_weights,
                pseudocount=pseudocount,
                min_valid=min_valid, min_valid_weight=min_valid_weight,
            )
        vectors = {
            key: np.concatenate(
                [
                    blocks["ABC"[a] + "ABC"[b]][np.ix_(pair_axes[key][a], pair_axes[key][b])].ravel()
                    for a, b in PAIRS
                ]
            )
            for key, blocks in pair_scores.items()
        }
        if evaluate_triples:
            vectors["threebody"] = np.array([getattr(row, threebody_score) for row in triples])
        if collect:
            score_sink(mi, dca_blocks)
        return pair_scores, dca_blocks, triples, vectors, diagnostics

    pair_scores, dca_blocks, triples, observed, diagnostics = score_data(
        msa, weights, mi_result, collect=score_sink is not None
    )
    null = {}
    if permutations:

        # The null counter uses a common upper-tail comparison. Negation implements
        # the registered lower tail without altering the signed score in exports.
        def oriented(vectors):
            return {
                key: -values if specs[key].direction == "lower" else values
                for key, values in vectors.items()
            }

        def refit(shuffled):
            w = refit_weights(shuffled)
            return oriented(score_data(shuffled, w, draw=True)[3])

        null = permutation_test(
            msa,
            oriented({k: v for k, v in observed.items() if k != "threebody" or withheld is None}),
            refit,
            permutations=permutations,
            seed=seed,
            strata=strata,
        )
    out.mkdir(parents=True, exist_ok=True)

    def write(name, fields, rows):
        outputs[name] = _table(out / f"{name}.{fmt}", fields, rows, fmt)

    def pair_row(a, b, i, j):
        return {"pair": "ABC"[a] + "ABC"[b], **_site(maps[a], i, "x"), **_site(maps[b], j, "y")}

    def triple_row(index):
        return {
            "candidate_id": candidate_id(maps, fixed[index]),
            **{
                key: value
                for prefix, m, j in zip("abc", maps, fixed[index])
                for key, value in _site(m, j, prefix).items()
            },
        }

    ranking = {}
    if dca:
        for a, b in PAIRS:
            pair = "ABC"[a] + "ABC"[b]
            name = f"dca_{pair.lower()}"
            block = dca_blocks[pair]
            options = {
                "score": dca_score, "max_pairs": max_pairs, "min_valid": min_valid,
                "require_reference": require_reference, "min_valid_weight":min_valid_weight,
            }
            path = write_dca_table(out / f"{name}.{fmt}", pair, block, maps[a], maps[b], **options)
            coverage = pair_coverage(block, maps[a], maps[b], **options, apc_universe=apc_universe)
            ranking[name] = coverage
            outputs[name] = {"path": path.name, "rows": coverage["exported"]}
    if threebody:
        triple_fields = [
            "threebody_rms",
            "co_information_bits",
            "n_valid",
            "valid_weight",
            "observed_cells",
            "possible_cells",
        ]
        if local_fit is not None:
            triple_fields += [
                "connected_information_bits",
                "target_co_information_bits",
                "status",
                "information_status",
            ]
            outputs["local_model_fits"] = {
                "path": local_fit["diagnostic_table"],
                "rows": local_fit["diagnostic_rows"],
            }
            outputs["local_state_support"] = {
                "path": local_fit["frozen_support"],
                "rows": sum(m.n_sites for m in maps),
            }
        values = observed["threebody"]
        finite = int(np.isfinite(values).sum())
        # Keep tested candidates in source coordinates even if the model cannot score them.
        def support_rows():
            for index, row in enumerate(triples):
                yield {
                    **triple_row(index),
                    "n_valid": row.n_valid,
                    "valid_weight": row.valid_weight,
                    "count_pass": row.n_valid >= min_valid,
                    "weight_pass": row.valid_weight >= min_valid_weight,
                    "score_finite": bool(np.isfinite(values[index])),
                    "observed_cells": row.observed_cells
                    if row.possible_cells or row.n_valid == 0
                    else None,
                    "possible_cells": row.possible_cells
                    if row.possible_cells or row.n_valid == 0
                    else None,
                    "support_scope": "all retained sites"
                    if threebody_model == "global"
                    else "candidate sites",
                }

        write(
            "threebody_support",
            [
                *TRIPLE_FIELDS,
                "n_valid",
                "valid_weight",
                "count_pass",
                "weight_pass",
                "score_finite",
                "observed_cells",
                "possible_cells",
                "support_scope",
            ],
            support_rows(),
        )
        ranking["threebody_abc"] = {
            "total": eligible,
            "tested": len(fixed),
            "finite": finite,
            "exported": finite,
            "selection": selection,
            "complete_candidates": True,
            "exhaustive": len(fixed) == eligible,
        }
        write(
            "threebody_abc",
            [
                "rank",
                "competition_rank",
                "statistic",
                "direction",
                "score",
                *TRIPLE_FIELDS,
                *triple_fields,
            ],
            (
                {
                    "rank": rank,
                    "competition_rank": competition,
                    "statistic": threebody_score,
                    "direction": specs["threebody"].direction,
                    "score": values[i],
                    **triple_row(i),
                    **{k: output_value(getattr(triples[i], k)) for k in triple_fields},
                }
                for i, rank, competition in ranked_indices(values, threebody_score)
            ),
        )

    def coordinate_rows(method):
        if method == "threebody":
            for i in range(len(fixed)):
                yield triple_row(i)
        else:
            for a, b in PAIRS:
                for i in pair_axes[method][a]:
                    for j in pair_axes[method][b]:
                        yield pair_row(a, b, i, j)

    for method, (p, q) in null.items():
        spec = specs[method]
        write(
            "permutation_" + method,
            [*(TRIPLE_FIELDS if method == "threebody" else PAIR_FIELDS), *NULL_FIELDS],
            (
                {
                    **row,
                    "statistic": spec.name,
                    "direction": spec.direction,
                    "score": value,
                    "permutation_p": pv,
                    "bh_q": qv,
                    "null_draws": permutations,
                }
                for row, value, pv, qv in zip(coordinate_rows(method), observed[method], p, q)
                if np.isfinite(value)
            ),
        )
    if labels:
        methods = {}
        for method, blocks in pair_scores.items():
            labelled_scores = {}
            for a, b in PAIRS:
                pair = "ABC"[a] + "ABC"[b]
                x = {pos: i for i, pos in enumerate(maps[a].reference_positions) if pos is not None}
                y = {pos: i for i, pos in enumerate(maps[b].reference_positions) if pos is not None}
                for key in labels:
                    name, i, j = key
                    if name == pair and i in x and j in y:
                        labelled_scores[key] = blocks[pair][x[i], y[j]]
            methods[method] = contact_metrics(labels, labelled_scores)
        result = {"methods": methods, "mi_statistic": score}
        if dca:
            result["dca_statistic"] = dca_score
        (out / "contacts.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        outputs["contacts"] = {"path": "contacts.json", "rows": len(labels)}
    return {
        "settings": settings,
        "outputs": outputs,
        "ranking": ranking,
        "selection_scope": {
            "mode": mode,
            "scope": manifest["scope"],
            "data_overlap": mode in SCREENED
            or manifest.get("file_origin") in {"same_data", "training"},
            "prior_evidence": manifest["prior_source"],
            "external_prior": manifest["used_external_prior"],
            "screened": manifest["screened_triples"],
            "uniform_reserved": manifest["uniform_reserved"],
            "selected_original_coordinates_sha256": manifest[
                "selected_original_coordinates_sha256"
            ],
        }
        if manifest is not None
        else None,
        "global_fit": diagnostics if threebody_model == "global" else {},
        "candidate_manifest": "candidate_manifest.json" if manifest is not None else None,
        "local_fit": local_fit,
        "triple_calibration": {
            "status": "withheld"
            if withheld
            else "whole_component_shuffle"
            if permutations
            else "not_requested",
            "reason": withheld,
            "permutations": 0 if withheld else permutations,
        }
        if threebody
        else None,
        "candidates": {
            "eligible": eligible,
            "tested": len(fixed),
            "finite": int(np.count_nonzero(np.isfinite(observed.get("threebody", [])))),
            "selection": selection,
        },
        "hypotheses": {k: len(v) for k, v in observed.items()},
        "calibrated_hypotheses": {k: len(observed[k]) for k in null},
        "null_model": "independent whole-component row permutations within strata"
        if permutations
        else None,
    }
