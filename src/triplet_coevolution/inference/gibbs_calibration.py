"""Calibrate nested held-out gain and optional local marginal I3 under a common pair law."""

from math import prod

import numpy as np

from triplet_coevolution.inference.gibbs_bootstrap import prepare_gibbs
from triplet_coevolution.inference.maxent import MaxEntOptions, fit_joint
from triplet_coevolution.validation.clade_splits import identity
from triplet_coevolution.validation.null_models import _m2_statistic
from triplet_coevolution.validation.null_policy import adjusted, mc_interval
from triplet_coevolution.validation.null_runner import ReplicateStreams, run_null


def _local_values(rows, weights, dimensions, terms, local_options):
    values = []
    for term in terms:
        shape = tuple(dimensions[i] for i in term)
        codes = np.ravel_multi_index(rows[:, term].T, shape)
        counts = np.bincount(codes, weights=weights, minlength=prod(shape)).reshape(shape)
        fit = fit_joint(counts, options=local_options, n_valid=len(rows))
        values.append(_m2_statistic(fit, "connected_information_bits")[1])
    return np.array(values)


def _tail(family, index, draws):
    valid = int(family.valid_draws[index])
    exceed = int(family.exceedances[index])
    report = {
        "valid_draws": valid,
        "invalid_draws": draws - valid,
        "p_floor": 1 / (draws + 1),
        "exceedances": exceed,
        "p_conservative": None,
        "tail_wilson95": None,
    }
    if not family.observed_valid[index]:
        report["status"] = "withheld_undefined_observed_statistic"
    elif valid != draws:
        report["status"] = "withheld_failed_refit_or_undefined_draw"
    else:
        report.update(
            status="available",
            p_conservative=float(family.p[index]),
            tail_wilson95=mc_interval(exceed, 0, draws),
        )
    return report


def gibbs_comparison(
    fit, test_rows, test_weight, observed, policy, draws, seed, fold_id, folds, refit_gain
):
    pair, extra = fit.models["pair_only"], fit.models["pair_plus_k"]
    if not extra.triples:
        return {
            "method": "gibbs",
            "status": "withheld_no_nested_contrast",
            "requested_draws": draws,
            "draws": 0,
            "p_conservative": None,
            "tail_wilson95": None,
        }
    layout = pair.layout()
    terms = extra.triples if policy.local_i3 else ()
    if policy.local_i3 and not 1 <= len(terms) <= 32:
        raise ValueError("Gibbs local_i3 requires 1 to 32 prespecified or coordinate-only triples")
    local_options = MaxEntOptions(prior_mass=1.0)
    cells = sum(prod(layout.dimensions[i] for i in term) for term in terms)
    if cells * (draws + 1) * folds * local_options.max_iterations > 300_000_000:
        raise ValueError("Gibbs local_i3 exceeds 300 million planned cell-iterations")
    if cells * 8 * 24 > layout.options.max_workspace_bytes:
        raise ValueError("Gibbs local_i3 exceeds max_workspace_bytes")
    name = "nested_gibbs_pair.v1." + fold_id
    pilot = ReplicateStreams(seed, name, 0).generator("diagnostic_chains")
    chains, diagnostics = prepare_gibbs(
        layout,
        pair.theta,
        policy,
        rows=len(fit.rows) + len(test_rows),
        draws=draws,
        folds=folds,
        rng=pilot,
    )
    summary = {
        "method": "gibbs",
        "requested_draws": draws,
        "draws": 0,
        "p_conservative": None,
        "tail_wilson95": None,
        "configurations": prod(layout.dimensions),
        "diagnostics": diagnostics,
        "seed": seed,
        "rng_namespace": name,
        "pilot_key": "diagnostic_chains",
        "endpoint_key": "independent_endpoints",
        "pair_model_sha256": pair.digest(),
        "test_missing_mask_sha256": identity((test_rows < 0).tolist()),
        "statistic": "held-out row-weighted mean conditional log-loss gain, nats/site",
        "unit": "independent finite-burn endpoints from the training pair law; fixed panel, masks and exogenous weights; both conditional models refitted",
        "inference_scope": "Approximate plug-in parametric tail for the prespecified fold. Wilson limits quantify replicate Monte Carlo uncertainty only.",
    }
    if diagnostics["mixing_status"] != "passed":
        return {**summary, "status": "withheld_gibbs_mixing"}
    held = (test_rows >= 0).all(axis=1) & (test_weight > 0)

    def local_values(rows):
        return _local_values(rows[held], test_weight[held], layout.dimensions, terms, local_options)

    observed_scores = {"gain": [observed]}
    if terms:
        observed_scores["local_i3"] = local_values(test_rows)

    def simulate(rep, streams):
        rows = chains.endpoints(
            len(fit.rows) + len(test_rows), policy, streams.generator("independent_endpoints")
        )
        train, test = np.split(rows, [len(fit.rows)])
        train[fit.rows < 0] = -1
        test[test_rows < 0] = -1
        gain = refit_gain(train, test)
        values = {"gain": [gain]}
        if terms:
            values["local_i3"] = (
                local_values(test) if np.isfinite(gain) else np.full(len(terms), np.nan)
            )
        return values

    run = run_null(
        observed_scores,
        simulate,
        draws=draws,
        seed=seed,
        null_name=name,
        directions=dict.fromkeys(observed_scores, "higher"),
        identity={
            "pair": pair.digest(),
            "training": fit.design["training_design_sha256"],
            "test_mask": summary["test_missing_mask_sha256"],
            "test_weights": test_weight.tolist(),
            "sampler": policy.metadata(),
        },
        joint_draws=True,
    )
    summary.update(draws=draws, **_tail(run.families["gain"], 0, draws))
    if terms:
        stats = observed_scores["local_i3"]
        local = [
            {
                "sites": [list(pair.sites[j].key) for j in term],
                "observed_bits": float(stats[i]) if np.isfinite(stats[i]) else None,
                **_tail(run.families["local_i3"], i, draws),
            }
            for i, term in enumerate(terms)
        ]
        bh, by = adjusted(
            [r["p_conservative"] if r["p_conservative"] is not None else 1.0 for r in local]
        )
        for i, r in enumerate(local):
            r["bh_q"] = float(bh[i]) if r["status"] == "available" else None
            r["by_q"] = float(by[i]) if r["status"] == "available" else None
        summary.update(
            local_i3=local,
            local_i3_estimand="Held-out full-panel positive-weight complete rows. Local KL from the joint target to its pairwise maximum-entropy fit in bits; symmetric Dirichlet total mass 1; fixed site alphabets. BH/BY family contains every requested K coordinate in this fold.",
        )
    return summary
