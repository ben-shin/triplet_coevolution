"""De novo size and power study of the fitted iid local M2 bootstrap."""

import argparse
import json
import platform
from math import sqrt
from statistics import NormalDist
from time import perf_counter

import numpy as np

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.inference.maxent import FrozenSupport, MaxEntOptions
from triplet_coevolution.inference.mi import weighted_mi
from triplet_coevolution.validation.null_models import matching_null, prepare_m2_bootstrap


def _binary_msa(code):
    parts = [
        ComponentMSA(
            c, "dna", tuple(f"{c}{i}" for i in range(len(code))), tuple("AC"[s] for s in col)
        )
        for c, col in zip("ABC", code.T)
    ]
    return MatchedTripletMSA(
        *parts, tuple(TripletMatch(str(i), *(c.ids[i] for c in parts)) for i in range(len(code)))
    )


def _rate(rejected, repetitions):
    freq = rejected / repetitions
    z95 = NormalDist().inv_cdf(0.975)
    corr = 1 + z95**2 / repetitions
    center = (freq + z95**2 / (2 * repetitions)) / corr
    margin = z95 * sqrt(freq * (1 - freq) / repetitions + z95**2 / (4 * repetitions**2)) / corr
    return {
        "rejected": rejected,
        "rate": freq,
        "wilson95": [max(0, center - margin), min(1, center + margin)],
    }


def study(repetitions: int, draws: int, depth: int, seed: int) -> dict:
    if repetitions < 1 or depth < 2 or draws < 19:
        raise ValueError("positive repetitions, depth >=2, and >=19 draws required at alpha=.05")
    cases = ("independent", "pairwise_triangle", "pairwise_plus_parity")
    support = FrozenSupport(("dna",) * 3, (((0, 1, 1, 1),),) * 3)
    opts = MaxEntOptions(prior_mass=0.0)
    x, y, z = np.indices((2, 2, 2)) * 2 - 1
    out = {}
    for model, name in enumerate(cases):
        energy = 0.2 * x - 0.1 * y + 0.15 * z
        if model > 0:
            energy += 0.35 * x * y - 0.18 * x * z + 0.25 * y * z
        if model == 2:
            energy += 0.32 * x * y * z
        p = np.exp(energy)
        p /= p.sum()
        reject = invalid = failed_source = 0
        elapsed = []
        for rep in range(repetitions):
            sample_seed, refit_seed = np.random.SeedSequence([seed, model, rep]).spawn(2)
            rng = np.random.default_rng(sample_seed)
            code = np.stack(
                np.unravel_index(rng.choice(8, p=p.ravel(), size=depth), p.shape), axis=1
            )
            msa = _binary_msa(code)
            t = perf_counter()
            fitted = prepare_m2_bootstrap(msa, (0, 0, 0), support=support, options=opts)
            failed_source += fitted.status != "ok"
            draw_rng = np.random.default_rng(refit_seed)
            null = np.array([fitted.draw(draw_rng) for _ in range(draws)])
            invalid += int(np.count_nonzero(~np.isfinite(null)))
            tie_atol = 64 * np.finfo(float).eps * max(1, abs(fitted.observed))
            exceed = np.count_nonzero(~np.isfinite(null) | (null >= fitted.observed - tie_atol))
            pvalue = (1 + exceed) / (1 + draws) if fitted.status == "ok" else 1
            reject += int(pvalue <= 0.05)
            elapsed.append(perf_counter() - t)
        out[name] = {
            **_rate(reject, repetitions),
            "undefined_draws": invalid,
            "unavailable_observed": failed_source,
            "median_seconds_per_cohort": float(np.median(elapsed)),
        }
    return {
        "seed": seed,
        "repetitions": repetitions,
        "draws": draws,
        "depth": depth,
        "alpha": 0.05,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "results": out,
    }


def matching_study(repetitions: int, draws: int, depth: int, seed: int) -> dict:
    """Compare two assignment hypotheses under a known two-stratum sampling model."""
    if repetitions < 1 or depth < 4 or draws < 19:
        raise ValueError("positive repetitions, depth >=4 and >=19 draws required")
    group = np.arange(depth) % 2
    labels = tuple("xy"[i] for i in group)
    freqs = np.where(group, 0.8, 0.2)
    out = {}
    for model, law in enumerate(("independent_within_group", "copy_ab_within_group")):
        results = {"whole_component": 0, "within_stratum": 0}
        for rep in range(repetitions):
            data, *streams = np.random.SeedSequence([seed, model, rep]).spawn(3)
            rng = np.random.default_rng(data)
            code = (rng.random((depth, 3)) < freqs[:, None]).astype(int)
            if model:
                copied = rng.random(depth) < 0.35
                code[copied, 1] = code[copied, 0]
            msa = _binary_msa(code)
            observed = weighted_mi(code[:, 0], code[:, 1], 2, 2)
            for name, stream in zip(results, streams):
                plan = matching_null(msa, name, strata=labels if name == "within_stratum" else None)
                rng = np.random.default_rng(stream)
                exceed = 0
                for _ in range(draws):
                    a, b, _c = plan.row_orders(rng)
                    val = weighted_mi(code[a, 0], code[b, 1], 2, 2)
                    exceed += int(val >= observed - 64 * np.finfo(float).eps)
                results[name] += int((1 + exceed) / (1 + draws) <= 0.05)
        out[law] = {name: _rate(n, repetitions) for name, n in results.items()}
    return {
        "experiment": "matching",
        "seed": seed,
        "repetitions": repetitions,
        "draws": draws,
        "depth": depth,
        "alpha": 0.05,
        "results": out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--draws", type=int, default=99)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--seed", type=int, default=771095)
    parser.add_argument("--experiment", choices=("m2", "matching"), default="m2")
    arguments = vars(parser.parse_args())
    run = study if arguments.pop("experiment") == "m2" else matching_study
    print(json.dumps(run(**arguments), indent=2))
