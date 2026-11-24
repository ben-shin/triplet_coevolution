"""Reproducible exact-small discovery, known-law calibration and sparse support sensitivity."""

import json
import platform
from itertools import product
from math import sqrt
from time import perf_counter

import numpy as np
from _resources import peak_resident_bytes

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.candidates import select_candidates
from triplet_coevolution.inference.maxent import MaxEntOptions, fit_joint
from triplet_coevolution.inference.mi import triplet_mi

MODEL_NAMES = (
    "independent",
    "pair_chain",
    "pair_triangle",
    "latent_clade",
    "pure_xor",
    "biased_noisy_xor",
    "pair_and_K",
)
MODES = ("uniform", "triangle", "wedge", "entropy", "hybrid")
L = 4
N = 192
REPS = 48
NULL_DRAWS = 39
BUDGET = 12
ROOT_SEED = 34309
OPTIONS = MaxEntOptions(prior_mass=0.5)
COORDS = tuple(product(range(L), repeat=3))


def law(name):
    a, b, c = np.indices((2, 2, 2))
    x, y, z = 2 * a - 1, 2 * b - 1, 2 * c - 1
    if name == "pure_xor":
        return ((a ^ b) == c).astype(float) / 4
    if name == "biased_noisy_xor":
        return (
            np.where(a, 0.24, 0.76) * np.where(b, 0.69, 0.31) * np.where((a ^ b) == c, 0.87, 0.13)
        )
    if name == "latent_clade":
        p = np.zeros((2, 2, 2))
        for mix, theta in ((0.28, (0.90, 0.83, 0.88)), (0.72, (0.09, 0.19, 0.13))):
            p += (
                mix
                * np.where(a, theta[0], 1 - theta[0])
                * np.where(b, theta[1], 1 - theta[1])
                * np.where(c, theta[2], 1 - theta[2])
            )
        return p
    params = {
        "independent": (0.0, 0.0, 0.0, 0.0),
        "pair_chain": (0.7, 0.65, 0.0, 0.0),
        "pair_triangle": (0.65, 0.55, 0.35, 0.0),
        "pair_and_K": (0.35, 0.24, -0.16, 0.65),
    }
    ab, bc, ac, k = params[name]
    p = np.exp(0.15 * x - 0.1 * y + 0.2 * z + ab * x * y + bc * y * z + ac * x * z + k * x * y * z)
    return p / p.sum()


def binary_m2_oracle(p):
    """Binary pair marginals leave a single parity contrast, solved independently of IPF."""
    parity = (np.indices((2, 2, 2)).sum(axis=0) % 2) == 0
    direction = np.where(parity, 1.0, -1.0)
    lo, hi = -p[parity].min(), p[~parity].min()
    for _ in range(100):
        t = (lo + hi) / 2
        q = p + t * direction
        # A pair-only law has equal products over positive and negative parity.
        if q[parity].prod() > q[~parity].prod():
            hi = t
        else:
            lo = t
    return p + (0.5 * (lo + hi)) * direction


def fixed_clade(theta):
    result = np.ones((2, 2, 2))
    for v, x in zip(theta, np.indices((2, 2, 2))):
        result *= np.where(x, v, 1 - v)
    return result


def clade_transfer(maps, axes):
    # Added as a distinct held-group stress test. A hidden mixture is dependent
    # marginally even though the sites are independent within each source clade.
    p0 = fixed_clade((0.90, 0.83, 0.88))
    p1 = fixed_clade((0.09, 0.19, 0.13))
    found = []
    effects = []
    for j in range(REPS):
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, 77, j]))
        target = tuple(map(int, rng.integers(L, size=3)))
        competing = tuple((v + 1) % L for v in target)
        train = make_panel(rng, p0, target, competing, N)
        test = make_panel(rng, p1, target, competing, N)
        msa = as_msa(np.concatenate([train, test], axis=0))
        chosen = select_candidates(
            maps,
            axes,
            mode="hybrid",
            maximum=BUDGET,
            seed=ROOT_SEED + 101 * j,
            msa=msa,
            training_ids=[m.observation_id for m in msa.matches[:N]],
        )
        found.append(target in chosen.triples)
        effects.append(fit_i3(counts(test[:, np.arange(3), target])))
    v = np.asarray(effects)
    rng = np.random.default_rng(1541)
    intervals = np.quantile(
        v[rng.integers(len(v), size=(4000, len(v)))].mean(axis=1), [0.025, 0.975]
    )
    return {
        "train_clade": 0,
        "held_clade": 1,
        "replicates": REPS,
        "hybrid_target_inclusion": rate(found),
        "held_target_mean_I3": float(v.mean()),
        "mean_I3_bootstrap95": intervals.tolist(),
        "train_true_I3": fit_joint(p0).score.connected_information_bits,
        "held_true_I3": fit_joint(p1).score.connected_information_bits,
    }


def counts(rows):
    return np.bincount(4 * rows[:, 0] + 2 * rows[:, 1] + rows[:, 2], minlength=8).reshape(2, 2, 2)


def draws(rng, p, n):
    return np.stack(np.unravel_index(rng.choice(8, n, p=p.ravel()), (2, 2, 2)), axis=1)


def make_panel(rng, p, target, competing, n):
    x = rng.integers(2, size=(n, 3, L))
    for t, prob in ((target, p), (competing, law("pair_triangle"))):
        chosen = draws(rng, prob, n)
        for k, pos in enumerate(t):
            x[:, k, pos] = chosen[:, k]
    return x


def as_msa(x):
    ids = tuple(f"r{i}" for i in range(len(x)))
    matches = tuple(TripletMatch(i, i, i, i) for i in ids)
    comps = tuple(
        ComponentMSA(c, m, ids, tuple("".join(v) for v in np.array(list(alphabet))[x[:, i, :]]))
        for i, (c, m, alphabet) in enumerate(
            zip("ABC", ("protein", "dna", "rna"), ("DV", "GT", "CU"))
        )
    )
    return MatchedTripletMSA(*comps, matches)


def fit_i3(table, opts=OPTIONS):
    score = fit_joint(table, options=opts, n_valid=int(table.sum())).score
    return score.connected_information_bits


def rate(values):
    n = len(values)
    k = sum(values)
    frac = k / n
    z = 1.959963984540054
    center = (frac + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * sqrt(frac * (1 - frac) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "successes": int(k),
        "n": n,
        "fraction": frac,
        "wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def run():
    started = perf_counter()
    maps = tuple(
        ReferenceCoordinates(
            c, "simulated_" + c, tuple(range(L)), tuple(range(1, L + 1)), ("G",) * L
        )
        for c in "ABC"
    )
    axes = (tuple(range(L)),) * 3
    scenarios = {}
    for model, name in enumerate(MODEL_NAMES):
        p = law(name)
        theoretical = fit_joint(p)
        assert theoretical.score.status == "ok"
        null = binary_m2_oracle(p)
        discrepancy = float(np.max(np.abs(null - theoretical.p2)))
        if discrepancy > 1e-7:
            raise ValueError(f"binary root oracle and IPF differ for {name}: {discrepancy}")
        records = {m: {"included": [], "ranked_first": []} for m in MODES}
        exhaustive_top = []
        rejected = []
        sparse = []
        invalid = 0
        for j in range(REPS):
            rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, model, 2, j]))
            target = tuple(map(int, rng.integers(L, size=3)))
            competing = tuple((v + 1) % L for v in target)
            train = make_panel(rng, p, target, competing, N)
            test = make_panel(rng, p, target, competing, N)
            msa = as_msa(train)
            raw = triplet_mi(msa)
            mats = {k: getattr(raw, k.lower()).mi_bits for k in ("AB", "AC", "BC")}
            held = {}
            for t in COORDS:
                v = fit_i3(counts(test[:, np.arange(3), t]))
                if np.isfinite(v):
                    held[t] = v
                else:
                    invalid += 1

            # Finite scores rank descending; exact ties in original-coordinate order.
            def rank(t, scores=held):
                return -scores[t], t

            exhaustive_top.append(bool(held) and min(held, key=rank) == target)
            reference_rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, model, 1, j]))
            ref = np.array(
                [
                    fit_i3(reference_rng.multinomial(N, null.ravel()).reshape(2, 2, 2))
                    for _ in range(NULL_DRAWS)
                ]
            )
            if not np.isfinite(ref).all():
                raise RuntimeError(f"oracle calibration failed in {name} replicate {j}")
            actual = held.get(target, np.nan)
            rejected.append(
                bool(
                    np.isfinite(actual)
                    and (1 + np.count_nonzero(ref >= actual - 1e-12)) / (NULL_DRAWS + 1) <= 0.05
                )
            )
            for method in MODES:
                opts = (
                    {"pair_scores": mats, "msa": msa}
                    if method in ("triangle", "wedge", "hybrid")
                    else {"msa": msa}
                )
                selection = select_candidates(
                    maps, axes, mode=method, maximum=BUDGET, seed=ROOT_SEED + 101 * j, **opts
                )
                selected = [t for t in selection.triples if t in held]
                records[method]["included"].append(target in selection.triples)
                records[method]["ranked_first"].append(
                    bool(selected) and min(selected, key=rank) == target
                )
            # A nested sparse prefix of the independent evaluation observations.
            vals = test[:, np.arange(3), target]
            results = []
            for n in (24, N):
                for prior in (0.0, 0.5, 8.0):
                    sc = fit_joint(
                        counts(vals[:n]), options=MaxEntOptions(prior_mass=prior), n_valid=n
                    ).score
                    results.append(
                        [sc.connected_information_bits, sc.observed_cells, sc.prior_fraction]
                    )
            sparse.append(results)
        arr = np.asarray(sparse)
        sensitivities = {}
        for i, (n, prior) in enumerate(product((24, N), (0.0, 0.5, 8.0))):
            v = arr[:, i, 0]
            good = np.isfinite(v)
            sensitivities[f"N{n}_alpha{prior:g}"] = {
                "finite": int(good.sum()),
                "mean_I3": float(v[good].mean()) if good.any() else None,
                "mean_occupied_cells": float(arr[:, i, 1].mean()),
                "prior_fraction": float(arr[0, i, 2]),
            }
        scenarios[name] = {
            "truth_I3_bits": theoretical.score.connected_information_bits,
            "binary_oracle_max_abs_difference": discrepancy,
            "truth_co_information_bits": theoretical.score.co_information_bits,
            "known_M2_rejection": rate(rejected),
            "exhaustive_held_top1": rate(exhaustive_top),
            "candidate_rates": {
                m: {k: rate(v) for k, v in rec.items()} for m, rec in records.items()
            },
            "sensitivity": sensitivities,
            "nonfinite_heldout_candidate_scores": invalid,
        }
    return {
        "algorithm": "fixed panel discovery and known-law M2 Monte Carlo v1",
        "seed": ROOT_SEED,
        "replicates": REPS,
        "rows_per_train_and_test": N,
        "universe": L**3,
        "candidate_budget": BUDGET,
        "known_null_draws": NULL_DRAWS,
        "scenarios": scenarios,
        "held_clade_contrast": clade_transfer(maps, axes),
        "runtime": {
            "seconds": perf_counter() - started,
            "peak_rss_bytes": peak_resident_bytes(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
