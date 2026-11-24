"""Prespecified held-out loss and exact fitted pair-null study in independent categorical panels."""

import argparse
import json
import platform
from itertools import combinations, product
from time import perf_counter

import numpy as np
from null_calibration import _rate

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.nested import compare_nested
from triplet_coevolution.inference.nested_solver import (
    Conditionals,
    Layout,
    NestedOptions,
    exact_law,
    optimize,
)
from triplet_coevolution.validation.clade_splits import make_split_plan
from triplet_coevolution.validation.null_runner import TIE_EPS, ReplicateStreams

CASES = ("pair_only", "pair_and_k", "weak_k", "latent_omitted", "mediator_conditioned")


def clade_transfer(repetitions, seed):
    """Four training and three fresh held-out clades with independent field shifts."""
    labels = np.repeat(np.arange(7), [30] * 4 + [24] * 3)
    ids = tuple(f"row{i}" for i in range(len(labels)))
    groups = {i: f"clade{g}" for i, g in zip(ids, labels, strict=True)}
    plan = make_split_plan(ids, groups, test_groups=["clade4", "clade5", "clade6"])
    maps = tuple(
        ReferenceCoordinates(role, role + "-external", (0,), (1,), ("A",)) for role in "ABC"
    )
    codes = np.array(list(product(range(2), repeat=3)))
    a, b, c = (2 * codes - 1).T
    out = {}
    for case in ("persistent", "reversed_in_test_clades"):
        effects = []
        positive_intervals = 0
        for rep in range(repetitions):
            rng = ReplicateStreams(seed, "nested_clade_transfer.v1", rep).generator(case)
            rows = np.empty((len(ids), 3), dtype=int)
            for g in range(7):
                h = rng.normal(0, 0.22, size=3)
                k = 0.55 if g < 4 or case == "persistent" else -0.35
                energy = (
                    h[0] * a + h[1] * b + h[2] * c + 0.25 * a * b - 0.20 * b * c + k * a * b * c
                )
                law = np.exp(energy - energy.max())
                mask = labels == g
                rows[mask] = codes[rng.choice(8, size=int(mask.sum()), p=law / law.sum())]
            parts = tuple(
                ComponentMSA(
                    role, "dna", tuple(role + i for i in ids), tuple("AC"[x] for x in rows[:, j])
                )
                for j, role in enumerate("ABC")
            )
            msa = MatchedTripletMSA(
                *parts,
                tuple(TripletMatch(i, *(part.ids[j] for part in parts)) for j, i in enumerate(ids)),
            )
            fit = compare_nested(
                msa,
                maps,
                plan,
                options=NestedOptions(l2=0.025, min_state_count=1, tolerance=1e-5),
                triples=[(1, 1, 1)],
                bootstrap_draws=200,
                seed=seed + rep,
            )
            gain = fit.summary["clade_macro_gain_nats_per_site"]
            if gain is not None:
                effects.append(gain)
                positive_intervals += int(fit.summary["heldout_clade_interval"]["lower"] > 0.0)
        arr = np.array(effects)
        rng = ReplicateStreams(seed, "nested_transfer_summary.v1", 0).generator(case)
        means = arr[rng.integers(len(arr), size=(2000, len(arr)))].mean(axis=1) if len(arr) else []
        out[case] = {
            "mean_clade_macro_gain": float(arr.mean()) if len(arr) else None,
            "independent_cohort_bootstrap95": np.quantile(means, [0.025, 0.975]).tolist()
            if len(arr)
            else None,
            "positive_gain": _rate(int((arr > 0).sum()), repetitions),
            "positive_three_clade_percentile_interval": _rate(positive_intervals, repetitions),
            "invalid_cohorts": repetitions - len(arr),
        }
    return out


def truth(name):
    mediated = name in ("latent_omitted", "mediator_conditioned")
    q = (2,) * (4 if mediated else 3)
    x = np.array(list(product((0, 1), repeat=len(q))))
    a, b, c = (2 * x[:, :3] - 1).T
    energy = 0.12 * a - 0.15 * b + 0.1 * c + 0.35 * a * b - 0.28 * b * c + 0.12 * a * c
    if name in ("pair_and_k", "weak_k"):
        energy += (0.55 if name == "pair_and_k" else 0.08) * a * b * c
    if mediated:
        m = 2 * x[:, 3] - 1
        energy += 1.1 * m + 0.95 * a * m + 1.0 * b * m + 1.2 * c * m
    p = np.exp(energy - energy.max())
    p = p.reshape(q) / p.sum()
    return p.sum(axis=3) if name == "latent_omitted" else p


def _paired(rows, test, layouts):
    w = np.ones(len(rows))
    results = [optimize(l, rows, w) for l in layouts]
    if any(d["status"] != "converged" for _, d in results):
        return np.nan, None
    losses = [Conditionals(l, test).losses(t).mean() for (t, _), l in zip(results, layouts)]
    return float(losses[0] - losses[1]), results[0][0]


def study(repetitions=64, draws=39, seed=377210):
    if repetitions < 2 or draws < 19:
        raise ValueError("use at least two cohorts and nineteen draws")
    opt = NestedOptions(l2=0.025, tolerance=1e-5)
    out = {}
    start = perf_counter()
    for name in CASES:
        p = truth(name)
        shape = p.shape
        edges = tuple(combinations(range(len(shape)), 2))
        layouts = Layout(shape, edges, [], opt), Layout(shape, edges, [(0, 1, 2)], opt)
        positive = rejected = invalid = invalid_draws = 0
        gains = []
        for rep in range(repetitions):
            streams = ReplicateStreams(seed, "nested_study." + name, rep)

            def draw(rng, law, n):
                return np.stack(
                    np.unravel_index(rng.choice(law.size, size=n, p=law.ravel()), law.shape), axis=1
                )

            rng = streams.generator("cohort")
            train, test = draw(rng, p, 120), draw(rng, p, 72)
            gain, theta = _paired(train, test, layouts)
            if theta is None:
                invalid += 1
                continue
            positive += int(gain > 0)
            gains.append(gain)
            null, _ = exact_law(layouts[0], theta)
            exceed = 0
            for i in range(draws):
                rn = streams.generator("fitted_iid_null", i)
                sampled, _ = _paired(draw(rn, null, 120), draw(rn, null, 72), layouts)
                invalid_draws += int(not np.isfinite(sampled))
                exceed += int(
                    not np.isfinite(sampled) or sampled >= gain - max(1.0, abs(gain)) * TIE_EPS
                )
            rejected += int((1 + exceed) / (draws + 1) <= 0.05)
        arr = np.array(gains)
        rng = ReplicateStreams(seed, "nested_study.summary", 0).generator(name)
        means = (
            arr[rng.integers(len(arr), size=(2000, len(arr)))].mean(axis=1)
            if len(arr)
            else np.array([])
        )
        out[name] = {
            "positive_gain": _rate(positive, repetitions),
            "parametric_rejection": _rate(rejected, repetitions),
            "mean_heldout_gain": float(arr.mean()) if len(arr) else None,
            "mean_gain_cohort_bootstrap95": np.quantile(means, [0.025, 0.975]).tolist()
            if len(arr)
            else None,
            "invalid_cohorts": invalid,
            "invalid_refits": invalid_draws,
            "joint_states": p.size,
        }
    return {
        "seed": seed,
        "repetitions": repetitions,
        "draws": draws,
        "train_rows": 120,
        "test_rows": 72,
        "results": out,
        "clade_transfer": clade_transfer(repetitions, seed),
        "elapsed_seconds": perf_counter() - start,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("--repetitions", type=int, default=64)
    cmd.add_argument("--draws", type=int, default=39)
    cmd.add_argument("--seed", type=int, default=377210)
    print(json.dumps(study(**vars(cmd.parse_args())), indent=2))
