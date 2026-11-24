"""Offline categorical sampler validation. No input files or sequence exports."""

import argparse
import json
import platform
from itertools import combinations, pairwise, product
from math import log, prod, sqrt, tanh
from time import perf_counter

import numpy as np

from triplet_coevolution.inference.gibbs_bootstrap import GibbsOptions, prepare_gibbs
from triplet_coevolution.inference.nested_solver import Layout, NestedOptions, exact_law


def _case(dims, edges, strength, seed, n, options):
    layout = Layout(dims, edges, (), NestedOptions())
    theta = np.zeros(layout.size)
    rng = np.random.default_rng(seed)
    for term, arr in zip(layout.terms, layout.arrays(theta), strict=True):
        if strength is None:
            arr[:] = rng.normal(0, 0.2, arr.shape)
        elif len(term) == 2:
            arr[:] = np.array([[1, -1], [-1, 1]]) * strength
    theta = layout.projected(theta)
    started = perf_counter()
    sampler, report = prepare_gibbs(
        layout, theta, options, rows=n, draws=1, folds=1, rng=np.random.default_rng(seed + 1)
    )
    result = {
        "shape": dims,
        "joint_states": prod(dims),
        "status": report["mixing_status"],
        "tv_bound_all_rows": report["all_endpoint_tv_bound_at_this_fold_rate"],
        "failed_observables": report["indicator_failures"],
        "min_ess": min(x["ess"] for x in report["indicator_diagnostics"]),
        "max_finite_split_rhat": max(
            (
                x["split_rhat"]
                for x in report["indicator_diagnostics"]
                if x["split_rhat"] is not None
            ),
            default=None,
        ),
    }
    if result["status"] == "passed":
        rows = sampler.endpoints(n, options, np.random.default_rng(seed + 2))
        if strength is None:
            p, _ = exact_law(layout, theta)
            configs = np.array(list(product(*(range(q) for q in dims))))
            conditional_error = 0.0
            for i in range(len(dims)):
                est = sampler.probabilities(configs, i)
                for k, row in enumerate(configs):
                    index = tuple(slice(None) if j == i else v for j, v in enumerate(row))
                    exact = p[index] / p[index].sum()
                    conditional_error = max(
                        conditional_error, float(np.max(np.abs(exact - est[k])))
                    )
            empirical = np.bincount(np.ravel_multi_index(rows.T, dims), minlength=p.size) / n
            target = p.ravel()
        else:
            # Independent transfer-matrix oracle: in the zero-field open binary
            # chain adjacent spin products are independent, E[s_i s_j]=tanh(J)^d.
            empirical = np.r_[
                np.mean(rows == 1, axis=0),
                np.mean(rows[:, :-1] == rows[:, 1:], axis=0),
                np.mean(rows[:, 0] == rows[:, -1]),
            ]
            target = np.r_[
                np.full(len(dims), 0.5),
                np.full(len(dims) - 1, (1 + tanh(strength)) / 2),
                (1 + tanh(strength) ** (len(dims) - 1)) / 2,
            ]
            conditional_error = None
        se = np.sqrt(target * (1 - target) / n)
        result.update(
            max_moment_error=float(np.max(np.abs(empirical - target))),
            max_standard_error=float(se.max()),
            max_standardized_error=float(np.max(np.abs(empirical - target) / se)),
            simultaneous95_halfwidth=sqrt(log(2 * len(target) / 0.05) / (2 * n))
            + report["single_endpoint_tv_bound"],
            max_conditional_error=conditional_error,
        )
    result["seconds"] = perf_counter() - started
    return result


def comparison_study(seed):
    """Run both refits through the public comparison API on independently generated rows."""
    from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
    from triplet_coevolution.core.coordinates import ReferenceCoordinates
    from triplet_coevolution.inference.nested import compare_nested
    from triplet_coevolution.validation.clade_splits import make_split_plan

    rng = np.random.default_rng(seed)
    data = rng.integers(4, size=(72, 9))
    for i in range(1, 9):
        data[:, i] = np.where(rng.random(72) < 0.25, data[:, i - 1], data[:, i])
    ids = tuple(f"row{j:02d}" for j in range(72))
    parts = []
    for k, role in enumerate("ABC"):
        seq = ["".join("ACGT"[v] for v in row) for row in data[:, 3 * k : 3 * k + 3]]
        if role == "B":
            seq[2] = "?" + seq[2][1:]
            seq[51] = "?" + seq[51][1:]
        parts.append(ComponentMSA(role, "dna", tuple(role + j for j in ids), tuple(seq)))
    table = tuple(TripletMatch(j, *(p.ids[i] for p in parts)) for i, j in enumerate(ids))
    msa = MatchedTripletMSA(*parts, table)
    maps = tuple(
        ReferenceCoordinates(r, "ref" + r, (0, 1, 2), (1, 2, 3), tuple("AAA")) for r in "ABC"
    )
    groups = {j: str(i // 12) for i, j in enumerate(ids)}
    split = make_split_plan(ids, groups, test_groups=["4", "5"])
    reports = {}
    coords = [(r, i) for r in "ABC" for i in range(1, 4)]
    triangle = [(("A", 1), ("B", 1)), (("A", 1), ("C", 1)), (("B", 1), ("C", 1))]
    for key, method in (
        ("small_exact", "exact"),
        ("small_gibbs", "gibbs"),
        ("wide_gibbs", "gibbs"),
    ):
        wide = key == "wide_gibbs"
        edges = triangle + list(pairwise(coords)) if wide else triangle
        sampler = {"method": method}
        if wide:
            sampler["local_i3"] = True
        start = perf_counter()
        result = compare_nested(
            msa,
            maps,
            split,
            options=NestedOptions(l2=0.3, tolerance=1e-5),
            panel=None if wide else {key_0: [1] for key_0 in "ABC"},
            edges=edges,
            triples=[[1, 1, 1]],
            weight_options={"policy": "uniform"},
            null_draws=19 if wide else 63,
            null_sampler=sampler,
            seed=seed,
        )
        fold = result.fold_rows[0]
        out = fold["parametric_null"]
        reports[key] = {
            k: out[k]
            for k in ("status", "configurations", "p_conservative", "valid_draws", "tail_wilson95")
        }
        reports[key].update(
            seconds=perf_counter() - start,
            gain_nats_per_site=fold["gain_nats_per_site"],
            training_rows=fold["complete_training_rows"],
            test_rows=fold["complete_test_rows"],
        )
        if wide:
            reports[key]["local_i3"] = out.get("local_i3")
    return reports


def study(rows=8192, seed=319091):
    opt = GibbsOptions()
    return {
        "seed": seed,
        "rows_per_control": rows,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "settings": opt.metadata(),
        "small_exact": _case((2, 3, 2), tuple(combinations(range(3), 2)), None, seed, rows, opt),
        "large_transfer_matrix": _case(
            (2,) * 18, tuple((i, i + 1) for i in range(17)), 0.23, seed + 10, rows, opt
        ),
        "trapped": _case((2,) * 3, tuple(combinations(range(3), 2)), 14.0, seed + 20, rows, opt),
        "end_to_end": comparison_study(seed + 30),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=319091)
    args = parser.parse_args()
    if not 32 <= args.rows <= 32768:
        parser.error("choose 32 to 32,768 rows")
    print(json.dumps(study(args.rows, args.seed), indent=2, allow_nan=False))
