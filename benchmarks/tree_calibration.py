"""Reproducible calibration of a known independent-site law on a synthetic tree."""

import argparse
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from null_calibration import _rate

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.inference.mi import weighted_mi
from triplet_coevolution.validation.null_models import matching_null
from triplet_coevolution.validation.tree_model import compare_trees, load_tree, prepare_tree_null


def study(repetitions: int, draws: int, seed: int) -> dict:
    """Use exchangeable site columns to simulate one observed and B null pairs per cohort."""
    if repetitions < 1 or draws < 19:
        raise ValueError("use positive repetitions and at least 19 draws for a 5% threshold")
    obs = tuple(f"s{k}" for k in range(64))
    clades = [
        "(" + ",".join(f"{key}:0.035" for key in obs[i : i + 8]) + "):0.8" for i in range(0, 64, 8)
    ]
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.nwk"
        path.write_text("(" + ",".join(clades) + ");", encoding="ascii")
        tree = load_tree(path, obs)
        path.write_text("(" + ",".join(f"{t}:0.835" for t in obs) + ");", encoding="ascii")
        star = load_tree(path, obs)
    width = draws + 1
    parts = tuple(
        ComponentMSA(role, "dna", tuple(role + t for t in obs), ("A" * width,) * len(obs))
        for role in "ABC"
    )
    msa = MatchedTripletMSA(
        *parts, tuple(TripletMatch(t, *(c.ids[i] for c in parts)) for i, t in enumerate(obs))
    )
    probabilities = {r: [0.5, 0.5, 0.0, 0.0] for r in "ABC"}
    model = prepare_tree_null(msa, tree, frequency_policy="supplied", frequencies=probabilities)
    misspecified = prepare_tree_null(
        msa, star, frequency_policy="supplied", frequencies=probabilities
    )
    perm = matching_null(msa, "whole_component")
    tree_reject = shuffle_reject = plugin_reject = star_reject = copy_reject = 0
    start = perf_counter()
    for rep in range(repetitions):
        simulation, assignment, plugin_seed, star_seed = np.random.SeedSequence([seed, rep]).spawn(
            4
        )
        # Columns use identical externally specified frequencies, independently simulated roots
        # and branch histories. Column zero serves as the observation in each new cohort.
        latent = model.draw_codes(np.random.default_rng(simulation))
        a, b, _ = latent
        all_mi = np.array([weighted_mi(x, y, 2, 2) for x, y in zip(a.T, b.T)])
        tol = 64 * np.finfo(float).eps
        limit = all_mi[0] - tol
        tree_reject += int((1 + np.count_nonzero(all_mi[1:] >= limit)) / (draws + 1) <= 0.05)
        # Perfect coupling via identical full histories retains both sites' F81 marginal laws.
        copied = weighted_mi(a[:, 0], a[:, 0], 2, 2)
        copy_reject += int((1 + np.count_nonzero(all_mi[1:] >= copied - tol)) / (1 + draws) <= 0.05)
        # Freeze tip-frequency estimates from the actual observed column (equal row weights).
        # Repeated identical columns only batch independent parametric draws.
        repeated = tuple(
            ComponentMSA(c.name, c.molecule, c.ids, tuple("AC"[k] * width for k in v[:, 0]))
            for c, v in zip(parts, latent)
        )
        fitted = prepare_tree_null(
            MatchedTripletMSA(*repeated, msa.matches), tree, frequency_policy="empirical"
        )
        for label, alt, stream in (
            ("plugin", fitted, plugin_seed),
            ("star", misspecified, star_seed),
        ):
            u, v, _ = alt.draw_codes(np.random.default_rng(stream))
            extreme = sum(
                weighted_mi(x, y, 2, 2) >= limit for x, y in zip(u[:, :draws].T, v[:, :draws].T)
            )
            rejection = int((1 + extreme) / (1 + draws) <= 0.05)
            if label == "plugin":
                plugin_reject += rejection
            else:
                star_reject += rejection
        rng = np.random.default_rng(assignment)
        extreme = 0
        for _ in range(draws):
            i, j, _ = perm.row_orders(rng)
            extreme += int(weighted_mi(a[i, 0], b[j, 0], 2, 2) >= limit)
        shuffle_reject += int((1 + extreme) / (draws + 1) <= 0.05)
    return {
        "seed": seed,
        "cohorts": repetitions,
        "draws": draws,
        "depth": len(obs),
        "clades": 8,
        "tips_per_clade": 8,
        "stem": 0.8,
        "terminal": 0.035,
        "pi": [0.5, 0.5, 0, 0],
        "alpha": 0.05,
        "tree_sha256": tree.source_sha256,
        "known_tree": _rate(tree_reject, repetitions),
        "empirical_tip_frequencies": _rate(plugin_reject, repetitions),
        "known_tree_planted_identical_history": _rate(copy_reject, repetitions),
        "misspecified_star": _rate(star_reject, repetitions),
        "tree_comparison": compare_trees(tree, star),
        "unrestricted": _rate(shuffle_reject, repetitions),
        "elapsed_seconds": perf_counter() - start,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=320)
    parser.add_argument("--draws", type=int, default=99)
    parser.add_argument("--seed", type=int, default=982011)
    print(json.dumps(study(**vars(parser.parse_args())), indent=2))
