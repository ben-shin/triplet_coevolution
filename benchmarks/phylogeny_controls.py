"""Prespecified clade-confounding, within-block signal and held-out transfer study."""

import argparse
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from null_calibration import _rate

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.mi import weighted_mi
from triplet_coevolution.validation.clade_stability import evaluate_stability
from triplet_coevolution.validation.null_models import matching_null
from triplet_coevolution.validation.null_runner import TIE_EPS, ReplicateStreams
from triplet_coevolution.validation.tree_topology import compare_trees, load_tree

CASES = ("iid", "clade_only", "persistent", "training_clades_only")


def _tree(path, ids, groups):
    by_group = [
        [key + ":0.1" for key, g in zip(ids, groups, strict=True) if g == label]
        for label in sorted(set(groups))
    ]
    path.write_text(
        "(" + ",".join("(" + ",".join(leaves) + "):1.0" for leaves in by_group) + ");",
        encoding="ascii",
    )
    return load_tree(path, ids)


def _make_msa(ids, codes):
    parts = [
        ComponentMSA(
            role,
            "dna",
            tuple(role + "_" + key for key in ids),
            tuple("".join("AC"[x] for x in row) for row in arr),
        )
        for role, arr in zip("ABC", codes, strict=True)
    ]
    return MatchedTripletMSA(
        *parts, tuple(TripletMatch(key, *(c.ids[i] for c in parts)) for i, key in enumerate(ids))
    )


def study(repetitions=128, draws=59, seed=724261):
    if repetitions < 1 or draws < 19:
        raise ValueError("at least one independent repetition and 19 draws required")
    nclades, nper, sites = 8, 16, 3
    n = nclades * nper
    labels = np.repeat(np.arange(nclades), nper)
    ids = tuple(f"t{g}_{i}" for g in range(nclades) for i in range(nper))
    alternate = (labels + np.tile(np.arange(nper), nclades)) % nclades
    with TemporaryDirectory() as d:
        species = _tree(Path(d) / "species.nwk", ids, labels)
        component = _tree(Path(d) / "discordant.nwk", ids, alternate)
    organism_groups = species.clades_at_depth(1)
    alternate_groups = component.clades_at_depth(1)
    tree_cmp = compare_trees(species, component)
    rates = {}
    maps = tuple(
        ReferenceCoordinates(
            role,
            role + "-synthetic",
            tuple(range(sites)),
            tuple(range(1, sites + 1)),
            tuple("A" * sites),
        )
        for role in "ABC"
    )
    start = perf_counter()
    for name in CASES:
        counts = dict.fromkeys(
            (
                "whole",
                "organism_clade",
                "discordant_clade",
                "training_selected",
                "selected_and_sole_best_heldout",
            ),
            0,
        )
        for rep in range(repetitions):
            streams = ReplicateStreams(seed, "clade_study." + name, rep)
            rng = streams.generator("cohort")
            a, b, c = rng.integers(0, 2, (3, n, sites))
            pi = np.full(n, 0.5) if name == "iid" else np.where(labels % 2, 0.85, 0.15)
            a[:, 0] = (rng.random(n) < pi).astype(int)
            b[:, 0] = (rng.random(n) < pi).astype(int)
            if name in ("persistent", "training_clades_only"):
                coupled = labels < nclades if name == "persistent" else labels < nclades - 1
                b[coupled, 0] = a[coupled, 0] ^ (rng.random(coupled.sum()) < 0.08)
            msa = _make_msa(ids, (a, b, c))
            result = evaluate_stability(
                msa,
                maps,
                organism_groups,
                test_groups=[organism_groups[ids[-1]]],
                statistics=("mi_bits",),
                top_k=1,
            )
            pair = next(row for row in result.pair_rows if row["interface"] == "AB")
            fold = next(row for row in result.fold_rows if row["interface"] == "AB")
            target_selected = (pair["A_column"], pair["B_column"]) == (1, 1)
            counts["training_selected"] += int(target_selected)
            counts["selected_and_sole_best_heldout"] += int(
                target_selected and pair["test_rank"] == 1 and fold["test_cutoff_tie_size"] == 1
            )
            orig = weighted_mi(a[:, 0], b[:, 0], 2, 2)
            for key, groups in (
                ("whole", None),
                ("organism_clade", organism_groups),
                ("discordant_clade", alternate_groups),
            ):
                law = matching_null(
                    msa, "whole_component" if groups is None else "within_stratum", strata=groups
                )
                random = streams.generator(key)
                exceed = 0
                for _ in range(draws):
                    i, j, _ = law.row_orders(random)
                    exceed += (
                        weighted_mi(a[i, 0], b[j, 0], 2, 2) >= orig - max(1.0, abs(orig)) * TIE_EPS
                    )
                counts[key] += int((1 + exceed) / (draws + 1) <= 0.05)
        rates[name] = {k: _rate(int(v), repetitions) for k, v in counts.items()}
    return {
        "repetitions": repetitions,
        "draws": draws,
        "seed": seed,
        "groups": nclades,
        "observations_per_group": nper,
        "sites_per_role": sites,
        "alpha": 0.05,
        "organism_tree_sha256": species.source_sha256,
        "alternate_tree_sha256": component.source_sha256,
        "tree_sensitivity": tree_cmp,
        "results": rates,
        "elapsed_seconds": perf_counter() - start,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--draws", type=int, default=59)
    parser.add_argument("--seed", type=int, default=724261)
    print(json.dumps(study(**vars(parser.parse_args())), indent=2))
