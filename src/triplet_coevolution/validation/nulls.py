"""Matched-component permutation tests with online exceedance counts."""

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA, TripletMatch
from triplet_coevolution.validation.fdr import bh_adjust
from triplet_coevolution.validation.phylogeny import exchangeability_groups


def permute_components(msa, groups, rng):
    """Hold A fixed and independently shuffle whole B and C rows within groups."""
    components = [msa.a]
    for component in (msa.b, msa.c):
        perm = np.arange(msa.n_observations)
        for group in groups:
            perm[group] = rng.permutation(group)
        components.append(component.take_ids([component.ids[i] for i in perm]))
    matches = tuple(
        TripletMatch(original.observation_id, a, b, c)
        for original, a, b, c in zip(msa.matches, *(m.ids for m in components))
    )
    return MatchedTripletMSA(*components, matches)


def permutation_test(msa, observed, evaluate, *, permutations: int, seed: int, strata=None):
    """Calibrate full score vectors under independent matching within strata.

    The scorer is refit on every permutation. An undefined null draw counts
    as an exceedance, and untestable observed coordinates enter BH at p=1.
    """
    if permutations < 1:
        raise ValueError("permutations must be positive")
    groups = exchangeability_groups(msa.n_observations, strata)
    rng = np.random.default_rng(seed)
    counts = {key: np.zeros(len(vals), dtype=np.int64) for key, vals in observed.items()}
    for _ in range(permutations):
        simulated = evaluate(permute_components(msa, groups, rng))
        for key, actual in observed.items():
            null = simulated[key]
            # Count numerical ties conservatively, including ties from row ordering.
            tol = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(actual))
            counts[key] += ~np.isfinite(null) | (null >= actual - tol)
    result = {}
    for key, actual in observed.items():
        p = (counts[key] + 1) / (permutations + 1)
        p[~np.isfinite(actual)] = 1.0
        result[key] = p, bh_adjust(p)
    return result
