"""Preserve v1.0's reduction order, including permuted observed-state support."""

from itertools import combinations

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.inference import advanced
from triplet_coevolution.inference.mi import apc, triplet_mi
from triplet_coevolution.inference.scores import mi_with_apc_policy
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.msa.masking import apply_triplet_masks


def _fixture():
    rng = np.random.default_rng(1213)
    components = []
    for name, molecule, symbols in (
        ("A", "protein", "A-KF?"), ("B", "dna", "AC-T?"), ("C", "rna", "AG-U?")
    ):
        rows = rng.choice(list(symbols), size=(47, 5))
        # Every observation retains a canonical site for sequence reweighting.
        rows[:, 0] = rng.choice([s for s in symbols if s not in "-?"], size=len(rows))
        seqs = tuple("".join(row) for row in rows)
        ids = tuple(f"{name}{i}" for i in range(len(seqs)))
        components.append(ComponentMSA(name, molecule, ids, seqs))
    matches = tuple(
        TripletMatch(str(i), *ids) for i, ids in enumerate(zip(*(comp.ids for comp in components)))
    )
    original = MatchedTripletMSA(*components, matches)
    masks = ([True, False, True, False, True], [True, True, False, True, False], [True] * 5)
    subset = apply_triplet_masks(original, *masks)
    maps = tuple(
        subset_reference_map(build_reference_map(comp, comp.ids[0]), np.flatnonzero(mask))
        for comp, mask in zip(original.components, masks)
    )
    return subset.msa, maps


def _scalar_block(a, b, w, include_gaps, min_valid):
    # Arithmetic order in c511759: masked weights before summation, row and
    # column marginals from joint probabilities, then sum only the occupied terms.
    qs = tuple(len(CANONICAL[comp.molecule]) + int(include_gaps) for comp in (a, b))
    shape = (a.n_sites, b.n_sites)
    bits = np.full(shape, np.nan)
    counts = np.empty(shape, dtype=int)
    mass = np.empty(shape)
    for i, j in np.ndindex(shape):
        x, y = a.matrix[:, i], b.matrix[:, j]
        valid = (w > 0) & (x >= 0) & (x < qs[0]) & (y >= 0) & (y < qs[1])
        positive = w[valid]
        counts[i, j] = positive.size
        mass[i, j] = positive.sum()
        if positive.size >= min_valid:
            ix = np.ravel_multi_index((x[valid], y[valid]), qs)
            joint = np.bincount(ix, weights=positive, minlength=qs[0] * qs[1]).reshape(qs)
            joint /= mass[i, j]
            null = np.outer(joint.sum(axis=1), joint.sum(axis=0))
            occupied = joint > 0
            terms = joint[occupied] * np.log2(joint[occupied] / null[occupied])
            bits[i, j] = max(0.0, float(terms.sum()))
    return bits, counts, mass


def _assert_scalar(result, msa, weights, min_valid, include_gaps):
    for a, b in combinations(msa.components, 2):
        actual = getattr(result, (a.name + b.name).lower())
        bits, counts, mass = _scalar_block(a, b, weights, include_gaps, min_valid)
        np.testing.assert_array_equal(actual.mi_bits, bits)
        np.testing.assert_array_equal(actual.n_valid, counts)
        np.testing.assert_array_equal(actual.valid_weight, mass)
        np.testing.assert_array_equal(actual.mi_apc, apc(bits))


@pytest.mark.parametrize("include_gaps", [False, True])
def test_scalar_kernel_exact_on_masked_mixed_fractional_frequencies(include_gaps):
    msa, maps = _fixture()
    w = 1.0 / (1 + np.arange(msa.n_observations) % 9)
    w[::7] = 0.0
    scalar = triplet_mi(msa, w, kernel="v1_scalar", include_gaps=include_gaps, min_valid=13)
    _assert_scalar(scalar, msa, w, 13, include_gaps)
    assert scalar.kernel == mi_with_apc_policy(scalar, maps, "reference").kernel == "v1_scalar"
    assert triplet_mi(msa, w).kernel == "batched"


def test_kernel_inherited_by_all_null_draws(tmp_path, monkeypatch):
    msa, maps = _fixture()
    w = sequence_weights(msa)
    observed = triplet_mi(msa, w, kernel="v1_scalar", include_gaps=True)
    calls = []

    def inspect_draw(data, weights, **kwargs):
        draw = triplet_mi(data, weights, **kwargs)
        assert draw.kernel == "v1_scalar" and draw.include_gaps
        _assert_scalar(draw, data, weights.weights, 2, True)
        calls.append(draw)
        return draw

    monkeypatch.setattr(advanced, "triplet_mi", inspect_draw)
    run = advanced.write_advanced_results(
        tmp_path, msa, maps, w, mi_result=observed, permutations=4, apc_universe="reference"
    )
    assert len(calls) == 4 and run["settings"]["mi_kernel"] == "v1_scalar"
    with pytest.raises(ValueError, match="same kernel"):
        advanced.write_advanced_results(tmp_path, msa, maps, w, mi_result=observed, mi_kernel="batched")
    with pytest.raises(ValueError, match="kernel"):
        triplet_mi(msa, w, kernel="wrong")
