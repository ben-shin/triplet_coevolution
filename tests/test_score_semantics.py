"""Score registry, APC domains, ranking, and one-sided permutation regression cases."""

import csv
import json
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import build_reference_map
from triplet_coevolution.inference import advanced
from triplet_coevolution.inference.mi import apc, triplet_mi
from triplet_coevolution.inference.potts import DCABlock, direct_couplings
from triplet_coevolution.inference.scores import (
    dca_with_apc_policy,
    get_score,
    mi_with_apc_policy,
    pair_coverage,
    ranked_indices,
)
from triplet_coevolution.inference.threebody import fit_triples
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.validation.nulls import permute_components
from triplet_coevolution.validation.phylogeny import exchangeability_groups


def _mixed_alignment():
    # One XOR site plus variable, redundant and missing states in different alphabets.
    rows = [
        ("KE"[a] + ("-" if index == 0 else "ST"[r]) + "YF"[a ^ r],
         "AT"[b] + "CT"[r] + "AG"[b ^ r],
         "GU"[a ^ b] + "AC"[r] + "UA"[a ^ r])
        for index, (a, b, r) in enumerate(product(range(2), repeat=3))
    ]
    comps = tuple(
        ComponentMSA(name, molecule, tuple(f"{name}{i}" for i in range(len(rows))),
                     tuple(row[col] for row in rows))
        for col, (name, molecule) in enumerate(zip("ABC", ("protein", "dna", "rna")))
    )
    matches = [TripletMatch(str(i), *ids) for i, ids in enumerate(zip(*(c.ids for c in comps)))]
    msa = MatchedTripletMSA(*comps, matches)
    return msa, tuple(build_reference_map(c, c.ids[0]) for c in comps), sequence_weights(msa)


def _rows(directory, name):
    with (directory / f"{name}.tsv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_registry_rejects_incompatible_units_method_and_block():
    score = get_score("mi_bits", "mi", units="bits")
    with pytest.raises(ValueError, match="units"):
        get_score("mi_bits", units="nats")
    with pytest.raises(ValueError, match="belongs"):
        get_score("threebody_rms", "dca")
    with pytest.raises(ValueError, match="unknown score"):
        get_score("absolute_co_info")
    with pytest.raises(ValueError, match="incompatible"):
        score.values(object())
    spec = get_score("co_information_bits").metadata(threebody_model="global")
    assert spec["direction"] == "lower" and spec["backend"] == "weighted empirical entropies"
    assert "entire fitted panel" in spec["missing_policy"]


def test_rank_ties_multidimensional_cutoff_zeros_and_missing():
    scores = np.array([[1, np.nan, 3, 3], [3, np.nan, -1, 0]])
    assert list(ranked_indices(scores, "dca_apc", 2)) == [(2, 1, 1), (3, 2, 1)]
    assert list(ranked_indices(scores, "dca_apc")) == [
        (2, 1, 1), (3, 2, 1), (4, 3, 1), (0, 4, 4), (7, 5, 5), (6, 6, 6)
    ]
    assert list(ranked_indices(scores, "co_information_bits", 2)) == [(6, 1, 1), (7, 2, 2)]
    assert not list(ranked_indices(np.full(3, np.nan), "dca_apc"))
    assert list(ranked_indices(np.zeros(3), "dca_apc")) == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]


@pytest.mark.parametrize("name,sign", [("dca_apc", -1), ("co_information_bits", 1)])
def test_chunked_top_k_matches_full_sorted_indices(name, sign):
    rng = np.random.default_rng(19)
    values = rng.integers(-3, 4, 140_000).astype(float)
    values[::311] = np.nan
    values[131_075] = 9 * (-sign)
    expected = np.argsort(sign * values, kind="stable")[:51]
    result = list(ranked_indices(values, name, limit=51))
    assert [i for i, _, _ in result] == list(expected)
    assert result[0][2] == 1
    assert all(competition == 2 for _, _, competition in result[1:])


def test_corrected_dca_top_k_outside_raw_top_k_and_single_complete_sink(tmp_path, monkeypatch):
    msa, maps, weights = _mixed_alignment()
    raw = np.array([[10, 9, 8], [9, 8, 7], [2, 1, 6]], dtype=float)
    blocks = {
        pair: DCABlock(raw, apc(raw), np.full((3, 3), 8), np.ones((3, 3)))
        for pair in ("AB", "AC", "BC")
    }
    calls = []

    def fitted(*args, **kwargs):
        calls.append("fit")
        return blocks

    monkeypatch.setattr(advanced, "direct_couplings", fitted)
    captured = []
    manifest = advanced.write_advanced_results(
        tmp_path / "default", msa, maps, weights, dca=True, max_pairs=1,
        score_sink=lambda mi, dca: captured.append((mi, dca)),
    )
    assert calls == ["fit"] and len(captured) == 1 and captured[0][1] is blocks
    assert captured[0][0].ab.mi_bits.shape == raw.shape
    assert manifest["ranking"]["dca_ab"] == {
        "total": 9, "eligible": 9, "finite": 9, "excluded_coordinates": 0,
        "below_min_valid": 0, "nonfinite_after_support": 0, "exported": 1, "complete": False,
    }
    assert manifest["settings"]["dca_statistic"] == "dca_apc"
    json.dumps(manifest, allow_nan=False)
    row, = _rows(tmp_path / "default", "dca_ab")
    assert (row["x_original_col"], row["y_original_col"], row["statistic"]) == ("3", "3", "dca_apc")
    assert float(row["score"]) == pytest.approx(2.85)
    advanced.write_advanced_results(
        tmp_path / "legacy", msa, maps, weights, dca=True, max_pairs=None, dca_score="dca_frobenius",
    )
    ranked = _rows(tmp_path / "legacy", "dca_ab")
    assert len(ranked) == 9
    assert (ranked[0]["x_col"], ranked[0]["y_col"], ranked[0]["score"]) == ("1", "1", "10.0")
    assert float(ranked[-1]["dca_apc"]) == pytest.approx(-1.7)


def test_reference_policy_corrects_only_supported_mapped_axes_with_full_shape():
    msa, maps, weights = _mixed_alignment()
    result = triplet_mi(msa, weights)
    # A's middle column is missing in its coordinate reference. The last raw
    # entry is below the fitted min_valid threshold and already marked NaN.
    raw = np.array([[2., 1., 3.], [9., 3., 5.], [4., 0., np.nan]])
    n = np.full(raw.shape, 8)
    n[2, 2] = 1
    ab = replace(result.ab, mi_bits=raw, mi_apc=apc(raw), n_valid=n)
    base = replace(result, ab=ab)
    scoped = mi_with_apc_policy(base, maps, "reference")
    expected = np.array([[2.-2.*3./2., 1.-2.*.5/2., 3.-2.*3./2.],
                         [np.nan, np.nan, np.nan], [4.-2.*3./2., 0.-2.*.5/2., np.nan]])
    np.testing.assert_allclose(scoped.ab.mi_apc, expected, equal_nan=True)
    assert scoped.ab.mi_bits is raw
    np.testing.assert_array_equal(mi_with_apc_policy(scoped, maps, "retained").ab.mi_apc, apc(raw))
    assert pair_coverage(
        ab, *maps[:2], "mi_apc", apc_universe="reference", max_pairs=None,
    ) == {
        "total": 9, "eligible": 6, "finite": 5, "excluded_coordinates": 3,
        "below_min_valid": 1, "nonfinite_after_support": 0, "exported": 5, "complete": True,
    }
    # All-gap coordinate row yields an empty corrected domain, raw matrices survive.
    absent = replace(maps[0], reference_positions=(None,) * 3, symbols=("-",) * 3)
    empty = mi_with_apc_policy(base, (absent, *maps[1:]), "reference")
    assert np.isnan(empty.ab.mi_apc).all() and np.isnan(empty.ac.mi_apc).all()
    assert np.isfinite(empty.ab.mi_bits).any()
    zero = np.zeros((3, 3))
    all_blocks = {key: DCABlock(zero, zero, n, n) for key in ("AB", "AC", "BC")}
    remapped = dca_with_apc_policy(all_blocks, maps, "reference")
    assert (remapped["AB"].corrected[[0, 2]] == 0).all()
    assert np.isnan(remapped["AB"].corrected[1]).all()


@pytest.mark.parametrize("statistic", ["dca_apc", "dca_frobenius"])
def test_mixed_null_refits_selected_stat_and_one_sided_co_information(tmp_path, statistic):
    msa, maps, w = _mixed_alignment()
    draws, seed = 9, 102
    summary = advanced.write_advanced_results(
        tmp_path, msa, maps, w, dca=True, threebody=True, permutations=draws, seed=seed,
        dca_score=statistic, threebody_score="co_information_bits", max_pairs=1,
        apc_universe="reference",
    )
    dca_rows = _rows(tmp_path, "permutation_dca")
    co_rows = _rows(tmp_path, "permutation_threebody")
    dca = next(row for row in dca_rows if (
        row["pair"], row["x_original_col"], row["y_original_col"]
    ) == ("AB", "1", "2"))
    co = next(row for row in co_rows if all(row[f"{p}_original_col"] == "1" for p in "abc"))
    assert dca["statistic"] == statistic and dca["direction"] == "higher"
    assert co["statistic"] == "co_information_bits" and co["direction"] == "lower"
    assert float(co["score"]) == pytest.approx(-1)
    assert summary["hypotheses"]["dca"] == (21 if statistic == "dca_apc" else 27)
    assert len(_rows(tmp_path, "dca_ab")) == 1
    assert _rows(tmp_path, "threebody_abc")[0]["score"] == co["score"]
    rng = np.random.default_rng(seed)
    groups = exchangeability_groups(msa.n_observations, None)

    def statistics(data):
        weights = sequence_weights(data)
        raw = direct_couplings(data, weights)["AB"].frobenius
        sub = raw[[0, 2]]
        expected = (sub[0, 1] - sub[0].mean() * sub[:, 1].mean() / sub.mean()
                    if statistic == "dca_apc" else raw[0, 1])
        co_info = fit_triples(data, [(0, 0, 0)], weights)[0].co_information_bits
        return expected, co_info

    actual, signed = statistics(msa)
    assert float(dca["score"]) == pytest.approx(actual)
    null = [statistics(permute_components(msa, groups, rng)) for _ in range(draws)]
    tol = 64 * np.finfo(float).eps
    high = sum(value >= actual - tol * max(1, abs(actual)) for value, _ in null)
    low = sum(value <= signed + tol * max(1, abs(signed)) for _, value in null)
    assert float(dca["permutation_p"]) == (1 + high) / (draws + 1)
    assert float(co["permutation_p"]) == (1 + low) / (draws + 1) < 1
