"""Exchangeability, fitted-null sampling and exact panel oracles using synthetic rows."""

from itertools import combinations, product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.inference.maxent import (
    FrozenSupport,
    MaxEntOptions,
    fit_joint,
    fit_maxent_triples,
    freeze_support,
)
from triplet_coevolution.inference.weights import SequenceWeightResult
from triplet_coevolution.validation.null_models import matching_null, prepare_m2_bootstrap
from triplet_coevolution.validation.nulls import permute_components
from triplet_coevolution.validation.pair_sampler import enumerate_pair_law, fit_pair_panel


def _msa(rows):
    parts = tuple(
        ComponentMSA(
            c, m, tuple(f"{c}{i}" for i in range(len(rows))), tuple(row[k] for row in rows)
        )
        for k, (c, m) in enumerate(zip("ABC", ("protein", "dna", "rna")))
    )
    matches = tuple(
        TripletMatch(f"t{i}", *(comp.ids[i] for comp in parts)) for i in range(len(rows))
    )
    return MatchedTripletMSA(*parts, matches)


def _parity(repeats=1):
    return [tuple("AC"[i] for i in (a, b, a ^ b)) for a, b in product(range(2), repeat=2)] * repeats


def _support(msa):
    base = freeze_support(msa)
    return FrozenSupport(
        base.molecules,
        tuple(tuple((0,) + (1,) * (len(m) - 1) for m in sites) for sites in base.maps),
    )


@pytest.mark.parametrize(
    ("name", "held"),
    [("preserve_ab", "AB"), ("preserve_ac", "AC"), ("preserve_bc", "BC"), ("within_stratum", "A")],
)
def test_named_matching_law_and_sequence_identity(name, held):
    rows = _parity(2) + [("CC", "CC", "CC")]
    rows = [tuple(s if len(s) == 2 else s * 2 for s in row) for row in rows]
    msa = _msa(rows)
    labels = ["u"] * 5 + ["v"] * 3 + ["s"]
    mapping = dict(zip((m.observation_id for m in msa.matches), labels))
    plan = matching_null(msa, name, strata=mapping)
    draw = plan.draw(np.random.default_rng(153))
    meta = plan.metadata()
    assert meta["group_sizes"] == {"u": 5, "v": 3, "s": 1}
    assert meta["singleton_count"] == 1 and meta["movable_fraction"] == 8 / 9
    assert meta["permutation_entropy_bits"] == pytest.approx((3 - len(held)) * np.log2(120 * 6))
    assert meta["sequence_permutation_entropy_bits"] < meta["permutation_entropy_bits"]
    for old, new in zip(msa.components, draw.components):
        assert set(old.ids) == set(new.ids)
        if old.name in held:
            assert old is new
        else:
            assert old.ids != new.ids
        for i, (identifier, sequence) in enumerate(zip(new.ids, new.sequences)):
            old_i = old.ids.index(identifier)
            assert old.sequences[old_i] == sequence and labels[i] == labels[old_i]
    assert [m.observation_id for m in draw.matches] == list(mapping)


def test_unrestricted_matches_legacy_and_degenerate_strata_audited():
    msa = _msa(_parity(2))
    plan = matching_null(msa, "whole_component")
    rng1, rng2 = (np.random.default_rng(17) for _ in range(2))
    assert plan.draw(rng1).matches == permute_components(msa, plan.groups, rng2).matches
    with pytest.raises(ValueError, match="two observations"):
        matching_null(msa, "within_stratum", strata=msa.a.ids)
    with pytest.raises(ValueError, match="unrestricted"):
        matching_null(msa, "whole_component", strata=msa.a.ids)
    with pytest.raises(ValueError, match="explicit"):
        matching_null(msa, "within_stratum")
    with pytest.raises(ValueError, match="exactly"):
        matching_null(msa, "preserve_ab", strata={"wrong": "x"})
    same = _msa([("A", "A", "A")] * 3)
    meta = matching_null(same, "whole_component").metadata()
    assert (
        meta["warning"]
        and meta["sequence_movable_fraction"] == meta["sequence_permutation_entropy_bits"] == 0
    )


def test_bootstrap_preserves_masks_weight_units_margins_and_refits():
    msa = _msa(_parity(4) + [("?", "A", "C"), ("A", "C", "-"), ("C", "C", "C")])
    w = np.array([0.5, 1.5, 2, 1] * 4 + [1, 2, 0])
    support = _support(msa)
    opts = MaxEntOptions(prior_mass=0.8)
    plan = prepare_m2_bootstrap(
        msa, (0, 0, 0), support=support, options=opts, weights=w, weight_policy="fixed_design"
    )
    assert plan.status == "ok" and plan.valid_indices == tuple(range(16))
    assert plan.metadata()["valid_weight"] == 20
    actual = fit_maxent_triples(msa, [(0, 0, 0)], w, support=support, options=opts)[0]
    assert plan.observed == actual.connected_information_bits
    raw = np.bincount(np.array([0, 3, 5, 6] * 4), weights=w[:16], minlength=8).reshape((2,) * 3)
    p3 = (raw + 0.1) / 20.8
    for axis in range(3):
        np.testing.assert_allclose(plan.distribution.sum(axis=axis), p3.sum(axis=axis), atol=1e-9)
    r1, r2 = (np.random.default_rng(44) for _ in range(2))
    counts = plan.draw_counts(r1)
    assert counts.sum() == 20 and np.all(2 * counts == np.floor(2 * counts))
    expected = fit_joint(counts, options=opts, n_valid=16)
    assert plan.draw(r2) == expected.score.connected_information_bits
    total = np.sum([plan.draw_counts(r1) for _ in range(3000)], axis=0)
    np.testing.assert_allclose(total / total.sum(), plan.distribution, atol=0.007)


def test_m2_selection_support_and_invalid_eligibility():
    msa = _msa(_parity())
    support = _support(msa)
    opts = MaxEntOptions()

    def prepare(**kwargs):
        return prepare_m2_bootstrap(msa, (0, 0, 0), support=support, options=opts, **kwargs)

    with pytest.raises(ValueError, match="independently"):
        prepare(selection="same_data")
    with pytest.raises(ValueError, match="unit"):
        prepare(weights=[0.0, 1, 1, 1])
    with pytest.raises(ValueError, match="fixed_design"):
        prepare(weight_policy="equal_component")
    derived = SequenceWeightResult(
        np.ones(4),
        np.ones(4),
        4,
        4,
        ("A", "B", "C"),
        0.8,
        0.5,
        tuple(m.observation_id for m in msa.matches),
        policy="equal_component",
    )
    with pytest.raises(ValueError, match="sequence-derived"):
        prepare(weights=derived, weight_policy="fixed_design")
    with pytest.raises(ValueError, match="disjoint"):
        prepare_m2_bootstrap(
            msa,
            (0, 0, 0),
            options=opts,
            support=freeze_support(msa, training_observations=["t0"], min_state_count=1),
        )
    assert prepare(min_valid=5).status == "insufficient_rows"
    assert prepare(min_valid_weight=4.1).status == "insufficient_weight"
    low = prepare(score="co_information_bits")
    assert low.observed == -1 and low.metadata()["direction"] == "lower"
    assert prepare(score="threebody_rms").status == "undefined_observed_statistic"
    assert np.isnan(prepare(score="threebody_rms").draw(np.random.default_rng(5)))
    rng = np.random.default_rng(2)
    assert any(np.isnan(low.draw(rng)) for _ in range(10))
    dependent = _msa([("A", "A", "A"), ("A", "A", "C"), ("C", "C", "A"), ("C", "C", "C")])
    assert (
        prepare_m2_bootstrap(dependent, (0, 0, 0), options=opts, support=_support(dependent)).status
        == "no_verified_triple_contrast"
    )


def test_exact_mixed_panel_fit_reconstructs_pair_law_and_replays_missingness():
    shape = (2, 3, 2, 2)
    rng = np.random.default_rng(37)
    fields = [rng.normal(0, 0.2, q) for q in shape]
    couplings = {
        (i, j): rng.normal(0, 0.4, (shape[i], shape[j])) for i, j in combinations(range(4), 2)
    }
    law = enumerate_pair_law(fields, couplings)
    coords = np.array(list(product(*(range(q) for q in shape))))
    oracle = np.array(
        [
            np.exp(
                sum(h[r[i]] for i, h in enumerate(fields))
                + sum(v[r[i], r[j]] for (i, j), v in couplings.items())
            )
            for r in coords
        ]
    )
    np.testing.assert_allclose(law.ravel(), oracle / oracle.sum())
    # Adding any a-dependent additive log term to J(a,b) and subtracting it from h(a)
    # is a gauge transformation of the same joint probabilities.
    shifts = np.array([0.31, -0.16])
    equivalent = enumerate_pair_law(
        [fields[0] - shifts, *fields[1:]], couplings | {(0, 1): couplings[(0, 1)] + shifts[:, None]}
    )
    np.testing.assert_allclose(equivalent, law, atol=1e-15, rtol=0)
    rows = np.vstack([coords, [0, 0, 0, -1]])
    weights = np.r_[law.ravel() * 123, 4.25]
    opts = MaxEntOptions(tolerance=1e-12)
    panel = fit_pair_panel(rows, shape, weights=weights, options=opts, weight_policy="fixed_design")
    assert panel.status == "ok" and panel.n_valid == 24 and panel.valid_weight == pytest.approx(123)
    np.testing.assert_allclose(panel.probabilities, law, atol=1e-12, rtol=0)
    counts = np.zeros(shape)
    for _ in range(700):
        sampled = panel.draw(rng)[:-1]
        np.add.at(counts, tuple(sampled.T), 1)
    np.testing.assert_allclose(counts / counts.sum(), law, atol=0.007, rtol=0)
    assert panel.metadata()["configurations"] == 24
    draw, refit = panel.draw_refit(rng)
    assert np.array_equal(draw < 0, rows < 0) and refit.weight_policy == "fixed_design"
    # Complete-case panel fit excludes the last row, while the visible local statistic includes it.
    local = panel.local_score(rows, (0, 1, 2))
    raw = (law * 123).sum(axis=3)
    raw[0, 0, 0] += 4.25
    assert local == pytest.approx(fit_joint(raw, options=opts).score.connected_information_bits)
    new = fit_pair_panel(draw, shape, weights=weights, options=opts, weight_policy="fixed_design")
    assert refit.status == new.status
    if refit.status == "ok":
        np.testing.assert_array_equal(refit.probabilities, new.probabilities)
    with pytest.raises(ValueError, match="too many configurations"):
        fit_pair_panel(rows, (20, 20, 20, 20))
    with pytest.raises(ValueError, match="valid declared code"):
        fit_pair_panel(rows, (1, 3, 2, 2))


def test_pairwise_hidden_site_can_generate_local_higher_order_signal():
    spins = np.array([-1, 1])
    J = np.outer(spins, spins)
    law = enumerate_pair_law(
        [0.2 * spins, 0.1 * spins, -0.2 * spins, 0.5 * spins],
        {(0, 3): 0.9 * J, (1, 3): 0.8 * J, (2, 3): -1.0 * J},
    )
    local = fit_joint(law.sum(axis=3)).score
    assert local.status == "ok" and local.connected_information_bits > 0.003
    coords = np.array(list(product(range(2), repeat=4)))
    panel = fit_pair_panel(coords, (2,) * 4, weights=law.ravel(), weight_policy="fixed_design")
    assert panel.status == "ok"
    np.testing.assert_allclose(panel.probabilities, law, rtol=0, atol=1e-8)
    assert panel.local_score(coords, (0, 1, 2)) == pytest.approx(local.connected_information_bits)


def test_structural_zero_stays_impossible_in_bootstrap_and_forced_panel_abstains():
    shape = (2, 2, 3)
    mask = np.ones(shape, dtype=bool)
    mask[1, 1, 2] = False
    coordinates = np.argwhere(mask)
    msa = _msa([("AC"[a], "AC"[b], "ACG"[c]) for a, b, c in coordinates])
    default = _support(msa)
    support = FrozenSupport(default.molecules, (*default.maps[:2], ((0, 1, 2, 2),)))
    boot = prepare_m2_bootstrap(
        msa, (0, 0, 0), support=support, options=MaxEntOptions(prior_mass=1), allowed=mask
    )
    assert boot.status == "ok" and boot.distribution[1, 1, 2] == 0
    rng = np.random.default_rng(98)
    for _ in range(40):
        assert not boot.draw_counts(rng)[1, 1, 2]
    mask[1, 1, 2] = True
    assert not boot.allowed[1, 1, 2]
    forced = np.array([(a, a, b) for a, b in product(range(2), repeat=2)])
    panel = fit_pair_panel(forced, (2,) * 3)
    assert panel.status == "ok" and np.isnan(panel.local_score(forced, (0, 1, 2)))
