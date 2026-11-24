"""Exact and adversarial finite-distribution tests for the local M1/M2/M3 hierarchy."""

import json
from itertools import product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.inference.maxent import (
    FrozenSupport,
    MaxEntOptions,
    fit_joint,
    fit_maxent_grid,
    fit_maxent_triples,
    fit_pair_marginals,
    freeze_support,
)


def test_uniform_xor_never_treats_sampling_zeros_as_structural_zeros():
    counts = np.zeros((2, 2, 2))
    for a, b in product(range(2), repeat=2):
        counts[a, b, a ^ b] = 1
    fit = fit_joint(counts, n_valid=4)
    s = fit.score
    assert s.status == "ok" and s.active_pair_face_cells == s.possible_cells == 8
    np.testing.assert_allclose(fit.p1, np.full((2, 2, 2), 1 / 8))
    np.testing.assert_allclose(fit.p2, np.full((2, 2, 2), 1 / 8))
    assert s.h1_bits == s.h2_bits == 3 and s.h3_bits == 2
    assert s.connected_information_bits == pytest.approx(1, abs=1e-15)
    assert s.co_information_bits == s.target_co_information_bits == -1
    assert np.isnan(s.threebody_rms)
    assert s.deviance_nats == pytest.approx(8 * np.log(2))
    assert s.target_deviance_nats == pytest.approx(s.deviance_nats)
    assert s.expected_occupied_m2 == pytest.approx(8 * (1 - (7 / 8) ** 4))
    assert s.observed_cells == 4 and s.information_status == "sparse_observed_support"
    smoothed = fit_joint(counts, options=MaxEntOptions(prior_mass=1)).score
    per = fit_joint(counts, options=MaxEntOptions(prior_mode="per_cell", prior_mass=0.125)).score
    assert smoothed.connected_information_bits == per.connected_information_bits
    assert smoothed.prior_total == 1 and smoothed.prior_fraction == 0.2
    assert smoothed.prior_data_ratio == 0.25 and smoothed.co_information_bits == -1
    assert smoothed.target_co_information_bits == pytest.approx(
        -smoothed.connected_information_bits
    )
    assert smoothed.deviance_nats != pytest.approx(smoothed.target_deviance_nats)


@pytest.mark.parametrize("edges", [(0.28, 0, 0.49), (0.52, -0.39, 0.70)])
def test_positive_pairwise_chain_and_triangle_recover_exact_generating_law(edges):
    xyz = np.indices((2, 2, 2)) * 2 - 1
    x, y, z = xyz
    ab, ac, bc = edges
    p = np.exp(0.3 * x - 0.6 * y + 0.4 * z + ab * x * y + ac * x * z + bc * y * z)
    p /= p.sum()
    fit = fit_joint(70 * p, options=MaxEntOptions(tolerance=1e-12))
    assert fit.score.status == "ok"
    np.testing.assert_allclose(fit.p2, p, rtol=0, atol=1e-12)
    assert fit.score.connected_information_bits < 1e-19
    assert abs(fit.score.co_information_bits) > 1e-5
    assert abs(fit.score.h2_bits - fit.score.h3_bits) < 1e-12


def test_nonuniform_independence_and_planted_binary_k():
    expected = np.einsum("i,j,k->ijk", [0.2, 0.8], [0.3, 0.2, 0.5], [0.4, 0.1, 0.3, 0.2])
    result = fit_joint(expected, options=MaxEntOptions(tolerance=1e-12))
    np.testing.assert_allclose(result.p1, expected)
    np.testing.assert_allclose(result.p2, expected)
    assert result.score.connected_information_bits < 1e-25
    k = 0.72
    parity = np.prod(2 * np.indices((2, 2, 2)) - 1, axis=0)
    p = np.exp(k * parity)
    p /= p.sum()
    fit = fit_joint(55 * p)
    positive = 1 / (1 + np.exp(-2 * k))
    expected_bits = 1 + positive * np.log2(positive) + (1 - positive) * np.log2(1 - positive)
    assert fit.score.connected_information_bits == pytest.approx(expected_bits)
    assert fit.score.threebody_rms == pytest.approx(k)
    assert fit.score.co_information_bits == pytest.approx(-expected_bits)


def test_hidden_mixture_has_marginal_residual_without_generating_triple():
    # The three visible variables are conditionally independent given the latent class.
    # Nonlinear marginalization of that class can produce a local three-factor contrast.
    mix = np.zeros((2, 2, 2))
    for mass, probs in ((0.2, (0.95, 0.75, 0.8)), (0.8, (0.12, 0.24, 0.15))):
        mix += mass * np.einsum("i,j,k->ijk", *([1 - p, p] for p in probs))
    fit = fit_joint(mix, options=MaxEntOptions(tolerance=1e-12))
    assert fit.score.status == "ok" and fit.score.connected_information_bits > 1e-4
    assert fit.score.identity_error_bits <= 1e-12
    for axis in range(3):
        np.testing.assert_allclose(mix.sum(axis=axis), fit.p2.sum(axis=axis), rtol=0, atol=1e-12)


def test_binary_pair_fit_against_independent_scalar_stationarity_oracle():
    # Matching all binary pair marginals leaves one parameter t multiplying parity.
    # Maximum entropy has zero parity log contrast, whose derivative in t is positive.
    rng = np.random.default_rng(5421)
    sign = (-1) ** np.indices((2, 2, 2)).sum(axis=0)
    for _ in range(16):
        counts = rng.lognormal(sigma=1.3, size=(2, 2, 2))
        for mass in (0.0, 1.0):
            joint = (counts + mass / 8) / (counts.sum() + mass)
            lower, upper = -min(joint[sign > 0]), min(joint[sign < 0])
            for _step in range(60):
                midpoint = (lower + upper) / 2
                guess = joint + midpoint * sign
                if np.sum(sign * np.log(guess)) > 0:
                    upper = midpoint
                else:
                    lower = midpoint
            pair = joint + (lower + upper) / 2 * sign
            expected_bits = np.sum(joint * np.log2(joint / pair))
            opts = MaxEntOptions(tolerance=1e-12, prior_mass=mass)
            fit = fit_joint(counts, options=opts)
            assert fit.score.status == "ok" and np.isnan(fit.score.expected_occupied_m2)
            np.testing.assert_allclose(fit.p2, pair, rtol=0, atol=2e-12)
            assert fit.score.connected_information_bits == pytest.approx(expected_bits, abs=2e-12)
            assert fit.score.identity_error_bits <= 1e-12


def test_fractional_weight_rescaling_and_axis_permutation():
    counts = np.array([3, 19, 7, 2, 5, 1, 2, 9]).reshape(2, 2, 2)
    base = fit_joint(counts).score
    for factor in (0.4, 5.0):
        new = fit_joint(factor * counts.transpose((2, 0, 1))[::-1]).score
        assert new.connected_information_bits == pytest.approx(base.connected_information_bits)
        assert new.deviance_nats == pytest.approx(factor * base.deviance_nats)


def test_structural_zero_constrains_singles_and_pair_models_on_same_domain():
    mask = np.ones((3, 2, 2), dtype=bool)
    mask[1, 1, 1] = False
    counts = np.arange(1, 13).reshape(3, 2, 2).astype(float) * mask
    fit = fit_joint(
        counts,
        allowed=mask,
        n_valid=80,
        options=MaxEntOptions(prior_mode="per_cell", prior_mass=0.5, tolerance=1e-12),
    )
    assert fit.score.status == "ok" and fit.score.m1_kind == "support_constrained_singles"
    assert fit.score.prior_total == 5.5 and fit.score.possible_cells == 11
    assert all(p[1, 1, 1] == 0 for p in (fit.p1, fit.p2, fit.p3))
    for axis in range(3):
        complement = tuple(a for a in range(3) if a != axis)
        np.testing.assert_allclose(
            fit.p1.sum(axis=complement), fit.p3.sum(axis=complement), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(fit.p2.sum(axis=axis), fit.p3.sum(axis=axis), rtol=0, atol=1e-12)
    assert fit.score.h1_bits >= fit.score.h2_bits >= fit.score.h3_bits - 1e-12
    assert fit.score.identity_error_bits <= 1e-12 and np.isnan(fit.score.threebody_rms)
    with pytest.raises(ValueError, match="structural zero"):
        fit_joint(counts + 1, allowed=mask)


def test_zero_margins_constant_axis_nonidentifiability_and_prior_only_signal():
    counts = np.zeros((3, 3, 2))
    counts[0, 1, 0] = 2
    counts[0, 2, 1] = 3
    fit = fit_joint(counts)
    assert fit.score.status == "ok" and fit.score.information_status == "constant_empirical_axis"
    assert fit.score.connected_information_bits == pytest.approx(0, abs=1e-14)
    assert fit.score.active_pair_face_cells == 2 and fit.score.possible_cells == 18
    smoothed = fit_joint(counts, options=MaxEntOptions(prior_mass=6)).score
    assert smoothed.status == "prior_only_axis" and np.isnan(smoothed.connected_information_bits)
    assert np.isfinite(smoothed.target_kl_bits)
    assert fit_joint(counts * 0).score.status == "no_data"


def test_infeasible_marginals_iteration_limit_and_resource_guards():
    anti = np.array([[0, 0.5], [0.5, 0]])
    assert fit_pair_marginals(anti, anti, anti).status == "incompatible_support"
    changed = np.array([[0, 0.6], [0.4, 0]])
    assert fit_pair_marginals(anti, changed, anti).status == "inconsistent_margins"
    with pytest.raises(ValueError, match="sum to one"):
        fit_pair_marginals(anti, anti, anti * 0.1)
    with pytest.raises(ValueError, match="axes disagree"):
        fit_pair_marginals(anti, np.zeros((3, 2)), anti)
    counts = np.ones((2, 2, 2))
    counts[0, 0, 0] = counts[1, 1, 1] = 0
    failed = fit_joint(counts, options=MaxEntOptions(max_iterations=4))
    assert failed.p2 is None and failed.score.status == "iteration_limit"
    assert failed.score.ipf_iterations == 4 and failed.score.ipf_residual > 1e-5
    assert np.isnan(failed.score.connected_information_bits)
    with pytest.raises(ValueError, match="max_cells"):
        fit_joint(counts, options=MaxEntOptions(max_cells=7))


def _alignment():
    cols = (
        ("AA", "CC", "AA", "CC", "DD", "??", "-A", "ZA"),
        ("AA", "AA", "CC", "CC", "GG", "AA", "-A", "TA"),
        ("UU", "CC", "CC", "UU", "GG", "??", "-A", "UA"),
    )
    comps = tuple(
        ComponentMSA(c, m, tuple(f"{c}{i}" for i in range(len(rows))), rows)
        for c, m, rows in zip("ABC", ("protein", "dna", "rna"), cols)
    )
    return MatchedTripletMSA(
        *comps,
        tuple(TripletMatch(f"obs{i}", *ids) for i, ids in enumerate(zip(*(c.ids for c in comps)))),
    )


def test_training_support_replay_novel_states_common_rows_and_weight_scale(monkeypatch):
    msa = _alignment()
    training = ["obs0", "obs1", "obs2", "obs3"]
    declared = freeze_support(msa)
    assert declared.shape_for((0, 0, 0)) == (20, 4, 4)
    support = freeze_support(msa, training_observations=training, min_state_count=2)
    assert support.shape_for((0, 0, 0)) == (3, 3, 3)
    assert FrozenSupport.from_metadata(json.loads(json.dumps(support.metadata()))) == support
    mutated = support.metadata()
    mutated["min_state_count"] = 0
    with pytest.raises(ValueError, match="training"):
        FrozenSupport.from_metadata(mutated)
    with pytest.raises(ValueError, match="training"):
        freeze_support(msa, min_state_count=1)
    w = np.array([0.5, 0.5, 0.5, 0.5, 0, 1.0, 1.0, 1.0])
    import triplet_coevolution.inference.maxent as module

    actual = module.fit_joint
    calls = []

    def inspect(counts, **kw):
        calls.append(counts.copy())
        return actual(counts, **kw)

    monkeypatch.setattr(module, "fit_joint", inspect)
    opts = MaxEntOptions(prior_mass=1)
    grid = fit_maxent_grid(
        msa,
        [(0, 0, 0), (0, 0, 0)],
        w,
        support=support,
        options_sequence=[opts, opts, MaxEntOptions()],
    )
    assert len(calls) == 2 and grid[0][0] is grid[1][0]
    assert grid[0][0] is grid[0][1] and grid[2][0] is grid[2][1]
    assert grid[0][0].n_valid == 4 and grid[0][0].valid_weight == 2
    assert grid[0][0].possible_cells == 27 and grid[0][0].observed_cells == 4
    assert grid[2][0].connected_information_bits == pytest.approx(1)
    assert grid[0][0].connected_information_bits < 1
    new = fit_maxent_triples(msa, [(0, 0, 0)], w + [0, 0, 0, 0, 1, 0, 0, 0], support=support)[0]
    assert new.possible_cells == 27 and new.observed_cells == 5
    below = fit_maxent_triples(msa, [(0, 0, 0)], w, support=support, min_valid_weight=2.01)[0]
    assert below.status == "insufficient_weight" and np.isnan(below.connected_information_bits)
    assert fit_maxent_triples(msa, [(0, 0, 0)], w, min_valid=5)[0].status == "insufficient_rows"
    through = fit_maxent_triples(
        msa, [(0, 0, 0)], w, support=freeze_support(msa, include_gaps=True)
    )[0]
    assert through.n_valid == 5 and through.possible_cells == 21 * 5 * 5
    assert (
        fit_maxent_triples(msa, [(0, 0, 0)], options=MaxEntOptions(max_cells=10))[0].status
        == "resource_limit"
    )
