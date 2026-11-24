"""Independent conditional-probability, gradient, gauge, missingness and resource tests."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference._plm_target import TargetObjective, zero_sum
from triplet_coevolution.inference.plm import PLMOptions, fit_conditional, fit_plm, plm_preflight


def _msa(rows, molecules=("protein", "dna", "rna")):
    parts = tuple(
        ComponentMSA(c, m, tuple(f"{c}{i}" for i in range(len(rows))), tuple(r[j] for r in rows))
        for j, (c, m) in enumerate(zip("ABC", molecules))
    )
    return MatchedTripletMSA(
        *parts, tuple(TripletMatch(str(i), *(c.ids[i] for c in parts)) for i in range(len(rows)))
    )


def test_categorical_objective_normalization_and_every_gradient():
    x = np.array([[0, 1, 2], [1, 1, 0], [1, -1, 1], [0, 2, -1], [0, 0, 0], [-1, 2, 0], [0, 1, 2]])
    w = np.array([1, 2, 0.75, 0.25, 1, 0.2, 0])
    q = (2, 3, 3)
    opt = PLMOptions(lambda_h=0.13, lambda_j=0.26, batch_rows=2)
    obj = TargetObjective(x, q, 0, w, opt)
    params = np.linspace(-0.35, 0.41, obj.size)

    def independently_evaluate(theta):
        h = theta[:2]
        j = theta[2:].reshape(2, 3, 2)
        rows = [r for r in range(len(w)) if w[r] > 0 and np.min(x[r]) >= 0]
        means = []
        for p in (1, 2):
            counts = np.array([sum(w[r] for r in rows if x[r, p] == s) for s in range(3)])
            means.append(counts / counts.sum())
        losses = []
        for r in rows:
            logits = h.copy()
            for p, k in enumerate((1, 2)):
                if x[r, k] >= 0:
                    logits += j[p, x[r, k]] - means[p] @ j[p]
            prob = np.exp(logits) / np.exp(logits).sum()
            losses.append(-np.log(prob[x[r, 0]]))
        return np.average(losses, weights=w[rows]) + 0.5 * (
            opt.lambda_h * (h @ h) + opt.lambda_j * np.sum(j * j)
        )

    value, grad = obj.loss_grad(params)
    assert value == pytest.approx(independently_evaluate(params), abs=1e-14)
    delta = 1e-6 * np.eye(obj.size)
    reference = np.array(
        [
            (independently_evaluate(params + d) - independently_evaluate(params - d)) / (2e-6)
            for d in delta
        ]
    )
    np.testing.assert_allclose(grad, reference, rtol=0, atol=3e-10)


def test_no_predictors_penalized_logit_root_and_weight_scaling():
    x = np.array([[0], [1]])
    opt = PLMOptions(lambda_h=0.13, tolerance=1e-10)
    one = fit_conditional(x, [2], 0, [7.0, 3.0], options=opt)
    two = fit_conditional(x[::-1], [2], 0, [0.3, 0.7], options=opt)
    assert one.diagnostics["status"] == two.diagnostics["status"] == "converged"
    h = one.fields[0]
    assert one.fields[1] == pytest.approx(-h, abs=1e-12) and not one.coefficients
    assert 1 / (1 + np.exp(-2 * h)) + 0.13 * h == pytest.approx(0.7, abs=1e-10)
    np.testing.assert_allclose(one.fields, two.fields, atol=1e-10)
    assert one.diagnostics["weighted_centered_design_rank"] == 0


def test_chain_recovers_signed_direct_blocks_and_indirect_conditional_zero():
    x = np.array(list(product((0, 1), repeat=3)))
    a, b, c = (2 * x - 1).T
    w = np.exp(0.27 * a - 0.24 * b + 0.12 * c + 0.53 * a * b - 0.71 * b * c)
    opt = PLMOptions(lambda_h=1e-6, lambda_j=1e-6, tolerance=2e-9)
    fits = [fit_conditional(x, [2] * 3, i, w, options=opt) for i in range(3)]
    assert all(f.diagnostics["status"] == "converged" for f in fits)
    pattern = np.array([[1, -1], [-1, 1]])
    for i, j, k in ((0, 1, 0.53), (1, 2, -0.71), (0, 2, 0)):
        direct = zero_sum(fits[i].coefficients[j]).T
        reverse = zero_sum(fits[j].coefficients[i])
        np.testing.assert_allclose(direct, k * pattern, rtol=0, atol=3e-5)
        np.testing.assert_allclose(reverse, direct, rtol=0, atol=3e-5)
        shifted = direct + np.arange(2)[:, None] + np.array([0.2, -0.8])[None, :]
        np.testing.assert_allclose(zero_sum(shifted), direct, atol=1e-14)
    # Omitting B exposes mediated marginal dependence.
    short = fit_conditional(x[:, [0, 2]], [2, 2], 0, w, options=opt)
    assert np.linalg.norm(zero_sum(short.coefficients[1])) > 0.3


@pytest.mark.parametrize("molecules", [("protein", "dna", "rna"), ("dna", "rna", "protein")])
def test_mixed_fit_symmetry_parallel_original_axes_and_gap_excluded_norm(molecules):
    letters = [CANONICAL[m] for m in molecules]
    rows = [tuple(letters[j][z] for j, z in enumerate(row)) for row in product(range(3), repeat=3)]
    rows.extend([("-", "A", "A"), ("A", "?", "C"), ("?", "C", "-")])
    w = np.array([np.exp(0.8 * (r[0] == r[1]) - 0.35 * (r[0] == r[2])) for r in rows])
    msa = _msa(rows, molecules)
    saved = {}
    opt = PLMOptions(include_gaps=True, tolerance=2e-7)
    run = fit_plm(msa, w, options=opt, coupling_sink=lambda pair, i, j, m: saved.update({pair: m}))
    parallel = fit_plm(msa, w, options=replace(opt, workers=3, batch_rows=9))
    assert run.diagnostics["status"] == parallel.diagnostics["status"] == "ok"
    assert len(run.diagnostics["targets"]) == 3
    for name in ("AB", "AC", "BC"):
        a, b = map("ABC".index, name)
        q1, q2 = len(letters[a]), len(letters[b])
        block = saved[name]
        assert block.shape == (q1, q2)
        np.testing.assert_allclose(block.sum(0), 0, atol=1e-12)
        np.testing.assert_allclose(block.sum(1), 0, atol=1e-12)
        assert np.linalg.norm(block) == pytest.approx(run.blocks[name].frobenius[0, 0])
        assert run.blocks[name].frobenius == pytest.approx(
            parallel.blocks[name].frobenius, abs=1e-10
        )
        support = [
            k
            for k, row in enumerate(rows)
            if "?" not in row and row[a] in letters[a] and row[b] in letters[b]
        ]
        assert run.blocks[name].n_valid[0, 0] == len(support)
        assert run.blocks[name].valid_weight[0, 0] == pytest.approx(w[support].sum())
    swapped = _msa([(c, b, a) for a, b, c in rows], molecules[::-1])
    flip = fit_plm(swapped, w, options=opt)
    np.testing.assert_allclose(flip.blocks["BC"].frobenius, run.blocks["AB"].frobenius, atol=1e-10)
    complete = fit_plm(msa, w, options=replace(opt, missing_policy="complete", include_gaps=False))
    assert complete.blocks["AB"].n_valid.item() == 27
    assert run.blocks["BC"].n_valid.item() > complete.blocks["BC"].n_valid.item()


def test_fields_only_imbalanced_and_missing_predictor_has_no_induced_contrast():
    rows = np.array(list(product((0, 1), (0, 1, 2), (0, 1))))
    marginal = [[0.15, 0.85], [0.25, 0.05, 0.70], [0.40, 0.60]]
    weights = np.array([np.prod([marginal[i][v] for i, v in enumerate(row)]) for row in rows])
    # Predictor is missing more often for outcome 1. Conditional on observed status, it
    # remains independent of this outcome. Marginal mean coding is essential with L2 fields.
    masked = rows.copy()
    masked[:, 1] = -1
    fraction = np.where(rows[:, 0] == 1, 0.65, 0.1)
    x = np.vstack([rows, masked])
    w = np.r_[weights * (1 - fraction), weights * fraction]
    res = fit_conditional(x, [2, 3, 2], 0, w, options=PLMOptions(lambda_h=0.2, lambda_j=0.01))
    assert res.diagnostics["status"] == "converged"
    assert res.diagnostics["positive_rows_excluded_incomplete"] == 12
    for matrix in res.coefficients.values():
        assert np.linalg.norm(zero_sum(matrix)) < 1e-10
    with pytest.raises(ValueError, match="complete"):
        PLMOptions(missing_policy="mean_predictor")


def test_independent_masks_keep_mediator_conditioned_explicitly():
    states = np.array(list(product((0, 1), repeat=3)))
    a, b, c = (2 * states - 1).T
    mass = np.exp(0.27 * a - 0.24 * b + 0.12 * c + 0.53 * a * b - 0.71 * b * c)
    rows, weights = [], []
    for row, w in zip(states, mass):
        for mask in product((False, True), repeat=3):
            data = row.copy()
            data[list(mask)] = -1
            rows.append(data)
            weights.append(w * 0.2 ** sum(mask) * 0.8 ** (3 - sum(mask)))
    opt = PLMOptions(lambda_h=1e-6, lambda_j=1e-6, tolerance=2e-9)
    fit = fit_conditional(np.asarray(rows), [2] * 3, 0, weights, options=opt)
    assert (
        fit.diagnostics["rows"] == 8 and fit.diagnostics["positive_rows_excluded_incomplete"] == 56
    )
    assert fit.diagnostics["weight_sum"] == pytest.approx(0.8**3 * mass.sum())
    for j, expected in ((1, 0.53), (2, 0)):
        assert zero_sum(fit.coefficients[j])[0, 0] == pytest.approx(expected, abs=8e-6)


def test_conditioning_panels_change_mediated_signal_and_reorder_sites_and_rows():
    x = np.array(list(product((0, 1), repeat=4)))
    a, b, c, _noise = (2 * x - 1).T
    w = np.exp(0.65 * a * b - 0.7 * b * c)
    text = [("AC"[r[0]] + "AC"[r[3]], "AC"[r[1]], "AC"[r[2]]) for r in x]
    msa = _msa(text, ("dna",) * 3)
    opt = PLMOptions(lambda_h=0.0003, lambda_j=0.0007, tolerance=5e-10)
    full = fit_plm(msa, w, options=opt)
    pairwise = fit_plm(msa, w, options=opt, conditioning="pairwise")
    assert full.diagnostics["status"] == pairwise.diagnostics["status"] == "ok"
    assert pairwise.blocks["AC"].frobenius[0, 0] > 4 * full.blocks["AC"].frobenius[0, 0]
    assert {t["panel"] for t in pairwise.diagnostics["targets"]} == {"AB", "AC", "BC"}
    # A column exchange and common row reversal do not change its identified edges.
    permuted = _msa([(a[::-1], b, c) for a, b, c in reversed(text)], ("dna",) * 3)
    flipped = fit_plm(permuted, w[::-1], options=opt)
    for group in ("AB", "AC"):
        np.testing.assert_allclose(
            full.blocks[group].frobenius, flipped.blocks[group].frobenius[::-1], atol=1e-7
        )


def test_resource_limit_invalid_input_constant_and_optimizer_failure_are_explicit():
    rows = [("A" * 24, "C" * 24, "A" * 24), ("C" * 24, "A" * 24, "C" * 24)]
    msa = _msa(rows, ("protein",) * 3)
    with pytest.raises(ValueError, match="memory_mib"):
        plm_preflight(msa, PLMOptions(memory_mib=1))
    with pytest.raises(ValueError, match="scratch"):
        plm_preflight(msa, PLMOptions(scratch_mib=1))
    with pytest.raises(ValueError, match="within their declared support"):
        fit_conditional(np.array([[0, -2], [1, 1]]), [2, 2], 0)
    small = _msa([("A", "A", "A"), ("C", "C", "C"), ("C", "A", "C"), ("C", "C", "A")])
    failed = fit_plm(small, options=PLMOptions(max_iterations=1, tolerance=1e-12))
    assert failed.diagnostics["status"] == "partial_nonconvergence"
    assert failed.diagnostics["target_status_counts"] == {"iteration_limit": 3}
    assert all(np.isnan(b.frobenius).all() for b in failed.blocks.values())
    constant = fit_plm(_msa([("A", "A", "A"), ("A", "C", "C"), ("A", "A", "C"), ("A", "C", "A")]))
    assert constant.diagnostics["target_status_counts"]["constant_target"] == 1
    assert np.isnan(constant.blocks["AB"].frobenius.item()) and np.isfinite(
        constant.blocks["BC"].frobenius.item()
    )
