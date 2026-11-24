"""Independent gradient, normalized-law, fit isolation, serialization and null-refit oracles."""

import json
from dataclasses import replace
from itertools import combinations, product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.nested import (
    NestedOptions,
    compare_nested,
    evaluate_model,
    fit_nested,
    load_model,
    write_nested,
)
from triplet_coevolution.inference.nested_solver import (
    ConditionalObjective,
    Conditionals,
    Layout,
    canonical_terms,
    exact_law,
    optimize,
)
from triplet_coevolution.validation.clade_splits import make_split_plan, subset_rows
from triplet_coevolution.validation.pair_sampler import enumerate_pair_law


def source(n=60):
    rng = np.random.default_rng(41683)
    a = rng.integers(2, size=(n, 2))
    b = rng.integers(2, size=(n, 1))
    c = (a[:, 0] ^ b[:, 0])[:, None]
    c[::13] ^= 1
    ids = tuple(f"t{i}" for i in range(n))
    groups = {key: "g" + str(i // 10) for i, key in enumerate(ids)}
    parts = tuple(
        ComponentMSA(
            r, "dna", tuple(r + k for k in ids), tuple("".join("AC"[x] for x in row) for row in arr)
        )
        for r, arr in zip("ABC", (a, b, c))
    )
    msa = MatchedTripletMSA(
        *parts, tuple(TripletMatch(key, *(p.ids[i] for p in parts)) for i, key in enumerate(ids))
    )
    maps = tuple(
        ReferenceCoordinates(
            p.name,
            p.name + "-ref",
            tuple(4 + 3 * i for i in range(p.n_sites)),
            tuple(10 + i for i in range(p.n_sites)),
            ("A",) * p.n_sites,
        )
        for p in parts
    )
    return msa, maps, groups


def test_weighted_projected_gradient_all_parameters_and_exact_conditional_losses():
    rng = np.random.default_rng(15)
    shape = (2, 3, 2)
    layout = Layout(
        shape,
        list(combinations(range(3), 2)),
        [(0, 1, 2)],
        NestedOptions(l2=0.031, field_l2=0.02, triple_l2=0.09),
    )
    x = np.array(list(product(*[range(q) for q in shape])))
    w = rng.uniform(0.2, 2, len(x))
    obj = ConditionalObjective(layout, x, w)
    theta = rng.normal(0, 0.25, layout.size)
    loss, grad = obj.evaluate(theta)
    delta = 1e-6
    fd = []
    for dx in np.eye(layout.size) * delta:
        fd.append(
            (obj.evaluate(theta + dx, False)[0] - obj.evaluate(theta - dx, False)[0]) / (2 * delta)
        )
    np.testing.assert_allclose(grad, fd, rtol=2e-6, atol=4e-10)
    theta = layout.projected(theta)
    law, log_z = exact_law(layout, theta)
    expected = np.zeros((len(x), 3))
    for i, state in enumerate(x):
        for j in range(3):
            ctx = tuple(slice(None) if k == j else v for k, v in enumerate(state))
            expected[i, j] = -np.log(law[tuple(state)] / law[ctx].sum())
    np.testing.assert_allclose(Conditionals(layout, x).losses(theta), expected, atol=1e-14)
    penalty = sum(
        layout.options.penalty(len(term)) * (t * t).sum() / 2
        for term, t in zip(layout.terms, layout.arrays(theta))
    )
    assert loss == pytest.approx(np.average(expected.mean(axis=1), weights=w) + penalty, abs=1e-12)
    assert np.isfinite(log_z)


def test_gauge_decomposition_and_independent_exact_pair_sampler_oracle():
    rng = np.random.default_rng(12)
    shape = (2, 3, 2)
    rows = list(product(*[range(q) for q in shape]))
    tensors = {
        term: rng.normal(size=tuple(shape[i] for i in term))
        for k in (1, 2, 3)
        for term in combinations(range(3), k)
    }
    fixed = canonical_terms(shape, tensors)
    for x in rows:

        def energy(t, x=x):
            return sum(arr[tuple(x[i] for i in term)] for term, arr in t.items())

        assert energy(fixed) == pytest.approx(energy(tensors), abs=1e-14)
    for term, t in fixed.items():
        for axis in range(len(term)):
            np.testing.assert_allclose(t.mean(axis=axis), 0, atol=5e-15)
    l = Layout(shape, list(combinations(range(3), 2)), [], NestedOptions())
    vec = np.concatenate([fixed[t].ravel() for t in l.terms])
    p, _ = exact_law(l, vec)
    independent = enumerate_pair_law(
        [fixed[(i,)] for i in range(3)], {t: fixed[t] for t in l.terms if len(t) == 2}
    )
    np.testing.assert_allclose(p, independent, rtol=1e-14)


def test_exact_known_laws_nested_capacity_and_omitted_mediator():
    shape = (2,) * 4
    rows = np.array(list(product(range(2), repeat=4)))
    a, b, c, m = (rows * 2 - 1).T
    energy = 0.9 * m + 1.1 * m * a + 1.2 * m * b + 0.85 * m * c + 0.21 * a * b
    p = np.exp(energy - energy.max()).reshape(shape)
    p /= p.sum()
    marginal = p.sum(axis=3)
    signs = (a * b * c).reshape(shape)[:, :, :, 0]
    assert abs(np.mean(signs * np.log(marginal))) > 0.1
    opt = NestedOptions(l2=1e-6, tolerance=8e-7, max_iterations=1600)
    for data, law, expect in ((rows, p, False), (rows[:8, :3], marginal, True)):
        sites = law.ndim
        data = np.array(list(product(range(2), repeat=sites)))
        edge = list(combinations(range(sites), 2))
        lp = Layout((2,) * sites, edge, [], opt)
        lk = Layout((2,) * sites, edge, [(0, 1, 2)], opt)
        fits = [optimize(l, data, law.ravel()) for l in (lp, lk)]
        assert all(diag["status"] == "converged" for _, diag in fits)
        full = lk.arrays(fits[1][0])[-1]
        gain = np.average(
            (
                Conditionals(lp, data).losses(fits[0][0])
                - Conditionals(lk, data).losses(fits[1][0])
            ).mean(axis=1),
            weights=law.ravel(),
        )
        if expect:
            assert abs(full).max() > 0.1 and gain > 0.001
        else:
            assert abs(full).max() < 0.001 and abs(gain) < 1e-7


def test_heldout_isolation_state_encoding_and_checked_saved_models(tmp_path):
    msa, maps, groups = source()
    plan = make_split_plan(groups, groups, test_groups=["g3", "g4", "g5"])
    opt = NestedOptions(l2=0.04, min_state_count=2)
    kw = {
        "options": opt,
        "candidate_options": {"mode": "triangle", "maximum": 1},
        "weight_options": {"policy": "clade_balanced"},
        "bootstrap_draws": 35,
        "seed": 102,
    }
    base = compare_nested(msa, maps, plan, **kw)
    fold = plan.folds[0]
    changed = tuple(
        ComponentMSA(
            c.name,
            c.molecule,
            c.ids,
            tuple(
                "T" * len(s) if row.observation_id in fold.test_ids else s
                for s, row in zip(c.sequences, msa.matches)
            ),
        )
        for c in msa.components
    )
    altered = compare_nested(MatchedTripletMSA(*changed, msa.matches), maps, plan, **kw)
    model = base.models[fold.fold_id]["pair_plus_k"]
    assert model.digest() == altered.models[fold.fold_id]["pair_plus_k"].digest()
    other = base.models[fold.fold_id]["pair_only"]
    assert model.metadata["training_design_sha256"] == other.metadata["training_design_sha256"]
    assert model.sites == other.sites and model.edges == other.edges and not other.triples
    assert (
        base.summary["clade_macro_gain_nats_per_site"]
        != altered.summary["clade_macro_gain_nats_per_site"]
    )
    assert base.summary["heldout_clade_interval"]["status"] == "available"
    report = write_nested(tmp_path, base)
    art = report["model_artifacts"][fold.fold_id]["pair_plus_k"]
    loaded = load_model(tmp_path / art["path"])
    assert loaded.digest() == art["model_sha256"]
    held = subset_rows(msa, fold.test_ids)
    scores = evaluate_model(loaded, held, maps)
    assert scores["scoreable_rows"] == 30 and scores["row_ids"] == fold.test_ids
    np.testing.assert_allclose(
        scores["loss_per_row_site_nats"].mean(axis=0),
        [v["pair_plus_k_loss_nats"] for v in base.site_rows],
        atol=5e-13,
    )
    with pytest.raises(ValueError, match="source panel"):
        evaluate_model(loaded, held, (replace(maps[0], reference_id="altered"), *maps[1:]))
    fields = np.load(tmp_path / art["path"])
    original = bytearray(fields["header"].tobytes())
    v = json.loads(original)
    v["fit"]["training_design_sha256"] = "x"
    bad = tmp_path / "bad.npz"
    np.savez(bad, header=np.frombuffer(json.dumps(v).encode(), np.uint8), theta=fields["theta"])
    with pytest.raises(ValueError, match="checksum"):
        load_model(bad)


def test_common_masks_degenerate_graph_status_and_hard_guards():
    msa, maps, groups = source()
    plan = make_split_plan(groups, groups, test_groups=["g5"])
    src = msa.a
    with_gaps = ComponentMSA(
        src.name,
        src.molecule,
        src.ids,
        tuple("-" + s[1:] if i % 4 == 0 else s for i, s in enumerate(src.sequences)),
    )
    gapped = MatchedTripletMSA(with_gaps, msa.b, msa.c, msa.matches)
    opts = NestedOptions(min_state_count=2)
    result = compare_nested(gapped, maps, plan, options=opts, edges=[], triples=[])
    row = result.fold_rows[0]
    assert (
        row["gain_nats_per_site"] == 0
        and row["complete_training_rows"] == 37
        and row["complete_test_rows"] == 8
    )
    assert result.summary["heldout_clade_interval"]["status"] == "not_requested"
    m = result.models[plan.folds[0].fold_id]
    assert np.array_equal(m["pair_only"].theta, m["pair_plus_k"].theta) and not m["pair_only"].edges
    gap_state = compare_nested(
        gapped, maps, plan, options=replace(opts, include_gaps=True), edges=[], triples=[]
    )
    assert gap_state.fold_rows[0]["complete_training_rows"] == 50
    with pytest.raises(ValueError, match="pair faces"):
        compare_nested(msa, maps, plan, options=opts, edges=[], triples=[(5, 5, 5)])
    with pytest.raises(ValueError, match="unknown panel"):
        compare_nested(msa, maps, plan, options=opts, triples=[(9, 5, 5)])
    with pytest.raises(ValueError, match="coefficients"):
        fit_nested(msa, maps, options=replace(opts, max_coefficients=3))
    with pytest.raises(ValueError, match="workspace"):
        fit_nested(msa, maps, options=replace(opts, max_workspace_bytes=64))
    failed = compare_nested(
        msa, maps, plan, options=replace(opts, max_iterations=1, tolerance=1e-13)
    )
    assert failed.fold_rows[0]["status"] == "fit_not_converged" and not failed.site_rows
    few = compare_nested(msa, maps, plan, options=replace(opts, min_valid=100))
    assert few.fold_rows[0]["status"] == "insufficient_complete_training_support" and not few.models
    invalid = replace(plan, folds=(replace(plan.folds[0], train_ids=tuple(groups)),))
    with pytest.raises(ValueError, match="isolation"):
        compare_nested(msa, maps, invalid, options=opts)


def test_exact_null_refits_support_guard_and_fixed_screen():
    msa, maps, groups = source()
    plan = make_split_plan(groups, groups, test_groups=["g5"])
    args = {
        "panel": {"A": [5], "B": [5], "C": [5]},
        "triples": [(5, 5, 5)],
        "null_draws": 3,
        "seed": 17,
    }
    result = compare_nested(msa, maps, plan, **args)
    evidence = result.fold_rows[0]["parametric_null"]
    assert (
        evidence["status"] == "available"
        and evidence["valid_draws"] + evidence["invalid_draws"] == 3
    )
    assert (
        evidence["p_conservative"] == (1 + evidence["exceedances"] + evidence["invalid_draws"]) / 4
    )
    limited = compare_nested(msa, maps, plan, options=NestedOptions(max_joint_states=8), **args)
    assert (
        limited.fold_rows[0]["parametric_null"]["status"] == "unavailable_exact_joint_state_budget"
    )
    adaptive = compare_nested(msa, maps, plan, options=NestedOptions(min_state_count=1), **args)
    assert "adaptive" in adaptive.fold_rows[0]["parametric_null"]["status"]


def test_optional_lbfgs_agrees_with_numpy_gradient_on_mixed_panel():
    pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(413)
    x = np.column_stack([rng.integers(q, size=96) for q in (2, 3, 2)])
    mass = rng.uniform(0.1, 2.0, len(x))
    opt = NestedOptions(l2=0.06, tolerance=1e-8, max_iterations=1200)
    dims = (2, 3, 2)
    terms = list(combinations(range(3), 2))
    lp = Layout(dims, terms, [(0, 1, 2)], opt)
    lb = Layout(dims, terms, [(0, 1, 2)], replace(opt, optimizer="lbfgs"))
    tp, dp = optimize(lp, x, mass)
    tb, db = optimize(lb, x, mass)
    assert dp["status"] == db["status"] == "converged"
    assert dp["objective"] == pytest.approx(db["objective"], abs=1e-12)
    np.testing.assert_allclose(
        Conditionals(lp, x).losses(tp), Conditionals(lb, x).losses(tb), atol=6e-7
    )


def test_state_pooling_and_screen_exclude_incomplete_and_zero_mass_training_rows():
    msa, maps, _groups = source()
    parts = []
    for c in msa.components:
        seq = list(c.sequences)
        if c.name == "A":
            seq[0], seq[1] = "G" + seq[0][1:], "G" + seq[1][1:]
        if c.name == "C":
            seq[0] = "?"
        parts.append(ComponentMSA(c.name, c.molecule, c.ids, tuple(seq)))
    src = MatchedTripletMSA(*parts, msa.matches)
    supplied = {r.observation_id: float(i != 1) for i, r in enumerate(msa.matches)}
    result = fit_nested(
        src,
        maps,
        options=NestedOptions(min_state_count=1),
        candidate_options={"mode": "triangle", "maximum": 1},
        weight_options={"policy": "supplied", "supplied_weights": supplied},
    )
    learned = result.models["pair_only"].sites[0]
    assert learned.native_to_bin[2] == learned.native_to_bin[3]
    assert result.design["candidate_screen"]["screen_observation_count"] == 58
    assert result.design["positive_complete_rows"] == 58
