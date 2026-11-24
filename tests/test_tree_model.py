"""Strict input, analytic transition and simulation checks for the independent-site tree null."""

import json

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.validation.tree_model import (
    compare_trees,
    f81_transition,
    load_tree,
    prepare_tree_null,
)


def _msa(seqs):
    parts = tuple(
        ComponentMSA(c, mol, tuple(f"{c}{i}" for i in range(len(rows))), tuple(rows))
        for c, mol, rows in zip("ABC", ("protein", "dna", "rna"), seqs)
    )
    return MatchedTripletMSA(
        *parts,
        tuple(
            TripletMatch(f"t{i}", *(c.ids[i] for c in parts)) for i in range(parts[0].n_sequences)
        ),
    )


def _tree(tmp_path, text, labels=None, mapping=None):
    path = tmp_path / "scenario.nwk"
    path.write_text(text, encoding="utf8")
    return load_tree(path, labels or ("t0", "t1"), tip_map=mapping)


@pytest.mark.parametrize(
    "text,err",
    [
        ("(t0:0,t0:0);", "duplicate"),
        ("(t0,t1:0);", "length"),
        ("(t0:1,t1:-2);", "nonnegative"),
        ("(t0:1,t1:nan);", "length"),
        ("(t0:1,t1:1e999);", "finite"),
        ("(t0:0,t1:0);(t0:1,t1:1);", "single"),
        ("(t0:1,t1:1)", "complete"),
        ("(t0:0,t1:0.2zzz);", "length"),
        ("(t0:0,:0);", "label"),
        ("(t0:0,t2:0);", "bijection"),
        ("(t0:0,t1:0):1:1;", "complete"),
        ("(t0:1,t1:1);[unclosed", "single"),
    ],
)
def test_rejects_malformed_or_unaligned_newick(tmp_path, text, err):
    with pytest.raises(ValueError, match=err):
        _tree(tmp_path, text)


def test_quoted_tip_mapping_and_root_audit(tmp_path):
    mapping = tmp_path / "tips.tsv"
    mapping.write_text("tip_label\tobservation_id\nBob's leaf\tt1\na\tt0\n", encoding="utf8")
    tree = _tree(tmp_path, "[&U]('Bob''s leaf':0[foo],a:1.2E-2)99:0.4;", mapping=mapping)
    meta = tree.metadata()
    assert meta["root_stem_ignored_at_stationarity"] == 0.4 and meta["newick_root_marker"] == "U"
    assert meta["tip_labels_in_observation_order"] == ["a", "Bob's leaf"]
    assert meta["tip_map_file_sha256"] and meta["zero_length_edges"] == 1
    np.testing.assert_array_equal(tree.lengths[tree.tips], [0.012, 0])
    with pytest.raises(ValueError, match="bijection"):
        _tree(tmp_path, "(a:1,b:1);", mapping={"a": "t0", "b": "t0"})
    mapping.write_text("tip_label\tobservation_id\na\tt0\na\tt1\n", encoding="utf8")
    with pytest.raises(ValueError, match="repeated"):
        _tree(tmp_path, "(a:1,b:1);", mapping=mapping)


def test_polytomy_rerooting_lengths_and_named_clades(tmp_path):
    ids = ("t0", "t1", "t2", "t3")
    a = _tree(tmp_path, "((t0:1,t1:2):1,(t2:3,t3:4):2);", ids)
    b = _tree(tmp_path, "(t0:0.4,(t1:2,(t3:4,t2:3):3):0.6);", ids)
    c = _tree(tmp_path, "((t0:1,t2:3):2,(t1:2,t3:4):1);", ids)
    star = _tree(tmp_path, "(t3:4,t1:2,t2:3,t0:1);", ids)
    same = compare_trees(a, b)
    assert same["unrooted_rf"] == 0 and same["branch_score_distance"] == pytest.approx(0)
    assert same["rooted_clade_symmetric_difference"] > 0
    conflict = compare_trees(a, c)
    assert conflict["unrooted_rf"] == 2 and conflict["normalized_unrooted_rf"] == 1
    assert conflict["branch_score_distance"] == pytest.approx(np.sqrt(18))
    assert compare_trees(a, star)["branch_score_distance"] == pytest.approx(3)
    assert star.metadata()["polytomies"] == 1
    clades = a.clades_at_depth(1)
    assert clades["t0"] == clades["t1"] != clades["t2"] == clades["t3"]
    assert (
        len(set(a.clades_at_depth(0).values())) == 1
        and len(set(a.clades_at_depth(50).values())) == 4
    )
    with pytest.raises(ValueError, match="identical mapped"):
        compare_trees(a, _tree(tmp_path, "(t0:0,t1:0);"))


def test_f81_analytic_stationarity_semigroup_zero_and_expected_rate():
    pi = np.array([0.65, 0.15, 0.12, 0.08])
    p = f81_transition(pi, 0.21)
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_allclose(pi @ p, pi)
    np.testing.assert_allclose(pi[:, None] * p, (pi[:, None] * p).T)
    np.testing.assert_allclose(f81_transition(pi, 0.11) @ f81_transition(pi, 0.10), p)
    np.testing.assert_allclose(f81_transition(pi, 2, rate_scale=0.105), p)
    np.testing.assert_array_equal(f81_transition(pi, 0), np.eye(4))
    np.testing.assert_array_equal(f81_transition([1, 0], 100), np.eye(2))
    np.testing.assert_allclose(f81_transition(pi, 1e5), np.broadcast_to(pi, (4, 4)))
    rate = np.sum(pi * (1 - np.diag(f81_transition(pi, 1e-6)))) / 1e-6
    assert rate == pytest.approx(1, abs=2e-6)
    with pytest.raises(ValueError, match="sum to one"):
        f81_transition([1, 2], 0.2)


def test_unary_ladder_and_zero_length_resolution(tmp_path):
    t0 = "(" * 1300 + "t0:0" + "):0" * 1300
    tree = _tree(tmp_path, f"({t0},t1:1);")
    assert tree.metadata()["unary_nodes"] == 1300
    assert compare_trees(tree, _tree(tmp_path, "(t0:0,t1:1);"))["branch_score_distance"] == 0
    ids = tuple(f"t{i}" for i in range(4))
    zero = _tree(tmp_path, "((t0:1,t1:1):0,t2:1,t3:1);", ids)
    star = _tree(tmp_path, "(t0:1,t1:1,t2:1,t3:1);", ids)
    result = compare_trees(zero, star)
    assert result["unrooted_rf"] == 1 and result["unrooted_rf_positive_edges"] == 0
    assert result["branch_score_distance"] == 0


def test_empirical_variable_and_constant_sites_and_frequency_contract(tmp_path):
    msa = _msa([["AA", "AC", "AD"], ["TT", "TC", "TA"], ["UU", "UC", "UA"]])
    tree = _tree(tmp_path, "(t0:1e6,t1:1e6,t2:1e6);", ("t0", "t1", "t2"))
    model = prepare_tree_null(msa, tree, weights=[1, 3, 0])
    assert model.metadata()["sites"]["B"]["constant_sites_0based"] == [0]
    # Unobserved A from the zero-weight row stays absent without a prior.
    np.testing.assert_allclose(model.frequencies[1][1], [0, 0.75, 0, 0.25])
    draw = model.draw(np.random.default_rng(35))
    assert all(s[0] == "U" for s in draw.c.sequences)
    with pytest.raises(ValueError, match="only to empirical"):
        prepare_tree_null(msa, tree, frequency_policy="uniform", prior_mass=1)
    with pytest.raises(ValueError, match="supplied frequencies"):
        prepare_tree_null(msa, tree, frequency_policy="supplied", frequencies={"A": [1, 0]})
    with pytest.raises(ValueError, match="rate_scale"):
        prepare_tree_null(msa, tree, rate_scale=-0.1)


def test_two_tip_simulator_against_exact_pair_law(tmp_path):
    nsites = 32000
    msa = _msa([["A" * nsites] * 2] * 3)
    tree = _tree(tmp_path, "(t0:0.12,t1:0.32);")
    p = np.array([0.65, 0.25, 0.1, 0])
    freqs = {"A": np.r_[p, np.zeros(16)], "B": p, "C": p}
    null = prepare_tree_null(msa, tree, frequency_policy="supplied", frequencies=freqs)
    # A stationary reversible process puts joint probability pi_a P_ab(t1+t2) on a tip pair.
    counts = np.bincount(
        (null.draw_codes(np.random.default_rng(84))[1] * np.array([[4], [1]])).sum(axis=0),
        minlength=16,
    )
    np.testing.assert_allclose(
        counts.reshape(4, 4) / nsites, p[:, None] * f81_transition(p, 0.44), rtol=0, atol=0.007
    )
    assert np.all(counts.reshape(4, 4)[3] == 0)


def test_tree_weights_mask_constant_ambiguity_and_identity(tmp_path):
    msa = _msa([["AX.C", "AC-D", "?A-C"], ["T-GN", "C.T?", "G-TA"], ["U.?A", "A-GC", "C-GA"]])
    ids = ("t0", "t1", "t2")
    tree = _tree(tmp_path, "(t1:0,t0:0,t2:9999);", ids)
    w = np.array([2.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="no positive-weight canonical"):
        prepare_tree_null(msa, tree, weights=w)
    model = prepare_tree_null(msa, tree, weights=w, prior_mass=1)
    meta = model.metadata()
    json.dumps(meta, allow_nan=False)
    assert meta["sites"]["B"]["alphabet"] == "ACGT" and meta["sites"]["C"]["alphabet"] == "ACGU"
    np.testing.assert_allclose(model.frequencies[1][0], [0.25 / 4, 1.25 / 4, 0.25 / 4, 2.25 / 4])
    assert meta["sites"]["B"]["no_positive_canonical_data_sites_0based"] == [1, 3]
    r1, r2 = (np.random.default_rng(36) for _ in range(2))
    draw = model.draw(r1)
    assert model.draw(r2) == draw and draw.matches == msa.matches
    for c, new in zip(msa.components, draw.components):
        assert c.ids == new.ids
        for row, newrow in zip(c.sequences, new.sequences):
            for old, symbol in zip(row, newrow):
                if old in "XN?-.":
                    assert old == symbol
    latent = model.draw_codes(r1)
    for comp in latent:
        np.testing.assert_array_equal(comp[0], comp[1])
    alternative = _tree(tmp_path, "((t0:0,t1:0):1,t2:0);", ids)
    mixed = prepare_tree_null(
        msa, {"A": tree, "B": alternative, "C": tree}, frequency_policy="uniform", rate_scale=0
    )
    assert set(mixed.metadata()["component_tree_comparisons"]) == {"AB", "BC"}
    for comp in mixed.draw_codes(r1):
        np.testing.assert_array_equal(comp[1:], np.broadcast_to(comp[:1], comp[1:].shape))
    with pytest.raises(ValueError, match="ordering"):
        prepare_tree_null(msa, _tree(tmp_path, "(t0:0,t1:0,t2:0);", ids[::-1]))
