import csv
import json
from itertools import product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.inference.advanced import write_advanced_results
from triplet_coevolution.inference.global_potts import global_triples
from triplet_coevolution.inference.potts import direct_couplings
from triplet_coevolution.inference.threebody import fit_triples, sample_candidates
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.msa.masking import apply_triplet_masks
from triplet_coevolution.validation.fdr import bh_adjust
from triplet_coevolution.validation.nulls import permutation_test, permute_components
from triplet_coevolution.validation.phylogeny import exchangeability_groups
from triplet_coevolution.validation.structure import contact_metrics


def alignment(rows):
    components = []
    for site, (name, molecule) in enumerate(zip("ABC", ("protein", "dna", "rna"))):
        ids = tuple(f"{name}{i}" for i in range(len(rows)))
        seqs = tuple(row[site] for row in rows)
        components.append(ComponentMSA(name, molecule, ids, seqs))
    matches = [
        TripletMatch(str(i), *ids) for i, ids in enumerate(zip(*(c.ids for c in components)))
    ]
    return MatchedTripletMSA(*components, matches)


def xor_rows():
    return [("AC"[a], "AC"[b], "AC"[a ^ b]) for a, b in product(range(2), repeat=2)]


def test_dca_removes_an_indirect_chain_association():
    rows = []
    for a, b, c in product("AC", repeat=3):
        # A and C are conditionally independent given B.
        rows.extend([(a, b, c)] * ((4 if a == b else 1) * (4 if b == c else 1)))
    scores = direct_couplings(alignment(rows), ridge=0.01)
    assert scores["AB"].frobenius[0, 0] == pytest.approx(scores["BC"].frobenius[0, 0])
    assert scores["AB"].frobenius[0, 0] > 10 * scores["AC"].frobenius[0, 0]
    for block in direct_couplings(alignment(list(product("AC", repeat=3)))).values():
        assert block.frobenius[0, 0] == pytest.approx(0, abs=1e-13)


def test_dca_missing_states_and_regularization():
    msa = alignment([("A", "T", "U"), ("C", "A", "G"), ("?", "T", "-"), ("D", "C", "U")])
    blocks = direct_couplings(msa, [1, 2, 1, 0], min_valid=2)
    assert blocks["AB"].n_valid[0, 0] == 2
    assert blocks["AB"].valid_weight[0, 0] == 3
    assert np.isfinite(blocks["AB"].frobenius).all()
    assert np.isnan(direct_couplings(msa, min_valid=5)["BC"].frobenius[0, 0])
    with pytest.raises(ValueError, match="ridge"):
        direct_couplings(msa, ridge=0)


def test_dca_dual_equals_primal_on_same_mixed_distribution():
    rows = [("AC", "TG", "UG"), ("CC", "AG", "GC"), ("AD", "C?", "UC"), ("DA", "TA", "GU")]
    dual = direct_couplings(alignment(rows), [1, 2, 3, 4], ridge=0.075)
    # Replication leaves weighted covariances unchanged and switches to the feature-space solve.
    primal = direct_couplings(alignment(rows * 8), [1, 2, 3, 4] * 8, ridge=0.075)
    for pair in dual:
        np.testing.assert_allclose(dual[pair].frobenius, primal[pair].frobenius, atol=1e-12)


def test_wide_low_depth_dca_does_not_allocate_feature_covariance():
    states = "ACDEFGHIKLMNPQRSTVWY"
    ids = tuple(str(i) for i in range(len(states)))
    components = [ComponentMSA(c, "protein", ids, tuple(s * 70 for s in states)) for c in "ABC"]
    msa = MatchedTripletMSA(*components, [TripletMatch(i, i, i, i) for i in ids])
    # 4,200 features, 20 rows. A 20x20 solve suffices.
    blocks = direct_couplings(msa)
    assert all(
        block.frobenius.shape == (70, 70) and np.isfinite(block.frobenius).all()
        for block in blocks.values()
    )


def test_local_three_factor_closed_forms():
    triple = [(0, 0, 0)]
    xor = fit_triples(alignment(xor_rows() * 4), triple)[0]
    assert xor.threebody_rms == pytest.approx(np.log((4 + 0.5) / 0.5) / 2)
    assert xor.co_information_bits == pytest.approx(-1)
    assert (xor.observed_cells, xor.possible_cells) == (4, 8)
    pair_only = [
        row for row in product("AC", repeat=3) for _ in range(4 if row[0] == row[1] else 1)
    ]
    assert fit_triples(alignment(pair_only), triple)[0].threebody_rms == pytest.approx(0, abs=1e-12)
    missing = fit_triples(alignment([*xor_rows(), ("?", "A", "A")]), triple, [2, 1, 1, 1, 9])[0]
    assert (missing.n_valid, missing.valid_weight) == (4, 5)


def test_joint_three_factor_matches_analytic_penalized_parity_fit():
    msa = alignment(xor_rows() * 2 + [("?", "A", "A")])
    (result,), info = global_triples(msa, [(0, 0, 0)], l2=0.1)
    assert info["status"] == "converged" and info["gradient_max"] < 1e-6
    assert info["complete_observations"] == result.n_valid == 8
    assert result.co_information_bits == pytest.approx(-1)
    # H = J * parity. The three conditional losses sum to 3 log(1+exp(-2J)).
    # Eight tensor entries +/- J give gradient 8*l2*J - 6/(1+exp(2J)).
    j = result.threebody_rms
    assert 8 * 0.1 * j == pytest.approx(6 / (1 + np.exp(2 * j)), abs=1e-5)
    pair_only = [
        (a, b, c) for a, b, c in product("AC", repeat=3) for _ in range(5 if a == b else 1)
    ]
    (result,), _ = global_triples(alignment(pair_only), [(0, 0, 0)])
    assert result.threebody_rms < 1e-10


def test_global_all_constant_or_inadequate_complete_rows():
    msa = alignment([("A", "A", "A"), ("A", "A", "A")])
    (score,), info = global_triples(msa, [(0, 0, 0)])
    assert score.threebody_rms == 0 and info["variable_sites"] == 0
    (score,), info = global_triples(msa, [(0, 0, 0)], min_valid=3)
    assert np.isnan(score.threebody_rms) and info["status"] == "insufficient complete rows"


def test_global_distinguishes_two_triples_in_one_model():
    rows = [(a + extra, b, c) for a, b, c in xor_rows() for extra in "AC"]
    (parity, independent), info = global_triples(alignment(rows), [(0, 0, 0), (1, 0, 0)])
    assert info["variable_sites"] == 4 and info["status"] == "converged"
    assert parity.threebody_rms > 0.5
    assert independent.threebody_rms < 1e-10


def test_candidate_budget_and_seed_without_cartesian_allocation():
    axes = (range(100_000),) * 3
    selected, n = sample_candidates(axes, 5, 777)
    assert n == 10**15 and len(set(selected)) == 5
    assert (selected, n) == sample_candidates(axes, 5, 777)
    assert selected != sample_candidates(axes, 5, 778)[0]


def test_stratified_shuffles_and_conservative_nulls():
    msa = alignment(xor_rows())
    groups = exchangeability_groups(4, ["g1", "g2", "g1", "g2"])
    draw = permute_components(msa, groups, np.random.default_rng(22))
    assert draw.a.ids == msa.a.ids
    for group in groups:
        for component, original in zip(draw.components, msa.components):
            assert {component.ids[i] for i in group} == {original.ids[i] for i in group}
    with pytest.raises(ValueError, match="at least two"):
        exchangeability_groups(4, list("abcd"))
    obs = {"x": np.array([3.0, 2.0, np.nan])}
    p, q = permutation_test(
        msa, obs, lambda _: {"x": np.array([1.0, np.nan, 1.0])}, permutations=9, seed=0
    )["x"]
    np.testing.assert_allclose(p, [0.1, 1.0, 1.0])
    np.testing.assert_allclose(q, [0.3, 1.0, 1.0])
    np.testing.assert_allclose(bh_adjust([0.04, 0.01, 0.03, 0.002]), [0.04, 0.02, 0.04, 0.008])


def test_contact_ties_and_unlabelled_pairs():
    result = contact_metrics({1: 1, 2: 0, 3: 1, 4: 0}, {1: 2.0, 2: 2.0, 3: 1.0, 99: 100.0})
    assert result["scored"] == 3 and result["unscored"] == 1
    assert result["auroc"] == pytest.approx(0.25)
    assert result["average_precision"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_outputs_provenance_and_null_reproducibility(tmp_path):
    rows = [(a + "-A", b + "C", c + "A") for a, b, c in xor_rows()] * 3
    msa = alignment(rows)
    original_maps = [build_reference_map(c, c.ids[0], start=101) for c in msa.components]
    filtered = apply_triplet_masks(msa, [True, False, True], None, None)
    maps = [
        subset_reference_map(m, cols)
        for m, cols in zip(
            original_maps, (filtered.a_columns, filtered.b_columns, filtered.c_columns)
        )
    ]
    weights = sequence_weights(filtered.msa)
    candidates = tmp_path / "candidates.tsv"
    candidates.write_text("c_original_col\tb_original_col\ta_original_col\n1\t1\t1\n2\t2\t3\n")
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "x_component\tx_reference_id\tx_reference_pos\ty_component\ty_reference_id\ty_reference_pos\tcontact\n"
        "A\tA0\t101\tB\tB0\t101\t1\nB\tB0\t102\tA\tA0\t102\t0\n"
    )
    for run in ("one", "two"):
        result = write_advanced_results(
            tmp_path / run,
            filtered.msa,
            maps,
            weights,
            dca=True,
            threebody=True,
            triplet_candidates=candidates,
            permutations=9,
            seed=7,
            contacts=labels,
            max_pairs=1,
        )
        assert result["hypotheses"] == {"mi": 12, "dca": 12, "threebody": 2}
        assert result["candidates"] == {
            "eligible": 8,
            "tested": 2,
            "finite": 2,
            "selection": "file",
        }
        assert result["outputs"]["permutation_dca"]["rows"] == 12
        assert result["outputs"]["dca_ab"]["rows"] == 1
    for path in (tmp_path / "one").iterdir():
        assert path.read_bytes() == (tmp_path / "two" / path.name).read_bytes()
    with (tmp_path / "one" / "threebody_abc.tsv").open() as handle:
        ranks = list(csv.DictReader(handle, delimiter="\t"))
    assert [r["a_original_col"] for r in ranks] == ["1", "3"]
    assert ranks[1]["a_col"] == "2" and ranks[1]["a_reference_pos"] == "102"
    metrics = json.loads((tmp_path / "one" / "contacts.json").read_text())
    assert metrics["methods"]["dca"]["scored"] == 2


@pytest.mark.parametrize("limit,expected", [(4, "uniform"), (8, "exhaustive"), (9, "exhaustive")])
def test_candidate_manifest_identifies_selection(tmp_path, limit, expected):
    msa = alignment([(a * 2, b * 2, c * 2) for a, b, c in xor_rows()])
    maps = [build_reference_map(c, c.ids[0]) for c in msa.components]
    result = write_advanced_results(
        tmp_path, msa, maps, sequence_weights(msa), threebody=True, max_triplets=limit
    )
    assert result["candidates"] == {
        "eligible": 8, "tested": min(limit, 8), "finite": min(limit, 8), "selection": expected,
    }
