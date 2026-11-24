"""Independent blocked split, bootstrap, tie-credit and held-out leakage checks."""

import csv
import json

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.validation._clade_scores import rank_scores
from triplet_coevolution.validation.clade_splits import identity, load_split_plan
from triplet_coevolution.validation.clade_stability import (
    bootstrap_sample,
    evaluate_stability,
    make_split_plan,
    write_stability,
)


def sample(sizes=(4, 3, 6, 5)):
    labels = {f"row.{g}/{i}": f"clade{g}" for g, n in enumerate(sizes) for i in range(n)}
    ids = tuple(labels)
    rng = np.random.default_rng(715)
    a = rng.integers(0, 2, (len(ids), 3))
    b = rng.integers(0, 2, (len(ids), 3))
    c = rng.integers(0, 2, (len(ids), 3))
    b[:, 1] = a[:, 0]
    c[:, 2] = a[:, 0] ^ b[:, 2]
    parts = tuple(
        ComponentMSA(
            r,
            "dna",
            tuple(r + "-" + key for key in ids),
            tuple("".join("AC"[i] for i in row) for row in arr),
        )
        for r, arr in zip("ABC", (a, b, c), strict=True)
    )
    matches = tuple(TripletMatch(key, *(p.ids[i] for p in parts)) for i, key in enumerate(ids))
    maps = tuple(
        ReferenceCoordinates(r, r + "-reference", (4, 7, 11), (26, None, 27), ("A", "-", "C"), 26)
        for r in "ABC"
    )
    return MatchedTripletMSA(*parts, matches), maps, labels


def test_exact_manifest_partition_replay_and_tamper(tmp_path):
    msa, maps, groups = sample((3, 2, 1))
    result = evaluate_stability(msa, maps, groups, source={"external": True})
    assert len(result.plan.folds) == 3
    for fold in result.plan.folds:
        assert not set(fold.test_ids) & set(fold.train_ids)
        assert not set(fold.test_groups) & set(fold.train_groups)
        assert set(fold.test_ids + fold.train_ids) == set(groups)
    write_stability(tmp_path, result)
    path = tmp_path / "split_manifest.json"
    assert load_split_plan(path, groups).metadata() == result.plan.metadata()
    with pytest.raises(ValueError, match="order"):
        load_split_plan(path, reversed(groups))
    data = json.loads(path.read_text())
    data["folds"][0]["train_ids"].append(data["folds"][0]["test_ids"][0])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_split_plan(path)
    data["plan_sha256"] = identity({k: v for k, v in data.items() if k != "plan_sha256"})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="partition"):
        load_split_plan(path)
    for bad in (
        {k: v for k, v in groups.items() if k != next(iter(groups))},
        {**groups, "other": "s"},
        {**groups, next(iter(groups)): "  "},
    ):
        with pytest.raises(ValueError, match="clade"):
            make_split_plan(groups, bad)
    with pytest.raises(ValueError, match="known"):
        make_split_plan(groups, groups, test_groups=["absent"])
    with pytest.raises(ValueError, match="training"):
        make_split_plan(groups, groups, test_groups=sorted(set(groups.values())))
    assert (
        make_split_plan(groups, dict.fromkeys(groups, "same")).metadata()["status"]
        == "insufficient_independent_blocks"
    )


def test_block_draw_keeps_triple_and_aliased_clades():
    msa, _maps, groups = sample((1, 3, 5, 7))
    indexes = {r.observation_id: i for i, r in enumerate(msa.matches)}
    fold = make_split_plan(groups, groups, test_groups=["clade3"]).folds[0]
    draws = [
        bootstrap_sample(
            msa, groups, seed=583, fold_id=fold.fold_id, replicate=i, eligible_ids=fold.train_ids
        )
        for i in range(12)
    ]
    assert any(len(set(d.sampled_groups)) < 3 for d in draws)
    assert len({len(d.source_ids) for d in draws}) > 1
    for i, d in enumerate(draws):
        assert len(set(d.groups.values())) == 3 and all(g != "clade3" for g in d.sampled_groups)
        assert d.source_ids == tuple(
            key for g in d.sampled_groups for key in fold.train_ids if groups[key] == g
        )
        assert len(set(d.groups)) == len(d.source_ids)
        for part, orig in zip(d.msa.components, msa.components):
            assert list(part.sequences) == [orig.sequences[indexes[key]] for key in d.source_ids]
            assert len(set(part.ids)) == len(d.source_ids)
        assert (
            d.sampled_groups
            == bootstrap_sample(
                msa,
                groups,
                seed=583,
                fold_id=fold.fold_id,
                replicate=i,
                eligible_ids=fold.train_ids,
            ).sampled_groups
        )
    with pytest.raises(ValueError, match="splits a clade"):
        bootstrap_sample(
            msa, groups, seed=0, fold_id=fold.fold_id, replicate=0, eligible_ids=tuple(groups)[2:]
        )


def test_holdout_cannot_alter_training_states_weights_candidates_or_masks(tmp_path):
    msa, maps, groups = sample()
    held = "clade2"
    changed = [
        ComponentMSA(
            c.name,
            c.molecule,
            c.ids,
            tuple(
                s if groups[m.observation_id] != held else "TTT"
                for s, m in zip(c.sequences, msa.matches)
            ),
        )
        for c in msa.components
    ]
    altered = MatchedTripletMSA(*changed, msa.matches)
    settings = {
        "statistics": (
            "mi_bits",
            "mi_apc",
            "dca_frobenius",
            "dca_apc",
            "connected_information_bits",
            "co_information_bits",
            "threebody_rms",
        ),
        "test_groups": [held],
        "top_k": 2,
        "min_site_entropy": 0.1,
        "candidate_options": {"mode": "hybrid", "maximum": 5},
        "maxent_options": {"prior_mass": 1.0},
        "min_state_count": 2,
        "weight_options": {"policy": "equal_component"},
    }
    r1 = evaluate_stability(msa, maps, groups, **settings)
    r2 = evaluate_stability(altered, maps, groups, **settings)
    assert r1.summary["source_msa_sha256"] != r2.summary["source_msa_sha256"]
    assert r1.summary["training_designs"] == r2.summary["training_designs"]
    assert [r["training_score"] for r in r1.pair_rows] == [
        r["training_score"] for r in r2.pair_rows
    ]
    d = next(iter(r1.summary["training_designs"].values()))
    assert set(d["state_support"]["training_observations"]) == set(d["train_ids"])
    assert (
        set(d["train_ids"]).isdisjoint(r1.plan.folds[0].test_ids)
        and len(d["triple_candidates"]) == 5
    )
    assert [r["test_score"] for r in r1.pair_rows] != [r["test_score"] for r in r2.pair_rows]
    write_stability(tmp_path, r1)
    assert {
        r["interface"]
        for r in csv.DictReader((tmp_path / "stability_pairs.tsv").open(), delimiter="\t")
    } == {"AB", "AC", "BC", "ABC"}


def test_train_only_weights_unequal_blocks_percentiles_and_original_columns():
    msa, maps, groups = sample()
    opts = {
        "test_groups": ["clade2"],
        "bootstrap_draws": 24,
        "seed": 639,
        "top_k": 2,
        "require_reference": True,
        "weight_options": {"policy": "clade_balanced"},
        "min_valid": 2,
    }
    r = evaluate_stability(msa, maps, groups, **opts)
    assert all(row["weight_sum"] == pytest.approx(3.0) for row in r.bootstrap_rows)
    assert all(row["test_weight_sum"] == pytest.approx(1.0) for row in r.fold_rows)
    assert any(row["bootstrap_interval_status"] == "available" for row in r.pair_rows)
    assert all(row["universe_size"] == 4 for row in r.fold_rows)
    assert all(
        row.get("A_column") in (None, 5, 12) and row.get("A_position") in (None, 26, 27)
        for row in r.pair_rows
    )
    # Independent oracle, direct unweighted replication of an exact binary MI in a held-out block.
    key = next(p for p in r.pair_rows if p["interface"] == "AB")
    a, b = [maps[i].original_positions.index(key[f"{name}_column"]) for i, name in enumerate("AB")]
    rows = [i for i, m in enumerate(msa.matches) if groups[m.observation_id] == "clade2"]
    hist = np.zeros((2, 2))
    for i in rows:
        hist[msa.a.matrix[i, a], msa.b.matrix[i, b]] += 1
    hist /= hist.sum()
    expected = np.outer(hist.sum(axis=1), hist.sum(axis=0))
    present = hist > 0
    assert key["test_score"] == pytest.approx(
        (hist[present] * np.log2(hist[present] / expected[present])).sum()
    )
    # Reconstruct block-instance weighting from original sequences and recorded clade draws.
    boot_scores = []
    for draw in r.bootstrap_rows:
        hist = np.zeros((2, 2))
        for clade in draw["sampled_groups"]:
            members = [i for i, m in enumerate(msa.matches) if groups[m.observation_id] == clade]
            for i in members:
                hist[msa.a.matrix[i, a], msa.b.matrix[i, b]] += 1 / len(members)
        hist /= hist.sum()
        expected = np.outer(hist.sum(axis=1), hist.sum(axis=0))
        present = hist > 0
        boot_scores.append(
            float((hist[present] * np.log2(hist[present] / expected[present])).sum())
        )
    np.testing.assert_allclose(
        [key["bootstrap_score_lower"], key["bootstrap_score_upper"]],
        np.quantile(boot_scores, [0.025, 0.975]),
        atol=1e-14,
    )
    duplicate = evaluate_stability(msa, maps, groups, **opts)
    assert duplicate.pair_rows == r.pair_rows and duplicate.bootstrap_rows == r.bootstrap_rows


def test_singletons_degenerate_weights_and_budget():
    msa, maps, groups = sample((4, 2, 1))
    r = evaluate_stability(msa, maps, groups, test_groups=["clade2"], bootstrap_draws=20)
    assert all(x["test_finite"] == 0 for x in r.fold_rows)
    assert all(
        x["bootstrap_interval_status"] == "insufficient_training_blocks" for x in r.pair_rows
    )
    weights = {k: float(g != "clade0") for k, g in groups.items()}
    r = evaluate_stability(
        msa,
        maps,
        groups,
        test_groups=["clade0"],
        weight_options={"policy": "supplied", "supplied_weights": weights},
    )
    assert all(x["test_weight_sum"] == 0 and x["test_finite"] == 0 for x in r.fold_rows)
    with pytest.raises(ValueError, match="exactly"):
        evaluate_stability(
            msa, maps, groups, weight_options={"policy": "supplied", "supplied_weights": {"x": 1.0}}
        )
    with pytest.raises(ValueError, match="max_fits"):
        evaluate_stability(msa, maps, groups, max_fits=2)
    with pytest.raises(ValueError, match="max_pair_cells"):
        evaluate_stability(msa, maps, groups, max_pair_cells=2)
    with pytest.raises(ValueError, match="interval workspace"):
        evaluate_stability(msa, maps, groups, top_k=10000, bootstrap_draws=60)
    many = {str(i): str(i) for i in range(2000)}
    with pytest.raises(ValueError, match="row assignments"):
        make_split_plan(many, many)
    for policy in (
        "uniform",
        "equal_component",
        "concatenated",
        "cluster_representatives",
        "component",
    ):
        opts = {"policy": policy}
        if policy == "component":
            opts["component"] = "B"
        assert evaluate_stability(
            msa, maps, groups, test_groups=["clade0"], weight_options=opts
        ).fold_rows


def test_boundary_credit_and_tied_rank_correlations():
    ranks = rank_scores(np.array([0.1, 0.9, 0.7, 0.7, 0.7, np.nan]), "mi_bits", 3)
    assert ranks.top.tolist() == [1, 2, 3]
    np.testing.assert_allclose(ranks.credit, [0, 1, 2 / 3, 2 / 3, 2 / 3, 0])
    np.testing.assert_allclose(ranks.mid, [5, 1, 3, 3, 3, np.nan], equal_nan=True)
    assert ranks.cutoff_tie_size == 3 and ranks.cutoff_tie_selected == 2
    assert rank_scores(np.array([-0.4, 0.8, -0.9]), "co_information_bits", 1).top.tolist() == [2]
