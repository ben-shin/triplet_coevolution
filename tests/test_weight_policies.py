"""Independent checks of weight semantics, row binding, support cutoffs and rematching."""

import csv
import os
from fractions import Fraction
from math import fsum
from pathlib import Path

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.core.coordinates import build_reference_map
from triplet_coevolution.inference.advanced import write_advanced_results
from triplet_coevolution.inference.global_potts import global_triples
from triplet_coevolution.inference.mi import apc, triplet_mi, weighted_mi
from triplet_coevolution.inference.potts import direct_couplings
from triplet_coevolution.inference.scores import pair_coverage
from triplet_coevolution.inference.threebody import fit_triples
from triplet_coevolution.inference.weights import (
    rematch_evaluator,
    sequence_weights,
    weighting_preflight,
)
from triplet_coevolution.io.score_store import from_results, load_score_store, write_score_store
from triplet_coevolution.validation.nulls import permute_components


def make(parts, ids):
    comps = tuple(
        ComponentMSA(k, m, tuple(f"{k}_{key}" for key in ids), seqs)
        for k, m, seqs in zip("ABC", ("protein", "dna", "rna"), parts)
    )
    rows = tuple(TripletMatch(key, *(c.ids[i] for c in comps)) for i, key in enumerate(ids))
    return MatchedTripletMSA(*comps, rows)


def raw_comparisons(parts, policy, threshold, overlap):
    # Strings are compared independently of the production bit packing.
    n = len(parts[0].ids)
    truth = np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(i):
            pairs = []
            for part in parts:
                letters = CANONICAL[part.molecule]
                sites = [
                    (x, y)
                    for x, y in zip(part.sequences[i], part.sequences[j])
                    if x in letters and y in letters
                ]
                pairs.append((sum(x == y for x, y in sites), len(sites), part.n_sites))
            if policy == "concatenated":
                pairs = [tuple(map(sum, zip(*pairs)))]
            if all(o and float(Fraction(o, L)) >= overlap for _, o, L in pairs):
                identity = sum(Fraction(m, o) for m, o, _ in pairs) / len(pairs)
                truth[i, j] = truth[j, i] = float(identity) >= threshold
    return truth


@pytest.mark.parametrize(
    "policy,component",
    [
        ("equal_component", None),
        ("concatenated", None),
        ("cluster_representatives", None),
        ("component", "A"),
        ("component", "C"),
    ],
)
def test_blocked_mixed_identity_against_string_reference(policy, component):
    msa = make(
        [
            ("A?AACCA", "AXAACCA", "AAAA-CA", "CCCAACA", "CCCAACA"),
            ("TT", "TT", "TC", "TC", "T-"),
            ("U?U", "UU-", "UUU", "CUA", "CUA"),
        ],
        ("z", "x", "v", "m", "b"),
    )
    opts = {"policy": policy, "component": component}
    parts = (getattr(msa, component.lower()),) if component else msa.components
    graph = raw_comparisons(parts, policy, 0.75, 0.55)
    native = sequence_weights(msa, 0.75, 0.55, **opts)
    limit = weighting_preflight(msa, **opts)["peak_bytes"] - 400
    bounded = sequence_weights(msa, 0.75, 0.55, memory_budget_bytes=limit, **opts)
    assert bounded.preflight["peak_bytes"] <= limit
    assert bounded.preflight["block_rows"] < msa.n_observations
    np.testing.assert_array_equal(bounded.neighbor_counts, graph.sum(axis=0))
    np.testing.assert_array_equal(native.weights, bounded.weights)
    if policy == "cluster_representatives":
        # Reachability by independent transitive closure.
        reach = graph.copy()
        for k in range(len(graph)):
            reach |= reach[:, k, None] & reach[k, None, :]
        reps = [min(np.array(native.observation_ids)[neighbors]) for neighbors in reach]
        assert native.cluster_ids == tuple(reps)
    else:
        np.testing.assert_array_equal(native.weights, 1 / graph.sum(axis=0))


def test_concatenation_unequal_lengths_and_one_absent_component():
    msa = make([("A" * 24, "A" * 24), ("AA", "CC"), ("UU", "GG")], ("x", "y"))
    assert sequence_weights(msa, policy="concatenated").n_eff == 1
    assert sequence_weights(msa).n_eff == 2
    assert sequence_weights(msa, policy="component", component="C").n_eff == 2
    partial = make([("A" * 24, "A" * 24), ("AA", "--"), ("UU", "GG")], ("x", "y"))
    assert sequence_weights(partial, policy="concatenated").n_eff == 1
    with pytest.raises(ValueError, match="canonical"):
        sequence_weights(partial)


def test_deterministic_single_linkage_and_row_binding():
    ids = ("z", "a", "m")
    seqs = ("AAAA", "AAAC", "AACC")
    msa = make([seqs, seqs, seqs], ids)
    options = {"identity_threshold": 0.75, "policy": "cluster_representatives"}
    result = sequence_weights(msa, **options)
    assert [r["cluster_id"] for r in result.table_rows()] == ["a"] * 3
    assert result.weights.tolist() == [0, 1, 0]
    assert result.neighbor_counts.tolist() == [2, 3, 2]
    perm = (2, 0, 1)
    reordered = make([[seqs[i] for i in perm]] * 3, tuple(ids[i] for i in perm))
    new = sequence_weights(reordered, **options)
    assert sorted(result.table_rows(), key=lambda r: r["observation_id"]) == sorted(
        new.table_rows(), key=lambda r: r["observation_id"]
    )
    for scorer, args in [
        (triplet_mi, ()),
        (direct_couplings, ()),
        (fit_triples, ([(0, 0, 0)],)),
        (global_triples, ([(0, 0, 0)],)),
    ]:
        with pytest.raises(ValueError, match="IDs or order"):
            scorer(reordered, *args, weights=result)


def test_metadata_validation_and_effective_depth():
    msa = make([("A", "A", "C")] * 3, ("b", "a", "c"))
    labels = {"a": "east", "c": "west", "b": "east"}
    result = sequence_weights(msa, policy="clade_balanced", clades=labels)
    assert result.neighbor_counts is None and result.summary()["clade_sizes"] == {
        "east": 2,
        "west": 1,
    }
    assert result.kish == pytest.approx(8 / 3) and result.n_eff == 2
    w = {"a": 4, "b": 0, "c": 2}
    supplied = sequence_weights(msa, policy="supplied", supplied_weights=w)
    assert supplied.weights.tolist() == [0, 4, 2]
    assert supplied.kish == pytest.approx(1.8) and supplied.n_eff == 6
    assert weighting_preflight(msa, policy="supplied")["unordered_distinct_row_pairs"] == 0
    for invalid in [
        {**w, "bad": 0},
        {"a": 4},
        {"a": False, "b": 1, "c": 0},
        {"a": float("nan"), "b": 1, "c": 0},
        dict.fromkeys(w, 0),
        dict.fromkeys(w, 1e308),
        dict.fromkeys(w, -1),
    ]:
        with pytest.raises(ValueError):
            sequence_weights(msa, policy="supplied", supplied_weights=invalid)
    for kwargs in [
        {"policy": "clade_balanced", "clades": {**labels, "a": " "}},
        {"policy": "clade_balanced"},
        {"policy": "equal_component", "clades": labels},
        {"policy": "component"},
        {"policy": "supplied", "supplied_weights": w, "component": "C"},
        {"memory_budget_bytes": 1},
    ]:
        with pytest.raises(ValueError):
            sequence_weights(msa, **kwargs)
    with pytest.raises(ValueError, match="nonempty observation ID"):
        sequence_weights(make([("A", "C")] * 3, ("a", " ")))
    with pytest.raises(ValueError, match="duplicate observation"):
        MatchedTripletMSA(
            *msa.components,
            tuple(TripletMatch("same", m.a_id, m.b_id, m.c_id) for m in msa.matches),
        )


@pytest.mark.parametrize(
    "policy",
    [
        "equal_component",
        "concatenated",
        "component",
        "cluster_representatives",
        "clade_balanced",
        "supplied",
    ],
)
def test_component_null_uses_correct_design(policy):
    rows = ("AAAA", "AAAA", "AAAC", "CCCC", "CCCC")
    ids = ("z", "s", "p", "q", "a")
    msa = make([rows, rows, rows], ids)
    clades = dict(zip(ids, ("near", "near", "near", "far", "far")))
    options = {"policy": policy}
    if policy == "component":
        options["component"] = "B"
    elif policy == "supplied":
        options["supplied_weights"] = dict(zip(ids, (1, 0.75, 0.5, 2, 0)))
    elif policy == "clade_balanced":
        options["clades"] = clades
    w = sequence_weights(msa, **options)
    strata = tuple(clades.values()) if policy == "clade_balanced" else None
    groups = (np.arange(3), np.arange(3, 5)) if strata else (np.arange(5),)
    evaluate = rematch_evaluator(msa, w, strata)
    changed = False
    for seed in range(8):
        shuffled = permute_components(msa, groups, np.random.default_rng(seed))
        direct = sequence_weights(shuffled, **options)
        np.testing.assert_array_equal(evaluate(shuffled).weights, direct.weights)
        changed |= not np.array_equal(direct.weights, w.weights)
    if policy not in ("clade_balanced", "supplied"):
        assert changed
    if policy == "clade_balanced":
        with pytest.raises(ValueError, match="nested"):
            rematch_evaluator(msa, w, None)


@pytest.mark.parametrize("kernel", ["batched", "v1_scalar"])
def test_threshold_before_apc_fit_null_and_archive(tmp_path, kernel):
    msa = make([("AA", "CA", "CC", "AC")] * 2 + [("AA", "CA", "C?", "A-")], ("r", "s", "t", "v"))
    weights = sequence_weights(
        msa, policy="supplied", supplied_weights={"r": 0.25, "s": 0.25, "t": 0.5, "v": 0}
    )
    maps = tuple(build_reference_map(c, c.ids[0]) for c in msa.components)
    mi = triplet_mi(msa, weights, kernel=kernel, min_valid_weight=0.75)
    assert mi.ac.n_valid.tolist() == [[3, 2], [3, 2]]
    assert mi.ac.valid_weight.tolist() == [[1, 0.5], [1, 0.5]]
    assert np.isnan(mi.ac.mi_bits[:, 1]).all()
    np.testing.assert_array_equal(mi.ac.mi_apc, apc(mi.ac.mi_bits))
    dca = direct_couplings(msa, weights, min_valid_weight=0.75)
    assert np.isnan(dca["AC"].frobenius[:, 1]).all()
    np.testing.assert_array_equal(dca["AC"].corrected, apc(dca["AC"].frobenius))
    local = fit_triples(msa, [(0, 0, 1)], weights, min_valid_weight=0.75)[0]
    assert np.isnan(local.threebody_rms) and local.valid_weight == 0.5 and local.n_valid == 2
    joint, status = global_triples(msa, [(0, 0, 0)], weights, min_valid_weight=0.75)
    assert np.isnan(joint[0].threebody_rms) and status["status"] == "insufficient complete weight"
    assert np.isfinite(
        weighted_mi(np.array([0, 1]), np.array([0, 1]), 2, 2, [0.25, 0.25], min_valid_weight=0.5)
    )
    assert np.isnan(
        weighted_mi(
            np.array([0, 1]),
            np.array([0, 1]),
            2,
            2,
            [0.25, 0.25],
            min_valid_weight=np.nextafter(0.5, 1.0),
        )
    )
    coverage = pair_coverage(mi.ac, maps[0], maps[2], "mi_apc", min_valid_weight=0.75)
    assert coverage["below_min_valid_weight_after_count"] == 2 and coverage["finite"] == 2
    store = from_results(
        maps,
        mi,
        dca,
        provenance={
            "software_version": "test",
            "inputs": dict.fromkeys(("A", "B", "C", "matches"), "0" * 64),
        },
    )
    restored = load_score_store(write_score_store(tmp_path / "archive", store))
    assert restored.eligible("AC").tolist() == [[True, False], [True, False]]
    assert restored.to_results()[0].kernel == kernel and restored.min_valid_weight == 0.75
    manifest = write_advanced_results(
        tmp_path / "null",
        msa,
        maps,
        weights,
        permutations=3,
        min_valid_weight=0.75,
        mi_result=mi,
        dca=True,
        threebody=True,
    )
    with (tmp_path / "null" / "threebody_support.tsv").open() as f:
        support = list(csv.DictReader(f, delimiter="\t"))
    assert len(support) == 8
    low = [r for r in support if r["weight_pass"] == "False"]
    assert len(low) == 4 and all(
        r["count_pass"] == "True" and r["score_finite"] == "False" for r in low
    )
    assert {r["c_original_col"] for r in low} == {"2"}
    assert manifest["settings"]["min_valid_weight"] == 0.75
    assert manifest["outputs"]["permutation_mi"]["rows"] == 8
    with pytest.raises(ValueError, match="matching min_valid"):
        write_advanced_results(
            tmp_path / "invalid", msa, maps, weights, mi_result=mi, min_valid_weight=1.1
        )


def test_different_reduction_orders_on_inclusive_weight_cutoff():
    vals = np.array([0.1] * 30)
    n = len(vals)
    ids = tuple(str(i) for i in range(n))
    msa = make([tuple("AC"[i % 2] for i in range(n))] * 3, ids)
    cutoff = fsum(vals)
    for threshold, passed in [(cutoff, True), (np.nextafter(cutoff, np.inf), False)]:
        for kernel in ("batched", "v1_scalar"):
            outcome = triplet_mi(msa, vals, kernel=kernel, min_valid_weight=threshold)
            assert np.isfinite(outcome.ab.mi_bits[0, 0]) == passed
        assert (
            np.isfinite(
                direct_couplings(msa, vals, min_valid_weight=threshold)["AB"].frobenius[0, 0]
            )
            == passed
        )
        assert (
            np.isfinite(
                fit_triples(msa, [(0, 0, 0)], vals, min_valid_weight=threshold)[
                    0
                ].co_information_bits
            )
            == passed
        )


def test_private_weighting_holdout_and_legacy_sum(monkeypatch):
    root = os.environ.get("TRIPLET_GATABC_WEIGHT_ROOT")
    if not root:
        pytest.skip("Set TRIPLET_GATABC_WEIGHT_ROOT to verify the private prepared input package")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from weighting_policies import taxon_test

    result = taxon_test(Path(root))
    assert result["baseline_sum"] == 505.49112803812943
    assert result["baseline_kish"] == pytest.approx(590.9818664111334, abs=1e-11, rel=0)
    assert (result["n"], result["train_n"], result["held_n"], result["held_taxa"]) == (
        888,
        736,
        152,
        151,
    )
    assert result["held_canonical_symbols"] == 154904
    assert result["policies"]["equal_component"]["paired_delta"] == pytest.approx(
        [0.031267367688479056, 0.009372406222918642, 0.05299164258118836], abs=1e-12, rel=0
    )


def test_native_archive_histogram_lexical_numeric_keys_and_collisions(tmp_path):
    ids = tuple(f"obs{i}" for i in range(12))
    cohort = make([("A",) * 10 + ("C",) * 2] * 3, ids)
    weights = sequence_weights(cohort)
    assert weights.summary()["neighbor_count_distribution"] == {"10": 10, "2": 2}
    maps = tuple(build_reference_map(c, c.ids[0]) for c in cohort.components)
    meta = {
        "software_version": "test",
        "inputs": dict.fromkeys(("A", "B", "C", "matches"), "0" * 64),
        "weighting": weights.summary(),
        "extra_counts": {2: 3, 10: 1},
    }
    store = from_results(maps, triplet_mi(cohort, weights), provenance=meta)
    restored = load_score_store(write_score_store(tmp_path / "good", store))
    assert restored.provenance["extra_counts"] == {"2": 3, "10": 1}
    store.provenance["extra_counts"] = {2: 0, "2": 1}
    with pytest.raises(ValueError, match="duplicate JSON field"):
        write_score_store(tmp_path / "bad", store)
