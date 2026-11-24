"""Candidate geometry, evidence isolation, protected sampling and replay identities."""

import csv
import json
from itertools import product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.candidates import (
    CandidateInputError,
    _draw,
    candidate_id,
    select_candidates,
    write_candidate_manifest,
)
from triplet_coevolution.inference.threebody import sample_candidates
from triplet_coevolution.inference.weights import sequence_weights


def mapped():
    return (
        ReferenceCoordinates("A", "ref-A", (1, 8, 12), (7, None, 8), ("F", "-", "W"), 7),
        ReferenceCoordinates("B", "ref-B", (0, 3, 9), (21, 17, 12), ("T", "C", "A"), 21, "-"),
        ReferenceCoordinates("C", "ref-C", (0, 2, 4), (4, 5, 6), ("U", "U", "G"), 4),
    )


def observed(seed=827):
    rng = np.random.default_rng(seed)
    ids = tuple(f"obs{i}" for i in range(24))
    parts = tuple(
        ComponentMSA(c, m, ids, tuple("".join(row) for row in rng.choice(list(v), (24, 3))))
        for c, m, v in zip("ABC", ("protein", "dna", "rna"), ("FS", "CG", "UG"))
    )
    return MatchedTripletMSA(*parts, tuple(TripletMatch(i, i, i, i) for i in ids))


def tsv(path, head, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(head)
        writer.writerows(rows)
    return path


def test_uniform_legacy_order_signature_and_retained_vs_reference(tmp_path):
    maps = mapped()
    axes = ((0, 2), (0, 1, 2), (0, 1, 2))
    result = select_candidates(maps, axes, maximum=7, seed=46)
    old, n = sample_candidates(axes, 7, 46)
    assert result.triples == tuple(old) and n == result.summary["eligible_triples"] == 18
    assert (
        result.summary["retained_cartesian_triples"] == 27
        and result.summary["coordinate_exclusions"] == 9
    )
    assert sum(result.batches(3), ()) == result.triples
    manifest = write_candidate_manifest(tmp_path, result)
    assert json.loads((tmp_path / "candidate_manifest.json").read_text()) == manifest
    with (tmp_path / "candidate_selection.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 7
    assert [r["candidate_id"] for r in rows] == [candidate_id(maps, t) for t in old]
    assert all(int(r["a_original_col"]) in (2, 13) and r["a_reference_pos"] for r in rows)
    assert manifest["scope"] == "coordinate_only"
    assert select_candidates(maps, axes, mode="exhaustive", maximum=18).triples == tuple(
        product(*axes)
    )
    with pytest.raises(ValueError, match="exhaustive"):
        select_candidates(maps, axes, mode="exhaustive", maximum=17)
    empty = select_candidates(maps, ([], axes[1], axes[2]), maximum=1)
    assert empty.triples == () and empty.summary["eligible_triples"] == 0
    with pytest.raises(ValueError, match="increasing"):
        select_candidates(maps, ((2, 0), *axes[1:]))


def test_explicit_and_external_evidence_are_reference_and_source_specific(tmp_path):
    maps = mapped()
    axes = (tuple(range(3)),) * 3
    f = tsv(
        tmp_path / "defined.tsv",
        ("a_original_col", "b_original_col", "c_original_col"),
        [(2, 4, 5)],
    )
    explicit = select_candidates(maps, axes, mode="explicit", candidate_file=f)
    assert explicit.triples == ((0, 1, 2),)
    assert explicit.summary["file_origin_provenance"] == "assumed_legacy_default"
    adaptive = select_candidates(
        maps, axes, mode="explicit", candidate_file=f, file_origin="same_data"
    )
    assert (
        adaptive.summary["same_data_screen"]
        and adaptive.summary["file_origin_provenance"] == "declared"
    )
    header = [f"{r}_{k}" for r in "abc" for k in ("reference_id", "reference_pos")]
    ref = tsv(tmp_path / "prior.tsv", header, [("ref-A", 7, "ref-B", 17, "ref-C", 6)])
    prior = select_candidates(
        maps, axes, mode="prior", candidate_file=ref, prior_source="held-out annotation v1"
    )
    assert (
        prior.triples == explicit.triples
        and prior.summary["file_sha256"] != explicit.summary["file_sha256"]
    )
    assert (
        prior.summary["used_external_prior"]
        and prior.summary["prior_source"] == "held-out annotation v1"
    )
    with pytest.raises(ValueError, match="evidence source"):
        select_candidates(maps, axes, mode="prior", candidate_file=ref)
    tsv(
        ref,
        header,
        [("ref-A", 7, "ref-B", 17, "ref-C", 6)] * 2
        + [("other", 7, "ref-B", 17, "ref-C", 6), ("ref-A", "na", "ref-B", 17, "ref-C", 6)],
    )
    with pytest.raises(CandidateInputError) as failed:
        select_candidates(maps, axes, mode="explicit", candidate_file=ref)
    assert failed.value.counts == {"rows": 4, "duplicates": 1, "invalid": 1, "outside_axes": 1}
    tsv(ref, header + [header[0]], [("ref-A", 7, "ref-B", 17, "ref-C", 6, "ref-A")])
    with pytest.raises(CandidateInputError, match="requires original"):
        select_candidates(maps, axes, mode="explicit", candidate_file=ref)


@pytest.mark.parametrize("mode", ["triangle", "wedge"])
def test_pair_screen_matches_independent_brute_ranking_with_nan_edges(mode):
    maps = mapped()
    axes = (tuple(range(3)),) * 3
    rng = np.random.default_rng(66)
    blocks = {p: rng.integers(8, size=(3, 3)).astype(float) for p in ("AB", "AC", "BC")}
    blocks["AB"][1, :] = np.nan
    blocks["AC"][1, 1:] = np.nan
    threshold = 3 if mode == "triangle" else 2
    expected = []
    for a, b, c in product(*axes):
        edge = [blocks["AB"][a, b], blocks["AC"][a, c], blocks["BC"][b, c]]
        valid = sorted((e for e in edge if np.isfinite(e)), reverse=True)
        if len(valid) >= threshold:
            expected.append((-valid[threshold - 1], (a, b, c)))
    expected.sort()
    native = select_candidates(maps, axes, mode=mode, pair_scores=blocks, maximum=7)
    assert native.triples == tuple(sorted(t for _, t in expected[:7]))
    assert native.summary["screen_ineligible"][mode] == 27 - len(expected)
    assert native.summary["same_data_screen"] and native.summary["screened_triples"] == 27


def test_hybrid_quota_survives_adversarial_pair_scores_and_prior(tmp_path):
    maps = mapped()
    axes = (tuple(range(3)),) * 3
    msa = observed()
    blocks = {p: np.full((3, 3), np.nan) for p in ("AB", "AC", "BC")}
    forced = select_candidates(maps, axes, maximum=2, seed=11).triples
    outside = next(t for t in product(*axes) if t not in forced)
    f = tsv(
        tmp_path / "prior.tsv",
        [f"{r}_original_col" for r in "abc"],
        [[m.original_columns[j] + 1 for m, j in zip(maps, t)] for t in (forced[0], outside)],
    )
    selected = select_candidates(
        maps,
        axes,
        mode="hybrid",
        maximum=6,
        seed=11,
        uniform_quota=2,
        msa=msa,
        pair_scores=blocks,
        candidate_file=f,
        prior_source="archived annotation",
        min_entropy_bits=8,
    )
    assert [
        (t, m) for t, (m, v) in zip(selected.triples, selected.reasons) if m == "uniform_reserved"
    ] == [(t, "uniform_reserved") for t in forced]
    assert set(selected.triples) == set(forced) | {outside}
    assert (
        selected.summary["duplicate_proposals"] >= 1
        and selected.summary["screen_ineligible"]["wedge"] > 0
    )
    assert selected.summary["uniform_reserved"] == 2
    with pytest.raises(ValueError, match=">= 1"):
        select_candidates(maps, axes, mode="hybrid", maximum=2, uniform_quota=0)
    with pytest.raises(ValueError, match="exceeds"):
        select_candidates(maps, axes, mode="hybrid", maximum=2, uniform_quota=3)


def test_scope_and_training_evidence_immune_to_evaluation_rows():
    maps = mapped()
    axes = (tuple(range(3)),) * 3
    msa = observed()
    ids = tuple(x.observation_id for x in msa.matches[:10])
    others = observed(seed=779)
    edited = MatchedTripletMSA(
        *(
            ComponentMSA(c.name, c.molecule, c.ids, c.sequences[:10] + e.sequences[10:])
            for c, e in zip(msa.components, others.components)
        ),
        msa.matches,
    )
    config = {"mode": "hybrid", "maximum": 6, "training_ids": ids, "uniform_quota": 1}

    def chosen(data):
        w = sequence_weights(data)
        return select_candidates(maps, axes, msa=data, weights=w, **config)

    before, after = chosen(msa), chosen(edited)
    assert (
        before.summary == after.summary
        and before.triples == after.triples
        and before.reasons == after.reasons
    )
    assert (
        before.summary["scope"] == "training_rows_only" and before.summary["screen_source_sha256"]
    )
    with pytest.raises(ValueError, match="internal fits"):
        select_candidates(maps, axes, msa=msa, pair_scores={}, **config)
    with pytest.raises(ValueError, match="proper subset"):
        select_candidates(maps, axes, msa=msa, mode="entropy", training_ids=[*ids, ids[-1]])
    with pytest.raises(ValueError, match="declared"):
        select_candidates(maps, axes, msa=msa, weights=np.ones(24), **config)


def test_large_universe_limits_streamed_probe_and_arbitrary_integer_sampling():
    n = 1024
    maps = tuple(
        ReferenceCoordinates(c, "ref" + c, tuple(range(n)), tuple(range(1, n + 1)), ("A",) * n)
        for c in "ABC"
    )
    axes = (range(n),) * 3
    arr = np.broadcast_to(np.array(1.0), (n, n))
    selected = select_candidates(
        maps,
        axes,
        mode="triangle",
        maximum=13,
        screen_budget=100,
        seed=8,
        pair_scores=dict.fromkeys(("AB", "AC", "BC"), arr),
    )
    assert selected.summary["eligible_triples"] == 1_073_741_824
    assert len(selected.triples) == 13 and selected.summary["screened_triples"] == 100
    assert selected.summary["outside_screen_pool"] == n**3 - 100
    assert len(set(_draw(2**80, 12, 11))) == 12 and _draw(2**80, 12, 11) == _draw(2**80, 12, 11)
    with pytest.raises(ValueError, match="supported"):
        select_candidates(maps, axes, mode="uniform", maximum=100_001)


def test_benchmark_pairwise_projection_has_an_independent_binary_truth_oracle(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from candidate_discovery import MODEL_NAMES, binary_m2_oracle, law

    from triplet_coevolution.inference.maxent import fit_joint

    for name in MODEL_NAMES:
        target = law(name)
        pair_only = binary_m2_oracle(target)
        fitted = fit_joint(target)
        np.testing.assert_allclose(fitted.p2, pair_only, rtol=0, atol=1e-7)
        for axis in range(3):
            np.testing.assert_allclose(target.sum(axis=axis), pair_only.sum(axis=axis), atol=1e-14)
        actual = target > 0
        oracle_i3 = np.sum(target[actual] * np.log2(target[actual] / pair_only[actual]))
        assert fitted.score.connected_information_bits == pytest.approx(oracle_i3, abs=1e-8)
        if name in ("independent", "pair_chain", "pair_triangle"):
            assert abs(oracle_i3) < 1e-12
        elif name == "pure_xor":
            assert oracle_i3 == pytest.approx(1.0)
        else:
            assert oracle_i3 > 1e-3


def test_entropy_screen_tracks_canonical_support_and_retained_column_order():
    ids = tuple(f"r{i}" for i in range(8))
    seqs = ("AA?", "AA?", "AA?", "AA?", "AC?", "AC?", "AC-", "ACA")
    comps = tuple(ComponentMSA(c, "dna", ids, seqs) for c in "ABC")
    msa = MatchedTripletMSA(*comps, tuple(TripletMatch(i, i, i, i) for i in ids))
    chosen = select_candidates(
        mapped(),
        (range(3),) * 3,
        mode="entropy",
        maximum=20,
        msa=msa,
        min_site_coverage=0.5,
        min_entropy_bits=0.8,
    )
    assert chosen.triples == ((1, 1, 1),)
    assert chosen.reasons == (("entropy", 1.0),)
    assert chosen.summary["screen_ineligible"]["entropy"] == 26
