"""Exact numerical regressions for blocked and rematched scale paths."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.core.coordinates import build_reference_map
from triplet_coevolution.inference import weights as weight_module
from triplet_coevolution.inference.advanced import write_advanced_results
from triplet_coevolution.inference.mi import _site_mi, mutual_information_block
from triplet_coevolution.inference.weights import sequence_weights


def mixed_msa(n, lengths, seed=17):
    rng = np.random.default_rng(seed)
    groups = np.arange(n) % 7
    comps = []
    for name, molecule, length in zip("ABC", ("protein", "dna", "rna"), lengths, strict=True):
        states = np.array(list(CANONICAL[molecule]))
        matrix = rng.choice(states, (7, length))[groups]
        change = rng.random(matrix.shape)
        matrix[change < 0.13] = rng.choice(states, (change < 0.13).sum())
        matrix[:, 1:][change[:, 1:] > 0.91] = "-"
        matrix[:, 1:][change[:, 1:] > 0.96] = "?"
        comps.append(
            ComponentMSA(
                name,
                molecule,
                tuple(f"{name}{i}" for i in range(n)),
                tuple("".join(x) for x in matrix),
            )
        )
    matches = tuple(TripletMatch(str(i), *(c.ids[i] for c in comps)) for i in range(n))
    return MatchedTripletMSA(*comps, matches)


def reference_counts(components, threshold, min_overlap):
    """Unpacked row-by-row reference with exact common denominators."""
    n = components[0].n_sequences
    counts = np.empty(n, dtype=int)
    for i in range(n):
        numerator = np.zeros(n, dtype=np.int64)
        denominator = np.ones(n, dtype=np.int64)
        eligible = np.ones(n, dtype=bool)
        for comp in components:
            valid = (comp.matrix >= 0) & (comp.matrix < len(CANONICAL[comp.molecule]))
            shared = valid & valid[i]
            overlap = shared.sum(axis=1)
            matched = (shared & (comp.matrix == comp.matrix[i])).sum(axis=1)
            numerator = numerator * overlap + matched * denominator
            denominator *= overlap
            eligible &= (overlap > 0) & (overlap / comp.n_sites >= min_overlap)
        eligible[eligible] = (
            numerator[eligible] / (denominator[eligible] * len(components)) >= threshold
        )
        eligible[i] = True
        counts[i] = eligible.sum()
    return counts


@pytest.mark.parametrize("fallback", [False, True])
def test_packed_weights_across_word_and_tile_boundaries(monkeypatch, fallback):
    if fallback:
        monkeypatch.delattr(np, "bitwise_count", raising=False)
    msa = mixed_msa(267, (71, 13, 8))
    for data, components in ((msa, msa.components), (msa.a, (msa.a,))):
        result = sequence_weights(data, identity_threshold=0.8, min_overlap=0.65)
        counts = reference_counts(components, 0.8, 0.65)
        np.testing.assert_array_equal(result.neighbor_counts, counts)
        np.testing.assert_array_equal(result.weights, 1.0 / counts)
        assert result.n_eff == (1.0 / counts).sum()


@pytest.mark.parametrize("include_gaps", [False, True])
def test_batched_mi_matches_scalar_across_tiles(include_gaps):
    msa = mixed_msa(503, (3, 138, 2))
    w = np.random.default_rng(11).random(msa.n_observations)
    w[::4] = 0
    result = mutual_information_block(
        msa.a, msa.b, weights=w, include_gaps=include_gaps, min_valid=340
    )
    expect = np.array(
        [
            _site_mi(x, y, 20 + include_gaps, 4 + include_gaps, w, 340)
            for x in msa.a.matrix.T
            for y in msa.b.matrix.T
        ]
    ).reshape(3, 138, 3)
    np.testing.assert_array_equal(result.n_valid, expect[:, :, 1])
    np.testing.assert_allclose(result.valid_weight, expect[:, :, 2], atol=1e-12)
    np.testing.assert_allclose(result.mi_bits, expect[:, :, 0], atol=1e-12, equal_nan=True)


def test_permutation_cache_and_bounded_fallback_match_independent_reweights(monkeypatch, tmp_path):
    msa = mixed_msa(58, (7, 5, 3))
    maps = tuple(build_reference_map(c, c.ids[0]) for c in msa.components)
    w = sequence_weights(msa)
    opts = {
        "permutations": 9,
        "seed": 70,
        "strata": [str(i % 3) for i in range(58)],
        "threebody": True,
        "max_triplets": 6,
    }
    from triplet_coevolution.inference import advanced

    original = advanced._WeightComparisons

    class Reference:
        def __init__(self, components, **kwargs):
            pass

        def weights(self, threshold, overlap, components):
            return 1.0 / reference_counts(components, threshold, overlap)

    monkeypatch.setattr(advanced, "_WeightComparisons", Reference)
    write_advanced_results(tmp_path / "ref", msa, maps, w, **opts)
    monkeypatch.setattr(advanced, "_WeightComparisons", original)
    for label, cap in (("cached", 1000000), ("blocked", 0)):
        monkeypatch.setattr(weight_module, "_CACHE_BYTES", cap)
        write_advanced_results(tmp_path / label, msa, maps, w, **opts)
        for method in ("mi", "threebody"):
            filename = f"permutation_{method}.tsv"
            assert (tmp_path / label / filename).read_bytes() == (
                tmp_path / "ref" / filename
            ).read_bytes()


def test_prepared_sweep_keeps_row_groups_and_reference_coordinates(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from _matched_inputs import prepared_subset

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    args = [
        "analyze",
        "--matches",
        "pairs.tsv",
        "--strata-column",
        "clade",
        "--b-ref-row",
        "B0",
        "--b-start",
        "100",
        "--b-strand",
        "-",
        "--min-coverage",
        "0.5",
    ]
    msa = mixed_msa(15, (3, 4, 5))
    for c in msa.components:
        key = c.name.lower()
        (prepared / f"{key}.fa").write_text(
            "".join(f">{i}\n{s}\n" for i, s in zip(c.ids, c.sequences, strict=True))
        )
        args.extend(
            [
                f"--{key}",
                f"{key}.fa",
                f"--{key}-ref",
                "display" if key == "b" else c.ids[0],
                f"--{key}-type",
                c.molecule.value,
            ]
        )
    with (prepared / "pairs.tsv").open("w", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["observation_id", "a_id", "b_id", "c_id", "clade"])
        for i, m in enumerate(msa.matches):
            writer.writerow([m.observation_id, m.a_id, m.b_id, m.c_id, f"clade{i % 2}"])
    (prepared / "analyze_args.json").write_text(json.dumps(args))
    previous = set()
    for n in (7, 11):
        subset, maps, strata, settings = prepared_subset(tmp_path, n, 11, seed=91)
        observed = [int(m.observation_id) for m in subset.matches]
        assert previous <= set(observed)
        assert strata == [f"clade{i % 2}" for i in observed]
        assert settings["strata_column"] == "clade"
        assert maps[1].reference_id == "display" and maps[1].reference_positions[0] == 100
        previous = set(observed)
    with pytest.raises(ValueError, match="requested 16 matched rows.*contains 15"):
        prepared_subset(prepared, 7, 16, seed=91)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "benchmarks/matched_scaling.py"),
        "--prepared",
        str(tmp_path),
        "--sizes",
        "7",
        "11",
        "--seed",
        "91",
        "--repeats",
        "1",
        "--permutations",
        "2",
    ]
    report = json.loads(subprocess.check_output(command, text=True))
    assert [(r["n"], r["job"]) for r in report["results"]] == [
        (7, "weights"),
        (7, "permutation"),
        (11, "weights"),
        (11, "permutation"),
    ]
    assert not list(tmp_path.rglob("permutation_*.tsv"))
