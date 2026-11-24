import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def geometry(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("followup_geometry")


def test_deterministic_calipers_ties_exclusion_and_no_reuse(geometry):
    # Two candidates compete for two equally close controls. A third candidate is out of range.
    f = np.full((6, 7), 0.5)
    f[2, 1] = 0.95
    f[5, 0] = 0.7
    ids = [20, 30, 40, 11, 10, 12]
    _, _, matched = geometry.select_matched(f, [9, 8, 7, 1, 2, 3], ids, 3)
    assert [(ids[i], ids[j]) for i, j in matched] == [(20, 10), (30, 11)]
    f[5, 0] = np.inf
    with pytest.raises(ValueError):
        geometry.select_matched(f, [9, 8, 7, 1, 2, 3], ids, 3)


def test_hand_calculated_geometry_with_unknowns(geometry):
    d = np.array([[4, 8, 10], [12, 12, 12], [3, np.nan, 4], [2, 3, 5]], float)
    out = geometry.paired_geometry(d, np.array([[0, 1], [2, 3]]), 8)
    assert out["complete_pairs"] == 1 and out["unresolved_pairs"] == 1
    assert out["candidate_only_wedge_or_triangle"] == 1 and out["paired_rate_difference"] == 1
    assert out["paired_mean_span_difference_angstrom"] == -2
    assert out["groups"]["candidate"] == {
        "zero_one_wedge_triangle": [0, 0, 1, 0],
        "unknown": 1,
        "event_lower": 1.0,
        "event_upper": 1.0,
    }
    assert out["groups"]["control"]["zero_one_wedge_triangle"] == [1, 0, 0, 1]


def test_independent_smoothed_xor_information(geometry):
    counts = np.zeros((2, 2, 2), int)
    for a in range(2):
        for b in range(2):
            counts[a, b, a ^ b] = 10
    opt = geometry.MaxEntOptions(
        prior_mode="total", prior_mass=1, tolerance=1e-8, max_iterations=1000
    )
    p, q = 10.125 / 41, 0.125 / 41
    expected = 4 * (p * np.log2(8 * p) + q * np.log2(8 * q))
    assert geometry.fit_joint(
        counts, n_valid=40, options=opt
    ).score.connected_information_bits == pytest.approx(expected)
    assert geometry.fit_joint(
        np.full((2, 2, 2), 5), n_valid=40, options=opt
    ).score.connected_information_bits == pytest.approx(0, abs=1e-12)


def test_two_stage_cli_table_contract(tmp_path, geometry):
    tab = tmp_path / "c.tsv"
    distance = tmp_path / "d.tsv"
    frozen = tmp_path / "selection.json"
    output = tmp_path / "out.json"
    tab.write_text(
        "id\tconnected_information_bits\t"
        + "\t".join(geometry.FEATURES)
        + "\n"
        + "".join(
            str(i) + "\t" + str(i) + "\t" + ("\t".join(["0.5"] * 7)) + "\n" for i in range(10)
        )
    )
    geometry.match_table(tab, frozen)
    distance.write_text("id\tAB\tAC\tBC\n9\t2\t3\t4\n")
    geometry.evaluate_table(frozen, distance, output)
    import json

    r = json.loads(output.read_text())["matched_geometry"]["8"]
    assert r["groups"]["candidate"]["zero_one_wedge_triangle"] == [0, 0, 0, 1]
    assert r["groups"]["control"]["unknown"] == 1
    assert r["complete_pairs"] == 0 and r["paired_rate_difference"] is None
