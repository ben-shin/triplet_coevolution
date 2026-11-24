"""Deterministic small controls for the full registered recovery study."""

import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks import recovery_matrix as study


def test_fixed_margin_oracles():
    # A standard two-sided Fisher result; and the full three-way, fixed-margin fiber.
    assert study.fixed_margin_p(np.array([[1, 9], [11, 3]])) == pytest.approx(0.0027594561852200836)
    parity = np.indices((2, 2, 2)).sum(axis=0) % 2
    endpoint = 2 * parity
    # Weights for the three admissible tables are 1/16, 1, 1/16.
    assert study.fixed_margin_p(endpoint) == pytest.approx(1 / 9)
    assert study.fixed_margin_p(np.ones((2, 2, 2), int)) == 1
    assert study.fixed_margin_p(np.zeros((2, 2, 2), int)) == 1
    assert study.fixed_margin_p(endpoint[::-1].transpose(2, 0, 1)) == pytest.approx(1 / 9)
    for invalid in (np.ones((2, 3), int), np.ones((2, 2)) / 2, -np.ones((2, 2), int)):
        with pytest.raises(ValueError, match="nonnegative integer"):
            study.fixed_margin_p(invalid)


def test_registered_law_matches_analytic_generation_and_donor_marginal():
    cells, _ = study.grid()
    cell = next(c for c in cells if c.q == 0.08 and c.parity_strength == 0.75)
    a, b, c = np.indices((2, 2, 2))
    p_a = np.where(a, cell.q, 1 - cell.q)
    p_b = cell.pair_strength * (a == b) + (1 - cell.pair_strength) * np.where(b, cell.q, 1 - cell.q)
    p_c = cell.parity_strength * ((a ^ b) == c) + (1 - cell.parity_strength) * np.where(
        c, cell.q, 1 - cell.q
    )
    p = p_a * p_b * p_c
    triple = (1, 2, 5)
    for index, wrong in enumerate((0, 0.35, 1)):
        source = replace(cell, missing=0, wrong=wrong)
        x = study.generate(source, triple, 150_000, np.random.default_rng(883 + index))
        truth = (1 - wrong) * p + wrong * p.sum(axis=2, keepdims=True) * p.sum(
            axis=(0, 1), keepdims=True
        )
        np.testing.assert_allclose(study.table(x, triple) / len(x), truth, atol=0.003)
        # Replacing an unrelated C partner retains the marginal frequencies of both A and B.
        np.testing.assert_allclose(x[:, triple[:2]].mean(axis=0), cell.q, atol=0.003)
    for k in (0, 0.25, 0.75):
        conditional = k * ((a ^ b) == c) + (1 - k) * np.where(c, cell.q, 1 - cell.q)
        logp = np.log(p_a * p_b * conditional)
        contrast = np.sum(logp * (-1) ** (a + b + c))
        assert (abs(contrast) > 1e-8) == (k > 0)


def test_holdout_changes_cannot_select_candidates_or_prior_and_validation_can_choose_prior():
    cells, _ = study.grid()
    cell = cells[35]
    target = (0, 2, 4)
    train, valid, test = [
        study.generate(cell, target, n, study.stream(54, 35, 7, j))
        for j, n in enumerate((192, 96, 192), 1)
    ]
    clean = study.evaluate_cohorts(train, valid, test, target, cell)
    masked = study.evaluate_cohorts(train, valid, np.full_like(test, -1), target, cell)
    assert clean["selected"] == masked["selected"]
    assert clean["tuning_choices"] == masked["tuning_choices"]
    assert clean["p"] and not masked["p"]
    assert masked["gain"] is None
    # Validation selects the member of a fixed grid. Test frequencies never enter this choice.
    codes = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int8)
    even = (codes.sum(axis=1) % 2) == 0
    training = np.repeat(codes, np.where(even, 5, 1), axis=0)
    m_even, _ = study.tune(training, np.repeat(codes[even], 4, axis=0), (0, 1, 2))
    m_evenless, _ = study.tune(training, np.repeat(codes[~even], 4, axis=0), (0, 1, 2))
    assert m_even["M3"][1] == 0.5 and m_evenless["M3"][1] == 8


def test_registered_cells_failure_denominators_and_reorder_replay():
    cells, p = study.grid()
    assert len(cells) == 64 and p["repetitions_per_cell"] == 64
    a, b = (cells[i] for i in (1, 30))
    args = [(c, 3, p["root_seed"]) for c in (a, b)]
    assert [study.run_cell(x) for x in args] == [study.run_cell(x) for x in reversed(args)][::-1]
    failure = study.run_cell((replace(a, missing=1), 3, p["root_seed"]))
    assert failure["rates"]["pair_raw_correct"]["n"] == 3
    assert failure["rates"]["pair_raw_correct"]["successes"] == 0
    assert failure["rates"]["triple_evaluable"]["successes"] == 0
    assert failure["rates"]["triple_raw_any_per_evaluable"]["wilson95"] is None
    assert failure["test_M3_minus_M2_nats"]["mean"] is None
    assert failure["bh_false_discovery_proportion"]["mean"] == 0
    assert failure["screen_and_fit_counts"]["invalid"]["triple_no_candidate"] == 3


def test_published_schedule_and_denominators():
    path = Path(__file__).resolve().parents[1] / "benchmarks/recovery_matrix_results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["registration_sha256_lf"] == study.HASH
    assert result["settings"] == 64 and result["replicates"] == 4096
    for i, row in enumerate(result["cells"]):
        assert row["setting"]["id"] == i and row["replicates"] == 64
        for k, n in (("pair", 4), ("triple", 8)):
            assert sum(row["screen_and_fit_counts"][k + "_screen_candidates"].values()) == n * 64
            assert (
                row["rates"][k + "_raw_any_per_evaluable"]["n"]
                == row["rates"][k + "_evaluable"]["successes"]
            )
        assert row["test_gain_positive_evaluable"]["n"] == row["test_M3_minus_M2_nats"]["n"]
