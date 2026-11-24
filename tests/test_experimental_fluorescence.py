"""Independent finite-cube truth, holdout isolation and optional public data replay."""

import hashlib
import json
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from benchmarks.experimental_fluorescence import (
    PRIMARY,
    REPEATS,
    cube_design,
    evaluate,
    heldout_predictions,
    metrics,
    order_variance,
    read_workbook,
    source_files,
    walsh,
)


def test_exact_cube_allele_orientation_contrasts_and_permutations():
    # Direct factorial contrasts on a known four-bit subject, independent of matrix construction.
    i = np.arange(16)
    a, b, c, d = (1 - 2 * ((i >> j) & 1) for j in (3, 2, 1, 0))
    response = 7 + a + 2 * b * d - 3 * a * b * c
    coeff = walsh(response)
    np.testing.assert_allclose(coeff[[0, 8, 5, 14]], [7, 1, 2, -3])
    assert np.count_nonzero(coeff) == 4
    direct = np.mean(
        [
            sum(y * ((-1) ** k.bit_count()) for k, y in enumerate(response[offset::2]))
            for offset in (0, 1)
        ]
    )
    assert direct == 8 * coeff[14]
    assert walsh(response[i ^ 8])[14] == -coeff[14]
    permuted = response.reshape([2] * 4).transpose(3, 1, 2, 0).ravel()
    np.testing.assert_allclose(
        order_variance(permuted)["variance_fractions_by_order"], [1 / 14, 4 / 14, 9 / 14, 0]
    )
    assert order_variance(np.ones(8))["variance_fractions_by_order"] is None
    assert metrics(np.ones(8), np.ones(8))["r_squared"] is None
    for invalid in ([1, 2, 3], [0, np.nan, 1, 2]):
        with pytest.raises(ValueError, match="complete finite"):
            walsh(invalid)


def test_fixed_hash_assignment_and_held_out_values_cannot_change_predictions():
    x, terms, folds, weights = cube_design(7, 3)
    expected = [int(hashlib.sha256(f"{i:07b}".encode()).hexdigest(), 16) % 5 for i in range(128)]
    np.testing.assert_array_equal(folds, expected)
    assert np.array_equal(weights, [i.bit_count() for i in range(128)])
    assert folds[0] == int(hashlib.sha256(b"0000000").hexdigest(), 16) % 5
    truth = 0.2 + x[:, terms.index((0, 1, 4))] + 0.3 * x[:, terms.index((2, 4))]
    predictions = heldout_predictions(x, truth, folds)
    np.testing.assert_allclose(predictions, truth, atol=1e-13)
    changed = truth.copy()
    changed[folds == 0] += np.linspace(1, 2, (folds == 0).sum())
    np.testing.assert_array_equal(
        heldout_predictions(x, changed, folds)[folds == 0], predictions[folds == 0]
    )
    with pytest.raises(ValueError, match="rank"):
        heldout_predictions(np.ones((128, 2)), truth, folds)
    with pytest.raises(ValueError, match="five"):
        heldout_predictions(x, truth, folds % 3)


def workbook(path, genotypes, *, header="combined"):
    ss = ["binary", "genotype", "brightness", header, *genotypes]
    shared, sheet = ET.Element("sst"), ET.Element("worksheet")
    for s in ss:
        ET.SubElement(ET.SubElement(shared, "si"), "t").text = s
    data = ET.SubElement(sheet, "sheetData")
    for i, (a, b) in enumerate([(0, 2), (1, 3), *[(k + 4, k) for k in range(len(genotypes))]], 1):
        row = ET.SubElement(data, "row", r=str(i))
        for col, value in [("A", a), ("B", b)]:
            cell = ET.SubElement(row, "c", r=f"{col}{i}", t="s" if i <= 2 or col == "A" else "n")
            if not (i == 4 and col == "B"):
                ET.SubElement(cell, "v").text = str(value)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", ET.tostring(shared))
        z.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet))


def test_parser_keys_on_author_bits_and_rejects_missing_duplicate_or_shifted_data(tmp_path):
    specs = {"A": ("binary", "genotype"), "B": ("brightness", "combined")}
    p = tmp_path / "fixture.xlsx"
    ids = ["'11'", "'00'", "'10'", "'01'"]
    workbook(p, ids)
    np.testing.assert_array_equal(read_workbook(p, specs, 2)["B"], [np.nan, 3, 2, 0])
    for bad, msg in [
        (ids[:-1], "incomplete"),
        (ids + ids[:1], "duplicate"),
        (ids[:-1] + ["'1'"], "fixed-width"),
    ]:
        workbook(p, bad)
        with pytest.raises(ValueError, match=msg):
            read_workbook(p, specs, 2)
    workbook(p, ids, header="single channel")
    with pytest.raises(ValueError, match="columns"):
        read_workbook(p, specs, 2)


def test_incomplete_processed_response_keeps_separate_complete_replicate():
    n = 128
    finite = np.arange(n, dtype=float)
    missing = finite.copy()
    missing[5] = np.nan
    result = evaluate(
        {"J": missing, "C": np.ones(n), "D": np.zeros(n), "E": np.zeros(n)},
        {"C": missing.copy(), "E": finite},
        7,
    )
    assert result["status"] == "partial_incomplete_response"
    assert set(result["analysis"]) == {"data4_trial2"}
    assert result["outcome_support"]["primary"]["missing_or_nonfinite"] == 1
    assert result["read_count_diagnostics"]["zero_both_colors"] == n
    assert "trial_comparison" not in result


def test_opt_in_public_workbooks_against_explicit_parity_design_and_normal_equations():
    location = os.environ.get("TRIPLET_FLUORESCENCE_ROOT")
    if not location:
        pytest.skip("Set TRIPLET_FLUORESCENCE_ROOT to replay both public author tables")
    a, b = source_files(Path(location), False)
    primary, trials = read_workbook(a, PRIMARY), read_workbook(b, REPEATS)
    result = evaluate(primary, trials)
    published = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "benchmarks/experimental_fluorescence_results.json"
        ).read_text("utf-8")
    )
    assert result["fold_sha256"] == published["fold_sha256"]
    assert result["outcome_support"] == published["outcome_support"]
    assert result["read_count_diagnostics"] == published["read_count_diagnostics"]
    y = np.column_stack([primary["J"], trials["E"]])
    i = np.arange(8192)
    sign = np.array([(-1) ** j.bit_count() for j in range(8192)])
    masks = sorted((j for j in range(8192) if j.bit_count() <= 3), key=int.bit_count)
    design = sign[i[:, None] & masks].astype(float)
    folds = np.array(
        [int(hashlib.sha256(format(j, "013b").encode()).hexdigest(), 16) % 5 for j in range(8192)]
    )
    for degree, p in enumerate((14, 92, 378), 1):
        predictions = np.empty_like(y)
        for fold in range(5):
            test = folds == fold
            train = design[~test, :p]
            coef = np.linalg.solve(train.T @ train, train.T @ y[~test])
            predictions[test] = design[test, :p] @ coef
        for j, label in enumerate(("primary", "data4_trial2")):
            fit = result["analysis"][label]["held_out_by_degree"][str(degree)]
            oracle = metrics(y[:, j], predictions[:, j])
            reference = published["analysis"][label]["held_out_by_degree"][str(degree)]
            for metric in ("rmse", "r_squared"):
                np.testing.assert_allclose(oracle[metric], fit[metric], rtol=0, atol=1e-12)
                np.testing.assert_allclose(oracle[metric], reference[metric], rtol=0, atol=1e-12)
    direct = np.sum(((design[:, 92:].T @ y) / 8192) ** 2, axis=0)
    for j, label in enumerate(("primary", "data4_trial2")):
        np.testing.assert_allclose(
            direct[j],
            result["analysis"][label]["walsh"]["squared_coefficients_by_order"][2],
            rtol=1e-13,
        )
