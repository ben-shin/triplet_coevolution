"""Exercise the score-table replay on hand-calculated synthetic data."""

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks.validate_gatabc import MANIFEST, validate_legacy_tables


def _write(path, header, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(records)


@pytest.fixture
def sample(tmp_path):
    refs = {c: f"synthetic-{c}" for c in "ABC"}
    id_fields = [
        f"{side}_{field}"
        for side in "xy"
        for field in ("component", "reference_id", "reference_pos")
    ]
    labels = tmp_path / "labels.tsv"
    coordinates = [(1, 1), (1, 2), (2, 1), (2, 2)]
    pairs = ("AB", "AC", "BC")
    _write(
        labels,
        [*id_fields, "contact"],
        [(a, refs[a], 1, b, refs[b], y, int(y == 1)) for a, b in pairs for y in (1, 2)],
    )
    values = {
        "mi_bits": [1, 1, 2, 0],
        "mi_apc": [3, 1, 3, 0],
        "dca_frobenius": [0, 1, 2, 2],
        "dca_apc": [4, 4, 4, 4],
    }
    for a, b in pairs:
        for family, fields in (
            ("mi", ["mi_bits", "mi_apc"]),
            ("dca", ["dca_frobenius", "dca_apc"]),
        ):
            records = []
            for i, (x, y) in enumerate(coordinates):
                records.append(
                    [
                        a,
                        refs[a],
                        x,
                        b,
                        refs[b],
                        y,
                        "GA"[x - 1],
                        "GA"[y - 1],
                        a + b,
                        *(values[f][i] for f in fields),
                    ]
                )
            _write(
                tmp_path / f"{family}_{a.lower()}{b.lower()}.tsv",
                [*id_fields, "x_symbol", "y_symbol", "pair", *fields],
                records,
            )
    metrics = {}
    for field, auc, ap in (
        ("mi_bits", 0.5, 0.5),
        ("mi_apc", 1, 1),
        ("dca_frobenius", 0, 0.5),
        ("dca_apc", 0.5, 0.5),
    ):
        metrics[field] = {scope: [auc, ap] for scope in (*pairs, "pooled")}
        metrics[field]["macro_auroc"] = auc
    return tmp_path, {
        "paths": {"labels": "labels.tsv", "full_score_directory": "."},
        "supplied_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tmp_path.glob("*.tsv")
        },
        "references": refs,
        "cohort": {
            "reference_eligible_pairs": dict.fromkeys(pairs, 4),
            "reference_eligible_columns": dict.fromkeys("ABC", 2),
        },
        "labels": {"counts": {p: {"positive": 1, "negative": 1} for p in pairs}},
        "metrics": metrics,
        "metric_definitions": {"absolute_tolerances": {"interface": 1e-9, "pooled_or_macro": 1e-9}},
        "reference_pairs": [
            {
                "pair": "AB",
                "positions": [1, 1],
                "symbols": ["G", "G"],
                "label": "positive",
                "ranks": {"mi_bits": 2, "mi_apc": 1, "dca_frobenius": 4},
            },
            {
                "pair": "AB",
                "positions": [2, 1],
                "symbols": ["A", "G"],
                "label": "unlabeled",
                "ranks": {"mi_apc": 1, "dca_apc": 1},
            },
        ],
    }


def test_replay_synthetic_ties_and_unknowns(sample):
    root, manifest = sample
    result = validate_legacy_tables(root, manifest)
    assert result["status"] == "passed"
    assert result["score_tables"]["mi_ab.tsv"]["checked_ranks"]["AB:1-1"] == {
        "mi_bits": 2,
        "mi_apc": 1,
    }
    assert len(result["score_tables"]) == 6
    assert result["metrics"]["mi_bits"]["pooled"]["scored"] == 6
    assert result["metrics"]["dca_frobenius"]["pooled"]["average_precision"] == 0.5


@pytest.mark.parametrize(
    "case", ["hash", "rank", "metric", "labels", "truncated", "duplicate", "reference"]
)
def test_replay_detects_disagreement(sample, case):
    root, manifest = sample
    if case == "hash":
        (root / "labels.tsv").write_bytes((root / "labels.tsv").read_bytes() + b"\n")
        error = "label SHA-256"
    elif case == "rank":
        manifest["reference_pairs"][0]["ranks"]["mi_bits"] = 3
        error = "competition rank"
    elif case == "metric":
        manifest["metrics"]["mi_bits"]["pooled"][1] = 0.75
        error = "pooled average_precision"
    elif case == "labels":
        manifest["reference_pairs"][1]["label"] = "negative"
        error = "reference label"
    else:
        path = root / "mi_ac.tsv"
        lines = path.read_text().splitlines()
        if case == "truncated":
            lines.pop()
            error = "eligible pairs"
        elif case == "duplicate":
            lines[-1] = lines[-2]
            error = "duplicate reference pair"
        else:
            lines[-1] = lines[-1].replace("synthetic-C", "other-reference")
            error = "reference"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        validate_legacy_tables(root, manifest)


def test_documented_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["supplied_sha256"]["metadata/contacts_2G5H.tsv"] == (
        "01041f913c8aa13e7be79b02b7f97d84d52d9ec10d1841fa0b16e1b2d688137b"
    )
    assert sum(manifest["cohort"]["reference_eligible_pairs"].values()) == 315492
    assert manifest["metrics"]["dca_apc"]["pooled"] == [0.6343996, 0.6553872]


def test_regenerated_scores_keep_byte_and_numerical_checks_distinct(sample):
    root, manifest = sample
    manifest["supplied_sha256"]["mi_ab.tsv"] = "0" * 64
    with pytest.raises(ValueError, match="archived SHA-256"):
        validate_legacy_tables(root, manifest)
    result = validate_legacy_tables(root, manifest, recomputed_scores=root)
    assert result["score_source"] == "recomputed"
    assert result["score_tables"]["mi_ab.tsv"]["matches_archived_sha256"] is False


def test_compact_scores_cover_the_same_labels(sample):
    root, manifest = sample
    relative = "regression/labeled_pairwise_scores.tsv"
    path = root / relative
    path.parent.mkdir()
    records = []
    for pair in ("AB", "AC", "BC"):
        for family, raw, apc in (("mi", "mi_bits", "mi_apc"), ("dca", "dca_frobenius", "dca_apc")):
            with (root / f"{family}_{pair.lower()}.tsv").open() as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row["x_reference_pos"] == "1":
                        y = int(row["y_reference_pos"])
                        records.append((pair, 1, y, int(y == 1), row[raw], row[apc], family))
    header = (
        "pair",
        "x_reference_pos",
        "y_reference_pos",
        "contact",
        "raw_score",
        "apc_score",
        "method_family",
    )
    _write(path, header, records)
    manifest["supplied_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = validate_legacy_tables(root, manifest, compact=True)
    assert result["metrics"] == validate_legacy_tables(root, manifest)["metrics"]
    assert result["score_tables"][relative]["rows"] == 12
    _write(path, header, records[:-1])
    manifest["supplied_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="finite labeled scores"):
        validate_legacy_tables(root, manifest, compact=True)


def test_optional_original_gatabc_score_tables():
    location = os.environ.get("TRIPLET_GATABC_LEGACY_ROOT")
    if not location:
        pytest.skip("Private GatABC score tables require explicit TRIPLET_GATABC_LEGACY_ROOT")
    result = validate_legacy_tables(
        Path(location), json.loads(MANIFEST.read_text(encoding="utf-8"))
    )
    assert result["status"] == "passed"


def test_optional_compact_gatabc_scores():
    location = os.environ.get("TRIPLET_GATABC_COMPACT_ROOT")
    if not location:
        pytest.skip("Private compact bundle requires explicit TRIPLET_GATABC_COMPACT_ROOT")
    result = validate_legacy_tables(
        Path(location), json.loads(MANIFEST.read_text(encoding="utf-8")), compact=True
    )
    assert result["status"] == "passed"
