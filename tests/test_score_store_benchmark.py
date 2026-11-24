"""Truth fixtures for full score universes, explicit structural evidence and tied thresholds."""

import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.mi import MIBlock, TripletMI
from triplet_coevolution.inference.potts import DCABlock
from triplet_coevolution.io.score_store import (
    PAIRS,
    from_results,
    import_legacy_scores,
    load_score_store,
    write_score_store,
)
from triplet_coevolution.validation.benchmark import benchmark_scores, write_merged_scores


def fixture_store():
    maps = tuple(ReferenceCoordinates(c, "ref" + c, (1, 8), (10, 20), ("G", "A")) for c in "ABC")
    raw = np.array([[2.0, 2.0], [1.0, 3.0]])
    corr = raw - 2.0
    n = np.full((2, 2), 4, dtype=int)
    w = np.full((2, 2), 3.0, dtype=float)
    mi = TripletMI(
        *(MIBlock(*p, raw.copy(), corr.copy(), n.copy(), w.copy()) for p in PAIRS), 4, 3.0, False, 2
    )
    dca = {p: DCABlock(raw.copy(), corr.copy(), n.copy(), w.copy()) for p in PAIRS}
    return from_results(
        maps,
        mi,
        dca,
        provenance={
            "software_version": "fixture",
            "inputs": {
                k: {"sha256": "0" * 64, "path": "/private/example"}
                for k in ("A", "B", "C", "matches")
            },
        },
    )


def labels(path, records, extra=()):
    columns = [
        f"{side}_{field}"
        for side in "xy"
        for field in ("component", "reference_id", "reference_pos")
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([*columns, "contact", *extra])
        for pair, x, y, value, *more in records:
            writer.writerow(
                [pair[0], "ref" + pair[0], x, pair[1], "ref" + pair[1], y, value, *more]
            )
    return path


def test_thresholds_ties_unknown_top_and_export(tmp_path):
    store = fixture_store()
    path = labels(
        tmp_path / "contacts.tsv", [("AB", 10, 10, 1), ("AB", 10, 20, 0), ("AB", 20, 10, 1)]
    )
    result = benchmark_scores(store, path, top_k=(1, 4))
    ds = result["statistics"]["mi_bits"]["interfaces"]["AB"]
    assert ds["auroc"] == 0.25
    assert ds["average_precision"] == pytest.approx(7 / 12)
    assert ds["curves"] == {
        "thresholds": [2, 1],
        "true_positives": [1, 2],
        "false_positives": [1, 1],
        "precision": [0.5, 2 / 3],
        "recall": [0.5, 1.0],
        "false_positive_rate": [1.0, 1.0],
        "initial": {"true_positives": 0, "false_positives": 0, "precision": None},
    }
    assert ds["top_k"]["1"]["unknown"] == 1 and ds["top_k"]["1"]["precision"] is None
    assert ds["top_k"]["4"]["verified_positive_lower_bound"] == 0.5
    assert ds["top_k"]["4"]["enrichment"] is None
    assert ds["coverage"]["eligible_prevalence"] is None
    assert ds["uncertainty"]["status"] == "unavailable"
    write_merged_scores(tmp_path / "scores.tsv", store, path)
    with (tmp_path / "scores.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 12
    assert [r["mi_bits_competition_rank"] for r in rows[:4]] == ["2", "2", "4", "1"]
    assert rows[3]["contact"] == "" and rows[3]["label_status"] == "unlabeled"
    assert rows[2]["mi_apc"] == "-1.0"
    assert all(r["score_uncertainty"] == "unavailable" for r in rows)


def test_finite_common_set_training_holdout_and_cluster_bootstrap(tmp_path):
    store = fixture_store()
    store.scores["dca_apc"]["AB"][0, 0] = np.nan
    store.n_valid["BC"][1, 1] = 1
    rows = [
        ("AB", 10, 10, 1, "first", "test", "resolved", 3.0),
        ("AB", 10, 20, 0, "first", "test", "resolved", 14.0),
        ("AB", 20, 10, 1, "second", "test", "resolved", 6.0),
        ("AC", 20, 10, 1, "second", "train", "resolved", 2.0),
        ("AC", 10, 10, "", "", "test", "unresolved", ""),
        ("BC", 20, 20, 0, "third", "test", "resolved", 22.0),
        ("BC", 50, 90, 1, "third", "test", "resolved", 1.0),
    ]
    path = labels(
        tmp_path / "labels.tsv", rows, ("cluster_id", "split", "status", "distance_angstrom")
    )
    res = benchmark_scores(store, path, bootstrap=40, seed=53)
    assert res == benchmark_scores(store, path, bootstrap=40, seed=53)
    assert {x["reason"] for x in res["excluded_labels"]} == {
        "missing_score_in_comparison",
        "non_test_split",
        "unresolved",
        "insufficient_pair_support",
        "outside_retained_reference_axes",
    }
    for method in res["statistics"].values():
        assert method["pooled"]["scored"] == 2
        assert method["pooled"]["uncertainty"]["units"] == 2
        assert method["pooled"]["uncertainty"]["auroc"]["valid_draws"] < 40
    a = res["statistics"]["mi_bits"]["interfaces"]["AB"]
    assert a["coverage"]["labeled_finite"] == 3 and a["coverage"]["common_labeled"] == 2
    assert a["distance"]["n"] == 3


@pytest.mark.parametrize("truth,ap", [(1, 1.0), (0, None)])
def test_degenerate_binary_truth_is_explicit(tmp_path, truth, ap):
    store = fixture_store()
    path = labels(tmp_path / "c.tsv", [("AB", 10, 10, truth)])
    r = benchmark_scores(store, path)["statistics"]["mi_bits"]
    assert (
        r["interfaces"]["AB"]["auroc"] is None and r["interfaces"]["AB"]["average_precision"] == ap
    )
    assert (
        r["interfaces"]["AC"]["scored"] == 0 and r["interfaces"]["AC"]["average_precision"] is None
    )


def test_round_trip_benchmark_integrity_and_coordinate_visibility(tmp_path):
    store = fixture_store()
    store.scores["mi_apc"]["AC"][:] = np.nan
    m = store.maps[0]
    store = replace(
        store,
        maps=(
            ReferenceCoordinates("A", m.reference_id, m.original_columns, (None, 20), ("-", "A")),
            *store.maps[1:],
        ),
    )
    path = labels(tmp_path / "c.tsv", [("AB", 10, 10, 1), ("AB", 20, 20, 0)])
    mf = write_score_store(tmp_path / "fit", store)
    assert "/private/example" not in mf.read_text()
    restored = load_score_store(mf)
    assert store.maps == restored.maps
    assert benchmark_scores(store, path) == benchmark_scores(restored, path)
    assert restored.to_results()[0].ac.mi_apc.shape == (2, 2)
    assert (
        restored.eligible("AB").sum() == 4
        and restored.eligible("AB", reference_only=True).sum() == 2
    )
    with pytest.raises(FileExistsError):
        write_score_store(mf.parent, store)
    archive = mf.parent / "pair_scores.npz"
    archive.write_bytes(archive.read_bytes()[:-4])
    with pytest.raises(ValueError, match="payload checksum"):
        load_score_store(mf)
    data = json.loads(mf.read_text())
    data["provenance"]["software_version"] = "forged"
    mf.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest checksum"):
        load_score_store(mf)


def test_labels_cannot_substitute_references_or_duplicate_coordinates(tmp_path):
    store = fixture_store()
    p = labels(tmp_path / "a.tsv", [("AB", 10, 10, 1), ("BA", 10, 10, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        benchmark_scores(store, p)
    p.write_text(p.read_text().replace("refB", "foreign"))
    with pytest.raises(ValueError, match="reference"):
        benchmark_scores(store, p)


def test_legacy_csv_rectangle_rejects_truncation_and_retains_source_columns(tmp_path):
    store = fixture_store()
    axes = {
        "A": [(1, 2, 10), (3, 9, 20)],
        "B": [(1, 2, 10), (2, 9, 20)],
        "C": [(1, 2, 10), (2, 9, 20)],
    }
    tables = {}
    for pair in PAIRS:
        a, b = pair
        path = tmp_path / f"mi_{pair.lower()}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            header = [
                "pair",
                *(
                    f"{side}_{key}"
                    for side in "xy"
                    for key in (
                        "component",
                        "reference_id",
                        "col",
                        "original_col",
                        "reference_pos",
                        "symbol",
                    )
                ),
                "mi_bits",
                "mi_apc",
                "n_valid",
                "valid_weight",
            ]
            writer.writerow(header)
            for (i, o, x), (j, u, y) in ((l, r) for l in axes[a] for r in axes[b]):
                writer.writerow(
                    [
                        pair,
                        a,
                        "ref" + a,
                        i,
                        o,
                        x,
                        "G",
                        b,
                        "ref" + b,
                        j,
                        u,
                        y,
                        "A" if b == "C" else "G",
                        0.25,
                        -0.5,
                        4,
                        3.0,
                    ]
                )
        tables[pair] = {"path": path.name, "rows": 4}
    components = {
        c: {
            "retained_columns": [2, 7, 9] if c == "A" else [2, 9],
            "retained_sites": 3 if c == "A" else 2,
            "reference_id": "ref" + c,
            "reference_start": 1,
        }
        for c in "ABC"
    }
    summary = {
        "schema_version": 1,
        "software_version": "1.0.0",
        "components": components,
        "outputs": {"tables": tables},
        "matched_observations": 4,
        "effective_depth": 3.0,
        "inputs": store.provenance["inputs"],
        "settings": {"max_pairs": 10, "min_valid": 2, "require_reference": True},
    }
    p = tmp_path / "run_summary.json"
    p.write_text(json.dumps(summary))
    imported = import_legacy_scores(tmp_path)
    assert imported.maps[0].site(1).current_column == 3
    artifact = write_score_store(tmp_path / "converted", imported)
    assert load_score_store(artifact).maps[0].site(1) == imported.maps[0].site(1)
    summary["settings"]["max_pairs"] = 4
    p.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="truncated or unprovably complete"):
        import_legacy_scores(tmp_path)
    summary["settings"]["max_pairs"] = 10
    summary["outputs"]["tables"]["AC"]["rows"] = 3
    p.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="truncated"):
        import_legacy_scores(tmp_path)


def test_compact_oracle_and_full_legacy_refit(tmp_path):
    root = os.environ.get("TRIPLET_GATABC_COMPACT_ROOT")
    source = os.environ.get("TRIPLET_GATABC_REFIT_SCORE_DIR")
    if not root or not source:
        pytest.skip(
            "Set TRIPLET_GATABC_COMPACT_ROOT and TRIPLET_GATABC_REFIT_SCORE_DIR to check the private historical refit"
        )
    root = Path(root)
    scores = import_legacy_scores(Path(source))
    assert [m.n_sites for m in scores.maps] == [477, 474, 94]
    artifact = write_score_store(tmp_path / "artifact", scores)
    result = benchmark_scores(load_score_store(artifact), root / "metadata/contacts_2G5H.tsv")
    expected = json.loads((root / "regression/metrics.json").read_text())["pairwise"]
    for field, res in result["statistics"].items():
        for p, scope in {**res["interfaces"], "ALL": res["pooled"]}.items():
            for metric in ("auroc", "average_precision"):
                if field == "mi_bits" and metric == "average_precision":
                    # This frozen Windows refit splits one six-positive AB tie by a binary64 ULP.
                    if p == "ALL":
                        assert scope[metric] == pytest.approx(0.5630831846447621, abs=1e-14, rel=0)
                    if p == "AB":
                        assert scope[metric] == pytest.approx(0.4261777935, abs=5e-11, rel=0)
                    if p in {"ALL", "AB"}:
                        continue
                assert scope[metric] == pytest.approx(expected[field][p][metric], abs=1e-14, rel=0)
    assert not result["excluded_labels"]
    assert result["statistics"]["dca_apc"]["interfaces"]["AC"]["top_k"]["10"]["positive"] == 9
    # The compact archive preserves every labeled pair's original binary64 score. Use only
    # these values for the independent original-label comparison, never for full-axes top K.
    original = {
        s: {p: np.full(v.shape, np.nan) for p, v in blocks.items()}
        for s, blocks in scores.scores.items()
    }
    indices = {
        m.component: {v: i for i, v in enumerate(m.reference_positions)} for m in scores.maps
    }
    with (root / "regression/labeled_pairwise_scores.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pair = row["pair"]
            i, j = (
                indices[c][int(row[k + "_reference_pos"])] for k, c in zip("xy", pair, strict=True)
            )
            family = (
                ("mi_bits", "mi_apc")
                if row["method_family"] == "mi"
                else ("dca_frobenius", "dca_apc")
            )
            for f, col in zip(family, ("raw_score", "apc_score"), strict=True):
                original[f][pair][i, j] = float(row[col])
    oracle = benchmark_scores(replace(scores, scores=original), root / "metadata/contacts_2G5H.tsv")
    for field, stats in oracle["statistics"].items():
        for scope, metrics in {**stats["interfaces"], "ALL": stats["pooled"]}.items():
            for metric in ("auroc", "average_precision"):
                assert metrics[metric] == pytest.approx(
                    expected[field][scope][metric], abs=1e-14, rel=0
                )
