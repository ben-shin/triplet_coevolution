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


@pytest.fixture()
def legacy_rectangle(tmp_path):
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
                "rank",
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
            for rank, ((i, o, x), (j, u, y)) in enumerate(
                ((l, r) for l in axes[a] for r in axes[b]), start=1
            ):
                writer.writerow(
                    [
                        rank,
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
        "settings": {"max_pairs": 10, "min_valid": 2, "require_reference": True, "score": "mi_apc"},
    }
    p = tmp_path / "run_summary.json"
    p.write_text(json.dumps(summary))
    return summary


def test_legacy_csv_rectangle_rejects_truncation_and_retains_source_columns(
    tmp_path, legacy_rectangle
):
    p = tmp_path / "run_summary.json"
    summary = legacy_rectangle
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


def test_sparse_uncapped_legacy_support_and_ranked_views(tmp_path, legacy_rectangle):
    summary = legacy_rectangle
    data = tmp_path / "mi_ab.csv"
    data.write_text("\n".join(data.read_text().splitlines()[:-1]) + "\n")
    summary["outputs"]["tables"]["AB"]["rows"] = 3
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    store = import_legacy_scores(tmp_path)
    restored = load_score_store(write_score_store(tmp_path / "artifact", store))
    assert restored.n_valid["AB"][1, 1] == -1
    assert np.isnan(restored.valid_weight["AB"][1, 1]) and np.isnan(
        restored.scores["mi_bits"]["AB"][1, 1]
    )
    from triplet_coevolution.score_views import write_ranked_views

    destination = tmp_path / "export"
    destination.mkdir()
    tables = write_ranked_views(destination, restored)
    assert [t.rows for t in tables] == [3, 4, 4]
    path = labels(tmp_path / "contact.tsv", [("AB", 20, 20, 1), ("AB", 10, 20, 0)])
    result = benchmark_scores(restored, path)
    coverage = result["statistics"]["mi_bits"]["interfaces"]["AB"]["coverage"]
    assert (
        coverage["eligible_pairs"],
        coverage["mapped_reference_pairs"],
        coverage["unreported_support"],
    ) == (3, 4, 1)
    assert not coverage["support_and_axes_complete"] and coverage["eligible_prevalence"] is None
    assert result["excluded_labels"][0]["reason"] == "unreported_legacy_support"
    write_merged_scores(tmp_path / "merged.tsv", restored, path)
    with (tmp_path / "merged.tsv").open() as stream:
        missing = next(
            r
            for r in csv.DictReader(stream, delimiter="\t")
            if r["support_status"] == "unreported_legacy_support"
        )
    assert (
        missing["n_valid"]
        == missing["valid_weight"]
        == missing["eligible"]
        == missing["mi_bits"]
        == ""
    )
    assert missing["x_current_col"] == "3" and missing["x_original_col"] == "9"


def test_unreported_legacy_axis_keeps_original_source_metadata(tmp_path, legacy_rectangle):
    for p in ("AB", "AC"):
        table = tmp_path / f"mi_{p.lower()}.csv"
        table.write_text(table.read_text().splitlines()[0] + "\n")
        legacy_rectangle["outputs"]["tables"][p]["rows"] = 0
    (tmp_path / "run_summary.json").write_text(json.dumps(legacy_rectangle))
    store = load_score_store(write_score_store(tmp_path / "store", import_legacy_scores(tmp_path)))
    assert store.maps[0].reference_id == "refA" and store.maps[0].reference_positions == ()
    assert store.provenance["source_retained_columns"]["A"] == [2, 7, 9]
    assert store.scores["mi_bits"]["AB"].shape == (0, 2)
    r = benchmark_scores(store, labels(tmp_path / "labels.tsv", [("AB", 20, 10, 1)]))
    assert r["excluded_labels"] == [
        {"pair": "AB", "positions": [20, 10], "reason": "outside_retained_reference_axes"}
    ]


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_header",
        "missing_header",
        "duplicate_coordinate",
        "nonfinite_ranked",
        "bad_rank",
        "drop_row",
        "bad_declared_digest",
    ],
)
def test_legacy_rejects_unreliable_view(tmp_path, legacy_rectangle, corruption):
    table = tmp_path / "mi_ab.csv"
    with table.open(newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    if corruption == "duplicate_header":
        header.append("mi_bits")
        for row in rows[1:]:
            row.append("999")
    elif corruption == "missing_header":
        header[header.index("mi_bits")] = "alternate"
    elif corruption == "duplicate_coordinate":
        rows[-1] = rows[-2].copy()
        rows[-1][0] = "4"
    elif corruption == "nonfinite_ranked":
        rows[1][header.index("mi_apc")] = "nan"
    elif corruption == "bad_rank":
        rows[1][0] = "2"
    elif corruption == "drop_row":
        rows.pop()
    else:
        legacy_rectangle["outputs"]["tables"]["AB"]["sha256"] = "0" * 64
        (tmp_path / "run_summary.json").write_text(json.dumps(legacy_rectangle))
    with table.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError):
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
