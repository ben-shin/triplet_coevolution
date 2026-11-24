"""Numerical, portability and presentation checks for the offline archive explorer."""

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from html.parser import HTMLParser
from urllib.parse import unquote

import numpy as np
import pytest

from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.io.score_store import ScoreStore
from triplet_coevolution.reporting.explore import write_explorer
from triplet_coevolution.reporting.explore_data import (
    PAIRS,
    artifact,
    benchmark_preview,
    common_ranks,
    records,
    score_grid,
    write_label_scores,
)
from triplet_coevolution.reporting.explore_evidence import (
    benchmark_view,
    structure_view,
    support_view,
)
from triplet_coevolution.reporting.explore_figures import colors, export_maps, grid_svg
from triplet_coevolution.reporting.network import RankedTable


def store(shape=3):
    n = shape
    axes = tuple(
        ReferenceCoordinates(
            c,
            "ref" + c,
            tuple(2 * j for j in range(n)),
            tuple(range(n, 0, -1)),
            tuple("A" * n),
            start=n,
            strand="-",
        )
        for c in "ABC"
    )
    raw = np.arange(n * n, dtype=float).reshape(n, n)
    values = {
        "mi_bits": raw,
        "mi_apc": raw - 3,
        "dca_frobenius": raw[::-1],
        "dca_apc": raw[::-1] - 3,
    }
    return ScoreStore(
        axes,
        {k: dict.fromkeys(PAIRS, v) for k, v in values.items()},
        dict.fromkeys(PAIRS, np.full((n, n), 4)),
        dict.fromkeys(PAIRS, np.full((n, n), 3.2)),
        {"min_valid_weight": 2},
        2,
        False,
    )


def test_rank_coordinates_are_exact_with_ties_support_and_missing_reference():
    s = store()
    x = replace(s.maps[0], reference_positions=(3, None, 1), symbols=("A", "-", "A"))
    n = np.full((3, 3), 4)
    n[2, 1] = 1
    s = replace(
        s,
        require_reference=True,
        maps=(x, *s.maps[1:]),
        dca_counts=dict.fromkeys(PAIRS, n),
        dca_weights=s.valid_weight,
    )
    mi = s.scores["mi_apc"]["AB"]
    mi[0, 0] = np.nan
    mi[0, 1] = mi[0, 2]
    assert score_grid(s, "AB", "mi_apc")["finite"] == 5
    assert score_grid(s, "AB", "dca_apc")["finite"] == 5
    result = common_ranks(s, "AB", "mi_apc", "dca_apc")
    coords = [(0, 1), (0, 2), (2, 0), (2, 2)]
    dca = s.scores["dca_apc"]["AB"]
    expected = [
        (
            2 * i + 1,
            2 * j + 1,
            1 + sum(mi[k] > mi[i, j] for k in coords),
            1 + sum(dca[k] > dca[i, j] for k in coords),
        )
        for i, j in coords
    ]
    assert result == {"common": 4, "points": expected}
    mi[:] = np.nan
    assert common_ranks(s, "AB", "mi_apc", "dca_apc") == {"common": 0, "points": []}


def test_binned_means_preserve_finite_zeros_and_missing(tmp_path):
    s = store(61)
    data = s.scores["mi_apc"]["AB"]
    data[:] = np.nan
    data[0, 0], data[1, 1], data[0, -1] = 0, 0, -2
    g = score_grid(s, "AB", "mi_apc")
    assert g["values"].shape == (28, 28)
    assert g["counts"][0, 0] == 2 and g["values"][0, 0] == 0
    assert np.isnan(g["values"][0, 1]) and g["values"][0, -1] == -2
    assert g["x"][0] == "1–5" and g["y"][-1] == "119–121"
    assert g["max_abs"] == 2
    svg = grid_svg(
        g,
        title="A < B",
        units="bits",
        maximum=2,
        signed=True,
        palette="accessible",
        identifier="ab",
    )
    ET.fromstring(svg)
    assert "A &lt; B" in svg and colors("accessible", True)[8] in svg and "missing-ab" in svg
    pdf_png = pytest.importorskip("matplotlib")
    assert pdf_png
    result = export_maps(
        tmp_path,
        "mi_apc",
        {"AB": g},
        maximum=2,
        units="bits",
        signed=True,
        palette="amber-blue",
        formats=["svg", "png", "pdf"],
    )
    assert len(result) == 3 and all((tmp_path / p).stat().st_size > 512 for p in result)
    assert (tmp_path / result[2]).read_bytes().startswith(b"%PDF")
    assert (tmp_path / result[1]).read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "p",
    ["../private.tsv", "/private", "D:private.tsv", "https://a/path", "a\\b", "./file", "a/../x"],
)
def test_only_run_relative_artifacts(tmp_path, p):
    with pytest.raises(ValueError, match="artifact"):
        artifact(tmp_path, p)


def test_curves_keep_endpoints_support_and_full_metrics():
    curves = {
        "thresholds": list(range(200)),
        "false_positive_rate": np.linspace(0, 1, 200).tolist(),
        "recall": np.linspace(0.2, 1, 200).tolist(),
        "precision": [0.7] * 200,
    }
    result = benchmark_preview(
        {
            "statistics": {
                "mi_bits": {
                    "interfaces": {
                        "AB": {"class_support": "both_classes", "auroc": 0.621, "curves": curves},
                        "AC": {
                            "class_support": "only_negatives",
                            "curves": {**curves, "recall": None},
                        },
                    }
                }
            }
        }
    )
    ab, ac, _ = result["methods"]
    assert ab["roc"][0] == [0, 0.2] and ab["roc"][-1] == [1, 1] and len(ab["roc"]) == 128
    assert ab["auroc"] == 0.621 and ab["thresholds"] == 200
    assert ac["status"] == "only_negatives" and ac["pr"] == [] and ac["roc"] == []


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []
        self.scripts = []
        self.dataset = ""
        self.inside = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs["href"])
        if tag == "script":
            self.scripts.append(attrs)
            self.inside = attrs.get("id") == "explore-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.inside = False

    def handle_data(self, text):
        if self.inside:
            self.dataset += text


def test_full_table_download_and_html_injection_bounded(tmp_path):
    s = store()
    attack = '</script><img src="exfil" onerror="alert(1)"> &'
    p = "ranking #1.tsv"
    with (tmp_path / p).open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["rank", "score", "category", "reference"])
        writer.writerows((i, i, "close", attack if i == 0 else "x" * 210) for i in range(2000))
    (tmp_path / "report.html").write_text("overview")
    table = RankedTable("AB", "MI", p, "mi_bits", ("x", "y"), 2000)
    write_explorer(tmp_path, {"outputs": {}}, [table], store=s)
    html = (tmp_path / "explore.html").read_text(encoding="utf8")
    page = Page()
    page.feed(html)
    assert attack not in html and "\\u003c/script\\u003e" in html
    assert len(set(page.ids)) == len(page.ids)
    assert all("src" not in item for item in page.scripts)
    sample = json.loads(page.dataset)["tables"][0]
    assert len(sample["rows"]) == 40 and attack in sample["rows"][0]
    assert max(map(len, sample["rows"][1])) == 200
    assert all(
        h[1:] in page.ids if h.startswith("#") else (tmp_path / unquote(h)).is_file()
        for h in page.links
    )
    assert p in {unquote(h) for h in page.links}
    assert len(html) < 200_000


def test_score_label_join_reverse_orientation_split_and_unmapped(tmp_path):
    s = store()
    s.n_valid["AB"][0, 1] = 1
    header = "x_component\tx_reference_id\tx_reference_pos\ty_component\ty_reference_id\ty_reference_pos\tcontact\tsplit\tstatus\n"
    (tmp_path / "labels.tsv").write_text(
        header
        + "B\trefB\t2\tA\trefA\t3\t1\ttrain\tresolved\nA\trefA\t2\tC\trefC\t1\t0\ttest\tresolved\nA\trefA\t999\tB\trefB\t2\t\t\tunresolved\n"
    )
    result = list(records(tmp_path, write_label_scores(tmp_path, "labels.tsv", s)))
    aa, bb, cc = result
    assert (
        aa["pair"],
        aa["x_original_col"],
        aa["y_original_col"],
        aa["contact"],
        aa["mi_bits"],
        aa["split"],
    ) == ("AB", "1", "3", "1", "", "train")
    assert bb["mi_bits"] == "5.0" and bb["contact"] == "0" and bb["split"] == "test"
    assert (
        cc["x_original_col"] == cc["n_valid"] == cc["mi_bits"] == ""
        and cc["label_status"] == "unresolved"
    )


def test_structure_overlay_joins_by_reference_retains_unknown_and_source_kind(tmp_path):
    s = store()
    sub = tmp_path / "structure"
    sub.mkdir()
    (sub / "structure_summary.json").write_text(
        json.dumps(
            {
                "source_kind": "predicted",
                "entry_id": "synthetic geometry",
                "assembly": "1",
                "model": 1,
                "definitions": {"metric": "heavy", "close_inclusive": 8},
            }
        )
    )
    (sub / "distances.tsv").write_text(
        "pair\tx_reference_id\tx_reference_pos\ty_reference_id\ty_reference_pos\tdistance_angstrom\tcontact\tcategory\nAB\trefA\t3\trefB\t1\t5\t1\tclose\nAB\trefA\t2\trefB\t2\t\t\tunresolved\nAB\twrongRef\t1\trefB\t1\t1\t1\tclose\n"
    )
    evidence = {("AB", "refA", "3", "refB", "1"): {}}
    markers, label, detail = structure_view(
        tmp_path, {"structure": "structure/structure_summary.json"}, s, network_evidence=evidence
    )
    assert np.argwhere(markers["AB"]).tolist() == [[0, 2]] and not markers["BC"].any()
    assert "predicted heavy distance ≤ 8 Å" == label and "Distance measured" in detail
    assert next(iter(evidence.values())) == {
        "structure_source_kind": "predicted",
        "distance_angstrom": "5",
        "structure_category": "close",
    }


def test_unknown_display_settings_and_older_without_archive(tmp_path):
    with pytest.raises(ValueError, match="palette"):
        write_explorer(tmp_path, {"outputs": {}}, [], palette="bright")
    with pytest.raises(ValueError, match="formats"):
        write_explorer(tmp_path, {"outputs": {}}, [], figure_formats=["../jpg"])
    write_explorer(tmp_path, {"outputs": {}}, [])
    page = (tmp_path / "explore.html").read_text(encoding="utf8")
    assert "full matrices are unavailable" in page and "no accompanying" in page


def test_zero_and_unmeasured_legends_and_old_curve_size_limit(tmp_path):
    s = store()
    s.scores["mi_apc"]["AB"][:] = 0

    def svg():
        g = score_grid(s, "AB", "mi_apc")
        return grid_svg(
            g,
            title="AB",
            units="bits",
            maximum=g["max_abs"],
            signed=True,
            palette="accessible",
            identifier="ab",
        )

    assert "All eligible scores are zero" in svg()
    s.scores["mi_apc"]["AB"][:] = np.nan
    assert "No eligible finite scores" in svg()
    path = tmp_path / "benchmark.json"
    with path.open("wb") as f:
        f.truncate(16 * 1024**2 + 1)
    block = benchmark_view(tmp_path, {"benchmark": "benchmark.json"})
    assert "16 MiB" in block and 'href="benchmark.json"' in block


def test_graphml_structural_provenance_attaches_to_measured_pair_only(tmp_path):
    s = store()
    sub = tmp_path / "structure"
    sub.mkdir()
    (sub / "structure_summary.json").write_text(
        json.dumps(
            {"source_kind": "synthetic", "definitions": {"metric": "heavy", "close_inclusive": 8}}
        )
    )
    (sub / "distances.tsv").write_text(
        "pair\tx_reference_id\tx_reference_pos\ty_reference_id\ty_reference_pos\tdistance_angstrom\tcontact\tcategory\nAB\trefA\t3\trefB\t2\t5\t1\tclose\n"
    )
    tables = []
    for method, kind, components, prefixes in (
        ("mi_bits", "MI", "AB", "xy"),
        ("threebody_rms", "Three-body Potts", "ABC", "abc"),
    ):
        row = {"rank": "1", method: "1", "n_valid": "4"}
        for i, (role, p) in enumerate(zip(components, prefixes, strict=True)):
            row.update(
                {
                    p + "_component": role,
                    p + "_original_col": str(1 + 2 * i),
                    p + "_reference_id": "ref" + role,
                    p + "_reference_pos": str(3 - i),
                }
            )
        path = method + ".tsv"
        with (tmp_path / path).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        tables.append(RankedTable(components, kind, path, method, tuple(prefixes), 1))
    write_explorer(
        tmp_path, {"outputs": {"structure": "structure/structure_summary.json"}}, tables, store=s
    )
    root = ET.parse(tmp_path / "network_structure.graphml")
    edges = [
        {d.get("key"): d.text for d in e.findall("{*}data")} for e in root.findall(".//{*}edge")
    ]
    assert (
        len(edges) == 4
        and edges[0]["structure_source_kind"] == "synthetic"
        and edges[0]["distance_angstrom"] == "5"
    )
    assert all(e == {"kind": "membership", "evidence_kind": "factor membership"} for e in edges[1:])


def test_taxonomic_depth_and_clade_holdout_are_distinct(tmp_path):
    (tmp_path / "heldout.html").write_text("folds")
    summary = {
        "outputs": {"conditional_report": "heldout.html"},
        "conditional": {
            "grid": [{"l2": 0.1, "evaluated_folds": 3, "clade_macro_gain_nats_per_site": -0.2}]
        },
        "qc_diagnostics": {
            "metadata": {
                "taxon": {
                    "status": "supplied",
                    "column": "taxon_id",
                    "input": {"distinct": 7},
                    "retained": {"distinct": 5, "missing": 2},
                },
                "clade": {"status": "unavailable"},
            }
        },
    }
    text = support_view(tmp_path, summary)
    assert (
        "<td>7</td><td>5</td><td>2</td>" in text
        and "Clade mean gain (nats/site)" in text
        and "<td>-0.2</td>" in text
    )
