"""Verify observed statistics, stored coordinates and offline views through the CLI."""

import csv
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.io.score_store import load_score_store

RUNNER = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def invoke(*arguments):
    result = RUNNER.invoke(app, [str(x) for x in arguments])
    assert result.exit_code == 0, (result.output, result.exception)


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def mixed(out, *args):
    demo = EXAMPLES / "mixed"
    invoke(
        "analyze",
        "--a",
        demo / "protein.fasta",
        "--b",
        demo / "dna.fasta",
        "--c",
        demo / "rna.fasta",
        "--matches",
        demo / "matches.tsv",
        "--b-type",
        "dna",
        "--c-type",
        "rna",
        "--a-ref",
        "prot_ref",
        "--b-ref",
        "dna_ref",
        "--c-ref",
        "rna_ref",
        "--dca",
        "--out",
        out,
        *args,
    )


def test_native_selection_and_reusable_full_scores(tmp_path, monkeypatch):
    from triplet_coevolution.inference import advanced

    out, views, metrics, report = (tmp_path / p for p in ("run", "views", "metrics", "report"))
    labels = EXAMPLES / "workflow" / "synthetic_labels.tsv"
    calls = []
    fit = advanced.direct_couplings

    def counted(*args, **kwargs):
        calls.append(1)
        return fit(*args, **kwargs)

    monkeypatch.setattr(advanced, "direct_couplings", counted)
    mixed(out, "--max-pairs", 1, "--contacts", labels)
    assert len(calls) == 1
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["schema_version"] == 2 and summary["ranking"]["dca"] == "dca_apc"
    assert summary["ranking"]["mi_coverage"]["AB"]["exported"] == 1
    assert summary["ranking"]["mi_coverage"]["AB"]["finite"] > 1
    store = load_score_store(out / "scores")
    assert store.maps[0].original_positions == (1, 3, 4, 5)
    assert np.any(store.scores["dca_apc"]["AB"] < 0)
    (best,) = rows(out / "dca_ab.tsv")
    assert float(best["score"]) == np.nanmax(store.scores["dca_apc"]["AB"])
    native = json.loads((out / "benchmark.json").read_text())
    assert set(native["statistics"]) == {"mi_bits", "mi_apc", "dca_apc", "dca_frobenius"}
    for statistic in native["statistics"].values():
        assert set(statistic["interfaces"]) == {"AB", "AC", "BC"}
    invoke("benchmark", "--scores", out, "--labels", labels, "--merged", "--out", metrics)
    assert (out / "benchmark.json").read_bytes() == (metrics / "benchmark.json").read_bytes()
    assert any(r["label_status"] == "unlabeled" for r in rows(metrics / "merged_scores.tsv"))
    invoke(
        "export",
        "--scores",
        out / "scores",
        "--all-pairs",
        "--dca-score",
        "dca_frobenius",
        "--out",
        views,
    )
    exported = rows(views / "dca_ab.tsv")
    assert len(exported) == 16
    assert float(exported[0]["score"]) == np.nanmax(store.scores["dca_frobenius"]["AB"])
    invoke("report", "--scores", out, "--dca-score", "dca_frobenius", "--out", report)
    assert "dca_frobenius · higher first" in (report / "report.html").read_text(encoding="utf-8")
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    network = ET.parse(out / "network.graphml")
    methods = {node.text for node in network.findall(".//g:data[@key='method']", ns)}
    assert methods == {"mi_apc", "dca_apc"}
    assert len(calls) == 1


def test_regenerate_verifies_outputs_and_recovers_missing_report(tmp_path):
    original, rebuilt = tmp_path / "original", tmp_path / "rebuilt"
    mixed(original, "--legacy", "--threebody", "--permutations", 2)
    old = json.loads((original / "run_summary.json").read_text())
    assert old["ranking"]["legacy"] and old["ranking"]["dca"] == "dca_frobenius"
    (original / "report.html").unlink()
    invoke("report", "--run", original, "--out", rebuilt)
    new = json.loads((rebuilt / "run_summary.json").read_text())
    assert new["artifact_sha256"] == old["artifact_sha256"]
    assert (
        new["artifact_sha256"]["report.html"]
        == hashlib.sha256((rebuilt / "report.html").read_bytes()).hexdigest()
    )
    table = original / "dca_ab.tsv"
    table.write_bytes(table.read_bytes() + b"\n")
    failure = RUNNER.invoke(
        app, ["report", "--run", str(original), "--out", str(tmp_path / "invalid")]
    )
    assert failure.exit_code == 1 and "checksum" in failure.output
    assert not (tmp_path / "invalid").exists()


def test_score_options_in_raw_config_and_png(tmp_path):
    pytest.importorskip("matplotlib")
    import shutil

    project = tmp_path / "project"
    shutil.copytree(EXAMPLES / "workflow", project)
    config = project / "run.json"
    data = json.loads(config.read_text())
    data["analysis"].update(
        dca_score="dca_apc",
        threebody_score="co_information_bits",
        apc_universe="reference",
        max_pairs=1,
        plots=True,
        permutations=2,
        benchmark_k=[1, 3],
    )
    config.write_text(json.dumps(data))
    out = tmp_path / "workflow"
    invoke("run", "--config", config, "--out", out)
    stage = out / "analysis"
    summary = json.loads((stage / "run_summary.json").read_text())
    assert summary["outputs"]["figures"]["dca"] == "figures/dca_apc_triplet.png"
    assert summary["outputs"]["figures"]["threebody"] == "figures/co_information_bits_abc.png"
    benchmark = json.loads((stage / "benchmark.json").read_text())
    assert "3" in benchmark["statistics"]["dca_apc"]["interfaces"]["AB"]["top_k"]
    assert (stage / summary["outputs"]["figures"]["dca"]).read_bytes().startswith(b"\x89PNG")
    assert "co_information_bits · lower first" in (stage / "report.html").read_text(
        encoding="utf-8"
    )
    assert (
        summary["artifact_sha256"]["report.html"]
        == hashlib.sha256((stage / "report.html").read_bytes()).hexdigest()
    )
    for filename, statistic in [
        ("permutation_threebody.tsv", "co_information_bits"),
        ("permutation_dca.tsv", "dca_apc"),
    ]:
        assert {r["statistic"] for r in rows(stage / filename)} == {statistic}
    assert rows(stage / "threebody_abc.tsv")[0]["direction"] == "lower"


def test_unknown_labels_remain_outside_native_comparison(tmp_path):
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "x_component\tx_reference_id\tx_reference_pos\ty_component\ty_reference_id\t"
        "y_reference_pos\tcontact\tstatus\tsplit\n"
        "A\tprot_ref\t1\tB\tdna_ref\t1\t\tunresolved\ttest\n"
        "A\tprot_ref\t2\tB\tdna_ref\t2\t1\tresolved\ttrain\n"
        "A\tprot_ref\t3\tB\tdna_ref\t3\t0\tresolved\ttest\n"
    )
    out = tmp_path / "run"
    mixed(out, "--contacts", labels, "--benchmark-k", 1)
    result = json.loads((out / "benchmark.json").read_text())
    for statistic in result["statistics"].values():
        assert statistic["pooled"]["scored"] == 1
        assert statistic["pooled"]["auroc"] is None
        assert statistic["pooled"]["negatives_scored"] == 1
    assert {r["reason"] for r in result["excluded_labels"]} == {"unresolved", "non_test_split"}
