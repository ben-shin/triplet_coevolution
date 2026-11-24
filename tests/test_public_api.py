"""Use the installed-style library and CLI on local, reproducible mixed-alphabet projects."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from triplet_coevolution import api
from triplet_coevolution.cli import app
from triplet_coevolution.io.score_store import load_score_store
from triplet_coevolution.templates import TEMPLATES, init_project


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf8")
    return path


def command(*args):
    result = CliRunner().invoke(app, [str(arg) for arg in args])
    assert result.exit_code == 0, (result.output, result.exception)
    return result


@pytest.mark.parametrize("template", TEMPLATES)
def test_runnable_template_and_common_coordinates(tmp_path, template):
    project = tmp_path / "project"
    command("init", "--template", template, "--out", project)
    req = api.AnalysisRequest.from_config(project / "analysis.json")
    assert tuple(getattr(req, k).molecule for k in "abc") == TEMPLATES[template]
    command("score", "--config", project / "analysis.json", "--out", tmp_path / "score")
    store = load_score_store(tmp_path / "score" / "scores")
    assert all(
        tuple(m.reference_positions) == (start, None, start + 1, start + 2)
        for m, start in zip(store.maps, (11, 51, 101), strict=True)
    )
    assert all(
        float(store.scores["mi_bits"][pair][0, 0]) == pytest.approx(0, abs=1.0e-12)
        for pair in ("AB", "AC", "BC")
    )
    with (tmp_path / "score" / "threebody_abc.tsv").open(encoding="utf8", newline="") as handle:
        row = next(
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if tuple(row[k + "_original_col"] for k in "abc") == ("1", "1", "1")
        )
    assert float(row["co_information_bits"]) == pytest.approx(-1.0)
    folder = api.run_folders(project / "input", tmp_path / "folder", online=False)
    assert folder.report.is_file() and folder.summary["exit_code"] == 0
    assert folder.summary["counts"]["resolved_groups"] == 16


def test_library_and_archive_after_input_removal(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    request = api.AnalysisRequest.from_config(init_project(project, "protein-rna-rna"))
    run = api.analyze(request, tmp_path / "fit")
    assert run.manifest.is_file() and run.report.is_file()
    label = tmp_path / "labels.tsv"
    shutil.copyfile(project / "labels.tsv", label)
    from triplet_coevolution import analysis_engine

    monkeypatch.setattr(
        analysis_engine, "execute", lambda *a, **k: pytest.fail("regeneration fitted a model")
    )
    shutil.rmtree(project)
    view = api.report(tmp_path / "from_store", scores=run.directory, labels=label)
    assert view.report.is_file() and (view.directory / "explore.html").is_file()
    assert (view.directory / "scores/pair_scores.npz").is_file()
    restored = api.report(tmp_path / "restored", run=run.directory)
    assert restored.summary["ranking"] == run.summary["ranking"]
    assert api.benchmark(
        view.directory / "scores", label, tmp_path / "benchmark", seed=7
    ).summary == read(run.directory / "benchmark.json")
    assert api.select(run.directory, tmp_path / "chosen").summary == read(
        run.directory / "candidate_manifest.json"
    )
    assert not capsys.readouterr().out


def test_config_sources_and_migration(tmp_path):
    project = tmp_path / "base"
    config = init_project(project)
    child = tmp_path / "overrides" / "child.policy"
    child.parent.mkdir()
    put(
        child,
        {"extends": "../base/analysis.json", "analysis": {"max_pairs": 3, "benchmark_k": [2]}},
    )
    leaf = put(
        tmp_path / "leaf",
        {"extends": "overrides/child.policy", "analysis": {"benchmark_k": [3, 5]}},
    )
    req = api.AnalysisRequest.from_config(leaf)
    assert req.analysis["max_pairs"] == 3 and req.analysis["benchmark_k"] == (3, 5)
    assert req.a.alignment.parent.parent.parent == project
    fit = api.analyze(req, tmp_path / "fit")
    assert [r["sha256"] for r in fit.summary["configuration"]["sources"]] == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (leaf, child, config)
    ]
    put(child, {"extends": "../base/analysis.json", "analysis": {"max_pairs": 4}})
    with pytest.raises(ValueError, match="configuration changed"):
        api.analyze(req, tmp_path / "stale")
    assert not (tmp_path / "stale/run_summary.json").exists()
    put(child, {"extends": "../leaf"})
    with pytest.raises(ValueError, match="cycle"):
        api.AnalysisRequest.from_config(leaf)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"analysis": {"unknown_flag": False}}, "unknown_flag"),
        ({"analysis": {"min_valid": "two"}}, "analysis.min_valid"),
        ({"analysis": 2}, "analysis must be"),
        ({"schema_version": 2}, "schema_version"),
        ({"components": {"C": {"alignment": 7}}}, "components.C.alignment"),
        ({"components": {"B": {"reference_start": False}}}, "components.B.reference_start"),
    ],
)
def test_rejects_invalid_config_early(tmp_path, patch, message):
    init_project(tmp_path / "base")
    config = put(tmp_path / "bad.json", {"extends": "base/analysis.json", **patch})
    with pytest.raises((ValueError, TypeError), match=message):
        api.AnalysisRequest.from_config(config)


def test_raw_library_entrypoint_is_quiet(tmp_path, capsys):
    fixture = Path(__file__).resolve().parents[1] / "examples/workflow/run.json"
    child = tmp_path / "config"
    child.write_text(
        json.dumps({"extends": str(fixture), "analysis": {"plots": False}}), encoding="utf8"
    )
    result = api.run(child, tmp_path / "run")
    assert result.report.is_file() and result.summary["matched_groups"] == 8
    assert (
        result.summary["inputs"]["config.parent.1"]["sha256"]
        == hashlib.sha256(fixture.read_bytes()).hexdigest()
    )
    assert not capsys.readouterr().out


def test_embedded_structural_coordinates_and_downloads(tmp_path):
    from triplet_coevolution.structure.workflow import run_structure

    structure_config = Path(__file__).resolve().parents[1] / "examples/structure/config.json"
    structures = tmp_path / "distances"
    run_structure(structure_config, structures)
    refs = read(structure_config)["references"]
    roles = []
    for r, first in zip("ABC", ("ACDE", "FWYK", "AUGC"), strict=True):
        path = tmp_path / f"{r}.fasta"
        path.write_text(f">{refs[r]['id']}\n{first}\n>{r}other\n{first[::-1]}\n", encoding="utf8")
        roles.append(api.ComponentInput(path, refs[r]["molecule"], refs[r]["id"]))
    matches = tmp_path / "matches.tsv"
    matches.write_text(
        "observation_id\ta_id\tb_id\tc_id\none\tenzyme_fragment\tpartner_fragment\trna_strand\ntwo\tAother\tBother\tCother\n"
    )
    run = api.analyze(api.AnalysisRequest(*roles, matches), tmp_path / "scored")
    report = api.report(tmp_path / "geometric-view", scores=run.directory, structure=structures)
    metadata = read(report.directory / "structure/structure_summary.json")
    assert (
        metadata["bundle"]["source_manifest_sha256"]
        == hashlib.sha256((structures / "structure_summary.json").read_bytes()).hexdigest()
    )
    for name in ("mapping.tsv", "distances.tsv", "viewer/view_chimerax.py", "viewer/C.cif"):
        assert (
            hashlib.sha256((report.directory / "structure" / name).read_bytes()).hexdigest()
            == metadata["output_sha256"][name]
        )
    # A clean independent native structural run with the wrong physical reference must abstain.
    mismatch = read(structures / "structure_summary.json")
    mismatch["inputs"]["references"]["C"]["reference_id"] = "different molecule"
    put(structures / "structure_summary.json", mismatch)
    with pytest.raises(ValueError, match="differs for C"):
        api.report(tmp_path / "wrong", scores=run.directory, structure=structures)
    assert not (tmp_path / "wrong/view_summary.json").exists()
