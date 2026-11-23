"""Isolated synthetic fixtures for the folder-first evidence CLI and native analysis handoff."""

import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.family_config import family_config
from triplet_coevolution.io.alignments import read_msa

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "families"


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def invoke(base, out, *args):
    return CliRunner().invoke(app, ["evidence", "--input", str(base), "--out", str(out), *args])


def test_folder_evidence_and_independent_replay(tmp_path, monkeypatch):
    base = tmp_path / "input"
    shutil.copytree(EXAMPLE, base)
    flat = base / "ATPase" / "family.sto"
    compressed = flat.with_suffix(".sto.gz")
    compressed.write_bytes(gzip.compress(flat.read_bytes()))
    flat.unlink()
    out = tmp_path / "prepared"
    result = invoke(base, out, "--analyze")
    assert result.exit_code == 0, (result.output, result.exception)
    summary = json.loads((out / "workflow_summary.json").read_text())
    assert summary["ready_for_analysis"] and summary["counts"]["resolved_groups"] == 3
    group_statuses = {row["genome_id"]: row["status"] for row in rows(out / summary["group_audit"])}
    assert group_statuses == {
        "genome1": "resolved", "genome2": "resolved", "genome3": "resolved",
        "genome4": "ambiguous", "genome5": "incomplete",
    }
    assert (
        summary["inputs"]["A.alignment"]["sha256"]
        == hashlib.sha256(compressed.read_bytes()).hexdigest()
    )
    assert summary["components"]["A"]["source_rows"] == 7
    assert summary["folder_assignment"] == {"A": "ATPase", "B": "Partner", "C": "guide_RNA"}
    prepared = out / "prepared"
    matches = rows(prepared / "matches.tsv")
    assert [r["a_id"] for r in matches] == ["A1/1-4", "A2/1-4", "A3/1-4"]
    for part, name in zip("ABC", ["ATPase", "Partner", "guide_RNA"]):
        msa = read_msa(prepared / f"{part}.sto", part, "rna" if part == "C" else "protein")
        expected = [r[f"{part.lower()}_id"] for r in matches]
        assert list(msa.ids) == expected
        replay_path = out / summary["components"][part]["replay_alignment"]
        assert list(read_msa(replay_path, part, msa.molecule).ids) == [
            *expected,
            f"{part}REF/21-24",
        ]
        assert all(
            r["component_name"] == name
            for r in rows(out / summary["evidence"])
            if r["component"] == part
        )
    coords = [
        r["reference_position"]
        for r in rows(out / summary["reference_coordinates"])
        if r["component"] == "A"
    ]
    assert coords == ["21", "", "22", "23", "24"]
    audit = rows(out / summary["evidence"])
    assert {"Aabsent", "Aunknown/1-4", "A4/1-4", "A4/11-14"} <= {r["sequence_id"] for r in audit}
    assert not any(r["selected"] == "1" for r in audit if r["sequence_id"].startswith("A4/"))
    analysis = out / "analysis"
    assert "../metadata/evidence.tsv" in (analysis / "report.html").read_text(encoding="utf-8")
    analysis_manifest = json.loads((analysis / "run_summary.json").read_text())
    assert analysis_manifest["components"]["A"]["reference_id"] == "AREF"
    assert analysis_manifest["components"]["A"]["reference_row_id"] == "AREF/21-24"
    monkeypatch.chdir(prepared)
    replay = tmp_path / "replay"
    args = json.loads((out / summary["analyze_args"]).read_text())
    run = CliRunner().invoke(app, [*args, "--out", str(replay)])
    assert run.exit_code == 0, run.output
    for pair in ("ab", "ac", "bc"):
        assert (replay / f"mi_{pair}.tsv").read_bytes() == (
            analysis / f"mi_{pair}.tsv"
        ).read_bytes()
    assert invoke(base, out).exit_code != 0


def test_folder_discovery_without_reference_or_mappings(tmp_path):
    directory = tmp_path / "input"
    for name in ("z enzyme", "A family", "rna"):
        (directory / name).mkdir(parents=True)
        (directory / name / "label.stk").write_text("# STOCKHOLM 1.0\nfoo XY--\n//\n")
    out = tmp_path / "out"
    res = invoke(directory, out, "--analyze")
    assert res.exit_code == 0, (res.output, res.exception)
    summary = json.loads((out / "workflow_summary.json").read_text())
    assert summary["folder_assignment"] == {"A": "A family", "B": "rna", "C": "z enzyme"}
    assert summary["counts"]["input_rows"] == 3
    assert len(rows(out / summary["evidence"])) == 3
    assert not summary["ready_for_analysis"] and summary["analysis_requested"]
    assert not (out / "analysis").exists() and not (out / "prepared" / "A.sto").exists()
    assert not rows(out / summary["matches"])
    assert not summary["analyze_args"] and "set reference_id" in res.output


def test_matched_rows_without_reference_are_exported(tmp_path):
    base = tmp_path / "input"
    shutil.copytree(EXAMPLE, base)
    (base / "ATPase" / "component.json").write_text('{"molecule":"protein"}')
    out = tmp_path / "out"
    assert invoke(base, out).exit_code == 0
    data = json.loads((out / "workflow_summary.json").read_text())
    assert len(rows(out / data["matches"])) == 3
    assert data["components"]["A"]["exported_rows"] == 3 and not data["ready_for_analysis"]
    assert not (out / "analysis").exists()


def test_descending_nucleotide_reference_keeps_original_strand_and_coordinates(tmp_path):
    directory = tmp_path / "input"
    shutil.copytree(EXAMPLE, directory)
    rna = directory / "guide_RNA" / "family.sto"
    rna.write_text(rna.read_text().replace("CREF/21-24", "CREF/24-21"))
    out = tmp_path / "out"
    result = invoke(directory, out, "--analyze")
    assert result.exit_code == 0, (result.output, result.exception)
    summary = json.loads((out / "workflow_summary.json").read_text())
    assert summary["components"]["C"]["reference_strand"] == "-"
    coords = [
        r["reference_position"]
        for r in rows(out / summary["reference_coordinates"])
        if r["component"] == "C"
    ]
    assert coords == ["24", "", "23", "22", "21"]
    scores = rows(out / "analysis" / "mi_ac.tsv")
    assert {(r["y_original_col"], r["y_reference_pos"]) for r in scores} == {
        ("1", "24"),
        ("3", "23"),
        ("4", "22"),
        ("5", "21"),
    }


def test_synchronized_files_for_uneven_reference_membership(tmp_path):
    directory = tmp_path / "input"
    shutil.copytree(EXAMPLE, directory)
    (directory / "Partner" / "component.json").write_text(
        '{"molecule":"protein","reference_id":"B1"}'
    )
    out = tmp_path / "out"
    result = invoke(directory, out)
    assert result.exit_code == 0, (result.output, result.exception)
    data = json.loads((out / "workflow_summary.json").read_text())
    matches = rows(out / data["matches"])
    for key, comp in data["components"].items():
        msa = read_msa(out / comp["alignment"], key, comp["molecule"])
        assert msa.ids == tuple(row[f"{key.lower()}_id"] for row in matches)
        assert comp["exported_rows"] == len(matches)
    assert data["components"]["B"]["replay_alignment"] == "prepared/B.sto"
    assert data["components"]["A"]["replay_alignment"] == "prepared/A.with_reference.sto"
    assert not (out / "prepared/B.with_reference.sto").exists()


def test_zero_matches_and_coordinate_only_references(tmp_path):
    directory = tmp_path / "input"
    shutil.copytree(EXAMPLE, directory)
    (directory / "ATPase" / "targets.txt").write_text("absent\n")
    out = tmp_path / "out"
    result = invoke(directory, out)
    assert result.exit_code == 0, (result.output, result.exception)
    summary = json.loads((out / "workflow_summary.json").read_text())
    assert not rows(out / summary["matches"]) and not summary["analyze_args"]
    for name, comp in summary["components"].items():
        assert comp["exported_rows"] == 0 and comp["alignment"] is None
        assert not (out / "prepared" / f"{name}.sto").exists()
        assert read_msa(out / comp["replay_alignment"], name, comp["molecule"]).ids == (
            f"{name}REF/21-24",
        )


@pytest.mark.parametrize(
    "problem", ["many_files", "bad_order", "ambiguous_ref", "conflict_start", "inside"]
)
def test_mistakes_leave_original_inputs_and_no_partial_output(tmp_path, problem):
    folder = tmp_path / "input"
    shutil.copytree(EXAMPLE, folder)
    out = tmp_path / "new"
    if problem == "many_files":
        (folder / "ATPase" / "other.sto").write_text("")
    elif problem == "bad_order":
        (folder / "project.json").write_text(
            '{"components":{"A":"ATPase","B":"ATPase","C":"guide_RNA"}}'
        )
    elif problem == "ambiguous_ref":
        (folder / "ATPase" / "component.json").write_text('{"reference_id":"A4"}')
    elif problem == "conflict_start":
        (folder / "ATPase" / "component.json").write_text(
            '{"reference_id":"AREF","reference_start":5}'
        )
    else:
        out = folder / "new"
    result = invoke(folder, out)
    assert result.exit_code != 0
    assert "error:" in result.output
    assert not out.exists()


def test_explicit_path_config_and_component_filtered_uniprot(tmp_path):
    file = tmp_path / "x.json"
    file.write_text(
        json.dumps(
            {
                "components": {
                    part: {
                        "alignment": str(EXAMPLE / folder / "family.sto"),
                        "molecule": "rna",
                        "family": "Demo" + part,
                    }
                    for part, folder in zip("ABC", ("ATPase", "Partner", "guide_RNA"))
                }
            }
        )
    )
    cfg, paths, order = family_config(file, None)
    assert len(paths) == 4 and order is None and not cfg["uniprot"]
    with pytest.raises(ValueError, match="UniProt requires protein"):
        family_config(file, None, uniprot=True)


def test_mocked_online_cache_is_published_and_readonly_replayable(tmp_path, monkeypatch):
    from triplet_coevolution.evidence import uniprot

    directory = tmp_path / "input"
    shutil.copytree(EXAMPLE, directory)
    for name in ("family.sto", "targets.txt"):
        file = directory / "ATPase" / name
        file.write_text(file.read_text().replace("Aunknown", "P12345"))
    monkeypatch.setattr(uniprot, "_request", lambda url: (
        {"results": []}, {"x-uniprot-release": "mock", "x-uniprot-release-date": "01-Jan-2000"}
    ))
    online = tmp_path / "mock"
    result = invoke(directory, online, "--uniprot")
    assert result.exit_code == 0, (result.output, result.exception)
    published = online / "metadata" / "uniprot.sqlite"
    assert published.stat().st_size > 0
    snapshot = directory / "uniprot.sqlite"
    shutil.copyfile(published, snapshot)
    before = snapshot.read_bytes()
    monkeypatch.setattr(uniprot, "_request", lambda url: pytest.fail("unexpected network request"))
    for mode in ("--offline", "--uniprot"):
        out = tmp_path / mode[2:]
        result = invoke(directory, out, mode)
        assert result.exit_code == 0, (result.output, result.exception)
        assert (out / "metadata/evidence.tsv").read_bytes() == (online / "metadata/evidence.tsv").read_bytes()
        assert json.loads((out / "metadata/evidence_summary.json").read_text())["resolver"]["cache_hits"] >= 1
        assert snapshot.read_bytes() == before
