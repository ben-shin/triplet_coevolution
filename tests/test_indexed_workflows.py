"""Coordinate and domain audits at the public CLI entry points."""

import csv
import gzip
import hashlib
import json
import random
import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.validation.null_archive import RefitData


def rows(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def invoke(*args):
    return CliRunner().invoke(app, [str(a) for a in args])


def test_a3m_selection_and_relocation_follow_native_format(tmp_path):
    data = tmp_path / "source"
    data.mkdir()
    matches = data / "matches.tsv"
    matches.write_text("observation_id\ta_id\tb_id\tc_id\nobs0\ts0\ts0\ts0\nobs1\ts1\ts1\ts1\n")
    text = ">ref\nCtA.G\n>s0\nAAA\n>s1\nCCC\n" + "".join(
        f">decoy{i:04d}\nCGG\n" for i in range(1200)
    )
    source = data / "alignment.unusual"
    source.write_bytes(gzip.compress(text.encode()))
    args = []
    for role in "abc":
        args += [
            "--" + role,
            source,
            "--" + role + "-format",
            "a3m",
            "--" + role + "-ref",
            "ref",
            "--" + role + "-type",
            "dna",
        ]
    plan = tmp_path / "plan"
    res = invoke("plan", *args, "--matches", matches, "--out", plan, "--memory-mib", "1")
    assert res.exit_code == 0, res.output
    src = json.loads((plan / "plan.json").read_text())["source_selection"]["A"]
    assert src["source_rows"] == 1203 and src["selected_rows"] == 3
    assert src["selected_symbols"] == 11 and src["selected_match_cells"] == 9
    out = tmp_path / "analysis"
    res = invoke("analyze", *args, "--matches", matches, "--out", out)
    assert res.exit_code == 0, res.output
    data_loaded = RefitData.load(out)
    assert data_loaded.msa.n_observations == 2
    pairs = rows(out / "mi_ab.tsv")
    assert {r["x_reference_pos"] for r in pairs} == {"1", "3", "4"}
    relocated = tmp_path / "other.bin"
    source.rename(relocated)
    assert RefitData.load(out, inputs={c: relocated for c in "ABC"}).msa.n_observations == 2


def test_prepare_complete_unassigned_ledger_and_retained_insertions(tmp_path, monkeypatch):
    name = tmp_path / "records.a3m"
    seq = ">ref\nCtA.G\n>s0\nAAA\n>s1\nCCC\n" + "".join(f">x{i:04d}\nAGC\n" for i in range(1002))
    name.write_text(seq)
    config = tmp_path / "prepare.json"
    evidence = tmp_path / "groups.tsv"
    evidence.write_text(
        "group_id\tcomponent\tsequence_id\n"
        + "".join(f"obs{i}\t{c}\ts{i}\n" for c in "ABC" for i in range(2))
    )
    config.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "evidence": evidence.name,
                "components": {
                    c: {
                        "molecule": "dna",
                        "alignment": name.name,
                        "reference_id": "ref",
                        "reference_start": 30,
                        "reference_strand": "-",
                    }
                    for c in "ABC"
                },
            }
        )
    )
    out = tmp_path / "prep"
    res = invoke("prepare", "--config", config, "--out", out)
    assert res.exit_code == 0, res.output
    qc = json.loads((out / "matching_qc.json").read_text())
    assert (
        qc["unassigned_sequence_counts"]["A"] == 1003
        and len(qc["unassigned_sequences"]["A"]) == 1000
    )
    assert len(rows(out / qc["complete_unassigned_tables"]["A"])) == 1003
    assert qc["alignments"]["A"]["selection"]["coordinate_space"] == "a3m_match_columns"
    assert "CtA.G" in (out / "A.a3m").read_text()
    assert "x0001" not in (out / "A.a3m").read_text()
    replay = json.loads((out / "analyze_args.json").read_text())
    assert all(replay[replay.index(f"--{role}-strand") + 1] == "-" for role in "abc")
    monkeypatch.chdir(out)
    result = invoke(*replay, "--out", tmp_path / "reverse")
    assert result.exit_code == 0, result.output
    assert {r["x_reference_pos"] for r in rows(tmp_path / "reverse" / "mi_ab.tsv")} == {
        "30",
        "28",
        "27",
    }


def test_raw_repeat_compatibility_and_conservative_wildcard(tmp_path):
    motif = "MACWFTEDR"
    reference = tmp_path / "ref.fa"
    reference.write_text(">query\n" + motif + "\n")
    database = tmp_path / "db.fa"
    database.write_text(">twice\n" + motif + "GGGGGG" + motif + "\n>once\n" + motif + "\n")
    cfg = {
        "components": {
            c: {
                "reference": reference.name,
                "database": database.name,
                "molecule": "protein",
                "reference_id": "query",
                "min_identity": 1,
                "min_coverage": 1,
            }
            for c in "ABC"
        },
        "evidence": "groups.tsv",
    }
    ev = tmp_path / "groups.tsv"
    ev.write_text(
        "group_id\tcomponent\tsequence_id\n" + "".join(f"one\t{c}\ttwice\n" for c in "ABC")
    )
    path = tmp_path / "raw.json"
    for version in (1, 2):
        cfg["schema_version"] = version
        path.write_text(json.dumps(cfg))
        out = tmp_path / f"v{version}"
        result = invoke("run", "--config", path, "--out", out)
        assert result.exit_code == (0 if version == 1 else 3), (result.output, result.exception)
        manifest = json.loads((out / "workflow_summary.json").read_text())
        assert manifest["matched_groups"] == (1 if version == 1 else 0)
        domain = rows(out / "search" / "A_discovery" / "hit_audit.tsv")
        assert len(domain) >= 2
        if version == 2:
            assert {row["row_id"] for row in domain if row["seq_id"] == "twice"} == {
                "twice/1-9",
                "twice/16-24",
            }
            assert (out / manifest["report"]).is_file() and not (out / "analysis").exists()
            assert (
                hashlib.sha256(
                    (out / "search" / "A_discovery" / "hit_audit.tsv").read_bytes()
                ).hexdigest()
                == manifest["search"]["A"]["discovery"]["audit_sha256"]
            )
    ev.write_text(
        ev.read_text().replace("twice", "twice/16-24") + "".join(f"two\t{c}\tonce\n" for c in "ABC")
    )
    out = tmp_path / "v2-selected"
    run = invoke("run", "--config", path, "--out", out)
    assert run.exit_code == 0, (run.output, run.exception)
    assert {r["a_id"] for r in rows(out / "prepared" / "matches.tsv")} == {"twice/16-24", "once"}


@pytest.mark.skipif(
    not (shutil.which("hmmsearch") and shutil.which("hmmbuild")), reason="requires HMMER"
)
def test_actual_hmmer_v1_best_and_v2_distinct_domains(tmp_path):
    motif = "".join(random.Random(12).choices("MACWFTEDRKLGPYHSQVIN", k=64))
    (tmp_path / "seed.sto").write_text(f"# STOCKHOLM 1.0\none {motif}\ntwo {motif}\n//\n")
    subprocess.run(
        ["hmmbuild", "--amino", "model.hmm", "seed.sto"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    (tmp_path / "ref.fa").write_text(">seed\n" + motif + "\n")
    (tmp_path / "db.fa").write_text(
        ">two_copies\n" + motif + "PPGG" + motif + "\n>single\n" + motif + "\n"
    )
    ev = tmp_path / "groups.tsv"
    ev.write_text(
        "group_id\tcomponent\tsequence_id\n"
        + "".join(f"one\t{c}\ttwo_copies\ntwo\t{c}\tsingle\n" for c in "ABC")
    )
    path = tmp_path / "settings.json"
    cfg = {
        "components": {
            c: {
                "reference": "ref.fa",
                "database": "db.fa",
                "reference_id": "seed",
                "molecule": "protein",
                "backend": "hmmer",
                "profile": "model.hmm",
            }
            for c in "ABC"
        },
        "evidence": ev.name,
    }
    for version in (1, 2):
        cfg["schema_version"] = version
        path.write_text(json.dumps(cfg))
        out = tmp_path / f"result-{version}"
        r = invoke("run", "--config", path, "--out", out)
        assert r.exit_code == 0, (r.output, r.exception)
        manifest = json.loads((out / "workflow_summary.json").read_text())
        assert manifest["matched_groups"] == (2 if version == 1 else 1)
        a = rows(out / "search" / "A_discovery" / "hit_audit.tsv")
        copies = [row for row in a if row["seq_id"] == "two_copies"]
        assert len(copies) == 2
        if version == 1:
            assert sum(row["selected_legacy"] in ("True", "1") for row in copies) == 1
        else:
            assert len({row["row_id"] for row in copies}) == 2
            assert not any(row["alignment_status"] == "aligned" for row in copies)
