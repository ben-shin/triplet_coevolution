"""Exercise installed engine binaries against small, generated local databases."""

import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from Bio.Seq import Seq
from typer.testing import CliRunner

from triplet_coevolution.search.retrieve import retrieve
from triplet_coevolution.search.shared import program_versions


@pytest.mark.parametrize(
    "backend,molecule",
    [
        ("hmmer", "protein"),
        ("hmmer", "dna"),
        ("hmmer", "rna"),
        ("infernal", "rna"),
        ("psiblast", "protein"),
    ],
)
def test_installed_engine(backend, molecule, tmp_path, monkeypatch):
    needed = {
        "hmmer": ("hmmbuild", "hmmsearch" if molecule == "protein" else "nhmmer"),
        "infernal": ("cmbuild", "cmcalibrate", "cmsearch"),
        "psiblast": ("makeblastdb", "psiblast"),
    }[backend]
    if any(shutil.which(exe) is None for exe in needed):
        pytest.skip(f"optional local executable(s) required: {', '.join(needed)}")
    monkeypatch.setenv("BLAST_USAGE_REPORT", "false")
    rng = random.Random(127)
    alphabet = {"protein": "ACDEFGHIKLMNPQRSTVWY", "dna": "ACGT", "rna": "ACGU"}[molecule]
    reference = "".join(rng.choices(alphabet, k=64))
    if backend == "infernal":
        reference += str(Seq(reference[:12]).reverse_complement_rna())
    reversed_target = (
        str(Seq(reference).reverse_complement_rna())
        if molecule == "rna"
        else str(Seq(reference).reverse_complement())
        if molecule == "dna"
        else reference
    )
    ref = tmp_path / "ref.fasta"
    ref.write_text(f">ref\n{reference}\n", encoding="ascii")
    db = tmp_path / "sequence database.fasta"
    db.write_text(f">fwd comment\n{reference}\n>target|2\n{reversed_target}\n", encoding="ascii")
    comp = {
        "backend": backend,
        "database": db,
        "reference": ref,
        "molecule": molecule,
        "reference_id": "ref",
        "iterations": 2,
        "evalue": 0.1,
    }
    if backend in ("hmmer", "infernal"):
        sto = tmp_path / "seeds.sto"
        ss = "(" * 12 + "." * (len(reference) - 24) + ")" * 12
        sto.write_text(
            f"# STOCKHOLM 1.0\nseed1 {reference}\nseed2 {reference}\n#=GC SS_cons {ss}\n//\n",
            encoding="ascii",
        )
        comp["profile"] = tmp_path / "model"
        commands = (
            [
                [
                    "hmmbuild",
                    "--" + ("amino" if molecule == "protein" else molecule),
                    comp["profile"],
                    sto,
                ]
            ]
            if backend == "hmmer"
            else [
                ["cmbuild", comp["profile"], sto],
                [
                    "cmcalibrate",
                    "--cpu",
                    "1",
                    "--noforecast",
                    "-L",
                    "0.05",
                    "--tailp",
                    "0.1",
                    comp["profile"],
                ],
            ]
        )
        for cmd in commands:
            subprocess.run(cmd, check=True, capture_output=True)
    hits, sequences = retrieve(comp, reference)
    programs = program_versions(comp)
    assert all(p["version"][0].isdigit() and Path(p["path"]).is_file() for p in programs)
    assert [p["name"] for p in programs] == (
        ["makeblastdb", "psiblast"] if backend == "psiblast" else [needed[-1]]
    )
    assert set(hits) == {"fwd", "target|2"}
    for key, hit in hits.items():
        assert hit.first >= 1 and hit.last <= len(reference)
        assert hit.last - hit.first + 1 >= len(reference) * 0.8
        assert hit.score > 0 and 0 <= hit.evalue <= 0.1
        assert sequences[key] in reference
    assert hits["fwd"].strand == "+"
    assert hits["target|2"].strand == ("+" if molecule == "protein" else "-")
    if molecule != "protein":
        other, sequences = retrieve(comp, reversed_target)
        assert other["fwd"].strand == "-" and other["fwd"].model_strand == "+"
        assert other["target|2"].strand == "+" and other["target|2"].model_strand == "-"
        assert all(sequence in reversed_target for sequence in sequences.values())
    if backend == "psiblast":
        # Restart an actual binary PSSM checkpoint produced by the installed BLAST+.
        for args in (
            ["makeblastdb", "-in", str(ref), "-dbtype", "prot", "-out", "training"],
            [
                "psiblast",
                "-query",
                str(ref),
                "-db",
                "training",
                "-num_iterations",
                "2",
                "-out_pssm",
                "checkpoint",
                "-out",
                "training.txt",
            ],
        ):
            subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)
        hits_from_checkpoint, _ = retrieve({**comp, "profile": tmp_path / "checkpoint"}, reference)
        assert set(hits_from_checkpoint) == set(hits)


@pytest.mark.parametrize("molecule,exe", [("protein", "hmmsearch"), ("dna", "nhmmer")])
def test_partial_reference_hit_and_copy_replay(molecule, exe, tmp_path, monkeypatch):
    """Real local domains on both observed rows, full seed for reported coordinates."""
    from triplet_coevolution.cli import app
    from triplet_coevolution.io.alignments import read_msa

    if any(shutil.which(name) is None for name in ("hmmbuild", exe)):
        pytest.skip("requires HMMER")
    rng = random.Random(127)
    alphabet = "ACDEFGHIKLMNPQRSTVWY" if molecule == "protein" else "ACGT"
    core = "".join(rng.choices(alphabet, k=64))
    reference = "A" * 9 + core + "A" * 9
    raw = reference if molecule == "protein" else str(Seq(reference).reverse_complement())
    (tmp_path / "seeds.sto").write_text(
        f"# STOCKHOLM 1.0\none {core}\ntwo {core}\n//\n", encoding="ascii"
    )
    subprocess.run(
        ["hmmbuild", "--amino" if molecule == "protein" else "--dna", "model.hmm", "seeds.sto"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    (tmp_path / "Aref.fa").write_text(">ref\n" + reference + "\n", encoding="ascii")
    (tmp_path / "Adb.fa").write_text(
        f">ref\n{raw}\n>copy\n{raw}\n>ref.coordinate\n{raw}\n", encoding="ascii"
    )
    (tmp_path / "R.fa").write_text(">anchor\nMALKTGVF\n", encoding="ascii")
    (tmp_path / "D.fa").write_text(">x\nMALKTGVF\n>y\nMALRTGVF\n", encoding="ascii")
    (tmp_path / "evidence.tsv").write_text(
        "group_id\tcomponent\tsequence_id\n"
        "one\tA\tref\none\tB\tx\none\tC\tx\ntwo\tA\tcopy\ntwo\tB\ty\ntwo\tC\ty\n",
        encoding="ascii",
    )
    other = {
        "backend": "reference",
        "reference": "R.fa",
        "database": "D.fa",
        "reference_id": "anchor",
        "molecule": "protein",
    }
    cfg = {
        "schema_version": 1,
        "evidence": "evidence.tsv",
        "components": {
            "A": {
                "backend": "hmmer",
                "profile": "model.hmm",
                "reference": "Aref.fa",
                "database": "Adb.fa",
                "reference_id": "ref",
                "reference_start": 101,
                "molecule": molecule,
            },
            "B": other,
            "C": other,
        },
    }
    config = tmp_path / "run.json"
    (tmp_path / "contacts.tsv").write_text(
        "x_component\tx_reference_id\tx_reference_pos\ty_component\ty_reference_id\ty_reference_pos\tcontact\n"
        "A\tref\t110\tB\tanchor\t1\t1\nA\tref\t111\tB\tanchor\t1\t0\n",
        encoding="ascii",
    )
    cfg["analysis"] = {"contacts": "contacts.tsv"}
    config.write_text(json.dumps(cfg), encoding="ascii")
    out = tmp_path / "full"
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--config", str(config), "--out", str(out)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "workflow_summary.json").read_text(encoding="utf-8"))
    (binary,) = manifest["search"]["A"]["executables"]
    assert binary["name"] == exe and binary["version"][0].isdigit()
    assert Path(binary["path"]).is_file()
    prepared = out / "prepared"
    qc = json.loads((prepared / "matching_qc.json").read_text(encoding="utf-8"))
    msa = read_msa(prepared / "A.fasta", "A", molecule)
    aligned = dict(zip(msa.ids, msa.sequences))
    with (out / "search" / "A_hits.tsv").open(encoding="utf-8", newline="") as handle:
        hits = {row["seq_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    first, last = int(hits["ref"]["first"]), int(hits["ref"]["last"])
    assert (first, last) == (10, 73)
    assert hits["ref"]["strand"] == ("-" if molecule == "dna" else "+")
    assert aligned["ref"] == aligned["copy"] == "-" * 9 + core + "-" * 9
    assert aligned[qc["alignments"]["A"]["reference_row_id"]] == reference
    assert qc["alignments"]["A"]["reference_row_id"] == "ref.coordinate.1"
    with (prepared / "matches.tsv").open(encoding="utf-8", newline="") as handle:
        assert [r["a_id"] for r in csv.DictReader(handle, delimiter="\t")] == ["ref", "copy"]
    with (out / "analysis" / "mi_ab.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows and all(r["x_reference_id"] == "ref" for r in rows)
    assert min(int(r["x_reference_pos"]) for r in rows) == 110
    monkeypatch.chdir(prepared)
    replay_args = json.loads((out / manifest["analyze_args"]).read_text(encoding="utf-8"))
    replay = runner.invoke(app, [*replay_args, "--out", str(tmp_path / "replay")])
    assert replay.exit_code == 0, replay.output
    for pair in ("ab", "ac", "bc"):
        name = f"mi_{pair}.tsv"
        assert (out / "analysis" / name).read_bytes() == (tmp_path / "replay" / name).read_bytes()
    assert (tmp_path / "replay" / "contacts.json").read_bytes() == (
        out / "analysis" / "contacts.json"
    ).read_bytes()
