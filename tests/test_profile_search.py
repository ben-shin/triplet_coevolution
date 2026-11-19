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
    variant = reference[:28] + next(c for c in alphabet if c != reference[28]) + reference[29:]
    db.write_text(
        f">fwd comment\n{reference}\n>target|2\n{reversed_target}\n>variant\n{variant}\n",
        encoding="ascii",
    )
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
    assert set(hits) == {"fwd", "target|2", "variant"}
    for key, hit in hits.items():
        assert hit.first >= 1 and hit.last <= len(reference)
        assert hit.last - hit.first + 1 >= len(reference) * 0.8
        assert hit.score > 0 and 0 <= hit.evalue <= 0.1
        assert sequences[key] in (variant if key == "variant" else reference)
    assert hits["variant"].projection_method == (
        "query_interval" if backend == "psiblast" else "profile_locus"
    )
    assert (
        hits["variant"].placement.aligned[28 - hits["variant"].placement.first + 1] == variant[28]
    )
    assert hits["fwd"].strand == "+"
    assert hits["target|2"].strand == ("+" if molecule == "protein" else "-")
    if molecule != "protein":
        other, sequences = retrieve(comp, reversed_target)
        assert other["fwd"].strand == "-" and other["fwd"].model_strand == "+"
        assert other["target|2"].strand == "+" and other["target|2"].model_strand == "-"
        reverse_variant = str(
            Seq(variant).reverse_complement_rna()
            if molecule == "rna"
            else Seq(variant).reverse_complement()
        )
        assert all(
            sequence in (reverse_variant if key == "variant" else reversed_target)
            for key, sequence in sequences.items()
        )
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


@pytest.mark.parametrize(
    "molecule,copies,linker,flipped",
    [
        ("protein", 1, "", False),
        ("dna", 1, "", False),
        ("protein", 2, "", False),
        ("protein", 3, "GGWW", False),
        ("dna", 2, "CCC", True),
    ],
)
def test_partial_reference_hit_and_copy_replay(
    molecule, copies, linker, flipped, tmp_path, monkeypatch
):
    """Real local domains on both observed rows, full seed for reported coordinates."""
    from triplet_coevolution.cli import app
    from triplet_coevolution.io.alignments import read_msa
    from triplet_coevolution.search import hmmer

    exe = "hmmsearch" if molecule == "protein" else "nhmmer"
    if any(shutil.which(name) is None for name in ("hmmbuild", exe)):
        pytest.skip("requires HMMER")
    rng = random.Random(127)
    alphabet = "ACDEFGHIKLMNPQRSTVWY" if molecule == "protein" else "ACGT"
    core = "".join(rng.choices(alphabet, k=64))
    reference = "A" * 9 + core + "A" * 9 if copies == 1 else linker.join([core] * copies)
    model = str(Seq(core).reverse_complement()) if flipped else core
    raw = reference if molecule == "protein" else str(Seq(reference).reverse_complement())
    (tmp_path / "seeds.sto").write_text(
        f"# STOCKHOLM 1.0\none {model}\ntwo {model}\n//\n", encoding="ascii"
    )
    subprocess.run(
        ["hmmbuild", "--amino" if molecule == "protein" else "--dna", "model.hmm", "seeds.sto"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    (tmp_path / "Aref.fa").write_text(">ref\n" + reference + "\n", encoding="ascii")
    database = tmp_path / "Adb.fa"
    source = f">ref\n{raw}\n>copy\n{raw}\n>ref.coordinate\n{raw}\n"
    if copies > 1:
        contextual = reference + "G"
        if molecule == "dna":
            contextual = str(Seq(contextual).reverse_complement())
        source += f">context\n{contextual}\n"
    database.write_text(source, encoding="ascii")
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
                "reference_start": 51,
                "molecule": molecule,
            },
            "B": other,
            "C": other,
        },
    }
    config = tmp_path / "run.json"
    # Determine the expected coordinates from the search program's raw target intervals,
    # independent of the workflow's projection or output tables.
    tool = {"evalue": 0.01, "molecule": molecule, "profile": tmp_path / "model.hmm"}
    domains = list(hmmer.search(tool, database, tmp_path))
    expected = {}
    lengths = {"ref": len(reference), "copy": len(reference)}
    if copies > 1:
        lengths["context"] = len(reference) + 1
    for seq_id, length in lengths.items():
        observed = min(
            (h for h in domains if h.seq_id == seq_id and h.evalue <= 0.01),
            key=lambda h: (-h.score, h.evalue, h.strand, h.first, h.last),
        )
        x, y = (
            (observed.first, observed.last)
            if molecule == "protein"
            else (length - observed.last + 1, length - observed.first + 1)
        )
        expected[seq_id] = x, y
    x, y = expected["ref"]
    if copies == 1:
        assert (x, y) == (10, 73)
    if copies == 2 and not linker:
        assert (x, y) == (1, 64)
    (tmp_path / "contacts.tsv").write_text(
        "x_component\tx_reference_id\tx_reference_pos\ty_component\ty_reference_id\ty_reference_pos\tcontact\n"
        f"A\tref\t{x + 50}\tB\tanchor\t1\t1\nA\tref\t{x + 51}\tB\tanchor\t1\t0\n",
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
    for seq_id, (first, last) in expected.items():
        row = hits[seq_id]
        assert (int(row["reference_first"]), int(row["reference_last"])) == (first, last)
        assert (int(row["reference_pos_first"]), int(row["reference_pos_last"])) == (
            first + 50,
            last + 50,
        )
        method = "profile_locus" if seq_id == "context" else "source_interval"
        assert row["projection_method"] == method
        search_msa = read_msa(out / "search" / "A.fasta", "A", molecule)
        assert dict(zip(search_msa.ids, search_msa.sequences))[seq_id] == (
            "-" * (first - 1) + reference[first - 1 : last] + "-" * (len(reference) - last)
        )
    assert hits["ref"]["strand"] == ("-" if molecule == "dna" else "+")
    assert (
        aligned["ref"]
        == aligned["copy"]
        == "-" * (x - 1) + reference[x - 1 : y] + "-" * (len(reference) - y)
    )
    assert aligned[qc["alignments"]["A"]["reference_row_id"]] == reference
    assert qc["alignments"]["A"]["reference_row_id"] == "ref.coordinate.1"
    with (prepared / "matches.tsv").open(encoding="utf-8", newline="") as handle:
        assert [r["a_id"] for r in csv.DictReader(handle, delimiter="\t")] == ["ref", "copy"]
    with (out / "analysis" / "mi_ab.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows and all(r["x_reference_id"] == "ref" for r in rows)
    assert min(int(r["x_reference_pos"]) for r in rows) == x + 50
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
    if copies > 1:
        database.write_text(source + f">isolated\n{core}\n", encoding="ascii")
        ambiguous = tmp_path / "ambiguous"
        result = runner.invoke(app, ["run", "--config", str(config), "--out", str(ambiguous)])
        assert result.exit_code == 1 and "isolated: ambiguous reference locus" in result.output
        assert not ambiguous.exists()
