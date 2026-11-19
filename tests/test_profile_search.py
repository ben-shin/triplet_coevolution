"""Exercise installed engine binaries against small, generated local databases."""

import random
import shutil
import subprocess

import pytest
from Bio.Seq import Seq

from triplet_coevolution.search.retrieve import retrieve


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
