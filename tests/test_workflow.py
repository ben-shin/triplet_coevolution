import csv
import json
import random
import shutil
from pathlib import Path

import pytest
from Bio import Align
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.config import load_config
from triplet_coevolution.io.alignments import read_msa
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.matching.orthology import read_evidence
from triplet_coevolution.matching.triplets import match_groups
from triplet_coevolution.msa.builders import anchored_msa, from_trace, tied_placements
from triplet_coevolution.search.retrieve import retrieve

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "workflow"
runner = CliRunner()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_prepare_reference_order_and_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "prepared"
    result = runner.invoke(
        app, ["prepare", "--config", str(EXAMPLE / "prepare.json"), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    qc = _json(out / "matching_qc.json")
    assert (qc["matched_groups"], qc["excluded_groups"]) == (8, 1)
    assert qc["alignments"]["A"]["matched_mean_gap_fraction"] == pytest.approx(0.2)
    matches = _rows(out / "matches.tsv")
    assert [row["group_id"] for row in qc["groups"]] == [
        row["observation_id"] for row in matches
    ] + ["partial"]
    assert all(row["cohort"] == "demo" for row in matches)
    for name, molecule, ref in [
        ("A", "protein", "prot_ref"),
        ("B", "dna", "dna_ref"),
        ("C", "rna", "rna_ref"),
    ]:
        msa = read_msa(out / f"{name}.fasta", name, molecule)
        assert list(msa.ids) == [ref, *(row[f"{name.lower()}_id"] for row in matches)]
    # The standalone prepare output can be passed directly to the original entry point.
    analysis = runner.invoke(
        app,
        [
            "analyze",
            "--a",
            str(out / "A.fasta"),
            "--b",
            str(out / "B.fasta"),
            "--c",
            str(out / "C.fasta"),
            "--matches",
            str(out / "matches.tsv"),
            "--a-ref",
            "prot_ref",
            "--b-ref",
            "dna_ref",
            "--c-ref",
            "rna_ref",
            "--b-type",
            "dna",
            "--c-type",
            "rna",
            "--out",
            str(tmp_path / "stats"),
        ],
    )
    assert analysis.exit_code == 0, analysis.output


def test_run_example_full_outputs_and_published_provenance(tmp_path):
    out = tmp_path / "results"
    result = runner.invoke(
        app, ["run", "--config", str(EXAMPLE / "run.json"), "--out", str(out), "--no-plots"]
    )
    assert result.exit_code == 0, result.output
    assert ".triplet-in-progress" not in result.output
    workflow = _json(out / "workflow_summary.json")
    assert workflow["matched_groups"] == 8
    assert (out / workflow["report"]).is_file()
    analysis = _json(out / workflow["analysis_summary"])
    for name in ("A", "B", "C", "matches"):
        path = Path(analysis["inputs"][name]["path"])
        assert path.parent == out / "prepared" and path.is_file()
    assert analysis["matched_observations"] == 8 and not analysis["settings"]["plots"]
    assert analysis["advanced"]["candidates"]["tested"] == 800
    top = _rows(out / "analysis" / "threebody_abc.tsv")[0]
    assert [top[f"{c}_original_col"] for c in "abc"] == ["2", "5", "3"]
    assert float(top["co_information_bits"]) == pytest.approx(-1)
    hits = {r["seq_id"]: r for r in _rows(out / "search" / "B_hits.tsv")}
    assert hits["dna03"]["strand"] == "-"
    qc = _json(out / workflow["matching_qc"])
    third = next(row for row in qc["groups"] if row["group_id"] == "obs03")
    assert third["components"]["A"]["selected"] == "prot03"
    assert len(third["components"]["A"]["candidates"]) == 2
    assert (out / "analysis" / "mi_ab.tsv").is_file()
    before = (out / "workflow_summary.json").read_bytes()
    again = runner.invoke(app, ["run", "--config", str(EXAMPLE / "run.json"), "--out", str(out)])
    assert again.exit_code == 1
    assert (out / "workflow_summary.json").read_bytes() == before


def test_ambiguous_missing_and_priority_groups(tmp_path):
    evidence = tmp_path / "groups.tsv"
    evidence.write_text(
        "group_id\tcomponent\tsequence_id\tpriority\n"
        "G\tA\ta1\t1\nG\tA\ta2\t2\nG\tB\tb\t1\nG\tC\tc\t1\n"
        "H\tB\tmissing\t1\n",
        encoding="utf-8",
    )
    groups, _, _ = read_evidence(evidence)
    ids = {"A": {"a1", "a2"}, "B": {"b"}, "C": {"c"}}
    rows, audit = match_groups(ids, groups, "exclude")
    assert rows == [] and "A: ambiguous" in audit["groups"][0]["reasons"]
    assert "B: missing" in audit["groups"][1]["reasons"]
    rows, _ = match_groups(ids, groups, "priority")
    assert [r.a_id for r in rows] == ["a2"]
    for value in ("1", ""):
        evidence.write_text(evidence.read_text().replace("a2\t2", f"a2\t{value}"), encoding="utf-8")
        tied, _, _ = read_evidence(evidence)
        assert not match_groups(ids, tied, "priority")[0]
        if value == "1":
            evidence.write_text(evidence.read_text().replace("a2\t1", "a2\t2"), encoding="utf-8")


@pytest.mark.parametrize(
    "bad,reason",
    [
        ("G\tA\ta\t1\nx\tA\ta\t2", "reused"),
        ("G\tA\ta\tnan", "nonfinite"),
        ("G\tZ\ta\t1", "invalid"),
        ("G\tA\ta", "malformed"),
    ],
)
def test_invalid_evidence(tmp_path, bad, reason):
    evidence = tmp_path / "g.tsv"
    evidence.write_text("group_id\tcomponent\tsequence_id\tpriority\n" + bad, encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        read_evidence(evidence)


def test_anchored_insertions_and_deletions():
    ref = "ACDEFGH"
    msa = anchored_msa("A", "protein", "ref", ref, {"insert": "ACDKEFGH", "delete": "ACEFGH"})
    assert dict(zip(msa.ids, msa.sequences)) == {"ref": ref, "insert": ref, "delete": "AC-EFGH"}


def test_distinct_coordinate_row_for_partial_same_id_hit():
    msa = anchored_msa("A", "protein", "ref", "MACDEG", {"ref": "ACDE", "ref.coordinate": "ACDE"})
    assert dict(zip(msa.ids, msa.sequences)) == {
        "ref.coordinate.1": "MACDEG",
        "ref": "-ACDE-",
        "ref.coordinate": "-ACDE-",
    }


@pytest.mark.parametrize("fragment", ["ATATATAT", "ATCTATAT"])
def test_overlapping_tied_reference_loci(fragment):
    with pytest.raises(ValueError, match="ambiguous reference locus"):
        anchored_msa("A", "dna", "ref", "CATATATATATG", {"fragment": fragment})


def test_portable_trace_retains_reference_region_and_rejects_unresolved_repeat(tmp_path):
    core = "DKPMVYCG"
    ref = core + "AAAA" + core
    path = tmp_path / "db.fa"
    comp = {
        "backend": "reference",
        "molecule": "protein",
        "reference_id": "ref",
        "database": path,
        "min_coverage": 0.3,
        "min_identity": 0.8,
    }
    path.write_text(">variant\nAAAADKPMVYCGT\n", encoding="ascii")
    hits, _ = retrieve(comp, ref)
    hit = hits["variant"]
    assert hit.projection_method == "query_trace"
    assert (hit.placement.first, hit.placement.last) == (9, 20)
    assert hit.placement.row(len(ref)) == "-" * 8 + "AAAA" + core
    path.write_text(f">isolated\n{core}\n", encoding="ascii")
    with pytest.raises(ValueError, match="isolated: ambiguous reference locus"):
        retrieve(comp, ref)


def test_fitting_spans_match_exhaustive_short_tracebacks():
    rng = random.Random(137)
    aligner = Align.PairwiseAligner(
        mode="global",
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-4,
        extend_gap_score=-1,
        wildcard="?",
    )
    aligner.left_deletion_score = aligner.right_deletion_score = 0
    for _ in range(150):
        ref = "".join(rng.choices("ACT?", k=rng.randrange(3, 13)))
        domain = "".join(rng.choices("ACT?", k=rng.randrange(2, 9)))
        alignments = aligner.align(ref, domain)
        first = from_trace(alignments[0], domain, "+")
        if first is None:
            continue
        expected = {
            (int(a.aligned[0][0][0]) + 1, int(a.aligned[0][-1][-1]))
            for a in alignments
            if len(a.aligned[0])
        }
        actual = {
            (p.first, p.last) for p in tied_placements(ref, domain, "dna", first, 0, len(ref))
        }
        assert actual == expected


@pytest.mark.parametrize("additional", [" a_id ", "clade\t clade ", " "])
def test_prepare_rejects_colliding_normalized_headers(tmp_path, additional):
    cfg = load_config(EXAMPLE / "prepare.json", "prepare")
    evidence = tmp_path / "evidence.tsv"
    cfg["evidence"] = str(evidence)
    evidence.write_text("group_id\tcomponent\tsequence_id\t" + additional + "\n", encoding="utf-8")
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(cfg, default=str), encoding="utf-8")
    result = runner.invoke(app, ["prepare", "--config", str(path), "--out", str(tmp_path / "out")])
    assert result.exit_code == 1 and "error:" in result.output
    assert not (tmp_path / "out").exists()


def test_padded_header_round_trip(tmp_path):
    cfg = load_config(EXAMPLE / "prepare.json", "prepare")
    evidence = tmp_path / "evidence.tsv"
    cfg["evidence"] = str(evidence)
    evidence.write_text(
        " group_id \tcomponent \t sequence_id\t cohort \n"
        "obs\tA\tprot01\tdemo\nobs\tB\tdna01\tdemo\nobs\tC\trna01\tdemo\n",
        encoding="utf-8",
    )
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(cfg, default=str), encoding="utf-8")
    out = tmp_path / "out"
    result = runner.invoke(app, ["prepare", "--config", str(path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    table = read_triplet_table(out / "matches.tsv")
    assert table.extra_columns == ("cohort",)
    assert table.annotations == ({"cohort": "demo"},)


def test_profile_missing_alphabet_field_value(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "w")
    path = tmp_path / "w" / "run.json"
    cfg = _json(path)
    a = cfg["components"]["A"]
    a.update(backend="hmmer", profile="broken.hmm")
    del a["min_coverage"], a["min_identity"]
    path.write_text(json.dumps(cfg), encoding="utf-8")
    (path.parent / "broken.hmm").write_text("HMMER3/f\nALPH    \n//\n", encoding="ascii")
    out = tmp_path / "out"
    result = runner.invoke(app, ["run", "--config", str(path), "--out", str(out)])
    assert result.exit_code == 1
    assert "broken.hmm" in result.output and "ALPH" in result.output
    assert not out.exists()


def test_reference_retrieval_boundaries_and_orientation(tmp_path):
    ref = "AGTCCTGCAA"
    database = tmp_path / "targets.fasta"
    database.write_text(">seq\nAATTGCAGGACTCCC\n>noise\nCCCCCCCC\n", encoding="utf-8")
    comp = {
        "reference_id": "reference",
        "backend": "reference",
        "database": database,
        "molecule": "dna",
        "min_identity": 1.0,
        "min_coverage": 1.0,
    }
    hits, seqs = retrieve(comp, ref)
    assert seqs == {"seq": ref}
    assert (hits["seq"].first, hits["seq"].last, hits["seq"].strand) == (3, 12, "-")
    database.write_text(">reference\nCCCCCCCC\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with reference"):
        retrieve(comp, ref)


@pytest.mark.parametrize(
    "key,value",
    [
        ("min_identity", 2.0),
        ("evalue", 0.01),
        ("backend", "other"),
        ("reference_start", True),
        ("extra", 0),
        ("reference", "missing.fasta"),
    ],
)
def test_config_rejects_invalid_component_options(tmp_path, key, value):
    shutil.copytree(EXAMPLE, tmp_path / "w")
    path = tmp_path / "w" / "run.json"
    cfg = _json(path)
    cfg["components"]["A"][key] = value
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path, "run")
