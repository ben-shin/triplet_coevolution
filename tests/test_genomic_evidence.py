"""Genome annotation controls for the ordinary folder and evidence paths."""

import csv
import gzip
import json
from contextlib import closing
from urllib.error import HTTPError

import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.evidence import EvidenceRow, generate_evidence, uniprot
from triplet_coevolution.evidence.provider_policy import ProviderPolicy, request_context

ASSEMBLY = "GCF_000000001.1"
ROLES = {"A": "protein", "B": "protein", "C": "rna"}
EXPECTED = {k: ["fixture:" + k] for k in ROLES}


def table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def record(kind, lo, hi, strand, attrs):
    return f"contig.1\tfixture\t{kind}\t{lo}\t{hi}\t.\t{strand}\t{'0' if kind == 'CDS' else '.'}\t{attrs}"


def annotation(tmp_path, features, *, gzip_input=False):
    data = (
        "##gff-version 3\n#!genome-build-accession NCBI_Assembly:"
        + ASSEMBLY
        + "\n"
        + "\n".join(features)
        + "\n##FASTA\n>unused\nACGT\n"
    ).encode()
    path = tmp_path / ("local.gff3.gz" if gzip_input else "local.gff3")
    path.write_bytes(gzip.compress(data) if gzip_input else data)
    return path


def features():
    result = []
    for i in (1, 2):
        offset = (i - 1) * 100
        result.extend(
            [
                record("operon", 1 + offset, 99 + offset, ".", f"ID=system%3A{i}"),
                record(
                    "gene",
                    3 + offset,
                    30 + offset,
                    "+",
                    f"ID=gene{i};locus_tag=A{i};Parent=system%3A{i}",
                ),
                record(
                    "CDS",
                    5 + offset,
                    10 + offset,
                    "+",
                    f"ID=cdsA{i};protein_id=A{i};Parent=gene{i};orthogroup=fixture:A",
                ),
                record(
                    "CDS",
                    15 + offset,
                    20 + offset,
                    "+",
                    f"ID=cdsA{i};protein_id=A{i};Parent=gene{i};orthogroup=fixture:A",
                ),
                record(
                    "CDS",
                    35 + offset,
                    52 + offset,
                    "+",
                    f"ID=B{i};Parent=system%3A{i};orthogroup=fixture:B",
                ),
                record(
                    "tRNA",
                    56 + offset,
                    70 + offset,
                    "-",
                    f"ID=C{i};Parent=system%3A{i};orthogroup=fixture:C",
                ),
            ]
        )
    return result


def evidence(tmp_path, seqs, data, *, source=None, automatic=False):
    rows = {k: [EvidenceRow(name) for name in ids] for k, ids in seqs.items()}
    out = tmp_path / "metadata"
    summary = generate_evidence(
        rows,
        out,
        expected_orthogroups=EXPECTED,
        molecule_types=ROLES,
        uniprot_components=("A", "B"),
        automatic=automatic,
        gff3=[{"path": data, **(source or {})}],
    )
    return (
        summary,
        table(out / "evidence.tsv"),
        table(out / "selected_evidence.tsv"),
        table(out / "groups.tsv"),
    )


def test_folder_offline_gff3_separates_two_systems_without_user_mapping(tmp_path, monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("unexpected remote call")

    monkeypatch.setattr(uniprot, "urlopen", prohibited)
    directory = tmp_path / "input"
    directory.mkdir()
    annotation(directory, features(), gzip_input=True)
    for k, molecule in ROLES.items():
        part = directory / k
        part.mkdir()
        seqs = ["ACDE", "ACDF"] if k != "C" else ["ACGU", "ACGA"]
        (part / "family.sto").write_text(
            "# STOCKHOLM 1.0\n" + "".join(f"{k}{n} {s}\n" for n, s in enumerate(seqs, 1)) + "//\n"
        )
        (part / "component.json").write_text(
            json.dumps({"molecule": molecule, "reference_id": k + "1"})
        )
    (directory / "project.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_orthogroups": EXPECTED,
                "gff3": [{"path": "local.gff3.gz", "release": "synthetic-1"}],
                "providers": {"cache_generation": "release_1"},
            }
        )
    )
    out = tmp_path / "out"
    run = CliRunner().invoke(
        app, ["run", "--input", str(directory), "--offline", "--out", str(out)]
    )
    summary = json.loads((out / "workflow_summary.json").read_text())
    assert summary["counts"]["resolved_groups"] == 2, (run.output, summary)
    assert (out / "report.html").is_file()
    selected = table(out / "metadata" / "selected_evidence.tsv")
    assert {(row["neighborhood_id"], row["sequence_id"]) for row in selected} == {
        ("system:" + str(i), k + str(i)) for i in (1, 2) for k in "ABC"
    }
    detail = table(out / "metadata" / "evidence.tsv")
    assert all(row["assembly_id"] == ASSEMBLY for row in detail)
    a = json.loads(next(r["gff3_annotation"] for r in detail if r["sequence_id"] == "A1"))
    assert a["segments"] == [[5, 10], [15, 20]] and a["gene_loci"] == ["A1"]
    c = next(r for r in detail if r["sequence_id"] == "C2")
    assert c["genomic_strand"] == "-" and c["replicon_id"] == "contig.1"
    audit = json.loads((out / "metadata" / "evidence_summary.json").read_text())
    assert (
        audit["request_policy"]["http_attempts"] == 0
        and audit["gff3_sources"][0]["release"] == "synthetic-1"
    )


def test_window_overlap_and_multiple_genes_do_not_break_ties(tmp_path):
    rows = features()[:6]
    rows = [r.replace(";Parent=system%3A1", "") for r in rows if "\toperon\t" not in r]
    rows.append(record("CDS", 75, 91, "+", "ID=otherA;Alias=A1;orthogroup=fixture:A"))
    gff = annotation(tmp_path, rows)
    windows = tmp_path / "windows.tsv"
    windows.write_text(
        "assembly_id\treplicon_id\tstart\tend\tstrand\tsystem_id\n"
        + f"{ASSEMBLY}\tcontig.1\t1\t99\t.\tone\n"
        + f"{ASSEMBLY}\tcontig.1\t60\t95\t+\ttwo\n"
    )
    summary, audit, selected, groups = evidence(
        tmp_path,
        {"A": ["A1"], "B": ["B1"], "C": ["contig.1/70-56"]},
        gff,
        source={"system_windows": windows},
    )
    assert not selected and summary["counts"]["resolved_groups"] == 0
    assert {g["neighborhood_id"] for g in groups} == {"one", "two"}
    assert {
        json.loads(r["gff3_annotation"])["feature_id"] for r in audit if r["component"] == "A"
    } == {"cdsA1", "otherA"}
    assert next(r for r in audit if r["component"] == "C")["genomic_strand"] == "-"


@pytest.mark.parametrize("qualifier", ["partial=true", "pseudo=true"])
def test_competing_unfit_feature_blocks_resolved_call(tmp_path, qualifier):
    rows = features()[:6] + [
        record("CDS", 75, 91, "+", f"ID=Afrag;{qualifier};Parent=system%3A1;orthogroup=fixture:A")
    ]
    summary, audit, selected, _ = evidence(
        tmp_path, {"A": ["A1", "Afrag"], "B": ["B1"], "C": ["C1"]}, annotation(tmp_path, rows)
    )
    assert not selected and not summary["counts"]["resolved_groups"]
    assert json.loads(next(r for r in audit if r["sequence_id"] == "Afrag")["gff3_annotation"])[
        "limitations"
    ]


@pytest.mark.parametrize(
    "conflict",
    [
        record("CDS", 40, 50, "-", "ID=cdsA1;protein_id=A1"),
        record("CDS", 40, 50, "+", "ID=cdsA1;protein_id=other"),
    ],
)
def test_conflicting_multipart_feature_is_rejected(tmp_path, conflict):
    data = annotation(tmp_path, features()[:6] + [conflict])
    with pytest.raises(ValueError, match="conflicting"):
        evidence(tmp_path, {c: [c + "1"] for c in ROLES}, data)


def test_provider_budget_and_progress_preserve_identifiers(tmp_path, monkeypatch):
    events = []
    attempts = []

    def refused(req, timeout):
        attempts.append(timeout)
        raise HTTPError(req.full_url, 429, "rate limited", {"Retry-After": "0"}, None)

    monkeypatch.setattr(uniprot, "urlopen", refused)
    with request_context(
        {"max_requests": 1, "timeout_seconds": 3, "progress_every": 1}, events.append
    ) as ctx:
        with pytest.raises(ValueError, match="budget exhausted"):
            uniprot._request("https://rest.uniprot.org/uniprotkb/search?query=unpublished_id")
        ctx.advance(3)
    assert (
        attempts == [3]
        and events[0]["http_attempts"] == 1
        and "unpublished_id" not in json.dumps(events)
    )
    with pytest.raises(ValueError, match="cache_generation"):
        ProviderPolicy(cache_generation="../bad")


def test_committed_batch_survives_interrupt_and_reopens_offline(tmp_path, monkeypatch):
    from triplet_coevolution.automatic_inputs import lookup_snapshot, save_lookup_batch

    metadata = tmp_path / "active"
    shared = tmp_path / "generation1"
    resume = tmp_path / "resume"
    metadata.mkdir()
    resume.mkdir()

    def ckpt(filename, keys):
        save_lookup_batch(metadata, shared, filename, keys)
        raise KeyboardInterrupt

    hits = {
        "results": [
            {
                "primaryAccession": "P60010",
                "uniProtkbId": "ACT_YEAST",
                "organism": {"taxonId": 559292},
                "uniProtKBCrossReferences": [],
            }
        ]
    }
    monkeypatch.setattr(uniprot, "_request", lambda url: (hits, {"x-uniprot-release": "fixture-1"}))
    with (
        pytest.raises(KeyboardInterrupt),
        request_context(checkpoint=ckpt),
        closing(uniprot.Resolver(metadata / "uniprot.sqlite", True)) as client,
    ):
        client.resolve(["P60010", "P12345"])
    snapshot = lookup_snapshot(None, shared, resume, "uniprot.sqlite", False)
    monkeypatch.setattr(uniprot, "_request", lambda url: pytest.fail("offline made a request"))
    with closing(uniprot.Resolver(snapshot, False)) as client:
        result = client.resolve(["P60010", "P12345"])
        assert result["P60010"]["mappings"][0]["protein_id"] == "P60010"
        assert "P12345" not in result and client.provenance()["cache_misses"] == 1
