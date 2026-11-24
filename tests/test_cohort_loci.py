import gzip
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("cohort_loci")


def test_count_distinct_5s_on_different_replicons_and_ignore_exons(tmp_path, parser):
    rows = []
    for acc in ["NC_000001.1", "NC_000002.1"]:
        rows.extend(
            [
                f"{acc}\tPGAP\tgene\t1\t100\t.\t+\t.\tID=gene-t;gene=rrf;locus_tag=t;gene_biotype=rRNA",
                f"{acc}\tPGAP\trRNA\t1\t100\t.\t+\t.\tID=rna-t;Parent=gene-t;product=5S ribosomal RNA;locus_tag=t",
                f"{acc}\tPGAP\texon\t1\t100\t.\t+\t.\tID=exon-t;Parent=rna-t;product=5S ribosomal RNA;locus_tag=t",
            ]
        )
    path = tmp_path / "synthetic.gff.gz"
    path.write_bytes(gzip.compress(("\n".join(rows) + "\n").encode()))
    candidates = parser.annotation_candidates(path)
    assert len(candidates["5S"]) == 2
    assert parser.gene_screen(candidates)[1]["5S"] == "multiple_loci"


def test_single_split_or_pseudogene_is_not_admitted(tmp_path, parser):
    path = tmp_path / "synthetic.gff.gz"
    body = (
        "NC_000001.1\tP\tgene\t1\t90\t.\t+\t.\tID=gene-1;gene=rpoC2;locus_tag=x\n"
        "NC_000001.1\tP\tCDS\t1\t90\t.\t+\t0\tID=cds-1;Parent=gene-1;product=DNA-directed RNA polymerase subunit beta%27;"
        "protein_id=WP_123456.1\n"
        "NC_000001.1\tP\tpseudogene\t201\t410\t.\t-\t.\tID=gene-2;gene=rplE;pseudo=true;locus_tag=y\n"
    )
    path.write_bytes(gzip.compress(body.encode()))
    c = parser.annotation_candidates(path)
    good, bad = parser.gene_screen(c)
    assert not good
    assert "numbered_paralog_or_split_locus" in bad["rpoC"]
    assert "pseudogene" in bad["rplE"]
    assert "rpoB" not in c


def test_projection_preserves_reference_axis_and_counts_insertion(parser):
    ref = "ACGUACGUACGU"
    aligned, detail = parser.project("5S", "ACGUACGUAACGU", ref)
    assert aligned == ref
    assert detail["inserted_query_residues"] == 1
    assert detail["query_coverage"] == pytest.approx(12 / 13)
    assert parser.project("5S", "ACGUNC", ref)[0] is None
