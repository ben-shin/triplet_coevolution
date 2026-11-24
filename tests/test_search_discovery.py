"""Offline imports, stable domain identities, search caches and external process gates."""

import csv
import gzip
import shutil
from pathlib import Path

import pytest

from triplet_coevolution.search.discovery import discover_component, validate_search_component
from triplet_coevolution.search.hit_tables import read_hit_table
from triplet_coevolution.search.local_tools import run_tool


def _base(tmp):
    r = "CTAGCCTAGT"
    ref = tmp / "ref.fa.gz"
    db = tmp / "db.fa.gz"
    with gzip.open(ref, "wt") as f:
        f.write(">query.1\n" + r + "\n")
    with gzip.open(db, "wt") as f:
        f.write(
            ">sp|T.2 desc taxonomy cannot establish a genome\n"
            + r
            + "AAC"
            + r
            + "\n>other\n"
            + r
            + "\n>poly\nAAAAAAAAAA\n"
        )
    t = tmp / "blast.tsv"

    def line(query, id, qs, qe, ts, te, ql, tl):
        return f"{query}\t{id}\t100\t10\t0\t0\t{qs}\t{qe}\t{ts}\t{te}\t1e-6\t20\t{ql}\t{tl}\n"

    t.write_text(
        "".join(
            [
                line("query.1", "sp|T.2", 1, 10, 1, 10, 10, 23),
                line("query.1", "sp|T.2", 1, 10, 14, 23, 10, 23),
                line("query.1", "other", 1, 10, 1, 10, 10, 10),
                line("query.1", "poly", 1, 10, 1, 10, 10, 10),
                line("query.1", "absent", 1, 10, 1, 10, 10, 10),
                "bad\trow\n",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "reference": ref,
        "reference_id": "query.1",
        "database": db,
        "molecule": "dna",
        "backend": "import",
        "hit_format": "blast6",
        "hit_table": t,
        "evalue": 0.01,
        "min_entropy_bits": 1.0,
    }, r


def test_import_identity_database_join_sparse_alignment_and_cache(tmp_path):
    cfg, seq = _base(tmp_path)
    cache = tmp_path / "cache"
    wanted = ["sp|T.2/14-23", "absent"]
    result = discover_component(
        "B", cfg, seq, tmp_path / "r1", wanted_ids=iter(wanted), cache_dir=cache
    )
    assert result.manifest["requested_ids"] == sorted(wanted)
    assert result.manifest["requested_ids_without_accepted_hit"] == ["absent"]
    assert result.manifest["selection_settings"]["min_entropy_bits"] == 1
    assert len(result.manifest["input_sha256"]["hit_table"]) == 64
    assert set(result.hits) == {"sp|T.2/1-10", "sp|T.2/14-23", "other"}
    assert result.msa.ids == ("query.1", "sp|T.2/14-23") and result.msa.sequences == (seq, seq)
    audit = list(csv.DictReader((tmp_path / "r1/hit_audit.tsv").open(), delimiter="\t"))
    assert sum(r["status"] == "rejected" for r in audit) == 3
    assert {r["reason"] for r in audit if r["reason"]} >= {
        "target_absent_from_named_database",
        "below_coverage_or_complexity_threshold",
    }
    chosen = next(r for r in audit if r["row_id"] == "sp|T.2/14-23")
    assert (
        chosen["seq_id"] == "sp|T.2"
        and chosen["query_id"] == "query.1"
        and chosen["model_coverage"] == "1.0"
    )
    assert chosen["reference_first"] == "1" and chosen["reference_last"] == "10"
    assert len({r["domain_id"] for r in audit if r["seq_id"] == "sp|T.2"}) == 2
    replay = discover_component(
        "B", cfg, seq, tmp_path / "r2", wanted_ids={"other"}, cache_dir=cache
    )
    assert replay.manifest["search_cache_hit"] and replay.msa.ids == ("query.1", "other")
    assert replay.manifest["requested_ids_sha256"] != result.manifest["requested_ids_sha256"]
    assert list(cache.glob("*/manifest.json"))
    with (cache / result.manifest["raw_key"] / "import.tbl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        discover_component("B", cfg, seq, tmp_path / "r3", cache_dir=cache)


def test_blast_reverse_query_and_subject_strands_and_missing_model_length(tmp_path):
    cfg, seq = _base(tmp_path)
    p = cfg["hit_table"]
    p.write_text("query.1 other 100 10 0 0 10 1 10 1 1e-7 12\n")
    hit = next(read_hit_table(p, "blast6")).hit
    assert (hit.source_strand, hit.query_strand, hit.strand) == ("-", "-", "+")
    assert (hit.first, hit.last, hit.model_first, hit.model_last) == (1, 10, 1, 10)
    assert hit.model_length is None
    out = discover_component("B", {**cfg, "min_model_coverage": 0.8}, seq, tmp_path / "a")
    assert not out.hits and out.manifest["rejection_reasons"] == {"model_coverage_unavailable": 1}
    p.write_text(p.read_text() + "foreign other 100 10 0 0 1 10 1 10 1e-7 12\n")
    with pytest.raises(ValueError, match="multiple queries"):
        discover_component("B", cfg, seq, tmp_path / "b")
    out = discover_component(
        "B", {**cfg, "hit_query": "query.1", "hit_model_length": 10}, seq, tmp_path / "c"
    )
    assert out.hits.keys() == {"other"}


@pytest.mark.parametrize(
    ("fmt", "text", "expected"),
    [
        (
            "hmmsearch_domtblout",
            "q2 P.2 120 hm ALT.3 40 4e-9 51 0 2 3 3e-8 9e-7 38 0 3 36 60 94 55 97 .95 d",
            (60, 94, 3, 36, "+", "hm", 40),
        ),
        (
            "nhmmer_tblout",
            "q2 P.2 h D.2 3 36 94 60 97 55 120 - 7e-8 35 .5 d",
            (60, 94, 3, 36, "-", "h", None),
        ),
        (
            "cmsearch_tblout",
            "q2 P.2 cm9 RF.1 cm 4 31 90 60 - no 1 .5 0 43.5 3e-7 ! d",
            (60, 90, 4, 31, "-", "cm9", None),
        ),
    ],
)
def test_reported_coordinate_formats(tmp_path, fmt, text, expected):
    p = tmp_path / "hits"
    p.write_text("# header\n" + text + "\nwrong\n")
    valid, bad = tuple(read_hit_table(p, fmt))
    h = valid.hit
    assert (
        h.first,
        h.last,
        h.model_first,
        h.model_last,
        h.strand,
        h.query_id,
        h.model_length,
    ) == expected
    assert h.source_accession == "P.2" and bad.reason and bad.hit is None and bad.line == 3


def test_missing_bin_profile_validation_and_bounded_timeout(tmp_path):
    import sys

    cfg, seq = _base(tmp_path)
    profile = tmp_path / "bad.hmm"
    profile.write_text("HMMER3/f\nALPH\n//\n")
    with pytest.raises(ValueError, match="ALPH"):
        discover_component(
            "B",
            {**cfg, "alignment_backend": "hmmalign", "alignment_profile": profile},
            seq,
            tmp_path / "a",
        )
    for bad in ({"cpu": 0}, {"iterations": 4}, {"seed": 0}, {"min_target_coverage": 2}):
        with pytest.raises(ValueError):
            validate_search_component({**cfg, **bad}, "B", tmp_path)
    log = []
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_tool(
            ["python", "-c", "import time;time.sleep(4)"],
            tmp_path,
            {"timeout_seconds": 1},
            [{"name": "python", "path": sys.executable}],
            log,
        )
    assert log[0]["timeout_seconds"] == 1


def test_external_real_jack_profile_covariance_and_cache(tmp_path):
    import sys

    needed = (
        "jackhmmer",
        "hmmsearch",
        "hmmalign",
        "hmmbuild",
        "cmbuild",
        "cmcalibrate",
        "cmsearch",
        "cmalign",
        "psiblast",
        "makeblastdb",
    )
    if any(not shutil.which(tool) for tool in needed):
        pytest.skip("requires local HMMER, Infernal and BLAST+")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    try:
        from search_discovery import exercise

        report = exercise(tmp_path)
    finally:
        sys.path.pop(0)
    assert report["jack"]["rounds"] >= 2 and report["jack"]["converged"]
    assert report["jack_reuse"]["cache_hit"] and report["hmmalign"]["insert_columns"] >= 2


def test_partial_raw_search_retries_without_publishing_incomplete_results(tmp_path, monkeypatch):
    from triplet_coevolution.search import local_tools

    cfg, ref = _base(tmp_path)
    build = local_tools._build

    def interrupted(raw, comp, reference, programs):
        raw.mkdir()
        (raw / "partial.tbl").write_text("unfinished")
        raise OSError("interrupted")

    monkeypatch.setattr(local_tools, "_build", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        discover_component("A", cfg, ref, tmp_path / "failed", cache_dir=tmp_path / "cache")
    assert not (tmp_path / "failed/raw").exists()
    assert not list((tmp_path / "cache").glob("*/manifest.json"))
    monkeypatch.setattr(local_tools, "_build", build)
    retry = discover_component("A", cfg, ref, tmp_path / "retry", cache_dir=tmp_path / "cache")
    assert not retry.manifest["search_cache_hit"] and retry.manifest["accepted_domains"] == 3


def test_portable_repeated_nonoverlapping_domains_and_selected_post_alignment_qc(tmp_path):
    cfg, ref = _base(tmp_path)
    cfg = {k: v for k, v in cfg.items() if k not in ("hit_table", "hit_format", "evalue")}
    cfg.update(backend="reference", min_identity=1.0, min_coverage=0.9)
    result = discover_component("A", cfg, ref, tmp_path / "r", wanted_ids={"sp|T.2/14-23"})
    assert {"sp|T.2/1-10", "sp|T.2/14-23"} <= result.hits.keys()
    assert result.msa.ids == ("query.1", "sp|T.2/14-23")
    assert result.hits["other"].placement is None


def test_search_direction_and_translated_blast_headers_are_checked(tmp_path):
    p = tmp_path / "input.tbl"
    p.write_text("# Program: hmmscan\n")
    with pytest.raises(ValueError, match="search direction"):
        list(read_hit_table(p, "hmmsearch_domtblout"))
    p.write_text("# BLASTX 2.16.0+\n")
    with pytest.raises(ValueError, match="translated BLAST"):
        list(read_hit_table(p, "blast6"))
