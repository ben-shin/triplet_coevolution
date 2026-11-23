import csv
import json
import sqlite3
from contextlib import closing
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from triplet_coevolution.evidence import EvidenceRow, generate_evidence, uniprot

HEADERS = {"x-uniprot-release": "fixture-v1", "x-uniprot-release-date": "01-Jan-2000"}


def entry(ac, name, proteomes=(), groups=()):
    return {
        "primaryAccession": ac,
        "uniProtkbId": name,
        "organism": {"taxonId": 559292},
        "uniProtKBCrossReferences": [
            *({"database": "Proteomes", "id": p} for p in proteomes),
            *({"database": "OrthoDB", "id": group} for group in groups),
            {"database": "OMA", "id": "MEMBER1"},
            {"database": "eggNOG", "id": "member:demo"},
        ],
    }


def test_rest_fixture_batches_metadata_and_replays_verified_misses(tmp_path, monkeypatch):
    calls = []
    act = entry("P60010", "ACT_YEAST", ["UP000002311", "UP000000001"], ["5132116at2759"])
    pyg = entry("P00549", "KPYK1_YEAST", ["UP000002311"], ["108365at2759"])

    def request(url):
        path = urlsplit(url)
        params = parse_qs(path.query)
        calls.append((path.path, params))
        if path.path == "/proteomes/search":
            assert "genome_assembly" in params["fields"][0]
            return {
                "results": [
                    {"id": "UP000002311", "genomeAssembly": {"assemblyId": "GCA_000146045.2"}}
                ]
            }, HEADERS
        if "cursor" in params:
            return {"results": [pyg]}, HEADERS
        if "sec_acc" in params["query"][0]:
            return {"results": []}, HEADERS
        assert "active:true" in params["query"][0]
        assert "id:KPYK1_YEAST" in params["query"][0] and "accession:P60010" in params["query"][0]
        assert "sequence" not in params["fields"][0]
        return {"results": [act]}, {
            **HEADERS,
            "Link": '<https://rest.uniprot.org/uniprotkb/search?cursor=2>; rel="next"',
        }

    monkeypatch.setattr(uniprot, "_request", request)
    cache = tmp_path / "snapshot.sqlite"
    with closing(uniprot.Resolver(cache, True)) as live:
        result = live.resolve(["P60010", "KPYK1_YEAST", "A0A000ZZZ9"])
        assert len(calls) == 4
        ids = {r["genome_id"] for r in result["P60010"]["mappings"]}
        assert ids == {"GCA_000146045.2", "UP000000001"}
        assert result["P60010"]["mappings"][0]["orthogroups"] == ["OrthoDB:5132116at2759"]
        assert result["A0A000ZZZ9"]["mappings"] == []
        assert result["KPYK1_YEAST"]["mappings"][0]["protein_id"] == "P00549"
        assert live.provenance()["releases"] == [
            {"release": "fixture-v1", "release_date": "01-Jan-2000"}
        ]
    before = cache.read_bytes()
    monkeypatch.setattr(
        uniprot, "_request", lambda url: pytest.fail("offline resolver attempted network")
    )
    with closing(uniprot.Resolver(cache, False)) as offline:
        assert offline.resolve([*result, "P12345"]) == result
        assert offline.provenance()["cache_misses"] == 1
        assert offline.provenance()["cache_hits"] == 3
    assert cache.read_bytes() == before


def test_secondary_accession_alternatives_associated_only_to_their_query(monkeypatch):
    def request(url):
        query = parse_qs(urlsplit(url).query)["query"][0]
        if "sec_acc:P12345" in query:
            return {
                "results": [entry("P60010", "ACT_YEAST"), entry("P00549", "KPYK1_YEAST")]
            }, HEADERS
        return {"results": []}, HEADERS

    monkeypatch.setattr(uniprot, "_request", request)
    with closing(uniprot.Resolver(None, True)) as client:
        result = client.resolve(["P12345", "P45678", "P12345-5"])
        assert [r["protein_id"] for r in result["P12345"]["mappings"]] == ["P60010", "P00549"]
        assert not result["P45678"]["mappings"]
        assert not result["P12345-5"]["mappings"]


def test_http_error_never_poison_cache_and_pagination_checks(monkeypatch, tmp_path):
    cache = tmp_path / "cache.sqlite"
    headers = {
        **HEADERS,
        "Link": '<https://rest.uniprot.org/uniprotkb/search?cursor=2>; rel="next"',
    }

    def partial(url):
        if "cursor=" in url:
            raise ValueError("HTTP 503")
        return {"results": [entry("P60010", "ACT_YEAST")]}, headers

    monkeypatch.setattr(uniprot, "_request", partial)
    with closing(uniprot.Resolver(cache, True)) as resolver:
        with pytest.raises(ValueError, match="503"):
            resolver.resolve(["P60010"])
        assert resolver.db.execute("SELECT COUNT(*) FROM lookups").fetchone()[0] == 0
        monkeypatch.setattr(uniprot, "_request", lambda url: ({}, HEADERS))
        with pytest.raises(ValueError, match="results list"):
            resolver.resolve(["P60010"])
        monkeypatch.setattr(
            uniprot, "_request", lambda url: ({"results": []}, {**HEADERS, "x-total-results": "1"})
        )
        with pytest.raises(ValueError, match="incomplete UniProt pages"):
            resolver.resolve(["P60010"])
        monkeypatch.setattr(
            uniprot,
            "_request",
            lambda url: (
                {"results": []},
                {**HEADERS, "Link": '<https://outside.invalid/>; rel="next"'},
            ),
        )
        with pytest.raises(ValueError, match="pagination link"):
            resolver.resolve(["P60010"])


def test_retries_read_only_request_honors_429_and_rejects_400(monkeypatch):
    class Response(BytesIO):
        headers = HEADERS

    calls, sleeps = [], []

    def request(req, timeout):
        calls.append((req, timeout))
        if len(calls) == 1:
            raise HTTPError(req.full_url, 429, "rate limit", {"Retry-After": "2"}, None)
        return Response(b'{"results":[]}')

    monkeypatch.setattr(uniprot, "urlopen", request)
    monkeypatch.setattr(uniprot.time, "sleep", sleeps.append)
    assert uniprot._request("https://rest.uniprot.org/uniprotkb/search")[0] == {"results": []}
    assert sleeps == [2]
    assert len(calls) == 2

    def bad(req, timeout):
        raise HTTPError(req.full_url, 400, "bad query", Message(), None)

    monkeypatch.setattr(uniprot, "urlopen", bad)
    with pytest.raises(ValueError, match="HTTP 400"):
        uniprot._request("https://rest.uniprot.org/uniprotkb/search")
    assert uniprot._retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", 0) == 0
    with pytest.raises(ValueError, match="resume later"):
        uniprot._retry_delay("300", 0)


def test_invalid_cache_rejected_without_mutating_other_tables(tmp_path):
    unrelated = tmp_path / "data.sqlite"
    with closing(sqlite3.connect(unrelated)) as db:
        db.execute("CREATE TABLE unrelated(a TEXT)")
    before = unrelated.read_bytes()
    with pytest.raises(ValueError, match="cache format"):
        uniprot.Resolver(unrelated, True)
    assert unrelated.read_bytes() == before
    with pytest.raises(ValueError, match="does not exist"):
        uniprot.Resolver(tmp_path / "missing.sqlite", False)


def _annotate(tmp_path, name, live, row, cache):
    source = tmp_path / "bc.tsv"
    source.write_text(
        "component\tidentifier\tprotein_id\tgenome_id\torthogroup\tsource\n"
        "B\tb\tb-protein\tg\tcurated:B\tfixture\n"
        "C\tc\tc-locus\tg\tcurated:C\tfixture\n",
        encoding="utf-8",
    )
    directory = tmp_path / name
    summary = generate_evidence(
        {"A": [row], "B": [EvidenceRow("b")], "C": [EvidenceRow("c")]},
        directory,
        manifest=source,
        uniprot=live,
        cache=cache,
        uniprot_components={"A"},
        expected_orthogroups={"B": ["curated:B"], "C": ["curated:C"]},
        reference_anchors={"A": row.row_id},
        reference_rows={"A": row},
    )
    with (directory / "evidence.tsv").open(encoding="utf-8", newline="") as stream:
        records = [r for r in csv.DictReader(stream, delimiter="\t") if r["component"] == "A"]
    return summary, records


def _service(monkeypatch, components=(), competing=False):
    original = entry("P60010", "ACT_YEAST", [], ["5132116at2759"])
    proteome = {"database": "Proteomes", "id": "UP000002311"}
    for value in components:
        original["uniProtKBCrossReferences"].append(
            {**proteome, "properties": [{"key": "Component", "value": value}]}
        )
    if not components:
        original["uniProtKBCrossReferences"].append(proteome)

    def respond(url):
        parsed = urlsplit(url)
        if parsed.path == "/proteomes/search":
            return {
                "results": [{"id": "UP000002311", "genomeAssembly": {"assemblyId": "g"}}]
            }, HEADERS
        query = parse_qs(parsed.query)["query"][0]
        hits = []
        if "accession:P60010" in query or "id:ACT_YEAST" in query:
            hits.append(original)
        if competing and "id:OLD_YEAST" in query:
            hits.append(entry("P00549", "OLD_YEAST", ["UP000002311"], ["108365at2759"]))
        return {"results": hits}, HEADERS

    monkeypatch.setattr(uniprot, "_request", respond)


@pytest.mark.parametrize(
    ("name", "competing", "expected_source", "is_resolved", "lookup_status"),
    [
        ("OLD_YEAST", False, "reference_orthogroup", 1, "uniprot_not_found"),
        ("ACT_YEAST", False, "reference_orthogroup", 1, "supported"),
        ("OLD_YEAST", True, "ambiguous_reference_orthology", 0, "multiple_mappings"),
    ],
)
def test_alias_outcomes_and_anchor_identities_survive_offline_replay(
    tmp_path, monkeypatch, name, competing, expected_source, is_resolved, lookup_status
):
    _service(monkeypatch, competing=competing)
    cache = tmp_path / "cache.sqlite"
    row = EvidenceRow(name + "/1-8", accession="P60010")
    live, audit = _annotate(tmp_path, "live", True, row, cache)
    assert live["counts"]["resolved_groups"] == is_resolved
    assert live["roles"]["A"]["source"] == expected_source
    attempts = {r["lookup_id"]: r for r in audit}
    assert attempts[name]["role_status"] == lookup_status
    if lookup_status == "uniprot_not_found":
        assert attempts[name]["group_id"] == "" and attempts[name]["selected"] == "0"
        assert attempts["P60010"]["selected"] == "1"
    if lookup_status == "multiple_mappings":
        assert {r["protein_id"] for r in audit} == {"P00549", "P60010"}
    monkeypatch.setattr(uniprot, "_request", lambda url: pytest.fail("offline network call"))
    offline, replayed = _annotate(tmp_path, "offline", False, row, cache)
    assert offline["counts"] == live["counts"]
    assert offline["roles"] == live["roles"]
    assert replayed == audit
    assert (tmp_path / "live" / "selected_evidence.tsv").read_bytes() == (
        tmp_path / "offline" / "selected_evidence.tsv"
    ).read_bytes()


def test_unknown_cache_alias_and_only_negative_lookup_remain_auditable(tmp_path, monkeypatch):
    _service(monkeypatch)
    cache = tmp_path / "cache.sqlite"
    with closing(uniprot.Resolver(cache, True)) as client:
        client.resolve(["P60010", "EMPTY_YEAST"])
    monkeypatch.setattr(uniprot, "_request", lambda url: pytest.fail("offline network call"))
    report, records = _annotate(
        tmp_path, "mixed", False, EvidenceRow("NOTIN_YEAST", "P60010"), cache
    )
    assert {r["role_status"] for r in records} == {"cache_miss", "supported"}
    assert report["counts"]["resolved_groups"] == 1
    assert report["roles"]["A"]["source"] == "reference_orthogroup"
    report, records = _annotate(tmp_path, "none", False, EvidenceRow("EMPTY_YEAST"), cache)
    assert [r["role_status"] for r in records] == ["uniprot_not_found"]
    assert report["roles"]["A"]["source"] == "missing_reference_orthology"
    assert report["counts"]["resolved_groups"] == 0


@pytest.mark.parametrize(
    ("components", "resolved"),
    [
        (["Chromosome I", "Chromosome II", "Chromosome I"], 0),
        (["Chromosome I", "Chromosome I"], 1),
    ],
)
def test_replicon_alternatives_and_duplicate_xrefs_in_new_and_legacy_snapshots(
    tmp_path, monkeypatch, components, resolved
):
    _service(monkeypatch, components=components)
    cache = tmp_path / "cache.sqlite"
    row = EvidenceRow("P60010/1-8")
    live, records = _annotate(tmp_path, "live", True, row, cache)
    assert live["counts"]["resolved_groups"] == resolved
    assert len(records) == len(set(components))
    assert {r["replicon_id"] for r in records} == set(components)
    assert all(r["neighborhood_id"] == "" for r in records)
    assert all(
        len([x for x in json.loads(r["cross_references"]) if x["database"] == "Proteomes"])
        == len(set(components))
        for r in records
    )
    if not resolved:
        assert all(
            r["group_status"] == "ambiguous" and r["role_status"] == "multiple_replicon_annotations"
            for r in records
        )
    monkeypatch.setattr(uniprot, "_request", lambda url: pytest.fail("offline network call"))
    _, replayed = _annotate(tmp_path, "current", False, row, cache)
    assert replayed == records
    # Original snapshots retained a single mapping per UPID with all raw cross-references.
    with closing(sqlite3.connect(cache)) as db:
        payload = json.loads(
            db.execute("SELECT payload FROM lookups WHERE id='P60010'").fetchone()[0]
        )
        payload.pop("mapping_schema")
        old = payload["mappings"][0]
        old.pop("replicon_id")
        old["cross_references"].append(
            next(x for x in old["cross_references"] if x["database"] == "Proteomes")
        )
        payload["mappings"] = [old]
        db.execute("UPDATE lookups SET payload=? WHERE id='P60010'", (json.dumps(payload),))
        db.commit()
    before = cache.read_bytes()
    legacy, replayed = _annotate(tmp_path, "legacy", False, row, cache)
    assert legacy["counts"] == live["counts"]
    assert replayed == records
    assert before == cache.read_bytes()
