import csv
import json
from contextlib import closing
from http.client import BadStatusLine, IncompleteRead
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from triplet_coevolution.evidence import EvidenceRow, generate_evidence, uniprot
from triplet_coevolution.evidence import nucleotide as nt
from triplet_coevolution.evidence.stockholm import local_placement

ASSEMBLY = "GCA_987654321.1"
REFSEQ = "GCF_987654321.2"
PROFILE = {"A": "PF99991", "B": "PF99992", "C": "RF99993"}
TYPES = {"A": "protein", "B": "protein", "C": "rna"}


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def annotated(*, c=None, genome=ASSEMBLY):
    return {
        "A": [
            EvidenceRow(
                "local-a", annotations=(("ASSEMBLY", genome), ("LOCUS", "local-a"), ("OX", "91"))
            )
        ],
        "B": [
            EvidenceRow(
                "local-b",
                annotations=(
                    ("DR", f"Assembly; {genome};"),
                    ("LOCUS", "local-b"),
                    ("DE", "x OS=Fixture OX=91"),
                ),
            )
        ],
        "C": c
        if c is not None
        else [EvidenceRow("AB123456.2/18-11", annotations=(("ASSEMBLY", genome),))],
    }


def generate(path, inputs, **opts):
    return generate_evidence(
        inputs,
        path,
        automatic=True,
        molecule_types=TYPES,
        uniprot_components={"A", "B"},
        automatic_profiles=PROFILE,
        **opts,
    )


def test_typed_stockholm_embedded_support_without_mapping_or_online(tmp_path, monkeypatch):
    monkeypatch.setattr(nt, "_request_json", lambda url: pytest.fail("unnecessary request"))
    monkeypatch.setattr(uniprot, "_request", lambda url: pytest.fail("unnecessary request"))
    s = generate(tmp_path, annotated(), uniprot=True, nucleotide_online=True)
    assert s["counts"]["resolved_groups"] == 1
    rec = table(tmp_path / "evidence.tsv")
    assert [(r["component"], r["assembly_id"]) for r in rec] == [(c, ASSEMBLY) for c in "ABC"]
    assert (rec[2]["genomic_start"], rec[2]["genomic_end"], rec[2]["genomic_strand"]) == (
        "18",
        "11",
        "-",
    )
    assert rec[1]["organism_name"] == "Fixture"
    assert rec[1]["taxon_id"] == "91"
    assert "profile" in rec[2]["confidence"]
    assert s["providers"]["status"] == "complete"


@pytest.mark.parametrize(
    ("row", "molecule", "query"),
    [
        (EvidenceRow("ref|NC_123456.2|/18-11"), "dna", "NC_123456.2"),
        (EvidenceRow("gb|AB123456.1|/1-8"), "rna", "AB123456.1"),
        (
            EvidenceRow("opaque", accession="URS000000000A", annotations=(("OX", "91"),)),
            "rna",
            "URS000000000A_91",
        ),
        (EvidenceRow("P60010"), "dna", ""),
        (EvidenceRow("NM_012345.2/1-4"), "rna", ""),
        (EvidenceRow("opaque-alias", selectors=("NC_123456.1",)), "rna", ""),
        (EvidenceRow("URS000000000A/1-5", accession="URS000000000A_91"), "rna", "URS000000000A_91"),
    ],
)
def test_nucleotide_identifiers_are_derived_only_from_typed_source_rows(row, molecule, query):
    assert local_placement(row, molecule)[0] == query


def test_taxonomy_without_genome_and_conflicting_stockholm_annotations_are_auditable(tmp_path):
    rows = annotated(
        c=[
            EvidenceRow("local-c", annotations=(("OX", "91"), ("OS", "Fixture"), ("LOCUS", "c"))),
            EvidenceRow(
                "bad", annotations=(("ASSEMBLY", ASSEMBLY), ("ASSEMBLY", REFSEQ), ("LOCUS", "bad"))
            ),
        ]
    )
    summary = generate(tmp_path, rows)
    assert summary["counts"]["resolved_groups"] == 0
    records = {r["sequence_id"]: r for r in table(tmp_path / "evidence.tsv")}
    assert records["local-c"]["taxon_id"] == "91" and records["local-c"]["genome_id"] == ""
    assert records["bad"]["role_status"] == "conflicting_stockholm_annotations"
    assert len(json.loads(records["bad"]["stockholm_annotations"])) == 3


def test_ena_pages_exact_version_interval_and_cache_replay(tmp_path, monkeypatch):
    urls = []

    def request(url):
        urls.append(url)
        assert url.startswith(nt.ENA + "?")
        args = parse_qs(urlparse(url).query)
        assert args["result"] == ["sequence"]
        assert set(args["fields"][0].split(",")) == {
            "accession",
            "sequence_version",
            "assembly_accession",
            "tax_id",
            "base_count",
        }
        assert args["query"] == ['accession="AB123456"']
        return [
            {
                "accession": "AB123456",
                "sequence_version": "2",
                "assembly_accession": ASSEMBLY,
                "tax_id": "91",
                "base_count": "80",
            }
        ]

    monkeypatch.setattr(nt, "_request_json", request)
    cache = tmp_path / "baseline.sqlite"
    rows = annotated(
        c=[
            EvidenceRow("AB123456.2/18-11"),
            EvidenceRow("AB123456.2/101-108"),
            EvidenceRow("AB123456.1/1-8"),
        ]
    )
    s = generate(tmp_path / "fresh", rows, nucleotide_cache=cache, nucleotide_online=True)
    assert len(urls) == 1
    records = {r["sequence_id"]: r for r in table(tmp_path / "fresh" / "evidence.tsv")}
    assert records["AB123456.2/18-11"]["protein_id"] == "AB123456.2/18-11"
    assert records["AB123456.2/101-108"]["role_status"] == "interval_outside_sequence"
    assert records["AB123456.1/1-8"]["role_status"] == "nucleotide_not_found"
    before = cache.read_bytes()
    monkeypatch.setattr(nt, "_request_json", lambda url: pytest.fail("offline request"))
    replay = generate(tmp_path / "replayed", rows, nucleotide_cache=cache)
    assert replay["counts"] == s["counts"]
    assert (tmp_path / "fresh" / "evidence.tsv").read_bytes() == (
        tmp_path / "replayed" / "evidence.tsv"
    ).read_bytes()
    assert cache.read_bytes() == before


@pytest.mark.parametrize(("different", "resolved"), [(False, 1), (True, 0), (None, 0)])
def test_refseq_pair_joins_only_with_version_specific_identical_pair(
    tmp_path, monkeypatch, different, resolved
):
    calls = []

    def request(url):
        calls.append(url)
        if url.startswith(nt.SUMMARY):
            return {
                "result": {
                    "uids": ["23"],
                    "23": {"accessionversion": "NC_123456.2", "taxid": 91, "slen": 1000},
                }
            }
        if "/sequence_assemblies" in url:
            return {"accessions": [REFSEQ]}
        assert "/accession/" + REFSEQ + "/dataset_report" in url
        pair = {"accession": ASSEMBLY}
        if different is not None:
            pair["refseq_genbank_are_different"] = different
        return {"reports": [{"accession": REFSEQ, "assembly_info": {"paired_assembly": pair}}]}

    monkeypatch.setattr(nt, "_request_json", request)
    monkeypatch.setattr(nt.time, "sleep", lambda _: None)
    s = generate(tmp_path, annotated(c=[EvidenceRow("NC_123456.2/1-8")]), nucleotide_online=True)
    assert s["counts"]["resolved_groups"] == resolved
    assert len(calls) == 3
    c = table(tmp_path / "evidence.tsv")[-1]
    assert c["nucleotide_accession"] == "NC_123456.2" and (c["genome_id"] == ASSEMBLY) == bool(
        resolved
    )
    assert json.loads(c["assembly_aliases"]) == ([REFSEQ, ASSEMBLY] if resolved else [])


def test_rnacentral_active_species_enumerates_paged_loci_and_does_not_project_urs_range(
    tmp_path, monkeypatch
):
    def request(url):
        assert url.startswith(nt.RNACENTRAL + "/URS000000000A/genome-locations/91?")
        end = 205 if "page=2" in url else 105
        return {
            "results": [
                {
                    "chromosome": "I",
                    "start": end - 5,
                    "end": end,
                    "strand": -1,
                    "assembly": {"gca_accession": ASSEMBLY},
                }
            ],
            "next": None
            if end == 205
            else nt.RNACENTRAL + "/URS000000000A/genome-locations/91?page=2",
        }

    monkeypatch.setattr(nt, "_request_json", request)
    monkeypatch.setattr(nt.time, "sleep", lambda _: None)
    rows = annotated(c=[EvidenceRow("URS000000000A_91/1-5"), EvidenceRow("URS000000000B")])
    s = generate(tmp_path, rows, nucleotide_online=True)
    assert s["counts"]["resolved_groups"] == 0
    c = [r for r in table(tmp_path / "evidence.tsv") if r["component"] == "C"]
    assert {r["genomic_start"] for r in c if r["protein_id"]} == {"100", "200"}
    assert {r["role_status"] for r in c} == {"multiple_mappings", "rnacentral_taxon_required"}


def test_nucleotide_outage_is_audited_once_for_all_chunks_and_can_retry(tmp_path, monkeypatch):
    calls = []

    def fail(url):
        calls.append(url)
        raise HTTPError(url, 429, "slow down", {"Retry-After": "90"}, None)

    monkeypatch.setattr(nt, "_request_json", fail)
    cache = tmp_path / "failure.sqlite"
    ids = [f"AB{i:06}.1" for i in range(125)]
    with closing(nt.NucleotideResolver(cache, True)) as client:
        first = client.resolve(ids)
        assert {r["status"] for r in first.values()} == {"provider_unavailable"}
        assert client.db.execute("SELECT count(*) FROM annotations").fetchone()[0] == 0
        assert "ENA" in client.failures and len(calls) == 1
    monkeypatch.setattr(nt, "_request_json", lambda url: [])
    with closing(nt.NucleotideResolver(cache, True)) as client:
        assert client.resolve([ids[0]])[ids[0]]["status"] == "nucleotide_not_found"
    with closing(nt.NucleotideResolver(cache, False)) as client:
        assert client.resolve([ids[1]])[ids[1]]["status"] == "nucleotide_cache_miss"


def test_auto_uniprot_outage_preserves_healthy_rows_and_partial_status(tmp_path, monkeypatch):
    calls = []

    def fail(url):
        calls.append(url)
        raise ValueError("simulated service refusal")

    monkeypatch.setattr(uniprot, "_request", fail)
    rows = annotated()
    rows["A"].extend(EvidenceRow(f"P{i:05}") for i in range(150))
    s = generate(tmp_path, rows, uniprot=True, cache=tmp_path / "uni.sqlite")
    assert len(calls) == 1
    assert s["providers"]["status"] == "partial"
    assert s["providers"]["lookup_status_counts"]["provider_unavailable"] == 150
    assert s["counts"]["selected_rows"] == 3
    assert any(
        r["provider_error"] == "simulated service refusal" for r in table(tmp_path / "evidence.tsv")
    )


def test_ordinary_lowercase_local_names_and_selectors_are_not_remote_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(
        uniprot, "_request", lambda url: pytest.fail("unrecognized local identifier")
    )
    rows = annotated()
    rows["A"].append(EvidenceRow("private_actor/1-4", selectors=("P60010",)))
    generate(tmp_path, rows, uniprot=True)
    assert any(
        r["role_status"] == "unrecognized_identifier" for r in table(tmp_path / "evidence.tsv")
    )


def test_unresolved_ncbi_membership_preserves_completed_siblings_and_can_retry(
    tmp_path, monkeypatch
):
    urls = []

    def query(url):
        urls.append(url)
        if url.startswith(nt.SUMMARY):
            return {
                "result": {
                    "uids": ["1", "2"],
                    "1": {"accessionversion": "NC_000001.1", "slen": 500},
                    "2": {"accessionversion": "NC_000002.1", "slen": 70},
                }
            }
        if "/NC_000002.1/" in url:
            raise HTTPError(url, 403, "denied", {}, None)
        return {"accessions": [ASSEMBLY]}

    monkeypatch.setattr(nt, "_request_json", query)
    monkeypatch.setattr(nt.time, "sleep", lambda _: None)
    cache = tmp_path / "batch.sqlite"
    keys = ["NC_000001.1", "NC_000002.1"]
    with closing(nt.NucleotideResolver(cache, True)) as client:
        first = client.resolve(keys)
    assert first[keys[0]]["placements"][0]["assembly_id"] == ASSEMBLY
    assert first[keys[1]]["status"] == "provider_unavailable" and len(urls) == 3
    with closing(nt.NucleotideResolver(cache, False)) as client:
        replay = client.resolve(keys)
    assert replay[keys[0]] == first[keys[0]]
    assert replay[keys[1]]["status"] == "nucleotide_cache_miss"


_URS = "URS000000000A_91"
_LOCATION_URL = nt.RNACENTRAL + "/URS000000000A/genome-locations/91"


def _page(starts, *, following=None, count=None):
    locations = [
        {"assembly": ASSEMBLY, "chromosome": "II", "start": start, "end": start + 20, "strand": "+"}
        for start in starts
    ]
    return {
        "results": locations,
        "next": following,
        **({"count": count} if count is not None else {}),
    }


@pytest.mark.parametrize(
    "pages",
    [
        [_page([10], count=2)],
        [{"count": 1, "results": _page([10])["results"]}],
        [_page([10], following="")],
        [_page([10], count=1, following=_LOCATION_URL + "?page=2")],
        [_page([10], count=2, following=_LOCATION_URL + "?page=2"), _page([80, 95], count=3)],
        [_page([10], count=2, following=_LOCATION_URL + "?page=2"), _page([80])],
        [_page([10], count=2, following=_LOCATION_URL + "?page=2"), _page([10], count=2)],
        [_page([10], following=_LOCATION_URL + "?page=3")],
        [_page([10], following=_LOCATION_URL + "?page=2&page_size=200")],
        [_page([10], following=_LOCATION_URL + "?page=2&chromosome=II")],
        [_page([10], following=_LOCATION_URL + "?page=2"), _page([])],
        [_page([10, 10])],
        [None],
    ],
)
def test_incomplete_rna_location_result_is_neither_selected_nor_cached(
    tmp_path, monkeypatch, pages
):
    responses = iter(pages)
    monkeypatch.setattr(nt, "_request_json", lambda url: next(responses))
    monkeypatch.setattr(nt.time, "sleep", lambda _: None)
    cache = tmp_path / "rna.sqlite"
    s = generate(
        tmp_path / "failed",
        annotated(c=[EvidenceRow(_URS)]),
        nucleotide_cache=cache,
        nucleotide_online=True,
    )
    assert s["providers"]["status"] == "partial" and "RNAcentral" in s["providers"]["failures"]
    assert s["counts"]["resolved_groups"] == 0
    assert any(
        r["role_status"] == "provider_unavailable" for r in table(tmp_path / "failed/evidence.tsv")
    )
    with closing(nt.NucleotideResolver(cache, False)) as client:
        assert client.resolve([_URS])[_URS]["status"] == "nucleotide_cache_miss"


@pytest.mark.parametrize(
    "pages,expected",
    [
        ([_page([], count=0)], []),
        ([_page([])], []),
        (
            [_page([10], count=2, following=_LOCATION_URL + "?page=2"), _page([80], count=2)],
            [10, 80],
        ),
        ([_page([10], following=_LOCATION_URL + "?page=2"), _page([80])], [10, 80]),
    ],
)
def test_complete_rna_locations_replace_legacy_positive_or_negative(
    tmp_path, monkeypatch, pages, expected
):
    cache = tmp_path / "rna.sqlite"
    with closing(nt.NucleotideResolver(cache, True)) as client:
        client.db.execute(
            "INSERT INTO annotations VALUES (?, ?)",
            (
                "RNAcentral:" + _URS,
                json.dumps(
                    {
                        "source": "RNAcentral",
                        "status": "mapped" if not expected else "nucleotide_not_found",
                        "placements": _page([999])["results"] if not expected else [],
                    }
                ),
            ),
        )
        client.db.commit()
    before = cache.read_bytes()
    with closing(nt.NucleotideResolver(cache, False)) as client:
        legacy = client.resolve([_URS])[_URS]
        assert legacy["status"] == "provider_unavailable" and "older" in legacy["provider_error"]
    assert cache.read_bytes() == before
    replies = iter(pages)
    monkeypatch.setattr(nt, "_request_json", lambda url: next(replies))
    monkeypatch.setattr(nt.time, "sleep", lambda _: None)
    with closing(nt.NucleotideResolver(cache, True)) as client:
        verified = client.resolve([_URS])[_URS]
        assert not client.failures
        assert [p["genomic_start"] for p in verified["placements"]] == expected
        assert verified["status"] == ("mapped" if expected else "nucleotide_not_found")
    before = cache.read_bytes()
    with closing(nt.NucleotideResolver(cache, False)) as client:
        assert client.resolve([_URS])[_URS] == verified and not client.failures
    assert cache.read_bytes() == before


@pytest.mark.parametrize(
    "error", [IncompleteRead(b"part", 8), BadStatusLine("incomplete response")]
)
@pytest.mark.parametrize("service", ["ENA", "UniProt"])
def test_http_read_failures_retry_and_preserve_same_batch_or_cached_siblings(
    tmp_path, monkeypatch, error, service
):
    calls = []

    class Response(BytesIO):
        def __init__(self, body=None):
            self.headers = {}
            self.ok = body is not None
            super().__init__(json.dumps(body).encode())

        def read(self, *args):
            if not self.ok:
                raise error
            return super().read(*args)

    def open_url(req, *, timeout):
        calls.append(req.full_url)
        if service == "UniProt" and "sec_acc" not in req.full_url:
            return Response(
                {"results": [{"primaryAccession": "P12345", "uniProtkbId": "FIX_YEAST"}]}
            )
        return Response()

    source = nt if service == "ENA" else uniprot
    monkeypatch.setattr(source, "urlopen", open_url)
    monkeypatch.setattr(source.time, "sleep", lambda _: None)
    cache = tmp_path / "lookup.sqlite"
    if service == "ENA":
        rows = annotated(c=[EvidenceRow("AB123456.1/1-8")])
        summary = generate(tmp_path / "audit", rows, nucleotide_cache=cache, nucleotide_online=True)
        assert len(calls) == 2
        with closing(nt.NucleotideResolver(cache, False)) as replay:
            assert replay.resolve(["AB123456.1"])["AB123456.1"]["status"] == "nucleotide_cache_miss"
    else:
        rows = annotated()
        rows["A"].extend([EvidenceRow("P12345"), EvidenceRow("P23456")])
        summary = generate(tmp_path / "audit", rows, cache=cache, uniprot=True)
        assert len(calls) == 5
        with closing(uniprot.Resolver(cache, False)) as replay:
            stored = replay.resolve(["P12345", "P23456"])
            assert (
                set(stored) == {"P12345"}
                and stored["P12345"]["mappings"][0]["protein_id"] == "P12345"
            )
    assert summary["providers"]["status"] == "partial"
    assert (
        sum(
            r["role_status"] == "provider_unavailable"
            for r in table(tmp_path / "audit/evidence.tsv")
        )
        == 1
    )


def test_short_bounded_read_with_valid_json_and_truncated_content_length(monkeypatch):
    class Response(BytesIO):
        def __init__(self, body):
            super().__init__(body)
            self.headers = {"Content-Length": "60"}

    monkeypatch.setattr(nt, "urlopen", lambda req, timeout: Response(b"[]"))
    monkeypatch.setattr(nt.time, "sleep", lambda _: None)
    with closing(nt.NucleotideResolver(None, True)) as client:
        response = client.resolve(["AB123456"])
        assert response["AB123456"]["status"] == "provider_unavailable"
        assert client.stats["ENA_requests"] == 2


def test_proteome_fetch_failure_keeps_only_fully_verified_primary_results(tmp_path, monkeypatch):
    cache = tmp_path / "primary.sqlite"

    def primary(ac, up):
        return {
            "primaryAccession": ac,
            "uniProtkbId": ac + "_FIX",
            "uniProtKBCrossReferences": [{"database": "Proteomes", "id": p} for p in up],
        }

    def respond(url):
        if "/proteomes/" in url:
            raise BadStatusLine("unreadable proteome response")
        return {
            "results": [
                primary("P12345", ["UP000000001"]),
                primary("P23456", ["UP000000002", "UP000000001"]),
            ]
        }, {}

    monkeypatch.setattr(uniprot, "_request", respond)
    with closing(uniprot.Resolver(cache, True, audit_errors=True)) as client:
        client._store("proteome:UP000000001", [ASSEMBLY, "release", "date"])
        client.db.commit()
        result = client.resolve(["P12345", "P23456"])
        assert result["P12345"]["mappings"][0]["genome_id"] == ASSEMBLY
        assert "provider_error" in result["P23456"]
    with closing(uniprot.Resolver(cache, False)) as client:
        assert set(client.resolve(result)) == {"P12345"}


@pytest.mark.parametrize("service", ["ENA", "UniProt"])
def test_single_read_interruption_recovers_before_circuit_or_negative_cache(
    tmp_path, monkeypatch, service
):
    calls = []

    class Response(BytesIO):
        def __init__(self, body):
            super().__init__(body)
            self.headers = {}

        def read(self, *args):
            if len(calls) == 1:
                raise IncompleteRead(b"part", 20)
            return super().read(*args)

    def open_url(req, **kwargs):
        calls.append(req.full_url)
        return Response(b"[]" if service == "ENA" else b'{"results":[]}')

    source = nt if service == "ENA" else uniprot
    monkeypatch.setattr(source, "urlopen", open_url)
    monkeypatch.setattr(source.time, "sleep", lambda _: None)
    if service == "ENA":
        with closing(nt.NucleotideResolver(tmp_path / "cache", True)) as client:
            result = client.resolve(["AB123456.1"])
            assert result["AB123456.1"]["status"] == "nucleotide_not_found" and not client.failures
    else:
        with closing(uniprot.Resolver(tmp_path / "cache", True, audit_errors=True)) as client:
            result = client.resolve(["TEST_YEAST"])
            assert result["TEST_YEAST"]["mappings"] == [] and not client.failure
    assert len(calls) == 2
