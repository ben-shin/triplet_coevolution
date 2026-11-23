import csv
import json
from contextlib import closing
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
