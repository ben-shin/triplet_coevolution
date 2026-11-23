import csv
import json

import pytest

from triplet_coevolution.evidence import EvidenceRow, generate_evidence
from triplet_coevolution.evidence.nucleotide import NucleotideResolver

GCA = "GCA_987654321.1"
GCF = "GCF_987654321.2"
OTHER = "GCA_987654322.1"
PROFILES = {"A": "PF99991", "B": "RF99992", "C": "RF99993"}
TYPES = {"A": "protein", "B": "rna", "C": "rna"}


def _read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _base():
    return {
        c: [
            EvidenceRow(
                f"{c}{i}",
                annotations=(
                    ("ASSEMBLY", genome),
                    ("LOCUS", f"{c}locus{i}"),
                    ("ORTHOGROUP", "demo:" + c),
                ),
            )
            for i, genome in enumerate([GCA, OTHER])
        ]
        for c in "ABC"
    }


def _generate(path, rows, **kwargs):
    return generate_evidence(
        rows,
        path,
        automatic=True,
        molecule_types=TYPES,
        automatic_profiles=PROFILES,
        uniprot_components={"A"},
        **kwargs,
    )


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("opaque", ()),
        ("URS000000000B_91", ()),
        ("conflicting_names", (("LOCUS", "first"), ("LOCUS", "second"))),
    ],
)
@pytest.mark.parametrize("roles", ["profile", "configured"])
def test_incomplete_genome_placement_counts_possible_competitor_locally(
    tmp_path, name, extra, roles
):
    rows = _base()
    if roles == "profile":
        rows = {
            c: [EvidenceRow(r.row_id, annotations=r.annotations[:2]) for r in group]
            for c, group in rows.items()
        }
    rows["B"].append(
        EvidenceRow(
            name,
            annotations=(
                ("ASSEMBLY", GCA),
                ("REPLICON", "chrI"),
                ("OX", "91"),
                ("ORTHOGROUP", "demo:B"),
                *extra,
            ),
        )
    )
    summary = _generate(
        tmp_path,
        rows,
        expected_orthogroups=({c: ["demo:" + c] for c in "ABC"} if roles == "configured" else {}),
    )
    groups = {r["genome_id"]: r for r in _read(tmp_path / "groups.tsv")}
    assert summary["counts"]["resolved_groups"] == 1
    assert groups[GCA]["b_candidates"] == "2" and "B:multiple_candidates" in groups[GCA]["reasons"]
    assert groups[OTHER]["status"] == "resolved"
    audit = next(r for r in _read(tmp_path / "evidence.tsv") if r["sequence_id"] == name)
    assert (audit["assembly_id"], audit["genome_id"], audit["group_id"]) == (GCA, GCA, GCA)
    assert (audit["taxon_id"], audit["replicon_id"], audit["selected"]) == ("91", "chrI", "0")
    assert audit["role_status"] not in {"supported", "profile_member"}


def test_independently_wrong_role_excludes_unresolved_genome_competitor(tmp_path):
    rows = _base()
    rows["B"].append(
        EvidenceRow("opaque", annotations=(("ASSEMBLY", GCA), ("ORTHOGROUP", "demo:other")))
    )
    summary = _generate(tmp_path, rows, expected_orthogroups={c: ["demo:" + c] for c in "ABC"})
    assert summary["counts"]["resolved_groups"] == 2
    row = next(r for r in _read(tmp_path / "evidence.tsv") if r["sequence_id"] == "opaque")
    assert row["group_id"] == GCA and row["role_status"] == "orthogroup_mismatch"


def test_an_opaque_protein_label_with_known_genome_cannot_supply_a_stable_locus(tmp_path):
    rows = _base()
    rows["A"][0] = EvidenceRow("opaque", annotations=(("ASSEMBLY", GCA),))
    summary = _generate(tmp_path, rows)
    assert summary["counts"]["resolved_groups"] == 1
    groups = {r["genome_id"]: r for r in _read(tmp_path / "groups.tsv")}
    assert groups[GCA]["status"] == "incomplete" and groups[GCA]["a_candidates"] == "1"
    row = next(r for r in _read(tmp_path / "evidence.tsv") if r["sequence_id"] == "opaque")
    assert row["group_id"] == GCA and row["protein_id"] == "" and row["selected"] == "0"


def test_partial_genome_competitor_excluded_by_learned_row_orthology(tmp_path):
    rows = _base()
    rows["B"].append(
        EvidenceRow("opaque", annotations=(("ASSEMBLY", GCA), ("ORTHOGROUP", "demo:other")))
    )
    summary = _generate(tmp_path, rows)
    assert summary["counts"]["resolved_groups"] == 2
    assert summary["roles"]["B"]["source"] == "family_orthogroup_consensus"
    row = next(r for r in _read(tmp_path / "evidence.tsv") if r["sequence_id"] == "opaque")
    assert (
        row["role_status"] == "orthogroup_mismatch"
        and row["identity_status"] == "missing_nucleotide_identifier"
    )


@pytest.mark.parametrize(
    ("start", "stop", "strand", "resolved"),
    [
        (100, 120, "+", 0),
        (101, 120, "+", 1),
        (100, 120, "-", 1),
    ],
)
def test_cross_role_physical_locus_with_verified_assembly_alias(
    tmp_path, monkeypatch, start, stop, strand, resolved
):
    rna_ids = ["URS000000000A_91", "URS000000000B_91"]
    placements = {
        rna_ids[0]: (GCF, 100, 120, "+"),
        rna_ids[1]: (GCA, start, stop, strand),
    }

    def resolve(self, keys):
        return {
            key: {
                "status": "mapped",
                "source": "RNAcentral",
                "placements": [
                    {
                        "assembly_id": placements[key][0],
                        "nucleotide_accession": key,
                        "replicon_id": "chrI" if key == rna_ids[0] else "chromosome 1",
                        "taxon_id": "91",
                        "genomic_start": placements[key][1],
                        "genomic_end": placements[key][2],
                        "genomic_strand": placements[key][3],
                    }
                ],
            }
            for key in keys
        }

    monkeypatch.setattr(NucleotideResolver, "resolve", resolve)
    monkeypatch.setattr(
        NucleotideResolver,
        "assembly_pairs",
        lambda self, keys: {
            GCF: {"canonical": GCA, "paired_record": {"refseq_genbank_are_different": False}}
        },
    )
    rows = {"A": [_base()["A"][0]], "B": [EvidenceRow(rna_ids[0])], "C": [EvidenceRow(rna_ids[1])]}
    summary = _generate(tmp_path, rows)
    assert summary["counts"]["resolved_groups"] == resolved
    group = _read(tmp_path / "groups.tsv")[0]
    if not resolved:
        assert (
            group["status"] == "ambiguous"
            and "shared_genomic_locus_across_roles" in group["reasons"]
        )
    evidence = _read(tmp_path / "evidence.tsv")
    b = next(r for r in evidence if r["component"] == "B")
    assert (b["assembly_id"], b["genome_id"], json.loads(b["assembly_aliases"])) == (
        GCF,
        GCA,
        [GCF, GCA],
    )
    assert {r["nucleotide_accession"] for r in evidence if r["component"] != "A"} == set(rna_ids)
