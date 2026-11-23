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


def _pair(monkeypatch, verified):
    monkeypatch.setattr(
        NucleotideResolver,
        "assembly_pairs",
        lambda self, keys: {
            key: {
                "canonical": GCA,
                "paired_record": {"accession": GCA, "refseq_genbank_are_different": False},
            }
            for key in keys
            if verified and key == GCF
        },
    )


def _third(rows):
    for comp in "ABC":
        rows[comp].append(
            EvidenceRow(
                comp + "third",
                annotations=(
                    ("ASSEMBLY", "GCA_987654323.1"),
                    ("LOCUS", comp + "third"),
                    ("ORTHOGROUP", "demo:" + comp),
                ),
            )
        )


@pytest.mark.parametrize("verified", [True, False])
def test_local_multi_alias_values_collapse_only_when_verified_and_original_locus_is_preserved(
    tmp_path, monkeypatch, verified
):
    _pair(monkeypatch, verified)
    rows = _base()
    rows["C"][0] = EvidenceRow(
        "local-record",
        annotations=(
            ("LOCUS", "known-C"),
            ("ASSEMBLY", GCF),
            ("ASSEMBLY", GCA),
            ("ASSEMBLY", GCF),
            ("REPLICON", "chrIII"),
            ("NEIGHBORHOOD", ""),
            ("ORTHOGROUP", "demo:C"),
        ),
    )
    summary = _generate(tmp_path, rows)
    found = [r for r in _read(tmp_path / "evidence.tsv") if r["sequence_id"] == "local-record"]
    assert summary["counts"]["resolved_groups"] == 1 + verified
    if verified:
        assert len(found) == 1 and found[0]["selected"] == "1"
        assert found[0]["protein_id"] == "known-C" and found[0]["replicon_id"] == "chrIII"
        assert set(json.loads(found[0]["assembly_aliases"])) == {GCF, GCA}
    else:
        assert {r["genome_id"] for r in found} == {GCF, GCA}
        assert all(r["protein_id"] == r["replicon_id"] == r["neighborhood_id"] == "" for r in found)


@pytest.mark.parametrize("wrong", [True, False])
def test_multiple_proteome_constraints_restrict_only_the_assemblies_documented_for_them(
    tmp_path, wrong
):
    rows = _base()
    _third(rows)
    proteomes = ("UP100000001", "UP100000002")
    for i, up in enumerate(proteomes):
        row = rows["A"][i]
        rows["A"][i] = EvidenceRow(row.row_id, annotations=(*row.annotations, ("PROTEOME", up)))
    rows["B"].append(
        EvidenceRow(
            "uncertain",
            annotations=(
                ("PROTEOME", proteomes[0]),
                ("PROTEOME", proteomes[1]),
                ("PROTEOME", proteomes[0]),
                ("ORTHOGROUP", "demo:different" if wrong else "demo:B"),
                ("REPLICON", "chr1"),
                ("REPLICON", "plasmid7"),
                ("NEIGHBORHOOD", "x"),
                ("NEIGHBORHOOD", "y"),
            ),
        )
    )
    summary = _generate(tmp_path, rows, expected_orthogroups={c: ["demo:" + c] for c in "ABC"})
    assert summary["counts"]["resolved_groups"] == (3 if wrong else 1)
    audit = [r for r in _read(tmp_path / "evidence.tsv") if r["sequence_id"] == "uncertain"]
    assert {(r["genome_id"], r["proteome_id"]) for r in audit} == {
        (GCA, proteomes[0]),
        (OTHER, proteomes[1]),
    }
    assert all(r["replicon_id"] == r["neighborhood_id"] == "" for r in audit)
    assert all(r["role_status"] == "orthogroup_mismatch" for r in audit) == wrong
    groups = {r["genome_id"]: r for r in _read(tmp_path / "groups.tsv")}
    for g in (GCA, OTHER):
        assert groups[g]["b_candidates"] == ("1" if wrong else "2")


def test_versionless_constraint_holds_known_versions_without_selecting_one(tmp_path):
    newer = GCA.removesuffix(".1") + ".2"
    rows = _base()
    rows = {
        comp: [
            EvidenceRow(
                r.row_id,
                annotations=tuple((k, newer if v == OTHER else v) for k, v in r.annotations),
            )
            for r in rs
        ]
        for comp, rs in rows.items()
    }
    _third(rows)
    rows["B"].append(EvidenceRow("version-unknown", annotations=(("ASSEMBLY", GCA.split(".")[0]),)))
    summary = _generate(tmp_path, rows)
    assert summary["counts"]["resolved_groups"] == 1
    audit = [r for r in _read(tmp_path / "evidence.tsv") if r["sequence_id"] == "version-unknown"]
    assert {r["genome_id"] for r in audit} == {GCA, newer}
    assert {r["assembly_id"] for r in audit} == {GCA.split(".")[0]}
    assert all(r["selected"] == "0" for r in audit)


def test_distinct_domains_never_invent_roles_when_family_support_is_absent(tmp_path):
    rows = _base()
    rows = {
        c: [
            EvidenceRow(
                r.row_id, annotations=tuple((k, v) for k, v in r.annotations if k != "ORTHOGROUP")
            )
            for r in group
        ]
        for c, group in rows.items()
    }
    summary = generate_evidence(
        rows,
        tmp_path,
        automatic=True,
        molecule_types=TYPES,
        uniprot_components={"A"},
        automatic_profiles={},
    )
    assert summary["counts"]["resolved_groups"] == 0
    assert all(r["role_status"] == "no_role_anchor" for r in _read(tmp_path / "evidence.tsv"))


def test_proteome_with_two_documented_assemblies_only_narrows_with_a_specific_build(tmp_path):
    rows = _base()
    _third(rows)
    for i in (0, 1):
        row = rows["A"][i]
        rows["A"][i] = EvidenceRow(
            row.row_id, annotations=(*row.annotations, ("PROTEOME", "UP100000009"))
        )
    rows["B"].append(EvidenceRow("unlocalized", annotations=(("PROTEOME", "UP100000009"),)))
    summary = _generate(tmp_path, rows)
    assert summary["counts"]["resolved_groups"] == 1
    evidence = _read(tmp_path / "evidence.tsv")
    named = [r for r in evidence if r["sequence_id"] == "unlocalized"]
    assert {r["genome_id"] for r in named} == {GCA, OTHER}
    assert all(r["proteome_id"] == "UP100000009" and not r["assembly_id"] for r in named)
    assert all(r["role_status"] == "supported" for r in evidence if r["component"] == "A")
