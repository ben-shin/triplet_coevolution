import csv

import pytest

from triplet_coevolution.evidence import EvidenceRow, generate_evidence
from triplet_coevolution.evidence.identifiers import identifiers
from triplet_coevolution.matching.orthology import read_evidence

FIELDS = (
    "component",
    "identifier",
    "protein_id",
    "genome_id",
    "orthogroup",
    "source",
    "taxon_id",
    "assembly_id",
    "proteome_id",
    "neighborhood_id",
    "replicon_id",
)
EXPECTED = {c: [f"curated:{c}"] for c in "ABC"}


def manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        table = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        table.writeheader()
        for comp, key, protein, genome, og, extras in rows:
            table.writerow(
                dict(
                    component=comp,
                    identifier=key,
                    protein_id=protein,
                    genome_id=genome,
                    orthogroup=og,
                    **{"source": "collection-v2", **extras},
                )
            )
    return path


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def complete(genome="g", locus=""):
    return [
        (c, c + "1", c + ".p1", genome, "curated:" + c, {"neighborhood_id": locus}) for c in "ABC"
    ]


def inventory(rows):
    return {c: [EvidenceRow(i) for i in sorted({r[1] for r in rows if r[0] == c})] for c in "ABC"}


@pytest.mark.parametrize(
    ("name", "ac", "query", "coords"),
    [
        ("sp|P60010|ACT_YEAST/3-20", None, ("P60010",), (3, 20)),
        ("tr|A0A024RBG1|A0A024RBG1_HUMAN/1-7", None, ("A0A024RBG1",), (1, 7)),
        ("P12345-2.4/3-10", None, ("P12345-2",), (3, 10)),
        ("ACT_YEAST/1-2", "P60010.2", ("ACT_YEAST", "P60010"), (1, 2)),
        ("PF00001/1-7", None, (), (1, 7)),
        ("RF00010.2", None, (), (None, None)),
    ],
)
def test_identifiers_keep_source_row_and_isoform(name, ac, query, coords):
    aliases, keys, start, end = identifiers(EvidenceRow(name, ac))
    assert (keys, (start, end)) == (query, coords)
    assert name in aliases


def test_curated_role_supported_selection_and_uncertainty(tmp_path):
    rows = complete() + [
        ("A", "wrong", "another", "g", "curated:other", {}),
        ("C", "rna", "locus:rrna1", "", "curated:C", {"taxon_id": "559292"}),
    ]
    inv = inventory(rows)
    inv["C"] += [EvidenceRow("RF00010/12-27", present=False)]
    out = tmp_path / "out"
    summary = generate_evidence(
        inv,
        out,
        manifest=manifest(tmp_path / "local.tsv", rows),
        expected_orthogroups=EXPECTED,
        uniprot_components={"A", "B"},
        component_names={"C": "RNA subunit"},
    )
    assert summary["counts"] == {
        "input_rows": 6,
        "groups": 1,
        "selected_rows": 3,
        "resolved_groups": 1,
        "audit_rows": 6,
    }
    audit = {r["sequence_id"]: r for r in read(out / "evidence.tsv")}
    assert audit["wrong"]["role_status"] == "orthogroup_mismatch"
    assert audit["wrong"]["confidence"] == "role_excluded"
    assert audit["rna"]["role_status"] == "missing_genome"
    assert audit["rna"]["component_name"] == "RNA subunit"
    assert audit["RF00010/12-27"]["present"] == "0"
    assert audit["B1"]["source"] == "collection-v2"
    assert audit["B1"]["confidence"] == "genome_and_role_supported"
    groups, _, _ = read_evidence(out / "selected_evidence.tsv")
    assert [(c.component, c.sequence_id) for c in groups["g"]] == [(c, c + "1") for c in "ABC"]


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ([("A", "another", "p2", "g", "curated:A", {})], "A:multiple_candidates"),
        ([("A", "A1", "p2", "g", "curated:A", {})], "A:multiple_mappings"),
        ([("A", "A1", "A.p1", "elsewhere", "curated:A", {})], "A:multiple_mappings"),
        ([("A", "A1", "A.p1", "g", "curated:other", {})], "A:orthology_conflict"),
        ([("C", "unknown", "rna", "g", "", {})], "C:missing_orthology"),
        ([("C", "maybe", "rna", "g", "other:some-group", {})], "C:orthology_not_comparable"),
    ],
)
def test_competing_evidence_is_retained_without_cartesian_triplets(tmp_path, extra, reason):
    rows = complete() + extra
    summary = generate_evidence(
        inventory(rows),
        tmp_path / "out",
        manifest=manifest(tmp_path / "local.tsv", rows),
        expected_orthogroups=EXPECTED,
    )
    assert summary["counts"]["resolved_groups"] == 0
    assert len(read(tmp_path / "out" / "evidence.tsv")) == len(rows)
    assert read(tmp_path / "out" / "selected_evidence.tsv") == []
    groups = {r["group_id"]: r for r in read(tmp_path / "out" / "groups.tsv")}
    assert reason in groups["g"]["reasons"]


def test_resolution_uses_role_reference_and_scope_with_separate_genomes(tmp_path):
    rows = [
        (
            c,
            c + "1",
            c + "1",
            "",
            f"OrthoDB:{n}at2759",
            {"assembly_id": "GCA_demo.1", "proteome_id": "UP_demo"},
        )
        for n, c in enumerate("ABC", 1)
    ]
    refs = [
        (c, "ref-" + c, c + "ref", "", f"OrthoDB:{n}at2759", {}) for n, c in enumerate("ABC", 1)
    ]
    refs.append(("A", "ref-A", "Aref", "", "OrthoDB:88at42", {}))
    anchors = {c: "ref-" + c for c in "ABC"}
    source = manifest(tmp_path / "local.tsv", rows + refs)
    summary = generate_evidence(
        inventory(rows),
        tmp_path / "ok",
        manifest=source,
        reference_anchors=anchors,
        orthology_scope="2759",
    )
    assert summary["counts"]["resolved_groups"] == 1
    assert summary["roles"]["A"]["orthogroups"] == ["OrthoDB:1at2759"]
    assert summary["roles"]["B"]["source"] == "reference_orthogroup"
    assert read(tmp_path / "ok" / "groups.tsv")[0]["genome_id"] == "GCA_demo.1"
    summary = generate_evidence(
        inventory(rows), tmp_path / "amb", manifest=source, reference_anchors=anchors
    )
    assert summary["counts"]["resolved_groups"] == 0
    assert summary["roles"]["A"]["source"] == "ambiguous_reference_orthology"


def test_multiple_aligned_domains_and_explicit_domain_override(tmp_path):
    rows = complete()
    rows[0] = ("A", "P60010", "P60010", "g", "curated:A", {})
    source = manifest(tmp_path / "local.tsv", rows)
    inv = inventory(rows)
    inv["A"] = [EvidenceRow("sp|P60010|ACT_YEAST/5-15"), EvidenceRow("sp|P60010|ACT_YEAST/55-64")]
    s = generate_evidence(inv, tmp_path / "amb", manifest=source, expected_orthogroups=EXPECTED)
    assert s["counts"]["resolved_groups"] == 0
    field = read(tmp_path / "amb" / "evidence.tsv")[1]
    assert (field["lookup_id"], field["domain_start"], field["domain_end"]) == (
        "P60010",
        "55",
        "64",
    )
    rows.append(("A", inv["A"][1].row_id, "P60010", "g", "curated:other", {}))
    s = generate_evidence(
        inv, tmp_path / "ok", manifest=manifest(source, rows), expected_orthogroups=EXPECTED
    )
    assert s["counts"]["resolved_groups"] == 1


def test_out_of_panel_reference_rows_retain_exact_domain_and_accession(tmp_path):
    rows = complete()
    refs = [
        ("A", "source-a/9-15", "a-ref", "", "curated:A", {}),
        ("B", "P00549", "b-ref", "", "curated:B", {}),
        ("C", "display-c", "c-ref", "", "curated:C", {}),
    ]
    anchors = {c: "display-" + c.lower() for c in "ABC"}
    reference_rows = {
        "A": EvidenceRow("source-a/9-15", selectors=("display-a",)),
        "B": EvidenceRow("source-b", "P00549.2", selectors=("display-b",)),
        "C": EvidenceRow("source-c", selectors=("display-c",)),
    }
    out = tmp_path / "out"
    result = generate_evidence(
        inventory(rows),
        out,
        manifest=manifest(tmp_path / "m.tsv", rows + refs),
        reference_anchors=anchors,
        reference_rows=reference_rows,
    )
    assert result["counts"]["resolved_groups"] == 1
    assert result["roles"]["A"]["reference"] == "display-a"
    assert result["roles"]["A"]["reference_row_id"] == "source-a/9-15"
    assert len(read(out / "evidence.tsv")) == 3


@pytest.mark.parametrize(
    ("second", "protein", "group", "resolved", "reason"),
    [
        ("alias/1-8", "p", "curated:other", 0, "A:orthology_conflict"),
        ("alias/1-8", "p", "curated:A", 0, "A:multiple_candidates"),
        ("alias/10-17", "p", "curated:other", 1, ""),
        ("alias/8-1", "p", "curated:other", 1, ""),
        ("alias/1-8", "different", "curated:other", 1, ""),
    ],
)
def test_compare_sourced_claims_only_on_known_matching_molecular_intervals(
    tmp_path, second, protein, group, resolved, reason
):
    rows = complete()[1:] + [
        ("A", "one/1-8", "p", "g", "curated:A", {"source": "curator-one"}),
        ("A", second, protein, "g", group, {"source": "curator-two"}),
    ]
    out = tmp_path / "out"
    result = generate_evidence(
        inventory(rows),
        out,
        manifest=manifest(tmp_path / "m.tsv", rows),
        expected_orthogroups=EXPECTED,
    )
    assert result["counts"]["resolved_groups"] == resolved
    audit = [r for r in read(out / "evidence.tsv") if r["component"] == "A"]
    assert {r["source"] for r in audit} == {"curator-one", "curator-two"}
    if reason:
        assert all(reason in r["group_reasons"] for r in audit)
        assert all(r["selected"] == "0" for r in audit)
    if reason == "A:orthology_conflict":
        assert all(r["role_status"] == "orthology_conflict" for r in audit)


def test_unknown_intervals_do_not_merge_rows_and_concordant_mappings_do_not_inflate_counts(
    tmp_path,
):
    rows = complete()[1:] + [
        ("A", "one", "p", "g", "curated:A", {"source": "curator-one"}),
        ("A", "same-row-alias", "p", "g", "curated:A", {"source": "curator-two"}),
    ]
    source = manifest(tmp_path / "m.tsv", rows)
    inv = inventory(rows)
    inv["A"] = [EvidenceRow("one", selectors=("same-row-alias",))]
    result = generate_evidence(
        inv, tmp_path / "one", manifest=source, expected_orthogroups=EXPECTED
    )
    assert result["counts"]["resolved_groups"] == 1
    assert read(tmp_path / "one" / "groups.tsv")[0]["a_candidates"] == "1"
    result = generate_evidence(
        inventory(rows), tmp_path / "two", manifest=source, expected_orthogroups=EXPECTED
    )
    assert result["counts"]["resolved_groups"] == 0
    assert read(tmp_path / "two" / "groups.tsv")[0]["a_candidates"] == "2"


def test_reference_interval_disagreement_is_visible_when_anchor_is_outside_targets(tmp_path):
    rows = complete()[1:] + [
        ("A", "target/1-8", "same", "g", "curated:A", {"source": "target collection"}),
        ("A", "seed/1-8", "same", "g", "curated:other", {"source": "reference collection"}),
    ]
    inv = inventory(rows)
    inv["A"] = [EvidenceRow("target/1-8")]
    out = tmp_path / "out"
    report = generate_evidence(
        inv,
        out,
        manifest=manifest(tmp_path / "map.tsv", rows),
        expected_orthogroups={"B": ["curated:B"], "C": ["curated:C"]},
        reference_anchors={"A": "seed/1-8"},
    )
    assert report["counts"]["resolved_groups"] == 0
    assert report["roles"]["A"]["source"] == "ambiguous_reference_orthology"
    assert report["roles"]["A"]["observed_anchor_orthogroups"] == ["curated:A", "curated:other"]
    assert report["roles"]["A"]["anchor_evidence"][0]["source"] == "reference collection"
    assert read(out / "evidence.tsv")[0]["source"] == "target collection"


def test_taxon_only_and_conflicting_assembly_or_shared_molecule(tmp_path):
    for label, rows, reason in [
        (
            "tax",
            [(c, k, p, "559292", og, {"taxon_id": "559292"}) for c, k, p, g, og, _ in complete()],
            None,
        ),
        (
            "assembly",
            [(c, k, p, g, og, {"assembly_id": c}) for c, k, p, g, og, _ in complete()],
            "conflicting_assemblies",
        ),
        (
            "fusion",
            [(c, k, "shared" if c != "C" else p, g, og, {}) for c, k, p, g, og, _ in complete()],
            "shared_molecule_across_roles",
        ),
    ]:
        out = tmp_path / label
        s = generate_evidence(
            inventory(rows),
            out,
            manifest=manifest(tmp_path / (label + ".tsv"), rows),
            expected_orthogroups=EXPECTED,
        )
        assert s["counts"]["selected_rows"] == 0
        if reason:
            assert reason in read(out / "groups.tsv")[0]["reasons"]
        else:
            assert all(r["group_status"] == "unplaced" for r in read(out / "evidence.tsv"))


@pytest.mark.parametrize(
    ("locus", "labels", "expected"),
    [
        ("region", ("chr1", "chr2", "chr1"), "conflicting_neighborhood_replicons"),
        (
            "region",
            ("plasmid pA", "Chromosome I", "Chromosome I"),
            "conflicting_neighborhood_replicons",
        ),
        ("", ("plasmid pA", "Chromosome I", "Chromosome I"), ""),
        ("region", ("Chromosome I", "Chr 1", "chr1"), ""),
        ("region", ("Chromosome I", "Chr 2", "chrI"), "conflicting_neighborhood_replicons"),
        ("region", ("chr1", "NC_000001.1", "chr1"), "unverified_neighborhood_replicons"),
        (
            "region",
            ("scaffold 1", "scaffold 2", "scaffold 1"),
            "conflicting_neighborhood_replicons",
        ),
        ("region", ("chr1", "Unplaced scaffold", ""), ""),
        ("region", ("plasmid pA", "plasmid pa", ""), "conflicting_neighborhood_replicons"),
    ],
)
def test_named_neighborhood_requires_consistent_localized_replicons(
    tmp_path, locus, labels, expected
):
    rows = [
        (c, i, p, g, og, {**meta, "replicon_id": rep})
        for (c, i, p, g, og, meta), rep in zip(complete(locus=locus), labels, strict=True)
    ]
    # Known orthogroup exclusion on another replicon cannot refute this neighborhood.
    rows.append(
        (
            "A",
            "excluded",
            "wrong",
            "g",
            "curated:other",
            {"neighborhood_id": locus, "replicon_id": "plasmid other"},
        )
    )
    out = tmp_path / "out"
    report = generate_evidence(
        inventory(rows),
        out,
        manifest=manifest(tmp_path / "manifest.tsv", rows),
        expected_orthogroups=EXPECTED,
    )
    assert report["counts"]["resolved_groups"] == int(not expected)
    group = read(out / "groups.tsv")[0]
    assert group["reasons"] == expected
    assert group["status"] == ("ambiguous" if expected else "resolved")
    assert {r["replicon_id"] for r in read(out / "evidence.tsv")} == {*labels, "plasmid other"}


@pytest.mark.parametrize(
    ("identifier", "protein", "resolved"),
    [
        ("DEMO_YEAST", "P23456", 0),
        ("DEMO_YEAST", "P12345", 1),
        ("P12345", "P23456", 1),
        ("DEMO_YEAST/1-20", "P23456", 1),
        ("unrelated_selector", "P23456", 0),
    ],
)
def test_curated_molecule_link_must_cover_the_declared_accession(
    tmp_path, identifier, protein, resolved
):
    rows = complete()[1:] + [("A", identifier, protein, "g", "curated:A", {})]
    inv = inventory(rows)
    row = EvidenceRow("DEMO_YEAST/1-20", accession="P12345", selectors=("unrelated_selector",))
    inv["A"] = [row]
    out = tmp_path / "out"
    report = generate_evidence(
        inv,
        out,
        manifest=manifest(tmp_path / "manifest.tsv", rows),
        expected_orthogroups={"B": ["curated:B"], "C": ["curated:C"]},
        reference_anchors={"A": "my display"},
        reference_rows={"A": row},
    )
    assert report["counts"]["resolved_groups"] == resolved
    assert report["roles"]["A"]["source"] == (
        "reference_orthogroup" if resolved else "unlinked_reference_accession"
    )
    assert report["roles"]["A"]["declared_accessions"] == {"P12345": ["stockholm_accession"]}


def test_missing_neighborhood_member_does_not_add_replicon_contradiction(tmp_path):
    rows = [
        (c, i, p, g, og, {**m, "replicon_id": "chr2" if c == "B" else "chr1"})
        for c, i, p, g, og, m in complete(locus="local")
    ]
    inv = inventory(rows)
    inv["B"] = [EvidenceRow("B1", present=False)]
    generate_evidence(
        inv,
        tmp_path / "out",
        manifest=manifest(tmp_path / "m.tsv", rows),
        expected_orthogroups=EXPECTED,
    )
    group = read(tmp_path / "out" / "groups.tsv")[0]
    assert group["status"] == "incomplete"
    assert group["reasons"] == "B:absent_from_alignment"


def test_unlinked_candidate_cannot_leak_conflicting_orthology_to_verified_same_interval(tmp_path):
    rows = complete()[1:] + [
        ("A", "DEMO_YEAST", "P23456", "g", "curated:wrong", {}),
        ("A", "P23456", "P23456", "g", "curated:A", {}),
        ("A", "seed", "seed", "other", "curated:A", {}),
    ]
    inv = inventory(rows)
    inv["A"] = [EvidenceRow("DEMO_YEAST/1-20", "P12345"), EvidenceRow("P23456/1-20")]
    out = tmp_path / "out"
    report = generate_evidence(
        inv,
        out,
        manifest=manifest(tmp_path / "map.tsv", rows),
        expected_orthogroups={"B": ["curated:B"], "C": ["curated:C"]},
        reference_anchors={"A": "seed"},
    )
    assert report["counts"]["resolved_groups"] == 1
    audit = {r["sequence_id"]: r for r in read(out / "evidence.tsv")}
    assert audit["P23456/1-20"]["role_status"] == "supported"
    assert audit["DEMO_YEAST/1-20"]["role_status"] == "unlinked_accession"


@pytest.mark.parametrize(("extra", "resolved"), [(False, 0), (True, 1)])
def test_distinct_biological_reference_accession_needs_verified_link_even_for_exact_curated_row(
    tmp_path, extra, resolved
):
    rows = complete() + [("A", "P12345/1-20", "local-protein", "g", "curated:A", {})]
    if extra:
        rows.append(("A", "P23456", "local-protein", "g", "curated:different-domain", {}))
    inv = inventory(complete())
    out = tmp_path / "out"
    report = generate_evidence(
        inv,
        out,
        manifest=manifest(tmp_path / "map.tsv", rows),
        expected_orthogroups={"B": ["curated:B"], "C": ["curated:C"]},
        reference_anchors={"A": "P23456"},
        reference_rows={"A": EvidenceRow("P12345/1-20")},
    )
    assert report["counts"]["resolved_groups"] == resolved
    assert report["roles"]["A"]["declared_accessions"] == {
        "P12345": ["row_id"],
        "P23456": ["reference_id"],
    }
    assert report["roles"]["A"]["source"] == (
        "reference_orthogroup" if resolved else "unlinked_reference_accession"
    )


def test_missing_selector_unknown_position_and_no_anchor(tmp_path):
    rows = complete(locus="near-1")
    inv = inventory(rows)
    inv["C"] = [EvidenceRow("C1", present=False)]
    source = manifest(tmp_path / "local.tsv", rows)
    s = generate_evidence(inv, tmp_path / "absent", manifest=source, expected_orthogroups=EXPECTED)
    assert s["counts"]["selected_rows"] == 0
    assert "C:absent_from_alignment" in read(tmp_path / "absent" / "groups.tsv")[0]["reasons"]
    rows += [("A", "unplaced", "p2", "g", "curated:A", {})]
    source = manifest(source, rows)
    s = generate_evidence(
        inventory(rows), tmp_path / "mixed", manifest=source, expected_orthogroups=EXPECTED
    )
    assert s["counts"]["selected_rows"] == 0
    assert "unlocalized_genome_candidates" in read(tmp_path / "mixed" / "groups.tsv")[1]["reasons"]
    s = generate_evidence(inventory(rows), tmp_path / "unanchored", manifest=source)
    assert s["counts"]["selected_rows"] == 0
    assert s["roles"]["C"]["source"] == "no_role_anchor"


def test_duplicate_inventory_and_bad_manifest_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="duplicate A alignment"):
        generate_evidence({"A": [EvidenceRow("x")] * 2, "B": [], "C": []}, tmp_path / "out")
    bad = tmp_path / "bad.tsv"
    bad.write_text("component\tidentifier\tsource\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid curated evidence header"):
        generate_evidence(dict.fromkeys("ABC", ()), tmp_path / "out", manifest=bad)


def test_overlapping_roles_and_large_ties_are_audited_linearly(tmp_path):
    rows = [
        (c, f"{c}.{i}", f"{c}.{i}", "g", "curated:shared" if c != "C" else "curated:C", {})
        for c in "ABC"
        for i in range(150)
    ]
    inv = {c: iter(alignments) for c, alignments in inventory(rows).items()}
    report = generate_evidence(
        inv,
        tmp_path / "out",
        manifest=manifest(tmp_path / "sample.tsv", rows),
        expected_orthogroups={"A": ["curated:shared"], "B": ["curated:shared"], "C": ["curated:C"]},
    )
    assert report["counts"]["audit_rows"] == 450
    assert report["counts"]["selected_rows"] == 0
    group = read(tmp_path / "out" / "groups.tsv")[0]
    assert group["a_candidates"] == group["b_candidates"] == group["c_candidates"] == "150"
    assert "A:overlapping_role_orthology" in group["reasons"]
