import csv
from contextlib import contextmanager

import pytest

from triplet_coevolution.auto_family import choose_reference, infer_molecule
from triplet_coevolution.core.alphabets import encode_sequence
from triplet_coevolution.evidence import EvidenceRow, generate_evidence
from triplet_coevolution.io.stockholm import open_stockholm
from triplet_coevolution.matching.orthology import read_evidence


@contextmanager
def stockholm(tmp_path, body):
    path = tmp_path / "source.sto"
    path.write_text("# STOCKHOLM 1.0\n" + body + "\n//\n", encoding="utf-8")
    with open_stockholm(path) as inventory:
        yield inventory


@pytest.mark.parametrize(
    ("body", "molecule", "source"),
    [
        ("#=GF AC PF99991.3\nrow ACG", "protein", "family_accession"),
        ("#=GF AC RF99991.1\nrow Acgt", "rna", "family_accession"),
        ("#=GF TP DNA\nrow ACG", "dna", "typed_metadata"),
        ("#=GS row MOLECULE RNA\nrow ACT", "rna", "typed_metadata"),
        ("sp|P12345|DEMO ACGT", "protein", "typed_metadata"),
        ("row/4-1 ACGT", "dna", "genomic_alphabet"),
        ("NC_000001.1/1-4 ACGT", "dna", "genomic_alphabet"),
        ("NR_999999.1 ACGT", "rna", "typed_metadata"),
        ("row LEIF", "protein", "protein_specific_residues"),
        ("row " + "acgt" * 5, "dna", "nucleotide_alphabet_heuristic"),
        ("row " + "acgu" * 5, "rna", "nucleotide_alphabet_heuristic"),
        ("row ACGT", None, "ambiguous_alphabet"),
        ("row ACGT" + "-" * 20, None, "ambiguous_alphabet"),
        ("row " + "AACGT" * 5 + "DH", None, "ambiguous_alphabet"),
        ("row ACG", None, "ambiguous_alphabet"),
        ("row --?.", None, "ambiguous_alphabet"),
        ("row " + "ACGTU" * 5, None, "ambiguous_alphabet"),
        ("#=GF AC RF99991\nsp|P12345|DEMO ACGT", None, "conflicting_molecule_metadata"),
        ("#=GF TP RNA\n#=GF TP DNA\nrow ACG", None, "conflicting_molecule_metadata"),
        ("#=GF AC RF99991\nrow LEIF", None, "incompatible_source_alignment"),
        ("#=GF AC PF99991\nrow/4-1 ACGT", None, "conflicting_molecule_metadata"),
    ],
)
def test_molecule_decisions(tmp_path, body, molecule, source):
    with stockholm(tmp_path, body) as inv:
        decision = infer_molecule(inv)
        assert (decision["molecule"], decision["source"]) == (molecule, source)
        if molecule is None:
            assert decision["profile"] is None and not decision["transcribe_t"]


def test_explicit_override_audits_contradiction_and_disables_profile(tmp_path):
    with stockholm(tmp_path, "#=GF AC PF99991\nrow ACGT") as inv:
        decision = infer_molecule(inv, "dna")
        assert decision["molecule"] == "dna" and decision["source"] == "configured"
        assert decision["warnings"] and decision["profile"] is None


def test_rfam_transcription_preserves_reference_source_and_all_feature_tracks(tmp_path):
    body = (
        "#=GF AC RF99991.3\n#=GS ref DR RefSeq; NC_000001.1;\n"
        "#=GS ref/8-5 OX 42\nref/8-5 t.-A\nalt gt-u\n#=GR ref/8-5 PP T..t\n\n"
        "ref/8-5 uc\nalt AC\n#=GR ref/8-5 PP T.\n#=GC RF TttTtT"
    )
    with stockholm(tmp_path, body) as inv:
        ref = inv.row("ref/8-5")
        assert ref.annotations == (("DR", "RefSeq; NC_000001.1;"), ("OX", "42"))
        assert next(inv.find("ref")).annotations == next(inv.rows()).annotations == ref.annotations
        result = infer_molecule(inv)
        assert result["transcribe_t"] and result["profile"] == "RF99991"
        target = tmp_path / "output.sto"
        original = inv.extract(
            [ref.row_id, "alt"],
            target,
            reference_row_id=ref.row_id,
            transcribe_t=True,
            validate_fragment=lambda _, fragment: encode_sequence(fragment, "rna"),
        )
        assert original == "t.-Auc"
        output = target.read_text()
        assert "ref/8-5 u.-Auc\n" in output and "alt gu-uAC\n" in output
        assert "#=GR ref/8-5 PP T..tT." in output and "#=GC RF TttTtT" in output
        assert "ref/8-5 t.-A" in (tmp_path / "source.sto").read_text()


def test_reference_uses_exact_matched_canonical_coverage_then_interval_and_name(tmp_path):
    body = (
        "#=GF AC PF99991\nunmatched LEIFEL\n"
        "best_unanchored L-EI-.\nb/5-7 LE-I..\na/9-11 -LEI..\n"
        "unknown BZ??..\nshort/2-3 Q----F\nmixed --XBZJ"
    )
    with stockholm(tmp_path, body) as inv:
        ids = ["unknown", "mixed", "short/2-3", "best_unanchored", "b/5-7", "a/9-11"]
        for order in (ids, reversed(ids)):
            row, audit = choose_reference(inv, order, "protein")
            assert (row.row_id, audit["canonical_residues"], audit["eligible_rows"]) == (
                "a/9-11",
                3,
                4,
            )
        assert choose_reference(inv, ["unknown", "mixed"], "protein")[0] is None
        assert choose_reference(inv, [], "protein")[0] is None
        assert choose_reference(inv, ids, None)[0] is None


PROFILES = {"A": "PF99991", "B": "PF99992", "C": "RF99991"}


def _evidence(tmp_path, records, profiles=PROFILES, **kwargs):
    mapping = tmp_path / "local.tsv"
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        table = csv.writer(handle, delimiter="\t")
        table.writerow(
            ("component", "identifier", "protein_id", "genome_id", "orthogroup", "source")
        )
        table.writerows((*r, "synthetic") for r in records)
    inv = {c: [EvidenceRow(i) for i in sorted({r[1] for r in records if r[0] == c})] for c in "ABC"}
    return generate_evidence(
        inv, tmp_path / "out", manifest=mapping, automatic_profiles=profiles, **kwargs
    )


def _genomes(n, with_groups=True):
    return [
        (c, f"{c}row{i}", f"{c}mol{i}", f"g{i}", "demo:" + c if with_groups else "")
        for c in "ABC"
        for i in range(n)
    ]


def _read(tmp_path, name):
    with (tmp_path / "out" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_consensus_independent_genomes_role_and_sparse_profile_tiers(tmp_path):
    rows = _genomes(3)
    rows = [(*r[:-1], "" if r[1] == "Crow2" else r[-1]) for r in rows]
    report = _evidence(tmp_path, rows)
    assert report["role_evidence_counts"] == {
        "orthogroup_supported": 2,
        "profile_membership_included": 1,
    }
    assert report["roles"]["C"]["consensus_evidence"] == [
        {"orthogroup": "demo:C", "distinct_genomes": 2}
    ]
    audit = {r["sequence_id"]: r for r in _read(tmp_path, "evidence.tsv")}
    assert audit["Crow2"]["role_status"] == "profile_member"
    assert audit["Crow2"]["confidence"] == "genome_and_family_profile"
    assert audit["Crow0"]["role_status"] == "supported"
    _, meta, _ = read_evidence(tmp_path / "out" / "selected_evidence.tsv")
    assert meta["g2"]["role_evidence"] == "profile_membership_included"
    assert meta["g0"]["role_evidence"] == "orthogroup_supported"


@pytest.mark.parametrize(
    ("alteration", "expected", "reason"),
    [
        ("profile_only", 2, ""),
        ("same_profile", 0, "no_role_anchor"),
        ("shared_orthogroup", 0, "unresolved_family_orthology"),
        ("competing_within_role", 0, "unresolved_family_orthology"),
        ("paralog", 1, "multiple_candidates"),
        ("cross_genome", 0, "multiple_mappings"),
        ("shared_molecule", 1, "shared_molecule_across_roles"),
        ("configured_requires_orthology", 0, "missing_orthology"),
        ("no_profile", 0, "no_role_anchor"),
    ],
)
def test_profile_tier_and_conflict_gates(tmp_path, alteration, expected, reason):
    profiles = dict(PROFILES)
    rows = _genomes(2, with_groups=False)
    options = {}
    if alteration == "same_profile":
        profiles["B"] = profiles["A"]
    if alteration == "shared_orthogroup":
        rows = [(*r[:-1], "demo:shared" if r[0] in "AB" and r[3] == "g0" else "") for r in rows]
    if alteration == "competing_within_role":
        rows = [(*r[:-1], "demo:" + r[3] if r[0] == "A" else "") for r in rows]
    if alteration == "paralog":
        rows.append(("B", "repeat", "other", "g0", ""))
    if alteration == "cross_genome":
        rows.append(("A", "extra", "Amol0", "g1", ""))
    if alteration == "shared_molecule":
        rows = [(c, i, "Amol0" if i == "Brow0" else p, g, og) for c, i, p, g, og in rows]
    if alteration == "configured_requires_orthology":
        options["expected_orthogroups"] = {"A": ["demo:A"]}
    if alteration == "no_profile":
        profiles["C"] = None
    report = _evidence(tmp_path, rows, profiles, **options)
    assert report["counts"]["resolved_groups"] == expected
    groups = _read(tmp_path, "groups.tsv")
    if reason:
        assert any(reason in g["reasons"] for g in groups)
    for g in groups:
        assert g["role_evidence"] == (
            "profile_membership_included" if g["status"] == "resolved" else ""
        )


def test_singleton_metadata_without_profile_does_not_establish_repeated_role(tmp_path):
    report = _evidence(tmp_path, _genomes(1), {})
    assert report["counts"]["resolved_groups"] == 0
    assert all(not d["consensus_evidence"] for d in report["roles"].values())


def test_declared_role_supported_across_genomes_without_profile(tmp_path):
    report = _evidence(tmp_path, _genomes(2), {})
    assert report["role_evidence_counts"] == {"orthogroup_supported": 2}
    assert all(d["source"] == "family_orthogroup_consensus" for d in report["roles"].values())


def test_explicit_orthology_scope_disallows_profile_substitution(tmp_path):
    report = _evidence(tmp_path, _genomes(2), orthology_scope="42")
    assert report["counts"]["resolved_groups"] == 0
