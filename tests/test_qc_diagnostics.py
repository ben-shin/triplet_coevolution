"""QC row/site thresholds, deterministic ordering and exact biological coordinates."""

import csv
import hashlib
import json
from itertools import product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.io.metadata import TripletTable
from triplet_coevolution.msa.diagnostics import QCPolicy, inspect_qc, write_qc
from triplet_coevolution.msa.masking import apply_triplet_masks, column_mask
from triplet_coevolution.msa.qc import triplet_qc
from triplet_coevolution.msa.support import MissingRowSupport, site_masks


def _input(columns, molecules=("protein", "dna", "rna"), metadata=False):
    components = tuple(
        ComponentMSA(c, mol, tuple(c + str(i) for i in range(len(seqs))), tuple(seqs))
        for c, mol, seqs in zip("ABC", molecules, columns, strict=True)
    )
    matches = tuple(
        TripletMatch(f"id{i}", *ids)
        for i, ids in enumerate(zip(*(part.ids for part in components)))
    )
    msa = MatchedTripletMSA(*components, matches)
    maps = tuple(
        build_reference_map(c, c.ids[0], start=40, strand="-" if c.molecule.value == "dna" else "+")
        for c in components
    )
    table = TripletTable(
        matches,
        ("taxon_id", "stratum") if metadata else (),
        tuple(
            {"taxon_id": ("tax1", "tax2", "", "tax1")[i % 4], "stratum": "s" + str(i % 2)}
            for i in range(len(matches))
        )
        if metadata
        else (),
    )
    return msa, maps, table


def _mixed():
    return _input(
        (
            ("AK-K", "AK.K", "A??K", "A--K"),
            ("A-TC", "A.TC", "GTT?", "AT-?"),
            ("U?GU", "U?GU", "C-?G", "A-?U"),
        ),
        metadata=True,
    )


def test_default_mask_and_measured_joint_support_match_enumeration(tmp_path):
    msa, maps, table = _mixed()
    report = inspect_qc(msa, maps, table)
    assert not report.blockers and report.row_indices == tuple(range(4))
    expected = triplet_qc(msa)
    for raw, measure, mask in zip(msa.components, expected.components, report.masks):
        np.testing.assert_array_equal(mask, column_mask(measure))
        retained = getattr(report.filtered.msa, raw.name.lower())
        assert retained.sequences == tuple(
            "".join(seq[i] for i in np.flatnonzero(mask)) for seq in raw.sequences
        )
    for name, desc in report.summary["joint_support"].items():
        cols = [getattr(report.filtered.msa, c.lower()) for c in name]
        ncan = ngap = 0
        for r in range(4):
            for sites in product(*(range(c.n_sites) for c in cols)):
                vals = [(c.matrix[r, i], len(CANONICAL[c.molecule])) for c, i in zip(cols, sites)]
                ncan += all(0 <= x < q for x, q in vals)
                ngap += all(x == q for x, q in vals)
        assert desc["joint_canonical_cells"] == ncan and desc["joint_gap_cells"] == ngap
    assert report.summary["exact_encoded_duplicates"]["ABC"]["largest_group"] == 2
    assert report.summary["metadata"]["taxon"]["retained"]["missing"] == 1
    assert report.summary["metadata"]["clade"]["status"] == "unavailable"
    files = write_qc(tmp_path, report)
    persisted = json.loads((tmp_path / files["summary"]).read_text())
    assert persisted["retained_observations"] == 4
    with (tmp_path / files["sites"]).open(newline="") as stream:
        sites = list(csv.DictReader(stream, delimiter="\t"))
    site = next(s for s in sites if s["component"] == "A" and s["source_column"] == "2")
    assert (int(site["n_gap"]), int(site["n_missing"]), int(site["n_canonical"])) == (1, 1, 2)
    assert json.loads(site["canonical_state_counts"]) == {"K": 2}


def test_fixed_point_can_recover_columns_and_reports_nonconvergence():
    msa, maps, table = _input((("A--", "A-C", "A-C", "-C-"), ("A", "A", "A", "-"), ("U",) * 4))
    opts = {"min_coverage": 0.6, "max_gap_fraction": 1, "max_missing_fraction": 1}
    cfg = {"rows": {"B": {"min_canonical": 1}}}
    one = inspect_qc(msa, maps, table, column_options=opts, policy=QCPolicy.from_config(cfg))
    repeated = QCPolicy.from_config({**cfg, "iteration": "fixed_point"})
    fixed = inspect_qc(msa, maps, table, column_options=opts, policy=repeated)
    assert one.row_indices == fixed.row_indices == (0, 1, 2)
    assert one.maps[0].original_positions == (1,) and fixed.maps[0].original_positions == (1, 3)
    assert fixed.summary["passes"][1]["column_changes"]["A"]["reentered"] == [3]
    assert fixed.summary["row_exclusion_reason_counts"] == {"B.min_canonical": 1}
    assert (
        next(r for r in fixed.rows if r["observation_id"] == "id3" and r["component"] == "B")[
            "exclusion_pass"
        ]
        == 1
    )
    guard = inspect_qc(
        msa,
        maps,
        table,
        column_options=opts,
        policy=QCPolicy.from_config({**cfg, "iteration": "fixed_point", "max_passes": 1}),
    )
    with pytest.raises(MissingRowSupport, match="did not converge"):
        guard.require_scoring()
    again = inspect_qc(msa, maps, table, column_options=opts, policy=repeated)
    assert again.summary == fixed.summary


def test_rows_first_versus_columns_first_and_empty_attrition():
    msa, maps, table = _input((("A--", "A--", "A-C", "A--"), ("T",) * 4, ("U",) * 4))
    cfg = {"rows": {"A": {"min_coverage": 0.75}}}
    col = inspect_qc(msa, maps, table, policy=QCPolicy.from_config(cfg))
    row = inspect_qc(msa, maps, table, policy=QCPolicy.from_config({**cfg, "order": "rows_first"}))
    assert col.row_indices == tuple(range(4)) and not col.blockers
    assert not row.row_indices and not row.filtered and all(not s["retained"] for s in row.sites)
    assert "QC excluded every" in row.blockers[0]
    assert row.summary["row_exclusion_reason_counts"] == {"A.min_coverage": 4}
    assert all(s["final_rows"] == 0 and s["entropy_bits"] is None for s in row.sites)
    json.dumps(row.summary, allow_nan=False)


def test_reference_and_original_panels_preserve_reverse_source_maps(tmp_path):
    msa, maps, table = _mixed()
    masks = ([True, False, True, True], [False, False, True, True], [True, False, False, True])
    narrow = apply_triplet_masks(msa, *masks)
    maps = tuple(subset_reference_map(m, np.flatnonzero(mask)) for m, mask in zip(maps, masks))
    original = tmp_path / "original.tsv"
    original.write_text("component\toriginal_column\nA\t4\nB\t3\nC\t4\n", encoding="utf-8")
    mapped = tmp_path / "reference.tsv"
    mapped.write_text(
        "component\treference_id\treference_position\nA\tA0\t42\nB\tB0\t39\nC\tC0\t43\n",
        encoding="utf-8",
    )
    file1 = site_masks(original, narrow.msa.components, maps)
    file2 = site_masks(mapped, narrow.msa.components, maps)
    for m1, m2 in zip(file1, file2):
        np.testing.assert_array_equal(m1, m2)
    qc = inspect_qc(narrow.msa, maps, table, panel_masks=file2)
    assert qc.summary["panel_sha256"] == hashlib.sha256(original.read_text().encode()).hexdigest()
    assert [m.original_positions for m in qc.maps] == [(4,), (3,), (4,)]
    assert qc.maps[1].reference_positions == (39,) and qc.maps[1].strand == "-"
    bad = mapped.read_text().replace("B0", "another_ref")
    mapped.write_text(bad)
    with pytest.raises(ValueError, match="out-of-range"):
        site_masks(mapped, narrow.msa.components, maps)


def test_entropy_length_flags_and_molecule_orders():
    msa, maps, table = _input(
        (
            ("TTTT", "T-T?", "T?T?", "TTTT"),
            ("U-CU", "U-GU", "U-GU", "U-GU"),
            ("KKKK", "KRKR", "KKKK", "KRKR"),
        ),
        molecules=("dna", "rna", "protein"),
        metadata=True,
    )
    cfg = {
        "rows": {"A": {"min_length_ratio": 1.0}, "C": {"min_entropy_bits": 1}},
        "sites": {"B": {"min_canonical": 2, "min_states": 1}},
        "clade_column": "stratum",
    }
    result = inspect_qc(msa, maps, table, policy=QCPolicy.from_config(cfg))
    assert result.row_indices == (3,)
    assert result.summary["metadata"]["clade"]["retained"]["counts"] == {"s1": 1}
    assert result.summary["row_exclusion_reason_counts"] == {
        "A.min_length_ratio": 2,
        "C.min_entropy_bits": 2,
    }
    csite = next(s for s in result.sites if s["component"] == "C")
    assert csite["entropy_bits"] == 0 and len(csite["canonical_alphabet"]) == 20
    assert {len(s["canonical_alphabet"]) for s in result.sites if s["component"] in "AB"} == {4}
    assert result.summary["retained_row_flags"]["A.single_canonical_state"] == 1
    with pytest.raises(ValueError, match="unknown QC taxon"):
        inspect_qc(msa, maps, table, policy=QCPolicy(taxon_column="nosuch"))
    with pytest.raises(ValueError, match="invalid symbol"):
        _input((("U",), ("C",), ("K",)), molecules=("dna", "rna", "protein"))


def test_require_reference_and_explicit_undefined_coverage():
    msa, maps, table = _input((("--", "GG"), ("T", "T"), ("U", "U")))
    # All A columns meet the inclusive v1 support threshold; its reference row is all gaps.
    base = inspect_qc(msa, maps, table)
    with pytest.raises(MissingRowSupport, match="no canonical residues"):
        base.require_scoring()
    strict = inspect_qc(
        msa,
        maps,
        table,
        policy=QCPolicy.from_config({"rows": {"A": {"min_reference_coverage": 0.1}}}),
    )
    assert not strict.row_indices
    assert all(r["reference_coverage"] is None for r in strict.rows if r["component"] == "A")
    assert strict.summary["row_exclusion_reason_counts"] == {"A.min_reference_coverage": 2}


@pytest.mark.parametrize(
    "setting",
    [
        {"rows": {"D": {}}},
        {"sites": {"A": {"min_states": True}}},
        {"rows": {"A": {"min_length_ratio": 2, "max_length_ratio": 1}}},
        {"sites": {"A": {"min_coverage": 1.1}}},
        {"order": "random"},
        {"max_passes": 0},
    ],
)
def test_malformed_policy_rejected(setting):
    with pytest.raises(ValueError):
        QCPolicy.from_config(setting)


def test_policy_versions_and_duplicate_json_keys(tmp_path):
    policy = tmp_path / "rules.json"
    for content in ('{"schema_version":true}', '{"schema_version":2}', '{"rows":{},"rows":{}}'):
        policy.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            QCPolicy.from_config(policy)
    policy.write_text('{"schema_version":1}', encoding="utf-8")
    assert QCPolicy.from_config(policy).as_dict() == QCPolicy().as_dict()
