import gzip
from io import StringIO

import pytest
from Bio import AlignIO

from triplet_coevolution.core.coordinates import reference_positions
from triplet_coevolution.io.stockholm import open_stockholm

REF = "ref/8-12"
DOMAIN = "sp|P12345-2|ALT_A/6-9"
REPEAT = "sp|P12345-2|ALT_A/20-24"
FAMILY = f"""# STOCKHOLM 1.0
#=GF ID green
#=GF AC PF00001.3
#=GF DE an example with wrapped rows
#=GF SQ 4
#=GF TN original-tree
#=GF NH (ref,other);
#=GS ref OS organism with spaces
#=GS {REF} AC Q54321.2
#=GS sp|P12345-2|ALT_A AC P12345-2.1
#=GS {REPEAT} DR example; annotation;
{REF}   aC.-
#=GR ref PP 12..
{DOMAIN} --AG
{REPEAT} tt..
unused  ABCD
#=GC RF xx..
#=GC SS_cons HH..

unused  zX?T
{REPEAT} gT-A
{REF}   GTA.
#=GR ref PP 345.
{DOMAIN} c..T
#=GC RF xx.x
#=GC SS_cons H..E
//
"""


@pytest.mark.parametrize("compressed", [False, True])
def test_preserve_exact_rows_columns_metadata_and_reference(tmp_path, compressed):
    path = tmp_path / "source.sto"
    raw = ("\ufeff\n" + FAMILY.replace("unused  ABCD", "unused\tABCD")).encode()
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    selected = tmp_path / "selected.sto"
    with open_stockholm(path, temp_dir=tmp_path) as inv:
        assert (inv.family_id, inv.accession, inv.n_rows, inv.n_columns) == (
            "green",
            "PF00001.3",
            4,
            8,
        )
        assert [r.row_id for r in inv.rows()] == [REF, DOMAIN, REPEAT, "unused"]
        assert inv.row("P12345-2") is None
        assert [r.row_id for r in inv.find("P12345-2")] == [DOMAIN, REPEAT]
        assert [r.row_id for r in inv.find("ALT_A")] == [DOMAIN, REPEAT]
        assert list(inv.find("P12345")) == []
        assert (inv.row(REF).start, inv.row(REF).end, inv.row(REF).accession) == (8, 12, "Q54321.2")
        reference = inv.extract([REPEAT, REF], selected, reference_row_id=REF)
        assert reference == "aC.-GTA."
        assert reference_positions(reference, 8) == (8, 9, None, None, 10, 11, 12, None)
        first = selected.read_bytes()
        inv.extract([REPEAT, REF], selected)
        assert first == selected.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["selected.sto", "source.sto"]
    lines = selected.read_text().splitlines()
    assert "#=GF SQ 2" in lines
    assert "#=GF DE an example with wrapped rows" in lines
    assert not any(line.startswith(("#=GF TN ", "#=GF NH ", "unused ")) for line in lines)
    assert f"{REF} aC.-GTA." in lines
    assert f"{REPEAT} tt..gT-A" in lines
    assert f"#=GS {REF} OS organism with spaces" in lines
    assert f"#=GS {REPEAT} DR example; annotation;" in lines
    assert f"#=GR {REF} PP 12..345." in lines
    original = AlignIO.read(StringIO(FAMILY), "stockholm")
    subset = AlignIO.read(selected, "stockholm")
    assert [r.id for r in subset] == [REPEAT, REF]
    assert subset.column_annotations == original.column_annotations
    by_name = {r.id: r for r in original}
    for actual in subset:
        expected = by_name[actual.id]
        assert (actual.seq, actual.annotations, actual.letter_annotations, actual.dbxrefs) == (
            expected.seq,
            expected.annotations,
            expected.letter_annotations,
            expected.dbxrefs,
        )


@pytest.mark.parametrize("compressed", [False, True])
def test_select_one_record_even_when_accession_is_after_rows(tmp_path, compressed):
    path = tmp_path / "collection"
    extra = "# STOCKHOLM 1.0\nx CD\n#=GF ID blue\n#=GF AC PF00002.5\n//\n"
    raw = (FAMILY + extra).encode()
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    for selector in ("blue", "PF00002.5", "PF00002"):
        with open_stockholm(path, family=selector) as inv:
            assert [(r.row_id, r.start) for r in inv.rows()] == [("x", None)]
            assert inv.n_columns == 2
            assert inv.extract(["x"], tmp_path / "blue.sto", reference_row_id="x") == "CD"
    with pytest.raises(ValueError, match="multiple Stockholm records"), open_stockholm(path):
        pass
    with pytest.raises(ValueError, match="not found"), open_stockholm(path, family="PF00002.6"):
        pass
    path.write_text(extra + extra.replace(".5", ".6"))
    with (
        pytest.raises(ValueError, match="multiple Stockholm records"),
        open_stockholm(path, family="PF00002"),
    ):
        pass


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("x AC\nx GT\n", "duplicate"),
        ("x AC\ny G\n", "unequal sequence fragment"),
        ("x AC\ny GT\n\nx AC\n", "unequal alignment width"),
        ("x/2-6 AC-G\n", "domain span"),
        ("x/5-2 AC-G\n", "forward domain"),
        ("x/0-2 ACG\n", "positive domain"),
        ("x/2-z ACG\n", "malformed domain"),
        ("x ACG\n#=GC RF xx\n", "unequal alignment width"),
        ("x ACG\n#=GR x PP 12\n", "unequal alignment width"),
        ("x ACG\n#=GC RF x x\n", "invalid aligned"),
        ("x ACG\n#=GS missing AC foo\n", "absent row"),
        ("x ACG\n#=GR missing SS xxx\n", "absent or ambiguous"),
        ("x/2-3 AC\nx/4-5 AC\n#=GR x SS xx\n", "absent or ambiguous"),
        ("x AC\n#=GS x AC A\n#=GS x AC B\n", "conflicting"),
        ("x/2-3 AC\n#=GS x AC A\n#=GS x/2-3 AC B\n", "conflicting"),
        ("#=GF SQ 2\nx AC\n", "disagrees"),
        ("#=GF ID x\n", "no sequences"),
    ],
)
def test_invalid_structure_fails_before_extraction(tmp_path, body, message):
    source = tmp_path / "bad.sto"
    source.write_text("# STOCKHOLM 1.0\n" + body + "//\n")
    with pytest.raises(ValueError, match=message), open_stockholm(source, molecule="protein"):
        pass


@pytest.mark.parametrize("molecule", [None, "rna", "dna"])
def test_reverse_strand_interval_preserves_orientation(tmp_path, molecule):
    path = tmp_path / "locus.sto"
    path.write_text("# STOCKHOLM 1.0\nrna/48-45 A.-CGU\n//\n")
    with open_stockholm(path, molecule=molecule) as inv:
        row = inv.row("rna/48-45")
        assert (row.start, row.end) == (48, 45)
        assert (
            inv.extract([row.row_id], tmp_path / "s.sto", reference_row_id=row.row_id) == "A.-CGU"
        )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("x AC\n//\n", "missing # STOCKHOLM"),
        ("# STOCKHOLM 1.0\nx AC\n", "missing //"),
        ("# STOCKHOLM 1.0\nx AC\n//\nx GC\n", "unexpected content"),
    ],
)
def test_record_boundaries(tmp_path, text, message):
    path = tmp_path / "bad.sto"
    path.write_text(text)
    with pytest.raises(ValueError, match=message), open_stockholm(path):
        pass


def test_missing_repeated_changed_input_or_missing_reference_preserve_output(tmp_path):
    source = tmp_path / "x.sto"
    source.write_text(FAMILY)
    output = tmp_path / "result.sto"
    output.write_bytes(b"previous")
    with open_stockholm(source) as inv:
        for rows, ref in [([], None), ([REF, REF], None), (["absent"], None), ([REF], REPEAT)]:
            with pytest.raises(ValueError):
                inv.extract(rows, output, reference_row_id=ref)
        with pytest.raises(ValueError, match="distinct paths"):
            inv.extract([REF], source)
        source.write_text(FAMILY + "\n")
        with pytest.raises(ValueError, match="changed"):
            inv.extract([REF], output)
    assert output.read_bytes() == b"previous"


def test_interleaving_without_blank_block_separator(tmp_path):
    source = tmp_path / "wrapped.sto"
    source.write_text("# STOCKHOLM 1.0\nx AA\ny C-\n#=GC RF .x\nx T\ny c\n#=GC RF x\n//\n")
    with open_stockholm(source) as inv:
        inv.extract(["y", "x"], tmp_path / "reordered.sto")
    assert (tmp_path / "reordered.sto").read_text().splitlines() == [
        "# STOCKHOLM 1.0",
        "y C-c",
        "x AAT",
        "#=GC RF .xx",
        "//",
    ]


def test_gzip_subset_is_reproducible_across_paths(tmp_path):
    path = tmp_path / "input.sto"
    path.write_text(FAMILY)
    with open_stockholm(path) as inv:
        inv.extract([REF], tmp_path / "a.sto.gz")
        inv.extract([REF], tmp_path / "b.sto.gz")
    first = (tmp_path / "a.sto.gz").read_bytes()
    assert first == (tmp_path / "b.sto.gz").read_bytes()
    assert int.from_bytes(first[4:8], "little") == 0
    assert REF.encode() in gzip.decompress(first)
