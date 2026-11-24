import gzip
from contextlib import ExitStack

import pytest

from triplet_coevolution.io.alignments import open_msa


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("fmt", ["fasta", "a3m"])
def test_index_domains_offsets_wrapped_case_and_reverse_reference(tmp_path, compressed, fmt):
    path = tmp_path / "source.unknown"
    if fmt == "a3m":
        records = (
            ">two/15-20 label\nacTg.-CG\n",
            ">two/1-3\nT-.CG\n",
            ">ref/30-25 details\ntT.\n-CaGg\n",
        )
        coordinates, projected, raw_count = (29, None, 28, 26), ("T-CG", "T-CG"), 21
    else:
        records = (">two/15-17 label\naC-G\n", ">two/1-3\nT-GC\n", ">ref/30-28 details\nT.\nCG\n")
        coordinates, projected, raw_count = (30, None, 29, 28), ("T-GC", "T.CG"), 12
    raw = ("\ufeff\n" + "".join(records)).encode()
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    out = tmp_path / ("selection." + fmt + (".gz" if compressed else ""))
    reference = records[2].split()[0][1:]
    with open_msa(path, "B", "dna", fmt.upper(), temp_dir=tmp_path) as inv:
        assert (inv.n_columns, inv.n_rows, inv.raw_symbols) == (4, 3, raw_count)
        assert tuple(inv.ids()) == (records[0].split()[0][1:], "two/1-3", reference)
        assert tuple(inv.ids(sorted_names=True)) == tuple(sorted(inv.ids()))
        assert [(r.start, r.end) for r in inv.find("two")] == [
            (15, 20 if fmt == "a3m" else 17),
            (1, 3),
        ]
        assert not inv.contains("two") and inv.contains("two/1-3")
        with pytest.raises(ValueError, match="missing ids"):
            inv.select(["two"])
        selected = inv.select(["two/1-3", reference], output=out)
        assert selected.msa.sequences == projected
        mapping = selected.reference_map("biological-id", 30, row_id=reference, strand="-")
        assert mapping.reference_id == "biological-id"
        assert mapping.original_positions == (1, 2, 3, 4)
        assert mapping.reference_positions == coordinates
        assert inv.selected_raw_symbols(["two/1-3", reference]) == (13 if fmt == "a3m" else 8)
    with open_msa(out, "B", "dna") as replay:
        assert tuple(replay.ids()) == ("two/1-3", reference)
        restored = replay.select(replay.ids()).reference_map(
            "biological-id", 30, row_id=reference, strand="-"
        )
        assert restored == mapping
    assert {p.name for p in tmp_path.iterdir()} == {path.name, out.name}


def test_native_stockholm_full_columns_and_selected_tracks(tmp_path):
    path = tmp_path / "family.sto.gz"
    body = (
        "\ufeff\n# STOCKHOLM 1.0\n#=GF AC RF00001.1\n#=GF SQ 3\n#=GS x AC Q12345\n"
        "#=GS omitted DE unrelated\nx a.\nref/5-7 UC\nomitted UG\n#=GR x PP 12\n"
        "#=GC RF x.\n\nref/5-7 -A\nomitted AC\nx C-\n#=GR x PP 3.\n#=GC RF .x\n//\n"
    )
    path.write_bytes(gzip.compress(body.encode()))
    out = tmp_path / "panel.sto"
    with open_msa(path, "C", "rna") as inv:
        assert [r.row_id for r in inv.find("Q12345")] == ["x"]
        selected = inv.select(["ref/5-7", "x"], output=out)
        assert selected.msa.sequences == ("UC-A", "A-C-")
        assert selected.reference_map("ref/5-7", 5).reference_positions == (5, 6, None, 7)
    saved = out.read_text()
    assert "omitted" not in saved
    for line in ("#=GC RF x..x", "#=GR x PP 123.", "#=GS x AC Q12345", "#=GF SQ 2", "x a.C-"):
        assert line in saved


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("bad.fa", ">ok\nAC\n>unused\nATZ\n", "invalid seq"),
        ("bad.fa", ">ok\nAC\n>unused\nA\n", "width"),
        ("bad.fa", ">ok\nAC\n>other\nCC\n>other duplicate\nGG\n", "duplicate"),
        ("bad.a3m", ">ok\nAC\n>unused\nArc-Z\n", "invalid seq"),
        ("bad.a3m", ">ok\nAgtC\n>unused\nAgGC\n", "width"),
        ("bad.a3m", ">ok\nAC\n>ss_pred\nH\n", "display track"),
        ("bad.a3m", ">ok\nAC\n>aa_unknown\nHH\n", "display annotation"),
        ("bad.a3m", ">ok\nAC\n>unused\ng.t\n", "nonzero"),
        ("bad.sto", "# STOCKHOLM 1.0\nok AC\nunused ZZ\n//\n", "invalid sequence"),
        ("bad.sto", "# STOCKHOLM 1.0\nok AC\nunused AC\n\nok AG\n//\n", "width"),
        ("bad.sto", "# STOCKHOLM 1.0\nok AC\nunused AC\n", "terminator"),
        ("bad.sto", "# STOCKHOLM 1.0\nok AC\nunused AC\n#=GR unused PP 1\n//\n", "width"),
    ],
)
def test_unselected_malformed_rows_fail_without_temporary_files(tmp_path, filename, data, message):
    source = tmp_path / filename
    source.write_text(data)
    out = tmp_path / "selected"
    out.write_bytes(b"previous-result")
    with (
        pytest.raises(ValueError, match=message),
        open_msa(source, "C", "dna", temp_dir=tmp_path) as inv,
    ):
        inv.select(["ok"], output=out)
    assert out.read_bytes() == b"previous-result"
    assert set(tmp_path.iterdir()) == {source, out}


@pytest.mark.parametrize(
    ("filename", "body"),
    [
        ("one.fa.gz", ">ok\nAC\n"),
        ("one.sto.gz", "# STOCKHOLM 1.0\nok AC\n//\n"),
    ],
)
def test_truncated_gzip_even_after_last_record(tmp_path, filename, body):
    source = tmp_path / filename
    source.write_bytes(gzip.compress(body.encode())[:-3])
    with (
        pytest.raises(ValueError, match="truncated|corrupt"),
        open_msa(source, "A", "dna", temp_dir=tmp_path),
    ):
        pytest.fail("invalid file exposed an inventory")
    assert list(tmp_path.iterdir()) == [source]


def test_a3m_query_display_annotations_are_not_observations(tmp_path):
    source, dest = tmp_path / "input.a3m", tmp_path / "selection.a3m"
    source.write_text(">ss_pred\nHEC\n>ss_conf PSIPRED confidence\n987\n>x\nTAC\n>y\nTgAC\n")
    with open_msa(source, "A", "dna") as inv:
        assert tuple(inv.ids()) == ("x", "y") and inv.raw_symbols == 7
        result = inv.select(["y"], output=dest)
    assert result.provenance["a3m_omitted_query_display_tracks"] == ["ss_conf", "ss_pred"]
    assert dest.read_text() == ">y\nTgAC\n"


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("one.fa", ">ok full title\nAC\n"),
        ("one.sto", "# STOCKHOLM 1.0\nok AC\n//\n"),
    ],
)
def test_atomic_failure_and_immutable_source(tmp_path, monkeypatch, filename, data):
    import os

    source, out = tmp_path / filename, tmp_path / "selected"
    source.write_text(data)
    out.write_bytes(b"old")
    replace = os.replace

    def interrupt_last_replace(src, dest):
        if dest == out:
            raise KeyboardInterrupt()
        return replace(src, dest)

    with open_msa(source, "A", "dna", temp_dir=tmp_path) as inv, ExitStack() as stack:
        patch = stack.enter_context(monkeypatch.context())
        patch.setattr(os, "replace", interrupt_last_replace)
        with pytest.raises(KeyboardInterrupt):
            inv.select(["ok"], output=out)
        assert out.read_bytes() == b"old"
        source.write_text(data + "\n")
        with pytest.raises(ValueError, match="changed"):
            inv.select(["ok"], output=out)
    assert set(tmp_path.iterdir()) == {source, out}
