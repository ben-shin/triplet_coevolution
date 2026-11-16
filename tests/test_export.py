
# tests for io/export.py ranked mi and ref pos

import csv
from dataclasses import replace

import numpy as np
import pytest

from triplet_coevolution.core.alignment import (
  ComponentMSA,
  MatchedTripletMSA,
  TripletMatch,
)
from triplet_coevolution.core.coordinates import (
  build_reference_map,
  subset_reference_map,
)
from triplet_coevolution.inference.mi import MIBlock, TripletMI, triplet_mi
from triplet_coevolution.io.export import (
  FIELDS,
  ranked_mi_rows,
  write_mi_table,
  write_triplet_mi_tables,
)


# ref maps have gaps and nonconsecutive original aln cols
@pytest.fixture
def ref_maps():
  a = ComponentMSA("A", "protein", ("a_ref",), ("A-CD.E",))
  b = ComponentMSA("B", "dna", ("b_ref",), ("G-T?",))
  c = ComponentMSA("C", "rna", ("c_ref",), ("A?CU",))

  a_map = subset_reference_map(build_reference_map(a, "a_ref", 101), (0, 2, 4))
  b_map = subset_reference_map(build_reference_map(b, "b_ref", 30), (0, 1))
  c_map = subset_reference_map(build_reference_map(c, "c_ref", 70), (0, 2))
  return a_map, b_map, c_map


# fixed scores let us check ranking without recalculating mi
@pytest.fixture
def ab_block():
  return MIBlock(
    component_x="A",
    component_y="B",
    mi_bits=np.array([
      [0.2, 0.8],
      [0.7, np.nan],
      [0.4, 0.7],
    ]),
    mi_apc=np.array([
      [0.1, 0.4],
      [0.5, np.nan],
      [-0.1, 0.2],
    ]),
    n_valid=np.array([
      [6, 5],
      [5, 0],
      [2, 3],
    ]),
    valid_weight=np.array([
      [3.0, 2.5],
      [2.5, 0.0],
      [1.0, 1.5],
    ]),
  )


# three mi blocks w different shapes and comp names
@pytest.fixture
def triplet_result(ab_block):
  ac = MIBlock(
    "A", "C",
    np.full((3, 2), 0.3),
    np.full((3, 2), 0.1),
    np.full((3, 2), 6, dtype=int),
    np.full((3, 2), 3.0),
  )
  bc = MIBlock(
    "B", "C",
    np.full((2, 2), 0.4),
    np.full((2, 2), 0.2),
    np.full((2, 2), 6, dtype=int),
    np.full((2, 2), 3.0),
  )
  return TripletMI(ab_block, ac, bc, 6, 3.0, False, 2)


# apc should be default ranking metric
# also check original msa cols and ref res
def test_default_apc_ranking_and_coordinates(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  rows = ranked_mi_rows(ab_block, a_map, b_map, max_pairs=3)

  assert [row["mi_apc"] for row in rows] == pytest.approx([0.5, 0.4, 0.2])
  assert [row["rank"] for row in rows] == [1, 2, 3]
  assert rows[0]["pair"] == "AB"
  assert rows[0]["x_col"] == 2
  assert rows[0]["x_original_col"] == 3
  assert rows[0]["x_reference_pos"] == 102
  assert rows[0]["x_symbol"] == "C"
  assert rows[0]["x_reference_id"] == "a_ref"
  assert rows[0]["y_col"] == 1
  assert rows[0]["y_original_col"] == 1
  assert rows[0]["y_reference_pos"] == 30
  assert rows[0]["y_symbol"] == "G"
  assert rows[0]["y_reference_id"] == "b_ref"
  assert rows[0]["mi_bits"] == pytest.approx(0.7)
  assert rows[0]["n_valid"] == 5
  assert rows[0]["valid_weight"] == pytest.approx(2.5)
  assert tuple(rows[0]) == FIELDS


# raw mi should have its own ranking order
def test_raw_mi_ranking(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  rows = ranked_mi_rows(ab_block, a_map, b_map, score="mi_bits", max_pairs=3)

  assert [row["mi_bits"] for row in rows] == pytest.approx([0.8, 0.7, 0.7])
  assert [(row["x_col"], row["y_col"]) for row in rows] == [
    (1, 2), (2, 1), (3, 2),
  ]


# None means export every valid pair
# nan cells should always be skipped
def test_all_pairs_and_nan_exclusion(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  rows = ranked_mi_rows(ab_block, a_map, b_map, max_pairs=None)

  assert len(rows) == 5
  assert (2, 2) not in [(row["x_col"], row["y_col"]) for row in rows]


# equal scores follow current aln col order
def test_tie_breaking(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  equal = replace(ab_block, mi_apc=np.zeros((3, 2)))
  rows = ranked_mi_rows(equal, a_map, b_map, max_pairs=3)

  assert [(row["x_col"], row["y_col"]) for row in rows] == [
    (1, 1), (1, 2), (2, 1),
  ]


# min_valid should filter pairs using their own obs depth
def test_min_valid_filter(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  rows = ranked_mi_rows(ab_block, a_map, b_map, min_valid=5)

  assert len(rows) == 3
  assert all(row["n_valid"] >= 5 for row in rows)
  assert [row["mi_apc"] for row in rows] == pytest.approx([0.5, 0.4, 0.1])


# ref gaps keep their original aln position when allowed
def test_reference_gap_positions(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  rows = ranked_mi_rows(ab_block, a_map, b_map, max_pairs=None)

  y_gap = next(row for row in rows if row["x_col"] == 1 and row["y_col"] == 2)
  x_gap = next(row for row in rows if row["x_col"] == 3 and row["y_col"] == 1)

  assert y_gap["y_reference_pos"] is None
  assert y_gap["y_symbol"] == "-"
  assert y_gap["y_original_col"] == 2
  assert x_gap["x_reference_pos"] is None
  assert x_gap["x_symbol"] == "."
  assert x_gap["x_original_col"] == 5


# require_reference removes pairs without numbered ref sites
def test_require_reference_positions(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  rows = ranked_mi_rows(ab_block, a_map, b_map, require_reference=True)

  assert [(row["x_col"], row["y_col"]) for row in rows] == [(2, 1), (1, 1)]
  assert all(row["x_reference_pos"] is not None for row in rows)
  assert all(row["y_reference_pos"] is not None for row in rows)


# no qualified obs should return an empty table
def test_no_qualified_pairs(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  assert ranked_mi_rows(ab_block, a_map, b_map, min_valid=100) == []


# a missing apc matrix can still export raw mi
def test_raw_mi_when_apc_missing(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  missing = replace(ab_block, mi_apc=np.full((3, 2), np.nan))

  assert ranked_mi_rows(missing, a_map, b_map) == []
  assert len(ranked_mi_rows(missing, a_map, b_map, score="mi_bits")) == 5


# export shouldnt change the original mi arrays
def test_no_input_mutation(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  names = ("mi_bits", "mi_apc", "n_valid", "valid_weight")
  before = tuple(getattr(ab_block, name).copy() for name in names)

  ranked_mi_rows(ab_block, a_map, b_map, max_pairs=2)

  for name, expected in zip(names, before):
    np.testing.assert_array_equal(getattr(ab_block, name), expected)


# csv should preserve headers and write ref gaps as empty cells
def test_write_csv_values(tmp_path, ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  path = tmp_path / "results" / "ab.csv"
  saved = write_mi_table(path, ab_block, a_map, b_map, max_pairs=2)

  assert saved == path

  with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    assert tuple(reader.fieldnames) == FIELDS

  assert len(rows) == 2
  assert rows[0]["rank"] == "1"
  assert rows[0]["x_reference_pos"] == "102"
  assert rows[0]["n_valid"] == "5"
  assert float(rows[0]["mi_apc"]) == pytest.approx(0.5)
  assert rows[1]["y_reference_pos"] == ""


# tsv should use tabs and keep the rank order
def test_write_tsv_values(tmp_path, ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  path = write_mi_table(tmp_path / "ab.tsv", ab_block, a_map, b_map, max_pairs=2)

  with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    rows = list(reader)
    assert tuple(reader.fieldnames) == FIELDS

  assert [row["rank"] for row in rows] == ["1", "2"]
  assert [row["pair"] for row in rows] == ["AB", "AB"]


# empty exports should still retain the header
def test_write_empty_table(tmp_path, ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  path = write_mi_table(
    tmp_path / "empty.csv", ab_block, a_map, b_map, min_valid=100,
  )

  with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    assert tuple(reader.fieldnames) == FIELDS
    assert list(reader) == []


# file extension controls csv or tsv delimiter
def test_uppercase_extension(tmp_path, ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  path = write_mi_table(tmp_path / "ab.CSV", ab_block, a_map, b_map)
  assert path.is_file()


# only csv and tsv output paths are supported
def test_reject_wrong_extension(tmp_path, ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  with pytest.raises(ValueError, match="output path"):
    write_mi_table(tmp_path / "ab.json", ab_block, a_map, b_map)


# reject unsupported scoring choices
@pytest.mark.parametrize("score", ["mi", "apc", "other"])
def test_invalid_score(ab_block, ref_maps, score):
  a_map, b_map, _ = ref_maps
  with pytest.raises(ValueError, match="score"):
    ranked_mi_rows(ab_block, a_map, b_map, score=score)


# reject invalid limits and bools that act like ints
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_invalid_max_pairs(ab_block, ref_maps, value):
  a_map, b_map, _ = ref_maps
  with pytest.raises(ValueError, match="max_pairs"):
    ranked_mi_rows(ab_block, a_map, b_map, max_pairs=value)


# min_valid must be a positive int
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_min_valid(ab_block, ref_maps, value):
  a_map, b_map, _ = ref_maps
  with pytest.raises(ValueError, match="min_valid"):
    ranked_mi_rows(ab_block, a_map, b_map, min_valid=value)


# require_reference must be a bool
@pytest.mark.parametrize("value", ["yes", 1, None])
def test_invalid_require_reference(ab_block, ref_maps, value):
  a_map, b_map, _ = ref_maps
  with pytest.raises(TypeError, match="require_reference"):
    ranked_mi_rows(ab_block, a_map, b_map, require_reference=value)


# wrong mi and reference types should fail
@pytest.mark.parametrize("wrong", [None, "not a block"])
def test_invalid_block_type(wrong, ref_maps):
  a_map, b_map, _ = ref_maps
  with pytest.raises(TypeError, match="MIBlock"):
    ranked_mi_rows(wrong, a_map, b_map)


def test_invalid_reference_map(ab_block, ref_maps):
  _, b_map, _ = ref_maps
  with pytest.raises(TypeError, match="ReferenceCoordinates"):
    ranked_mi_rows(ab_block, None, b_map)


# comp labels must match the ref maps
def test_mismatched_component_names(ab_block, ref_maps):
  a_map, _, c_map = ref_maps
  with pytest.raises(ValueError, match="dont match"):
    ranked_mi_rows(ab_block, a_map, c_map)


# all four mi arrays must match both ref map lengths
@pytest.mark.parametrize(
  "field",
  ["mi_bits", "mi_apc", "n_valid", "valid_weight"],
)
def test_wrong_score_shapes(ab_block, ref_maps, field):
  a_map, b_map, _ = ref_maps
  changed = replace(ab_block, **{field: np.zeros((1, 1))})
  with pytest.raises(ValueError, match="shape"):
    ranked_mi_rows(changed, a_map, b_map)


# raw mi cant contain negative or infinite scores
@pytest.mark.parametrize("value", [-0.2, np.inf, -np.inf])
def test_invalid_raw_mi(ab_block, ref_maps, value):
  a_map, b_map, _ = ref_maps
  values = ab_block.mi_bits.copy()
  values[0, 0] = value
  with pytest.raises(ValueError, match="mi scores"):
    ranked_mi_rows(replace(ab_block, mi_bits=values), a_map, b_map)


# apc can be negative but not infinite
def test_invalid_apc_infinity(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  values = ab_block.mi_apc.copy()
  values[0, 0] = np.inf
  with pytest.raises(ValueError, match="mi scores"):
    ranked_mi_rows(replace(ab_block, mi_apc=values), a_map, b_map)


def test_negative_apc_is_allowed(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  rows = ranked_mi_rows(ab_block, a_map, b_map, max_pairs=None)
  assert any(row["mi_apc"] < 0 for row in rows)


# n_valid must be nonnegative ints
@pytest.mark.parametrize("value,dtype", [(-1, int), (1.5, float)])
def test_invalid_n_valid(ab_block, ref_maps, value, dtype):
  a_map, b_map, _ = ref_maps
  values = ab_block.n_valid.astype(dtype)
  values[0, 0] = value
  with pytest.raises(ValueError, match="n_valid"):
    ranked_mi_rows(replace(ab_block, n_valid=values), a_map, b_map)


# valid_weight cant be negative or infinite
@pytest.mark.parametrize("value", [-0.2, np.inf, np.nan])
def test_invalid_valid_weight(ab_block, ref_maps, value):
  a_map, b_map, _ = ref_maps
  values = ab_block.valid_weight.copy()
  values[0, 0] = value
  with pytest.raises(ValueError, match="valid_weight"):
    ranked_mi_rows(replace(ab_block, valid_weight=values), a_map, b_map)


# write the ab ac bc files w the correct comp labels
def test_write_triplet_tables(tmp_path, triplet_result, ref_maps):
  paths = write_triplet_mi_tables(
    tmp_path / "out", triplet_result, *ref_maps, max_pairs=None,
  )

  assert set(paths) == {"ab", "ac", "bc"}
  assert {name: path.name for name, path in paths.items()} == {
    "ab": "mi_ab.tsv",
    "ac": "mi_ac.tsv",
    "bc": "mi_bc.tsv",
  }

  for name, count in (("ab", 5), ("ac", 6), ("bc", 4)):
    with paths[name].open(newline="", encoding="utf-8") as handle:
      rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == count
    assert {row["pair"] for row in rows} == {name.upper()}


# change fmt and max_pairs for all three outputs
def test_triplet_csv_and_limit(tmp_path, triplet_result, ref_maps):
  paths = write_triplet_mi_tables(
    tmp_path / "out", triplet_result, *ref_maps, fmt="csv", max_pairs=1,
  )

  for path in paths.values():
    assert path.suffix == ".csv"
    with path.open(newline="", encoding="utf-8") as handle:
      assert len(list(csv.DictReader(handle))) == 1


# reject invalid triplet arguments
def test_invalid_triplet_result(tmp_path, ref_maps):
  with pytest.raises(TypeError, match="TripletMI"):
    write_triplet_mi_tables(tmp_path / "out", None, *ref_maps)


@pytest.mark.parametrize("fmt", ["json", "CSV", "txt"])
def test_invalid_triplet_format(tmp_path, triplet_result, ref_maps, fmt):
  with pytest.raises(ValueError, match="fmt"):
    write_triplet_mi_tables(
      tmp_path / "out", triplet_result, *ref_maps, fmt=fmt,
    )


# check every comp before creating any export files
def test_triplet_preflight_validation(tmp_path, triplet_result, ref_maps):
  a_map, b_map, _ = ref_maps
  out_dir = tmp_path / "out"

  with pytest.raises(ValueError, match="dont match"):
    write_triplet_mi_tables(out_dir, triplet_result, a_map, b_map, b_map)

  assert not out_dir.exists()


# use actual mi output to verify the final csv scores
def test_real_mi_to_export(tmp_path):
  a = ComponentMSA(
    "A", "protein",
    ("a1", "a2", "a3", "a4"),
    ("A", "A", "C", "C"),
  )
  b = ComponentMSA(
    "B", "dna",
    ("b1", "b2", "b3", "b4"),
    ("G", "G", "T", "T"),
  )
  c = ComponentMSA(
    "C", "rna",
    ("c1", "c2", "c3", "c4"),
    ("A", "C", "A", "C"),
  )

  matches = tuple(
    TripletMatch(f"obs{i}", f"a{i}", f"b{i}", f"c{i}")
    for i in range(1, 5)
  )
  msa = MatchedTripletMSA(a, b, c, matches)
  mi = triplet_mi(msa)

  maps = (
    build_reference_map(a, "a1"),
    build_reference_map(b, "b1"),
    build_reference_map(c, "c1"),
  )
  paths = write_triplet_mi_tables(
    tmp_path, mi, *maps, score="mi_bits",
  )

  with paths["ab"].open(newline="", encoding="utf-8") as handle:
    row = next(csv.DictReader(handle, delimiter="\t"))

  assert float(row["mi_bits"]) == pytest.approx(1)
  assert row["x_reference_pos"] == "1"
  assert row["y_reference_pos"] == "1"
  assert row["n_valid"] == "4"
  assert float(row["valid_weight"]) == pytest.approx(4)
