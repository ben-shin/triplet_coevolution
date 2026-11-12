# tests for io/metadata.py matching tables and provenance

import pytest

from triplet_coevolution.core.alignment import TripletMatch
from triplet_coevolution.io.metadata import (
  REQUIRED_COLUMNS,
  TripletTable,
  read_triplet_map,
  read_triplet_table,
  write_triplet_map,
  write_triplet_table,
)


# reusable example of two matched observations
@pytest.fixture
def matches():
  return (
    TripletMatch("obs1", "a1", "b1", "c1"),
    TripletMatch("obs2", "a2", "b2", "c2"),
  )


# reusable table with additional biological metadata
@pytest.fixture
def table(matches):
  return TripletTable(
    matches=matches,
    extra_columns=("taxon_id", "evidence"),
    annotations=(
      {"taxon_id": "0093", "evidence": "synteny, curated"},
      {"taxon_id": "0094", "evidence": ""},
    ),
  )


# check required column names and order
def test_required_columns():
  assert REQUIRED_COLUMNS == (
    "observation_id", "a_id", "b_id", "c_id"
  )


# read basic tsv and preserve row order
def test_read_tsv(tmp_path, matches):
  path = tmp_path / "matches.tsv"
  path.write_text(
    "observation_id\ta_id\tb_id\tc_id\n"
    "obs1\ta1\tb1\tc1\n"
    "obs2\ta2\tb2\tc2\n"
  )

  result = read_triplet_table(path)

  assert result.matches == matches
  assert result.n_observations == 2
  assert result.extra_columns == ()
  assert result.annotations == ({}, {})
  assert read_triplet_map(path) == matches


# read csv with optional metadata and quoted commas
def test_read_csv_with_metadata(tmp_path, matches):
  path = tmp_path / "matches.csv"
  path.write_text(
    "observation_id,a_id,b_id,c_id,taxon_id,evidence\n"
    'obs1,a1,b1,c1,0093,"synteny, curated"\n'
    "obs2,a2,b2,c2,0094,orthology\n"
  )

  result = read_triplet_table(path)

  assert result.matches == matches
  assert result.extra_columns == ("taxon_id", "evidence")
  assert result.annotations[0]["taxon_id"] == "0093"
  assert result.annotations[0]["evidence"] == "synteny, curated"


# column order should not affect matching
def test_reordered_headers(tmp_path, matches):
  path = tmp_path / "matches.csv"
  path.write_text(
    "c_id,evidence,a_id,observation_id,b_id\n"
    "c1,curated,a1,obs1,b1\n"
    "c2,curated,a2,obs2,b2\n"
  )

  result = read_triplet_table(path)

  assert result.matches == matches
  assert result.extra_columns == ("evidence",)


# normalize BOM and surrounding whitespace
def test_bom_and_whitespace(tmp_path, matches):
  path = tmp_path / "matches.tsv"
  path.write_text(
    "\ufeff observation_id \t a_id \t b_id \t c_id \n"
    " obs1 \t a1 \t b1 \t c1 \n"
    " obs2 \t a2 \t b2 \t c2 \n"
  )

  assert read_triplet_map(path) == matches


# read/write should preserve values in both formats
@pytest.mark.parametrize("extension", [".tsv", ".csv"])
def test_round_trip_basic(tmp_path, matches, extension):
  path = tmp_path / "nested" / f"matches{extension}"

  returned = write_triplet_map(path, list(matches))

  assert returned == path
  assert path.is_file()
  assert read_triplet_map(path) == matches


# preserve additional metadata during export
@pytest.mark.parametrize("extension", [".tsv", ".csv"])
def test_round_trip_with_metadata(tmp_path, table, extension):
  path = tmp_path / f"matches{extension}"

  write_triplet_table(path, table)
  restored = read_triplet_table(path)

  assert restored == table


# allow custom extensions with explicit delimiter
def test_custom_delimiter(tmp_path, matches):
  path = tmp_path / "matches.txt"

  write_triplet_map(path, matches, delimiter="\t")

  assert read_triplet_map(path, delimiter="\t") == matches


# accession strings can be shared across different components
def test_same_accession_across_components(tmp_path):
  path = tmp_path / "matches.tsv"
  path.write_text(
    "observation_id\ta_id\tb_id\tc_id\n"
    "obs1\tshared\tshared\tshared\n"
  )

  result = read_triplet_map(path)

  assert result[0] == TripletMatch(
    "obs1", "shared", "shared", "shared"
  )


# reject files that dont exist
def test_missing_file(tmp_path):
  with pytest.raises(FileNotFoundError, match="not found"):
    read_triplet_map(tmp_path / "missing.tsv")


# reject unsupported extensions
def test_unsupported_extension(tmp_path):
  path = tmp_path / "matches.txt"
  path.write_text("observation_id,a_id,b_id,c_id\n")

  with pytest.raises(ValueError, match="expected tsv/csv"):
    read_triplet_table(path)


# reject unsupported delimiters
def test_unsupported_delimiter(tmp_path):
  path = tmp_path / "matches.csv"
  path.write_text("observation_id,a_id,b_id,c_id\n")

  with pytest.raises(ValueError, match="expected tsv/csv"):
    read_triplet_table(path, delimiter=";")


# reject empty and header-only files
@pytest.mark.parametrize(
  "content,error",
  [
    ("", "empty or has no header"),
    ("observation_id,a_id,b_id,c_id\n", "no matches"),
  ],
)
def test_empty_tables(tmp_path, content, error):
  path = tmp_path / "empty.csv"
  path.write_text(content)

  with pytest.raises(ValueError, match=error):
    read_triplet_map(path)


# required column names are case sensitive
@pytest.mark.parametrize(
  "header",
  [
    "observation_id,a_id,b_id",
    "Observation_id,a_id,b_id,c_id",
  ],
)
def test_missing_required_headers(tmp_path, header):
  path = tmp_path / "bad.csv"
  path.write_text(header + "\nobs1,a1,b1,c1\n")

  with pytest.raises(ValueError, match="missing required columns"):
    read_triplet_map(path)


# reject duplicate and empty header names
@pytest.mark.parametrize(
  "header",
  [
    "observation_id,a_id,b_id,c_id,a_id",
    "observation_id,a_id,b_id,c_id, ",
  ],
)
def test_invalid_headers(tmp_path, header):
  path = tmp_path / "bad.csv"
  path.write_text(header + "\nobs1,a1,b1,c1,extra\n")

  with pytest.raises(ValueError, match="column names"):
    read_triplet_table(path)


# reject rows with too many or too few fields
@pytest.mark.parametrize(
  "row",
  [
    "obs1,a1,b1",
    "obs1,a1,b1,c1,extra",
  ],
)
def test_invalid_field_count(tmp_path, row):
  path = tmp_path / "bad.csv"
  path.write_text(
    "observation_id,a_id,b_id,c_id\n"
    + row + "\n"
  )

  with pytest.raises(ValueError, match="malformed"):
    read_triplet_map(path)


# reject blank values in any required field
@pytest.mark.parametrize("field", REQUIRED_COLUMNS)
def test_missing_required_values(tmp_path, field):
  row = {
    "observation_id": "obs1",
    "a_id": "a1",
    "b_id": "b1",
    "c_id": "c1",
  }
  row[field] = "  "

  path = tmp_path / "bad.tsv"
  path.write_text(
    "\t".join(REQUIRED_COLUMNS) + "\n"
    + "\t".join(row.values()) + "\n"
  )

  with pytest.raises(ValueError, match=f"missing {field}"):
    read_triplet_map(path)


# reject duplicate IDs within each component/observation column
@pytest.mark.parametrize("field", REQUIRED_COLUMNS)
def test_duplicate_ids(tmp_path, field):
  row1 = ["obs1", "a1", "b1", "c1"]
  row2 = ["obs2", "a2", "b2", "c2"]
  index = REQUIRED_COLUMNS.index(field)
  row2[index] = row1[index]

  path = tmp_path / "bad.csv"
  path.write_text(
    ",".join(REQUIRED_COLUMNS) + "\n"
    + ",".join(row1) + "\n"
    + ",".join(row2) + "\n"
  )

  with pytest.raises(ValueError, match=f"duplicate {field}"):
    read_triplet_map(path)


# normalize list inputs into tuples
def test_table_normalizes_lists(matches):
  result = TripletTable(matches=list(matches))

  assert isinstance(result.matches, tuple)
  assert result.matches == matches
  assert result.annotations == ({}, {})


# direct match construction should reject surrounding whitespace
def test_direct_match_rejects_whitespace(matches):
  incorrect = (
    TripletMatch(" obs1 ", "a1", "b1", "c1"),
    matches[1],
  )

  with pytest.raises(ValueError, match="surrounding spaces"):
    TripletTable(matches=incorrect)


# reject empty records and wrong object types
@pytest.mark.parametrize(
  "records,error_type",
  [
    ((), ValueError),
    (("not a TripletMatch",), TypeError),
  ],
)
def test_invalid_match_records(records, error_type):
  with pytest.raises(error_type):
    TripletTable(matches=records)


# reject invalid extra metadata columns
@pytest.mark.parametrize(
  "columns",
  [
    ("taxon_id", "taxon_id"),
    ("a_id",),
    ("",),
  ],
)
def test_invalid_optional_columns(matches, columns):
  with pytest.raises(ValueError, match="optional metadata column"):
    TripletTable(matches=matches, extra_columns=columns)


# annotation rows must match declared metadata columns
@pytest.mark.parametrize(
  "annotations,error",
  [
    (({"evidence": "curated"},), "number of observations"),
    (({"wrong": "x"}, {"evidence": "y"}), "incorrect optional"),
    (({"evidence": 5}, {"evidence": "y"}), "nonstring"),
  ],
)
def test_invalid_annotations(matches, annotations, error):
  with pytest.raises(ValueError, match=error):
    TripletTable(
      matches=matches,
      extra_columns=("evidence",),
      annotations=annotations,
    )


# extra columns require corresponding annotation rows
def test_metadata_requires_annotation_rows(matches):
  with pytest.raises(ValueError, match="incorrect optional"):
    TripletTable(
      matches=matches,
      extra_columns=("evidence",),
    )
