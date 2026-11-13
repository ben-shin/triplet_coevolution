# read and write hom correspondence tables for 3 component coevolution

import csv
from dataclasses import dataclass
from pathlib import Path

from triplet_coevolution.core.alignment import TripletMatch

# required columns in matching table - CASE SENSITIVE
REQUIRED_COLUMNS = ("observation_id", "a_id", "b_id", "c_id")

# support extensions and separators
DELIMITERS = {
  ".tsv": "\t",
  ".csv": ",",
}


def _delimiter_for(path: Path, delimiter: str | None) -> str:
  """choose csv/tsv delimiter but also allow explicit override"""

  if delimiter is None:
    delimiter = DELIMITERS.get(path.suffix.lower())

  if delimiter not in {"\t", ","}:
    raise ValueError(
      "expected tsv/csv or delimiter='\\t' / delimiter=','"
    )

  return delimiter

def _validate_matches(matches: tuple[TripletMatch, ...]) -> None:
  """check all matches are complete and one to one"""

  if not matches:
    raise ValueError("triplet table contains no matches")

  for field in REQUIRED_COLUMNS:
    seen = set()

    for row_number, match in enumerate(matches, start=1):
      if not isinstance(match, TripletMatch):
        raise TypeError(f"row {row_number} is not a TripletMatch")

      value = getattr(match, field)

      if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {row_number} has an empty {field}")

      if value != value.strip():
        raise ValueError(
          f"row {row_number} has surrounding spaces in {field}"
        )

      if value in seen:
        raise ValueError(f"duplicate {field}: {value}")

      seen.add(value)


@dataclass(frozen=True)
class TripletTable:
  """matched homs and optional observation metadata

  .matches contains TripletMatch objects
  .annotations retains extra columns like taxon_id/evidence
  """

  matches: tuple[TripletMatch, ...]
  extra_columns: tuple[str, ...] = ()
  annotations: tuple[dict[str, str], ...] = ()

  def __post_init__(self) -> None:
    matches = tuple(self.matches)
    extra_columns = tuple(self.extra_columns)
    annotations = tuple(self.annotations)

    _validate_matches(matches)

    if len(set(extra_columns)) != len(extra_columns):
      raise ValueError("duplicate optional metadata column")

    if any(
      not x or x in REQUIRED_COLUMNS
      for x in extra_columns
    ):
      raise ValueError("invalid optional metadata column")

    if not annotations:
      annotations = tuple({} for _ in matches)

    if len(annotations) != len(matches):
      raise ValueError(
        "annotations must match number of observations"
      )

    for row_number, annotation in enumerate(annotations, start=1):
      if set(annotation) != set(extra_columns):
        raise ValueError(
          f"incorrect optional metadata columns at row {row_number}"
        )

      if any(
        not isinstance(value, str)
        for value in annotation.values()
      ):
        raise ValueError(
          f"nonstring metadata at row {row_number}"
        )

    object.__setattr__(self, "matches", matches)
    object.__setattr__(self, "extra_columns", extra_columns)
    object.__setattr__(self, "annotations", annotations)

  @property
  def n_observations(self) -> int:
    """number of matched evolutionary observations"""
    return len(self.matches)


def read_triplet_table(
  path: str | Path,
  delimiter: str | None = None,
) -> TripletTable:
  """read csv/tsv matching table while preserving metadat

  requires observation_id, a_id, b_id, c_id and each seq id may appear once per component
  """

  path = Path(path).expanduser()

  # check input file
  if not path.is_file():
    raise FileNotFoundError(
      f"triplet table not found: {path}"
    )

  delimiter = _delimiter_for(path, delimiter)

  matches = []
  annotations = []

  # utf-8-sig also handles files containing a BOM
  with path.open(
    "r", encoding="utf-8-sig", newline=""
  ) as handle:

    reader = csv.DictReader(handle, delimiter=delimiter)

    if reader.fieldnames is None:
      raise ValueError(
        "triplet table is empty or has no header"
      )

    # normalize header names
    columns = tuple(
      column.strip()
      for column in reader.fieldnames
    )

    if not all(columns) or len(set(columns)) != len(columns):
      raise ValueError(
        "triplet table has empty or duplicate column names"
      )

    # check required columns
    missing = set(REQUIRED_COLUMNS) - set(columns)

    if missing:
      raise ValueError(
        f"missing required columns: {sorted(missing)}"
      )

    reader.fieldnames = list(columns)

    # preserve additional columns for provenance
    extra_columns = tuple(
      column for column in columns
      if column not in REQUIRED_COLUMNS
    )

    for row in reader:

      # reject rows containing too many or too few fields
      if None in row or any(
        value is None for value in row.values()
      ):
        raise ValueError(
          f"malformed triplet table at line {reader.line_num}"
        )

      row = {
        key: value.strip()
        for key, value in row.items()
      }

      # check that required identifiers are present
      for field in REQUIRED_COLUMNS:
        if not row[field]:
          raise ValueError(
            f"missing {field} at line {reader.line_num}"
          )

      # create a TripletMatch using the four required fields
      matches.append(
        TripletMatch(**{
          key: row[key]
          for key in REQUIRED_COLUMNS
        })
      )

      # retain optional biological metadata separately
      annotations.append({
        key: row[key]
        for key in extra_columns
      })

  return TripletTable(
    matches=tuple(matches),
    extra_columns=extra_columns,
    annotations=tuple(annotations),
  )


def read_triplet_map(
  path: str | Path,
  delimiter: str | None = None,
) -> tuple[TripletMatch, ...]:
  """read matching records without returning extra metadata"""

  return read_triplet_table(
    path, delimiter=delimiter
  ).matches


def write_triplet_table(
  path: str | Path,
  table: TripletTable,
  delimiter: str | None = None,
) -> Path:
  """export matches and optional metadata to csv/tsv"""

  path = Path(path).expanduser()
  delimiter = _delimiter_for(path, delimiter)

  path.parent.mkdir(parents=True, exist_ok=True)

  columns = REQUIRED_COLUMNS + table.extra_columns

  with path.open(
    "w", encoding="utf-8", newline=""
  ) as handle:

    writer = csv.DictWriter(
      handle,
      fieldnames=columns,
      delimiter=delimiter,
    )

    writer.writeheader()

    for match, annotation in zip(
      table.matches, table.annotations
    ):
      writer.writerow({
        "observation_id": match.observation_id,
        "a_id": match.a_id,
        "b_id": match.b_id,
        "c_id": match.c_id,
        **annotation,
      })

  return path


def write_triplet_map(
  path: str | Path,
  matches: tuple[TripletMatch, ...] | list[TripletMatch],
  delimiter: str | None = None,
) -> Path:
  """export basic four column matching table"""

  return write_triplet_table(
    path,
    TripletTable(matches=tuple(matches)),
    delimiter=delimiter,
  )
