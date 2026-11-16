# map msa cols to original aln cols and reference res positions

from dataclasses import dataclass
from numbers import Integral

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.alphabets import GAPS


def _check_positive_int(value: int, name: str) -> int:
  """require positive int for reference numbering"""

  if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
    raise ValueError(f"{name} must be a positive integer")
  return int(value)


def reference_positions(
  gapped_reference: str,
  start: int = 1,
) -> tuple[int | None, ...]:
  """number occupied ref sites, use None for aln gaps
  - and . do not advance numbering
  ? and ambiguous res count as occupied sites with unknown identity
  start is the 1 based pos of the first nongap ref site
  """

  start = _check_positive_int(start, "start")

  if not isinstance(gapped_reference, str) or not gapped_reference:
    raise ValueError("reference sequence must be a nonempty string")

  if any(
    not (symbol.isalpha() or symbol in GAPS or symbol == "?")
    for symbol in gapped_reference
  ):
    raise ValueError("ref seq has invalid symbols")

  positions = []
  next_position = start

  for symbol in gapped_reference:
    if symbol in GAPS:
      positions.append(None)
    else:
      positions.append(next_position)
      next_position += 1

  return tuple(positions)


@dataclass(frozen=True)
class SiteCoordinate:
  """site index in current msa, original aln, and ref seq"""

  current_column: int
  original_column: int
  reference_position: int | None
  reference_symbol: str


@dataclass(frozen=True)
class ReferenceCoordinates:
  """coordinate mapping for one component reference
  original_columns are 0 based pos in the unmasked input msa
  reference_positions are 1 based pos or None for ref gaps
  symbols are the original ref aln characters
  """

  component: str
  reference_id: str
  original_columns: tuple[int, ...]
  reference_positions: tuple[int | None, ...]
  symbols: tuple[str, ...]
  start: int = 1

  def __post_init__(self) -> None:
    columns = tuple(self.original_columns)
    positions = tuple(self.reference_positions)
    symbols = tuple(self.symbols)

    _check_positive_int(self.start, "start")

    if self.component not in {"A", "B", "C"}:
      raise ValueError("component must be A, B, or C")

    if not isinstance(self.reference_id, str) or not self.reference_id:
      raise ValueError("reference_id must be a nonempty string")

    if not columns or not (len(columns) == len(positions) == len(symbols)):
      raise ValueError("coordinate arrays must have equal nonzero lengths")

    if any(
      isinstance(i, bool) or not isinstance(i, Integral) or i < 0
      for i in columns
    ):
      raise ValueError("original columns must be nonnegative integers")

    if any(int(a) >= int(b) for a, b in zip(columns, columns[1:])):
      raise ValueError("original columns must be strictly increasing")

    if any(not isinstance(symbol, str) or len(symbol) != 1 for symbol in symbols):
      raise ValueError("reference symbols must be single characters")

    if any(
      (symbol in GAPS) != (pos is None)
      for symbol, pos in zip(symbols, positions)
    ):
      raise ValueError("reference gaps must have None coordinates")

    numbered = [pos for pos in positions if pos is not None]

    if any(
      isinstance(pos, bool) or not isinstance(pos, Integral) or pos < self.start
      for pos in numbered
    ):
      raise ValueError("reference positions must be positive integers")

    if any(int(a) >= int(b) for a, b in zip(numbered, numbered[1:])):
      raise ValueError("reference positions must be strictly increasing")

    object.__setattr__(self, "original_columns", tuple(int(i) for i in columns))
    object.__setattr__(
      self,
      "reference_positions",
      tuple(None if pos is None else int(pos) for pos in positions),
    )
    object.__setattr__(self, "symbols", symbols)

  @property
  def n_sites(self) -> int:
    """number of cols in the current mapped msa"""
    return len(self.original_columns)

  @property
  def original_positions(self) -> tuple[int, ...]:
    """1-based original aln cols for reporting"""
    return tuple(col + 1 for col in self.original_columns)

  def site(self, index: int) -> SiteCoordinate:
    """get site mapping from a 0-based current msa col"""

    if isinstance(index, bool) or not isinstance(index, Integral):
      raise TypeError("column index must be an integer")

    if index < 0 or index >= self.n_sites:
      raise IndexError("column index outside current msa")

    return SiteCoordinate(
      current_column=int(index) + 1,
      original_column=self.original_columns[index] + 1,
      reference_position=self.reference_positions[index],
      reference_symbol=self.symbols[index],
    )


def build_reference_map(
  msa: ComponentMSA,
  reference_id: str,
  start: int = 1,
) -> ReferenceCoordinates:
  """make col to ref mapping from unmasked component msa
  choose ref explicitly; do not assume the first seq is ref
  the ref can later be excluded during matched triplet selection
  """

  if not isinstance(msa, ComponentMSA):
    raise TypeError("build_reference_map expects a ComponentMSA")

  if reference_id not in msa.ids:
    raise ValueError(
      f"reference id {reference_id!r} not found in component {msa.name}"
    )

  reference = msa.sequences[msa.ids.index(reference_id)]

  return ReferenceCoordinates(
    component=msa.name,
    reference_id=reference_id,
    original_columns=tuple(range(msa.n_sites)),
    reference_positions=reference_positions(reference, start=start),
    symbols=tuple(reference),
    start=start,
  )


def subset_reference_map(
  mapping: ReferenceCoordinates,
  kept_columns: tuple[int, ...] | list[int],
) -> ReferenceCoordinates:
  """retain mapping after selecting cols with msa/masking.py
  kept_columns are 0 based indices in the CURRENT input mapping
  original msa cols remain unchanged even after repeated masking
  """

  if not isinstance(mapping, ReferenceCoordinates):
    raise TypeError("subset_reference_map expects ReferenceCoordinates")

  indices = tuple(kept_columns)

  if not indices:
    raise ValueError("at least one reference col must be kept")

  if any(
    isinstance(i, bool) or not isinstance(i, Integral)
    for i in indices
  ):
    raise TypeError("kept cols must be integer indices")

  if any(i < 0 or i >= mapping.n_sites for i in indices):
    raise IndexError("kept col outside current reference mapping")

  if any(a >= b for a, b in zip(indices, indices[1:])):
    raise ValueError("kept cols must be unique and increasing")

  return ReferenceCoordinates(
    component=mapping.component,
    reference_id=mapping.reference_id,
    original_columns=tuple(mapping.original_columns[i] for i in indices),
    reference_positions=tuple(mapping.reference_positions[i] for i in indices),
    symbols=tuple(mapping.symbols[i] for i in indices),
    start=mapping.start,
  )
