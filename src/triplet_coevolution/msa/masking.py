# select and remove unreliable msa cols while keeping original positions

from dataclasses import dataclass

import numpy as np

from triplet_coevolution.core.alignment import (
  ComponentMSA,
  MatchedTripletMSA,
)
from triplet_coevolution.msa.qc import ComponentQC


@dataclass(frozen=True)
class MaskedComponent:
  """filtered component and 0 based cols from its input msa"""

  msa: ComponentMSA
  original_columns: tuple[int, ...]

  @property
  def original_positions(self) -> tuple[int, ...]:
    """1 based input msa positions useful when reporting sites"""
    return tuple(i + 1 for i in self.original_columns)


@dataclass(frozen=True)
class MaskedTriplet:
  """filtered matched msa and 0 based input cols for each component"""

  msa: MatchedTripletMSA
  a_columns: tuple[int, ...]
  b_columns: tuple[int, ...]
  c_columns: tuple[int, ...]


def column_mask(
  qc: ComponentQC,
  min_coverage: float = 0.5,
  max_gap_fraction: float = 0.5,
  max_missing_fraction: float = 0.5,
  min_entropy_bits: float | None = None,
) -> np.ndarray:
  """return True for cols passing qc thresholds

  coverage, gaps, and missing use all seqs as denominator
  entropy uses canonical res only
  cols with no canonical res are always excluded
  """

  if not isinstance(qc, ComponentQC):
    raise TypeError("column_mask expects a ComponentQC")

  # thresholds must be finite fractions between 0 and 1
  for name, value in (
    ("min_coverage", min_coverage),
    ("max_gap_fraction", max_gap_fraction),
    ("max_missing_fraction", max_missing_fraction),
  ):
    if (
      isinstance(value, (bool, np.bool_))
      or not isinstance(value, (int, float, np.integer, np.floating))
      or not np.isfinite(value)
      or not 0 <= value <= 1
    ):
      raise ValueError(f"{name} must be between 0 and 1")

  if min_entropy_bits is not None:
    if (
      isinstance(min_entropy_bits, (bool, np.bool_))
      or not isinstance(min_entropy_bits, (int, float, np.integer, np.floating))
      or not np.isfinite(min_entropy_bits)
      or min_entropy_bits < 0
    ):
      raise ValueError("min_entropy_bits must be finite and nonnegative")

  # exclude incomplete cols but retain conserved cols by default
  keep = (
    (qc.n_canonical > 0)
    & (qc.coverage >= min_coverage)
    & (qc.gap_fraction <= max_gap_fraction)
    & (qc.missing_fraction <= max_missing_fraction)
  )

  if min_entropy_bits is not None:
    keep &= qc.entropy_bits >= min_entropy_bits

  keep.setflags(write=False)
  return keep


def apply_mask(
  msa: ComponentMSA,
  keep: np.ndarray | list[bool] | tuple[bool, ...] | None,
) -> MaskedComponent:
  """filter cols without changing seq rows or the original msa

  keep=None retains every input col
  at least one col must remain to construct a ComponentMSA
  """

  if not isinstance(msa, ComponentMSA):
    raise TypeError("apply_mask expects a ComponentMSA")

  if keep is None:
    keep = np.ones(msa.n_sites, dtype=bool)
  else:
    keep = np.asarray(keep)

  if keep.ndim != 1 or len(keep) != msa.n_sites:
    raise ValueError("mask length must equal number of msa cols")

  if keep.dtype.kind != "b":
    raise TypeError("mask must contain boolean values")

  selected = np.flatnonzero(keep)

  if selected.size == 0:
    raise ValueError("mask removed all msa cols")

  # filter string cols to preserve symbols (including - . ?)
  sequences = tuple(
    "".join(seq[i] for i in selected)
    for seq in msa.sequences
  )

  filtered = ComponentMSA(
    name=msa.name,
    molecule=msa.molecule,
    ids=msa.ids,
    sequences=sequences,
  )

  return MaskedComponent(
    msa=filtered,
    original_columns=tuple(int(i) for i in selected),
  )


def apply_triplet_masks(
  msa: MatchedTripletMSA,
  a_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
  b_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
  c_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
) -> MaskedTriplet:
  """filter each component independently, preserving matched seq rows"""

  if not isinstance(msa, MatchedTripletMSA):
    raise TypeError("apply_triplet_masks expects a MatchedTripletMSA")

  a = apply_mask(msa.a, a_mask)
  b = apply_mask(msa.b, b_mask)
  c = apply_mask(msa.c, c_mask)

  filtered = MatchedTripletMSA(
    a=a.msa,
    b=b.msa,
    c=c.msa,
    matches=msa.matches,
  )

  return MaskedTriplet(
    msa=filtered,
    a_columns=a.original_columns,
    b_columns=b.original_columns,
    c_columns=c.original_columns,
  )
