# sequence alignments for 3 body coev

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from triplet_coevolution.core.alphabets import (
  MoleculeType,
  encode_sequence,
)

@dataclass(frozen=True)
class ComponentMSA:
  """1 rectangular single molecule type alignment, sequence ids must be unique
  rows don't have cross component meanings (until they're matched with a tripletmatch table)
  """

  name: str
  molecule: MoleculeType | str
  ids: tuple[str, ...]
  sequences: tuple[str, ...]

  def __post_init__(self) -> None:
    if self.name not in ("A", "B", "C"):
      raise ValueError(
        "name must be a b or c"
      )

    molecule = MoleculeType(self.molecule)
    ids = tuple(self.ids)
    sequences = tuple(self.sequences)

    if not ids or len(ids) != len(sequences):
      raise ValueError(
        "ids and sequences cant be empty and has to be same in number"
      )

    if any(
      not isinstance(seq_id, str) or not seq_id.strip()
      for seq_id in ids
    ):
      raise ValueError(
        "every sequence id must be nonempty strings"
      )

    if len(set(ids)) != len(ids):
      raise ValueError(
        f"duplicate sequence ids in component (self.name)"
      )

    if any(not isinstance(seq, str) for seq in sequences):
      raise ValueError("all sequences must be strings")

    sequences = tuple(seq.upper() for seq in sequences)

    lengths = {len(seq) for seq in sequences}

    if len(lengths) != 1 or 0 in lengths:
      raise ValueError(
        f"component {self.name} can't be an empty rectangular msa"
      )

    for seq_id, seq in zip(ids, sequences):
      try:
        encode_sequence(seq, molecule)
      except ValueError as exc:
        raise ValueError(
          f"invalid seq {seq_id!r}: {exc}"
        ) from exc

    object.__setattr__(self, "molecule", molecule)
    object.__setattr__(self, "ids", ids)
    object.__setattr__(self, "sequences", sequences)

  @property
  def n_sequences(self) -> int:
    """number of seqs in this aln"""
    return len(self.ids)

  @property
  def n_sites(self) -> int:
    """number of aln pos"""
    return len(self.sequences[0])

  @cached_property
  def matrix(self) -> np.ndarray:
    """return n x l numerical aln matrix
    canonical res: 0 thru alphabet size - 1 (19 for prot 3 for rna/dna)
    gaps: alphabet size (20 for prot 4 for rna/dna)
    ambiguous/missing: -1
    """
    result = np.stack([
      encode_sequence(seq, self.molecule)
      for seq in self.sequences
    ])

    result.setflags(write=False)

    return result

  def take_ids(
    self,
    ordered_ids: tuple[str, ...] | list[str],
  ) -> "ComponentMSA":
    """select and reorder rows using explicit seq ids"""

    requested = tuple(ordered_ids)

    if not requested:
      raise ValueError(
        "cant select empty aln"
      )

    if len(set(requested)) != len(requested):
      raise ValueError(
        "requested seq ids have duplicates"
      )

    lookup = {
      seq_id: pos
      for pos, seq_id in enumerate(self.ids)
    }

    missing = [
      seq_id
      for seq_id in requested
      if seq_id not in lookup
    ]

    if missing:
      raise ValueError(
        f"component {self.name} is missing ids: {missing}"
      )

    return ComponentMSA(
      name=self.name,
      molecule=self.molecule,
      ids=requested,
      sequences=tuple(
        self.sequences[lookup[seq_id]]
        for seq_id in requested
      ),
    )


@dataclass(frozen=True)
class TripletMatch:
  """1 verified correspondence between a b c homs"""

  observation_id: str
  a_id: str
  b_id: str
  c_id: str


@dataclass(frozen=True)
class MatchedTripletMSA:
  """3 alns ordered by shared triplet matching table"""

  a: ComponentMSA
  b: ComponentMSA
  c: ComponentMSA
  matches: tuple[TripletMatch, ...]

  def __post_init__(self) -> None:
    if (
      self.a.name,
      self.b.name,
      self.c.name,
    ) != ("A", "B", "C"):
      raise ValueError(
        "components must be named a b c in this order"
      )

    matches = tuple(self.matches)

    if not matches or any(
      not isinstance(m, TripletMatch)
      for m in matches
    ):
      raise ValueError(
        "at least one valid TripletMatch is required"
      )

    if len({
      m.observation_id for m in matches
    }) != len(matches):
      raise ValueError(
        "duplicate observation ids in matching table"
      )

    if not (
      self.a.n_sequences
      == self.b.n_sequences
      == self.c.n_sequences
      == len(matches)
    ):
      raise ValueError(
        "each msa must have exactly 1 row per matched observation"
      )

    for component, field in (
      (self.a, "a_id"),
      (self.b, "b_id"),
      (self.c, "c_id"),
    ):
      expected = tuple(
        getattr(m, field)
        for m in matches
      )

      if component.ids != expected:
        raise ValueError(
          f"{component.name} aln isnt in triplet match row order, reorder w ComponentMSA.take_ids() first."
        )

    object.__setattr__(self, "matches", matches)

  @property
  def n_observations(self) -> int:
    """number of matched evol observations"""
    return len(self.matches)

  @property
  def components(
    self,
  ) -> tuple[ComponentMSA, ComponentMSA, ComponentMSA]:
    return (self.a, self.b, self.c)

  @cached_property
  def matrix(self) -> np.ndarray:
    """concatenate a b c using shared row order"""

    result = np.concatenate(
      [part.matrix for part in self.components],
      axis=1,
    )

    result.setflags(write=False)

    return result
