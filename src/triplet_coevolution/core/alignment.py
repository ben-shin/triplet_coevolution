# sequence alignments for 3 body coev

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from triplet_coevolution.src.triplet_coevolution.core.alphabets import (
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
    if 
