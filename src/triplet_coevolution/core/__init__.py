
# core sequence and alignment data structures

from .alphabets import (
  MoleculeType,
  CANONICAL,
  AMBIGUOUS,
  GAPS,
  MISSING,
  UNKNOWN_CODE,
  encode_sequence,
)

from .alignment import (
  ComponentMSA,
  TripletMatch,
  MatchedTripletMSA,
)


# public objects available from triplet_coevolution.core
__all__ = [
  "MoleculeType",
  "CANONICAL",
  "AMBIGUOUS",
  "GAPS",
  "MISSING",
  "UNKNOWN_CODE",
  "encode_sequence",
  "ComponentMSA",
  "TripletMatch",
  "MatchedTripletMSA",
]
