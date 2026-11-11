# alphabet definitions and numerical sequence encoding

from enum import Enum
import numpy as np

class MoleculeType(str, Enum):
  PROTEIN = "protein"
  DNA = "dna"
  RNA = "rna"

CANONICAL = {
  MoleculeType.PROTEIN: "ACDEFGHIKLMNPQRSTVWY",
  MoleculeType.DNA: "ACGT",
  MoleculeType.RNA: "ACGU",
}

AMBIGUOUS = {
  MoleculeType.PROTEIN: "BJOUXZ",
  MoleculeType.DNA: "BDHKMNRSVWY",
  MoleculeType.RNA: "BDHKMNRSVWY",
}

GAPS = set("-.")
MISSING = set("?")
UNKNOWN_CODE = -1

def encode_sequence(
  sequence: str,
  molecule: MoleculeType | str,
) -> np.ndarray:
  """encode sequence as integers
  canoncial = nonnegative
  gaps = 4 for dna/rna, 20 for protein
  ambiguous/missing = -1"""
  molecule = MoleculeType(molecule)

  if not sequence:
    raise ValueError("sequence can't be empty")

  alphabet = {
    letter: i
    for i, letter in enumerate(CANONICAL[molecule])
  }

  gap_code = len(alphabet)

  result = np.empty(len(sequence), dtype=np.int16)

  for i, symbol in enumerate(sequence.upper()):
    if symbol in alphabet:
      result[i] = alphabet[symbol]

    elif symbol in GAPS:
      result[i] = gap_code
    
    elif symbol in AMBIGUOUS[molecule] or symbol in MISSING:
      result[i] = UNKNOWN_CODE
    
    else:
      raise ValueError(
        f"invalid symbol {symbol!r} at pos {i + 1} for {molecule.value}"
      )

  return result
