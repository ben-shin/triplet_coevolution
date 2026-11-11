# alphabet definitions and numerical sequence encoding

from enum import Enum
import numpy as np

class MoleculeType(str, Enum):
  PROTEIN = "protein"
  DNA = "dna"
  RNA = "rna"

CANONICAL = {
  MoleculeType.PROTEIN: "ACDEFGHIKLMNPQRSTVWY",
  MoleculeType.DNA: "ATGC",
  MoleculeType.RNA: "AUGC",
}

AMBIGUOUS = {
  MoleculeType.PROTEIN: "BZJXUO",
  MoleculeType.DNA: "YRSWMKVHDB",
  MoleculeType.RNA: "YRSWMKVHDB",
}

MISSING = set("-.?")

def encode_sequence(
  sequence: str,
  molecule: MoleculeType | str,
) -> np.ndarray:
  """encode sequence as integers
  canoncial = nonnegative
  gaps/ambiguous = -1
  invalid gives ValueError"""
  molecule = MoleculeType(molecule)

  if not sequence:
    raise ValueError("sequence can't be empty")

  alphabet = {
    letter: i
    for i, letter in enumerate(CANONICAL[molecule])
  }

  result = np.empty(len(sequence), dtype=np.int16)

  for i, symbol in enumerate(sequence.upper()):
    if symbol in alphabet:
      result[i] = alphabet[symbol]
    elif symbol in AMBIGUOUS[molecule] or symbol in MISSING:
      result[i] = -1
    else:
      raise ValueError(
        f"invalid symbol {symbol!r} at pos {i + 1} for {molecule.value}"

  return result
