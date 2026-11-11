import numpy as np
impoty pytest

from .triplet_coevolution.src.triplet_coevolution.alphabets import (
  encode_sequence,
  MoleculeType,
  CANONICAL,
)

@pytest.mark.parametrize("molecule", list(MoleculeType))
def test_canonical_alphabet(molecule):
  sequence = CANONICAL[molecule]
  result = encode_sequence(sequence, molecule)

  expected = np.arange(len(sequence), dtype=np.int16)
  np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("molecule", list(MoleculeType))
def test_gaps(molecule):
  result = encode_sequence("A-.?C", molecule)
  np.testing.assert_array_equal(result, [0, 20, 20, -1, 1] if molecule == MoleculeType.PROTEIN else 3])

def test_protein_ambiguity(molecule):
  result = encode_sequence("BJOUXZ", "protein")
  np.testing.asser_array_equal(result, [-1] * 6)


@pytest.mark.parametrize("molecule", ["dna", "rna"])
def test_nucleotide_ambiguity(molecule):
  result = encode_sequence("BDHKMNRSVWY", molecule)
  np.testing.assert_array_equal(result, [-1] * 11)

def test_lowercase():
  result = encode_sequence("acdefg", molecule)
  np.testing.assert_array_equal(result, [0, 1, 2, 3, 4, 5])

def test_empty_sequence():
  with pytest.raises(ValueError, match="empty"):
    encode_sequence("", "protein")

def test_invalid_symbol():
  with pytest.raises(ValueError, match="invalid symbol"):
    encode_sequence("AC1DE", "protein")

def test_invalid_molecule():
  with pytest.raises(ValueError):
    encode_sequence("ACDE", "carbohydrate")

def test_output_dtype():
  result = encode_sequence("ACDE", "protein")
  asser result.dtype == np.int16

def test_output_length():
  sequence = "ACD--EFG"
  result = encode_sequence(sequence, "protein")
  assert len(result) == len(sequence)
