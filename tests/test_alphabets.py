import numpy as np
import pytest

from triplet_coevolution.core.alphabets import (
  CANONICAL,
  UNKNOWN_CODE,
  MoleculeType,
  encode_sequence,
)


# canonical encoding
@pytest.mark.parametrize("molecule", list(MoleculeType))
def test_canonical_alphabet(molecule):
  sequence = CANONICAL[molecule]
  result = encode_sequence(sequence, molecule)

  expected = np.arange(len(sequence), dtype=np.int16)
  np.testing.assert_array_equal(result, expected)


# alphabet size
@pytest.mark.parametrize(
  "molecule,expected_size",
  [
    (MoleculeType.PROTEIN, 20),
    (MoleculeType.DNA, 4),
    (MoleculeType.RNA, 4),
  ],
)
def test_alphabet_size(molecule, expected_size):
  assert len(CANONICAL[molecule]) == expected_size


# gaps and missing chars
@pytest.mark.parametrize(
  "molecule,gap_code",
  [
    (MoleculeType.PROTEIN, 20),
    (MoleculeType.DNA, 4),
    (MoleculeType.RNA, 4),
  ],
)
def test_gaps(molecule, gap_code):
  result = encode_sequence("A-.?C", molecule)

  expected = [0, gap_code, gap_code, -1, 1]
  np.testing.assert_array_equal(result, expected)


# ambiguous protein
def test_protein_ambiguity():
  result = encode_sequence("BJOUXZ", MoleculeType.PROTEIN)

  np.testing.assert_array_equal(
    result, [UNKNOWN_CODE] * 6
  )


# ambiguous nucleotide
@pytest.mark.parametrize("molecule", [
  MoleculeType.DNA,
  MoleculeType.RNA,
])
def test_nucleotide_ambiguity(molecule):
  result = encode_sequence("BDHKMNRSVWY", molecule)

  np.testing.assert_array_equal(
    result, [UNKNOWN_CODE] * 11
  )


# lowercase
@pytest.mark.parametrize("molecule", list(MoleculeType))
def test_lowercase(molecule):
  sequence = CANONICAL[molecule].lower()
  result = encode_sequence(sequence, molecule)

  expected = np.arange(len(sequence), dtype=np.int16)
  np.testing.assert_array_equal(result, expected)


# empty sequence
def test_empty_sequence():
  with pytest.raises(ValueError, match="empty"):
    encode_sequence("", MoleculeType.PROTEIN)


# invalid characters
def test_invalid_symbol():
  with pytest.raises(ValueError, match="invalid symbol"):
    encode_sequence("AC1DE", "protein")


# invalid molecule type
def test_invalid_molecule():
  with pytest.raises(ValueError):
    encode_sequence("ACDE", "carbohydrate")


# output data type
def test_output_dtype():
  result = encode_sequence("ACDE", MoleculeType.PROTEIN)

  assert isinstance(result, np.ndarray)
  assert result.dtype == np.int16


# output length
def test_output_length():
  sequence = "ACD--EFG"
  result = encode_sequence(sequence, MoleculeType.PROTEIN)

  assert len(result) == len(sequence)


# rna should reject T
def test_rna_rejects_thymine():
  with pytest.raises(ValueError, match="invalid symbol"):
    encode_sequence("ACGT", MoleculeType.RNA)


# dna should reject U
def test_dna_rejects_uracil():
  with pytest.raises(ValueError, match="invalid symbol"):
    encode_sequence("ACGU", MoleculeType.DNA)


# string molecule args
def test_string_molecule_argument():
  result = encode_sequence("ACGU", "rna")

  np.testing.assert_array_equal(result, [0, 1, 2, 3])


# missing and gaps are not the same
@pytest.mark.parametrize(
  "molecule,gap_code",
  [
    (MoleculeType.PROTEIN, 20),
    (MoleculeType.DNA, 4),
    (MoleculeType.RNA, 4),
  ],
)
def test_missing_vs_gap(molecule, gap_code):
  result = encode_sequence("A-?C", molecule)

  assert result[1] == gap_code
  assert result[2] == UNKNOWN_CODE
  assert result[1] != result[2]
