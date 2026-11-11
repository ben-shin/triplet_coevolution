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
  result = encode_sequence(


@pytest.mark.parametrize("molecule", ["dna", "rna"])
