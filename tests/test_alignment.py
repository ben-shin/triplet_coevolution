# tests for core/alignment.py

from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from triplet_coevolution.core.alignment import (
  ComponentMSA,
  MatchedTripletMSA,
  TripletMatch,
)
from triplet_coevolution.core.alphabets import MoleculeType

# reusable example - 2 observations with protein, dna, rna components
@pytest.fixture
def components():
  a = ComponentMSA(
    "A", "protein",
    ("a1", "a2"),
    ("ACD-", "AGD?"),
  )
  b = ComponentMSA(
    "B", "dna",
    ("b1", "b2"),
    ("ACGT", "A-GT"),
  )
  c = ComponentMSA(
    "C", "rna",
    ("c1", "c2"),
    ("AGCU", "AG-U"),
  )
  return a, b, c

# explicit corresp between homs across components
@pytest.fixture
def matches():
  return (
    TripletMatch("obs1", "a1", "b1", "c1"),
    TripletMatch("obs2", "a2", "b2", "c2"),
  )

# basic component properties
def test_component_dimensions(components):
  a, _, _ = components

  assert a.name == "A"
  assert a.molecule == MoleculeType.PROTEIN
  assert a.n_sequences == 2
  assert a.n_sites == 4
  assert a.matrix.shape == (2, 4)
  assert a.matrix.dtype == np.int16


# residues/gaps/missing use different codes
def test_component_encoding(components):
  a, b, c = components

  np.testing.assert_array_equal(a.matrix[0], [0, 1, 2, 20])
  np.testing.assert_array_equal(a.matrix[1], [0, 5, 2, -1])
  np.testing.assert_array_equal(b.matrix[1], [0, 4, 2, 3])
  np.testing.assert_array_equal(c.matrix[1], [0, 2, 4, 3])


# normalize lowercase, list should be tuples
@pytest.mark.parametrize(
  "molecule,sequence",
  [
    ("protein", "ac-?"),
    ("dna", "ac-?"),
    ("rna", "ac-?"),
  ],
)
def test_normalize_input(molecule, sequence):
  a = ComponentMSA("A", molecule, ["x"], [sequence])

  assert a.ids == ("x",)
  assert a.sequences == ("AC-?",)
  assert a.molecule == MoleculeType(molecule)
  

# cached matrix shouldn't be modified
def test_matrix_cached_and_read_only(components):
  a, _, _ = components

  assert a.matrix is a.matrix
  assert not a.matrix.flags.writeable

  with pytest.raises(ValueError):
    a.matrix[0, 0] = 10


# dataclass fields shouldn't be reassigned
def test_component_frozen(components):
  a, _, _ = components

  with pytest.raises(FrozenInstanceError):
    a.name = "B"


# take_ids should reorder rows based on ids not original positions
def test_take_ids_reorder(components):
  a, _, _ = components
  reordered = a.take_ids(["a2", "a1"])

  assert reordered.ids == ("a2", "a1")
  assert reordered.sequences == ("AGD?", "ACD-")
  np.testing.assert_array_equal(reordered.matrix[0], a.matrix[1])
  assert a.ids == ("a1", "a2")  # original remains unchanged


# should be able to select subset of homs
def test_take_ids_subset(components):
  a, _, _ = components
  subset = a.take_ids(["a2"])

  assert subset.ids == ("a2",)
  assert subset.matrix.shape == (1, 4)


# reject invalid component names
@pytest.mark.parametrize("name", ["D", "a", ""])
def test_invalid_component_name(name):
  with pytest.raises(ValueError, match="name"):
    ComponentMSA(name, "protein", ("x",), ("AC",))


# reject missing ids, duplicate ids, and nonrectangular alns
@pytest.mark.parametrize(
  "ids,sequences",
  [
    ((), ()),                           # empty MSA
    (("x",), ("AC", "AG")),             # unequal row counts
    (("x", "x"), ("AC", "AG")),         # duplicate IDs
    (("",), ("AC",)),                   # empty ID
    ((" ",), ("AC",)),                  # whitespace-only ID
    ((10,), ("AC",)),                   # non-string ID
    (("x", "y"), ("AC", "A")),          # unequal sequence lengths
    (("x",), ("",)),                    # empty sequence
    (("x",), (123,)),                   # non-string sequence
  ],
)
def test_invalid_component_rows(ids, sequences):
  with pytest.raises(ValueError):
    ComponentMSA("A", "protein", ids, sequences)


# invalid seq symbols should be rejected by alphabets.py
@pytest.mark.parametrize(
  "molecule,sequence",
  [
    ("protein", "AC1D"),
    ("dna", "ACGU"),      # dna must not contain U
    ("rna", "ACGT"),      # rna must not contain T
  ],
)
def test_invalid_component_sequence(molecule, sequence):
  with pytest.raises(ValueError, match="invalid seq"):
    ComponentMSA("A", molecule, ("x",), (sequence,))


# invalid molecule types should be rejected
def test_invalid_molecule_type():
  with pytest.raises(ValueError):
    ComponentMSA("A", "carbohydrate", ("x",), ("AC",))


# take_ids should reject invalid selections
# valid triplet should preserve matching table and observation count
# combined matrix is a b c with rows in matched order
# 3 components can have different aln lengths
# all 27 prot/dna/rna combinations should be allowed
# input row order has to agree with explicit matching table
# fixing row order with take_ids should allow construction
# a b c supplied in exact order
# missing or invalid triplet match records should fail
# observation ids need to be unique
# number of rows has to equal number of observations
# reject match records with unknown seq ids
# list of matches need to normalize to a tuple
