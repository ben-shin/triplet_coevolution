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
@pytest.mark.parametrize(
  "requested",
  [
    [],
    ["a1", "a1"],
    ["a1", "unknown"],
  ],
)
def test_invalid_take_ids(components, requested):
  a, _, _ = components

  with pytest.raises(ValueError):
    a.take_ids(requested)


# valid triplet should preserve matching table and observation count
def test_valid_triplet_alignment(components, matches):
  combined = MatchedTripletMSA(*components, matches)

  assert combined.matches == matches
  assert combined.components == components
  assert combined.n_observations == 2
  assert matches[0].observation_id == "obs1"
  assert matches[0].a_id == "a1"


# combined matrix is a b c with rows in matched order
def test_triplet_matrix(components, matches):
  combined = MatchedTripletMSA(*components, matches)

  assert combined.matrix.shape == (2, 12)
  assert combined.matrix is combined.matrix
  assert not combined.matrix.flags.writeable

  np.testing.assert_array_equal(
    combined.matrix[0],
    [0, 1, 2, 20, 0, 1, 2, 3, 0, 2, 1, 3],
  )
  np.testing.assert_array_equal(
    combined.matrix,
    np.concatenate([part.matrix for part in components], axis=1),
  )


# 3 components can have different aln lengths
def test_different_component_lengths(matches):
  a = ComponentMSA("A", "protein", ("a1", "a2"), ("ACD", "AGD"))
  b = ComponentMSA("B", "dna", ("b1", "b2"), ("AC", "AG"))
  c = ComponentMSA("C", "rna", ("c1", "c2"), ("A", "U"))

  combined = MatchedTripletMSA(a, b, c, matches)
  assert combined.matrix.shape == (2, 6)


# all 27 prot/dna/rna combinations should be allowed
@pytest.mark.parametrize("molecules", list(product(MoleculeType, repeat=3)))
def test_all_molecule_combinations(molecules, matches):
  a = ComponentMSA("A", molecules[0], ("a1", "a2"), ("AC", "AG"))
  b = ComponentMSA("B", molecules[1], ("b1", "b2"), ("AC", "AG"))
  c = ComponentMSA("C", molecules[2], ("c1", "c2"), ("AC", "AG"))

  combined = MatchedTripletMSA(a, b, c, matches)
  assert combined.matrix.shape == (2, 6)
  

# input row order has to agree with explicit matching table
@pytest.mark.parametrize("component_index", [0, 1, 2])
def test_reject_wrong_triplet_row_order(components, matches, component_index):
  modified = list(components)
  ids = modified[component_index].ids
  modified[component_index] = modified[component_index].take_ids(ids[::-1])

  with pytest.raises(ValueError, match="triplet match row order"):
    MatchedTripletMSA(*modified, matches)


# fixing row order with take_ids should allow construction
def test_correct_triplet_row_order(components, matches):
  a, b, c = components
  shuffled_b = b.take_ids(["b2", "b1"])
  corrected_b = shuffled_b.take_ids([m.b_id for m in matches])

  combined = MatchedTripletMSA(a, corrected_b, c, matches)
  assert combined.b.ids == ("b1", "b2")


# a b c supplied in exact order
def test_reject_wrong_component_order(components, matches):
  a, b, c = components

  with pytest.raises(ValueError, match="components must be named"):
    MatchedTripletMSA(b, a, c, matches)


# missing or invalid triplet match records should fail
@pytest.mark.parametrize("bad_matches", [(), ("not a match",)])
def test_reject_invalid_match_records(components, bad_matches):
  with pytest.raises(ValueError, match="TripletMatch"):
    MatchedTripletMSA(*components, bad_matches)


# observation ids need to be unique
def test_reject_duplicate_observation_ids(components):
  duplicate_matches = (
    TripletMatch("obs1", "a1", "b1", "c1"),
    TripletMatch("obs1", "a2", "b2", "c2"),
  )

  with pytest.raises(ValueError, match="duplicate observation ids"):
    MatchedTripletMSA(*components, duplicate_matches)


# number of rows has to equal number of observations
def test_reject_inconsistent_row_counts(components, matches):
  a, b, c = components

  with pytest.raises(ValueError, match="exactly 1 row"):
    MatchedTripletMSA(a.take_ids(["a1"]), b, c, matches)


# reject match records with unknown seq ids
def test_reject_unknown_homolog_in_triplet(components, matches):
  invalid_matches = (
    TripletMatch("obs1", "a1", "b1", "missing_c"),
    matches[1],
  )

  with pytest.raises(ValueError, match="triplet match row order"):
    MatchedTripletMSA(*components, invalid_matches)


# list of matches need to normalize to a tuple
def test_match_list_becomes_tuple(components, matches):
  combined = MatchedTripletMSA(*components, list(matches))

  assert isinstance(combined.matches, tuple)
  assert combined.matches == matches
