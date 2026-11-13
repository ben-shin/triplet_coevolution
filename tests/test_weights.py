
# tests for inference/weights.py redundancy and effective msa depth

import numpy as np
import pytest

from triplet_coevolution.core.alignment import (
  ComponentMSA,
  MatchedTripletMSA,
  TripletMatch,
)
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.msa.masking import apply_triplet_masks


# three obs: first two identical, third distinct
@pytest.fixture
def triplet_msa():
  a = ComponentMSA("A", "protein", ("a1", "a2", "a3"), ("AAAA", "AAAA", "CCCC"))
  b = ComponentMSA("B", "dna", ("b1", "b2", "b3"), ("AAAA", "AAAA", "CCCC"))
  c = ComponentMSA("C", "rna", ("c1", "c2", "c3"), ("AAAA", "AAAA", "CCCC"))
  matches = (
    TripletMatch("obs1", "a1", "b1", "c1"),
    TripletMatch("obs2", "a2", "b2", "c2"),
    TripletMatch("obs3", "a3", "b3", "c3"),
  )
  return MatchedTripletMSA(a, b, c, matches)


# identical obs should share statistical weight
@pytest.mark.parametrize("use_triplet", [False, True])
def test_identical_homologs(triplet_msa, use_triplet):
  msa = triplet_msa if use_triplet else triplet_msa.a
  result = sequence_weights(msa)

  np.testing.assert_array_equal(result.neighbor_counts, [2, 2, 1])
  np.testing.assert_allclose(result.weights, [0.5, 0.5, 1])
  assert result.n_eff == pytest.approx(2)
  assert result.n_observations == 3


# completely distinct obs retain full weight
def test_unrelated_homologs():
  msa = ComponentMSA("A", "rna", ("a1", "a2", "a3"), ("AAAA", "CCCC", "GGGG"))
  result = sequence_weights(msa)

  np.testing.assert_allclose(result.weights, [1, 1, 1])
  assert result.n_eff == pytest.approx(3)


# exactly threshold identity should count as similar
def test_inclusive_identity_threshold():
  msa = ComponentMSA("A", "protein", ("a1", "a2"), ("AAAA", "AAAC"))

  result = sequence_weights(msa, identity_threshold=0.75)
  np.testing.assert_array_equal(result.neighbor_counts, [2, 2])
  assert result.n_eff == pytest.approx(1)

  strict = sequence_weights(msa, identity_threshold=0.76)
  np.testing.assert_array_equal(strict.neighbor_counts, [1, 1])


# all gap positions must not inflate sequence similarity
def test_gaps_not_counted_as_matches():
  msa = ComponentMSA("A", "protein", ("a1", "a2"), ("A---", "C---"))
  result = sequence_weights(msa, identity_threshold=0.8, min_overlap=0.25)

  np.testing.assert_array_equal(result.neighbor_counts, [1, 1])


# missing and gap states should not be used as canonical res
def test_unknown_and_gap_ignored():
  msa = ComponentMSA("B", "dna", ("b1", "b2"), ("AC-?", "AC.N"))
  result = sequence_weights(msa, min_overlap=0.5)

  np.testing.assert_array_equal(result.neighbor_counts, [2, 2])
  assert result.n_eff == pytest.approx(1)


# short shared stretches should not create spurious redundancy
def test_min_overlap_rejects_tiny_region():
  msa = ComponentMSA("A", "protein", ("a1", "a2"), ("A???", "A---"))

  default = sequence_weights(msa)
  relaxed = sequence_weights(msa, min_overlap=0.25)
  np.testing.assert_array_equal(default.neighbor_counts, [1, 1])
  np.testing.assert_array_equal(relaxed.neighbor_counts, [2, 2])


# short rna/dna should contribute equally to long prot in triplet mode
def test_balanced_component_identity():
  a = ComponentMSA("A", "protein", ("a1", "a2"), ("AAAAAAAAAA", "AAAAAAAAAA"))
  b = ComponentMSA("B", "dna", ("b1", "b2"), ("AC", "GT"))
  c = ComponentMSA("C", "rna", ("c1", "c2"), ("AC", "GU"))
  matches = (
    TripletMatch("o1", "a1", "b1", "c1"),
    TripletMatch("o2", "a2", "b2", "c2"),
  )
  result = sequence_weights(MatchedTripletMSA(a, b, c, matches))

  # long prot identity alone must not dominate two discordant nt components
  np.testing.assert_array_equal(result.neighbor_counts, [1, 1])


# min overlap must hold independently for every component
def test_each_component_requires_overlap(triplet_msa):
  a, b, c = triplet_msa.components
  altered = ComponentMSA("C", "rna", c.ids, ("A???", "A---", "CCCC"))
  result = sequence_weights(
    MatchedTripletMSA(a, b, altered, triplet_msa.matches),
    min_overlap=0.5,
  )
  np.testing.assert_array_equal(result.neighbor_counts, [1, 1, 1])


# one obs always has one neighbor and full weight
def test_single_observation():
  msa = ComponentMSA("B", "dna", ("b1",), ("ACGT",))
  result = sequence_weights(msa)

  np.testing.assert_array_equal(result.neighbor_counts, [1])
  np.testing.assert_allclose(result.weights, [1])
  assert result.n_eff == pytest.approx(1)


# should still work after filtering aln cols with masking.py
def test_weights_after_masking(triplet_msa):
  filtered = apply_triplet_masks(
    triplet_msa,
    a_mask=[True, True, False, False],
    b_mask=[True, True, False, False],
    c_mask=[True, True, False, False],
  )
  result = sequence_weights(filtered.msa)

  assert result.n_observations == 3
  np.testing.assert_allclose(result.weights, [0.5, 0.5, 1])


# arrays should be read only to avoid accidental overwrites
def test_results_read_only(triplet_msa):
  result = sequence_weights(triplet_msa)

  assert not result.weights.flags.writeable
  assert not result.neighbor_counts.flags.writeable
  with pytest.raises(ValueError):
    result.weights[0] = 100


# original msa rows and encoded seqs should not change
def test_no_input_mutation(triplet_msa):
  original = triplet_msa.matrix.copy()
  sequence_weights(triplet_msa)
  np.testing.assert_array_equal(triplet_msa.matrix, original)


# invalid identity and overlap parameters should fail
@pytest.mark.parametrize("field,value", [
  ("identity_threshold", -0.01),
  ("identity_threshold", 1.01),
  ("identity_threshold", float("nan")),
  ("identity_threshold", True),
  ("min_overlap", -0.1),
  ("min_overlap", float("inf")),
  ("min_overlap", "0.5"),
])
def test_bad_thresholds(triplet_msa, field, value):
  with pytest.raises(ValueError, match=field):
    sequence_weights(triplet_msa, **{field: value})


# cannot calculate sequence identity if no canonical res in a seq
def test_reject_no_canonical_res():
  msa = ComponentMSA("A", "protein", ("a1", "a2"), ("????", "AAAA"))
  with pytest.raises(ValueError, match="no canonical res"):
    sequence_weights(msa)


# reject arbitrary data instead of accepting wrong shapes
def test_reject_wrong_input():
  with pytest.raises(TypeError, match="ComponentMSA or MatchedTripletMSA"):
    sequence_weights("AAAA")
