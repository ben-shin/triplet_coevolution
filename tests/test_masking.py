
# tests for msa/masking.py col selection and original pos mapping

import numpy as np
import pytest

from triplet_coevolution.core.alignment import (
  ComponentMSA,
  MatchedTripletMSA,
  TripletMatch,
)
from triplet_coevolution.msa.qc import component_qc
from triplet_coevolution.msa.masking import (
  column_mask,
  apply_mask,
  apply_triplet_masks,
)


# fixed, variable, all gap, and all missing prot cols
@pytest.fixture
def protein_msa():
  return ComponentMSA(
    "A", "protein",
    ("a1", "a2", "a3", "a4"),
    ("AA-?", "AC-?", "AA-?", "AC-?"),
  )


# mixed prot, dna, and rna components with valid matching
@pytest.fixture
def triplet_msa():
  a = ComponentMSA("A", "protein", ("a1", "a2"), ("AA-?", "AC-?"))
  b = ComponentMSA("B", "dna", ("b1", "b2"), ("ACG", "A-G"))
  c = ComponentMSA("C", "rna", ("c1", "c2"), ("AU?", "AG?"))
  matches = (
    TripletMatch("obs1", "a1", "b1", "c1"),
    TripletMatch("obs2", "a2", "b2", "c2"),
  )
  return MatchedTripletMSA(a, b, c, matches)


# basic qc filters keep only cols with enough canonical res
def test_default_column_mask(protein_msa):
  result = column_mask(component_qc(protein_msa))

  np.testing.assert_array_equal(result, [True, True, False, False])
  assert result.dtype == np.bool_
  assert not result.flags.writeable


# conserved cols should be kept by default
def test_keep_conserved_column(protein_msa):
  result = column_mask(component_qc(protein_msa))
  assert result[0]


# optional min entropy excludes conserved cols but keeps variable ones
def test_min_entropy_filter(protein_msa):
  result = column_mask(
    component_qc(protein_msa), min_entropy_bits=0.5
  )
  np.testing.assert_array_equal(result, [False, True, False, False])


# all gap and all missing cols excluded even when thresholds are permissive
def test_exclude_no_canonical_columns(protein_msa):
  result = column_mask(
    component_qc(protein_msa),
    min_coverage=0,
    max_gap_fraction=1,
    max_missing_fraction=1,
  )
  np.testing.assert_array_equal(result, [True, True, False, False])


# threshold comparisons are inclusive at the boundary
def test_threshold_boundaries():
  msa = ComponentMSA(
    "B", "dna", ("b1", "b2", "b3", "b4"), ("A", "A", "-", "?")
  )
  qc = component_qc(msa)

  assert column_mask(qc, 0.5, 0.25, 0.25)[0]
  assert not column_mask(qc, min_coverage=0.51)[0]
  assert not column_mask(qc, max_gap_fraction=0.24)[0]
  assert not column_mask(qc, max_missing_fraction=0.24)[0]


# reject invalid fraction thresholds rather than silently accepting them
@pytest.mark.parametrize(
  "field,value",
  [
    ("min_coverage", -0.1),
    ("min_coverage", 1.01),
    ("max_gap_fraction", float("nan")),
    ("max_missing_fraction", float("inf")),
    ("max_missing_fraction", True),
    ("max_gap_fraction", "0.5"),
  ],
)
def test_invalid_fraction_thresholds(protein_msa, field, value):
  with pytest.raises(ValueError, match=field):
    column_mask(component_qc(protein_msa), **{field: value})


# reject invalid entropy thresholds
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True])
def test_invalid_entropy_threshold(protein_msa, value):
  with pytest.raises(ValueError, match="min_entropy_bits"):
    column_mask(component_qc(protein_msa), min_entropy_bits=value)


# qc input should use ComponentQC rather than the msa itself
def test_invalid_qc_input(protein_msa):
  with pytest.raises(TypeError, match="ComponentQC"):
    column_mask(protein_msa)


# applying a mask removes specific cols and retains original msa col indices
def test_apply_mask_and_original_positions(protein_msa):
  result = apply_mask(protein_msa, [True, True, False, False])

  assert result.msa.sequences == ("AA", "AC", "AA", "AC")
  assert result.msa.ids == protein_msa.ids
  assert result.msa.n_sites == 2
  assert result.original_columns == (0, 1)
  assert result.original_positions == (1, 2)


# nonconsecutive cols should preserve input order and exact mapping
def test_nonconsecutive_columns(protein_msa):
  result = apply_mask(protein_msa, [True, False, True, False])

  assert result.msa.sequences == ("A-", "A-", "A-", "A-")
  assert result.original_columns == (0, 2)
  assert result.original_positions == (1, 3)


# applying None keeps all cols, without losing the positions
def test_none_keeps_all_columns(protein_msa):
  result = apply_mask(protein_msa, None)
  assert result.msa.sequences == protein_msa.sequences
  assert result.original_columns == (0, 1, 2, 3)


# masked arrays should not modify input seqs or matrices
def test_original_msa_unchanged(protein_msa):
  before = protein_msa.matrix.copy()
  result = apply_mask(protein_msa, [True, True, False, False])
  np.testing.assert_array_equal(protein_msa.matrix, before)
  assert protein_msa.sequences[0] == "AA-?"
  assert result.msa.matrix.shape == (4, 2)


# reject masks with incorrect length or number of dimensions
@pytest.mark.parametrize(
  "mask",
  [
    [True, False],
    [True] * 5,
    [[True, False], [True, False]],
  ],
)
def test_invalid_mask_shape(protein_msa, mask):
  with pytest.raises(ValueError, match="mask length"):
    apply_mask(protein_msa, mask)


# reject numeric masks that could accidentally be interpreted as bool
@pytest.mark.parametrize("mask", [[1, 1, 0, 0], ["T", "T", "F", "F"]])
def test_invalid_mask_dtype(protein_msa, mask):
  with pytest.raises(TypeError, match="boolean"):
    apply_mask(protein_msa, mask)


# all cols excluded should raise, rather than create an empty msa
def test_reject_masking_every_column(protein_msa):
  with pytest.raises(ValueError, match="all msa cols"):
    apply_mask(protein_msa, [False] * 4)


# reject wrong msa type
def test_reject_wrong_component_input():
  with pytest.raises(TypeError, match="ComponentMSA"):
    apply_mask("AGCU", [True] * 4)


# independently mask each component without changing shared row matches
def test_apply_triplet_masks(triplet_msa):
  result = apply_triplet_masks(
    triplet_msa,
    a_mask=[True, True, False, False],
    b_mask=[True, False, True],
    c_mask=[True, True, False],
  )

  assert result.msa.n_observations == 2
  assert result.msa.matches == triplet_msa.matches
  assert (result.msa.a.n_sites, result.msa.b.n_sites, result.msa.c.n_sites) == (2, 2, 2)
  assert result.msa.a.sequences == ("AA", "AC")
  assert result.msa.b.sequences == ("AG", "AG")
  assert result.msa.c.sequences == ("AU", "AG")
  assert result.msa.matrix.shape == (2, 6)
  assert result.a_columns == (0, 1)
  assert result.b_columns == (0, 2)
  assert result.c_columns == (0, 1)


# missing mask values should leave those components unchanged
def test_partial_triplet_mask(triplet_msa):
  result = apply_triplet_masks(
    triplet_msa, a_mask=[True, True, False, False]
  )

  assert result.msa.a.n_sites == 2
  assert result.msa.b.n_sites == triplet_msa.b.n_sites
  assert result.msa.c.n_sites == triplet_msa.c.n_sites
  assert result.b_columns == (0, 1, 2)
  assert result.c_columns == (0, 1, 2)


# wrong triplet input raises TypeError
def test_reject_wrong_triplet_input():
  with pytest.raises(TypeError, match="MatchedTripletMSA"):
    apply_triplet_masks("not a triplet")


# no component should be allowed to become empty after filtering
def test_reject_empty_triplet_component(triplet_msa):
  with pytest.raises(ValueError, match="all msa cols"):
    apply_triplet_masks(triplet_msa, c_mask=[False, False, False])


# qc masks should work directly with synchronized triplet alns
def test_qc_masks_to_triplet(triplet_msa):
  a_keep = column_mask(component_qc(triplet_msa.a), min_coverage=1)
  b_keep = column_mask(component_qc(triplet_msa.b), min_coverage=1)
  c_keep = column_mask(component_qc(triplet_msa.c), min_coverage=1)

  result = apply_triplet_masks(triplet_msa, a_keep, b_keep, c_keep)

  assert result.a_columns == (0, 1)
  assert result.b_columns == (0, 2)
  assert result.c_columns == (0, 1)
  assert result.msa.matrix.shape == (2, 6)
