
# tests for inference/mi.py weighted mi and apc

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.inference.mi import (
  weighted_mi,
  mutual_information_block,
  apc,
)


# perfect binary coupling should give one bit
def test_perfect_binary_mi():
  x = np.array([0, 1, 0, 1])
  y = np.array([2, 3, 2, 3])

  assert weighted_mi(x, y, 20, 4) == pytest.approx(1)


# independent binary states should give zero mi
def test_independent_binary_mi():
  x = np.array([0, 0, 1, 1])
  y = np.array([2, 3, 2, 3])

  assert weighted_mi(x, y, 20, 4) == pytest.approx(0)


# known 40/10/10/40 distribution gives 0.278 bits
def test_known_contingency_table():
  x = np.array([0] * 40 + [0] * 10 + [1] * 10 + [1] * 40)
  y = np.array([2] * 40 + [3] * 10 + [2] * 10 + [3] * 40)

  assert weighted_mi(x, y, 20, 4) == pytest.approx(0.2780719051)


# nonuniform weights change observed freqs
def test_weighted_state_frequencies():
  x = np.array([0, 0, 0, 1])
  y = np.array([2, 2, 2, 3])

  unweighted = weighted_mi(x, y, 20, 4)
  weighted = weighted_mi(
    x, y, 20, 4, weights=[1, 1, 1, 3]
  )

  assert unweighted == pytest.approx(0.811278124459)
  assert weighted == pytest.approx(1)


# pairwise missing data must be excluded from normalization
def test_pair_specific_weight_normalization():
  a = ComponentMSA(
    "A", "protein",
    ("a1", "a2", "a3", "a4"),
    ("A", "A", "C", "C"),
  )
  b = ComponentMSA(
    "B", "rna",
    ("b1", "b2", "b3", "b4"),
    ("G", "U", "G", "?"),
  )

  result = mutual_information_block(
    a, b, weights=[1, 1, 1, 10]
  )

  assert result.n_valid[0, 0] == 3
  assert result.valid_weight[0, 0] == pytest.approx(3)
  assert result.mi_bits[0, 0] == pytest.approx(0.2516291674)


# gaps excluded by default but can be included explicitly
def test_gap_handling():
  a = ComponentMSA(
    "A", "protein",
    ("a1", "a2", "a3", "a4"),
    ("A", "A", "-", "."),
  )
  b = ComponentMSA(
    "B", "rna",
    ("b1", "b2", "b3", "b4"),
    ("G", "G", "U", "U"),
  )

  excluded = mutual_information_block(a, b)
  included = mutual_information_block(
    a, b, include_gaps=True
  )

  assert excluded.mi_bits[0, 0] == pytest.approx(0)
  assert included.mi_bits[0, 0] == pytest.approx(1)


# check apc against known numeric values
def test_apc_numeric_example():
  matrix = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
  ])

  corrected = apc(matrix)

  np.testing.assert_allclose(
    corrected,
    [[-0.2, 0.2], [0.2, -0.2]],
  )


# all missing scores should stay nan
def test_apc_all_missing():
  matrix = np.full((2, 2), np.nan)
  result = apc(matrix)

  assert np.isnan(result).all()
