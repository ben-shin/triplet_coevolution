# tests for msa/synchronize.py and incomplete homolog search coverage

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, TripletMatch
from triplet_coevolution.io.metadata import TripletTable
from triplet_coevolution.msa.synchronize import (
  synchronize,
  unmatched_sequence_ids,
)


# 3 input MSAs: unequal hit counts and different row orders
@pytest.fixture
def components():
  a = ComponentMSA(
    "A", "protein",
    ("a_extra", "a2", "a1"),
    ("DD", "AG", "AC"),
  )
  b = ComponentMSA(
    "B", "dna",
    ("b2", "b1", "b_extra1", "b_extra2"),
    ("AT", "AC", "TT", "GG"),
  )
  c = ComponentMSA(
    "C", "rna",
    ("c1", "c_extra", "c2"),
    ("AU", "CC", "AG"),
  )
  return a, b, c


# only 2 complete triplets; unmatched hits should remain available
@pytest.fixture
def matches():
  return (
    TripletMatch("obs1", "a1", "b1", "c1"),
    TripletMatch("obs2", "a2", "b2", "c2"),
  )


# reorder according to matching table, not input row order
def test_synchronize_row_order(components, matches):
  result = synchronize(*components, matches)

  assert result.a.ids == ("a1", "a2")
  assert result.b.ids == ("b1", "b2")
  assert result.c.ids == ("c1", "c2")
  assert result.n_observations == 2
  assert result.matrix.shape == (2, 6)
  np.testing.assert_array_equal(result.matrix[0, :2], [0, 1])


# source alignments should not be modified
def test_preserve_original_msas(components, matches):
  before = tuple(x.ids for x in components)
  synchronize(*components, matches)
  assert tuple(x.ids for x in components) == before


# list of matches should work as well as a tuple
def test_list_of_matches(components, matches):
  result = synchronize(*components, list(matches))
  assert result.matches == matches


# accept TripletTable while preserving metadata in the original table
def test_table_input(components, matches):
  table = TripletTable(
    matches=matches,
    extra_columns=("taxon_id",),
    annotations=({"taxon_id": "1"}, {"taxon_id": "2"}),
  )
  result = synchronize(*components, table)
  assert result.matches == matches
  assert table.annotations[0]["taxon_id"] == "1"


# identify unmatched hits separately for each component
def test_unmatched_hit_report(components, matches):
  result = unmatched_sequence_ids(*components, matches)
  assert result == {
    "A": ("a_extra",),
    "B": ("b_extra1", "b_extra2"),
    "C": ("c_extra",),
  }


# require_all should reject extra hits when explicitly requested
def test_require_all(components, matches):
  with pytest.raises(ValueError, match="unmatched input sequences"):
    synchronize(*components, matches, require_all=True)


# allow strict mode when every sequence has a match
def test_require_all_without_extras(components, matches):
  a, b, c = components
  a = a.take_ids(["a1", "a2"])
  b = b.take_ids(["b1", "b2"])
  c = c.take_ids(["c1", "c2"])
  result = synchronize(a, b, c, matches, require_all=True)
  assert result.n_observations == 2
  assert unmatched_sequence_ids(a, b, c, matches) == {
    "A": (), "B": (), "C": (),
  }


# cannot match a sequence absent from its MSA
@pytest.mark.parametrize("index,field", [(0, "a_id"), (1, "b_id"), (2, "c_id")])
def test_missing_referenced_id(components, matches, index, field):
  from dataclasses import replace
  invalid = (
    replace(matches[0], **{field: "does_not_exist"}),
    matches[1],
  )
  with pytest.raises(ValueError, match=f"component {chr(65+index)} is missing"):
    synchronize(*components, invalid)


# reject duplicated accession assignments within a component
def test_reused_sequence_id(components, matches):
  invalid = (
    matches[0],
    TripletMatch("obs2", "a1", "b2", "c2"),
  )
  with pytest.raises(ValueError, match="duplicate a_id"):
    synchronize(*components, invalid)


# reject duplicate observation IDs
def test_duplicate_observations(components, matches):
  invalid = (
    matches[0],
    TripletMatch("obs1", "a2", "b2", "c2"),
  )
  with pytest.raises(ValueError, match="duplicate observation_id"):
    synchronize(*components, invalid)


# reject wrong component order
def test_wrong_component_order(components, matches):
  a, b, c = components
  with pytest.raises(ValueError, match="named A, B, C"):
    synchronize(b, a, c, matches)


# reject an empty matching table
def test_empty_matches(components):
  with pytest.raises(ValueError, match="no matches"):
    synchronize(*components, ())


# support a single verified triplet
def test_single_triplet(components, matches):
  result = synchronize(*components, matches[:1])
  assert result.n_observations == 1
  assert result.matrix.shape == (1, 6)
