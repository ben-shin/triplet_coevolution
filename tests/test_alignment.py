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

# FINISH TOMORROW
# basic component properties
# residues/gaps/missing use different codes
# normalize lowercase, list should be tuples
# cached matrix shouldn't be modified
# dataclass fields shouldn't be reassigned
# take_ids should reorder rows based on ids not original positions
# should be able to select subset of homs
# reject invalid component names
# reject missing ids, duplicate ids, and nonrectangular alns
# invalid seq symbols should be rejected by alphabets.py
# invalid molecule types should be rejected
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
