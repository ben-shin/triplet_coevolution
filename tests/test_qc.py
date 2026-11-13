
# tests for msa/qc.py aln depth, coverage, and conservation

import numpy as np
import pytest

from triplet_coevolution.core.alignment import (
  ComponentMSA,
  MatchedTripletMSA,
  TripletMatch,
)
from triplet_coevolution.core.alphabets import MoleculeType
from triplet_coevolution.msa.qc import component_qc, triplet_qc


# prot example: fixed, variable, all gap, and all missing cols
@pytest.fixture
def protein_msa():
  return ComponentMSA(
    "A", "protein",
    ("a1", "a2", "a3", "a4"),
    ("AA-?", "AC-?", "AA-?", "AC-?"),
  )


# mixed canonical, gap, and ambiguous nt symbols
@pytest.fixture
def dna_msa():
  return ComponentMSA(
    "B", "dna",
    ("b1", "b2", "b3", "b4"),
    ("A-?", "AC-", "ANC", "A.C"),
  )


# fully matched protein/dna/rna data with unequal component lengths
@pytest.fixture
def triplet_msa():
  a = ComponentMSA(
    "A", "protein",
    ("a1", "a2", "a3"),
    ("AAA", "AAC", "A-A"),
  )
  b = ComponentMSA(
    "B", "dna",
    ("b1", "b2", "b3"),
    ("AC", "A-", "AN"),
  )
  c = ComponentMSA(
    "C", "rna",
    ("c1", "c2", "c3"),
    ("AGCU", "AG-U", "AG?U"),
  )
  matches = (
    TripletMatch("obs1", "a1", "b1", "c1"),
    TripletMatch("obs2", "a2", "b2", "c2"),
    TripletMatch("obs3", "a3", "b3", "c3"),
  )
  return MatchedTripletMSA(a, b, c, matches)


# basic qc result fields and dimensions
def test_component_qc_dimensions(protein_msa):
  result = component_qc(protein_msa)

  assert result.name == "A"
  assert result.molecule == MoleculeType.PROTEIN
  assert result.n_sequences == 4
  assert result.n_sites == 4
  assert result.coverage.shape == (4,)
  assert result.n_canonical.shape == (4,)


# count canon res separately from gaps and unknowns
def test_canonical_counts(protein_msa):
  result = component_qc(protein_msa)

  np.testing.assert_array_equal(result.n_canonical, [4, 4, 0, 0])
  np.testing.assert_array_equal(result.coverage, [1, 1, 0, 0])
  np.testing.assert_array_equal(result.gap_fraction, [0, 0, 1, 0])
  np.testing.assert_array_equal(result.missing_fraction, [0, 0, 0, 1])


# fixed canon col has zero entropy and full conservation
def test_conserved_column(protein_msa):
  result = component_qc(protein_msa)

  assert result.entropy_bits[0] == pytest.approx(0)
  assert result.conservation[0] == pytest.approx(1)


# equal frequencies of a and c give one bit of entropy
def test_variable_column(protein_msa):
  result = component_qc(protein_msa)

  assert result.entropy_bits[1] == pytest.approx(1)
  assert result.conservation[1] == pytest.approx(0.5)


# an all gap or all missing column has undefined entropy
def test_no_canonical_residues(protein_msa):
  result = component_qc(protein_msa)

  assert np.isnan(result.entropy_bits[2:]).all()
  assert np.isnan(result.conservation[2:]).all()


# mixed nt symbols are counted using the correct denoms
def test_mixed_nucleotide_coverage(dna_msa):
  result = component_qc(dna_msa)

  np.testing.assert_array_equal(result.n_canonical, [4, 1, 2])
  np.testing.assert_allclose(result.coverage, [1, 0.25, 0.5])
  np.testing.assert_allclose(result.gap_fraction, [0, 0.5, 0.25])
  np.testing.assert_allclose(result.missing_fraction, [0, 0.25, 0.25])


# coverage, gaps, and missing fractions must sum to one
@pytest.mark.parametrize("molecule,sequences", [
  ("protein", ("A-?", "ACD", "A.?")),
  ("dna", ("A-?", "ACG", "A.N")),
  ("rna", ("A-?", "ACG", "A.N")),
])
def test_fractions_partition_columns(molecule, sequences):
  msa = ComponentMSA("A", molecule, ("x", "y", "z"), sequences)
  result = component_qc(msa)

  np.testing.assert_allclose(
    result.coverage + result.gap_fraction + result.missing_fraction,
    np.ones(msa.n_sites),
  )


# acgu at equal freqs give two bits of entropy
def test_rna_four_state_entropy():
  msa = ComponentMSA(
    "C", "rna", ("x1", "x2", "x3", "x4"),
    ("A", "C", "G", "U"),
  )
  result = component_qc(msa)

  assert result.entropy_bits[0] == pytest.approx(2)
  assert result.conservation[0] == pytest.approx(0.25)


# shannon entropy agrees with a known 3:1 freqs distribution
def test_uneven_state_frequencies():
  msa = ComponentMSA(
    "C", "rna", ("x1", "x2", "x3", "x4"),
    ("A", "A", "A", "C"),
  )
  result = component_qc(msa)

  assert result.entropy_bits[0] == pytest.approx(0.811278124459)
  assert result.conservation[0] == pytest.approx(0.75)


# 10. one observed canonical residue is not the same as full coverage
def test_one_canonical_residue():
  msa = ComponentMSA(
    "B", "dna", ("x", "y", "z", "w"),
    ("A", "-", "?", "N"),
  )
  result = component_qc(msa)

  assert result.coverage[0] == pytest.approx(0.25)
  assert result.entropy_bits[0] == pytest.approx(0)
  assert result.conservation[0] == pytest.approx(1)


# . and - have equiv gap semantics in alphabet encoding
def test_dot_and_dash_as_gaps():
  msa = ComponentMSA(
    "C", "rna", ("x", "y", "z"), (".", "-", "A"),
  )
  result = component_qc(msa)

  assert result.gap_fraction[0] == pytest.approx(2 / 3)
  assert result.coverage[0] == pytest.approx(1 / 3)


# mean metrics summarize the full aln including empty cols
def test_mean_metrics(protein_msa):
  result = component_qc(protein_msa)

  assert result.mean_coverage == pytest.approx(0.5)
  assert result.mean_gap_fraction == pytest.approx(0.25)
  assert result.mean_missing_fraction == pytest.approx(0.25)


# qc arrays should be read only to avoid accidental modification
def test_qc_arrays_are_read_only(protein_msa):
  result = component_qc(protein_msa)

  for array in (
    result.n_canonical, result.gap_fraction,
    result.missing_fraction, result.coverage,
    result.entropy_bits, result.conservation,
  ):
    assert not array.flags.writeable
    with pytest.raises(ValueError):
      array[0] = 10


# qc must not change the original encoded aln
def test_input_msa_is_unchanged(protein_msa):
  original = protein_msa.matrix.copy()
  component_qc(protein_msa)

  np.testing.assert_array_equal(protein_msa.matrix, original)


# triplet qc keeps a b c separate with shared obs depth
def test_triplet_qc(triplet_msa):
  result = triplet_qc(triplet_msa)

  assert result.n_observations == 3
  assert result.n_sites == 9
  assert tuple(x.name for x in result.components) == ("A", "B", "C")
  assert (result.a.n_sites, result.b.n_sites, result.c.n_sites) == (3, 2, 4)
  assert result.a.n_sequences == result.b.n_sequences == result.c.n_sequences == 3


# triplet qc results equal independently calculated comp qc
def test_triplet_equals_component_qc(triplet_msa):
  result = triplet_qc(triplet_msa)

  for part, qc in zip(triplet_msa.components, result.components):
    expected = component_qc(part)
    np.testing.assert_allclose(qc.coverage, expected.coverage)
    np.testing.assert_allclose(qc.entropy_bits, expected.entropy_bits, equal_nan=True)


# qc entry points should reject unsupported inputs
def test_reject_wrong_input_type():
  with pytest.raises(TypeError, match="ComponentMSA"):
    component_qc("ACGU")

  with pytest.raises(TypeError, match="MatchedTripletMSA"):
    triplet_qc("not an alignment")
