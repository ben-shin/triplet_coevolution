# tests for core/coordinates.py original msa and reference res positions

import pytest

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.coordinates import (
  ReferenceCoordinates,
  reference_positions,
  build_reference_map,
  subset_reference_map,
)
from triplet_coevolution.msa.masking import apply_mask


# reference with gaps, unknown res, and nonzero numbering offset
@pytest.fixture
def protein_msa():
  return ComponentMSA(
    "A", "protein",
    ("alt", "ref"),
    ("ATCDWEQ", "A-CD.E?"),
  )


# canonical and ambiguous res advance reference position, gaps dont
@pytest.mark.parametrize(
  "sequence,expected",
  [
    ("A-C.D", (1, None, 2, None, 3)),
    ("A?X-", (1, 2, 3, None)),
    ("--A--", (None, None, 1, None, None)),
    ("....", (None, None, None, None)),
  ],
)
def test_reference_positions(sequence, expected):
  assert reference_positions(sequence) == expected


# start should reflect a fragment beginning at reference res 101
def test_reference_start():
  assert reference_positions("A-CD", start=101) == (101, None, 102, 103)


# choose reference by id rather than first row
def test_build_reference_by_id(protein_msa):
  result = build_reference_map(protein_msa, "ref", start=101)

  assert result.component == "A"
  assert result.reference_id == "ref"
  assert result.n_sites == 7
  assert result.original_columns == (0, 1, 2, 3, 4, 5, 6)
  assert result.original_positions == (1, 2, 3, 4, 5, 6, 7)
  assert result.reference_positions == (
    101, None, 102, 103, None, 104, 105,
  )
  assert result.symbols == tuple("A-CD.E?")


# report 1-based current, original, and reference positions
def test_site_lookup(protein_msa):
  mapping = build_reference_map(protein_msa, "ref", start=101)

  assert mapping.site(2).current_column == 3
  assert mapping.site(2).original_column == 3
  assert mapping.site(2).reference_position == 102
  assert mapping.site(1).reference_position is None
  assert mapping.site(6).reference_symbol == "?"
  assert mapping.site(6).reference_position == 105


# selected cols keep original aln index after masking
def test_subset_after_masking(protein_msa):
  original = build_reference_map(protein_msa, "ref", start=101)
  masked = apply_mask(
    protein_msa,
    [True, False, True, False, False, True, True],
  )
  updated = subset_reference_map(
    original,
    masked.original_columns,
  )

  assert masked.msa.sequences[1] == "ACE?"
  assert updated.original_columns == (0, 2, 5, 6)
  assert updated.reference_positions == (101, 102, 104, 105)
  assert updated.site(2).current_column == 3
  assert updated.site(2).original_column == 6
  assert updated.site(2).reference_position == 104


# two sequential masks still retain original msa col numbering
def test_repeated_masking(protein_msa):
  original = build_reference_map(protein_msa, "ref")

  masked1 = apply_mask(
    protein_msa,
    [True, False, True, True, False, True, True],
  )
  map1 = subset_reference_map(original, masked1.original_columns)

  masked2 = apply_mask(
    masked1.msa,
    [False, True, False, True, True],
  )
  map2 = subset_reference_map(map1, masked2.original_columns)

  assert map2.original_columns == (2, 5, 6)
  assert map2.reference_positions == (2, 4, 5)
  assert map2.site(0).original_column == 3


# gap at ref position maps to None even if another seq has a residue
def test_reference_gap_not_counted(protein_msa):
  mapping = build_reference_map(protein_msa, "ref")

  assert protein_msa.sequences[0][1] == "T"
  assert mapping.site(1).reference_position is None


# dna and rna can use same coordinate mapping logic
@pytest.mark.parametrize(
  "molecule,reference",
  [
    ("dna", "A-CGT"),
    ("rna", "A-CGU"),
  ],
)
def test_nucleotide_coordinate_map(molecule, reference):
  msa = ComponentMSA("C", molecule, ("ref",), (reference,))
  mapping = build_reference_map(msa, "ref")

  assert mapping.reference_positions == (1, None, 2, 3, 4)


# ref can be excluded from downstream synchronized msa
# map must be made from original input first
def test_reference_can_be_excluded_after_matching(protein_msa):
  mapping = build_reference_map(protein_msa, "ref")
  selected = protein_msa.take_ids(["alt"])

  assert selected.ids == ("alt",)
  assert mapping.n_sites == selected.n_sites
  assert mapping.site(5).reference_position == 4


# missing or incorrect reference ids should fail
@pytest.mark.parametrize("reference_id", ["missing", "REF", ""])
def test_unknown_reference_id(protein_msa, reference_id):
  with pytest.raises(ValueError, match="not found"):
    build_reference_map(protein_msa, reference_id)


# input must be a component msa
def test_invalid_component_input():
  with pytest.raises(TypeError, match="ComponentMSA"):
    build_reference_map("ACDE", "ref")


# first reference pos must be a positive integer
@pytest.mark.parametrize("start", [0, -1, 1.5, True, "101"])
def test_invalid_start(start):
  with pytest.raises(ValueError, match="positive integer"):
    reference_positions("AC", start=start)


# empty and malformed reference sequences should fail
@pytest.mark.parametrize(
  "reference",
  ["", "A C", "A1C", None],
)
def test_invalid_reference_sequence(reference):
  with pytest.raises(ValueError, match="reference sequence"):
    reference_positions(reference)


# reject invalid subset indices, duplicates, and reordered cols
@pytest.mark.parametrize(
  "indices,error_type",
  [
    ([], ValueError),
    ([0, 0], ValueError),
    ([2, 1], ValueError),
    ([0, 100], IndexError),
    ([-1], IndexError),
    ([True], TypeError),
    ([0.0], TypeError),
  ],
)
def test_invalid_subset(protein_msa, indices, error_type):
  mapping = build_reference_map(protein_msa, "ref")

  with pytest.raises(error_type):
    subset_reference_map(mapping, indices)


# reject wrong coordinate map object
def test_invalid_map_input():
  with pytest.raises(TypeError, match="ReferenceCoordinates"):
    subset_reference_map("not a map", [0])


# reject out of bounds site lookup and invalid index types
def test_invalid_site_index(protein_msa):
  mapping = build_reference_map(protein_msa, "ref")

  with pytest.raises(IndexError):
    mapping.site(-1)

  with pytest.raises(IndexError):
    mapping.site(7)

  with pytest.raises(TypeError):
    mapping.site(True)


# coordinate arrays must be consistent when constructed directly
def test_reject_inconsistent_manual_mapping():
  with pytest.raises(ValueError, match="equal nonzero lengths"):
    ReferenceCoordinates(
      component="A",
      reference_id="ref",
      original_columns=(0, 1),
      reference_positions=(1,),
      symbols=("A", "-"),
    )


# ref gap must not be assigned a residue number
def test_reject_gap_coordinate_mismatch():
  with pytest.raises(ValueError, match="reference gaps"):
    ReferenceCoordinates(
      component="A",
      reference_id="ref",
      original_columns=(0, 1),
      reference_positions=(1, 2),
      symbols=("A", "-"),
    )
