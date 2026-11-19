# tests for reading fasta and stockholm msas

import numpy as np
import pytest

from triplet_coevolution.core.alphabets import MoleculeType
from triplet_coevolution.io.alignments import read_msa


# read aln fasta w/ lowercase, gaps, and missing positions
def test_read_fasta(tmp_path):
  path = tmp_path / "protein.fasta"
  path.write_text(">a1\nacd-\n>a2\nagD?\n")

  msa = read_msa(path, "A", MoleculeType.PROTEIN)

  assert msa.name == "A"
  assert msa.ids == ("a1", "a2")
  assert msa.sequences == ("ACD-", "AGD?")
  assert msa.matrix.shape == (2, 4)

  np.testing.assert_array_equal(msa.matrix[0], [0, 1, 2, 20])
  np.testing.assert_array_equal(msa.matrix[1], [0, 5, 2, -1])


# read stockholm rna aln
def test_read_stockholm(tmp_path):
  path = tmp_path / "rna.sto"
  path.write_text(
    "# STOCKHOLM 1.0\n"
    "c1 AGCU\n"
    "c2 AG-U\n"
    "//\n"
  )

  msa = read_msa(path, "C", "rna")

  assert msa.ids == ("c1", "c2")
  assert msa.n_sequences == 2


def test_interleaved_stockholm_with_bom(tmp_path):
  path = tmp_path / "interleaved.sto"
  path.write_text(
    "\n# STOCKHOLM 1.0\n#=GF ID example\nx\tAC\ny\tA-\n#=GC RF xx\n"
    "\nx GT\ny GC\n#=GC RF xx\n//\n\n",
    encoding="utf-8-sig",
  )
  assert read_msa(path, "A", "dna").sequences == ("ACGT", "A-GC")


@pytest.mark.parametrize("second_record", ["# STOCKHOLM 1.0\ny GT\n//\n", "# STOCKHOLM 1.0\n//\n"])
def test_stockholm_rejects_multiple_alignments(tmp_path, second_record):
  path = tmp_path / "two.sto"
  path.write_text("# STOCKHOLM 1.0\nx AC\n//\n" + second_record)
  with pytest.raises(ValueError, match="More than one"):
    read_msa(path, "A", "dna")


@pytest.mark.parametrize("extra", ["y GC\n", "//\n"])
def test_stockholm_content_after_terminator(tmp_path, extra):
  path = tmp_path / "malformed.sto"
  path.write_text("# STOCKHOLM 1.0\nx AC\n//\n" + extra)
  with pytest.raises(ValueError, match="unexpected content after //"):
    read_msa(path, "A", "dna")


def test_truncated_interleaved_stockholm(tmp_path):
  path = tmp_path / "truncated.sto"
  path.write_text("# STOCKHOLM 1.0\nx AC\ny GT\n\nx TG\ny GC\n")
  with pytest.raises(ValueError, match="missing // terminator"):
    read_msa(path, "A", "dna")


def test_fasta_with_bom(tmp_path):
  path = tmp_path / "bom.fa"
  path.write_text(">a\nAC\n>b\nGC\n", encoding="utf-8-sig")
  assert read_msa(path, "A", "dna").ids == ("a", "b")

# reject missing aln files
def test_missing_file(tmp_path):
  with pytest.raises(FileNotFoundError, match="not found"):
    read_msa(tmp_path / "absent.fasta", "A", "protein")


# reject empty files
def test_empty_file(tmp_path):
  path = tmp_path / "empty.fasta"
  path.write_text("")

  with pytest.raises(ValueError, match="no sequences"):
    read_msa(path, "A", "protein")


# reject unsupported formats
def test_unsupported_format(tmp_path):
  path = tmp_path / "alignment.a3m"
  path.write_text(">x\nACDE\n")

  with pytest.raises(ValueError, match="unsupported"):
    read_msa(path, "A", "protein")


# reject duplicate ids and unequal seq lengths
@pytest.mark.parametrize(
  "content",
  [
    ">x\nACG\n>x\nAGG\n", # duplicate ids named x
    ">x\nACG\n>y\nAG\n", # unequal seq lengths
  ],
)
def test_bad_fasta_alignmetn(tmp_path, content):
  path = tmp_path / "bad.fasta"
  path.write_text(content)

  with pytest.raises(ValueError):
    read_msa(path, "A", "protein")


# rna alns should reject T
def test_wrong_nucleotide(tmp_path):
  path = tmp_path / "rna.fa"
  path.write_text(">x\nACGT\n>y\nAGGU\n")

  with pytest.raises(ValueError, match="invalid seq"):
    read_msa(path, "C", "rna")
