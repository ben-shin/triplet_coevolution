# read aligned fasta and stockholm files into ComponentMSA objs

from pathlib import Path

from Bio import SeqIO

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.alphabets import MoleculeType

# extensions you can recognize. add more later if necessary
FILE_FORMATS = {
  ".fa" = "fasta",
  ".fas" = "fasta",
  ".fasta" = "fasta",
  ".faa" = "fasta",
  ".fna" = "fasta",
  ".ffn" = "fasta",
  ".sto" = "stockholm",
  ".sth" = "stockholm",
  ".stockholm" = "stockholm",
}


def read_msa(
  path: str | Path,
  name: str,
  molecule: MoleculeType | str,
  fmt: str | None = None,
) -> ComponentMSA:
  """read aligned fasta/stockholm file, returns validated ComponentMSA"""

  path = Path(path).expanduser()

  # make sure the file actually exists
  if not path.exists():
    raise FileNotFoundError(
      f"alignment file not found: {path}"
    )

  # determine file format from extension if not specified
  if fmt is None:
    fmt = FILE_FORMATS.get(path.suffix.lower())

  if fmt is None or fmt.lower() not in {"fasta", "stockholm"}:
    raise ValueError(
      f"unsupported alignment format for {path};"
      "use .fasta or .sto or specify fmt='fasta'/'stockholm"
    )

  fmt = fmt.lower()

  # read all seqs w/ biopython
  try:
    with path.open("r", encoding="utf-8") as handle:
      records = list(SeqIO.parse(handle, fmt))

  except ValueError as exc:
    raise ValueError(
      f"could not parse {fmt} alignment {path}: {exc}"
    ) from exc

  # reject empty aln files
  if not records:
    raise ValueError(
      f"no seqs found in aln: {path}"
    )

  # construct validated aln obj
  # duplicate ids, unequal lengths, and invalid symbols are checked by ComponentMSA
  return ComponentMSA(
    name=name,
    molecule=molecule,
    ids=tuple(record.id for record in records),
    sequences=tuple(str(record.seq) for record in records),
  )
