# read aligned fasta and stockholm files into ComponentMSA objs

from io import TextIOWrapper
from pathlib import Path

from Bio import AlignIO, SeqIO

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.alphabets import MoleculeType

# extensions you can recognize. add more later if necessary
FILE_FORMATS = {
  ".fa": "fasta",
  ".fas": "fasta",
  ".fasta": "fasta",
  ".faa": "fasta",
  ".fna": "fasta",
  ".ffn": "fasta",
  ".sto": "stockholm",
  ".sth": "stockholm",
  ".stockholm": "stockholm",
}


class _StockholmStream(TextIOWrapper):
  """Normalize legal tab separators for Biopython's space-delimited parser."""

  def readline(self, size: int = -1) -> str:
    return super().readline(size).replace("\t", " ")


def _validate_stockholm_record(handle: _StockholmStream) -> None:
  """Check the record boundaries and leave the stream at its header."""
  while True:
    header_offset = handle.tell()
    line = handle.readline()
    if line.strip() or not line:
      break
  if line.strip() != "# STOCKHOLM 1.0":
    raise ValueError("missing # STOCKHOLM 1.0 header")

  terminated = False
  for line in handle:
    marker = line.strip()
    if marker == "# STOCKHOLM 1.0":
      raise ValueError("More than one Stockholm record header")
    if terminated and marker:
      raise ValueError("malformed stockholm alignment: unexpected content after // terminator")
    if marker == "//":
      terminated = True
  if not terminated:
    raise ValueError("incomplete Stockholm alignment: missing // terminator")
  handle.seek(header_offset)


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
      f"unsupported alignment format for {path}. "
      "Use .fasta or .sto, or specify fmt='fasta' or 'stockholm'."
    )

  fmt = fmt.lower()

  # read all seqs w/ biopython
  try:
    if fmt == "stockholm":
      # Require one alignment, including any interleaved blocks within it.
      with _StockholmStream(path.open("rb"), encoding="utf-8-sig") as handle:
        _validate_stockholm_record(handle)
        records = AlignIO.read(handle, fmt)
    else:
      with path.open("r", encoding="utf-8-sig") as handle:
        records = list(SeqIO.parse(handle, fmt))
  except AssertionError as exc:
    # Biopython uses assertions for some malformed Stockholm records.
    raise ValueError(f"malformed {fmt} alignment {path}: invalid record structure") from exc
  except ValueError as exc:
    raise ValueError(
      f"could not parse {fmt} alignment {path}: {exc}"
    ) from exc

  # reject empty aln files
  if not records:
    raise ValueError(
      f"no sequences found in aln: {path}"
    )

  # construct validated aln obj
  # duplicate ids, unequal lengths, and invalid symbols are checked by ComponentMSA
  return ComponentMSA(
    name=name,
    molecule=molecule,
    ids=tuple(record.id for record in records),
    sequences=tuple(str(record.seq) for record in records),
  )
