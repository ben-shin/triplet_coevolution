"""Read local, ungapped FASTA records with exact identifiers."""

import gzip
from pathlib import Path

from Bio import SeqIO

from triplet_coevolution.core.alphabets import CANONICAL, encode_sequence


def read_sequences(path: Path, molecule):
    seen = set()
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_id = record.id
            if not seq_id or seq_id in seen:
                raise ValueError(f"{path}: duplicate or empty FASTA ID {seq_id!r}")
            seen.add(seq_id)
            sequence = str(record.seq).upper()
            if "-" in sequence or "." in sequence:
                raise ValueError(f"{path}: expected ungapped sequence for {seq_id}")
            try:
                encode_sequence(sequence, molecule)
            except ValueError as exc:
                raise ValueError(f"{path}, {seq_id}: {exc}") from exc
            yield seq_id, sequence
    if not seen:
        raise ValueError(f"no FASTA records in {path}")


def read_reference(comp):
    records = list(read_sequences(comp["reference"], comp["molecule"]))
    if len(records) != 1 or records[0][0] != comp["reference_id"]:
        raise ValueError("reference FASTA must contain exactly the configured reference_id")
    sequence = records[0][1]
    if not any(letter in CANONICAL[comp["molecule"]] for letter in sequence):
        raise ValueError("reference sequence contains no canonical symbols")
    return sequence


def check_profile(comp):
    """Require exactly one text HMM or CM of the declared alphabet."""
    if comp["backend"] not in ("hmmer", "infernal"):
        return
    marker = "HMMER3/" if comp["backend"] == "hmmer" else "INFERNAL1/"
    expected = {"protein": "amino", "dna": "dna", "rna": "rna"}[comp["molecule"]]
    count = 0
    alphabet = None
    opener = gzip.open if comp["profile"].suffix.lower() == ".gz" else open
    with opener(comp["profile"], "rt", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(marker):
                count += 1
            elif count == 1 and alphabet is None and line.split(maxsplit=1)[:1] == ["ALPH"]:
                tokens = line.split()
                if len(tokens) != 2:
                    raise ValueError(f"{comp['profile']}: ALPH header requires one alphabet value")
                alphabet = tokens[1].lower()
    if count != 1 or alphabet != expected:
        raise ValueError(
            f"{comp['profile']}: expected one text {marker} profile with alphabet {expected}"
        )
