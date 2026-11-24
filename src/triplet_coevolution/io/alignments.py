"""Validated, disk-indexed selection of a matched alignment and its reference rows."""

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.alphabets import MoleculeType
from triplet_coevolution.core.coordinates import (
    ReferenceCoordinates,
    build_reference_map,
    reference_positions,
)
from triplet_coevolution.io.fasta_index import allowed_symbols, open_fasta
from triplet_coevolution.io.stockholm import (
    _destination,
    _events,
    _header,
    _signature,
    _source,
    open_stockholm,
)

FILE_FORMATS = {
    **dict.fromkeys((".fa", ".fas", ".fasta", ".faa", ".fna", ".ffn"), "fasta"),
    **dict.fromkeys((".sto", ".sth", ".stockholm"), "stockholm"),
    ".a3m": "a3m",
}
_REMOVE_INSERTIONS = str.maketrans("", "", "abcdefghijklmnopqrstuvwxyz.")


@dataclass(frozen=True)
class SelectedAlignment:
    msa: ComponentMSA
    provenance: dict
    # Only A3M needs row-specific biological residues outside scoring columns.
    _raw_a3m: dict[str, str]

    def reference_map(self, reference_id, start=1, *, row_id=None, strand="+"):
        mapping = build_reference_map(self.msa, reference_id, start, row_id=row_id, strand=strand)
        if not self._raw_a3m:
            return mapping
        raw = self._raw_a3m[reference_id if row_id is None else row_id]
        physical = reference_positions(raw, start, strand=strand)
        match_positions = tuple(
            pos
            for symbol, pos in zip(raw, physical, strict=True)
            if symbol != "." and not symbol.islower()
        )
        return ReferenceCoordinates(
            self.msa.name,
            reference_id,
            mapping.original_columns,
            match_positions,
            mapping.symbols,
            start,
            strand,
        )


class AlignmentInventory:
    """Identifiers and source dimensions; the backing store lives until context exit."""

    def __init__(self, index, path, signature, name, molecule, fmt, scratch):
        self._index = index
        self._path = path
        self._signature = signature
        self._scratch = scratch
        self.name = name
        self.molecule = molecule
        self.format = fmt
        self.n_rows = index.n_rows
        self.n_columns = index.n_columns
        self.raw_symbols = self.n_rows * self.n_columns
        if fmt == "a3m":
            self.raw_symbols += index.insert_residues + index.padding_symbols

    def ids(self, *, sorted_names=False):
        return self._index.ids(sorted_names=sorted_names)

    def contains(self, row_id):
        return self._index.contains(row_id)

    def selected_raw_symbols(self, row_ids):
        if self.format == "a3m":
            return sum(self._index.raw_width(row_id) for row_id in row_ids)
        return sum(1 for _ in row_ids) * self.n_columns

    def rows(self):
        return self._index.rows()

    def find(self, identifier):
        """Yield every metadata alias hit, including separate domains. Selection is exact-ID."""
        return self._index.find(identifier)

    def _check_source(self):
        if _signature(self._path) != self._signature:
            raise ValueError(f"alignment input changed while reading {self._path}")

    def select(self, ordered_unique_ids, *, output=None):
        """Load only requested rows in caller order. An optional derivative keeps source format."""
        ids = tuple(ordered_unique_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("selection requires nonempty, unique exact row IDs")
        self._check_source()
        output = Path(output) if output is not None else None
        if output is not None and output.resolve() == self._path.resolve():
            raise ValueError("input and selected output must be distinct paths")
        compressed = output is not None and output.suffix.lower() == ".gz"
        raw_a3m = {}
        with tempfile.TemporaryDirectory(
            dir=output.parent if output else self._scratch
        ) as directory:
            staged = Path(directory) / ("selected.gz" if compressed else "selected")
            if self.format == "stockholm":
                self._index.extract(ids, staged)
                # extract stitches interleaved sequence and annotation tracks on disk.
                # Its derivative has one complete line per row in the requested order.
                with _source(staged) as handle:
                    _header(handle)
                    seqs = tuple(
                        value.replace(".", "-")
                        for _, kind, _, _, value in _events(handle)
                        if kind == "sequence"
                    )
            else:
                records = self._index.records(ids)
                if output is not None:
                    with _destination(staged, compressed) as handle:
                        for row_id in ids:
                            title, seq = records[row_id]
                            handle.write(f"{title}\n")
                            for i in range(0, len(seq), 80):
                                handle.write(seq[i : i + 80] + "\n")
                if self.format == "a3m":
                    raw_a3m = {row_id: records[row_id][1] for row_id in ids}
                    seqs = tuple(seq.translate(_REMOVE_INSERTIONS) for seq in raw_a3m.values())
                else:
                    seqs = tuple(records[row_id][1] for row_id in ids)
            msa = ComponentMSA(self.name, self.molecule, ids, seqs)
            self._check_source()
            if output is not None:
                os.replace(staged, output)
        provenance = {
            "format": self.format,
            "selection": "disk_index_exact_id",
            "source_rows": self.n_rows,
            "source_columns": self.n_columns,
            "selected_rows": len(ids),
            "selected_columns": self.n_columns,
            "coordinate_space": "a3m_match_columns" if raw_a3m else "source_alignment_columns",
        }
        if raw_a3m:
            provenance.update(
                a3m_source_insert_residues=self._index.insert_residues,
                a3m_source_insert_padding=self._index.padding_symbols,
                a3m_omitted_query_display_tracks=sorted(self._index.display_widths),
                a3m_selected_insert_residues=sum(
                    sum(symbol.islower() for symbol in seq) for seq in raw_a3m.values()
                ),
            )
        return SelectedAlignment(msa, provenance, raw_a3m)


@contextmanager
def open_msa(path, name, molecule, fmt=None, *, temp_dir=None):
    """Validate a whole plain/gzipped source before exposing its temporary row inventory."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"alignment file not found: {path}")
    if name not in ("A", "B", "C"):
        raise ValueError("component name must be A, B or C")
    molecule = MoleculeType(molecule)
    suffix_path = path.with_suffix("") if path.suffix.lower() == ".gz" else path
    fmt = FILE_FORMATS.get(suffix_path.suffix.lower()) if fmt is None else fmt.lower()
    if fmt not in ("fasta", "stockholm", "a3m"):
        raise ValueError(
            f"unsupported alignment format for {path}; specify fasta, stockholm or a3m"
        )
    signature = _signature(path)
    with tempfile.TemporaryDirectory(dir=temp_dir) as scratch:
        if fmt == "stockholm":
            manager = open_stockholm(path, temp_dir=scratch, molecule=molecule.value)
        else:
            manager = open_fasta(path, molecule, fmt == "a3m", scratch)
        with manager as index:
            if fmt == "stockholm":
                invalid = index.symbols.difference(map(chr, allowed_symbols(molecule)))
                if invalid:
                    raise ValueError(
                        f"invalid sequence symbol(s) for {molecule.value}: {sorted(invalid)}"
                    )
            inventory = AlignmentInventory(index, path, signature, name, molecule, fmt, scratch)
            inventory._check_source()
            yield inventory


def read_msa(path, name, molecule, fmt=None) -> ComponentMSA:
    """Read every sequence. For a matched subset or A3M reference map use open_msa instead."""
    with open_msa(path, name, molecule, fmt) as inventory:
        return inventory.select(inventory.ids()).msa
