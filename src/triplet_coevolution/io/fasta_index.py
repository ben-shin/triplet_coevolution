"""Temporary metadata and byte-offset index for aligned FASTA and A3M."""

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from triplet_coevolution.core.alphabets import AMBIGUOUS, CANONICAL
from triplet_coevolution.evidence.identifiers import identifiers
from triplet_coevolution.io.stockholm import StockholmRow, _domain, _source

_WHITESPACE = b" \t\n\r"
_INSERTS = b"abcdefghijklmnopqrstuvwxyz."
# These HH-suite display tracks contain symbols with different semantics to sequence rows.
_DISPLAY = {"ss_dssp", "sa_dssp", "ss_pred", "ss_conf", "ss_cons"}
_DISPLAY_SYMBOLS = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-?"


def allowed_symbols(molecule):
    upper = CANONICAL[molecule] + AMBIGUOUS[molecule] + "-.?"
    return (upper + upper.lower()).encode("ascii")


class FastaIndex:
    def __init__(self, db, path, molecule, a3m):
        self.db = db
        self.path = path
        self.a3m = a3m
        self.n_columns = None
        self.n_rows = 0
        self.insert_residues = 0
        self.padding_symbols = 0
        self.display_widths = {}
        self.db.executescript("""
            CREATE TABLE rows (row_id TEXT PRIMARY KEY, start INTEGER, end INTEGER,
                               head INTEGER, stop INTEGER, raw_width INTEGER) WITHOUT ROWID;
            CREATE UNIQUE INDEX source_order ON rows(head);
            CREATE TABLE alias (name TEXT, row_id TEXT, PRIMARY KEY(name,row_id)) WITHOUT ROWID;
        """)
        self._scan(molecule)

    def _scan(self, molecule):
        allowed = allowed_symbols(molecule)
        row_id = None
        head = width = occupied = raw_width = 0
        with _source(self.path) as source:
            offset = 0
            for number, raw in enumerate(source, 1):
                line_offset = offset
                offset += len(raw)
                line = raw.removeprefix(b"\xef\xbb\xbf") if number == 1 else raw
                if line.startswith(b">"):
                    if row_id is not None:
                        self._add(row_id, head, line_offset, width, raw_width, occupied, molecule)
                    title = line[1:].decode("utf-8").strip()
                    if not title:
                        raise ValueError(f"empty FASTA identifier at line {number}")
                    row_id = title.split(maxsplit=1)[0]
                    if (
                        self.a3m
                        and row_id not in _DISPLAY
                        and (
                            row_id.startswith(("ss_", "sa_", "aa_"))
                            or row_id == "Consensus"
                            or row_id.endswith("_consensus")
                        )
                    ):
                        raise ValueError(
                            f"unsupported A3M display annotation {row_id!r}; "
                            "supply sequence rows or supported SS/SA tracks"
                        )
                    head, width, raw_width, occupied = line_offset, 0, 0, 0
                    continue
                fragment = line.translate(None, _WHITESPACE)
                if not fragment:
                    continue
                if row_id is None:
                    raise ValueError(f"expected FASTA header before sequence at line {number}")
                display = self.a3m and row_id in _DISPLAY
                invalid = fragment.translate(None, _DISPLAY_SYMBOLS if display else allowed)
                if invalid:
                    raise ValueError(
                        f"invalid seq {row_id!r} for {molecule.value} at line "
                        f"{number}: {invalid[:1]!r}"
                    )
                dots = fragment.count(b".")
                raw_width += len(fragment)
                occupied += len(fragment) - fragment.count(b"-") - dots
                if self.a3m:
                    match_width = len(fragment.translate(None, _INSERTS))
                    if not display:
                        self.insert_residues += len(fragment) - match_width - dots
                        self.padding_symbols += dots
                    width += match_width
                else:
                    width += len(fragment)
            if row_id is not None:
                self._add(row_id, head, offset, width, raw_width, occupied, molecule)
        if not self.n_rows:
            raise ValueError(f"no sequences found in alignment: {self.path}")
        for track, length in self.display_widths.items():
            if length != self.n_columns:
                raise ValueError(
                    f"A3M display track {track!r} has {length} match columns, "
                    f"expected {self.n_columns}"
                )

    def _add(self, row_id, head, stop, width, raw_width, occupied, molecule):
        if self.a3m and row_id in _DISPLAY:
            if row_id in self.display_widths:
                raise ValueError(f"duplicate A3M display track {row_id!r}")
            self.display_widths[row_id] = width
            return
        if not width or (self.n_columns is not None and self.n_columns != width):
            raise ValueError(
                f"sequences must have equal, nonzero alignment widths: "
                f"{row_id!r} has {width}, expected {self.n_columns}"
            )
        self.n_columns = width
        _, start, end = _domain(row_id, occupied, molecule.value != "protein")
        try:
            self.db.execute(
                "INSERT INTO rows VALUES (?,?,?,?,?,?)", (row_id, start, end, head, stop, raw_width)
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate sequence IDs: {row_id!r}") from exc
        aliases, *_ = identifiers(StockholmRow(row_id, None, start, end))
        self.db.executemany("INSERT INTO alias VALUES (?,?)", ((name, row_id) for name in aliases))
        self.n_rows += 1

    def ids(self, *, sorted_names=False):
        order = "row_id" if sorted_names else "head"
        for (row_id,) in self.db.execute(f"SELECT row_id FROM rows ORDER BY {order}"):
            yield row_id

    def rows(self):
        for row_id, start, end in self.db.execute(
            "SELECT row_id,start,end FROM rows ORDER BY head"
        ):
            yield StockholmRow(row_id, None, start, end)

    def find(self, identifier):
        for row_id, start, end in self.db.execute(
            """
            SELECT r.row_id,r.start,r.end FROM alias a JOIN rows r ON a.row_id=r.row_id
            WHERE name=? ORDER BY head
        """,
            (identifier,),
        ):
            yield StockholmRow(row_id, None, start, end)

    def contains(self, row_id):
        return (
            self.db.execute("SELECT 1 FROM rows WHERE row_id=?", (row_id,)).fetchone() is not None
        )

    def raw_width(self, row_id):
        row = self.db.execute("SELECT raw_width FROM rows WHERE row_id=?", (row_id,)).fetchone()
        if row is None:
            raise ValueError(f"alignment is missing ids: {row_id!r}")
        return row[0]

    def records(self, ordered):
        offsets = []
        for row_id in ordered:
            bounds = self.db.execute(
                "SELECT head,stop FROM rows WHERE row_id=?", (row_id,)
            ).fetchone()
            if bounds is None:
                raise ValueError(f"alignment is missing ids: {row_id!r}")
            offsets.append((*bounds, row_id))
        # Uncompressed offsets work for both streams. Monotonic reads avoid gzip restarts.
        selected = {}
        with _source(self.path) as source:
            for head, stop, row_id in sorted(offsets):
                source.seek(head)
                title = (
                    source.readline().removeprefix(b"\xef\xbb\xbf").decode("utf-8").rstrip("\r\n")
                )
                seq = bytearray()
                while source.tell() < stop:
                    line = source.readline()
                    if not line:
                        raise ValueError("source alignment was truncated during extraction")
                    seq.extend(line.translate(None, _WHITESPACE))
                seq = seq.decode("ascii")
                selected[row_id] = (title, seq)
        return selected


@contextmanager
def open_fasta(path, molecule, a3m, scratch):
    with closing(sqlite3.connect(Path(scratch) / "fasta.sqlite")) as db:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA cache_size=-2048")
        db.execute("PRAGMA temp_store=FILE")
        yield FastaIndex(db, path, molecule, a3m)
