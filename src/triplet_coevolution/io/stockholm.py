"""Stream an annotated Stockholm family, indexing identifiers on temporary disk."""

import gzip
import os
import re
import sqlite3
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from io import TextIOWrapper
from itertools import chain
from pathlib import Path

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.evidence.identifiers import identifiers

_HEADER = b"# STOCKHOLM 1.0"
_DOMAIN = re.compile(r"(.+)/([0-9]+)-([0-9]+)$")
_PROTEIN = CANONICAL["protein"].encode("ascii")
_TRANSCRIBE = str.maketrans("Tt", "Uu")


@dataclass(frozen=True)
class StockholmRow:
    """The exact alignment name, optional GS AC, and 1-based inclusive domain interval."""

    row_id: str
    accession: str | None
    start: int | None
    end: int | None
    annotations: tuple[tuple[str, str], ...] = ()


@contextmanager
def _source(path):
    with path.open("rb") as raw:
        magic = raw.read(2)
        raw.seek(0)
        if magic == b"\x1f\x8b":
            try:
                with gzip.GzipFile(fileobj=raw) as stream:
                    yield stream
            except (EOFError, zlib.error) as exc:
                raise ValueError(f"truncated or corrupt gzip source {path}") from exc
        else:
            yield raw


def _header(stream):
    for line in stream:
        marker = line.removeprefix(b"\xef\xbb\xbf").strip()
        if marker == _HEADER:
            return stream.tell()
        if marker:
            break
    raise ValueError("missing # STOCKHOLM 1.0 header")


def _family_offset(stream, family):
    """Locate one record even when identifying GF lines follow its sequences."""
    offset = 0
    record = None
    selected = None
    matched = False
    for raw in stream:
        offset += len(raw)
        line = raw.removeprefix(b"\xef\xbb\xbf").strip()
        if line == _HEADER:
            if record is not None:
                raise ValueError("missing // terminator before next Stockholm header")
            record, matched = offset, False
        elif record is None:
            if line:
                raise ValueError("unexpected content outside Stockholm record")
        elif line == b"//":
            if matched:
                if selected is not None:
                    raise ValueError(f"multiple Stockholm records match family {family!r}")
                selected = record
            record = None
        elif line.startswith(b"#=GF"):
            fields = line.split(maxsplit=2)
            if len(fields) == 3 and fields[0] == b"#=GF" and fields[1] in (b"AC", b"ID"):
                value = fields[2].decode("utf-8")
                base = re.sub(r"\.\d+$", "", value) if fields[1] == b"AC" else value
                matched |= family in (value, base)
    if record is not None:
        raise ValueError("missing // terminator in Stockholm file")
    if selected is None:
        raise ValueError(f"Stockholm family {family!r} was not found")
    stream.seek(selected)
    return selected


def _events(stream):
    """Yield one bounded fragment or annotation at a time up to the terminator."""
    for number, raw in enumerate(stream, start=1):
        line = raw.decode("utf-8").strip()
        if not line:
            yield number, "break", "", "", ""
        elif line == "//":
            return
        elif line == "# STOCKHOLM 1.0":
            raise ValueError("missing // terminator before next Stockholm header")
        elif line.startswith("#="):
            kind, *fields = line.split(maxsplit=1)
            if kind not in ("#=GF", "#=GS", "#=GC", "#=GR"):
                continue
            per_row = kind in ("#=GS", "#=GR")
            parts = fields[0].split(maxsplit=2 if per_row else 1) if fields else []
            if len(parts) < 1 + per_row:
                raise ValueError(f"malformed {kind} at record line {number}")
            row = parts.pop(0) if per_row else ""
            tag, value = parts[0], parts[1] if len(parts) == 2 else ""
            if kind in ("#=GC", "#=GR") and (not value or len(value.split()) != 1):
                raise ValueError(f"invalid aligned {kind} feature {tag!r} at record line {number}")
            yield number, kind, row, tag, value
        elif not line.startswith("#"):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"expected a Stockholm row name and sequence at record line {number}"
                )
            yield number, "sequence", fields[0], "", fields[1]
    raise ValueError("incomplete Stockholm alignment: missing // terminator")


def _domain(row, occupied, allow_reverse):
    match = _DOMAIN.fullmatch(row)
    if not match:
        suffix = row.rpartition("/")[2]
        if "/" in row and re.match(r"[+-]?[0-9]+-", suffix):
            raise ValueError(f"malformed domain interval in {row!r}")
        return row, None, None
    base, first, last = match.groups()
    start, end = int(first), int(last)
    if min(start, end) < 1:
        raise ValueError(f"{row!r} requires positive domain coordinates")
    if end < start and not allow_reverse:
        raise ValueError(f"{row!r} requires a forward domain interval for a protein")
    if abs(end - start) + 1 != occupied:
        raise ValueError(f"{row!r} domain span disagrees with its {occupied} nongap positions")
    return base, start, end


def _check_feature_blocks(db):
    # A single complete annotation is absolute across the whole alignment. Wrapped
    # tracks instead need a matching sequence block for every fragment.
    bad = db.execute("""
        SELECT p.row_id, p.tag, p.line, p.columns, b.columns
        FROM pending_feature p
        JOIN track t ON t.row_id = p.row_id AND t.tag = p.tag
        LEFT JOIN sequence_block b ON b.section = p.section
        WHERE t.fragments > 1 AND (b.section IS NULL OR b.columns != p.columns)
        ORDER BY p.rowid LIMIT 1
    """).fetchone()
    if bad:
        row, tag, line, actual, expected = bad
        label = f"#=GR {row} {tag}" if row else f"#=GC {tag}"
        where = f"{label} at record line {line} has fragment width {actual}"
        if expected is None:
            raise ValueError(f"{where} and no sequence block in its section")
        raise ValueError(f"{where}, but sequence block width {expected} is required")


def _index(db, stream, allow_reverse):
    db.executescript("""
        CREATE TABLE track (row_id TEXT, tag TEXT, first_line INTEGER, section INTEGER,
            columns INTEGER, occupied INTEGER, fragments INTEGER, amino INTEGER, acg INTEGER,
            thymine INTEGER, uracil INTEGER,
            PRIMARY KEY (row_id, tag)) WITHOUT ROWID;
        CREATE TABLE gs (row_id TEXT PRIMARY KEY, accession TEXT) WITHOUT ROWID;
        CREATE TABLE gs_values (row_id TEXT, tag TEXT, value TEXT);
        CREATE INDEX gs_values_key ON gs_values(row_id);
        CREATE TABLE rows (ordinal INTEGER PRIMARY KEY, row_id TEXT UNIQUE, base TEXT,
            accession TEXT, start INTEGER, end INTEGER, amino INTEGER, acg INTEGER,
            thymine INTEGER, uracil INTEGER);
        CREATE INDEX on_base ON rows(base);
        CREATE TABLE alias (name TEXT, row_id TEXT, PRIMARY KEY (name, row_id)) WITHOUT ROWID;
        CREATE TABLE sequence_block (section INTEGER PRIMARY KEY, columns INTEGER);
        CREATE TABLE pending_feature (row_id TEXT, tag TEXT, line INTEGER, section INTEGER,
            columns INTEGER);
    """)
    gf = {}
    types, symbols = set(), set()
    section = sequence_sections = count_in_section = 0
    first_row = section_width = None
    for line, kind, row, tag, value in _events(stream):
        if kind == "break":
            section += 1
            first_row, count_in_section = None, 0
        elif kind == "#=GF":
            if tag == "TP" and value.strip().casefold() in ("protein", "dna", "rna"):
                types.add(value.strip().casefold())
            if tag in ("SQ", "ID", "AC"):
                if not value or len(value.split()) != 1 or tag in gf:
                    raise ValueError(f"invalid or repeated #=GF {tag} at record line {line}")
                gf[tag] = value
        elif kind == "#=GS":
            ac = value if tag == "AC" else None
            if tag == "AC" and (not ac or len(ac.split()) != 1):
                raise ValueError(f"invalid #=GS AC for {row!r}")
            updated = db.execute(
                """
                INSERT INTO gs VALUES (?, ?) ON CONFLICT(row_id) DO UPDATE
                SET accession = coalesce(excluded.accession, gs.accession)
                WHERE gs.accession IS NULL OR excluded.accession IS NULL
                   OR gs.accession = excluded.accession
            """,
                (row, ac),
            )
            if not updated.rowcount:
                raise ValueError(f"conflicting #=GS AC values for {row!r}")
            if tag != "AC":
                db.execute("INSERT INTO gs_values VALUES (?, ?, ?)", (row, tag, value))
        else:
            occupied = amino = acg = thymine = uracil = 0
            if kind == "sequence":
                # Some writers restart at the first row without a blank separator.
                if row == first_row and count_in_section >= 2:
                    section += 1
                    first_row = None
                if first_row is None:
                    first_row, section_width, count_in_section = row, len(value), 0
                    sequence_sections += 1
                    db.execute("INSERT INTO sequence_block VALUES (?, ?)", (section, section_width))
                if len(value) != section_width:
                    raise ValueError(
                        f"unequal sequence fragment widths at {row!r}, record line {line}"
                    )
                count_in_section += 1
                occupied = len(value) - value.count("-") - value.count(".")
                upper = value.encode("ascii", errors="replace").upper()
                symbols.update(upper)
                if not value.isascii():
                    symbols.add(255)
                amino = len(upper) - len(upper.translate(None, _PROTEIN))
                acg = len(upper) - len(upper.translate(None, b"ACG"))
                thymine, uracil = upper.count(b"T"), upper.count(b"U")
            elif first_row is None or len(value) != section_width:
                # The associated block might follow this annotation, or the source
                # might supply this entire feature on one line. Decide at record end.
                db.execute(
                    "INSERT INTO pending_feature VALUES (?, ?, ?, ?, ?)",
                    (row, tag, line, section, len(value)),
                )
            updated = db.execute(
                """
                INSERT INTO track VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(row_id, tag) DO UPDATE SET
                    columns = track.columns + excluded.columns,
                    occupied = track.occupied + excluded.occupied,
                    amino = track.amino + excluded.amino, acg = track.acg + excluded.acg,
                    thymine = track.thymine + excluded.thymine, uracil = track.uracil + excluded.uracil,
                    section = excluded.section, fragments = fragments + 1
                WHERE track.section != excluded.section
            """,
                (row, tag, line, section, len(value), occupied, amino, acg, thymine, uracil),
            )
            if not updated.rowcount:
                raise ValueError(f"duplicate {kind} {row!r} {tag!r} in one Stockholm block")

    first = db.execute("SELECT columns FROM track WHERE tag='' LIMIT 1").fetchone()
    if first is None:
        raise ValueError("Stockholm alignment contains no sequences")
    width = first[0]
    bad = db.execute(
        "SELECT row_id, tag FROM track WHERE columns != ? LIMIT 1", (width,)
    ).fetchone()
    if bad:
        raise ValueError(f"Stockholm row or annotation {bad} has unequal alignment width")
    min_blocks, max_blocks = db.execute(
        "SELECT min(fragments), max(fragments) FROM track WHERE tag=''"
    ).fetchone()
    if max_blocks > 1 and min_blocks != sequence_sections:
        raise ValueError("incomplete interleaved Stockholm block or duplicate row IDs")
    _check_feature_blocks(db)
    for row, line, occupied, *coverage in db.execute(
        "SELECT row_id, first_line, occupied, amino, acg, thymine, uracil FROM track WHERE tag=''"
    ):
        base, start, end = _domain(row, occupied, allow_reverse)
        db.execute(
            "INSERT INTO rows VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)",
            (line, row, base, start, end, *coverage),
        )
    orphan = db.execute("""
        SELECT row_id FROM gs WHERE NOT EXISTS (
            SELECT 1 FROM rows r WHERE r.row_id = gs.row_id OR r.base = gs.row_id
        ) LIMIT 1
    """).fetchone()
    if orphan:
        raise ValueError(f"Stockholm GS annotation refers to absent row {orphan[0]!r}")
    for row, tag in db.execute("""
        SELECT row_id, tag FROM track t WHERE tag != '' AND row_id != ''
        AND NOT EXISTS (SELECT 1 FROM rows r WHERE r.row_id=t.row_id)
    """):
        members = db.execute("SELECT row_id FROM rows WHERE base=? LIMIT 2", (row,)).fetchall()
        if len(members) != 1:
            raise ValueError(f"Stockholm GR {tag} refers to absent or ambiguous row {row!r}")
        if db.execute(
            "SELECT 1 FROM track WHERE row_id=? AND tag=?", (members[0][0], tag)
        ).fetchone():
            raise ValueError(f"Stockholm GR {tag} is duplicated for {members[0][0]!r} and {row!r}")
    for row, start, end, exact_ac, base_ac in db.execute("""
        SELECT r.row_id, r.start, r.end, e.accession, b.accession FROM rows r
        LEFT JOIN gs e ON e.row_id=r.row_id LEFT JOIN gs b ON b.row_id=r.base
        ORDER BY r.ordinal
    """):
        if exact_ac and base_ac and exact_ac != base_ac:
            raise ValueError(f"conflicting #=GS AC values for {row!r} and its base identifier")
        ac = exact_ac or base_ac
        if ac:
            db.execute("UPDATE rows SET accession=? WHERE row_id=?", (ac, row))
        aliases, _, _, _ = identifiers(StockholmRow(row, ac, start, end))
        db.executemany("INSERT INTO alias VALUES (?, ?)", ((key, row) for key in aliases))
    count = db.execute("SELECT count(*) FROM rows").fetchone()[0]
    if "SQ" in gf and (not gf["SQ"].isdecimal() or int(gf["SQ"]) != count):
        raise ValueError(f"#=GF SQ {gf['SQ']!r} disagrees with {count} Stockholm rows")
    db.executescript(
        "DROP TABLE track; DROP TABLE gs; DROP TABLE sequence_block; DROP TABLE pending_feature;"
    )
    return (
        count,
        width,
        gf,
        tuple(("TP", t) for t in sorted(types)),
        frozenset(chr(s) if s < 128 else "<non-ASCII>" for s in symbols),
    )


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@contextmanager
def _destination(path, compressed):
    if compressed:
        with (
            path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive,
            TextIOWrapper(archive, encoding="utf-8", newline="\n") as out,
        ):
            yield out
    else:
        with path.open("w", encoding="utf-8", newline="\n") as out:
            yield out


class StockholmInventory:
    """An inventory valid for the lifetime of the open_stockholm context."""

    def __init__(self, db, path, offset, signature, count, width, gf, annotations, symbols):
        self._db = db
        self._path = path
        self._offset = offset
        self._signature = signature
        self.n_rows = count
        self.n_columns = width
        self.family_id = gf.get("ID")
        self.accession = gf.get("AC")
        self.annotations = annotations
        self.symbols = symbols
        self.has_reverse, self.max_nucleotide_coverage = db.execute(
            "SELECT coalesce(max(start>end),0),max(acg+thymine+uracil) FROM rows"
        ).fetchone()

    def _row(self, fields):
        row_id, ac, start, end, base = fields
        annotations = tuple(
            self._db.execute(
                "SELECT tag,value FROM gs_values WHERE row_id IN (?,?) ORDER BY rowid",
                (row_id, base),
            )
        )
        return StockholmRow(row_id, ac, start, end, annotations)

    def canonical_coverage(self, row_id, molecule, *, transcribe_t=False):
        """Count known canonical residues in an exact row, including wrapped fragments."""
        fields = self._db.execute(
            "SELECT amino,acg,thymine,uracil FROM rows WHERE row_id=?", (row_id,)
        ).fetchone()
        if fields is None:
            raise ValueError(f"unknown Stockholm row {row_id!r}")
        amino, acg, thymine, uracil = fields
        if molecule == "protein":
            return amino
        if molecule == "rna":
            return acg + uracil + (thymine if transcribe_t else 0)
        if molecule == "dna":
            return acg + thymine
        raise ValueError(f"unknown molecule {molecule!r}")

    def identity_metadata(self):
        """Stream identifiers and explicitly typed annotations for molecule detection."""
        for base, ac in self._db.execute("SELECT base,accession FROM rows"):
            yield "identifier", base
            if ac:
                yield "identifier", ac
        yield from self._db.execute(
            "SELECT tag,value FROM gs_values WHERE upper(tag) IN ('DR','MOLECULE','TP')"
        )

    def rows(self) -> Iterator[StockholmRow]:
        """Iterate over row metadata in source order, without loading aligned sequences."""
        for fields in self._db.execute(
            "SELECT row_id, accession, start, end, base FROM rows ORDER BY ordinal"
        ):
            yield self._row(fields)

    def row(self, row_id: str) -> StockholmRow | None:
        """Return one exact, case-sensitive row ID, or None."""
        fields = self._db.execute(
            "SELECT row_id, accession, start, end, base FROM rows WHERE row_id = ?", (row_id,)
        ).fetchone()
        return self._row(fields) if fields else None

    def find(self, identifier: str) -> Iterator[StockholmRow]:
        """Yield every exact or accession/name alias match, retaining separate domain copies."""
        for fields in self._db.execute(
            """
            SELECT r.row_id, r.accession, r.start, r.end, r.base FROM alias a
            JOIN rows r ON r.row_id = a.row_id WHERE a.name = ? ORDER BY r.ordinal
        """,
            (identifier,),
        ):
            yield self._row(fields)

    def _check_source(self):
        if _signature(self._path) != self._signature:
            raise ValueError(f"Stockholm input changed while in use: {self._path}")

    def extract(
        self, row_ids, output, *, reference_row_id=None, validate_fragment=None, transcribe_t=False
    ):
        """Write ordered rows and optionally return one gapped reference.

        The optional validate_fragment(row_id, text) receives the scoring fragment in
        input order. With transcribe_t, only exported sequence tracks map T/t to U/u.
        The returned reference always has its original sequence characters.
        Raising an exception preserves any existing output file.
        """
        ordered = tuple(row_ids)
        selected = set(ordered)
        if not ordered or len(selected) != len(ordered):
            raise ValueError("Stockholm extraction requires a nonempty list of unique row IDs")
        if reference_row_id is not None and reference_row_id not in selected:
            raise ValueError("the reference row ID must be included in the requested rows")
        sources = defaultdict(list)
        for row_id in ordered:
            if self.row(row_id) is None:
                raise ValueError(f"requested Stockholm row {row_id!r} was not found")
            sources[row_id].append(row_id)
            if match := _DOMAIN.fullmatch(row_id):
                sources[match[1]].append(row_id)
        output = Path(output)
        if self._path.resolve() == output.resolve():
            raise ValueError("Stockholm input and output must be distinct paths")
        self._check_source()
        self._db.execute("""
            CREATE TABLE piece (row_id TEXT, tag TEXT, line INTEGER, value TEXT,
                                PRIMARY KEY(row_id, tag, line)) WITHOUT ROWID
        """)
        try:
            with tempfile.TemporaryDirectory(dir=output.parent) as staging:
                temp = Path(staging) / "selected"
                with _destination(temp, output.suffix.lower() == ".gz") as out:
                    out.write("# STOCKHOLM 1.0\n")
                    written_ac = set()
                    with _source(self._path) as stream:
                        stream.seek(self._offset)
                        for line, kind, row, tag, value in _events(stream):
                            if kind == "#=GF":
                                if tag == "SQ":
                                    value = str(len(ordered))
                                if tag in ("NH", "TN") and len(ordered) < self.n_rows:
                                    continue
                                out.write(f"#=GF {tag} {value}\n")
                            elif kind == "#=GS":
                                for target in sources.get(row, ()):
                                    if tag == "AC":
                                        if target in written_ac:
                                            continue
                                        written_ac.add(target)
                                    out.write(f"#=GS {target} {tag} {value}\n")
                            elif kind == "#=GR" and row not in selected and row in sources:
                                if self.row(row) is None:
                                    self._db.execute(
                                        "INSERT INTO piece VALUES (?, ?, ?, ?)",
                                        (sources[row][0], tag, line, value),
                                    )
                            elif kind == "#=GC" or (
                                kind in ("sequence", "#=GR") and row in selected
                            ):
                                if kind == "sequence" and validate_fragment is not None:
                                    validate_fragment(
                                        row, value.translate(_TRANSCRIBE) if transcribe_t else value
                                    )
                                self._db.execute(
                                    "INSERT INTO piece VALUES (?, ?, ?, ?)", (row, tag, line, value)
                                )
                    reference = self._write_tracks(out, ordered, reference_row_id, transcribe_t)
                    out.write("//\n")
                self._check_source()
                os.replace(temp, output)
                return reference
        finally:
            self._db.execute("DROP TABLE piece")

    def _write_tracks(self, out, ordered, reference_row_id, transcribe_t):
        reference = []
        for row in chain(ordered, ("",)):
            previous = None
            for tag, value in self._db.execute(
                "SELECT tag, value FROM piece WHERE row_id=? ORDER BY tag, line", (row,)
            ):
                if previous != tag:
                    if previous is not None:
                        out.write("\n")
                    label = row
                    if not row:
                        label = f"#=GC {tag}"
                    elif tag:
                        label = f"#=GR {row} {tag}"
                    out.write(label + " ")
                    previous = tag
                out.write(
                    value.translate(_TRANSCRIBE) if transcribe_t and row and not tag else value
                )
                if row == reference_row_id and not tag:
                    reference.append(value)
            if previous is not None:
                out.write("\n")
        return "".join(reference) if reference_row_id is not None else None


@contextmanager
def open_stockholm(path, *, family=None, temp_dir=None, molecule=None):
    """Open an on-disk row inventory for one plain or gzipped Stockholm family.

    A multi-record database requires a unique GF ID or AC selector. An accession
    without a version matches the corresponding versioned GF AC. Descending
    coordinates on reverse-strand rows are preserved unless molecule="protein".
    """
    path = Path(path).expanduser().absolute()
    signature = _signature(path)
    if molecule not in (None, "protein", "dna", "rna"):
        raise ValueError(f"unknown Stockholm molecule type {molecule!r}")
    with (
        tempfile.TemporaryDirectory(dir=temp_dir) as scratch,
        closing(sqlite3.connect(Path(scratch) / "inventory.sqlite")) as db,
    ):
        db.execute("PRAGMA journal_mode = OFF")
        db.execute("PRAGMA temp_store = FILE")
        db.execute("PRAGMA cache_size = -2048")
        with _source(path) as stream:
            offset = _header(stream) if family is None else _family_offset(stream, family)
            index = _index(db, stream, molecule != "protein")
            if family is None:
                for raw in stream:
                    if raw.strip() == _HEADER:
                        raise ValueError(
                            "multiple Stockholm records; specify a family ID or accession"
                        )
                    if raw.strip():
                        raise ValueError("unexpected content after // terminator")
        inventory = StockholmInventory(db, path, offset, signature, *index)
        inventory._check_source()
        yield inventory
