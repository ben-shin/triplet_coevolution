"""Explicit PDB file retrieval, immutable digests and local reference identities."""

import csv
import gzip
import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from Bio import SeqIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from triplet_coevolution.io.score_store import file_digest
from triplet_coevolution.structure.config import StructureConfig


def entry_matches(expected, observed):
    """Accept the equivalent numeric legacy and expanded PDB accessions."""

    def canonical(value):
        name = value.lower()
        return "pdb_0000" + name if len(name) == 4 else name

    return canonical(expected) == canonical(observed)


def validate_ambiguous_source(path, expected):
    """Verify an explicit remote entry even when the parser returns a choice before its ID."""
    with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
        fields = MMCIF2Dict(stream)
    values = fields.get("_entry.id", [])
    if len(values) != 1 or not entry_matches(expected, values[0]):
        raise ValueError("coordinate entry ID does not match the requested public PDB ID")


@contextmanager
def coordinate_source(
    config: StructureConfig, *, download: bool = False
) -> Iterator[tuple[Path, dict]]:
    """Verify local or cached bytes. Retrieve only an explicitly named public PDB entry."""
    expected = config.expected_sha256
    if config.coordinate_file:
        path = config.coordinate_file
        digest = file_digest(path)
        if expected and expected != digest:
            raise ValueError("local coordinate file SHA-256 mismatch")
        yield path, {"source": "local", "file": path.name, "sha256": digest}
        return

    entry = config.pdb_id
    url = f"https://files.rcsb.org/download/{entry}.cif.gz"
    directory = config.cache / entry
    record = directory / "source.json"
    if record.is_file():
        info = json.loads(record.read_text(encoding="utf-8"))
        if info.get("url") != url:
            raise ValueError("cached structure source does not match the requested ID")
        cached = directory / "coordinates.cif.gz"
        digest = file_digest(cached)
        if digest != info.get("sha256") or (expected and digest != expected):
            raise ValueError("cached structure checksum mismatch")
        yield cached, {**info, "source": "cache"}
        return
    if not download:
        raise ValueError(
            f"PDB entry {entry} is absent from the cache. Use --download to retrieve the named entry."
        )
    config.cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=config.cache) as scratch:
        staging = Path(scratch)
        candidate = staging / "coordinates.cif.gz"
        req = Request(url, headers={"User-Agent": "triplet-coevolution/structure-map"})
        with urlopen(req, timeout=30) as response, candidate.open("wb") as target:
            headers = response.headers
            version = {k: headers.get(k) for k in ("ETag", "Last-Modified") if headers.get(k)}
            shutil.copyfileobj(response, target, length=1024 * 1024)
        digest = file_digest(candidate)
        if expected and digest != expected:
            raise ValueError("downloaded structure checksum mismatch")
        with gzip.open(candidate, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        info = {
            "source": "download",
            "pdb_id": entry,
            "url": url,
            "sha256": digest,
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "http_version": version,
        }
        (staging / "source.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        yield candidate, info
        # Publish both cached files together only after the caller parses them.
        try:
            staging.rename(directory)
        except OSError:
            if not record.is_file() or file_digest(directory / "coordinates.cif.gz") != digest:
                raise ValueError(
                    "concurrent structure cache publication disagreed with downloaded bytes"
                ) from None


def read_references(config: StructureConfig):
    """Select each exact FASTA row, retaining the declared biological reference ID."""
    from triplet_coevolution.structure.mapping import ReferenceSequence

    references = {}
    provenance = {}
    for spec in config.references:
        sha = file_digest(spec.fasta)
        with spec.fasta.open("rb") as raw:
            compressed = raw.read(2) == b"\x1f\x8b"
        opener = gzip.open if compressed else open
        with opener(spec.fasta, "rt", encoding="utf-8-sig") as stream:
            sequence = None
            for row in SeqIO.parse(stream, "fasta"):
                if row.id == spec.row_id:
                    if sequence is not None:
                        raise ValueError(
                            f"duplicate reference FASTA ID in {spec.component}: {spec.row_id}"
                        )
                    sequence = str(row.seq)
        if not sequence:
            raise ValueError(f"missing reference FASTA row in {spec.component}: {spec.row_id}")
        if file_digest(spec.fasta) != sha:
            raise ValueError(f"reference FASTA changed while reading {spec.component}")
        references[spec.component] = ReferenceSequence(
            spec.component,
            spec.reference_id,
            sequence,
            spec.molecule,
            spec.start,
            spec.strand,
            spec.organism,
        )
        provenance[spec.component] = {
            "file": spec.fasta.name,
            "sha256": sha,
            "reference_id": spec.reference_id,
            "row_id": spec.row_id,
            "start": spec.start,
            "strand": spec.strand,
            "molecule": spec.molecule,
            "organism": spec.organism,
        }
    return references, provenance


def read_anchors(path: Path | None) -> dict[str, dict[int, int]]:
    if path is None:
        return {}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t", strict=True)
        required = {"component", "reference_position", "polymer_index"}
        if not rows.fieldnames or len(rows.fieldnames) != 3 or set(rows.fieldnames) != required:
            raise ValueError(
                "explicit map TSV requires component, reference_position and polymer_index"
            )
        for row in rows:
            if None in row or row["component"] not in ("A", "B", "C"):
                raise ValueError(f"invalid explicit map row {rows.line_num}")
            pos, polymer = int(row["reference_position"]), int(row["polymer_index"])
            role = result.setdefault(row["component"], {})
            if pos < 1 or polymer < 1 or pos in role:
                raise ValueError(
                    f"nonpositive or duplicate explicit map position at row {rows.line_num}"
                )
            role[pos] = polymer
    if not result:
        raise ValueError("explicit map TSV is empty")
    return result
