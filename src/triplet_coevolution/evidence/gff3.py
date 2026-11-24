"""Read local GFF3 annotations and preserve locus alternatives for evidence reconciliation."""

import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

from triplet_coevolution.evidence.nucleotide import ASSEMBLY

RNA = {
    "rna",
    "ncrna",
    "rrna",
    "trna",
    "mrna",
    "tmrna",
    "snrna",
    "snorna",
    "lnc_rna",
    "mirna",
    "primary_transcript",
}
TYPES = {
    "protein": {"cds"},
    "rna": RNA,
    "dna": {"gene", "pseudogene", "regulatory_region", "promoter", "origin_of_replication"},
}


def decode(raw):
    if re.search(r"%(?![0-9A-Fa-f]{2})", raw):
        raise ValueError("invalid GFF3 percent escape")
    value = unquote(raw, errors="strict")
    if any(ord(c) < 32 for c in value):
        raise ValueError("control character in GFF3 identity or annotation")
    return value


def attributes(raw):
    result = {}
    if raw == ".":
        return result
    for item in raw.rstrip(";").split(";"):
        name, sep, value = item.partition("=")
        name = decode(name)
        if not name or not sep or name in result:
            raise ValueError("malformed or duplicate GFF3 attribute")
        result[name] = [decode(v) for v in value.split(",") if v]
    return result


def parse_specs(value, base):
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError("gff3 must be a nonempty array of local annotation source objects")
    result = []
    for raw in value:
        if (
            not isinstance(raw, dict)
            or "path" not in raw
            or set(raw)
            - {
                "path",
                "assembly_id",
                "release",
                "components",
                "system_attribute",
                "orthogroup_attribute",
                "taxon_id",
                "system_windows",
            }
        ):
            raise ValueError("invalid GFF3 source settings")
        src = dict(raw)
        if not isinstance(src["path"], (str, Path)):
            raise TypeError("GFF3 source requires a local file")
        src["path"] = (Path(base) / src["path"]).resolve()
        if not src["path"].is_file():
            raise ValueError(f"GFF3 source missing: {src['path']}")
        if "system_windows" in src:
            src["system_windows"] = (Path(base) / src["system_windows"]).resolve()
            if not src["system_windows"].is_file():
                raise ValueError("system_windows must identify a local TSV file")
        if "assembly_id" in src and (
            not isinstance(src["assembly_id"], str) or not ASSEMBLY.fullmatch(src["assembly_id"])
        ):
            raise ValueError("GFF3 assembly_id requires a versioned GCA/GCF accession")
        roles = src.setdefault("components", list("ABC"))
        if (
            not isinstance(roles, list)
            or not roles
            or set(roles) - set("ABC")
            or len(set(roles)) != len(roles)
        ):
            raise ValueError("GFF3 components must list distinct roles A, B or C")
        for k in ("release", "system_attribute", "orthogroup_attribute", "taxon_id"):
            if k in src and (
                not isinstance(src[k], str) or not src[k] or any(ord(c) < 32 for c in src[k])
            ):
                raise ValueError(f"GFF3 {k} must be nonempty text")
        result.append(src)
    return result


def load_windows(db, path, assembly):
    db.executescript(
        "DROP TABLE IF EXISTS temp.system_windows; CREATE TEMP TABLE system_windows(seq TEXT,lo INTEGER,hi INTEGER,strand TEXT,name TEXT); CREATE INDEX temp.windows_index ON system_windows(seq,lo,hi);"
    )
    if not path:
        return None
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", strict=True)
        if reader.fieldnames != [
            "assembly_id",
            "replicon_id",
            "start",
            "end",
            "strand",
            "system_id",
        ]:
            raise ValueError("invalid GFF3 system_windows header")
        for row in reader:
            if (
                None in row
                or None in row.values()
                or row["assembly_id"] != assembly
                or not row["replicon_id"]
                or not row["system_id"]
                or row["strand"] not in ("+", "-", ".")
                or any(any(ord(c) < 32 for c in v) for v in row.values())
            ):
                raise ValueError(f"invalid system window at line {reader.line_num}")
            lo, hi = int(row["start"]), int(row["end"])
            if lo < 1 or hi < lo:
                raise ValueError("system windows require 1-based inclusive intervals")
            db.execute(
                "INSERT INTO system_windows VALUES(?,?,?,?,?)",
                (row["replicon_id"], lo, hi, row["strand"], row["system_id"]),
            )
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != digest:
            raise ValueError("system windows changed while reading")
    return digest


def load_annotations(db, sources, molecules):
    stats = []
    for src in parse_specs(sources, Path.cwd()) if sources else []:
        path = src["path"]
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
            f.seek(0)
            opener = gzip.open if f.read(2) == b"\x1f\x8b" else open
        assembly = src.get("assembly_id")
        db.executescript(
            "DROP TABLE IF EXISTS temp.features; CREATE TEMP TABLE features(fid TEXT,seq TEXT,kind TEXT,lo INTEGER,hi INTEGER,strand TEXT,attrs TEXT,line INTEGER); CREATE INDEX temp.feature_id ON features(fid);"
        )
        count = 0
        version = False
        with opener(path, "rt", encoding="utf-8-sig") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.rstrip("\r\n")
                if line == "##FASTA":
                    break
                if line.startswith("##gff-version"):
                    if line.split() != ["##gff-version", "3"]:
                        raise ValueError("GFF3 version 3 required")
                    version = True
                if line.startswith("#!genome-build-accession "):
                    found = line.split()[-1].split(":")[-1]
                    if not ASSEMBLY.fullmatch(found) or assembly and assembly != found:
                        raise ValueError(
                            "GFF3 assembly pragma conflicts with the versioned source assembly"
                        )
                    assembly = found
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) != 9 or not version:
                    raise ValueError(
                        f"GFF3 needs a version directive and nine columns (line {line_no})"
                    )
                seq, _, kind, lo, hi, _, strand, phase, raw_attrs = fields
                lo, hi = int(lo), int(hi)
                if (
                    lo < 1
                    or hi < lo
                    or strand not in ("+", "-", ".", "?")
                    or phase not in (".", "0", "1", "2")
                ):
                    raise ValueError(f"invalid GFF3 coordinates, strand or phase at line {line_no}")
                attrs = attributes(raw_attrs)
                if len(attrs.get("ID", [])) > 1:
                    raise ValueError(f"GFF3 feature has multiple IDs at line {line_no}")
                fid = next(iter(attrs.get("ID", [])), f"#line:{line_no}")
                db.execute(
                    "INSERT INTO features VALUES(?,?,?,?,?,?,?,?)",
                    (fid, decode(seq), kind.lower(), lo, hi, strand, json.dumps(attrs), line_no),
                )
                count += 1
        if not version or not assembly:
            raise ValueError(
                "GFF3 needs ##gff-version 3 and an explicit versioned assembly_id or matching genome-build-accession pragma"
            )
        collision = db.execute("""
            SELECT fid FROM (SELECT DISTINCT fid, seq, kind, strand FROM features)
            GROUP BY fid HAVING COUNT(*)>1 LIMIT 1
        """).fetchone()
        if collision:
            raise ValueError(
                f"GFF3 feature ID {collision[0]!r} has conflicting replicons, types or strands"
            )
        window_sha = load_windows(db, src.get("system_windows"), assembly)
        types = {role: TYPES.get(molecules.get(role), set()) for role in src["components"]}
        if not any(types.values()):
            raise ValueError(
                "GFF3 annotation requires at least one typed protein, DNA or RNA component"
            )
        mapped = 0
        for fid, seq, kind, strand, lo, hi in db.execute(
            "SELECT fid,seq,kind,strand,min(lo),max(hi) FROM features GROUP BY fid,seq,kind,strand ORDER BY seq,lo,fid"
        ):
            roles = [role for role, allowed in types.items() if kind in allowed]
            if not roles:
                continue
            segments = []
            tags = {}
            segment_identity = None
            for x, y, raw in db.execute(
                "SELECT lo,hi,attrs FROM features WHERE fid=? AND seq=? AND kind=? AND strand=? ORDER BY lo,hi",
                (fid, seq, kind, strand),
            ):
                claims = json.loads(raw)
                identity = tuple(
                    tuple(sorted(claims.get(key, ())))
                    for key in ("Parent", "locus_tag", "protein_id", "transcript_id")
                )
                if segment_identity is not None and identity != segment_identity:
                    raise ValueError(
                        f"GFF3 multipart feature {fid!r} has conflicting locus or parent claims"
                    )
                segment_identity = identity
                segments.append((x, y))
                for k, v in claims.items():
                    tags.setdefault(k, set()).update(v)
            aliases = {
                v
                for k in ("ID", "Alias", "locus_tag", "protein_id", "transcript_id")
                for v in tags.get(k, ())
            }
            dbx = tags.get("Dbxref", set())
            aliases.update(
                v.split(":", 1)[1]
                for v in dbx
                if v.startswith(
                    (
                        "UniProtKB:",
                        "UniProtKB/Swiss-Prot:",
                        "UniProtKB/TrEMBL:",
                        "RNAcentral:",
                        "RefSeq:",
                        "Genbank:",
                    )
                )
            )
            genomic_interval_id = None
            if len(segments) == 1 and strand in ("+", "-"):
                first, last = (lo, hi) if strand == "+" else (hi, lo)
                genomic_interval_id = f"{seq}/{first}-{last}"
            if not aliases and not genomic_interval_id:
                continue
            lineage = set()
            frontier = list(tags.get("Parent", ()))
            systems = set(tags.get(src.get("system_attribute", "operon"), ()))
            genes = set(tags.get("locus_tag", ()))
            blocked = set()
            if kind == "pseudogene" or "true" in tags.get("pseudo", ()) or "pseudogene" in tags:
                blocked.add("gff3_pseudogene")
            if "true" in tags.get("partial", ()) or {"start_range", "end_range"} & tags.keys():
                blocked.add("gff3_partial_feature")
            if strand not in ("+", "-"):
                blocked.add("gff3_unknown_strand")
            while frontier:
                parent = frontier.pop()
                if parent in lineage:
                    continue
                if len(lineage) >= 128:
                    raise ValueError("GFF3 ancestry exceeds 128 parent features")
                lineage.add(parent)
                for pk, ps, pl, ph, pstrand, pa in db.execute(
                    "SELECT kind,seq,lo,hi,strand,attrs FROM features WHERE fid=?", (parent,)
                ):
                    if (
                        ps != seq
                        or pl > lo
                        or ph < hi
                        or (pstrand in ("+", "-") and pstrand != strand)
                    ):
                        continue
                    pt = json.loads(pa)
                    if pk in ("gene", "pseudogene"):
                        genes.update(pt.get("locus_tag", [parent]))
                        if (
                            pk == "pseudogene"
                            or "true" in pt.get("pseudo", ())
                            or "pseudogene" in pt
                        ):
                            blocked.add("gff3_pseudogene")
                    if pk == "operon" and strand in ("+", "-") and pstrand in (strand, "."):
                        systems.add(parent)
                    if pk != "operon":
                        systems.update(pt.get(src.get("system_attribute", "operon"), []))
                    frontier.extend(pt.get("Parent", []))
            windows = sorted(
                {
                    v[0]
                    for v in db.execute(
                        "SELECT name FROM system_windows WHERE seq=? AND lo<=? AND hi>=? AND (strand='.' OR strand=?)",
                        (seq, lo, hi, strand),
                    )
                }
            )
            systems.update(windows)
            groups = sorted(
                set(tags.get(src.get("orthogroup_attribute", "orthogroup"), ()))
                | {v for v in dbx if v.startswith("OrthoDB:")}
            )
            fingerprint = json.dumps([assembly, seq, kind, fid, strand], separators=(",", ":"))
            tax = src.get("taxon_id", "")
            for role in roles:
                role_aliases = set(aliases)
                if genomic_interval_id and molecules[role] in ("dna", "rna"):
                    role_aliases.add(genomic_interval_id)
                for system in sorted(systems or {""}):
                    mapping = {
                        "protein_id": "gff3:" + fingerprint,
                        "genome_id": assembly,
                        "assembly_id": assembly,
                        "replicon_id": seq,
                        "genomic_start": lo,
                        "genomic_end": hi,
                        "genomic_strand": strand,
                        "neighborhood_id": system,
                        "orthogroups": groups,
                        "taxon_id": tax,
                        "source": "GFF3",
                        "source_type": "gff3",
                        "placement_source": "annotated locus",
                        "cross_references": sorted(dbx),
                        "molecule_type": molecules[role],
                        "release": src.get("release", ""),
                        "gff3": {
                            "feature_id": fid,
                            "type": kind,
                            "segments": segments,
                            "gene_loci": sorted(genes),
                            "annotation_ids": sorted(role_aliases),
                            "limitations": sorted(blocked),
                            "system_window_matches": windows,
                            "system_window_sha256": window_sha,
                            "source_sha256": digest,
                        },
                    }
                    raw = json.dumps(mapping, sort_keys=True)
                    db.executemany(
                        "INSERT INTO manifest VALUES(?,?,?)",
                        (
                            (role, v, raw)
                            for v in sorted(role_aliases)
                            if v and not any(c.isspace() for c in v)
                        ),
                    )
                    mapped += 1
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise ValueError("GFF3 source changed while indexing")
        stats.append(
            {
                "name": path.name,
                "sha256": digest,
                "assembly_id": assembly,
                "release": src.get("release"),
                "components": src["components"],
                "system_windows_sha256": window_sha,
                "features": count,
                "typed_mappings": mapped,
                "coordinates": "1-based inclusive on deposited replicon; multipart spans retained",
            }
        )
    return stats
