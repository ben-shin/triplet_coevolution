"""Batch active UniProtKB and proteome metadata lookups with a replayable SQLite cache."""

import json
import re
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .identifiers import accession

BASE = "https://rest.uniprot.org"
KB_FIELDS = "accession,id,organism_id,xref_proteomes,xref_orthodb,xref_eggnog,xref_oma"
PROTEOME_FIELDS = "upid,genome_assembly,organism_id"
SCHEMA = 1
MAPPING_SCHEMA = 2


class UniProtError(ValueError):
    """The response or transport failed; a biological absence cannot be inferred."""


def _replicon_annotations(payload):
    """Expand each original proteome association into its documented component alternatives."""
    version = payload.get("mapping_schema", 1)
    if version == MAPPING_SCHEMA:
        return payload
    if version != 1:
        raise UniProtError(f"unsupported UniProt mapping schema {version}")
    mappings = []
    for mapping in payload["mappings"]:
        # V1 cached the complete cross-references with their original properties, even
        # though it emitted only one molecular mapping per distinct proteome accession.
        refs = {json.dumps(r, sort_keys=True): r for r in mapping.get("cross_references", ())}
        cross_references = [refs[key] for key in sorted(refs)]
        components = {
            prop["value"].strip()
            for ref in cross_references
            if ref.get("database") == "Proteomes" and ref.get("id") == mapping.get("proteome_id")
            for prop in ref.get("properties", ())
            if prop.get("key") == "Component" and prop.get("value", "").strip()
        }
        for replicon in sorted(components) or [""]:
            mappings.append(
                {**mapping, "cross_references": cross_references, "replicon_id": replicon}
            )
    return {**payload, "mappings": mappings, "mapping_schema": MAPPING_SCHEMA}


def _retry_delay(value, attempt):
    if value:
        try:
            delay = int(value)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = 2**attempt
        if delay > 60:
            raise ValueError(f"UniProt asks to retry in {int(delay)} seconds; resume later")
        return max(0, delay)
    return 2**attempt


def _request(url):
    request = Request(
        url, headers={"Accept": "application/json", "User-Agent": "triplet-coevolution"}
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                if not isinstance((data := json.load(response)), dict):
                    raise UniProtError("invalid UniProt JSON object")
                return data, response.headers
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise ValueError(
                    f"UniProt returned HTTP {exc.code} for {url}; no missing-ID inference"
                ) from exc
            time.sleep(_retry_delay(exc.headers.get("Retry-After"), attempt))
        except (TimeoutError, URLError) as exc:
            if attempt == 3:
                raise ValueError(f"could not reach UniProt at {url}: {exc}") from exc
            time.sleep(2**attempt)


class Resolver:
    def __init__(self, cache: Path | None, live: bool):
        self.live = live
        self.stats = Counter()
        self.releases = set()
        self.cache_name = cache.name if cache else None
        self.db = None
        if cache is None and not live:
            return
        if cache is None:
            self.db = sqlite3.connect("")  # private temporary on-disk database
        elif live:
            self.db = sqlite3.connect(cache)
        else:
            if not cache.is_file():
                raise ValueError(f"UniProt cache does not exist: {cache}")
            self.db = sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if (
                live
                and version == 0
                and not self.db.execute("SELECT 1 FROM sqlite_master").fetchone()
            ):
                self.db.execute("CREATE TABLE lookups (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                self.db.execute(f"PRAGMA user_version={SCHEMA}")
            elif (
                version != SCHEMA
                or not self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lookups'"
                ).fetchone()
            ):
                raise ValueError(f"unrecognized UniProt cache format: {cache}")
        except BaseException:
            self.db.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()

    def search(self, endpoint, query, fields):
        params = {"query": query, "fields": fields, "format": "json", "size": 500}
        if endpoint == "uniprotkb":
            params["includeIsoform"] = "true"
        url = f"{BASE}/{endpoint}/search?{urlencode(params)}"
        records, seen = [], set()
        release, expected_total = None, None
        while url:
            if url in seen or urlparse(url)[:2] != ("https", "rest.uniprot.org"):
                raise ValueError("invalid UniProt pagination link")
            seen.add(url)
            self.stats["requests"] += 1
            payload, headers = _request(url)
            rows = payload.get("results")
            if not isinstance(rows, list):
                raise UniProtError("UniProt response is missing the results list")
            current = (
                headers.get("x-uniprot-release", ""),
                headers.get("x-uniprot-release-date", ""),
            )
            if release is not None and current != release:
                raise ValueError("UniProt release changed during pagination; retry the batch")
            release = current
            total = headers.get("x-total-results")
            if total is not None:
                if expected_total is not None and expected_total != int(total):
                    raise UniProtError("UniProt result count changed during pagination")
                expected_total = int(total)
            records.extend(rows)
            link = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
            url = link.group(1) if link else None
        if expected_total is not None and len(records) != expected_total:
            raise UniProtError("incomplete UniProt pages; retry the batch")
        self.releases.add(release)
        return records, release

    def _secondary(self, ids):
        """Exact association without assigning hits from a batched OR to the wrong query."""
        if not ids:
            return {}
        hits, release = self.search(
            "uniprotkb",
            "(" + " OR ".join("sec_acc:" + i for i in ids) + ") AND active:true",
            KB_FIELDS,
        )
        if not hits:
            return {}
        if len(ids) == 1:
            return {ids[0]: (hits, release)}
        split = len(ids) // 2
        return {**self._secondary(ids[:split]), **self._secondary(ids[split:])}

    def _fetch(self, ids):
        terms = [("accession:" if accession(i) else "id:") + i for i in ids]
        hits, kb_release = self.search(
            "uniprotkb", "(" + " OR ".join(terms) + ") AND active:true", KB_FIELDS
        )
        associated = {i: ([], kb_release) for i in ids}
        for hit in hits:
            if not hit.get("primaryAccession") or not hit.get("uniProtkbId"):
                raise ValueError("invalid active UniProtKB record without an accession/entry name")
            for key in {hit["primaryAccession"], hit["uniProtkbId"]} & associated.keys():
                associated[key][0].append(hit)
        missing = [i for i in ids if not associated[i][0] and accession(i) and "-" not in i]
        associated.update(self._secondary(missing))
        proteomes = sorted(
            {
                ref["id"]
                for records, _ in associated.values()
                for record in records
                for ref in record.get("uniProtKBCrossReferences", ())
                if ref["database"] == "Proteomes"
            }
        )
        assemblies = {}
        # Most proteomes recur across components/batches. Cache their verified absence as well.
        for i in range(0, len(proteomes), 48):
            chunk = proteomes[i : i + 48]
            need = []
            for proteome in chunk:
                row = self.db.execute(
                    "SELECT payload FROM lookups WHERE id=?", ("proteome:" + proteome,)
                ).fetchone()
                if row:
                    assemblies[proteome] = json.loads(row[0])
                else:
                    need.append(proteome)
            if need:
                rows, release = self.search(
                    "proteomes", " OR ".join("upid:" + v for v in need), PROTEOME_FIELDS
                )
                by_id = {r["id"]: r for r in rows}
                for proteome in need:
                    info = by_id.get(proteome, {}).get("genomeAssembly", {})
                    annotation = [info.get("assemblyId", ""), *release]
                    assemblies[proteome] = annotation
                    self._store("proteome:" + proteome, annotation)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        result = {}
        for identifier, (records, release) in associated.items():
            mappings = []
            for record in records:
                refs = record.get("uniProtKBCrossReferences", [])
                groups = sorted(
                    {
                        f"OrthoDB:{r['id']}"
                        for r in refs
                        if r["database"] == "OrthoDB" and re.fullmatch(r"\d+at\d+", r["id"])
                    }
                )
                memberships = sorted({r["id"] for r in refs if r["database"] == "Proteomes"})
                for proteome in memberships or [""]:
                    assembly, proteome_release, proteome_date = assemblies.get(
                        proteome, ["", "", ""]
                    )
                    mappings.append(
                        {
                            "protein_id": record["primaryAccession"],
                            "entry_name": record["uniProtkbId"],
                            "taxon_id": str(record.get("organism", {}).get("taxonId", "")),
                            "proteome_id": proteome,
                            "assembly_id": assembly,
                            "genome_id": assembly or proteome,
                            "orthogroups": groups,
                            "neighborhood_id": "",
                            "source": "UniProtKB",
                            "cross_references": refs,
                            "release": release[0],
                            "release_date": release[1],
                            "proteome_release": proteome_release,
                            "proteome_release_date": proteome_date,
                        }
                    )
            result[identifier] = _replicon_annotations(
                {
                    "mappings": mappings,
                    "release": release[0],
                    "release_date": release[1],
                    "fetched_at": now,
                }
            )
            self._store(identifier, result[identifier])
        self.db.commit()
        return result

    def _store(self, key, payload):
        self.db.execute(
            "INSERT OR REPLACE INTO lookups VALUES (?,?)",
            (key, json.dumps(payload, sort_keys=True)),
        )

    def resolve(self, identifiers):
        """Resolve at most one bounded chunk of normalized identifiers."""
        result, missing = {}, []
        for key in sorted(set(identifiers)):
            cached = (
                self.db.execute("SELECT payload FROM lookups WHERE id=?", (key,)).fetchone()
                if self.db is not None
                else None
            )
            if cached:
                self.stats["cache_hits"] += 1
                payload = json.loads(cached[0])
                if not isinstance(payload, dict) or not isinstance(payload.get("mappings"), list):
                    raise ValueError(f"invalid cache entry for {key}")
                self.releases.add((payload.get("release", ""), payload.get("release_date", "")))
                result[key] = _replicon_annotations(payload)
            else:
                missing.append(key)
        self.stats["cache_misses"] += len(missing)
        if self.live:
            for i in range(0, len(missing), 48):
                result.update(self._fetch(missing[i : i + 48]))
        return result

    def provenance(self):
        return {
            "mode": "live" if self.live else "cache" if self.cache_name else "curated_only",
            "uniprot": self.live,
            "cache": self.cache_name,
            "cache_schema": SCHEMA,
            "mapping_schema": MAPPING_SCHEMA,
            "endpoints": [f"{BASE}/uniprotkb/search", f"{BASE}/proteomes/search"],
            "releases": [{"release": r, "release_date": d} for r, d in sorted(self.releases)],
            "requests": self.stats["requests"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
        }
