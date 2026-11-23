"""Identifier-only nucleotide placement metadata. Sequence strings never enter this module."""

import json
import re
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from .identifiers import explicit_accession
from .uniprot import _retry_delay

ENA = "https://www.ebi.ac.uk/ena/portal/api/search"
NCBI = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome"
SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
RNACENTRAL = "https://rnacentral.org/api/v1/rna"
ASSEMBLY = re.compile(r"GC[AF]_\d{9}\.\d+")
_INSDC = re.compile(r"(?:[A-Z]\d{5}|[A-Z]{2}\d{6,8}|[A-Z]{4}\d{8,10}|[A-Z]{6}\d{9,11})(?:\.\d+)?")
_REFSEQ = re.compile(r"(?:N[CGTW]_\d{6,9}|NZ_(?:\d{6,9}|[A-Z]{2,6}\d{6,11}))(?:\.\d+)?")
_URS = re.compile(r"URS[A-F0-9]{10}(?:_[1-9]\d*)?")
_LIMIT = 4 * 1024 * 1024
_NOT_FOUND = object()


def nucleotide_key(value, *, allow_ambiguous=False):
    base = re.sub(r"/\d+-\d+$", "", value or "").upper()
    fields = base.split("|")
    if len(fields) >= 2 and fields[0] in {"REF", "GB", "EMB", "DBJ", "ENA", "RNACENTRAL"}:
        base, allow_ambiguous = fields[1], True
    if _URS.fullmatch(base) or _REFSEQ.fullmatch(base):
        return base
    if _INSDC.fullmatch(base) and (allow_ambiguous or not explicit_accession(base)):
        return base
    return ""


class ProviderError(ValueError):
    """An incomplete response or failed service, distinct from a verified record absence."""


def _request_json(url):
    try:
        with urlopen(
            Request(
                url, headers={"Accept": "application/json", "User-Agent": "triplet-coevolution"}
            ),
            timeout=10,
        ) as response:
            content = response.read(_LIMIT + 1)
            if len(content) > _LIMIT:
                raise ProviderError("metadata response exceeded 4 MiB")
            expected = response.headers.get("Content-Length")
            if expected is not None and int(expected) > len(content):
                # HTTPResponse.read(amount) can return short without raising on EOF.
                raise IncompleteRead(content, int(expected) - len(content))
            return json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderError("invalid metadata JSON") from exc


def _items(payload, key):
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ProviderError(f"missing metadata list {key!r}")
    return payload[key]


class NucleotideResolver:
    def __init__(self, path: Path | None, live: bool):
        self.live, self.path = live, path
        self.failures, self.last_request = {}, {}
        self.stats = Counter()
        if not live and path and not path.is_file():
            raise ValueError(f"nucleotide cache does not exist: {path}")
        self.db = sqlite3.connect(
            path if live and path else path.resolve().as_uri() + "?mode=ro" if path else "",
            uri=bool(path and not live),
        )
        try:
            if self.db.execute("PRAGMA application_id").fetchone()[0] != 1413697073:
                if (path and not live) or self.db.execute("SELECT 1 FROM sqlite_master").fetchone():
                    raise ValueError(f"not a nucleotide placement cache: {path}")
                self.db.execute("PRAGMA application_id=1413697073")
                self.db.execute("PRAGMA user_version=1")
                self.db.execute(
                    "CREATE TABLE annotations (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
            if self.db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError(f"unsupported nucleotide cache version: {path}")
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def request(self, service, url, *, allow_404=False):
        if service in self.failures:
            raise ProviderError(self.failures[service])
        interval = {"RNAcentral": 2.1, "NCBI": 0.36, "ENA": 0.2}[service]
        for attempt in range(2):
            delay = self.last_request.get(service, 0) + interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last_request[service] = time.monotonic()
            self.stats[f"{service}_requests"] += 1
            try:
                return _request_json(url)
            except HTTPError as exc:
                if exc.code == 404 and allow_404:
                    return _NOT_FOUND
                if attempt or exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise ProviderError(f"{service} HTTP {exc.code}") from exc
                try:
                    delay = _retry_delay(exc.headers.get("Retry-After"), attempt)
                except ValueError:
                    raise ProviderError(
                        f"{service} rate limit exceeds this run's retry window"
                    ) from exc
                if delay > 5:
                    raise ProviderError(f"{service} requests retry after {int(delay)}s") from exc
                time.sleep(delay)
            except (OSError, URLError, HTTPException) as exc:
                if attempt:
                    raise ProviderError(f"{service} transport error: {exc}") from exc
                time.sleep(1)
        raise AssertionError("unreachable")

    def _cached(self, prefix, keys, fetch, *, service, size):
        result, needed = {}, []
        for key in sorted(set(keys)):
            row = self.db.execute(
                "SELECT payload FROM annotations WHERE key=?", (prefix + key,)
            ).fetchone()
            if row:
                result[key] = json.loads(row[0])
                self.stats["cache_hits"] += 1
            elif self.live:
                needed.append(key)
            else:
                result[key] = {"status": "nucleotide_cache_miss"}
        for i in range(0, len(needed), size):
            chunk = needed[i : i + size]
            try:
                if service in self.failures:
                    raise ProviderError(self.failures[service])
                batch = fetch(chunk)
                if set(batch) != set(chunk):
                    raise ProviderError("incomplete identifier batch")
                for key, annotation in batch.items():
                    annotation["fetched_at"] = datetime.now(UTC).isoformat(timespec="seconds")
                    result[key] = annotation
                    self.db.execute(
                        "INSERT OR REPLACE INTO annotations VALUES (?,?)",
                        (prefix + key, json.dumps(annotation, sort_keys=True)),
                    )
                self.db.commit()
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError) as exc:
                self.db.rollback()
                self.failures.setdefault(service, str(exc) or type(exc).__name__)
                for key in chunk:
                    result[key] = {
                        "status": "provider_unavailable",
                        "provider_error": f"{service}: {self.failures[service]}",
                    }
        return result

    def _ena(self, keys):
        bases = sorted({k.split(".")[0] for k in keys})
        params = {
            "result": "sequence",
            "fields": "accession,sequence_version,assembly_accession,tax_id,base_count",
            "query": " OR ".join(f'accession="{b}"' for b in bases),
            "format": "json",
            "limit": "1000",
        }
        rows = self.request("ENA", ENA + "?" + urlencode(params))
        if not isinstance(rows, list) or len(rows) >= 1000:
            raise ProviderError("invalid or incomplete ENA sequence metadata batch")
        found = {k: [] for k in keys}
        for row in rows:
            ac = row["accession"].split(".")[0] + "." + str(int(row["sequence_version"]))
            for key in {ac, ac.split(".")[0]} & found.keys():
                assemblies = [
                    a for a in re.split(r"[;,\s]+", row.get("assembly_accession", "") or "") if a
                ]
                if any(not re.fullmatch(r"GC[AF]_\d{9}(?:\.\d+)?", a) for a in assemblies):
                    raise ProviderError("unrecognized ENA assembly accession list")
                for asm in assemblies or [""]:
                    found[key].append(
                        {
                            "nucleotide_accession": ac,
                            "assembly_id": asm if ASSEMBLY.fullmatch(asm) else "",
                            "reported_assembly": asm,
                            "length": int(row["base_count"]),
                            "taxon_id": str(row.get("tax_id", "")),
                            "replicon_id": ac,
                        }
                    )
        return {
            k: {
                "source": "ENA",
                "placements": v,
                "status": "mapped" if v else "nucleotide_not_found",
            }
            for k, v in found.items()
        }

    def _ncbi(self, keys):
        params = {
            "db": "nuccore",
            "id": ",".join(keys),
            "retmode": "json",
            "version": "2.0",
            "tool": "triplet_coevolution",
        }
        payload = self.request("NCBI", SUMMARY + "?" + urlencode(params))
        result = payload.get("result", {})
        uids = _items(result, "uids")
        by_key = {k: [] for k in keys}
        for uid in uids:
            record = result[str(uid)]
            ac = record.get("accessionversion", "")
            if not _REFSEQ.fullmatch(ac) or "." not in ac:
                raise ProviderError("NCBI nuccore did not confirm a versioned genomic accession")
            matched = {ac, ac.split(".")[0]} & by_key.keys()
            if not matched:
                continue
            for key in matched:
                by_key[key].append(
                    {
                        "assembly_id": "",
                        "nucleotide_accession": ac,
                        "length": int(record["slen"]),
                        "taxon_id": str(record.get("taxid", "")),
                        "replicon_id": ac,
                    }
                )
        return {
            k: {
                "source": "NCBI",
                "placements": v,
                "status": "mapped" if v else "nucleotide_not_found",
            }
            for k, v in by_key.items()
        }

    def _memberships(self, ac):
        data = self.request(
            "NCBI", f"{NCBI}/sequence_accession/{ac}/sequence_assemblies", allow_404=True
        )
        assemblies = [] if data is _NOT_FOUND or data == {} else _items(data, "accessions")
        if any(not ASSEMBLY.fullmatch(a) for a in assemblies):
            raise ProviderError("invalid NCBI assembly membership")
        return assemblies

    def _rnacentral(self, keys):
        (key,) = keys
        urs, taxon = key.split("_")
        path = f"/api/v1/rna/{urs}/genome-locations/{taxon}"
        url = (
            "https://rnacentral.org" + path + "?" + urlencode({"format": "json", "page_size": 100})
        )
        locations, seen, loci = [], set(), set()
        expected = None
        while url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query, keep_blank_values=True)
            if (
                (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"))
                != ("https", "rnacentral.org", path)
                or parsed.fragment
                or set(params) - {"format", "page_size", "page"}
                or params.get("format", ["json"]) != ["json"]
                or params.get("page_size", ["100"]) != ["100"]
                or (seen and params.get("page") != [str(len(seen) + 1)])
                or url in seen
                or len(seen) >= 50
            ):
                raise ProviderError("invalid or incomplete RNAcentral pagination")
            seen.add(url)
            if len(seen) > 1:
                # Keep the original page size even if a returned link omits it.
                url = (
                    RNACENTRAL
                    + f"/{urs}/genome-locations/{taxon}?"
                    + urlencode({"format": "json", "page_size": 100, "page": len(seen)})
                )
            data = self.request("RNAcentral", url, allow_404=len(seen) == 1)
            if data is _NOT_FOUND:
                break
            page = _items(data, "results")
            if "next" not in data or (
                data["next"] is not None and not isinstance(data["next"], str)
            ):
                raise ProviderError("missing or invalid RNAcentral next-page marker")
            following = data["next"]
            if following == "" or (not page and (following is not None or len(seen) > 1)):
                raise ProviderError("incomplete RNAcentral location page")
            total = data.get("count")
            if "count" in data and (type(total) is not int or total < 0):
                raise ProviderError("invalid RNAcentral location count")
            if len(seen) == 1:
                expected = total
            elif total != expected:
                raise ProviderError("RNAcentral location count changed during pagination")
            for loc in page:
                assembly = loc.get("assembly")
                asm = (
                    assembly.get("gca_accession")
                    if isinstance(assembly, dict)
                    else assembly
                    if isinstance(assembly, str) and ASSEMBLY.fullmatch(assembly)
                    else ""
                ) or loc.get("gca_accession", "")
                if asm and not ASSEMBLY.fullmatch(asm):
                    raise ProviderError(
                        "RNAcentral supplied an unversioned or invalid assembly accession"
                    )
                start, end = int(loc["start"]), int(loc["end"])
                strand = {"1": "+", "+": "+", "-1": "-", "-": "-"}.get(str(loc.get("strand")))
                if min(start, end) < 0 or end < start or not strand or not loc.get("chromosome"):
                    raise ProviderError("RNAcentral location lacks usable coordinates/strand")
                locus = (
                    asm or json.dumps(assembly, sort_keys=True),
                    str(loc["chromosome"]),
                    start,
                    end,
                    strand,
                )
                if locus in loci:
                    raise ProviderError(
                        "duplicate RNAcentral genomic location in paginated results"
                    )
                loci.add(locus)
                locations.append(
                    {
                        "assembly_id": asm,
                        "taxon_id": taxon,
                        "nucleotide_accession": key,
                        "replicon_id": str(loc["chromosome"]),
                        "genomic_start": start,
                        "genomic_end": end,
                        "genomic_strand": strand,
                        "reported_assembly": assembly,
                    }
                )
            if expected is not None and (
                len(locations) > expected or (following is None) != (len(locations) == expected)
            ):
                raise ProviderError("incomplete RNAcentral location count or page chain")
            url = following
        return {
            key: {
                "source": "RNAcentral",
                "placements": locations,
                "status": "mapped" if locations else "nucleotide_not_found",
            }
        }

    def resolve(self, keys):
        keys = set(keys)
        result = {}
        for service, pattern, fn, size in (
            ("ENA", _INSDC, self._ena, 40),
            ("NCBI", _REFSEQ, self._ncbi, 30),
            ("RNAcentral", _URS, self._rnacentral, 1),
        ):
            subset = {k for k in keys if pattern.fullmatch(k)}
            if service == "RNAcentral":
                for k in tuple(subset):
                    if "_" not in k:
                        result[k] = {"source": service, "status": "rnacentral_taxon_required"}
                        subset.remove(k)
            prefix = "RNAcentral:v2:" if service == "RNAcentral" else service + ":"
            records = self._cached(prefix, subset, fn, service=service, size=size)
            if service == "RNAcentral" and not self.live:
                for key, payload in records.items():
                    if (
                        payload.get("status") == "nucleotide_cache_miss"
                        and self.db.execute(
                            "SELECT 1 FROM annotations WHERE key=?", ("RNAcentral:" + key,)
                        ).fetchone()
                    ):
                        error = "older RNAcentral cache lacks verified pagination; refresh with a connected run"
                        self.failures[service] = error
                        records[key] = {"status": "provider_unavailable", "provider_error": error}
            result.update(records)
        # Cache sequence metadata and individual assembly memberships independently.
        # A failed link lookup must not discard other completed records from a batch.
        unplaced = {
            p["nucleotide_accession"]
            for info in result.values()
            if info.get("source") in {"ENA", "NCBI"}
            for p in info.get("placements", ())
            if not p["assembly_id"]
        }
        extra = self._cached(
            "membership:",
            unplaced,
            lambda chunk: {ac: {"assemblies": self._memberships(ac)} for ac in chunk},
            service="NCBI",
            size=1,
        )
        for key, info in list(result.items()):
            if info.get("source") not in {"ENA", "NCBI"} or not info.get("placements"):
                continue
            updated = []
            for placement in info["placements"]:
                ac = placement["nucleotide_accession"]
                supplement = extra.get(ac, {}) if not placement["assembly_id"] else {}
                if supplement.get("status") in {"provider_unavailable", "nucleotide_cache_miss"}:
                    result[key] = {**info, **supplement}
                    break
                for assembly in supplement.get("assemblies", ()) or [placement["assembly_id"]]:
                    updated.append(
                        {
                            **placement,
                            "assembly_id": assembly,
                            "placement_source": info["source"] + " + NCBI Datasets"
                            if supplement.get("assemblies")
                            else info["source"],
                        }
                    )
            else:
                result[key] = {**info, "placements": updated}
        return result

    def assembly_pairs(self, keys):
        def fetch(chunk):
            url = f"{NCBI}/accession/{','.join(chunk)}/dataset_report?" + urlencode(
                {"page_size": 100}
            )
            response = self.request("NCBI", url)
            if response.get("next_page_token"):
                raise ProviderError("incomplete NCBI assembly-pair report")
            answers = {k: {} for k in chunk}
            for report in _items(response, "reports"):
                ac = report.get("accession")
                if ac not in answers:
                    continue
                paired = report.get("assembly_info", {}).get("paired_assembly", {})
                gca = paired.get("accession") or report.get("paired_accession", "")
                different = paired.get("refseq_genbank_are_different")
                if (
                    isinstance(gca, str)
                    and gca.startswith("GCA_")
                    and ASSEMBLY.fullmatch(gca)
                    and paired.get("accession") == gca
                    and different is False
                    and not any(
                        paired.get(k)
                        for k in (
                            "differences",
                            "changed",
                            "only_refseq",
                            "only_genbank",
                            "manual_diff",
                        )
                    )
                ):
                    answers[ac] = {
                        "canonical": gca,
                        "source": "NCBI Datasets paired assembly",
                        "paired_record": paired,
                    }
            return answers

        return self._cached("pair:", keys, fetch, service="NCBI", size=30)
