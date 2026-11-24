"""Local, auditable domain discovery with independently selected alignment rows."""

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.io.profiles import check_profile, read_reference, read_sequences
from triplet_coevolution.msa.builders import assemble_msa, comparison, from_trace, oriented
from triplet_coevolution.search.discovery_config import (
    BACKENDS,
    SEARCH_KEYS,
    validate_search_component,
)
from triplet_coevolution.search.hit_tables import HitRecord, read_hit_table
from triplet_coevolution.search.local_tools import _hash, checksum, raw_search, tools_for
from triplet_coevolution.search.profile_alignment import profile_align, stockholm_names
from triplet_coevolution.search.retrieve import _place_hit, _support
from triplet_coevolution.search.shared import Hit

__all__ = ["BACKENDS", "SEARCH_KEYS", "discover_component", "validate_search_component"]


@dataclass(frozen=True)
class DiscoveryResult:
    hits: dict[str, Hit]
    msa: ComponentMSA
    reference_row_id: str
    manifest: dict
    programs: list


def _portable(comp, ref):
    possible = sum(c in CANONICAL[comp["molecule"]] for c in ref)
    for seq_id, seq in read_sequences(comp["database"], comp["molecule"]):
        intervals = [(0, len(seq))]
        any_hit = False
        while intervals:
            left, right = intervals.pop()
            window = seq[left:right]
            if right - left < comp["min_coverage"] * possible:
                continue
            options = []
            for strand in ("+",) if comp["molecule"] == "protein" else ("+", "-"):
                target = oriented(window, comp["molecule"], strand)
                aln = comparison(ref, target, comp["molecule"], "local")
                if aln is None:
                    continue
                place = from_trace(aln, target, strand)
                if place is None:
                    continue
                ident, cov = _support(ref, place, comp["molecule"], possible)
                a, b = map(int, aln.coordinates[1, [0, -1]])
                first, last = (
                    (left + a + 1, left + b) if strand == "+" else (right - b + 1, right - a)
                )
                options.append(
                    Hit(
                        seq_id,
                        first,
                        last,
                        strand,
                        aln.score,
                        identity=ident,
                        coverage=cov,
                        placement=place,
                        query_id=comp["reference_id"],
                        target_length=len(seq),
                        model_length=len(ref),
                        model_first=place.first,
                        model_last=place.last,
                        source_strand=strand,
                        query_strand="+",
                    )
                )
            if not options:
                continue
            best = max(options, key=lambda h: (h.score, h.coverage, h.identity, h.strand == "+"))
            if best.identity < comp["min_identity"] or best.coverage < comp["min_coverage"]:
                if not any_hit:
                    yield HitRecord(0, "", best, "below_portable_identity_or_reference_coverage")
                    any_hit = True
                continue
            yield HitRecord(0, "", best, None)
            any_hit = True
            intervals.extend([(left, best.first - 1), (best.last, right)])
        if not any_hit:
            yield HitRecord(0, seq_id, None, "no_portable_qualifying_interval")


def _model_length(comp):
    if "hit_model_length" in comp:
        return comp["hit_model_length"]
    if "profile" in comp and comp["backend"] in ("hmmer", "infernal"):
        opener = gzip.open if comp["profile"].suffix.lower() == ".gz" else open
        with opener(comp["profile"], "rt", encoding="ascii") as f:
            for line in f:
                m = re.match(r"^(?:LENG|CLEN)\s+(\d+)\s*$", line)
                if m:
                    return int(m[1])
    return None


def _profile_labels(path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    labels = set()
    with opener(path, "rt", encoding="ascii") as f:
        for line in f:
            parts = line.split()
            if parts[:1] in (["NAME"], ["ACC"]) and len(parts) == 2:
                labels.add(parts[1])
            if parts[:1] in (["HMM"], ["CM"]):
                break
    return labels


def _details(h):
    return {k: v for k, v in asdict(h).items() if k != "placement"}


def _span(key, hit):
    a, b = (
        (hit.first, hit.last) if (hit.source_strand or hit.strand) == "+" else (hit.last, hit.first)
    )
    return f"{key}/{a}-{b}"


def _digest_hit(h):
    return _hash(
        [
            h.seq_id,
            h.first,
            h.last,
            h.source_strand or h.strand,
            h.query_id,
            h.model_first,
            h.model_last,
            h.query_strand,
        ]
    )


def _record(row, raw_id=None):
    return {
        "line": row.line,
        "raw_target": raw_id,
        "raw_text": row.text,
        "reason": row.reason or "",
        "status": "rejected" if row.reason else "reported",
        **(_details(row.hit) if row.hit else {}),
    }


def _reference_anchors(info, raw, comp, ref):
    if not info or not info["result"].get("reference_table"):
        return []
    records = list(
        read_hit_table(
            raw / info["result"]["reference_table"],
            info["result"]["format"],
            max_rows=comp["max_candidates"],
        )
    )
    if any(r.reason for r in records):
        raise ValueError("malformed reference anchor hit table")
    expected = comp["reference_id"] if comp["backend"] == "import" else "seed"
    anchors = [
        r.hit for r in records if r.hit.seq_id == expected and r.hit.evalue <= comp["evalue"]
    ]
    if any(h.last > len(ref) for h in anchors):
        raise ValueError("reference anchor lies outside seed sequence")
    return anchors


def _mask_reason(h, comp):
    if comp["backend"] == "reference":
        return None
    if comp["backend"] in ("jackhmmer", "hmmer") and h.sequence_evalue is not None:
        if h.sequence_evalue > comp["evalue"] or h.conditional_evalue > comp["domain_evalue"]:
            return "above_reporting_threshold"
    elif h.evalue is None or h.evalue > comp["evalue"]:
        return "above_evalue_threshold"
    if comp.get("hit_query") and comp["hit_query"] != h.query_id:
        return "different_query"
    if comp["molecule"] == "protein" and (h.source_strand == "-" or h.query_strand == "-"):
        return "incompatible_protein_strand"
    return None


def discover_component(name, comp, reference, directory, *, wanted_ids=None, cache_dir=None):
    """Search all local records, audit all reported intervals, align selected stable IDs plus seed."""
    comp = validate_search_component(dict(comp), name, Path.cwd())
    if read_reference(comp) != reference:
        raise ValueError("supplied sequence differs from declared query/reference FASTA")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(wanted_ids, (str, bytes)):
        raise TypeError("wanted_ids must be a collection of exact sequence/domain identifiers")
    wanted = None
    if wanted_ids is not None:
        wanted = set()
        for value in wanted_ids:
            if not isinstance(value, str) or not value:
                raise ValueError("wanted_ids must contain exact nonempty identifiers")
            wanted.add(value)
    if comp["backend"] in ("hmmer", "infernal"):
        check_profile(comp)
    if comp["alignment_backend"] in ("hmmalign", "cmalign"):
        check_profile(
            {
                "backend": "hmmer" if comp["alignment_backend"] == "hmmalign" else "infernal",
                "profile": comp.get("alignment_profile", comp.get("profile")),
                "molecule": comp["molecule"],
            }
        )
    programs = tools_for(comp)
    info = None
    cache_hit = False
    raw = out / "raw"
    aliases = {}
    jack_rows = None
    if comp["backend"] == "reference":
        source_digests = {
            k: checksum(comp[k])
            for k in ("reference", "database", "alignment_profile")
            if k in comp
        }
        records = _portable(comp, reference)
    else:
        info, cache_hit = raw_search(
            comp, reference, programs, out, cache_dir or comp.get("cache_dir")
        )
        source_digests = info["identity"]["sources"]
        result = info["result"]
        records = read_hit_table(
            raw / result["table"], result["format"], max_rows=comp["max_candidates"]
        )
        if result["aliases"]:
            records = list(records)
            reported_ids = {r.hit.seq_id for r in records if r.hit is not None}
            with (raw / "aliases.tsv").open(encoding="utf-8") as f:
                for line in f:
                    alias, source, length = line.rstrip("\n").split("\t")
                    if alias in reported_ids:
                        aliases[alias] = (source, int(length))
        if comp["backend"] == "jackhmmer":
            jack_rows = stockholm_names(raw / "final.sto", comp["max_candidates"])
            if "seed" not in jack_rows:
                raise ValueError("JACKHMMER final Stockholm missing query seed")
    by_id = defaultdict(list)
    audit = []
    queries = set()
    native_bind = {}
    model_len = _model_length(comp)
    for record in records:
        if len(audit) >= comp["max_candidates"]:
            raise ValueError("search reports exceed max_candidates")
        h = record.hit
        a = _record(record, h.seq_id if h else None)
        audit.append(a)
        if h is None or record.reason:
            continue
        raw_id = h.seq_id
        if info and info["result"]["aliases"]:
            if raw_id not in aliases:
                raise ValueError("generated search returned an unknown database alias")
            h = replace(h, seq_id=aliases[raw_id][0])
        queries.add(h.query_id)
        if model_len is not None and h.model_length is not None and model_len != h.model_length:
            raise ValueError("declared model length differs from the source hit table")
        h = replace(h, model_length=h.model_length or model_len)
        if h.model_length is not None and h.model_last > h.model_length:
            a.update(status="rejected", reason="model_coordinate_out_of_bounds")
            continue
        a.update(_details(h), domain_id=_digest_hit(h))
        reason = _mask_reason(h, comp)
        if jack_rows is not None:
            spans = [(h.first, h.last)]
            if h.envelope_first:
                spans.append((h.envelope_first, h.envelope_last))
            key = next(
                (f"{raw_id}/{i}-{j}" for i, j in spans if f"{raw_id}/{i}-{j}" in jack_rows), None
            )
            if key is None:
                reason = "reported_outside_final_inclusion_alignment"
            else:
                native_bind[a["domain_id"]] = key
        if reason:
            a.update(status="rejected", reason=reason)
        else:
            by_id[h.seq_id].append((h, a))
    if len(queries) > 1 and not comp.get("hit_query"):
        raise ValueError("import table contains multiple queries; specify hit_query explicitly")
    anchors = _reference_anchors(info, raw, comp, reference)
    query = comp.get("hit_query") or next(iter(queries), None)
    if query is None and len({h.query_id for h in anchors}) == 1:
        query = anchors[0].query_id
    anchors = [h for h in anchors if h.query_id == query]
    if (
        comp["molecule"] != "protein"
        and comp["alignment_backend"] in ("hmmalign", "cmalign")
        and "alignment_profile" in comp
    ):
        model = comp["alignment_profile"]
        labels = _profile_labels(model)
        if (
            comp["backend"] == "import"
            and query is not None
            and (
                comp["hit_format"] not in ("nhmmer_tblout", "cmsearch_tblout")
                or (query not in labels and not any(h.query_accession in labels for h in anchors))
            )
        ):
            raise ValueError(
                "nucleotide alignment profile identity differs from the imported search model"
            )
        if comp["backend"] in ("hmmer", "infernal") and checksum(model) != checksum(
            comp["profile"]
        ):
            raise ValueError(
                "nucleotide alignment requires the same model as the validated reference strand"
            )
    if (
        comp["backend"] == "import"
        and comp["hit_format"] == "blast6"
        and query != comp["reference_id"]
        and not anchors
        and by_id
    ):
        raise ValueError(
            "BLAST query differs from the reference; supply reference_hits for that query"
        )
    hit_map = {}
    selected = {}
    binds = {}
    projected = {}
    audit_by_key = {}
    possible = sum(c in CANONICAL[comp["molecule"]] for c in reference)
    source_names = set(by_id)
    align = comp["alignment_backend"]
    for seq_id, seq in read_sequences(comp["database"], comp["molecule"]):
        if (
            seq_id == comp["reference_id"]
            and seq != reference
            and (comp["molecule"] == "protein" or oriented(seq, comp["molecule"], "-") != reference)
        ):
            raise ValueError(f"database record {seq_id!r} conflicts with reference sequence")
        hits = by_id.pop(seq_id, ())
        spans = Counter((h.first, h.last, h.source_strand or h.strand) for h, _ in hits)
        repeated = len(spans) > 1
        for hit, a in sorted(hits, key=lambda item: (-item[0].score, item[0].evalue or 0)):
            rid = _span(seq_id, hit)
            stem = rid if repeated or (wanted is not None and rid in wanted) else seq_id
            if stem != seq_id and stem in source_names:
                stem += "#" + a["domain_id"][:16]
            if stem in hit_map:
                if _digest_hit(hit) == _digest_hit(hit_map[stem]):
                    a.update(status="rejected", reason="duplicate_identical_domain")
                    continue
                stem = stem + "#" + a["domain_id"][:16]
                if stem in hit_map:
                    a.update(status="rejected", reason="duplicate_identical_domain")
                    continue
            if hit.last > len(seq) or (
                hit.target_length is not None and hit.target_length != len(seq)
            ):
                a.update(status="rejected", reason="target_length_or_coordinate_mismatch")
                continue
            frag = seq[hit.first - 1 : hit.last]
            canonical = Counter(x for x in frag if x in CANONICAL[comp["molecule"]])
            total = sum(canonical.values())
            entropy = (
                -sum((n / total) * math.log2(n / total) for n in canonical.values())
                if total
                else 0.0
            )
            mc = (
                (hit.model_last - hit.model_first + 1) / hit.model_length
                if hit.model_length
                else None
            )
            tc = len(frag) / len(seq)
            a.update(
                target_length=len(seq),
                target_coverage=tc,
                model_coverage=mc,
                domain_entropy_bits=entropy,
                row_id=stem,
            )
            if (
                comp["min_model_coverage"] > 0
                and (mc is None or mc < comp["min_model_coverage"])
                or tc < comp["min_target_coverage"]
                or entropy < comp["min_entropy_bits"]
            ):
                a.update(
                    status="rejected",
                    reason="model_coverage_unavailable"
                    if mc is None and comp["min_model_coverage"] > 0
                    else "below_coverage_or_complexity_threshold",
                )
                continue
            if mc is not None and mc > 1 or not total:
                a.update(status="rejected", reason="invalid_model_interval_or_no_canonical_target")
                continue
            hit = replace(hit, target_length=len(seq))
            a.update(status="retained", alignment_status="not_requested")
            hit_map[stem] = hit
            audit_by_key[stem] = a
            if wanted is not None and stem not in wanted:
                hit_map[stem] = replace(hit, placement=None)
                if comp["min_reference_coverage"] or comp["min_domain_coverage"]:
                    a["alignment_status"] = "not_requested_alignment_qc_unassessed"
                continue
            if align == "projection":
                context = dict(comp)
                if anchors or (comp["backend"] == "import" and comp["hit_format"] != "blast6"):
                    context["profile"] = True
                try:
                    place, method = _place_hit(context, reference, seq, hit, anchors, possible)
                except ValueError as exc:
                    # A nonunique or absent locus cannot enter the synchronized alignment.
                    if not any(
                        s in str(exc)
                        for s in (
                            "ambiguous reference locus",
                            "cannot place",
                            "no reference",
                            "no supported",
                            "no compatible reference locus",
                        )
                    ):
                        raise
                    a.update(
                        status="rejected", reason="unresolved_reference_locus", detail=str(exc)
                    )
                    del hit_map[stem]
                    continue
                ident, cov = _support(reference, place, comp["molecule"], possible)
                placed = replace(
                    hit,
                    placement=place,
                    projection_method=method,
                    strand=place.strand,
                    identity=ident,
                    coverage=cov,
                )
                hit_map[stem] = placed
                projected[stem] = place.row(len(reference))
                a.update(
                    identity=ident,
                    coverage=cov,
                    projection_method=method,
                    alignment_status="aligned",
                    reference_coverage=cov,
                    domain_coverage=round(cov * possible) / total,
                    reference_first=place.first,
                    reference_last=place.last,
                    insertions=len(frag) - sum(c != "-" for c in place.aligned),
                    deletions=projected[stem].count("-"),
                )
            else:
                if jack_rows is not None and align == "jackhmmer_stockholm":
                    token = native_bind[a["domain_id"]]
                    start, end = map(int, token.rsplit("/", 1)[1].split("-"))
                    frag = seq[start - 1 : end]
                    binds[stem] = token
                    a.update(stockholm_source_first=start, stockholm_source_last=end)
                selected[stem] = (hit, oriented(frag, comp["molecule"], hit.strand))
    for values in by_id.values():
        for _, a in values:
            a.update(status="rejected", reason="target_absent_from_named_database")
    if align == "projection":
        msa = assemble_msa(name, comp["molecule"], comp["reference_id"], reference, projected)
        meta = {
            "backend": "projection",
            "discarded_insertions": sum(a.get("insertions", 0) for a in audit),
            "reference_residues": len(reference),
        }
        refrow = msa.ids[0]
    elif not selected and align in ("hmmalign", "cmalign"):
        msa = assemble_msa(name, comp["molecule"], comp["reference_id"], reference, {})
        refrow = msa.ids[0]
        meta = {
            "backend": align,
            "status": "coordinate_seed_only_no_selected_domains",
            "reference_residues": len(reference),
        }
    else:
        best = max(anchors, key=lambda a: a.score, default=None)
        alternatives = [
            h
            for h in anchors
            if best and (h.strand, h.first, h.last) != (best.strand, best.first, best.last)
        ]
        margin = best.score - max(h.score for h in alternatives) if best and alternatives else None
        if (comp["molecule"] != "protein" and best is None) or (margin is not None and margin < 5):
            raise ValueError(
                "profile alignment requires one reference locus and strand with a five-bit lead over competing placements; supply a domain-specific anchor"
            )
        msa, refrow, placed, details, meta = profile_align(
            name, comp, reference, selected, raw, binds, best.strand if best else "+", out, programs
        )
        meta.update(
            reference_anchor_competitors=len(alternatives),
            reference_anchor_bit_margin=margin,
            reference_anchor_first=best.first if best else None,
            reference_anchor_last=best.last if best else None,
        )
        hit_map.update(placed)
        for k, h in placed.items():
            audit_by_key[k].update(
                **details[k],
                reference_first=h.placement.first,
                reference_last=h.placement.last,
                identity=h.identity,
                coverage=h.coverage,
                projection_method=h.projection_method,
                alignment_status="aligned",
            )
    excluded = set()
    for key, a in audit_by_key.items():
        if a["alignment_status"] == "aligned" and (
            a["reference_coverage"] < comp["min_reference_coverage"]
            or a["domain_coverage"] < comp["min_domain_coverage"]
        ):
            a.update(
                status="rejected",
                reason="below_mapped_reference_or_domain_coverage",
                alignment_status="removed_after_alignment_qc",
            )
            excluded.add(key)
            del hit_map[key]
    if excluded:
        msa = msa.take_ids([key for key in msa.ids if key == refrow or key not in excluded])
    fields = dict.fromkeys(
        (
            "line",
            "status",
            "reason",
            "alignment_status",
            "row_id",
            "domain_id",
            "seq_id",
            "first",
            "last",
            "strand",
            "query_id",
            "model_first",
            "model_last",
        )
    )
    for a in audit:
        fields.update(dict.fromkeys(a))
    with (out / "hit_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    if any(checksum(comp[k]) != v for k, v in source_digests.items()):
        raise ValueError("discovery source changed during filtering or alignment")
    manifest = {
        "schema": "triplet.discovery.v2",
        "role": name,
        "molecule": comp["molecule"],
        "backend": comp["backend"],
        "database_sha256": source_digests["database"],
        "input_sha256": source_digests,
        "database_release": comp.get("database_release"),
        "reference_id": comp["reference_id"],
        "reference_sha256": source_digests["reference"],
        "reference_row_id": refrow,
        "alignment": meta,
        "selection_settings": {
            key: comp[key]
            for key in (
                "min_identity",
                "min_coverage",
                "min_model_coverage",
                "min_target_coverage",
                "min_reference_coverage",
                "min_domain_coverage",
                "min_entropy_bits",
                "hit_model_length",
                "hit_query",
                "max_candidates",
                "evalue",
                "domain_evalue",
                "include_evalue",
                "include_domain_evalue",
            )
            if key in comp
        },
        "requested_ids": sorted(wanted) if wanted is not None else None,
        "requested_ids_sha256": _hash(sorted(wanted)) if wanted is not None else None,
        "requested_ids_without_accepted_hit": sorted(wanted - hit_map.keys()) if wanted else [],
        "accepted_domains": len(hit_map),
        "aligned_domains": sum(a.get("alignment_status") == "aligned" for a in audit),
        "reported_rows": len(audit),
        "status_counts": dict(Counter(a["status"] for a in audit)),
        "rejection_reasons": dict(Counter(a["reason"] for a in audit if a["reason"])),
        "source_coordinates": "one-based inclusive; exact target ID and reported strand; stable domain_id binds query and model/source intervals",
        "orthology_evidence": "sequence similarity only; no inferred taxonomy, orthology or genome identity",
        "programs": programs,
        "search_cache_hit": cache_hit,
        "raw_manifest": "raw/manifest.json" if info else None,
        "raw_key": info["key"] if info else None,
        "raw_manifest_sha256": info["manifest_sha256"] if info else None,
        "audit": "hit_audit.tsv",
        "audit_sha256": checksum(out / "hit_audit.tsv"),
    }
    (out / "discovery.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return DiscoveryResult(hit_map, msa, refrow, manifest, programs)
