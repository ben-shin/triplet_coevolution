"""Generate auditable, genome- and role-supported candidate triplets from row identifiers."""

import csv
import hashlib
import itertools
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

from .identifiers import (
    EvidenceRow,
    declared_accessions,
    explicit_accession,
    identifiers,
    lookup_id,
)
from .uniprot import Resolver

COMPONENTS = ("A", "B", "C")
REQUIRED = {"component", "identifier", "protein_id", "genome_id", "orthogroup", "source"}
OPTIONAL = {"taxon_id", "neighborhood_id", "assembly_id", "proteome_id", "replicon_id"}
GROUP_FIELDS = (
    "group_id",
    "genome_id",
    "neighborhood_id",
    "status",
    "reasons",
    "a_candidates",
    "b_candidates",
    "c_candidates",
)
SELECTED_FIELDS = ("group_id", "component", "sequence_id", "genome_id", "neighborhood_id")
AUDIT_FIELDS = (
    "group_id",
    "group_status",
    "group_reasons",
    "selected",
    "component",
    "component_name",
    "sequence_id",
    "present",
    "stockholm_accession",
    "domain_start",
    "domain_end",
    "selectors",
    "lookup_id",
    "protein_id",
    "entry_name",
    "genome_id",
    "assembly_id",
    "proteome_id",
    "replicon_id",
    "taxon_id",
    "neighborhood_id",
    "orthogroups",
    "role_status",
    "source",
    "source_type",
    "cross_references",
    "release",
    "release_date",
    "fetched_at",
    "proteome_release",
    "proteome_release_date",
    "confidence",
    "declared_accessions",
    "accession_links",
    "unlinked_accessions",
    "identity_status",
)


def _database(db):
    db.executescript("""
        CREATE TABLE manifest (comp TEXT, identifier TEXT, mapping TEXT);
        CREATE INDEX manifest_key ON manifest(comp,identifier);
        CREATE TABLE rows (id INTEGER PRIMARY KEY, comp TEXT, name TEXT, accession TEXT,
            start INTEGER, end INTEGER, present INTEGER, selectors TEXT, anchor INTEGER,
            declared TEXT, identity_status TEXT DEFAULT '',
            UNIQUE(comp,name,anchor));
        CREATE TABLE mappings (row_key INTEGER, protein TEXT, genome TEXT, locus TEXT,
            taxon TEXT, groups_json TEXT, raw TEXT, lookup TEXT, status TEXT);
        CREATE INDEX mappings_key ON mappings(row_key,protein,genome,locus);
        CREATE TABLE assignments (row_key INTEGER, protein TEXT, genome TEXT, locus TEXT,
            groups_json TEXT, replicons_json TEXT, PRIMARY KEY(row_key,protein,genome,locus));
        CREATE TABLE candidates (row_key INTEGER, comp TEXT, protein TEXT, genome TEXT, locus TEXT,
            group_id TEXT, present INTEGER, reason TEXT, PRIMARY KEY(row_key,protein,genome,locus));
        CREATE INDEX candidates_group ON candidates(group_id,comp);
        CREATE INDEX candidates_genome ON candidates(genome,locus);
        CREATE TABLE groups (group_id TEXT PRIMARY KEY, genome TEXT, locus TEXT, status TEXT,
            reasons TEXT, a INTEGER, b INTEGER, c INTEGER);
    """)


def _load_manifest(db, path):
    if path is None:
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        fields = [v.strip() for v in reader.fieldnames or []]
        if (
            len(set(fields)) != len(fields)
            or not REQUIRED <= set(fields)
            or set(fields) - REQUIRED - OPTIONAL
        ):
            raise ValueError(
                f"invalid curated evidence header in {path}; required columns: {', '.join(sorted(REQUIRED))}; optional: {', '.join(sorted(OPTIONAL))}"
            )
        reader.fieldnames = fields
        for row in reader:
            if None in row or any(
                v is None or v.strip() != v or any(ord(ch) < 32 for ch in v) for v in row.values()
            ):
                raise ValueError(f"malformed curated evidence in {path} at line {reader.line_num}")
            identifier = row.pop("identifier")
            comp = row.pop("component")
            if (
                comp not in COMPONENTS
                or not identifier
                or any(c.isspace() for c in identifier)
                or not row["source"]
            ):
                raise ValueError(
                    f"invalid component, identifier or source in {path} at line {reader.line_num}"
                )
            group = row.pop("orthogroup")
            row["orthogroups"] = [group] if group else []
            row["source_type"] = "curated"
            # A taxon names a population of organisms, not a particular assembly or proteome.
            stated_genome = row["genome_id"]
            if stated_genome and (
                stated_genome == row.get("taxon_id")
                or re.fullmatch(
                    r"(?:NCBI(?:Taxon(?:omy)?)?|Tax(?:on|id))[:_]\d+", stated_genome, re.IGNORECASE
                )
            ):
                row["genome_id"] = ""
            row["genome_id"] = (
                row["genome_id"] or row.get("assembly_id", "") or row.get("proteome_id", "")
            )
            db.execute(
                "INSERT INTO manifest VALUES (?,?,?)",
                (comp, identifier, json.dumps(row, sort_keys=True)),
            )
    with path.open("rb") as stream:
        return {"name": path.name, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def _curated(db, comp, row_id, aliases, declared):
    placeholders = ",".join("?" for _ in aliases)
    records = [
        (lookup, json.loads(raw), "")
        for lookup, raw in db.execute(
            f"SELECT DISTINCT identifier,mapping FROM manifest WHERE comp=? AND identifier IN ({placeholders}) ORDER BY identifier,mapping",
            (comp, *aliases),
        )
    ]
    if "/" in row_id and (specific := [r for r in records if r[0] == row_id]):
        # Domain-specific features take precedence. A second explicit accession
        # may still need separately sourced corroboration of the underlying molecule.
        verification = [
            r for r in records if r[0] != row_id and explicit_accession(r[0]) in declared
        ]
        return specific, verification
    return records, []


def _append(db, key, lookup, mapping, status=""):
    db.execute(
        "INSERT INTO mappings VALUES (?,?,?,?,?,?,?,?,?)",
        (
            key,
            mapping.get("protein_id", ""),
            mapping.get("genome_id", ""),
            mapping.get("neighborhood_id", ""),
            mapping.get("taxon_id", ""),
            json.dumps(mapping.get("orthogroups", [])),
            json.dumps(mapping, sort_keys=True),
            lookup,
            status,
        ),
    )


def _lookup_facts(queries, results, resolver):
    for query in queries:
        if query not in results:
            yield query, {}, "cache_miss" if resolver.cache_name else "no_curated_mapping"
            continue
        payload = results[query]
        stamp = {k: payload.get(k, "") for k in ("release", "release_date", "fetched_at")}
        if not payload["mappings"]:
            yield query, {"source": "UniProtKB", **stamp}, "uniprot_not_found"
        for mapping in payload["mappings"]:
            yield query, {**stamp, **mapping}, ""


def _accession_links(declared, row_id, facts):
    """Proof is specific to an accession and a destination molecule, never a shared selector."""
    by_protein = {}
    for lookup, mapping, status in facts:
        protein = mapping.get("protein_id", "")
        if status or not protein:
            continue
        links = by_protein.setdefault(protein, {})
        source = mapping.get("source", "")
        curated = mapping.get("source_type") == "curated"
        for ac, origins in declared.items():
            proof = ""
            if explicit_accession(protein) == ac:
                proof = "matching_primary"
            elif curated and explicit_accession(lookup) == ac:
                proof = "curated_accession"
            elif not curated and lookup == ac and "-" not in ac:
                # The resolver associates a different current accession only after an
                # accession-specific secondary-ID lookup, including in older snapshots.
                proof = "verified_secondary"
            elif (
                curated
                and lookup == row_id
                and not lookup_id(row_id)
                and not explicit_accession(row_id)
                and "stockholm_accession" in origins
            ):
                # A curator explicitly identified this domain or opaque alignment row.
                # A free-form coordinate-reference label cannot override another biological accession.
                proof = "curated_alignment_row"
            if proof:
                witness = {"proof": proof, "lookup": lookup, "source": source}
                if witness not in links.setdefault(ac, []):
                    links[ac].append(witness)
    return by_protein


def _save_facts(db, key, row_id, declared, facts, verification):
    links = _accession_links(declared, row_id, [*facts, *verification])
    proteins = {m["protein_id"] for _, m, status in facts if not status and m.get("protein_id")}
    missing = {p: sorted(declared.keys() - links.get(p, {}).keys()) for p in proteins}
    issue = ""
    verified = {
        m["protein_id"] for _, m, status in verification if not status and m.get("protein_id")
    }
    if len(proteins) > 1 or verified - proteins:
        issue = "multiple_mappings"
    elif any(missing.values()):
        issue = "unlinked_accession"
    if issue:
        db.execute("UPDATE rows SET identity_status=? WHERE id=?", (issue, key))
    for lookup, mapping, status in facts:
        protein = mapping.get("protein_id", "")
        if protein and declared:
            mapping = {
                **mapping,
                "accession_links": links.get(protein, {}),
                "unlinked_accessions": missing.get(protein, []),
            }
            if missing.get(protein):
                status = "unlinked_accession"
        _append(db, key, lookup, mapping, status)
    for lookup, mapping, status in verification:
        _append(db, key, lookup, mapping, status or "accession_verification_only")


def _collect(db, comp, records, resolver, allow_uniprot, *, anchor=False, reference_id=None):
    records = iter(records)
    while chunk := tuple(itertools.islice(records, 128)):
        pending = []
        for row in chunk:
            if not row.row_id or any(c.isspace() for c in row.row_id):
                raise ValueError(f"empty or invalid {comp} alignment identifier {row.row_id!r}")
            aliases, remote, start, end = identifiers(row)
            declared = declared_accessions(row, reference_id)
            aliases = tuple(dict.fromkeys((*aliases, *declared)))
            # Include a pipe header's embedded mnemonic as a cross-check. Its lookup
            # can reveal an obsolete or conflicting name but cannot override the accession.
            remote = {token for alias in aliases if (token := lookup_id(alias))} | set(remote)
            try:
                key = db.execute(
                    "INSERT INTO rows VALUES (NULL,?,?,?,?,?,?,?,?,?,'')",
                    (
                        comp,
                        row.row_id,
                        row.accession,
                        start,
                        end,
                        bool(getattr(row, "present", True)),
                        json.dumps(sorted(set(getattr(row, "selectors", ())))),
                        anchor,
                        json.dumps(declared, sort_keys=True),
                    ),
                ).lastrowid
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate {comp} alignment identifier {row.row_id!r}") from exc
            curated, verification = _curated(db, comp, row.row_id, aliases, declared)
            if curated:
                # Preserve curated feature assignments. Allow exact accession-to-molecule
                # verification for aliases without an independently documented link.
                needed = set()
                if allow_uniprot and declared and (resolver.live or resolver.cache_name):
                    links = _accession_links(declared, row.row_id, [*curated, *verification])
                    for protein in {m["protein_id"] for _, m, _ in curated if m.get("protein_id")}:
                        needed.update(declared.keys() - links.get(protein, {}).keys())
                pending.append((key, row.row_id, declared, curated, verification, sorted(needed)))
            elif (remote or declared) and allow_uniprot:
                pending.append(
                    (key, row.row_id, declared, [], [], sorted(set(remote) | declared.keys()))
                )
            else:
                reason = "no_curated_mapping" if not allow_uniprot else "unrecognized_identifier"
                _append(db, key, row.row_id, {}, reason)
        results = resolver.resolve(query for *_, queries in pending for query in queries)
        for key, row_id, declared, curated, verification, queries in pending:
            looked_up = list(_lookup_facts(queries, results, resolver))
            _save_facts(
                db,
                key,
                row_id,
                declared,
                curated or looked_up,
                [*verification, *looked_up] if curated else [],
            )


def _reconcile_intervals(db):
    """Combine claims on a known oriented interval; retain every distinct alignment row."""
    facts = db.execute("""
        SELECT r.comp,m.protein,m.genome,m.locus,r.start,r.end,r.identity_status,r.id,m.groups_json,
               json_extract(m.raw,'$.replicon_id') FROM mappings m JOIN rows r ON r.id=m.row_key
        WHERE m.status='' ORDER BY r.comp,m.protein,m.genome,m.locus,r.start,r.end,r.identity_status,r.id
    """)

    def interval(fact):
        comp, protein, genome, locus, start, end, issue, key = fact[:8]
        # Unbound local identifiers and unknown spans establish no equivalence between rows.
        return (
            comp,
            protein,
            genome,
            locus,
            start,
            end,
            None if protein and genome and start and not issue else key,
        )

    for placement, records in itertools.groupby(facts, key=interval):
        _, protein, genome, locus, *_ = placement
        keys, groups, replicons = set(), set(), set()
        for *_, row_key, memberships, replicon in records:
            keys.add(row_key)
            groups.update(json.loads(memberships))
            if replicon:
                replicons.add(replicon)
        group_values, replicon_values = json.dumps(sorted(groups)), json.dumps(sorted(replicons))
        db.executemany(
            "INSERT INTO assignments VALUES (?,?,?,?,?,?)",
            ((key, protein, genome, locus, group_values, replicon_values) for key in keys),
        )


def _role_expectations(db, supplied, anchors, scope):
    expected, audit = {}, {}
    for comp in COMPONENTS:
        configured = (supplied or {}).get(comp, ())
        values = tuple(configured)
        if isinstance(configured, str) or not all(isinstance(v, str) and v for v in values):
            raise ValueError(f"expected_orthogroups[{comp}] must be a list of nonempty strings")
        groups = set(values)
        ref = (anchors or {}).get(comp)
        decision = "configured" if groups else "no_role_anchor"
        evidence = []
        memberships = set()
        if ref and not groups:
            proteins = set()
            declared, identity = db.execute(
                "SELECT declared,identity_status FROM rows WHERE anchor=1 AND comp=?", (comp,)
            ).fetchone()
            for protein, group_json in db.execute(
                "SELECT protein,groups_json FROM assignments JOIN rows ON rows.id=row_key WHERE anchor=1 AND comp=? AND identity_status=''",
                (comp,),
            ):
                proteins.add(protein)
                memberships.update(json.loads(group_json))
            for protein, group_json, source, lookup, status in db.execute(
                "SELECT protein,groups_json,raw,lookup,status FROM mappings JOIN rows ON rows.id=row_key WHERE anchor=1 AND comp=? ORDER BY protein,lookup,raw",
                (comp,),
            ):
                ogs = json.loads(group_json)
                detail = json.loads(source)
                evidence.append(
                    {
                        "lookup_id": lookup,
                        "protein_id": protein,
                        "orthogroups": ogs,
                        "source": detail.get("source", ""),
                        "release": detail.get("release", ""),
                        "lookup_status": status or "mapped",
                        "replicon_id": detail.get("replicon_id", ""),
                        "accession_links": detail.get("accession_links", {}),
                        "unlinked_accessions": detail.get("unlinked_accessions", []),
                    }
                )
            eligible = {
                v
                for v in memberships
                if not scope or re.fullmatch(r"OrthoDB:\d+at" + re.escape(scope), v)
            }
            if identity:
                decision = (
                    "unlinked_reference_accession"
                    if identity == "unlinked_accession"
                    else "ambiguous_reference_orthology"
                )
            elif len(proteins) == 1 and "" not in proteins and len(eligible) == 1:
                groups, decision = eligible, "reference_orthogroup"
            elif len(proteins) > 1 or len(eligible) > 1:
                decision = "ambiguous_reference_orthology"
            else:
                decision = "missing_reference_orthology"
        expected[comp] = groups
        audit[comp] = {
            "reference": ref,
            "source": decision,
            "orthogroups": sorted(groups),
            "anchor_evidence": evidence,
        }
        if ref and not values:
            audit[comp]["declared_accessions"] = json.loads(declared)
            audit[comp]["observed_anchor_orthogroups"] = sorted(memberships)
    return expected, audit


def _namespace(group):
    if match := re.fullmatch(r"OrthoDB:\d+at(\d+)", group):
        return ("OrthoDB", match[1])
    return (group.partition(":")[0] if ":" in group else "", "")


def _role_status(found, expected):
    if not expected:
        return "no_role_anchor"
    if not found:
        return "missing_orthology"
    namespaces = {_namespace(v) for v in expected}
    comparable = {g for g in found if _namespace(g) in namespaces}
    if found & expected:
        return "orthology_conflict" if comparable - expected else "supported"
    return (
        "orthogroup_mismatch"
        if namespaces <= {_namespace(g) for g in comparable}
        else "orthology_not_comparable"
    )


def _group_id(genome, locus):
    return quote(genome, safe="") + ("@" + quote(locus, safe="") if locus else "") if genome else ""


def _classify(db, expected):
    distinguishing = {
        c: expected[c] - set().union(*(expected[other] for other in COMPONENTS if other != c))
        for c in COMPONENTS
    }
    for (
        key,
        comp,
        present,
        protein,
        genome,
        locus,
        raw_groups,
        raw_replicons,
        identity,
        n,
    ) in db.execute(
        """SELECT row_key,comp,present,protein,genome,locus,groups_json,replicons_json,identity_status,
            COUNT(*) OVER (PARTITION BY row_key) FROM assignments JOIN rows ON rows.id=row_key
            WHERE anchor=0"""
    ):
        groups = set(json.loads(raw_groups))
        reason = _role_status(groups, expected[comp])
        if reason == "supported" and not groups & distinguishing[comp]:
            reason = "overlapping_role_orthology"
        if n > 1:
            reason = "multiple_mappings"
        elif reason != "orthogroup_mismatch" and len(json.loads(raw_replicons)) > 1:
            reason = "multiple_replicon_annotations"
        elif not protein:
            reason = "missing_molecule_id"
        elif not genome:
            reason = "missing_genome"
        db.execute(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)",
            (
                key,
                comp,
                protein,
                genome,
                locus,
                _group_id(genome, locus),
                present,
                identity or reason,
            ),
        )
    # A stable molecule attributed to distinct biological placements under two row aliases
    # remains uncertain even if each alias individually had a single manifest entry.
    db.execute("""CREATE TEMP TABLE conflicts AS SELECT c.comp,protein FROM candidates c JOIN rows r ON r.id=c.row_key
        WHERE protein!='' AND r.identity_status=''
        GROUP BY c.comp,protein HAVING COUNT(DISTINCT group_id)>1""")
    db.execute("CREATE INDEX conflicts_key ON conflicts(comp,protein)")
    db.execute("""UPDATE candidates SET reason='multiple_mappings' WHERE EXISTS
        (SELECT 1 FROM conflicts WHERE conflicts.comp=candidates.comp AND conflicts.protein=candidates.protein)""")


def _replicon_label(raw):
    label = " ".join(raw.casefold().split())
    if label in {
        "",
        "-",
        ".",
        "?",
        "na",
        "n/a",
        "none",
        "not specified",
        "not available",
        "not assigned",
    } or re.search(
        r"\b(?:unknown|unplaced|unlocalized|unlocalised|unassembled|unassigned|unspecified|undetermined)\b",
        label,
    ):
        return None
    named = re.fullmatch(
        r"(chromosome|chr|plasmid|scaffold|contig)[\s:_-]*(.*)", raw.strip(), re.IGNORECASE
    )
    if named:
        kind, name = named.groups()
        kind = kind.lower()
        if name.lower() in {"un", "unknown", "unplaced", "unlocalized"} or (
            kind in {"chr", "chromosome"} and name.lower().startswith("un_")
        ):
            return None
        if kind in {"chr", "chromosome"}:
            roman = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix")
            if name.lower() in roman:
                name = str(roman.index(name.lower()) + 1)
            return "chromosome", name
        return kind, name
    return "local", raw


def _neighborhood_replicon_reasons(labels):
    reasons = set()
    for (kind_a, name_a), (kind_b, name_b) in itertools.combinations(labels, 2):
        if (
            {kind_a, kind_b} == {"chromosome", "plasmid"}
            or kind_a == kind_b
            and name_a
            and name_b
            and name_a != name_b
        ):
            reasons.add("conflicting_neighborhood_replicons")
        elif kind_a != kind_b:
            reasons.add("unverified_neighborhood_replicons")
    return reasons


def _summarize_groups(db):
    db.execute("""CREATE TEMP TABLE placements AS
        SELECT c.group_id AS gid,COUNT(DISTINCT NULLIF(m.taxon,'')) AS taxa,
        COUNT(DISTINCT NULLIF(json_extract(m.raw,'$.assembly_id'),'')) AS assemblies
        FROM candidates c JOIN mappings m ON m.row_key=c.row_key AND m.protein=c.protein
        AND m.genome=c.genome AND m.locus=c.locus WHERE c.reason!='orthogroup_mismatch' AND m.status=''
        GROUP BY c.group_id""")
    db.execute("CREATE INDEX placements_key ON placements(gid)")
    for gid, genome, locus in db.execute(
        "SELECT DISTINCT group_id,genome,locus FROM candidates WHERE group_id!='' ORDER BY group_id"
    ):
        counts, reasons = dict.fromkeys(COMPONENTS, 0), set()
        unique, all_ready = set(), True
        for comp, n, available, unresolved, distinct_proteins, protein in db.execute(
            """
            SELECT comp,COUNT(DISTINCT row_key),COUNT(DISTINCT CASE WHEN present THEN row_key END),
            MAX(reason!='supported'),COUNT(DISTINCT protein),MIN(protein) FROM candidates
            WHERE group_id=? AND reason!='orthogroup_mismatch' GROUP BY comp""",
            (gid,),
        ):
            counts[comp] = n
            if n > 1 or distinct_proteins > 1:
                reasons.add(f"{comp}:multiple_candidates")
            if not available:
                reasons.add(f"{comp}:absent_from_alignment")
            if unresolved:
                codes = db.execute(
                    "SELECT DISTINCT reason FROM candidates WHERE group_id=? AND comp=? AND reason NOT IN ('supported','orthogroup_mismatch')",
                    (gid, comp),
                )
                reasons.update(f"{comp}:{code}" for (code,) in codes)
            if n != 1 or available != 1 or unresolved:
                all_ready = False
            if protein:
                unique.add(protein)
        for comp in COMPONENTS:
            if not counts[comp]:
                reasons.add(f"{comp}:missing_candidate")
        if all_ready and all(counts.values()) and len(unique) != 3:
            reasons.add("shared_molecule_across_roles")
        if locus:
            labels = set()
            for comp, raw in db.execute(
                """SELECT c.comp,a.replicons_json FROM candidates c
                JOIN assignments a USING(row_key,protein,genome,locus)
                WHERE c.group_id=? AND c.present AND c.reason='supported'""",
                (gid,),
            ):
                if counts[comp] == 1:
                    labels.update(label for r in json.loads(raw) if (label := _replicon_label(r)))
            reasons.update(_neighborhood_replicon_reasons(labels))
        placement = db.execute(
            "SELECT taxa,assemblies FROM placements WHERE gid=?", (gid,)
        ).fetchone()
        if placement:
            if placement[0] > 1:
                reasons.add("conflicting_taxa")
            if placement[1] > 1:
                reasons.add("conflicting_assemblies")
        if (
            locus
            and db.execute(
                "SELECT 1 FROM candidates WHERE genome=? AND locus='' AND reason!='orthogroup_mismatch' LIMIT 1",
                (genome,),
            ).fetchone()
        ):
            reasons.add("unlocalized_genome_candidates")
        if (
            not locus
            and db.execute(
                "SELECT 1 FROM candidates WHERE genome=? AND locus!='' AND reason!='orthogroup_mismatch' LIMIT 1",
                (genome,),
            ).fetchone()
        ):
            reasons.add("unlocalized_genome_candidates")
        ambiguous = {
            "multiple_candidates",
            "multiple_mappings",
            "multiple_replicon_annotations",
            "orthology_conflict",
            "overlapping_role_orthology",
            "conflicting_taxa",
            "conflicting_assemblies",
            "conflicting_neighborhood_replicons",
            "unverified_neighborhood_replicons",
            "shared_molecule_across_roles",
            "unlocalized_genome_candidates",
        }
        status = (
            "ambiguous"
            if any(r.split(":")[-1] in ambiguous for r in reasons)
            else "incomplete"
            if reasons
            else "resolved"
        )
        db.execute(
            "INSERT INTO groups VALUES (?,?,?,?,?,?,?,?)",
            (gid, genome, locus, status, ";".join(sorted(reasons)), *counts.values()),
        )


def _write_table(path, header, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _audit_rows(db, names):
    query = """SELECT c.group_id,g.status,g.reasons,r.comp,r.name,r.present,r.accession,r.start,r.end,
        r.selectors,m.lookup,m.raw,m.status,c.reason,r.declared,r.identity_status FROM mappings m JOIN rows r ON m.row_key=r.id
        LEFT JOIN candidates c ON m.status='' AND c.row_key=r.id AND c.protein=m.protein AND c.genome=m.genome AND c.locus=m.locus
        LEFT JOIN groups g ON c.group_id=g.group_id WHERE r.anchor=0
        ORDER BY r.comp,r.name,c.group_id,c.protein,m.lookup,m.raw"""
    for (
        gid,
        gstatus,
        greasons,
        comp,
        row,
        present,
        ac,
        start,
        end,
        selectors,
        lookup,
        raw,
        mstatus,
        role,
        declared,
        issue,
    ) in db.execute(query):
        data = json.loads(raw)
        status = (
            "multiple_mappings"
            if issue == "multiple_mappings" and mstatus == "unlinked_accession"
            else mstatus or role
        )
        confidence = "uncertain"
        if role == "orthogroup_mismatch":
            confidence = "role_excluded"
        elif gstatus == "resolved" and role == "supported":
            confidence = (
                "neighborhood_and_role_supported"
                if data.get("neighborhood_id")
                else "genome_and_role_supported"
            )
        yield (
            gid,
            gstatus or "unplaced",
            greasons or "",
            int(gstatus == "resolved" and role == "supported"),
            comp,
            names.get(comp, comp),
            row,
            present,
            ac,
            start,
            end,
            selectors,
            lookup,
            *[
                data.get(k, "")
                for k in (
                    "protein_id",
                    "entry_name",
                    "genome_id",
                    "assembly_id",
                    "proteome_id",
                    "replicon_id",
                    "taxon_id",
                    "neighborhood_id",
                )
            ],
            json.dumps(data.get("orthogroups", [])),
            status,
            data.get("source", ""),
            data.get("source_type", "uniprot" if data.get("source") == "UniProtKB" else ""),
            json.dumps(data.get("cross_references", []), sort_keys=True),
            *[
                data.get(k, "")
                for k in (
                    "release",
                    "release_date",
                    "fetched_at",
                    "proteome_release",
                    "proteome_release_date",
                )
            ],
            confidence,
            declared,
            json.dumps(data.get("accession_links", {}), sort_keys=True),
            json.dumps(data.get("unlinked_accessions", [])),
            mstatus
            if mstatus == "unlinked_accession"
            else issue or ("mapped" if not mstatus else mstatus),
        )


def generate_evidence(
    rows,
    out_dir: Path,
    *,
    manifest: Path | None = None,
    uniprot: bool = False,
    cache: Path | None = None,
    expected_orthogroups=None,
    uniprot_components=COMPONENTS,
    component_names=None,
    reference_anchors=None,
    reference_rows=None,
    orthology_scope: str | None = None,
):
    """Write into the caller's staging metadata directory; return evidence_summary.json content."""
    if set(rows) != set(COMPONENTS):
        raise ValueError("expected evidence inventory for components A, B and C")
    if orthology_scope is not None and not re.fullmatch(r"\d+", orthology_scope):
        raise ValueError("orthology_scope must be an OrthoDB numeric taxonomy ID string")
    names = component_names or {}
    protein_components = set(uniprot_components)
    if (
        set(names)
        | set(expected_orthogroups or {})
        | set(reference_anchors or {})
        | set(reference_rows or {})
        | protein_components
    ) - set(COMPONENTS):
        raise ValueError("unknown evidence component; expected A, B or C")
    out_dir.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect("")) as db, closing(Resolver(cache, uniprot)) as resolver:
        _database(db)
        identity = _load_manifest(db, manifest)
        for comp in COMPONENTS:
            _collect(db, comp, rows[comp], resolver, comp in protein_components)
            if (ref := (reference_anchors or {}).get(comp)) and not (
                expected_orthogroups or {}
            ).get(comp):
                occurrence = db.execute(
                    "SELECT accession,start,end,selectors FROM rows WHERE comp=? AND name=?",
                    (comp, ref),
                ).fetchone()
                anchor = (
                    EvidenceRow(
                        ref,
                        occurrence[0],
                        occurrence[1],
                        occurrence[2],
                        selectors=tuple(json.loads(occurrence[3])),
                    )
                    if occurrence
                    else EvidenceRow(ref)
                )
                anchor = (reference_rows or {}).get(comp, anchor)
                _collect(
                    db,
                    comp,
                    [anchor],
                    resolver,
                    comp in protein_components,
                    anchor=True,
                    reference_id=ref,
                )
        _reconcile_intervals(db)
        expected, decisions = _role_expectations(
            db, expected_orthogroups, reference_anchors, orthology_scope
        )
        for comp, name in db.execute("SELECT comp,name FROM rows WHERE anchor=1"):
            decisions[comp]["reference_row_id"] = name
        _classify(db, expected)
        _summarize_groups(db)
        _write_table(
            out_dir / "groups.tsv",
            GROUP_FIELDS,
            db.execute("SELECT * FROM groups ORDER BY group_id"),
        )
        _write_table(out_dir / "evidence.tsv", AUDIT_FIELDS, _audit_rows(db, names))
        _write_table(
            out_dir / "selected_evidence.tsv",
            SELECTED_FIELDS,
            db.execute("""
            SELECT g.group_id,r.comp,r.name,g.genome,g.locus FROM groups g JOIN candidates c USING(group_id)
            JOIN rows r ON r.id=c.row_key WHERE g.status='resolved' AND c.reason='supported'
            ORDER BY g.group_id,r.comp"""),
        )
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM groups GROUP BY status"))
        resolved = statuses.get("resolved", 0)
        summary = {
            "schema_version": 1,
            "counts": {
                "input_rows": db.execute("SELECT COUNT(*) FROM rows WHERE anchor=0").fetchone()[0],
                "groups": sum(statuses.values()),
                "resolved_groups": resolved,
                "selected_rows": 3 * resolved,
                "audit_rows": db.execute(
                    "SELECT COUNT(*) FROM mappings JOIN rows ON id=row_key WHERE anchor=0"
                ).fetchone()[0],
            },
            "group_status_counts": statuses,
            "manifest": identity,
            "component_names": {c: names.get(c, c) for c in COMPONENTS},
            "roles": decisions,
            "orthology_scope": orthology_scope,
            "uniprot_components": sorted(protein_components),
            "resolver": {
                **resolver.provenance(),
                "precedence": "exact curated domain, then curated aliases, then UniProt for enabled molecules; declared accessions require independent molecule links",
            },
            "tables": ["evidence.tsv", "selected_evidence.tsv", "groups.tsv"],
        }
        if cache and cache.is_file():
            with cache.open("rb") as stream:
                summary["resolver"]["cache_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        with (out_dir / "evidence_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary
