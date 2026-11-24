"""Generate auditable, genome- and role-supported candidate triplets from row identifiers."""

import csv
import hashlib
import itertools
import json
import re
import sqlite3
from contextlib import closing, nullcontext
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from .gff3 import load_annotations
from .identifiers import (
    EvidenceRow,
    declared_accessions,
    explicit_accession,
    identifiers,
    lookup_id,
)
from .nucleotide import ASSEMBLY, ENA, NCBI, RNACENTRAL, SUMMARY, NucleotideResolver, nucleotide_key
from .provider_policy import current, request_context
from .stockholm import local_placement, nucleotide_facts
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
    "molecule_type",
    "nucleotide_accession",
    "genomic_start",
    "genomic_end",
    "genomic_strand",
    "placement_source",
    "organism_name",
    "provider_error",
    "assembly_aliases",
    "stockholm_annotations",
)


def _database(db):
    db.executescript("""
        CREATE TABLE manifest (comp TEXT, identifier TEXT, mapping TEXT);
        CREATE INDEX manifest_key ON manifest(comp,identifier);
        CREATE TABLE row_tags (row_key INTEGER PRIMARY KEY, tags TEXT);
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
            group_id TEXT, present INTEGER, reason TEXT, incomplete INTEGER,
            PRIMARY KEY(row_key,protein,genome,locus));
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
        if "provider_error" in payload:
            yield (
                query,
                {"source": "UniProtKB", "provider_error": payload["provider_error"]},
                "provider_unavailable",
            )
            continue
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
            elif mapping.get("source_type") == "gff3" and ac in mapping.get("gff3", {}).get(
                "annotation_ids", ()
            ):
                proof = "gff3_explicit_feature_accession"
            elif curated and explicit_accession(lookup) == ac:
                proof = "curated_accession"
            elif (
                source == "Stockholm"
                and mapping.get("source_type") == "stockholm"
                and "LOCUS" in {tag.upper() for tag, _ in mapping.get("stockholm_annotations", ())}
                and len(
                    [
                        a
                        for a, fields in declared.items()
                        if {"row_id", "stockholm_accession"} & set(fields)
                    ]
                )
                == 1
                and {"row_id", "stockholm_accession"} & set(origins)
            ):
                proof = "stockholm_locus"
            elif source == "UniProtKB" and not curated and lookup == ac and "-" not in ac:
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
        _append(
            db,
            key,
            lookup,
            {**mapping, "verification_only": True},
            status or "accession_verification_only",
        )


def _collect(
    db,
    comp,
    records,
    resolver,
    allow_uniprot,
    *,
    anchor=False,
    reference_id=None,
    molecule=None,
    family=None,
    nucleotides=None,
):
    records = iter(records)
    while chunk := tuple(itertools.islice(records, 128)):
        pending = []
        nuc_pending = []
        annotations = {}
        for row in chunk:
            if not row.row_id or any(c.isspace() for c in row.row_id):
                raise ValueError(f"empty or invalid {comp} alignment identifier {row.row_id!r}")
            aliases, remote, start, end = identifiers(row)
            if nucleotides and row.start is None and start is not None:
                row = EvidenceRow(
                    row.row_id,
                    row.accession,
                    start,
                    end,
                    getattr(row, "present", True),
                    getattr(row, "selectors", ()),
                    getattr(row, "annotations", ()),
                )
            declared = (
                {}
                if nucleotides and molecule in {"rna", "dna"}
                else declared_accessions(row, reference_id)
            )
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
            if nucleotides:
                nuc, embedded, meta = local_placement(row, molecule, family)
                ref_key = (
                    nucleotide_key(
                        reference_id, allow_ambiguous=bool(family and family.startswith("RF"))
                    )
                    if molecule in {"dna", "rna"}
                    else ""
                )
                if (
                    ref_key
                    and nuc
                    and ref_key != nuc
                    and not (
                        ("." not in ref_key and ref_key == nuc.split(".")[0])
                        or (ref_key.startswith("URS") and ref_key == nuc.split("_")[0])
                    )
                ):
                    embedded = [(row.row_id, meta, "conflicting_reference_nucleotide_identifier")]
                annotations[key] = meta
                if getattr(row, "annotations", ()):
                    db.execute(
                        "INSERT INTO row_tags VALUES (?,?)", (key, json.dumps(row.annotations))
                    )
                if not curated and embedded:
                    _save_facts(db, key, row.row_id, declared, embedded, [])
                    continue
                if not curated and nuc:
                    nuc_pending.append((key, row, nuc, meta))
                    continue
                if not curated and not allow_uniprot:
                    status = (
                        "missing_nucleotide_identifier"
                        if molecule in {"dna", "rna"}
                        else "unknown_molecule"
                    )
                    _append(db, key, row.row_id, meta, status)
                    continue
                if allow_uniprot:
                    # Send only confirmed public identifier syntax from the actual record.
                    # A local row name or arbitrary user-supplied selector is not a query.
                    tokens = []
                    for source in (row.row_id, row.accession):
                        if not source:
                            continue
                        original = re.sub(r"/\d+-\d+$", "", source)
                        parts = original.split("|")
                        tokens.extend(
                            parts[1:]
                            if len(parts) == 3 and parts[0].lower() in {"sp", "tr"}
                            else [original]
                        )
                    remote = {
                        lookup_id(a)
                        for a in tokens
                        if explicit_accession(a) or re.fullmatch(r"[A-Z0-9]{1,16}_[A-Z0-9]{3,5}", a)
                    }
                    remote.discard("")
                    if any(nucleotide_key(a) for a in (row.row_id, row.accession)):
                        remote = set()
                        declared = {}
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
                _append(db, key, row.row_id, annotations.get(key, {}), reason)
        if nuc_pending:
            results = nucleotides.resolve(nuc for _, _, nuc, _ in nuc_pending)
            for key, row, nuc, meta in nuc_pending:
                _save_facts(
                    db, key, row.row_id, {}, nucleotide_facts(row, nuc, results[nuc], meta), []
                )
        results = resolver.resolve(query for *_, queries in pending for query in queries)
        for key, row_id, declared, curated, verification, queries in pending:
            looked_up = list(_lookup_facts(queries, results, resolver))
            meta = annotations.get(key, {})
            facts = []
            for token, mapping, status in curated or looked_up:
                if (
                    not status
                    and meta.get("taxon_id")
                    and mapping.get("taxon_id")
                    and meta["taxon_id"] != mapping["taxon_id"]
                ):
                    status = "conflicting_stockholm_taxa"
                if mapping.get("source_type") == "gff3" and mapping["gff3"]["limitations"]:
                    status = status or mapping["gff3"]["limitations"][0]
                mapped = {**meta, **mapping}
                if not curated and meta:
                    if not mapped.get("taxon_id"):
                        mapped["taxon_id"] = meta.get("taxon_id", "")
                    if mapping.get("source") == "UniProtKB":
                        mapped.update(
                            {
                                k: mapping.get(k, "")
                                for k in (
                                    "genome_id",
                                    "assembly_id",
                                    "proteome_id",
                                    "neighborhood_id",
                                    "replicon_id",
                                )
                            }
                        )
                        mapped["provider_genome_id"] = mapping.get("genome_id", "")
                if mapping.get("source") == "UniProtKB" and mapping.get("source_type") != "curated":
                    mapped.update(source_type="uniprot", placement_source="UniProtKB Proteomes")
                if not curated and not status and meta.get("orthogroups"):
                    mapped["orthogroups"] = sorted(
                        set(meta["orthogroups"]) | set(mapped.get("orthogroups", ()))
                    )
                facts.append((token, mapped, status))
            _save_facts(
                db,
                key,
                row_id,
                declared,
                facts,
                [*verification, *looked_up] if curated else [],
            )
        if not anchor and (ctx := current()):
            ctx.advance(len(chunk))


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


def _automatic_roles(db, expected, decisions, profiles, scope):
    """Learn distinguishable memberships from independent genome placements before classifying."""
    for comp, profile in profiles.items():
        if (
            comp not in COMPONENTS
            or profile is not None
            and not re.fullmatch(r"(PF|RF)\d{5}", profile)
        ):
            raise ValueError(f"invalid automatic family profile {comp!r}: {profile!r}")
    db.create_function(
        "orthology_namespace", 1, lambda g: json.dumps(_namespace(g)), deterministic=True
    )
    db.executescript("""
        CREATE TEMP TABLE family_groups AS
        WITH placements AS (
            SELECT a.*,r.comp,r.present,r.identity_status,
                   count(*) OVER (PARTITION BY row_key) AS n
            FROM assignments a JOIN rows r ON r.id=a.row_key WHERE r.anchor=0
        ), conflicting_proteins AS (
            SELECT comp,protein FROM placements GROUP BY comp,protein
            HAVING count(DISTINCT genome)>1 OR count(DISTINCT locus)>1
        )
        SELECT DISTINCT p.comp,p.genome,j.value AS membership,orthology_namespace(j.value) AS ns
        FROM placements p, json_each(p.groups_json) j
        LEFT JOIN conflicting_proteins x ON x.comp=p.comp AND x.protein=p.protein
        WHERE p.n=1 AND p.present=1 AND p.identity_status='' AND p.protein!='' AND p.genome!=''
          AND x.protein IS NULL AND json_array_length(p.replicons_json)<=1;
        CREATE INDEX family_membership ON family_groups(membership,comp);
        CREATE INDEX family_namespace ON family_groups(comp,ns);
    """)
    profile_roles, disagreement = {}, set()
    for comp in COMPONENTS:
        profile = profiles.get(comp)
        distinct_profile = profile and list(profiles.values()).count(profile) == 1
        decision = decisions[comp]
        decision["family_profile"] = profile
        if decision["source"] != "no_role_anchor":
            continue
        # An explicit taxonomic scope asks for orthology evidence at that level.
        profile_roles[comp] = profile if distinct_profile and scope is None else None
        witnesses = []
        shared = competing = 0
        for ns, different, genomes, membership in db.execute(
            """
            SELECT ns,count(DISTINCT membership),count(DISTINCT genome),min(membership)
            FROM family_groups WHERE comp=? GROUP BY ns ORDER BY ns
        """,
            (comp,),
        ):
            if scope and json.loads(ns) != ["OrthoDB", scope]:
                continue
            if different > 1:
                competing += 1
                continue
            if db.execute(
                "SELECT 1 FROM family_groups WHERE membership=? AND comp!=? LIMIT 1",
                (membership, comp),
            ).fetchone():
                shared += 1
                continue
            if genomes >= 2:
                witnesses.append({"orthogroup": membership, "distinct_genomes": genomes})
        if competing or shared:
            disagreement.add(comp)
        expected[comp] = {w["orthogroup"] for w in witnesses}
        decision.update(
            {
                "source": "family_orthogroup_consensus"
                if witnesses
                else "family_profile"
                if profile_roles[comp]
                else "no_distinguishing_role",
                "orthogroups": sorted(expected[comp]),
                "consensus_evidence": witnesses,
                "competing_namespaces": competing,
                "shared_memberships": shared,
                "distinct_family_profile": bool(distinct_profile),
            }
        )
    return profile_roles, disagreement


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


def _classify(db, expected, profile_roles, disagreement):
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
        if profile_roles.get(comp) and reason in ("missing_orthology", "no_role_anchor"):
            if groups & set().union(*(expected[c] for c in COMPONENTS if c != comp)):
                reason = "overlapping_role_orthology"
            else:
                # Conflicting family evidence cannot be rescued by an
                # unannotated row. Missing or singleton evidence can.
                reason = "unresolved_family_orthology" if comp in disagreement else "profile_member"
        if n > 1:
            reason = "multiple_mappings"
        elif reason != "orthogroup_mismatch" and len(json.loads(raw_replicons)) > 1:
            reason = "multiple_replicon_annotations"
        elif not protein and reason != "orthogroup_mismatch":
            reason = "missing_molecule_id"
        elif not genome:
            reason = "missing_genome"
        db.execute(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,0)",
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


def _retain_incomplete_placements(db, expected, *, gff3_only=False):
    """Keep possible copies with genome evidence out of role learning but in the group audit."""
    facts = db.execute(
        """
        SELECT m.row_key,r.comp,m.protein,m.genome,m.locus,r.present,m.raw,m.status FROM mappings m
        JOIN rows r ON r.id=m.row_key WHERE r.anchor=0 AND m.status!='' AND m.genome!=''
          AND coalesce(json_extract(m.raw,'$.verification_only'),0)=0
        AND (?=0 OR json_extract(m.raw,'$.source_type')='gff3')
        ORDER BY m.row_key,r.comp,m.protein,m.genome,m.locus,r.present,m.status
    """,
        (int(gff3_only),),
    )
    for placement, records in itertools.groupby(facts, key=lambda v: v[:6]):
        key, comp, protein, genome, locus, present = placement
        possible = set()
        for *_, raw, status in records:
            data = json.loads(raw)
            # A failed biological ID lookup cannot exclude a row based on the
            # apparent molecule's role. Row-scoped or curated roles can exclude it.
            groups = (
                data.get("orthogroups", ())
                if data.get("source_type") in ("curated", "gff3")
                else data.get("stockholm_orthogroups", ())
            )
            if _role_status(set(groups), expected[comp]) != "orthogroup_mismatch":
                possible.add(status)
        reason = min(possible) if possible else "orthogroup_mismatch"
        db.execute(
            """INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,1)""",
            (key, comp, protein, genome, locus, _group_id(genome, locus), present, reason),
        )
    db.execute("""UPDATE candidates SET reason='multiple_mappings' WHERE reason!='orthogroup_mismatch'
        AND row_key IN (SELECT row_key FROM candidates WHERE reason!='orthogroup_mismatch'
                        GROUP BY row_key HAVING max(incomplete) AND count(DISTINCT group_id)>1)""")


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


def _shared_genomic_locus(db, gid):
    """Identify indistinguishable oriented nucleotide loci without collapsing RNA IDs."""
    seen = {}
    gene_roles = {}
    for comp, raw in db.execute(
        """
        SELECT DISTINCT c.comp,m.raw FROM candidates c JOIN mappings m USING(row_key,protein,genome,locus)
        WHERE c.group_id=? AND c.present AND c.reason IN ('supported','profile_member') AND m.status=''
          AND json_extract(m.raw,'$.genomic_start') IS NOT NULL
    """,
        (gid,),
    ):
        data = json.loads(raw)
        assembly = data.get("canonical_assembly_id") or data.get("assembly_id", "")
        if data.get("source_type") == "gff3":
            for gene in data.get("gff3", {}).get("gene_loci", ()):
                key = assembly, data.get("replicon_id"), gene
                if gene_roles.setdefault(key, comp) != comp:
                    return True
        start, end, strand = (
            data.get(k) for k in ("genomic_start", "genomic_end", "genomic_strand")
        )
        if (
            not ASSEMBLY.fullmatch(assembly)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or strand not in ("+", "-")
        ):
            continue
        # RNAcentral endpoints retain the provider's coordinate convention. Archive
        # placements use oriented 1-based genomic intervals from Stockholm rows.
        rna_coordinates = data.get("source") == "RNAcentral"
        label = data.get("replicon_id", "")
        if rna_coordinates and re.fullmatch(r"(?:[0-9]+|[IVX]+|[XYWZ]|MT)", label, re.IGNORECASE):
            label = "chromosome " + label
        replicon = _replicon_label(label)
        if not replicon or not replicon[1]:
            continue
        location = assembly, replicon, min(start, end), max(start, end), strand, rna_coordinates
        if seen.setdefault(location, comp) != comp:
            return True
    return False


def _summarize_groups(db):
    db.execute("""CREATE TEMP TABLE placements AS
        SELECT c.group_id AS gid,COUNT(DISTINCT NULLIF(m.taxon,'')) AS taxa,
        COUNT(DISTINCT NULLIF(coalesce(json_extract(m.raw,'$.canonical_assembly_id'),
                                      json_extract(m.raw,'$.assembly_id')),'')) AS assemblies
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
            MAX(reason NOT IN ('supported','profile_member')),COUNT(DISTINCT protein),MIN(protein) FROM candidates
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
                    "SELECT DISTINCT reason FROM candidates WHERE group_id=? AND comp=? AND reason NOT IN ('supported','profile_member','orthogroup_mismatch')",
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
        if all_ready and _shared_genomic_locus(db, gid):
            reasons.add("shared_genomic_locus_across_roles")
        if locus:
            labels = set()
            for comp, raw in db.execute(
                """SELECT c.comp,a.replicons_json FROM candidates c
                JOIN assignments a USING(row_key,protein,genome,locus)
                WHERE c.group_id=? AND c.present AND c.reason IN ('supported','profile_member')""",
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
            "unresolved_family_orthology",
            "conflicting_taxa",
            "conflicting_assemblies",
            "conflicting_neighborhood_replicons",
            "unverified_neighborhood_replicons",
            "shared_molecule_across_roles",
            "shared_genomic_locus_across_roles",
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


def _group_context(db, directory):
    """Count each decision reason once per group and each mapping issue once per row."""
    db.executescript("""
        CREATE TEMP TABLE context AS
        SELECT c.group_id AS gid, min(nullif(m.taxon,'')) AS taxon_id,
               count(distinct nullif(m.taxon,'')) AS n_taxa
        FROM candidates c JOIN mappings m USING(row_key,protein,genome,locus)
        WHERE c.reason!='orthogroup_mismatch' AND m.status=''
        GROUP BY c.group_id;
        CREATE UNIQUE INDEX context_key ON context(gid);
    """)

    def group_rows():
        for gid, genome, locus, status, tier, taxon, n_taxa in db.execute("""
            SELECT group_id,genome,locus,status,role_evidence,taxon_id,n_taxa
            FROM groups JOIN tiers USING(group_id) LEFT JOIN context ON gid=group_id
            ORDER BY group_id
        """):
            yield (gid, genome, locus, status, tier, taxon if n_taxa == 1 else "", n_taxa or 0)

    _write_table(
        directory / "group_context.tsv",
        (
            "group_id",
            "genome_id",
            "neighborhood_id",
            "status",
            "role_evidence",
            "taxon_id",
            "distinct_taxon_ids",
        ),
        group_rows(),
    )
    reason_counts = {}
    for (reasons,) in db.execute("SELECT reasons FROM groups WHERE reasons!=''"):
        for code in reasons.split(";"):
            reason_counts[code] = 1 + reason_counts.get(code, 0)
    mapping_counts = dict(
        db.execute("""
        SELECT m.status,count(distinct m.row_key) FROM mappings m JOIN rows r ON r.id=m.row_key
        WHERE r.anchor=0 AND m.status!='' GROUP BY m.status
    """)
    )
    repeated = dict(
        db.execute("""
        SELECT comp,count(*) FROM (
          SELECT DISTINCT c.group_id, c.comp, c.protein
          FROM candidates c JOIN rows r ON r.id=c.row_key WHERE r.anchor=0 AND r.present=1
          AND c.reason!='orthogroup_mismatch' AND c.protein!=''
          AND r.start IS NOT NULL AND r.end IS NOT NULL
          GROUP BY c.group_id,c.comp,c.protein HAVING count(distinct printf('%d:%d',r.start,r.end))>1
        ) GROUP BY comp
    """)
    )
    return {
        "groups_per_reason": reason_counts,
        "distinct_rows_per_mapping_issue": mapping_counts,
        "molecules_with_multiple_alignment_intervals_by_role": repeated,
    }


def _audit_rows(db, names, *, gff3=False):
    query = """SELECT c.group_id,g.status,g.reasons,r.comp,r.name,r.present,r.accession,r.start,r.end,
        r.selectors,m.lookup,m.raw,m.status,c.reason,r.declared,r.identity_status FROM mappings m JOIN rows r ON m.row_key=r.id
        LEFT JOIN candidates c ON (m.status='' OR c.incomplete=1) AND coalesce(json_extract(m.raw,'$.verification_only'),0)=0
        AND c.row_key=r.id AND c.protein=m.protein AND c.genome=m.genome AND c.locus=m.locus
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
        if role == "orthogroup_mismatch":
            status = role
        confidence = "uncertain"
        if role == "orthogroup_mismatch":
            confidence = "role_excluded"
        elif gstatus == "resolved" and not mstatus and role == "profile_member":
            confidence = (
                "neighborhood_and_family_profile"
                if data.get("neighborhood_id")
                else "genome_and_family_profile"
            )
        elif gstatus == "resolved" and not mstatus and role == "supported":
            confidence = (
                "neighborhood_and_role_supported"
                if data.get("neighborhood_id")
                else "genome_and_role_supported"
            )
        yield (
            gid,
            gstatus or "unplaced",
            greasons or "",
            int(gstatus == "resolved" and not mstatus and role in ("supported", "profile_member")),
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
            *[
                data.get(k, "")
                for k in (
                    "molecule_type",
                    "nucleotide_accession",
                    "genomic_start",
                    "genomic_end",
                    "genomic_strand",
                    "placement_source",
                    "organism_name",
                    "provider_error",
                )
            ],
            json.dumps(data.get("assembly_aliases", [])),
            json.dumps(data.get("stockholm_annotations", [])),
            *((json.dumps(data.get("gff3", {}), sort_keys=True),) if gff3 else ()),
        )


def _genome_aliases(db, resolver):
    """Collect identifiers before mapping verified synonyms to common genome keys."""
    db.executescript("""
        CREATE TEMP TABLE genome_claims(id TEXT PRIMARY KEY) WITHOUT ROWID;
        INSERT OR IGNORE INTO genome_claims SELECT genome FROM mappings WHERE genome!='';
        INSERT OR IGNORE INTO genome_claims SELECT json_extract(raw,'$.assembly_id') FROM mappings;
        INSERT OR IGNORE INTO genome_claims
            SELECT j.value FROM mappings m,json_each(m.raw,'$.stockholm_assemblies') j;
        INSERT OR IGNORE INTO genome_claims
            SELECT j.value FROM mappings m,json_each(m.raw,'$.stockholm_unversioned_assemblies') j;
        CREATE TEMP TABLE genome_aliases(id TEXT PRIMARY KEY,canonical TEXT,witness TEXT) WITHOUT ROWID;
        CREATE TEMP TABLE proteome_alternatives(id TEXT PRIMARY KEY,genomes TEXT) WITHOUT ROWID;
    """)
    if db.execute("SELECT 1 FROM genome_claims WHERE id GLOB 'GCA_*' LIMIT 1").fetchone():
        targets = (
            v
            for (v,) in db.execute("SELECT id FROM genome_claims WHERE id GLOB 'GCF_*' ORDER BY id")
            if ASSEMBLY.fullmatch(v)
        )
        while chunk := tuple(itertools.islice(targets, 30)):
            for original, relation in resolver.assembly_pairs(chunk).items():
                if canonical := relation.get("canonical"):
                    witness = {
                        "source": "NCBI Datasets",
                        "reported_assembly": original,
                        "paired_assembly": relation["paired_record"],
                    }
                    db.execute(
                        "INSERT INTO genome_aliases VALUES (?,?,?)",
                        (original, canonical, json.dumps(witness)),
                    )
    db.execute("INSERT OR IGNORE INTO genome_claims SELECT canonical FROM genome_aliases")
    # A single UniProt/curated placement or an unambiguous row assertion can link a
    # proteome to an assembly. Conflicting endpoints never establish an equivalence.
    for proteome, canonical_assemblies, raw_assemblies in db.execute("""
        WITH pairs AS (
            SELECT DISTINCT json_extract(raw,'$.proteome_id') AS up,
                   json_extract(raw,'$.assembly_id') AS assembly
            FROM mappings WHERE status='' AND json_extract(raw,'$.source_type') IN ('uniprot','curated')
            UNION SELECT DISTINCT json_extract(raw,'$.stockholm_proteomes[0]'),
                                  json_extract(raw,'$.stockholm_assemblies[0]')
            FROM mappings WHERE status='' AND json_extract(raw,'$.source_type')='stockholm'
            AND json_array_length(raw,'$.stockholm_proteomes')=1 AND json_array_length(raw,'$.stockholm_assemblies')=1)
        SELECT up,json_group_array(DISTINCT coalesce(a.canonical,p.assembly)),json_group_array(DISTINCT p.assembly)
        FROM pairs p LEFT JOIN genome_aliases a ON a.id=p.assembly
        WHERE up GLOB 'UP*' AND (p.assembly GLOB 'GCA_*' OR p.assembly GLOB 'GCF_*')
        GROUP BY up
    """):
        originals = sorted(json.loads(raw_assemblies))
        if re.fullmatch(r"UP\d{9}", proteome) and all(ASSEMBLY.fullmatch(a) for a in originals):
            builds = sorted(json.loads(canonical_assemblies))
            if len(builds) > 1:
                db.execute(
                    "INSERT INTO proteome_alternatives VALUES (?,?)", (proteome, json.dumps(builds))
                )
                continue
            canonical = builds[0]
            witness = {
                "source": "proteome-assembly placement",
                "proteome_id": proteome,
                "reported_assemblies": originals,
                "canonical_assembly": canonical,
            }
            witness["assembly_equivalences"] = [
                json.loads(record[0])
                for original in originals
                if (
                    record := db.execute(
                        "SELECT witness FROM genome_aliases WHERE id=?", (original,)
                    ).fetchone()
                )
            ]
            db.execute(
                "INSERT INTO genome_aliases VALUES (?,?,?)",
                (proteome, canonical, json.dumps(witness)),
            )


def _placement_options(data, status, resolve, versions, proteome_builds):
    """Normalize local constraints and a provider's intact fact before identifying conflicts."""
    explicit = data.get("source_type") == "curated" or data.get("verification_only")
    local = (
        []
        if explicit
        else data.get("stockholm_assemblies", []) + data.get("stockholm_proteomes", [])
    )
    unversioned = [] if explicit else data.get("stockholm_unversioned_assemblies", [])
    possibilities = {}
    if unversioned:
        # A stem supplies a finite set of possible *observed* versions. It
        # cannot authorize choosing one of them or projecting a provider locus.
        possibilities = {
            accession: stem
            for stem in unversioned
            for accession in versions(stem)
            if accession not in local
        }
        local += list(possibilities)
        status = status or "unversioned_stockholm_assembly"
    primary = data.get("genome_id", "")
    observed = list(dict.fromkeys([*local, *([primary] if primary else [])]))
    if not observed:
        return [(data, status)]
    destinations = {}
    witnesses, aliases = [], []
    explicit_builds = {resolve(v)[0] for v in observed if ASSEMBLY.fullmatch(v)}
    for identifier in observed:
        alternatives = proteome_builds(identifier) if identifier.startswith("UP") else ()
        if alternatives:
            # A proteome can span reported builds. An explicit compatible
            # assembly narrows its placement; the proteome alone leaves them open.
            builds = (
                sorted(explicit_builds)
                if explicit_builds and explicit_builds.issubset(alternatives)
                else alternatives
            )
            for genome in builds:
                destinations.setdefault(genome, []).append(identifier)
            witnesses.append(
                {
                    "source": "proteome-assembly placements",
                    "proteome_id": identifier,
                    "possible_assemblies": alternatives,
                }
            )
            continue
        canonical, witness = resolve(identifier)
        destinations.setdefault(canonical, []).append(identifier)
        if witness:
            witnesses.append(witness)
            if witness["source"] == "NCBI Datasets":
                aliases.extend((identifier, canonical))
    base = dict(data)
    for build, stem in possibilities.items():
        witnesses.append(
            {
                "source": "versionless Stockholm assembly",
                "reported_assembly": stem,
                "possible_observed_assembly": build,
            }
        )
    if witnesses:
        base["cross_references"] = [*data.get("cross_references", ()), *witnesses]
    if aliases:
        base["assembly_aliases"] = list(dict.fromkeys(aliases))
    if len(destinations) == 1:
        genome, originals = next(iter(destinations.items()))
        original_assembly = data.get("assembly_id") or next(
            (possibilities.get(v, v) for v in originals if ASSEMBLY.fullmatch(v)), ""
        )
        base.update(genome_id=genome, assembly_id=original_assembly)
        if original_assembly and resolve(original_assembly)[0] != original_assembly:
            base["canonical_assembly_id"] = resolve(original_assembly)[0]
        for key, value in data.get("stockholm_context", {}).items():
            if value and not base.get(key) and not explicit:
                base[key] = value
        return [(base, status)]

    # Incompatible placement assertions have finite alternatives. Only retain the
    # provider's genomic interval on its own genome. Stockholm tags stay as source
    # assertions; their lists do not define pairings of replicons, loci and assemblies.
    provider = data.get("source_type") in {"uniprot", "nucleotide"} and data.get(
        "provider_genome_id"
    )
    options = []
    conflict = "conflicting_stockholm_assembly" if provider else "conflicting_stockholm_annotations"
    for genome, originals in destinations.items():
        fact = {**base, "genome_id": genome}
        if provider and resolve(provider)[0] == genome:
            if assembly := fact.get("assembly_id"):
                fact["canonical_assembly_id"] = resolve(assembly)[0]
        else:
            fact.update(
                protein_id="",
                replicon_id="",
                neighborhood_id="",
                genomic_start="",
                genomic_end="",
                genomic_strand="",
                source="Stockholm",
                source_type="stockholm",
                placement_source="Stockholm GS",
                assembly_id=next(
                    (possibilities.get(v, v) for v in originals if ASSEMBLY.fullmatch(v)), ""
                ),
                proteome_id=next((v for v in originals if re.fullmatch(r"UP\d{9}", v)), ""),
            )
            fact["orthogroups"] = data.get("stockholm_orthogroups", [])
            if provider and provider in originals:
                fact.update(
                    source=data["source"],
                    source_type=data["source_type"],
                    placement_source="multiple documented proteome assemblies",
                )
        options.append((fact, status or conflict))
    return options


def _reconcile_genomes(db, resolver):
    _genome_aliases(db, resolver)

    @lru_cache(maxsize=4096)
    def resolve(identifier):
        entry = db.execute(
            "SELECT canonical,witness FROM genome_aliases WHERE id=?", (identifier,)
        ).fetchone()
        return (entry[0], json.loads(entry[1])) if entry else (identifier, None)

    @lru_cache(maxsize=512)
    def versions(stem):
        return tuple(
            v
            for (v,) in db.execute(
                "SELECT id FROM genome_claims WHERE id GLOB ? ORDER BY id", (stem + ".*",)
            )
            if ASSEMBLY.fullmatch(v)
        )

    @lru_cache(maxsize=4096)
    def proteome_builds(identifier):
        row = db.execute(
            "SELECT genomes FROM proteome_alternatives WHERE id=?", (identifier,)
        ).fetchone()
        return json.loads(row[0]) if row else []

    # Freeze the input row IDs. Extra alternatives are emitted once, after all
    # source lookups, and never become inputs to a second normalization pass.
    last = db.execute("SELECT max(rowid) FROM mappings").fetchone()[0]
    for entry, key, lookup, raw, status in db.execute(
        "SELECT rowid,row_key,lookup,raw,status FROM mappings WHERE rowid<=? ORDER BY rowid",
        (last,),
    ):
        data = json.loads(raw)
        for i, (fact, decision) in enumerate(
            _placement_options(data, status, resolve, versions, proteome_builds)
        ):
            if i:
                _append(db, key, lookup, fact, decision)
            elif fact != data or decision != status:
                db.execute(
                    """UPDATE mappings SET protein=?,genome=?,locus=?,groups_json=?,raw=?,status=?
                             WHERE rowid=?""",
                    (
                        fact.get("protein_id", ""),
                        fact.get("genome_id", ""),
                        fact.get("neighborhood_id", ""),
                        json.dumps(fact.get("orthogroups", [])),
                        json.dumps(fact, sort_keys=True),
                        decision,
                        entry,
                    ),
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
    automatic_profiles=None,
    automatic=False,
    molecule_types=None,
    nucleotide_cache=None,
    nucleotide_online=False,
    gff3=None,
    provider_policy=None,
    progress=None,
    checkpoint=None,
    total_rows=None,
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
    with (
        request_context(
            provider_policy, progress, checkpoint=checkpoint, total=total_rows
        ) as request_state,
        closing(sqlite3.connect("")) as db,
        closing(Resolver(cache, uniprot, audit_errors=automatic)) as resolver,
        (
            closing(NucleotideResolver(nucleotide_cache, nucleotide_online))
            if automatic
            else nullcontext()
        ) as nucleotides,
    ):
        _database(db)
        identity = _load_manifest(db, manifest)
        gff_sources = load_annotations(db, gff3, molecule_types or {})
        for comp in COMPONENTS:
            options = {
                "molecule": (molecule_types or {}).get(comp),
                "family": (automatic_profiles or {}).get(comp),
                "nucleotides": nucleotides if automatic else None,
            }
            _collect(db, comp, rows[comp], resolver, comp in protein_components, **options)
            if (ref := (reference_anchors or {}).get(comp)) and not (
                expected_orthogroups or {}
            ).get(comp):
                occurrence = db.execute(
                    "SELECT accession,start,end,selectors,coalesce(tags,'[]') FROM rows LEFT JOIN row_tags ON id=row_key WHERE comp=? AND name=?",
                    (comp, ref),
                ).fetchone()
                anchor = (
                    EvidenceRow(
                        ref,
                        occurrence[0],
                        occurrence[1],
                        occurrence[2],
                        selectors=tuple(json.loads(occurrence[3])),
                        annotations=tuple(tuple(pair) for pair in json.loads(occurrence[4])),
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
                    **options,
                )
        if automatic:
            _reconcile_genomes(db, nucleotides)
        _reconcile_intervals(db)
        expected, decisions = _role_expectations(
            db, expected_orthogroups, reference_anchors, orthology_scope
        )
        for comp, name in db.execute("SELECT comp,name FROM rows WHERE anchor=1"):
            decisions[comp]["reference_row_id"] = name
        profile_roles, disagreement = {}, set()
        if automatic_profiles is not None:
            profile_roles, disagreement = _automatic_roles(
                db, expected, decisions, automatic_profiles, orthology_scope
            )
        _classify(db, expected, profile_roles, disagreement)
        if automatic or gff_sources:
            _retain_incomplete_placements(db, expected, gff3_only=not automatic)
        _summarize_groups(db)
        db.executescript("""
            CREATE TEMP TABLE tiers AS SELECT g.group_id, CASE
                WHEN g.status!='resolved' THEN ''
                WHEN max(c.reason='profile_member') THEN 'profile_membership_included'
                ELSE 'orthogroup_supported' END AS role_evidence
            FROM groups g JOIN candidates c USING(group_id) GROUP BY g.group_id;
            CREATE UNIQUE INDEX tiers_group ON tiers(group_id);
        """)
        attrition = _group_context(db, out_dir)
        tier_fields = ("role_evidence",) if automatic_profiles is not None else ()
        tier_column = ",t.role_evidence" if tier_fields else ""
        _write_table(
            out_dir / "groups.tsv",
            GROUP_FIELDS + tier_fields,
            db.execute(
                f"SELECT g.*{tier_column} FROM groups g JOIN tiers t USING(group_id) ORDER BY group_id"
            ),
        )
        _write_table(
            out_dir / "evidence.tsv",
            AUDIT_FIELDS + (("gff3_annotation",) if gff_sources else ()),
            _audit_rows(db, names, gff3=bool(gff_sources)),
        )
        _write_table(
            out_dir / "selected_evidence.tsv",
            SELECTED_FIELDS + tier_fields,
            db.execute(f"""
            SELECT g.group_id,r.comp,r.name,g.genome,g.locus{tier_column} FROM groups g JOIN candidates c USING(group_id)
            JOIN tiers t USING(group_id) JOIN rows r ON r.id=c.row_key
            WHERE g.status='resolved' AND c.reason IN ('supported','profile_member')
            ORDER BY g.group_id,r.comp"""),
        )
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM groups GROUP BY status"))
        resolved = statuses.get("resolved", 0)
        summary = {
            "schema_version": 2 if gff_sources else 1,
            **({"gff3_sources": gff_sources} if gff_sources else {}),
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
            "attrition": attrition,
            "manifest": identity,
            "request_policy": request_state.metadata(),
            "component_names": {c: names.get(c, c) for c in COMPONENTS},
            "roles": decisions,
            "orthology_scope": orthology_scope,
            "uniprot_components": sorted(protein_components),
            "resolver": {
                **resolver.provenance(),
                "precedence": "exact curated domain, then curated aliases, then UniProt for enabled molecules; declared accessions require independent molecule links",
            },
            "tables": ["evidence.tsv", "selected_evidence.tsv", "groups.tsv", "group_context.tsv"],
        }
        if automatic_profiles is not None:
            summary["role_evidence_counts"] = dict(
                db.execute("""
                SELECT role_evidence,count(*) FROM tiers WHERE role_evidence!='' GROUP BY role_evidence
            """)
            )
        if cache and cache.is_file():
            with cache.open("rb") as stream:
                summary["resolver"]["cache_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        if automatic:
            failures = dict(nucleotides.failures)
            if resolver.failure:
                failures["UniProtKB"] = resolver.failure
            snapshot = {
                "cache": nucleotide_cache.name if nucleotide_cache else None,
                "online": nucleotide_online,
                "counts": dict(nucleotides.stats),
                "cache_schema": 1,
                "endpoints": {
                    "ENA Portal sequence metadata": ENA,
                    "NCBI nuccore ESummary": SUMMARY,
                    "NCBI Datasets genome": NCBI,
                    "RNAcentral genome-locations": RNACENTRAL,
                },
            }
            if nucleotide_cache and nucleotide_cache.is_file():
                with nucleotide_cache.open("rb") as stream:
                    snapshot["cache_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            attempts = dict(
                db.execute("""SELECT status,COUNT(DISTINCT row_key) FROM mappings m
                    JOIN rows r ON r.id=m.row_key WHERE r.anchor=0 AND status!='' GROUP BY status""")
            )
            summary["providers"] = {
                "status": "partial" if failures else "complete",
                "failures": failures,
                "nucleotide": snapshot,
                "lookup_status_counts": attempts,
            }
        with (out_dir / "evidence_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary
