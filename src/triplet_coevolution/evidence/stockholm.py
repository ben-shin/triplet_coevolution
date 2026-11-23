"""Interpret row-scoped Stockholm annotations independently of provider transport."""

import re

from .identifiers import declared_accessions
from .nucleotide import ASSEMBLY, nucleotide_key


def annotation_values(row):
    values = {}
    for tag, text in getattr(row, "annotations", ()):
        tag, value = tag.upper(), text.strip()
        if tag == "DE":
            if taxon := re.search(r"(?:^|\s)OX=([1-9]\d*)(?:\s|$)", value):
                values.setdefault("OX", set()).add(taxon[1])
            if organism := re.search(r"(?:^|\s)OS=(.+?)(?=\s[A-Z]{2}=|$)", value):
                values.setdefault("OS", set()).add(organism[1].strip())
            continue
        if tag == "DR":
            fields = [f.strip() for f in value.split(";")]
            if len(fields) < 2:
                continue
            db, value = fields[:2]
            tag = {
                "assembly": "ASSEMBLY",
                "ncbi_assembly": "ASSEMBLY",
                "proteomes": "PROTEOME",
                "orthodb": "ORTHOGROUP",
                "taxon": "OX",
            }.get(db.lower(), "")
            if db.lower() == "orthodb":
                if not re.fullmatch(r"\d+at\d+", value):
                    continue
                value = "OrthoDB:" + value
        if tag == "OX":
            match = re.fullmatch(r"(?:NCBI_TaxID=)?([1-9]\d*);?", value)
            value = match[1] if match else ""
        if tag == "ASSEMBLY" and not ASSEMBLY.fullmatch(value):
            tag = "INVALID_ASSEMBLY"
        if tag == "PROTEOME" and not re.fullmatch(r"UP\d{9}", value):
            tag = "INVALID_PROTEOME"
        if (
            tag
            in {
                "OX",
                "OS",
                "ASSEMBLY",
                "PROTEOME",
                "ORTHOGROUP",
                "LOCUS",
                "REPLICON",
                "NEIGHBORHOOD",
                "INVALID_ASSEMBLY",
                "INVALID_PROTEOME",
            }
            and value
        ):
            values.setdefault(tag, set()).add(value)
    return values


def nucleotide_identity(row, values, family=None):
    allow_ambiguous = bool(re.fullmatch(r"RF\d{5}(?:\.\d+)?", family or ""))
    ids = {
        key
        for token in (row.row_id, row.accession)
        if (key := nucleotide_key(token, allow_ambiguous=allow_ambiguous))
    }
    if len(ids) > 1:
        # A versionless source + its own single versioned accession refer to the same lookup.
        versions = {v for v in ids if "." in v}
        if len(versions) == 1 and ids == versions | {next(iter(versions)).split(".")[0]}:
            ids = versions
        elif all(v.startswith("URS") for v in ids):
            scoped = {v for v in ids if "_" in v}
            if len(scoped) != 1 or ids != scoped | {next(iter(scoped)).split("_")[0]}:
                return "", "conflicting_nucleotide_identifiers"
            ids = scoped
        else:
            return "", "conflicting_nucleotide_identifiers"
    key = next(iter(ids), "")
    if key.startswith("URS"):
        taxa = values.get("OX", set())
        if "_" in key and taxa and taxa != {key.split("_")[1]}:
            return key, "conflicting_stockholm_taxa"
        if "_" not in key and len(taxa) == 1:
            key += "_" + next(iter(taxa))
    return key, ""


def metadata_fields(row, molecule, values):
    return {
        "molecule_type": molecule or "",
        "stockholm_annotations": list(getattr(row, "annotations", ())),
        "organism_name": " | ".join(sorted(values.get("OS", ()))),
        "taxon_id": " | ".join(sorted(values.get("OX", ()))),
    }


def local_placement(row, molecule, family=None):
    """Return a definitive row-scoped assertion, or the nucleotide key to consult."""
    values = annotation_values(row)
    metadata = metadata_fields(row, molecule, values)
    nuc, issue = (
        nucleotide_identity(row, values, family) if molecule in {"dna", "rna"} else ("", "")
    )
    tagged = {k: sorted(v) for k, v in values.items()}

    def get(key):
        choices = values.get(key, ())
        return next(iter(choices)) if len(choices) == 1 else ""

    # Genome placement can survive unresolved molecule identity. Retain each
    # unambiguous row-scoped placement field even when an accession lookup fails.
    genome = get("ASSEMBLY") or get("PROTEOME")
    provenance = {
        **metadata,
        "source": "Stockholm",
        "source_type": "stockholm",
        "placement_source": "Stockholm GS",
        "orthogroups": sorted(values.get("ORTHOGROUP", ())),
        "stockholm_orthogroups": sorted(values.get("ORTHOGROUP", ())),
        "cross_references": [{"stockholm": tagged}],
        "genome_id": genome,
        "assembly_id": get("ASSEMBLY"),
        "proteome_id": get("PROTEOME"),
        "neighborhood_id": get("NEIGHBORHOOD"),
        "replicon_id": get("REPLICON"),
        "nucleotide_accession": nuc,
    }
    if (
        issue
        or any(len(v) > 1 for k, v in values.items() if k not in {"ORTHOGROUP", "OS"})
        or any(k.startswith("INVALID_") for k in values)
    ):
        return (
            nuc,
            [(row.row_id, provenance, issue or "conflicting_stockholm_annotations")],
            provenance,
        )

    if not genome:
        return nuc, [], provenance
    protein = get("LOCUS")
    if not protein:
        if molecule == "protein":
            acs = declared_accessions(row)
            if len(acs) > 1:
                return nuc, [(row.row_id, provenance, "unlinked_accession")], provenance
            base = re.sub(r"/\d+-\d+$", "", row.row_id)
            if acs:
                protein = next(iter(acs))
            elif re.fullmatch(r"(?:NP|XP|YP|WP)_\d+(?:\.\d+)?", base):
                protein = base
            else:
                # An opaque row label or family name is not a stable molecule ID.
                # A recognized mnemonic may still resolve to one via UniProt.
                return nuc, [], provenance
        elif nuc and "." in nuc and row.start is not None:
            protein = f"{nuc}/{row.start}-{row.end}"
        else:
            # A URS identifies a sequence that may occur at several positions. A genome
            # label alone supplies neither the position nor the number of copies.
            return nuc, [], provenance
    mapping = {
        **provenance,
        "protein_id": protein,
    }
    if nuc and not nuc.startswith("URS") and row.start is not None:
        mapping.update(
            genomic_start=row.start,
            genomic_end=row.end,
            genomic_strand="+" if row.end >= row.start else "-",
            replicon_id=mapping["replicon_id"] or nuc,
        )
    return nuc, [(row.row_id, mapping, "")], provenance


def nucleotide_facts(row, key, payload, local):
    stamp = {
        **local,
        "nucleotide_accession": key,
        "source": payload.get("source", "nucleotide metadata"),
        "source_type": "nucleotide",
        "fetched_at": payload.get("fetched_at", ""),
        "placement_source": payload.get("source", ""),
        "provider_error": payload.get("provider_error", ""),
    }
    if payload.get("status") != "mapped":
        return [(key, stamp, payload.get("status", "nucleotide_not_found"))]
    facts = {}
    assertions = annotation_values(row)
    for placement in payload["placements"]:
        ac = placement["nucleotide_accession"]
        assembly = placement.get("assembly_id") or local.get("assembly_id", "")
        common = {
            **stamp,
            **placement,
            "assembly_id": assembly,
            "genome_id": assembly or local.get("genome_id", ""),
        }
        for field in ("taxon_id", "proteome_id", "replicon_id"):
            common[field] = placement.get(field) or local.get(field, "")
        common["cross_references"] = [
            *local["cross_references"],
            {"provider": stamp["source"], "query": key, "placement": placement},
        ]
        if (
            local.get("taxon_id")
            and placement.get("taxon_id")
            and local["taxon_id"] != placement["taxon_id"]
        ):
            status = "conflicting_stockholm_taxa"
        elif (
            assertions.get("ASSEMBLY")
            and placement.get("assembly_id")
            and placement["assembly_id"] not in assertions["ASSEMBLY"]
        ):
            status = "conflicting_stockholm_assembly"
        elif "genomic_start" in placement and not placement.get("assembly_id"):
            status = "unverified_coordinate_assembly"
        elif "genomic_start" in placement:
            # RNAcentral gives locations of the entire RNA, possibly spanning introns.
            # A Stockholm /start-end suffix here indexes the RNA, not genomic bases.
            common["protein_id"] = (
                f"{ac}@{placement['replicon_id']}:{placement['genomic_start']}-{placement['genomic_end']}:{placement['genomic_strand']}"
            )
            status = ""
        elif row.start is None:
            status = "missing_locus_interval"
        elif max(row.start, row.end) > placement["length"]:
            status = "interval_outside_sequence"
        else:
            common.update(
                protein_id=f"{ac}/{row.start}-{row.end}",
                genomic_start=row.start,
                genomic_end=row.end,
                genomic_strand="+" if row.end >= row.start else "-",
            )
            status = ""
        if assertions.get("NEIGHBORHOOD"):
            common["neighborhood_id"] = next(iter(assertions["NEIGHBORHOOD"]))
        facts[ac, common.get("genome_id"), common.get("protein_id"), status] = (key, common, status)
    if local.get("genome_id") and not any(
        f[1].get("genome_id") == local["genome_id"] for f in facts.values()
    ):
        facts["stockholm"] = (row.row_id, local, "conflicting_stockholm_assembly")
    # A rejected alternate must not disappear and silently make a competing placement unique.
    if any(f[2] for f in facts.values()) and len(facts) > 1:
        return [(k, m, s or "incomplete_nucleotide_placements") for k, m, s in facts.values()]
    return list(facts.values())
