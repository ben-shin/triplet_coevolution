"""Find all annotated candidates before choosing a single intact locus per component."""

import gzip
import json
import re
from collections import defaultdict
from urllib.parse import unquote

from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
from cohort_sources import STAGE, atomic_json, request_file, source

RNA = "5S"
LOCI = ("rpoA", "rpoB", "rpoC", RNA, "rplE", "rplR")
ACCESSIONS = dict(
    zip(
        ("rpoA", "rpoB", "rpoC", "rplE", "rplR"),
        ("P0A7Z4", "P0A8V2", "P0A8T7", "P62399", "P0C018"),
        strict=True,
    )
)


def references():
    refdir = STAGE / "references"
    refs = {}
    for gene, acc in ACCESSIONS.items():
        p = request_file(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", refdir / f"{acc}.fasta")
        rec = SeqIO.read(p, "fasta")
        if rec.id.split("|")[1] != acc:
            raise ValueError(f"incorrect canonical reference {acc}")
        refs[gene] = (acc, str(rec.seq))
    assembly_members = {}
    entities = {}
    for entry in ["4YBB", "4YG2"]:
        path = request_file(
            f"https://www.rcsb.org/fasta/entry/{entry}/display", refdir / f"{entry}.fasta"
        )
        assembly_url = f"https://data.rcsb.org/rest/v1/core/assembly/{entry}/1"
        membership = request_file(assembly_url, refdir / f"{entry}-assembly1-metadata.json")
        data = json.loads(membership.read_text(encoding="utf-8"))
        assembly_members[entry] = {
            s for gen in data["pdbx_struct_assembly_gen"] for s in gen["asym_id_list"]
        }
        entities.update({r.id.split("|")[0]: r for r in SeqIO.parse(path, "fasta")})
        if entry == "4YBB":
            records = [
                r
                for r in SeqIO.parse(path, "fasta")
                if "5s ribosomal rna" in r.description.lower()
                or "|5s rrna|" in r.description.lower()
            ]
            if len({str(r.seq) for r in records}) != 1:
                raise ValueError("4YBB 5S entity is absent or has distinct sequences")
            refs[RNA] = (records[0].id.split("|")[0], str(records[0].seq).replace("T", "U"))
    role_records = {
        "rpoA": ("4YG2_1", ["A", "B", "G", "H"]),
        "rpoB": ("4YG2_2", ["C", "I"]),
        "rpoC": ("4YG2_3", ["D", "J"]),
        RNA: ("4YBB_23", ["RA", "WB"]),
        "rplE": ("4YBB_27", ["AC", "VA"]),
        "rplR": ("4YBB_36", ["EB", "KC"]),
    }
    witness = {}
    for gene, (entity, labels) in role_records.items():
        rec = entities[entity]
        directory_members = assembly_members[entity.split("_")[0]]
        seq = str(rec.seq).upper()
        aligned, stats = project(gene, seq, refs[gene][1])
        if aligned is None:
            raise ValueError(f"deposited reference entity {entity} fails sequence correspondence")
        witness[gene] = {
            "coordinate_id": refs[gene][0],
            "deposited_entity": entity,
            "deposited_entity_description": rec.description,
            "deposited_entity_length": len(seq),
            "axis_length": len(refs[gene][1]),
            "assembly_1_label_asym_ids": [i for i in labels if i in directory_members],
            "sequence_alignment": stats,
        }
    atomic_json(refdir / "correspondence.json", witness)
    return refs


def roles(attributes):
    gene = attributes.get("gene", "").lower()
    synonyms = {gene, *attributes.get("gene_synonym", "").lower().split(",")}
    product = attributes.get("product", "").lower().replace("′", "'").replace("’", "'")
    names = {role for role in LOCI if role.lower() in synonyms}
    if (
        any(re.fullmatch(r"rrf[a-z0-9]*", s) for s in synonyms)
        or product in ("5s ribosomal rna", "5s rrna")
        or "RFAM:RF00001" in attributes.get("Dbxref", "").split(",")
    ):
        names.add(RNA)
    if "ribosomal protein" in product or "ribosomal subunit protein" in product:
        if re.search(r"\b(?:u|b)?l5\b", product):
            names.add("rplE")
        if re.search(r"\b(?:u|b)?l18\b", product) and not re.search(
            "l18p?/l5e|l18 family protein", product
        ):
            names.add("rplR")
    if "rna polymerase" in product and not re.search(
        r"binding|assembly|inhibitor|associated", product
    ):
        if re.search(r"subunit alpha\b|alpha subunit", product):
            names.add("rpoA")
        if re.search(r"subunit beta(?:\b(?![-' ]?prime|')|$)|beta subunit", product):
            names.add("rpoB")
        if re.search(r"beta(?:[- ]?prime|')", product):
            names.add("rpoC")
    for s in synonyms:
        for name in ("rpoA", "rpoB", "rpoC", "rplE", "rplR"):
            if re.fullmatch(re.escape(name.lower()) + r"[0-9]+", s):
                names.add(name)
        if s == "rpobc":
            names.update(("rpoB", "rpoC"))
    return names


def annotation_candidates(path):
    genes = {}
    links = defaultdict(lambda: defaultdict(list))
    feature_roles = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as h:
        for line in h:
            if line.startswith("#"):
                continue
            p = line.rstrip("\r\n").split("\t")
            if len(p) != 9 or p[2] not in ("gene", "pseudogene", "CDS", "rRNA"):
                continue
            a = {
                k: unquote(v)
                for item in p[8].split(";")
                if "=" in item
                for k, v in [item.split("=", 1)]
            }
            is_gene = p[2] in ("gene", "pseudogene")
            key = (p[0], a.get("ID", "") if is_gene else a.get("Parent", a.get("ID", "")))
            called = roles(a)
            if is_gene and a.get("gene_biotype") == "protein_coding":
                called.discard(RNA)
            if is_gene:
                genes[key] = (a, p[2])
                if called:
                    feature_roles[key].update(called)
            elif called or key in feature_roles:
                feature_roles[key].update(called)
                links[key][p[2]].append((p, a))
    candidates = defaultdict(list)
    for key, called in feature_roles.items():
        gene, kind = genes.get(key, ({}, None))
        for role in called:
            target = "rRNA" if role == RNA else "CDS"
            features = links[key][target]
            reasons = []
            if len(called) > 1:
                reasons.append("conflicting_target_roles")
            if kind is None or len(features) != 1:
                reasons.append("missing_or_split_feature")
            if kind == "pseudogene" or gene.get("pseudo") == "true" or gene.get("pseudogene"):
                reasons.append("pseudogene")
            if gene.get("partial") == "true":
                reasons.append("partial")
            if role != RNA and re.fullmatch(
                re.escape(role) + r"\d+", gene.get("gene", ""), re.IGNORECASE
            ):
                reasons.append("numbered_paralog_or_split_locus")
            rec = {
                "role": role,
                "locus": gene.get("locus_tag", key[1]),
                "genomic_accession": key[0],
                "parent": key[1],
                "reasons": reasons,
            }
            if features:
                p, attr = features[0]
                rec.update(
                    start=int(p[3]),
                    end=int(p[4]),
                    strand=p[6],
                    accession=attr.get("protein_id"),
                    product=attr.get("product"),
                    feature_id=attr.get("ID"),
                )
                if attr.get("pseudo") == "true" or attr.get("pseudogene"):
                    reasons.append("pseudogene")
                if (
                    attr.get("partial") == "true"
                    or attr.get("start_range")
                    or attr.get("end_range")
                ):
                    reasons.append("partial")
                if p[6] not in ("+", "-") or not re.fullmatch(r"\w+\.\d+", key[0]):
                    reasons.append("ambiguous_genomic_coordinates")
                if role != RNA and not re.fullmatch(r"[A-Z]+_?\d+\.\d+", rec["accession"] or ""):
                    reasons.append("missing_versioned_protein")
                if role == "rpoC" and re.search(
                    r"\b(?:split|fragment|part\s+[12])\b", attr.get("product", ""), re.IGNORECASE
                ):
                    reasons.append("split_beta_prime")
            candidates[role].append(rec)
    return dict(candidates)


def seq_retrieval(m, chosen):
    records = {}
    wanted = {c["accession"]: locus for locus, c in chosen.items() if locus != RNA}
    if wanted:
        with gzip.open(source(m, "protein.faa.gz"), "rt") as f:
            for rec in SeqIO.parse(f, "fasta"):
                if rec.id in wanted:
                    key = wanted[rec.id]
                    if key in records:
                        raise ValueError(f"duplicate product FASTA record for {key}")
                    records[key] = str(rec.seq).upper()
    if RNA in chosen:
        c = chosen[RNA]
        location = f"{c['start']}..{c['end']}"
        if c["strand"] == "-":
            location = f"complement({location})"
        with gzip.open(source(m, "rna_from_genomic.fna.gz"), "rt") as f:
            for rec in SeqIO.parse(f, "fasta"):
                annotations = dict(re.findall(r"\[([^=\]]+)=([^\]]+)\]", rec.description))
                if (
                    annotations.get("locus_tag") != c["locus"]
                    or annotations.get("location") != location
                ):
                    continue
                if not rec.id.startswith("lcl|" + c["genomic_accession"] + "_"):
                    continue
                if RNA in records:
                    raise ValueError("duplicate rRNA locus in assembly FASTA")
                records[RNA] = str(rec.seq).upper().replace("T", "U")
    return records


def project(gene, sequence, reference):
    rna = gene == RNA
    canonical = "ACGU" if rna else "ACDEFGHIKLMNPQRSTVWY"
    allowed = canonical if rna else canonical + "BXZJUO"
    if not sequence or set(sequence) - set(allowed):
        return None, {"reason": "unsupported_sequence_symbol"}
    aligner = PairwiseAligner(mode="global")
    if rna:
        aligner.match_score, aligner.mismatch_score = 2, -1
        aligner.open_gap_score, aligner.extend_gap_score = -5, -1
        align_seq = sequence
    else:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score, aligner.extend_gap_score = -10, -0.5
        align_seq = re.sub(r"[JUO]", "X", sequence)
    alignment = aligner.align(reference, align_seq)[0]
    projected = ["-"] * len(reference)
    pairs = ident = placed = 0
    for (a, b), (x, y) in zip(*alignment.aligned, strict=True):
        projected[a:b] = sequence[x:y]
        placed += int(b - a)
        for l, r in zip(reference[a:b], sequence[x:y], strict=True):
            if l in canonical and r in canonical:
                pairs += 1
                ident += int(l == r)
    identity = ident / pairs if pairs else 0
    cov_ref = pairs / len(reference)
    cov_query = pairs / len(sequence)
    meta = {
        "identity": identity,
        "reference_coverage": cov_ref,
        "query_coverage": cov_query,
        "query_length": len(sequence),
        "inserted_query_residues": len(sequence) - placed,
        "reference_gaps": len(reference) - placed,
    }
    if identity < (0.60 if rna else 0.20) or min(cov_ref, cov_query) < (0.75 if rna else 0.65):
        return None, dict(meta, reason="alignment_gate")
    return "".join(projected), meta


def gene_screen(candidates):
    accepted = {}
    misses = {}
    for role in LOCI:
        items = candidates.get(role, [])
        if len(items) != 1:
            misses[role] = "multiple_loci" if items else "missing_locus"
        elif items[0]["reasons"]:
            misses[role] = ";".join(sorted(set(items[0]["reasons"])))
        else:
            accepted[role] = items[0]
    return accepted, misses
