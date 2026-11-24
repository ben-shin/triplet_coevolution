"""Check registered public cohort FASTAs, annotation witnesses and masks against source records."""

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from urllib.parse import unquote

from Bio import SeqIO
from cohort_sources import STAGE, digest


def rows(path):
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def seqhash(s):
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def main():
    freeze = json.loads((STAGE / "frozen.json").read_text())
    expected = defaultdict(dict)
    for name, summary in freeze["systems"].items():
        out = STAGE / name
        for f, sha in summary["files_sha256"].items():
            assert digest(out / f) == sha, f
        matches = rows(out / "matches.tsv")
        assert len({m["observation_id"] for m in matches}) == len(matches) == summary["accepted"]
        assert all(m["tip"] == "RS_" + m["observation_id"] for m in matches)
        phyla = Counter(m["phylum"] for m in matches)
        assert dict(phyla) == summary["phylum_counts"]
        for w, m in zip(rows(out / "prelabel-weights.tsv"), matches, strict=True):
            assert w["observation_id"] == m["observation_id"] and float(w["uniform"]) == 1.0
            assert float(w["equal_phylum"]) == len(matches) / (len(phyla) * phyla[m["phylum"]])
        masks = json.loads((out / "prelabel-sites.json").read_text())
        loci = rows(out / "loci.tsv")
        assert len(loci) == 3 * len(matches)
        for role, mask in masks.items():
            component = list(SeqIO.parse(out / f"{role}.fasta", "fasta"))
            assert component[0].id == mask["reference_id"]
            assert seqhash(str(component[0].seq)) == mask["reference_sha256"]
            assert all(len(r) == mask["reference_width"] for r in component)
            witnesses = [w for w in loci if w["role"] == role]
            for obs, r, w in zip(matches, component[1:], witnesses, strict=True):
                assert r.id == obs[role.lower() + "_id"] and w["assembly"] == obs["observation_id"]
                assert seqhash(str(r.seq)) == w["aligned_sha256"]
                expected[w["assembly"]][w["gene"]] = w
            canon = set("ACGU" if mask["gene"] == "5S" else "ACDEFGHIKLMNPQRSTVWY")
            positions = []
            for i, col in enumerate(zip(*(str(r.seq) for r in component[1:]), strict=True), 1):
                if 5 * sum(c in canon for c in col) >= 4 * len(col) and len(set(col) & canon) >= 2:
                    positions.append(i)
            assert positions == mask["retained_reference_1based"], (name, role)

    # Re-read the accepted source records independently of the candidate-screening parser.
    for assembly, by_gene in expected.items():
        raw = STAGE / "raw" / assembly
        proteins = {w["accession"]: w for gene, w in by_gene.items() if gene != "5S"}
        with gzip.open(next(raw.glob("*_protein.faa.gz")), "rt") as h:
            observed = Counter()
            for r in SeqIO.parse(h, "fasta"):
                if r.id in proteins:
                    assert seqhash(str(r.seq).upper()) == proteins[r.id]["raw_sha256"]
                    observed[r.id] += 1
        assert observed == dict.fromkeys(proteins, 1), assembly
        gff_seen = Counter()
        rrna_intervals = set()
        with gzip.open(next(raw.glob("*_genomic.gff.gz")), "rt", encoding="utf-8") as h:
            for line in h:
                if "\tCDS\t" not in line and "\trRNA\t" not in line:
                    continue
                acc, _, kind, start, end, _, strand, _, attributes = line.rstrip().split("\t")
                attr = {
                    k: unquote(v)
                    for k, sep, v in (s.partition("=") for s in attributes.split(";"))
                    if sep
                }
                coord = (acc, start, end, strand)
                if kind == "rRNA" and (
                    "RFAM:RF00001" in attr.get("Dbxref", "").split(",")
                    or attr.get("product", "").lower() in {"5s rrna", "5s ribosomal rna"}
                ):
                    rrna_intervals.add(coord)
                w = proteins.get(attr.get("protein_id"))
                if w is not None and coord == (
                    w["genomic_accession"],
                    w["start"],
                    w["end"],
                    w["strand"],
                ):
                    assert attr.get("locus_tag") == w["locus"]
                    gff_seen[w["accession"]] += 1
        assert gff_seen == dict.fromkeys(proteins, 1), assembly
        if "5S" in by_gene:
            w = by_gene["5S"]
            assert rrna_intervals == {
                (w["genomic_accession"], w["start"], w["end"], w["strand"])
            }, assembly
            with gzip.open(next(raw.glob("*_rna_from_genomic.fna.gz")), "rt") as h:
                rrna = [
                    r
                    for r in SeqIO.parse(h, "fasta")
                    if f"[locus_tag={w['locus']}]" in r.description
                    and r.id.startswith("lcl|" + w["genomic_accession"] + "_")
                ]
            assert (
                len(rrna) == 1
                and seqhash(str(rrna[0].seq).upper().replace("T", "U")) == w["raw_sha256"]
            )
    print(
        f"Verified {len(expected)} assemblies, {sum(map(len, expected.values()))} admitted loci and six aligned components."
    )


if __name__ == "__main__":
    main()
