"""Acquire and align two registered public three-component cohorts, using sequence data only."""

import argparse
import csv
import gzip
import hashlib
import json
import platform
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import Bio
from cohort_loci import RNA, annotation_candidates, gene_screen, project, references, seq_retrieval
from cohort_sources import HERE, REG_SHA, RULE, STAGE, atomic_json, digest, make_frame, source

SYSTEMS = RULE["target_loci"]
ACQUISITION_REVISION = "cohort-loci-v2"


def hashseq(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def fetch_and_align(member, refs):
    path = STAGE / "records" / (member["assembly"] + ".json")
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["protocol"] != REG_SHA:
            raise ValueError("cached assembly protocol changed")
        if (
            cached.get("acquisition_revision") == ACQUISITION_REVISION
            and not cached.get("error")
            and not cached.get("download_errors")
        ):
            return cached
    record = {
        "member": member,
        "protocol": REG_SHA,
        "systems": {},
        "acquisition_revision": ACQUISITION_REVISION,
    }
    try:
        candidates = annotation_candidates(source(member, "genomic.gff.gz"))
        chosen, miss = gene_screen(candidates)
        possible = {s for s, genes in SYSTEMS.items() if set(genes) <= chosen.keys()}
        selected = {g: chosen[g] for name in possible for g in SYSTEMS[name]}
        seqs = {}
        for rna in (False, True):
            fetch = {g: c for g, c in selected.items() if (g == RNA) == rna}
            if not fetch:
                continue
            try:
                seqs.update(seq_retrieval(member, fetch))
            except (OSError, EOFError) as exc:
                record.setdefault("download_errors", {})["RNA" if rna else "protein"] = str(exc)
                miss.update(dict.fromkeys(fetch, "sequence_source_unavailable"))
        evidence = {}
        for gene, feature in selected.items():
            raw = seqs.get(gene)
            if raw is None:
                miss.setdefault(gene, "no_exact_source_FASTA_record")
                continue
            reference = refs[gene][1]
            projected, stats = project(gene, raw, reference)
            evidence[gene] = {**feature, "projection": stats, "raw_sha256": hashseq(raw)}
            if projected is None:
                miss[gene] = stats["reason"]
            else:
                evidence[gene].update(raw=raw, aligned=projected, aligned_sha256=hashseq(projected))
        record.update(
            counts={g: len(candidates.get(g, [])) for genes in SYSTEMS.values() for g in genes},
            selected=evidence,
            candidates=candidates,
        )
        for name, genes in SYSTEMS.items():
            failures = {g: miss[g] for g in genes if g in miss}
            record["systems"][name] = {"admitted": not failures, "reason": failures}
    except (OSError, EOFError) as exc:
        record["error"] = str(exc)
        record["systems"] = {
            s: {"admitted": False, "reason": {"source": "unavailable"}} for s in SYSTEMS
        }
    atomic_json(path, record)
    return record


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_cohort(records, name, genes, refs):
    out = STAGE / name
    out.mkdir(exist_ok=True)
    accepted = [r for r in records if r["systems"][name]["admitted"]]
    phyla = Counter(r["member"]["phylum"] for r in accepted)
    reasons = Counter()
    for r in records:
        reasons.update(f"{g} {why}" for g, why in r["systems"][name]["reason"].items())
    matches = []
    witnesses = []
    role_sequences = {}
    for comp, gene in zip("ABC", genes, strict=True):
        ref_id, reference = refs[gene]
        role_sequences[gene] = []
        with (out / f"{comp}.fasta").open("w", encoding="ascii", newline="\n") as f:
            f.write(f">{ref_id}\n{reference}\n")
            for r in accepted:
                m = r["member"]
                ev = r["selected"][gene]
                row = f"{m['assembly']}|{gene}"
                aligned = ev["aligned"]
                role_sequences[gene].append(aligned)
                f.write(f">{row}\n{aligned}\n")
                witnesses.append(
                    {
                        "assembly": m["assembly"],
                        "role": comp,
                        "gene": gene,
                        **{
                            k: ev[k]
                            for k in (
                                "locus",
                                "genomic_accession",
                                "start",
                                "end",
                                "strand",
                                "accession",
                                "raw_sha256",
                                "aligned_sha256",
                            )
                        },
                        **{k: ev["projection"][k] for k in sorted(ev["projection"])},
                    }
                )
        # Keep the reference row distinct from observations, including if its source strain occurs.
    for r in accepted:
        m = r["member"]
        matches.append(
            {
                "observation_id": m["assembly"],
                **{
                    c.lower() + "_id": f"{m['assembly']}|{g}"
                    for c, g in zip("ABC", genes, strict=True)
                },
                "phylum": m["phylum"],
                "genus": m["genus"],
                "tip": m["tip"],
            }
        )
    write_tsv(
        out / "matches.tsv",
        matches,
        ["observation_id", "a_id", "b_id", "c_id", "phylum", "genus", "tip"],
    )
    if witnesses:
        write_tsv(out / "loci.tsv", witnesses, list(witnesses[0]))
    masks = {}
    for comp, g in zip("ABC", genes, strict=True):
        letters = set("ACGU" if g == RNA else "ACDEFGHIKLMNPQRSTVWY")
        cols = []
        for i, col in enumerate(zip(*role_sequences[g], strict=True), 1):
            valid = [x for x in col if x in letters]
            if len(valid) >= 0.8 * len(accepted) and len(set(valid)) >= 2:
                cols.append(i)
        masks[comp] = {
            "gene": g,
            "reference_id": refs[g][0],
            "reference_width": len(refs[g][1]),
            "reference_sha256": hashseq(refs[g][1]),
            "retained_reference_1based": cols,
        }
    atomic_json(out / "prelabel-sites.json", masks)
    weights = [
        {
            "observation_id": m["observation_id"],
            "uniform": 1.0,
            "equal_phylum": len(accepted) / (len(phyla) * phyla[m["phylum"]]),
        }
        for m in matches
    ]
    write_tsv(out / "prelabel-weights.tsv", weights, ["observation_id", "uniform", "equal_phylum"])
    output_files = [
        "A.fasta",
        "B.fasta",
        "C.fasta",
        "matches.tsv",
        "prelabel-sites.json",
        "prelabel-weights.tsv",
    ]
    if witnesses:
        output_files.append("loci.tsv")
    hashes = {name: digest(out / name) for name in output_files}
    summary = {
        "accepted": len(accepted),
        "screened": len(records),
        "distinct_phyla": len(phyla),
        "admission_gate": len(accepted) >= 128 and len(phyla) >= 4,
        "phylum_counts": dict(phyla),
        "exclusion_reasons_nonexclusive": dict(reasons),
        "sites": {
            c: {
                "reference_width": v["reference_width"],
                "retained": len(v["retained_reference_1based"]),
            }
            for c, v in masks.items()
        },
        "files_sha256": hashes,
        "protocol_sha256": REG_SHA,
    }
    atomic_json(out / "manifest.json", summary)
    return summary


def freeze(records, frame_summary, refs):
    systems = {name: write_cohort(records, name, genes, refs) for name, genes in SYSTEMS.items()}
    audit = []
    for r in records:
        m = r["member"]
        audit.append(
            {
                "assembly": m["assembly"],
                "phylum": m["phylum"],
                "genus": m["genus"],
                **{g: r.get("counts", {}).get(g) for gs in SYSTEMS.values() for g in gs},
                **{
                    s: (
                        "admitted"
                        if r["systems"][s]["admitted"]
                        else json.dumps(r["systems"][s]["reason"], sort_keys=True)
                    )
                    for s in sorted(SYSTEMS)
                },
            }
        )
    write_tsv(STAGE / "screen.tsv", audit, list(audit[0]))
    inputs = []
    for r in records:
        m = r["member"]
        chosen, _ = gene_screen(r.get("candidates", {}))
        possible = {s for s, genes in SYSTEMS.items() if set(genes) <= chosen.keys()}
        kinds = [] if "error" in r else ["genomic.gff.gz"]
        errors = r.get("download_errors", {})
        if possible and "protein" not in errors:
            kinds.append("protein.faa.gz")
        if "ribosome" in possible and "RNA" not in errors:
            kinds.append("rna_from_genomic.fna.gz")
        prefix = m["ftp"].rstrip("/").rsplit("/", 1)[1]
        for kind in kinds:
            inputs.append((m["assembly"], STAGE / "raw" / m["assembly"] / f"{prefix}_{kind}.json"))
    refdir = STAGE / "references"
    for p in sorted([*refdir.glob("*.fasta.json"), *refdir.glob("*-metadata.json.json")]):
        inputs.append(("reference", p))
    receipts = []
    for assembly, p in inputs:
        blob = p.with_suffix("")
        d = json.loads(p.read_text(encoding="utf-8"))
        if blob.stat().st_size != d["bytes"] or digest(blob) != d["sha256"]:
            raise ValueError(f"source bytes differ from acquisition receipt: {blob}")
        receipts.append(
            {
                "assembly": assembly,
                "file": blob.name,
                "url": d["url"],
                "sha256": d["sha256"],
                "bytes": d["bytes"],
            }
        )
    write_tsv(STAGE / "sources.tsv", receipts, ["assembly", "file", "url", "sha256", "bytes"])
    info = {
        "frame": frame_summary,
        "screened": len(records),
        "sources": len(receipts),
        "source_bytes": sum(r["bytes"] for r in receipts),
        "systems": systems,
        "source_manifest_sha256": digest(STAGE / "sources.tsv"),
        "screen_sha256": digest(STAGE / "screen.tsv"),
        "reference_correspondence_sha256": digest(refdir / "correspondence.json"),
        "python": platform.python_version(),
        "biopython": Bio.__version__,
        "phase": "sequence data only",
    }
    atomic_json(STAGE / "frozen.json", info)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-frame",
        action="store_true",
        help="Use the published capped registry when the historical NCBI assembly report is unavailable.",
    )
    args = parser.parse_args()
    published = HERE / "independent_cohort_freeze.json"
    if args.frozen_frame:
        summary = json.loads(published.read_text(encoding="utf-8"))["frame"]
        if summary["preregistration"] != REG_SHA:
            raise ValueError("published frame registration differs")
        with gzip.open(HERE / "independent_cohort_frame.json.gz", "rt", encoding="utf-8") as f:
            frame = json.load(f)
        STAGE.mkdir(parents=True, exist_ok=True)
        atomic_json(STAGE / "frame.json", frame)
        if digest(STAGE / "frame.json") != summary["frame_sha256"]:
            raise ValueError("published frame digest differs")
        atomic_json(STAGE / "frame-summary.json", summary)
    else:
        frame, summary = make_frame()
    refs = references()
    (STAGE / "records").mkdir(exist_ok=True)
    records = []
    batch = RULE["sampling"]["batch"]
    for start in range(0, len(frame), batch):
        with ThreadPoolExecutor(max_workers=5) as pool:
            for offset, r in enumerate(
                pool.map(lambda m: fetch_and_align(m, refs), frame[start : start + batch]), 1
            ):
                records.append(r)
                if offset % 32 == 0:
                    counts = {s: sum(r["systems"][s]["admitted"] for r in records) for s in SYSTEMS}
                    atomic_json(
                        STAGE / "progress.json", {"screened": len(records), "accepted": counts}
                    )
                    print("screened", len(records), counts, flush=True)
        enough = True
        for s in SYSTEMS:
            accepted = [r for r in records if r["systems"][s]["admitted"]]
            enough &= len(accepted) >= 128 and len({r["member"]["phylum"] for r in accepted}) >= 4
        if enough:
            break
    result = freeze(records, summary, refs)
    if published.is_file():
        prior = json.loads(published.read_text(encoding="utf-8"))
        if {k: v for k, v in result.items() if k != "python"} != {
            k: v for k, v in prior.items() if k != "python"
        }:
            raise ValueError("results differ from the frozen source and output digests")
    print("frozen", digest(STAGE / "frozen.json"), "screened", result["screened"], flush=True)


if __name__ == "__main__":
    main()
