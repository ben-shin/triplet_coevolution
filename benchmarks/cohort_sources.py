"""Freeze the registered complete-assembly frame and cache public RefSeq annotation files."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

from triplet_coevolution.validation.tree_topology import _parse

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results/public-experimental"
STAGE = RESULTS / "independent-cohorts"
REGISTRATION = HERE / "independent_cohort_protocol.json"
REG_SHA = "8ba73f61cf525a8b5a8880837198da01edee2d8349d30ab9cdfe943d27ada577"
RULE = json.loads(REGISTRATION.read_text(encoding="utf-8"))
PUBLIC = json.loads((HERE / "experimental_sources.json").read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@cache
def registered_downloads():
    path = HERE / "independent_cohort_sources.tsv.gz"
    if not path.exists():
        return {}
    published = json.loads((HERE / "independent_cohort_freeze.json").read_text(encoding="utf-8"))
    with gzip.open(path, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != published["source_manifest_sha256"]:
            raise ValueError("published source inventory mismatch")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return {row["url"]: row["sha256"] for row in csv.DictReader(handle, delimiter="\t")}


def atomic_json(path, data):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        except BaseException:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def request_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix + ".json")
    expected = registered_downloads().get(url)
    if path.is_file() and receipt.is_file():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            old["url"] == url
            and old["sha256"] == digest(path)
            and expected in (None, old["sha256"])
        ):
            return path
        raise ValueError(f"cached source mismatch for {path}")
    for attempt in range(4):
        tmp = None
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "triplet-coevolution public benchmark/1.0"}
            )
            with (
                urllib.request.urlopen(req, timeout=40) as stream,
                tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as dst,
            ):
                tmp = Path(dst.name)
                headers = dict(stream.headers)
                for chunk in iter(lambda: stream.read(1024 * 256), b""):
                    dst.write(chunk)
            if not tmp.stat().st_size:
                raise OSError("empty response")
            if path.suffix == ".gz":
                with gzip.open(tmp, "rb") as handle:
                    for _ in iter(lambda: handle.read(1024 * 512), b""):
                        pass
            record = {
                "url": url,
                "sha256": digest(tmp),
                "bytes": tmp.stat().st_size,
                "last_modified": headers.get("Last-Modified"),
                "etag": headers.get("ETag"),
            }
            if expected is not None and expected != record["sha256"]:
                raise ValueError(f"download differs from frozen source: {url}")
            os.replace(tmp, path)
            atomic_json(receipt, record)
            return path
        except (OSError, EOFError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 404, 410):
                raise
            if attempt == 3:
                raise
            time.sleep(1.5 * 2**attempt)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)


def source(assembly, kind):
    url = assembly["ftp"].rstrip("/")
    filename = url.rsplit("/", 1)[1] + "_" + kind
    return request_file(url + "/" + filename, STAGE / "raw" / assembly["assembly"] / filename)


def make_frame():
    if hashlib.sha256(REGISTRATION.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != REG_SHA:
        raise ValueError("cohort preregistration changed")
    sources = RESULTS / "sources"
    for name in [
        "bac120_r232.tree.gz",
        "bac120_taxonomy_r232.tsv.gz",
        "assembly_summary_refseq.txt",
    ]:
        if not (sources / name).is_file():
            request_file(PUBLIC[name]["url"], sources / name)
        if digest(sources / name) != PUBLIC[name]["sha256"]:
            raise ValueError(f"public source changed: {name}")
    with gzip.open(sources / "bac120_r232.tree.gz", "rt") as h:
        parsed = _parse(h.read())
    tips = {s for s, n in zip(parsed[2], parsed[3], strict=True) if not n}
    with gzip.open(sources / "bac120_taxonomy_r232.tsv.gz", "rt") as h:
        taxonomy = {t: lin for t, lin in csv.reader(h, delimiter="\t") if t in tips}
    groups = defaultdict(list)
    counted = Counter()
    witnessed = set()
    with (sources / "assembly_summary_refseq.txt").open(encoding="utf-8", newline="") as h:
        next(h)
        for row in csv.DictReader(h, delimiter="\t"):
            acc = row["#assembly_accession"]
            tip = "RS_" + acc
            if tip not in tips:
                continue
            if tip in witnessed:
                raise ValueError("duplicate NCBI assembly")
            witnessed.add(tip)
            phylum = next(
                (x[3:] for x in taxonomy.get(tip, "").split(";") if x.startswith("p__")), ""
            )
            if row["version_status"] != "latest":
                counted["noncurrent"] += 1
            elif row["assembly_level"] != "Complete Genome":
                counted["incomplete"] += 1
            elif (
                not phylum
                or not row["ftp_path"].startswith("https://ftp.ncbi.nlm.nih.gov/")
                or row["annotation_name"].strip() in ("", "na")
            ):
                counted["missing_phylum_annotation_or_ftp"] += 1
            else:
                groups[phylum].append(
                    {
                        "assembly": acc,
                        "tip": tip,
                        "phylum": phylum,
                        "genus": next(
                            (x[3:] for x in taxonomy[tip].split(";") if x.startswith("g__")), ""
                        ),
                        "name": row["organism_name"],
                        "ftp": row["ftp_path"],
                        "annotation": row["annotation_name"],
                    }
                )
    counted["GenBank_only"] = sum(t.startswith("GB_") for t in tips)
    counted["no_current_RefSeq_record"] = len(tips) - len(witnessed) - counted["GenBank_only"]
    seed = RULE["sampling"]["seed"]
    for members in groups.values():
        members.sort(
            key=lambda m: hashlib.sha256((seed + "/" + m["assembly"]).encode()).hexdigest()
        )
    ordered = []
    for offset in range(max(map(len, groups.values()))):
        for phylum, members in sorted(groups.items()):
            if offset < len(members):
                ordered.append(members[offset])
    STAGE.mkdir(parents=True, exist_ok=True)
    frame = ordered[: RULE["sampling"]["cap"]]
    atomic_json(STAGE / "frame.json", frame)
    summary = {
        "preregistration": REG_SHA,
        "all_tree_tips": len(tips),
        "frame_counts": dict(counted),
        "complete_eligible": len(ordered),
        "candidate_cap": len(frame),
        "eligible_phyla": len(groups),
        "sampled_phyla": len({m["phylum"] for m in frame}),
        "frame_sha256": digest(STAGE / "frame.json"),
    }
    atomic_json(STAGE / "frame-summary.json", summary)
    return frame, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefetch-first-batch", action="store_true")
    args = parser.parse_args()
    frame, summary = make_frame()
    print(json.dumps(summary), flush=True)
    if args.prefetch_first_batch:

        def grab(m):
            try:
                source(m, "genomic.gff.gz")
                return "downloaded"
            except OSError as exc:
                return f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=5) as pool:
            for i, state in enumerate(pool.map(grab, frame[: RULE["sampling"]["batch"]]), 1):
                if i % 32 == 0 or state != "downloaded":
                    print(i, state, flush=True)


if __name__ == "__main__":
    main()
