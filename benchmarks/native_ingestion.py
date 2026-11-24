"""Compare eager Biopython and production indexed ingestion using generated protein alignments."""

import argparse
import gzip
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _resources import peak_resident_bytes
from Bio import AlignIO, SeqIO

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.io.alignments import open_msa


def generate(path, args, size):
    rng = random.Random(args.seed)
    alphabet = b"ACDEFGHIKLMNPQRSTVWY--.X"
    translation = bytes(alphabet[i % len(alphabet)] for i in range(256))
    opener = gzip.open if args.gzip else open
    with opener(path, "wb") as out:
        if args.format == "stockholm":
            out.write(f"# STOCKHOLM 1.0\n#=GF SQ {size}\n".encode())
            for _ in range(args.blocks):
                for row in range(size):
                    out.write(
                        f"row{row} ".encode()
                        + rng.randbytes(args.block_width).translate(translation)
                        + b"\n"
                    )
                    if row % 101 == 0:
                        out.write(f"#=GR row{row} PP ".encode() + b"9" * args.block_width + b"\n")
                out.write(b"#=GC RF " + b"x" * args.block_width + b"\n\n")
            out.write(b"//\n")
        else:
            for row in range(size):
                out.write(f">row{row} annotated\n".encode())
                for _ in range(args.blocks):
                    out.write(rng.randbytes(args.block_width).translate(translation) + b"\n")
        return out.tell()


def worker(args):
    (size,) = args.rows
    ids = [f"row{i}" for i in random.Random(args.seed).sample(range(size), args.keep)]
    before = peak_resident_bytes()
    start = time.perf_counter()
    if args.worker == "indexed":
        with (
            tempfile.TemporaryDirectory() as temp,
            open_msa(args.source, "A", "protein", temp_dir=temp) as inv,
        ):
            selected = inv.select(ids).msa
            count, width = inv.n_rows, inv.n_columns
            index_bytes = sum(f.stat().st_size for f in Path(temp).rglob("*") if f.is_file())
    else:
        opener = gzip.open if args.gzip else open
        with opener(args.source, "rt", encoding="utf-8-sig") as handle:
            records = (
                AlignIO.read(handle, "stockholm")
                if args.format == "stockholm"
                else list(SeqIO.parse(handle, "fasta"))
            )
        full = ComponentMSA(
            "A", "protein", tuple(r.id for r in records), tuple(str(r.seq) for r in records)
        )
        selected = full.take_ids(ids)
        count, width = full.n_sequences, full.n_sites
        index_bytes = 0
    seconds = time.perf_counter() - start
    digest = hashlib.sha256()
    for row_id, seq in zip(selected.ids, selected.sequences, strict=True):
        digest.update(f">{row_id}\n{seq}\n".encode())
    print(
        json.dumps(
            {
                "source_shape": [count, width],
                "selected_sha256": digest.hexdigest(),
                "seconds": seconds,
                "peak_process_bytes": peak_resident_bytes(),
                "peak_before_bytes": before,
                "index_bytes_after_select": index_bytes,
            }
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[512, 4096, 16384])
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--block-width", type=int, default=256)
    p.add_argument("--keep", type=int, default=8)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=818)
    p.add_argument("--format", choices=["stockholm", "fasta"], default="stockholm")
    p.add_argument("--gzip", action="store_true")
    p.add_argument("--worker", choices=["indexed", "eager"], help=argparse.SUPPRESS)
    p.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(
        *args.rows, args.block_width, args.blocks, args.repeat, args.keep
    ) < 1 or args.keep > min(args.rows):
        p.error("dimensions and repeats must be positive; keep cannot exceed source rows")
    if args.worker:
        worker(args)
        return
    measurements = []
    for size in args.rows:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / ("raw.sto" if args.format == "stockholm" else "raw.fa")
            if args.gzip:
                source = source.with_suffix(source.suffix + ".gz")
            plain_bytes = generate(source, args, size)
            sample = {
                "rows": size,
                "source_bytes": source.stat().st_size,
                "uncompressed_bytes": plain_bytes,
            }
            for method in ("indexed", "eager"):
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    method,
                    "--source",
                    str(source),
                    "--rows",
                    str(size),
                    "--keep",
                    str(args.keep),
                    "--seed",
                    str(args.seed),
                    "--format",
                    args.format,
                ] + (["--gzip"] if args.gzip else [])
                runs = [
                    json.loads(subprocess.check_output(cmd, text=True)) for _ in range(args.repeat)
                ]
                assert all(
                    r["source_shape"] == [size, args.blocks * args.block_width] for r in runs
                )
                assert len({r["selected_sha256"] for r in runs}) == 1
                sample[method] = {
                    key: statistics.median(r[key] for r in runs)
                    for key in (
                        "seconds",
                        "peak_process_bytes",
                        "peak_before_bytes",
                        "index_bytes_after_select",
                    )
                }
                selected_hash = runs[0]["selected_sha256"]
                if method == "eager" and sample["selected_sha256"] != selected_hash:
                    raise RuntimeError("eager and indexed selected rows disagree")
                sample["selected_sha256"] = selected_hash
            measurements.append(sample)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "blocks": args.blocks,
                "block_width": args.block_width,
                "keep": args.keep,
                "repeat": args.repeat,
                "format": args.format,
                "gzip": args.gzip,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "measurements": measurements,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
