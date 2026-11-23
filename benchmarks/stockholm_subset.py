"""Measure identifier indexing and three-row extraction on deterministic synthetic Stockholm."""

import argparse
import gzip
import json
import tempfile
import time
from pathlib import Path

from _resources import peak_resident_bytes

from triplet_coevolution.io.stockholm import open_stockholm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--block-width", type=int, default=512)
    parser.add_argument("--gzip", action="store_true")
    args = parser.parse_args()
    if args.rows < 3 or min(args.blocks, args.block_width) < 1:
        parser.error("supply at least three rows and positive block dimensions")
    pattern = "ACGTacgt.-"
    repeated = pattern * (args.block_width // len(pattern) + 2)
    fragments = [repeated[i : i + args.block_width] for i in range(len(pattern))]
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "source.sto"
        selected = Path(scratch) / "selected.sto"
        opener = gzip.open if args.gzip else open
        with opener(source, "wb") as out:
            out.write(f"# STOCKHOLM 1.0\n#=GF SQ {args.rows}\n".encode())
            for block in range(args.blocks):
                for row in range(args.rows):
                    frag = fragments[(row + block) % len(fragments)]
                    out.write(f"row{row} {frag}\n".encode())
                out.write(("#=GC RF " + "x" * args.block_width + "\n\n").encode())
            out.write(b"//\n")
            plain_bytes = out.tell()
        baseline = peak_resident_bytes()
        start = time.perf_counter()
        picked = [args.rows - 1, 0, args.rows // 2]
        names = [f"row{i}" for i in picked]
        with open_stockholm(source) as inv:
            assert (inv.n_rows, inv.n_columns) == (args.rows, args.blocks * args.block_width)
            inv.extract(names, selected)
        seconds = time.perf_counter() - start
        sequences = [
            line.split() for line in selected.read_text().splitlines() if line.startswith("row")
        ]
        assert sequences == [
            [name, "".join(fragments[(row + b) % len(fragments)] for b in range(args.blocks))]
            for name, row in zip(names, picked, strict=True)
        ]
        print(
            json.dumps(
                {
                    **vars(args),
                    "input_bytes": source.stat().st_size,
                    "uncompressed_bytes": plain_bytes,
                    "peak_before_index_bytes": baseline,
                    "peak_process_bytes": peak_resident_bytes(),
                    "seconds_index_and_extract": round(seconds, 3),
                    "selected": names,
                }
            )
        )


if __name__ == "__main__":
    main()
