# Selecting rows from a Stockholm family

`triplet_coevolution.io.stockholm.open_stockholm` reads plain and gzip-compressed Stockholm 1.0 alignments. It indexes row names, aliases, domain intervals and alignment lengths on temporary disk. Iterating over `rows()` returns metadata in source order. Use `row()` for an exact alignment ID or `find()` for all rows associated with an identifier. Different occurrences of a domain in the same protein retain distinct row IDs.

```python
from pathlib import Path
from triplet_coevolution.io.stockholm import open_stockholm

with open_stockholm(Path("family.sto.gz"), family="PF00001", molecule="protein") as family:
    for row in family.find("P12345"):
        print(row.row_id, row.accession, row.start, row.end)

    # Use exact IDs established by the matching step, in observation order.
    reference = "reference/101-125"
    family.extract([reference, "homolog/25-49"], "selected.sto", reference_row_id=reference)
```

A file containing several records needs a unique family selector. Select by `#=GF ID`, versioned `#=GF AC`, or an accession without its numeric version. A selector triggers an initial scan of the file to detect duplicate matches, including in a gzip stream.

Extraction makes a second pass over the chosen record. It spools only the requested sequence fragments and their aligned annotations to disk, then writes the rows in the requested order. All source columns survive, including all-gap columns. Gap characters, lowercase letters, per-column `#=GC` tracks (including `RF`) and per-row `#=GR` tracks keep their source positions. `#=GS` annotations attached to a shared domain name are applied to each retained occurrence. The family description and other `#=GF` records are copied, the sequence count is updated, and source trees are omitted from strict subsets. Plain comments and whitespace are discarded. Output ending in `.gz` uses deterministic gzip compression.

Domain intervals are 1-based and inclusive. Their span must match the row's count of symbols other than `.` and `-`. A descending interval retains both endpoints in source order. Declaring `molecule="protein"` requires forward intervals. Missing or duplicate requested IDs, ambiguous aligned domain annotations and malformed row or feature lengths raise errors before output is replaced. For wrapped input, use a blank line between blocks, or restart each unseparated block at the first sequence of the preceding block.

Each fragment of a wrapped `#=GC` or `#=GR` track must appear in the same block as its sequence columns and have that block's width. Blocks can have different widths. A track supplied on one line can appear separately and must span the full alignment. The inventory validates these positions for all tracks, including annotations on rows that will be discarded. During extraction, callers can pass `validate_fragment(row_id, text)` to check selected sequence symbols before publication.

The Python working set scales with the longest input line, requested row IDs and optional returned reference string. Temporary disk usage scales with the row index, annotation metadata and selected sequence fragments. The caller can choose the index location with `temp_dir=`. Each extraction rescans its source record. Seeking to a later record in a gzip archive replays decompression up to that offset.

## Synthetic measurements

The following Windows measurements used Python 3.13.15. Each run generated a disposable file with 2,048 columns, then indexed it and verified the exact strings and order of three selected rows. Process peak includes the interpreter, SQLite and compression buffers. The periodic synthetic sequences compress particularly well.

| Rows | Layout | Stored bytes | Uncompressed bytes | Peak process MiB |
| ---: | --- | ---: | ---: | ---: |
| 4,000 | gzip, 4 blocks | 78,764 | 8,333,680 | 28.5 |
| 40,000 | gzip, 4 blocks | 771,613 | 83,477,681 | 30.0 |
| 40,000 | plain, 1 block | 82,310,981 | 82,310,981 | 29.8 |

Run the standalone synthetic measurement from an installation of this checkout.

```sh
python benchmarks/stockholm_subset.py --rows 4000 --blocks 4 --gzip
python benchmarks/stockholm_subset.py --rows 40000 --blocks 4 --gzip
python benchmarks/stockholm_subset.py --rows 40000 --blocks 1 --block-width 2048
```

The separate reader tests compare retained sequences, per-residue annotations, column annotations and accessions with Biopython. Because Biopython normalizes Stockholm gap notation, the tests also check the literal output characters against the source.

See the [Stockholm 1.0 format specification](https://sonnhammer.sbc.su.se/Stockholm.html).
