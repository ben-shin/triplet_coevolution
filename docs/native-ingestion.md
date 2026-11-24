# Native alignment ingestion

Native analysis resolves exact, case-sensitive IDs from the match table and includes the coordinate reference for each component. Evidence-based preparation checks candidate presence against each full inventory before choosing among candidates. A second copy of a domain therefore remains visible to the matching policy even when it will be excluded from scoring. Preparation writes a selected alignment in its native format. Its replay arguments point to that file.

## Formats and coordinates

| Input | Scoring columns | Selected derivative |
|---|---|---|
| Aligned FASTA | Every sequence column. Lowercase letters denote the same state as uppercase. Both gap symbols occupy a column. | Requested records in order, original residue text and FASTA titles, wrapped at 80 symbols. |
| Stockholm 1.0 | Every original alignment column, including non-RF columns. Scoring uses uppercase and normalizes dot gaps to hyphens, matching the historical Biopython reader. | Stitched sequence tracks keep the original case and dot gaps. Global per-column features including RF and SS_cons retain their full width. Selected per-row GS and GR annotations retain their content. GF SQ records the new count. Subsetting drops GF NH/TN trees referring to the full family. |
| A3M | Uppercase letters and deletion hyphens define the shared match columns. A question mark denotes an unknown occupied match state. Lowercase letters consume a residue in that row. A dot pads an insertion. Both are outside the scoring grid. | Raw sequence records retain the lowercase residues and padding. Scoring positions use 1-based match columns. Biological residue numbers advance across intervening lowercase residues in either numbering direction. |

For example, the A3M reference `tT.-CaGg` has four match columns `T-CG`. With reverse numbering starting at 30, their reference positions are 29, gap, 28 and 26. The leading lowercase residue occupies position 30. Unique physical character offsets in a raw A3M row depend on its insertions, so a column panel expressed in original alignment coordinates selects **match columns**. Complete raw sequence records in the derivative allow replay.

The A3M reader recognizes the HH-suite display records `ss_pred`, `ss_conf`, `ss_dssp`, `sa_dssp` and `ss_cons`. It checks their match widths and records their names in selection provenance. It omits these query-specific displays from the selected alignment, whose first row can differ from the original query. Recognized unsupported display/consensus conventions produce an input error. Concatenated multimer A3M files with a chain-length header require separate per-component inputs. [HH-suite A3M conventions](https://github.com/soedinglab/hh-suite/blob/master/scripts/reformat.pl) · [HH-suite display records](https://github.com/soedinglab/hh-suite/blob/master/src/hhalignment.cpp) · [Stockholm format specification](https://sonnhammer.sbc.su.se/Stockholm.html)

The native reader accepts a single Stockholm record, including interleaved blocks. The lower-level family-database reader also supports an explicit family ID or accession. A numeric `/first-last` domain suffix records the 1-based inclusive source interval. Its length must agree with the occupied residues. Descending intervals require a nucleotide component. Preparation and analysis retain the configured reference ID, first occupied position and strand as the numbering authority. Distinct domains sharing an accession keep separate exact row IDs.

## Validation, memory and replay

The source filename identifies the format, including `.a3m`, `.sto.gz` and `.fasta.gz`. An explicit format overrides the extension. For example, `open_msa(path, "C", "rna", fmt="a3m")` reads compressed or uncompressed A3M. The gzip magic bytes determine decompression. Readers accept a UTF-8 byte-order mark. Format and molecule type are separate inputs.

FASTA and A3M scans write exact IDs, aliases, domain intervals and byte offsets to temporary SQLite storage. The first pass validates widths, identifiers and molecule-specific characters on every row. Selected offsets are read in increasing order, including in gzip streams. Stockholm uses a disk-backed fragment and annotation inventory. Its second pass reassembles the requested sequences and tracks. Source signatures are checked around selection. Invalid records, changed sources, write failures and interrupts close the temporary database. Successful derivative publication uses an atomic replacement in the destination directory.

Indexing storage grows with identifier/annotation count. RAM depends on the largest physical input line, SQLite page caches and selected material, including any A3M insertions. Gzip streams can require a second decompression pass. The complete matched scoring alignment and downstream matrices still need their own budgets. A caller that explicitly requests all IDs through `read_msa` materializes the entire family. With an inventory, `inv.ids()` and `inv.rows()` stream original-order metadata, `inv.find(selector)` yields every domain/alias hit, and `inv.contains(row_id)` checks an exact ID. Resolve ambiguous selectors before calling `select`.

```python
from triplet_coevolution.io.alignments import open_msa

with open_msa("component.sto.gz", "A", "protein") as inventory:
    selected = inventory.select(["sample/5-20", "reference"], output="A.sto")
    scoring_alignment = selected.msa
    coordinates = selected.reference_map("reference", start=1)
```

## Measured native ingestion

The rerunnable benchmark uses four interleaved Stockholm blocks with deterministic variable amino acids, gaps and unknowns, a full RF track and a GR track on every 101st row. Eight sampled rows are read in a seeded order. Each method runs in a fresh process. The eager comparison reproduces the `4cf141a` Biopython parser and full `ComponentMSA` validation, with gzip decompression added for a like-for-like input. Both methods must produce the same ordered sequence digest. Data generation and process startup are outside the timed interval. Peak RSS includes the interpreter and imported dependencies.

Windows 11, AMD Ryzen 7 9800X3D, one CPU process per measurement, Python 3.13.15, Biopython 1.86, three repeats per method, seed 818. Values are medians. Throughput is the decompressed source size divided by the time for indexing, validation and extraction.

| Source rows × columns | Gzip MiB | Indexed seconds | Eager seconds | Indexed MiB/s | Eager MiB/s | Indexed peak MiB | Eager peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 × 1,024 | 0.32 | 0.063 | 0.038 | 8.34 | 13.79 | 43.93 | 44.70 |
| 4,096 × 1,024 | 2.58 | 0.197 | 0.345 | 21.17 | 12.12 | 44.68 | 58.82 |
| 16,384 × 1,024 | 10.33 | 0.668 | 1.304 | 25.09 | 12.85 | 46.42 | 106.00 |
| 32,768 × 4,096 | 80.61 | 3.256 | 10.308 | 40.08 | 12.66 | 46.32 | 483.46 |

At the largest size the uncompressed source occupied 130.50 MiB. Its eight selected rows had SHA-256 `61e03e705ecc0cecbd3caf8358c364cc2985785ba5c02386d94793124042846b` under both methods. The final temporary SQLite file occupied 4,628,480 bytes. The small case shows the fixed cost of indexing. Larger runs gave lower peak process memory and shorter wall times on this host.

```console
python benchmarks/native_ingestion.py --gzip --rows 512 4096 16384 --repeat 3
python benchmarks/native_ingestion.py --gzip --rows 32768 --block-width 1024 --repeat 3
```

The supplied GatABC evidence and aligned FASTA files also ran through native `prepare` in a temporary directory. Each source contained 889 rows. Widths were 3,699, 2,653 and 361. All 888 groups matched. The three selected FASTA files and `matches.tsv` were byte-identical to their supplied historical artifacts. The original collection of approximately 2.3 GB of Stockholm inputs was unavailable for this measurement. [GatABC provenance](../benchmarks/gatabc-regression.md)
