# Triplet coevolution

Score associations between alignment columns across three matched sets of protein, DNA or RNA sequences. Each row of a match table identifies one observation across components A, B and C. The `triplet analyze` command ranks all three cross-component pairs (AB, AC and BC) using weighted mutual information (MI), with optional average product correction (APC).

## Install

Use Python 3.11 or newer. From the repository root, create a virtual environment and activate it in your shell.

| Shell | Create | Activate |
| --- | --- | --- |
| Windows PowerShell | `py -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |
| macOS or Linux | `python3 -m venv .venv` | `. .venv/bin/activate` |

```shell
python -m pip install .
triplet --version
```

For heatmaps, install `python -m pip install ".[plots]"`. For development, use `python -m pip install -e ".[dev,plots]"`, then run `python -m pytest -q` and `ruff check src tests`.

## Run the example

The [example inputs](examples/mixed/) contain eight synthetic observations and three separate reference records. The DNA alignment is in reverse row order so the example exercises pairing by identifier. Run this line from the repository root with the environment active.

```shell
triplet analyze --a examples/mixed/protein.fasta --b examples/mixed/dna.fasta --c examples/mixed/rna.fasta --b-type dna --c-type rna --matches examples/mixed/matches.tsv --a-ref prot_ref --b-ref dna_ref --c-ref rna_ref --out results/mixed
```

Expect eight matched observations, an effective depth of 8.00 and 4, 4 and 3 retained columns in A, B and C. Protein alignment column 2 contains only gaps and is removed. In `mi_ab.tsv`, the first row pairs the first protein and DNA columns. They have 1 bit of raw MI across eight observations.

Choose a new or empty output directory for each run. To generate heatmaps, use `--out results/mixed-plots --plots` in place of `--out results/mixed` after installing the plotting extra.

## Prepare your inputs

Supply one aligned FASTA or one Stockholm alignment per component. Sequences within a component must have equal aligned lengths and unique, case-sensitive IDs. For FASTA, the ID is the first whitespace-delimited token after `>`. Each component defaults to `protein`. Set `--a-type`, `--b-type` or `--c-type` to `protein`, `dna` or `rna` as needed.

The match table is a UTF-8 TSV or CSV file with the following four required columns. The extension determines the separator. Additional columns can carry annotations.

| observation_id | a_id | b_id | c_id |
| --- | --- | --- | --- |
| obs01 | prot01 | dna01 | rna01 |
| obs02 | prot02 | dna02 | rna02 |

Each observation and each component ID must occur once in its respective column. Sequence IDs must occur in the corresponding alignment. Supply curated matches for homologs you intend to compare. Unlisted sequences are excluded from scoring. Each `--*-ref` identifies an alignment sequence for coordinate reporting and may be outside the match table.

## Read the results

| File | Contents |
| --- | --- |
| `mi_ab.tsv`, `mi_ac.tsv`, `mi_bc.tsv` | Site pairs ranked within each component pair. Each row gives both scores, both sets of coordinates, the jointly observed sequence count (`n_valid`) and their summed weight (`valid_weight`). |
| `run_summary.json` | Software version, input paths and SHA-256 checksums, settings, matched and effective depth, retained columns, QC and output inventory. |
| `figures/*.png` | With `--plots`, one heatmap per component pair and a combined view. |

All table coordinates start at 1. For example, `x_col` is the position in the filtered alignment, `x_original_col` is its original alignment column and `x_reference_pos` counts non-gap residues in the chosen reference. The `y_` fields describe the second component. A gap in the reference leaves its reference position blank. Use `--require-reference` to omit such rows, and `--a-start`, `--b-start` or `--c-start` to set the first occupied reference position.

By default, each table contains up to 1,000 pairs ranked by `mi_apc`. Use `--score mi_bits` to rank by raw MI, `--fmt csv` for comma-separated tables or `--max-pairs` to change the per-table limit. A header-only table means no pair passed the selected criteria. See `triplet analyze --help` for all options.

## How scores are calculated

1. **Filter columns.** Within each matched component, columns must have at least 50% canonical symbols, at most 50% gaps and at most 50% missing symbols by default. Change these limits with `--min-coverage`, `--max-gap-fraction` and `--max-missing-fraction`. Gaps (`-` or `.`), ambiguous codes and `?` are excluded from identity and MI calculations.
2. **Weight observations.** Each observation receives the reciprocal of its number of neighbors, including itself. A neighbor must meet a minimum overlap in every component (default 50%) and a threshold for the mean of the three component identities (default 80%). The sum of the weights is the effective depth.
3. **Score site pairs.** Weighted MI uses canonical symbols observed at both sites. The default `--min-valid 2` requires two observations with data at both sites. APC subtracts a background estimate calculated from row, column and overall mean MI within each component pair. Corrected scores can be negative.

V1 starts with aligned, matched inputs. Homolog search and pairing, direct coupling analysis, three-body inference and significance testing are outside this release. Shared ancestry, sampling and indirect associations can all contribute to high scores. Use the rankings as descriptive evidence when investigating candidate associations.
