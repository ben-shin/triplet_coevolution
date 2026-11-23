# Triplet coevolution

Search local sequence collections, build alignments, match homologs and measure coevolution between three protein, DNA or RNA components. The package scores all three pairs (AB, AC and BC) and can fit interactions that involve one site from each component (ABC).

| Start with | Command | Result |
| --- | --- | --- |
| Three named folders containing Stockholm families, including gzip-compressed files | `triplet run` | Matching audit, synchronized observations, automatic scoring and permutation benchmarks when supported, and an offline report |
| Three family folders with explicitly specified orthology and preparation settings | `triplet evidence` | Detailed candidate evidence, synchronized rows and a scoring handoff |
| Three local search databases, seeds or profiles, references and pairing evidence | `triplet run --config …` | Search hits, retrieved sequences, alignments, a pairing audit and the complete analysis |
| Three unpaired alignments and genome or orthology evidence | `triplet prepare` | Matched observations, QC and inputs for `analyze` |
| Three aligned components and a curated match table | `triplet analyze` | Ranked scores, benchmarks, a site network and an offline HTML report |

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

For PNG heatmaps, install `python -m pip install ".[plots]"`. For development, use `python -m pip install -e ".[dev,plots]"`, then run `python -m pytest -q` and `ruff check src tests`. The examples use local files. Additional search programs have their own installation requirements, listed in the [workflow guide](docs/workflows.md).

## Run the complete workflow

If you already have three Stockholm families, put each file in a separate folder under [input](input/README.md). Name the folders after your proteins or nucleic acids and run the following command from the repository root.

```shell
triplet run
```

The command uses family and row annotations, cached records and public identifier lookups to find unambiguous matches in a shared genomic context. It determines the scoring molecules and coordinate representatives, then scores and benchmarks supported observations. It prints the path to an offline `report.html` under a fresh `results/run-…` directory, including when evidence is insufficient. No configuration or prompts are required. To keep the run entirely local, use `triplet run --offline`. The [automatic run guide](docs/automatic.md) covers the data needed for a match, public services, inference limits and batch exit codes. For strict preparation with explicitly supplied roles, see the [three-folder evidence guide](docs/families.md).

For a workflow starting with unaligned sequences, run the bundled database-search example from the repository root. Paths in its configuration are relative to the JSON file.

```shell
triplet run --config examples/workflow/run.json --out results/full
```

The run searches the example databases, retrieves candidates, builds alignments, uses the supplied evidence to select partners, filters and weights the matched observations, then scores and benchmarks the result. Open `results/full/analysis/report.html` to inspect search hits, excluded groups, alignment QC, site rankings and benchmarks. Choose a **new or empty output directory** for every run.

The synthetic example retains eight observations and fits all 800 eligible site triples. Two triples tie for first in the local Potts ranking. At reference positions **A2/B5/C3** and **A4/B2/C3**, the raw MI of each constituent pair is zero and the signed co-information is −1 bit. In these constructed observations, the combination of the first two sites determines the state of the third. With 19 permutations, the first triple has an empirical p-value of 0.05 and a BH-adjusted value of 1 across the 800 tested triples. The four contact labels are also synthetic.

If you already have unpaired alignments, start at the pairing step.

```shell
triplet prepare --config examples/workflow/prepare.json --out results/prepared
```

Review `results/prepared/matching_qc.json` and `matches.tsv` before scoring. The prepared directory includes `analyze_args.json` with the reference IDs, coordinate rows and arguments for replaying the handoff. For the bundled configuration, the prepared alignments connect directly to the analysis command.

```shell
triplet analyze --a results/prepared/A.fasta --b results/prepared/B.fasta --c results/prepared/C.fasta --matches results/prepared/matches.tsv --a-ref prot_ref --b-ref dna_ref --c-ref rna_ref --b-type dna --c-type rna --dca --threebody --out results/from-prepared
```

See the [workflow guide](docs/workflows.md) for the component definitions, evidence schema, hit thresholds, ambiguity policy, expected output files and external-tool alternatives.

## Analyze curated matches

The [example inputs](examples/mixed/) contain eight synthetic observations and three separate reference records. The DNA alignment is in reverse row order so the example exercises pairing by identifier. Run this line from the repository root with the environment active.

```shell
triplet analyze --a examples/mixed/protein.fasta --b examples/mixed/dna.fasta --c examples/mixed/rna.fasta --b-type dna --c-type rna --matches examples/mixed/matches.tsv --a-ref prot_ref --b-ref dna_ref --c-ref rna_ref --out results/mixed
```

Expect eight matched observations, an effective depth of 8.00 and 4, 4 and 3 retained columns in A, B and C. Protein alignment column 2 contains only gaps and is removed. In `mi_ab.tsv`, the first row pairs the first protein and DNA columns. They have 1 bit of raw MI across eight observations.

To add mean-field DCA, a three-body fit and stratification-free permutation tests, append `--dca --threebody --permutations 19 --seed 7` to the example and use `--out results/mixed-models`. Nineteen draws are enough to exercise the workflow, with limited p-value resolution. To generate PNG heatmaps, add `--plots` after installing the plotting extra.

## Prepare your inputs

Supply one aligned FASTA or one Stockholm alignment per component. Sequences within a component must have equal aligned lengths and unique, case-sensitive IDs. For FASTA, the ID is the first whitespace-delimited token after `>`. Each component defaults to `protein`. Set `--a-type`, `--b-type` or `--c-type` to `protein`, `dna` or `rna` as needed.

The match table is a UTF-8 TSV or CSV file with the following four required columns. The extension determines the separator. Additional columns can carry annotations.

| observation_id | a_id | b_id | c_id |
| --- | --- | --- | --- |
| obs01 | prot01 | dna01 | rna01 |
| obs02 | prot02 | dna02 | rna02 |

Each observation and each component ID must occur once in its respective column. Sequence IDs must occur in the corresponding alignment. Supply curated matches for homologs you intend to compare. Unlisted sequences are excluded from scoring. Each `--*-ref` identifies the biological reference used in reported coordinates and contact labels. By default the alignment contains a row with that ID. If a full coordinate row has a different ID, supply `--a-ref-row`, `--b-ref-row` or `--c-ref-row`. A matched, trimmed hit can then retain its original identifier while the full reference defines the coordinates.

## Read the results

| File | Contents |
| --- | --- |
| `mi_ab.tsv`, `mi_ac.tsv`, `mi_bc.tsv` | Site pairs ranked within each component pair. Each row gives both scores, both sets of coordinates, the jointly observed sequence count (`n_valid`) and their summed weight (`valid_weight`). |
| `run_summary.json` | Software version, input paths and SHA-256 checksums, settings, matched and effective depth, retained columns, QC and output inventory. |
| `report.html` | Offline report with alignment composition, leading sites, model and benchmark results, settings and links to the full outputs. |
| `network.graphml` | Site network for graph viewers such as Cytoscape. Includes up to 25 results per exported ranking. Each scored triple is a separate node with three membership links. |
| `dca_*.tsv`, `threebody_abc.tsv` | With the corresponding model enabled, ranked pair couplings or three-site Potts scores and signed co-information, with coordinates and observation support. |
| `permutation_*.tsv`, `contacts.json` | With benchmarks enabled, empirical p-values and BH-adjusted values, or performance against supplied positive and negative reference contacts. |
| `figures/*.png` | With `--plots`, one heatmap per component pair and a combined view. |

All table coordinates start at 1. For example, `x_col` is the position in the filtered alignment, `x_original_col` is its original alignment column and `x_reference_pos` counts non-gap residues in the chosen reference. The `y_` fields describe the second component. A gap in the reference leaves its reference position blank. Use `--require-reference` to omit such rows, and `--a-start`, `--b-start` or `--c-start` to set the first occupied reference position.

By default, each table contains up to 1,000 pairs ranked by `mi_apc`. Use `--score mi_bits` to rank by raw MI, `--fmt csv` for comma-separated tables or `--max-pairs` to change the per-table limit. A header-only table means no pair passed the selected criteria. See `triplet analyze --help` for all options.

## How scores are calculated

1. **Filter columns.** Within each matched component, columns must have at least 50% canonical symbols, at most 50% gaps and at most 50% missing symbols by default. Change these limits with `--min-coverage`, `--max-gap-fraction` and `--max-missing-fraction`. Gaps (`-` or `.`), ambiguous codes and `?` are excluded from identity and MI calculations.
2. **Weight observations.** Each observation receives the reciprocal of its number of neighbors, including itself. A neighbor must meet a minimum overlap in every component (default 50%) and a threshold for the mean of the three component identities (default 80%). The sum of the weights is the effective depth.
3. **Score site pairs.** Weighted MI uses canonical symbols observed at both sites. The default `--min-valid 2` requires two observations with data at both sites. APC subtracts a background estimate calculated from row, column and overall mean MI within each component pair. Corrected scores can be negative.

4. **Fit optional Potts models.** Use `--dca` for regularized mean-field pairwise couplings and `--threebody` for local three-site fits. For a bounded panel, `--threebody-model global` with `--threebody` fits fields, pair terms and selected three-site terms jointly. Supply an explicit ABC candidate table with `--triplet-candidates` when screening a subset. Ranks from different methods have different scales.
5. **Benchmark signal.** Use `--permutations` and `--seed` for reproducible null draws, optionally restricted by a match-table annotation named with `--strata-column`. Supply `--contacts` for a comparison against labeled reference-coordinate contacts. Shared ancestry and indirect associations can contribute to high scores, and sparse observations can destabilize estimates. Interpret rankings alongside depth, coverage and benchmark resolution.

The [methods guide](docs/methods.md) describes statistical assumptions, regularization, candidate selection, null hypotheses, multiple testing, score interpretation and computational limits.

The [performance guide](docs/performance.md) provides reproducible synthetic benchmarks for weighting and permutation calibration. The [Stockholm reader guide](docs/stockholm.md) measures extraction memory as family size increases. Neither benchmark runs during installation or preparation.
