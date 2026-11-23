# Unattended three-family analysis

Create exactly three named component folders inside `input/` and put one Stockholm family alignment in each. Run this command from the repository root after installing the package.

```shell
triplet run
```

The program performs discovery, matching, coordinate selection, quality control, scoring and calibration in one invocation. No interactive decisions or intermediate commands are required. At completion, open the printed `report.html` path. The landing page links to the underlying audit and, when scoring finishes, a report with site rankings, benchmarks and a GraphML site network. Each default output directory under `results/` has a UTC timestamp and unique suffix.

`--input PATH` changes the input folder. `--out PATH` selects a new or empty destination. `--offline` confines lookup to local data and snapshots. `--plots` requests heatmaps when the plotting extra is installed. Optional, pre-existing `project.json`, per-component settings, mappings and target lists follow the [family project format](families.md). The separate `run --config FILE` command accepts [raw-search configurations](workflows.md).

## What the program chooses

* The three folder names determine roles A, B and C in ascending, case-sensitive order. Family files may be plain or gzip-compressed. Byte-identical decompressed copies collapse to one source. When full and seed files carry the same family identifier, a uniquely named full file with at least as many rows takes precedence. Different families or competing versions produce an input error with the affected names. A preconfigured `alignment` or `family` selector can pin a choice.
* The workflow determines a scoring molecule from explicit settings, identified Pfam or Rfam families, molecule-specific metadata and informative residue alphabets. Ambiguous or contradictory evidence can leave the type unknown. A recognized RNA represented with T can be scored as U. The manifest records transcription, source hashes and molecule decisions. The coordinate sidecar retains the source reference symbols.
* Rows must have support for one molecule or locus per role in a common genome or assembly. Strong role assignments use orthogroups. For an identified family, unambiguous membership can support a weaker role assignment when orthogroup evidence is unavailable. Both tiers appear separately in the audit. Multiple loci, unresolved alternatives or inconsistent placements can exclude an observation. A taxon shared by three records is insufficient to establish a common genome. Family membership and coevolution scores do not demonstrate a physical complex.
* A supplied coordinate reference takes precedence. Otherwise each component uses a matched row with maximal canonical coverage. An explicit coordinate interval, then lexical row ID, break ties. Interval-bearing rows preserve their numbering and strand. Rows without intervals receive row-relative numbering.

The [automatic inference reference](automatic-inference.md) gives the precedence rules, confidence tiers and coordinate tie-breaks.

## Lookup and replay

By default the workflow can query UniProt and, for recognized nucleotide identifiers, ENA, NCBI or RNAcentral. Requests use public identifiers and relevant taxonomy or assembly accessions. Alignment contents remain local. Where supported, row-level Stockholm annotations can provide the entire placement offline, including assembly and locus metadata. Curated mappings, cached records and predeclared orthogroups are also accepted. See the [evidence guide](evidence.md) for detailed identity rules.

The workflow keeps SQLite lookup snapshots under the output's `metadata/` directory when available. Successful lookup records are reusable on later invocations through a project-specific `.triplet-cache/` next to the run directory. Use `--cache-dir PATH` to choose a common cache location across output roots. `--offline` also accepts `input/uniprot.sqlite` and `input/nucleotide.sqlite`. If an external lookup fails, a provider-failure report preserves the audit and withholds inference because missing records could conceal alternative matches.

## Scoring and size limits

The default compact-panel analysis calculates pairwise MI, ridge-regularized mean-field DCA, and local three-site Potts scores on up to 64 seeded ABC candidates. Up to 19 whole-row permutations refit the models for empirical p-values and BH-adjusted values. The smallest empirical p-value with 19 draws is 0.05. A small sample or short null run limits interpretation. A documented omission appears in the report if an optional method cannot fit, and other eligible methods can still produce scores.

The scheduler uses original alignment widths and resolved group count before masking. Let `P` be the sum of the AB, AC and BC site-pair counts, `N` the matched depth and `W` the sum of alignment widths. The default total limits are one million pair cells and four billion `N² × W` weighting cells. If an unconfigured project exceeds these limits, the program selects a reproducible, smaller scoring panel from the resolved observations and, when needed, original alignment columns. The full matching audit, alignments and reference coordinate table remain available. The report states how many observations and sites entered the bounded fit. Its permutation results apply to that selection. A project that explicitly specifies analysis settings or limits keeps its requested scope and reports a resource limit if it cannot fit. Projects can set positive integers at `analysis_limits.pair_cells` and `analysis_limits.weight_cells`.

For the bounded fit, the scheduler targets 8–128 matched observations and selects observation IDs by seeded SHA-256 rank before examining scores. It retains all columns if possible. If widths also exceed budget, it chooses evenly spaced midpoints of original alignment intervals with an equal cap per component. Native `analyze --site-panel FILE` intersects those prespecified 1-based columns with the usual QC mask. Matched rows, chosen columns, reference numbering and the exact replay arguments are recorded. The full dataset remains available even if the minimum panel cannot fit.

The automatic DCA budget requires at most 80,000 pair cells, three million depth-by-feature cells and a solver dimension up to 512. The feature estimate uses the smaller of depth and alphabet size per original column. Nineteen permutations require `P ≤ 12,000`, `N × P ≤ 800,000` and `N² × W ≤ 10 million`. Five permutations run for panels with `P ≤ 80,000`, `N × P ≤ 4 million` and `N² × W ≤ 80 million`. Larger panels skip null refits. To preconfigure a method, seed, candidate set, draw count, plots or contact benchmark, set the corresponding field in the project's `analysis` object. Explicit choices override automatic scheduling within the overall limits and the model's own solver limits. See the [methods guide](methods.md) for interpretation and parameter details.

## Batch results

| Exit | Status in `workflow_summary.json` | Meaning |
| --- | --- | --- |
| 0 | `complete` | The effective scoring plan finished. |
| 1 | `invalid_input`, `analysis_error` | A source or setting could not be used. A safe writable destination receives a report. |
| 2 | CLI usage error | The command-line arguments were invalid. |
| 3 | `no_matches`, `insufficient_data`, `resource_limit`, `partial` | The run finished with no qualifying analysis or an explicitly reported method omission. |
| 4 | `provider_failure` | A public lookup failed. Completed evidence is preserved. |

Use `metadata/evidence.tsv` for row-level identities and alternatives, `metadata/groups.tsv` for group decisions and `prepared/matches.tsv` for selected observations. The synchronized Stockholm files contain only matched rows in table order. `prepared/reference_coordinates.tsv` links columns to the coordinate row. When available, `prepared/analyze_args.json` records the native analysis arguments for the full matched dataset. An automatically reduced fit has its own `prepared/panel_analyze_args.json` and `prepared/panel_matches.tsv`. For a site subset, `prepared/panel_columns.tsv` records the selected original columns. `analysis/run_summary.json` records the completed fit and its outputs. The workflow manifest records the exact sources, inferred roles and molecules, chosen references, methods, resource estimates and terminal status.
