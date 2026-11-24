# Plan and replay a matched analysis

Use the same input files and options for planning and scoring. From the repository root, inspect the bundled protein, DNA and RNA panel with this command.

```sh
triplet plan --a examples/mixed/protein.fasta --b examples/mixed/dna.fasta --c examples/mixed/rna.fasta --a-ref prot_ref --b-ref dna_ref --c-ref rna_ref --b-type dna --c-type rna --matches examples/mixed/matches.tsv --dca --threebody --permutations 19 --memory-mib 64 --out results/mixed-plan
```

The example starts with eight matched observations and retains 4, 4 and 3 columns. The 40 eligible pair coordinates comprise 16 AB, 12 AC and 12 BC pairs. The triple grid has 48 cells. With 19 draws the smallest attainable permutation p-value is 0.05. Change the command from `triplet plan` to `triplet analyze` and use a fresh output directory to score these inputs. `triplet analyze --dry-run` runs the same preflight. The Python functions `triplet_coevolution.cli.plan` and `analyze` accept the corresponding underscore options and `pathlib.Path` inputs.

## What the plan checks

`plan.json` records the input checksums, alignment dimensions, numbered and retained pair and triple universes, eligible candidates, variable sites and observed-state dimensions, DCA solver and global parameter limits, configured score and selection settings, exchangeability counts and attainable p-value. The QC tables identify every matched row and source site, the filtering pass and the reason for each decision. For the default analysis, planning computes QC and model dimensions without computing pair identities, fitting scores or executing a permutation. Group metadata and support are reported only when available. Some scoreability decisions depend on the actual chosen weights. See the [row and site policy guide](qc.md).

The `estimates` section reports the approximate numeric workspace and score-output size, MI observation-by-pair visits, floating-point operations for forming the observed DCA covariance or Gram matrix, maximum ranked pair rows and tested triple support rows. The weighting plan identifies the block size, prospective exact identity cache, unordered row comparisons and word operations. The global model can bound a very large candidate list before allocating it. Estimates depend on the actual synchronized, masked panel and requested models. Correlated residue states and compression affect final file size. Python objects, parser buffers, output publication, BLAS-internal workspaces and HTML/benchmark serialization add to measured process memory. Thread environment variables are recorded when supplied. Each fit uses one analysis process, and linear algebra libraries may use their own threads.

Supply a positive `--memory-mib` to check the source sequence-symbol lower bound before loading the native inputs and check estimated QC and scoring workspaces before allocating them. Plans that reach their QC or compute limit exit 3 and publish a reason in `plan.json` in a new or empty directory. Bad paths and invalid option combinations exit 1, and Typer argument errors exit 2. A ready plan exits 0. To reduce an oversized input, prepare a smaller matched panel or select source/reference coordinates with `--site-panel`. Native planning still reads the three supplied alignments to compute QC. The folder workflow can stream selected rows out of much larger source Stockholm files. [Family preparation](families.md)

## Choose weights and support

| `--weight-policy` | Required inputs and resulting weights |
| --- | --- |
| `equal_component` (default) | Self-inclusive reciprocal neighbor count under the mean of the three component identities. Each component must meet the overlap threshold. |
| `component` | `--weight-component A`, `B` or `C` chooses the alignment used for neighbor counts. |
| `concatenated` | Pool canonical matches and overlaps across components, allowing longer alignments to contribute more sites. |
| `cluster_representatives` | Build connected components of the equal-component neighbor graph. One lexicographically selected observation per cluster receives unit weight. |
| `clade_balanced` | `--clade-column NAME` gives each named group total weight one. Permutations use that column by default. An explicit `--strata-column` must be nested inside the clades. |
| `supplied` | `--weight-column NAME` reads finite nonnegative weights from the match table. Weights follow their original observation IDs under the anchored permutation design. |

The `--identity-threshold` and `--min-overlap` bounds are inclusive. Use `--min-valid` for the minimum number of positive-weight canonical observations and `--min-valid-weight` for their minimum total weight. Thresholds apply before APC and ranking, and apply to each permutation refit. For the local ABC score, support is measured at the three candidate sites. The global model uses complete observations across its fitted panel. The default weight threshold is zero. A data-defined low effective weight per site is descriptive and has no universal success cutoff. [Definitions, simulation and taxon-held-out comparison](../benchmarks/weighting-policies.md)

## Outputs and replay

| File | Contents |
| --- | --- |
| `qc_summary.json`, `qc_rows.tsv`, `qc_sites.tsv` | Phase ordering, input/retained dimensions, taxonomy status, exact duplicate groups, masks, diagnostics and exclusion reasons, including original and biological coordinates. |
| `weights.tsv` | Observation ID, final weight, neighbor count where applicable and any representative or clade. |
| `site_weight_support.tsv` | Counts of positive-weight canonical rows, canonical/gap/missing mass and canonical Kish count for every input column. |
| `scores/score_manifest.json` | Checksummed complete pair archives, fitted observation IDs, weight-table digest, QC policy and thresholds. |
| `threebody_support.tsv` or `.csv` | Support, threshold flags, occupancy and finite-score status for each requested triple when three-body scoring is enabled. |
| `analysis_replay.json` | Exact options for the native analysis, including input/panel/QC paths. Generated sibling inputs are relative to the analysis directory. |

The offline report lists input versus retained depth, sum of weights, Kish effective count and links to the diagnostics and plan. To repeat a fit, run this from the directory containing `analysis_replay.json`. Keep source files and sibling workflow directories in their recorded locations and use a fresh output.

```sh
python -c "import json, pathlib, subprocess; a=json.loads(pathlib.Path('analysis_replay.json').read_text(encoding='utf-8')); subprocess.run(['triplet', *a['arguments'], '--out', '../replayed-analysis'], check=True)"
```
