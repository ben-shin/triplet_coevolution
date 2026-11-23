# Scaling matched-row weights and permutation calibration

Run the benchmark from a checkout with the package installed. The default uses generated protein, DNA and RNA alignments. Each size contains all the observations at the previous size. Progress goes to stderr and the JSON report goes to stdout.

```shell
python benchmarks/matched_scaling.py --sizes 256 768 2048 4096 --repeats 3 --permutations 5 --seed 2026 > scaling.json
```

To time your prepared data, supply the folder containing `analyze_args.json`, that manifest itself, or its parent if the manifest lives under `prepared/`. Choose sizes up to the number of complete matches. For example, the following command benchmarks a prepared table with at least 1,000 rows and a `clade` annotation.

```shell
python benchmarks/matched_scaling.py --prepared path/to/prepared --sizes 100 300 1000 --strata-column clade --permutations 19 --repeats 3 --seed 47 > scaling.json
```

The runner uses a seeded random ordering of the match table and takes increasingly long prefixes. It restores correspondence-table order within each subset. Group labels follow the same row selection as the sequences. The runner reads the native reference rows, strand, residue offsets, molecule types, masks, weighting thresholds, minimum joint observations, reference requirement and MI score. QC runs separately on each subset, so retained columns can vary with depth. The JSON includes the retained dimensions and a SHA-256 digest of the ordered observation identifiers. The native strata column is used unless overridden. Without groups, prepared data uses unrestricted shuffling. Each calibrated subset needs at least one group with two observations.

The workload fits weighted MI/APC on all AB, AC and BC pairs and local three-body interactions on up to 16 coordinate triples. It runs the same permutation code that `triplet analyze` uses. Each draw recomputes weights on newly matched rows, refits the statistics, updates exceedance counts, and the completed run applies Benjamini–Hochberg correction. The benchmark writes its statistical tables under a temporary directory and removes them on completion. Timing outputs stay in the JSON report. Search, retrieval, evidence generation and external services are outside the benchmark.

Installed wheels include the scripts and their relative-import helpers under `share/triplet-coevolution/benchmarks` in the Python installation's data directory. Supply the path to the installed `matched_scaling.py` to run without a checkout.

## Comparing revisions

Use the same Python environment for both revisions. The baseline option takes the `src` directory of a second checkout or a local archive. The following commands use the local Git object for the measured baseline.

```shell
git worktree add --detach ../triplet-scale-baseline c511759
python benchmarks/matched_scaling.py --baseline-src ../triplet-scale-baseline/src --sizes 256 768 2048 4096 --lengths 72 30 24 --repeats 3 --permutations 5 --seed 2026 > comparison.json
```

The driver launches a fresh process for each revision, workload and size. It verifies which weighting module each baseline process loaded. It requires exact agreement on neighbor counts, weights, effective depth, the number of tests, empirical p-values and BH-adjusted values. It compares the observed scores at absolute and relative tolerances of 10⁻¹². Baselines used with a prepared manifest must support the coordinate options recorded in that manifest.

## What is timed

In the weighting workload, the matched alignment has already been validated and encoded. The timer includes packing valid-state and residue bits, every pair comparison and the final per-row weights. For permutation calibration, the original weights and MI values are available before the timer starts, as they are in the analysis CLI. The timer includes setup of reusable component counts, observed triple scores, all requested whole-component shuffles and refits, adjustment and calibration-table writes. Loading the input, generating the fixture and comparing revisions are outside the timers.

Peak memory is the process high-water mark, including imports, input storage, encoding, result tables and the timing harness. The prepared loader reads the input alignments and matching table before selecting rows, so RSS includes this input cost even for small subsets. The runner samples Windows process counters or Unix `getrusage` and reports RSS in bytes. Wall time depends on the machine, sequence lengths, missingness, null draws and other workloads. The timings below describe MI and local three-body calibration. DCA and global pseudolikelihood add their own per-draw fitting costs.

## Measured synthetic sweep

The fixture uses 64 founder triplets across 8 permutation strata. Each component redraws residues independently at 10% of positions, inserts 6% gaps and 2% ambiguous residues, and retains one canonical anchor column. The lengths are 72 protein residues, 30 DNA bases and 24 RNA bases. Every calibrated run evaluates 4,608 pair hypotheses and 16 triple hypotheses across five shuffles. Times below are medians of three runs against baseline `c511759`, on Windows 11 build 26300, AMD Ryzen 7 9800X3D (8 cores), CPython 3.13.15 and NumPy 2.5.5.

| Matched rows | Effective depth | Weights baseline → optimized (s) | Five-draw calibration baseline → optimized (s) | Weighting peak RSS baseline → optimized (MiB) | Calibration peak RSS baseline → optimized (MiB) |
|---:|---:|---:|---:|---:|---:|
| 256 | 77.702 | 0.0191 → 0.0053 | 0.6114 → 0.1442 | 36.1 → 41.3 | 42.6 → 42.6 |
| 768 | 90.562 | 0.1187 → 0.0278 | 1.2767 → 0.2892 | 36.8 → 43.1 | 43.0 → 44.2 |
| 2,048 | 88.488 | 0.7191 → 0.1519 | 4.7326 → 0.8302 | 39.4 → 45.5 | 45.0 → 68.1 |
| 4,096 | 90.074 | 2.7822 → 0.5937 | 16.3904 → 4.6522 | 43.9 → 48.4 | 47.4 → 51.8 |

Every size passed the counts, weights, effective-depth, score and calibration comparisons. At 2,048 rows the permutation cache used 24 MiB. At 4,096 rows a full cache would need 96 MiB, so the benchmark automatically switched to blocked comparisons for every shuffle. The 4,096-row run completed weighting 4.7 times faster and permutation calibration 3.5 times faster than the baseline.

## Memory and numerical behavior

Weighting packs validity and state bits into 64-column words. Bit counts recover exact integers for the joint overlap and matching canonical residues. Blocks on the upper triangle contribute to both rows' neighbor counts. Each component must satisfy the overlap criterion and the final rational mean identity is rounded once before the inclusive threshold comparison. Very long alignments use Python integer arithmetic when count products exceed exact floating-point integer representation.

Standalone weighting uses space linear in the encoded input size plus bounded block workspace. During repeated permutations the implementation can retain the per-component match and overlap **counts** in the smallest suitable unsigned integer type. The combined cache is limited to 32 MiB. When the cache would exceed that limit, the null scorer reuses packed component inputs and compares row blocks on demand. A new matching always gets its own triplet weights. MI uses separate weighted contingency tables for each site pair and batches these tables to limit temporary storage. Output score matrices and the online exceedance counters still scale with the number of hypotheses.
