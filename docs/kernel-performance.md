# Exact kernel performance

The native weighting, batched MI and local three-body paths retain the candidate list, row order and score definitions. The separate v1 scalar MI path remains available under `--legacy`. Mean-field DCA has an optional Cholesky solver for users who can accept floating-point rank sensitivity. All measurements here used a CPU.

## What changed

**Weights.** In an equal-component identity comparison of C components, let the first component have m identical sites among d jointly canonical sites, with d positive. The largest possible final mean identity is `(m + (C − 1) × d) / (C × d)`. If at most one eighth of a block's row pairs can still reach the threshold, the remaining components are evaluated only for those pairs. Actual overlaps and identities are still required in each remaining component. The test uses an inclusive cutoff. Pruning is restricted to the existing exact integer path with bounds at or below the binary64 exact-integer limit. Dense blocks, pooled concatenated identity, low thresholds and large-integer calculations keep their dense evaluation. Rematches use their actual component row permutations. Cache and block sizes stay within the weighting preflight budget.

**Mutual information.** On NumPy 2, nucleotide–nucleotide blocks with a common positive power-of-two weight can encode each state’s observations as bits. Intersecting the masks and counting set bits produces integer joint counts. Zero-weight, missing and optional gap states follow the same masks as the histogram kernel. Multiplication by the common weight is exact. Pair tiles contain at most 131,072 unsigned 64-bit intersection words (1 MiB). The regular histogram path handles fractional mixed weights, higher alphabets, small blocks, and NumPy 1.26. Both paths use the same MI and full-interface APC reductions. [NumPy population counts](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_count.html)

**Local three-body model.** A bounded cache holds per-column canonical masks and mixed-radix state contributions. Joint counts accumulate positive input weights in original observation order. For modest state spaces the fit drops missing marginal states after building the joint table. Larger state spaces use a compact mapping of observed state codes before counting. Both approaches recover the original complete-row, observed-state prior and gauge. The cache uses at most 2 MiB of numerical array data or 4,096 columns and is discarded after a fit. Its memory estimate includes per-candidate row temporaries. Candidate selection, null exchangeability and the max-entropy estimator are independent of this optimization.

**Mean-field DCA.** To request the optional solver, install the `optimize` extra and pass a DCA policy to `analyze --dca --dca-policy mf.json`.

```json
{"schema_version": 1, "backend": "mf", "factor_solver": "cholesky"}
```

The primal solver uses a Cholesky factor of the penalized covariance and solves against identity columns. The dual solver factors the penalized observation Gram matrix, solves the lower-triangular system for the design, and forms only the required site-pair blocks. Existing dimension and element guards apply to both solvers. When diagnostics are requested, up to four deterministic right-hand sides have a relative Frobenius residual recorded with their column indices. The default `numpy_legacy` retains the former linear algebra and ranking arithmetic. An explicit Cholesky request needs SciPy for planning and fitting. Its identity travels with the DCA design for observed fits, archives and null refits. [SciPy triangular solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html) · [SciPy positive-definite solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_solve.html)

## Measurements

The baseline was `e0ad9ae` on Windows 11, AMD Ryzen 7 9800X3D, Python 3.13.15, NumPy 2.5.5 with OpenBLAS 0.3.34.237.2 and one BLAS thread. The optional solver used SciPy 1.18.1. Before editing, profiling 2,048 synthetic matched observations with widths 96/64/48 found 0.184 s in weighting, 0.124 s in MI, 0.261 s in 1,024 triple fits and 0.919 s in joint DCA. Sorting appeared 3,072 times in the triple fits and the dual DCA general solve accounted for 0.324 s. Those instrumented durations guided implementation. The comparisons below use separate timing runs.

Each measurement is the median of three paired repetitions in fresh processes, alternating baseline/candidate order. Inputs, seed 93, reference mappings, gap patterns, 64 source genotypes, weights and candidate coordinates were fixed. Timed steps start after fixture loading and encoding. Peak RSS includes those steps and interpreter imports. The mixed fixture has a 20-state protein, 4-state DNA and 4-state RNA. Full MI includes 13,824 intercomponent pairs. Work units for weighting are distinct unordered row pairs, for MI and DCA are site pairs, and for local three-body fits are candidate triples.

| Rows | Kernel | Baseline s | Current s | Speed ratio | Current units/s | Peak RSS baseline / current MiB |
|---:|---|---:|---:|---:|---:|---:|
| 256 | Weights | 0.0057 | 0.0024 | 2.42× | 13.8 M | 40.2 / 38.4 |
| 2,048 | Weights | 0.1685 | 0.0609 | 2.77× | 34.4 M | 44.6 / 42.1 |
| 4,096 | Weights | 0.5301 | 0.1774 | 2.99× | 47.3 M | 48.9 / 46.9 |
| 2,048 | Full MI, identity weights | 0.1190 | 0.1179 | 1.01× | 117,210 | 44.8 / 41.6 |
| 4,096 | Full MI, identity weights | 0.2142 | 0.2135 | 1.00× | 64,743 | 48.0 / 47.0 |
| 2,048 | Full MI, unit/zero weights | 0.1231 | 0.1028 | 1.20× | 134,424 | 44.8 / 41.8 |
| 4,096 | Full MI, unit/zero weights | 0.2182 | 0.1880 | 1.16× | 73,527 | 48.4 / 46.8 |
| 512 | DNA–RNA MI, unit/zero weights | 0.0087 | 0.0030 | 2.91× | 1.02 M | 41.4 / 39.2 |
| 2,048 | DNA–RNA MI, unit/zero weights | 0.0257 | 0.0071 | 3.63× | 433,843 | 44.8 / 42.0 |
| 4,096 | DNA–RNA MI, unit/zero weights | 0.0487 | 0.0151 | 3.23× | 203,750 | 47.5 / 47.1 |
| 256 | Local triples | 0.1031 | 0.0724 | 1.43× | 14,153 | 40.0 / 38.1 |
| 2,048 | Local triples | 0.2128 | 0.0858 | 2.48× | 11,930 | 44.7 / 41.6 |
| 4,096 | Local triples | 0.3310 | 0.0991 | 3.34× | 10,332 | 48.5 / 46.9 |

Complete default score and support arrays were byte-identical, including APC and nonfinite cells. Coordinate and candidate digests matched. The 4/4/20 layout at 2,048 rows gave 2.64× for weights, 1.04× for identity-weighted full MI and 2.58× for triples. On a dense panel of 2,048 identical genotypes, weighting took 0.1750 s versus 0.1683 s, a 4% slowdown. The 128-row legacy check reproduced every scalar MI and mean-field DCA output byte. A seven-draw within-stratum matching calibration at 512 and 2,048 rows refitted weights on every draw. Exceedance-derived p and BH q arrays matched exactly, with full calibration times of 0.133 versus 0.155 s and 0.638 versus 0.749 s respectively.

The supplied GatABC prepared panel gave 888 matched rows, effective depth 505.491128 and retained widths 480/474/94. Weights took 0.0310 s versus 0.1346 s (4.34×). All 317,196 raw/APC MI cells took 2.255 versus 2.382 s (1.06×). The 1,024 prespecified local fits took 0.1214 versus 0.1587 s (1.31×). Complete score/support, candidates and original/reference coordinate digests agreed. Peak RSS for the three jobs ranged from 70.8 to 70.9 MiB in each version, including prepared input loading. This was a performance and regression measurement on the existing development panel. [Panel provenance](../benchmarks/gatabc-regression.md)

| Rows | DCA system | Default s | Cholesky s | Peak RSS default / Cholesky MiB | Largest raw/APC absolute difference |
|---:|---|---:|---:|---:|---:|
| 256 | Dual | 0.1752 | 0.1618 | 74.0 / 69.9 | 6.11 × 10⁻¹⁶ |
| 2,048 | Dual | 0.8416 | 0.5147 | 274.7 / 200.0 | 9.99 × 10⁻¹⁶ |
| 4,096 | Primal | 0.9853 | 0.9244 | 246.4 / 294.9 | 2.39 × 10⁻¹⁵ |

Support and six complete raw/APC rankings agreed on those three variable-site cases. A separate 80-row test deliberately duplicated and complemented binary columns. A maximum difference of 3.73 × 10⁻¹⁶ moved 30/30, 30/35 and 32/29 of 36 raw/APC rank positions on AB, AC and BC. This is why changing the factorization is an explicit numerical choice. The primal example also required more peak memory on this machine.

## Reproduce

```console
python benchmarks/kernel_performance.py --rows 256 2048 4096 --repeat 3
python benchmarks/kernel_performance.py --rows 512 2048 4096 --repeat 3 --jobs mi_nucleotides
python benchmarks/kernel_performance.py --rows 2048 --layout 4/4/20 --repeat 3
python benchmarks/kernel_performance.py --rows 2048 --dense --repeat 3 --jobs weights mi_unit triples
python benchmarks/kernel_performance.py --rows 128 --widths 11 9 7 --repeat 3 --jobs mi_legacy dca triples
python benchmarks/kernel_performance.py --rows 256 2048 4096 --repeat 3 --jobs dca_cholesky
python benchmarks/kernel_performance.py --rows 80 --seed 938 --repeat 3 --jobs dca_cholesky --tied-dca
python benchmarks/kernel_performance.py --rows 888 --repeat 3 --prepared-v1 PATH_TO_PREPARED --jobs weights mi triples
```

The driver reads a frozen Git archive locally or accepts `--reference-src PATH_TO_SRC`. It reports stage times, RSS, observed work and SHA-256 digests on stdout. It checks the entire numerical result for the default kernels. For optional DCA it checks exact supports, numerical tolerances and changed rank positions. The v1 prepared option accepts aligned FASTA and v1 threshold/match arguments. Use native analysis replay for projects with additional QC and model policies.
