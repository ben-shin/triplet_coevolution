# Held-out prediction of measured fluorescent-protein brightness

Adding third-order terms improved predictions for held-out genotypes in both published fluorescence trials. In the primary assay, pooled held-out R² rose from 0.87061 to 0.93007 and RMSE fell from 0.14352 to 0.10551 on the authors’ processed brightness scale. The second trial reproduced the direction and approximate size of the R² improvement. This experiment measures interactions among differences within one fluorescent protein. Cross-component coevolution, physical contacts and thermodynamic three-body energies require different evidence. [Poelwijk, Socolich and Ranganathan (2019)](https://doi.org/10.1038/s41467-019-12130-8) · [Registered evaluation](../benchmarks/independent_evaluation_policy.json)

| Measurement | Polynomial degree | Coefficients per fit | Held-out R² | Held-out RMSE |
|---|---:|---:|---:|---:|
| Primary | 1 | 14 | 0.13518 | 0.37105 |
| Primary | 2 | 92 | 0.87061 | 0.14352 |
| Primary | 3 | 378 | 0.93007 | 0.10551 |
| Second trial | 1 | 14 | 0.10075 | 0.53007 |
| Second trial | 2 | 92 | 0.85439 | 0.21330 |
| Second trial | 3 | 378 | 0.91737 | 0.16068 |

## What was measured

The two author workbooks each contained 8,192 distinct fixed-width binary genotypes, exactly the full 13-site combinatorial design. Supplementary Data 3, sheet `genodata`, supplies the primary outcome in column J, labeled “brightness / combined.” Supplementary Data 4 column C reproduces that outcome to within 5.0 × 10⁻¹⁰. Column E reports the second sorting and sequencing trial, which started from the same bottlenecked population. The trial-to-trial Pearson correlation was 0.98990. The unrescaled RMSE between trials was 0.22713. Differences in the two numerical response ranges affect absolute RMSE. [Supplement description](https://pmc-oa-opendata.s3.amazonaws.com/PMC6746860.1/41467_2019_12130_MOESM3_ESM.pdf) · [Primary workbook](https://pmc-oa-opendata.s3.amazonaws.com/PMC6746860.1/41467_2019_12130_MOESM7_ESM.xlsx) · [Replicate workbook](https://pmc-oa-opendata.s3.amazonaws.com/PMC6746860.1/41467_2019_12130_MOESM8_ESM.xlsx)

The source authors normalized red and blue enrichment using both parental spectral measurements, combined the channels, fitted a nonlinear brightness transform and regularized zero-brightness estimates with pseudocounts. Their primary power transform used an exponent of 0.44, chosen against low-order reconstruction on the entire factorial design. Each held-out fit here uses the published response as fixed. Its scores therefore assess genotype prediction conditional on a globally calibrated phenotype. They cannot independently validate the calibration against raw fluorescence. [Article methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746860/#Sec14)

All 8,192 processed responses were finite in both measurements. Primary values ranged from 0.15486 to 1.69245, and second-trial values ranged from 0.14177 to 2.84255. Each minimum occurred once. There were zero exact zeros on either processed scale. In primary sequencing counts, four genotypes had zero input reads, 1,331 had zero red reads, 119 had zero blue reads and 113 had zero reads in both color channels. Three had zero reads across all three columns. All four zero-input genotypes were retained as prescribed. The outcome tables do not flag which estimates were originally censored or received pseudocounts. The count tallies and minima cannot establish that number. [Full support counts](../benchmarks/experimental_fluorescence_results.json)

## Fold and variance definitions

The original registration fixed the assay, degrees and hash folds. The evaluation supplement at commit `0c09b900ea0fcc46d767a8470785af6203bd21d4` identified the processed outcome and replica before the numeric measurements were opened. The input is the printed 13-bit string in column A, in its original left-to-right order with the surrounding quotation marks removed. Each genotype’s fold is the big-endian unsigned integer SHA-256 of its ASCII bitstring modulo five. Applying that rule within each Hamming-weight stratum gives fold sizes of 1,688, 1,602, 1,654, 1,645 and 1,603. The hash determines the assignments even for the two singleton strata. The aggregate JSON includes every stratum-by-fold count and a checksum of the ordered fold vector. [Original registration](../benchmarks/experimental_protocol.json) · [Frozen analysis rules](../benchmarks/independent_evaluation_policy.json) · [Numerical record](../benchmarks/experimental_fluorescence_results.json)

The unpenalized models include the intercept and all allele products through the indicated degree, coding the printed zero allele as +1 and one as −1. There is no fitted encoder, feature selection or tuning parameter. Each of the five fits used training rows only, checked full column rank and predicted every held-out row once. Pooled R² uses total squared error over all held-out predictions and the total response sum of squares for all 8,192 genotypes. The third-degree model improved on the second-degree model in every fold. Fold-wise R² gains ranged from 0.05361 to 0.07066 in the primary measurement and from 0.05734 to 0.07517 in the second trial. The paired pooled R² gains were 0.05946 and 0.06298. Those two trial gains show empirical repeat variation. A confidence interval for population or between-system performance is unavailable from two trials sharing an initial population. The five folds also have overlapping training sets.

The full-cube Walsh calculation used equal mass on every genotype. Each coefficient is the mean of the response times its signed allele product. Its square contributes to the variance, with the constant excluded. The sum of all nonconstant contributions equaled the variance of the measured cube. The exactly third-order component comprised **0.010168 brightness² of 0.159199 total (6.387%)** in the primary assay and **0.021441 brightness² of 0.312458 (6.862%)** in the second. Orders four through thirteen contributed another 6.278% and 7.388%, respectively. These are descriptive decompositions on the specified processed scales. They include measurement variation. The predictive gains above come from separately fitted and held-out models. [All orders and fold results](../benchmarks/experimental_fluorescence_results.json)

## Replay and provenance

The replay verifies both protocol hashes, the two complete workbook SHA-256 digests, the column headers and the fixed-width genotype keys before fitting. It aligns tables by the binary labels and emits only aggregate measurements. Downloaded workbooks stay in the designated input directory. Both archive copies distributed by NCBI were byte-identical to the publisher downloads. The article carries CC BY 4.0 with exceptions for otherwise credited material. This repository contains a derived numerical analysis and attributes the source authors. [Article and licensing](https://doi.org/10.1038/s41467-019-12130-8) · [License terms](https://creativecommons.org/licenses/by/4.0/) · [Digests and versions](../benchmarks/experimental_fluorescence_results.json)

```sh
python benchmarks/experimental_fluorescence.py --download --input-dir results/phase11-fluorescence --out results/phase11-fluorescence/aggregate.json
python benchmarks/experimental_fluorescence.py --input-dir results/phase11-fluorescence --out results/phase11-fluorescence/replay.json
TRIPLET_FLUORESCENCE_ROOT=results/phase11-fluorescence python -m pytest -q tests/test_experimental_fluorescence.py
```

The public-data regression recomputes the two trial results using an independent integer-parity design, direct third-order inner products and normal-equation fits. Pooled R² and RMSE agreed with the reported least-squares values within 10⁻¹² absolute tolerance. Synthetic tests verify allele reversal, scaled factorial contrasts, prediction invariance when held-out outcomes change, duplicate and missing genotypes, incomplete phenotypes and a constant-response variance denominator.
