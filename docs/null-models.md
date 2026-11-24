# What the calibration models test

Select the control using the scientific question and the sampling design. A triplet can depart from random matching while agreeing closely with a pairwise probability model. All controls use the declared statistic and its direction. Larger connected information and larger RMS are more extreme. More negative signed co-information is more extreme.

## Random matching

| Null name | Exchangeable unit and constraint | Interpretation |
| --- | --- | --- |
| `whole_component` | Keep A anchored to observation IDs. Independently permute whole B and C sequence rows over observations. | Measures association above random cross-component correspondence. Includes every site's original sequence, missing symbols, and component row ID. |
| `within_stratum` | Perform the two permutations separately within each supplied group. | Tests residual correspondence after retaining those group memberships. Ancestry within a group can still induce association. |
| `preserve_ab` | Keep A and B fixed. Permute C, optionally within groups. | Tests a random assignment of C conditional on the original AB pairing and any blocks. |
| `preserve_ac` | Keep A and C fixed. Permute B, optionally within groups. | Tests random assignment of B conditional on AC. |
| `preserve_bc` | Keep B and C fixed. Permute A, optionally within groups. | Tests random assignment of A conditional on BC. |

The unit is an identified component sequence, so all columns from a particular molecule move together. A score on the preserved pair may remain invariant. A globally conditioned estimator or sequence-derived weights can still change when the third component moves. The preserved pair provides no independently randomized assignment of its two partners. The assignment law assumes exchangeability of the specified component rows within each permitted group. Ancestral structure, nonrandom matching, or a sampling design that violates this assumption can invalidate a tail probability. [Winkler et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)

With group sizes $n_g$ and $k$ independently shuffled components, the labeled assignment entropy is $k\sum_g\log_2(n_g!)$ bits. The metadata also reports entropy after collapsing identical sequences in each group/component, the fraction of observations in nonsingleton groups and the fraction whose sequence assignment can change. All-singleton designs raise an error. Repeated strings can yield zero sequence entropy even when row IDs can move. A digest binds ordered observation IDs to group labels.

## Local pairwise parametric bootstrap

`m2_bootstrap` tests a local, fitted, pairwise maximum-entropy model of one prespecified three-site marginal. Fit M2 to the common weighted joint rows on the frozen state space. Independently sample one categorical joint state from that fitted M2 per positive-weight complete observation. Reapply its original weight. Repeat the observed M2/M3 fit with identical support, prior, convergence thresholds and statistic. The simulated pair margins fluctuate across draws around their fitted targets. [Local model definitions](maxent.md) · [Parametric bootstrap with fitted interaction nulls](https://pmc.ncbi.nlm.nih.gov/articles/PMC2904826/)

Rows missing a required native state retain their excluded status. Grouped rare states and explicit gap states keep their original meanings. Structural zeros remain forbidden. Estimation assumes that the complete rows are iid draws from a common law conditional on the recorded observation design. Informative missingness, lineage dependence, or an outcome-derived weighting scheme require another sampling model. Uniform unit weights are supported directly. A fixed, externally specified design may attach unequal weights to observations. These weights influence the refit without being treated as independent replicate counts. Native identity-derived or clade weights cannot be transplanted into this iid bootstrap.

Only coordinate-only, externally prespecified or independently selected hypotheses can use the fixed-candidate law. When training determined the state map, its observation IDs must be disjoint from the test cohort. The caller must document prospective external selection. Selecting coordinates, the prior or the most favorable statistic on the same data changes the hypothesis family. Coordinate-local simulations do not yield a joint maximum-statistic distribution across hypotheses.

An unresolved fit, an undefined ranking statistic, empirical constant axes, or variation supplied only by the prior yield an unavailable result. The M2 bootstrap also checks that the fitted face contains at least one full $2\times2\times2$ subtable. Alternating parity mass on that subtable preserves all pair margins and changes a three-way contrast. This provides a sufficient identifiability check. More complicated structural supports can have valid contrasts without this witness and are conservatively withheld. The same eligibility checks apply on a refit. Invalid simulations remain in the requested denominator as conservative exceedances. A plug-in parametric-bootstrap tail has finite-sample approximation error, especially near support boundaries or with strong smoothing. The plus-one Monte Carlo correction accounts for simulation resolution. [Phipson and Smyth, 2010](https://arxiv.org/abs/1603.05766)

## Exact small conditioning panel

`pair_panel` fits one normalized pair-only law over the **entire retained conditioning panel**. The low-level fitter accepts 3–8 categorical columns, no more than 65,536 joint configurations, and the configured workspace cap. For each pair, the marginal target comes from the same prior-augmented count table of rows complete across the panel. Cyclic IPF projects on every pair marginal, starting from uniform mass over permitted Cartesian cells. It checks marginal residuals and the entropy/KL identity on convergence. Estimated probabilities are invariant to gauge choices for the underlying pair coefficients. The bounded enumerator also normalizes user-supplied fields and pair coefficient tables for reference calculations.

Each replicate samples independent rows from the full enumerated probability table, reapplies each missing cell, preserves the exogenous row weights, and refits the pair-only law on the original site panel. The local M2/M3 reported statistic is then refit on the available rows at its three selected sites, which can include rows missing an unrelated panel position. The sampling design assumes a common complete-data law and a fixed exogenous missingness pattern. Direct joint sampling removes burn-in, chain autocorrelation and a need for MCMC convergence diagnostics. Only a full Cartesian model with analytically forced zeros is supported for this control. Fit size limits and nonconvergence stop inference at the documented status.

Marginalizing a fourth site that couples pairwise to three others can give positive local connected information although the full generator contains only pair terms. This control asks whether local evidence is unusual under the specified full-panel pair model. A nested held-out improvement from explicit conditional triple terms is a separate estimand. Larger conditional models require a validated normalized generator, model selection and predictive refits. Ridge inverse covariance scores do not provide simulation probabilities. [Exact panel implementation](../src/triplet_coevolution/validation/pair_sampler.py) · [Known-law, exchangeability and refit tests](../tests/test_phase5_null_models.py)

## Reproducing the size and power experiment

From the checkout with the package installed, run the following synthetic experiment. The script prints JSON and does not require downloaded data.

```console
python benchmarks/null_calibration.py --depth 128 --draws 99 --repetitions 80 --seed 771095
python benchmarks/null_calibration.py --depth 128 --draws 99 --repetitions 320 --seed 771096
```

The test uses one prespecified binary-coordinate triple, unsmoothed frequencies, unit weights, an upper-tail connected-information statistic and a 5% nominal threshold. Every replicate cohort is generated anew. The baseline spin fields are $(0.2,-0.1,0.15)$ in natural-log units. The triangular law adds $(J_{AB},J_{AC},J_{BC})=(0.35,-0.18,0.25)$. The alternative also has $K_{ABC}=0.32$. The three law-specific studies use independent random streams for the data and fitted bootstrap. Wilson intervals describe variation between cohorts. The p-value grid spacing with 99 draws is 0.01. [Benchmark driver](../benchmarks/null_calibration.py)

| Generating law | Initial rejections (95% interval) | Independent follow-up (95% interval) | Undefined draws, initial / follow-up | Follow-up median time per cohort |
| --- | ---: | ---: | ---: | ---: |
| Independent sites | 4/80, 5.0% (2.0–12.2%) | 10/320, 3.1% (1.7–5.7%) | 0 / 0 | 0.050 s |
| Pairwise triangle | 8/80, 10.0% (5.2–18.5%) | 18/320, 5.6% (3.6–8.7%) | 0 / 1 | 0.086 s |
| Triangle plus parity | 62/80, 77.5% (67.2–85.3%) | 252/320, 78.8% (73.9–82.9%) | 2 / 17 | 0.103 s |

The first triangle cohort series had elevated rejection. The larger independent-seed follow-up kept the same laws, estimator, sample size, draw budget and threshold. All 1,200 observed datasets were eligible. Undefined replicates counted conservatively. These simulations cover the stated binary, independent-observation designs. Size and power in sparse mixed alphabets, related cohorts, or other prior regimes require separate validation. Timings include the observed fit and 99 refits per cohort on a Ryzen 7 9800X3D with Windows 11, Python 3.13.15 and NumPy 2.5.5.

### Group composition and exchangeability

```console
python benchmarks/null_calibration.py --experiment matching --depth 128 --draws 99 --repetitions 320 --seed 771097
```

Two fixed strata of 64 observations have state frequencies 0.2 and 0.8 respectively. A, B and C are independent within each stratum. A second model copies A into B with probability 0.35 within strata. For the same cohort, the benchmark tests one predetermined AB mutual-information value using both assignment laws. Ties count as exceedances.

| Generating law | Unrestricted assignment | Within-stratum assignment |
| --- | ---: | ---: |
| Independent within stratum | 314/320, 98.1% (95% CI 96.0–99.1%) | 9/320, 2.8% (1.5–5.3%) |
| Within-stratum AB copying | 320/320, 100% (98.8–100%) | 301/320, 94.1% (90.9–96.2%) |

Group composition alone makes the unrestricted random-correspondence hypothesis false. Restricting permutations preserves this composition. Within-group dependence from the added copying mechanism remains detectable. These synthetic observation groups test the block exchangeability assumption. They do not simulate a tree.
