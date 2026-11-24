# Simulation of partner errors, missing sites and sparse support

The study ran all 64 registered settings with 64 independent repetitions each. Under complete independence, the two-test Benjamini–Hochberg procedure made a discovery in 17 of 1,024 repetitions. With an AB coupling and no three-way interaction, the held-out triple test rejected in 17 of 1,024 repetitions. The strong planted triple was selected in training 848 times and confirmed by the held-out exact test 596 times. For the weak effect these counts fell to 457 and 142. Equal numbers of replicates were allocated to each setting. The pooled counts describe this particular matrix. [Prospective protocol](../benchmarks/recovery_matrix_protocol.json) · [Full results and confidence intervals](../benchmarks/recovery_matrix_results.json)

## Design

The protocol was committed at `a4ae12d93132ae671acb3063fe65252091af5c6b` before running the new comparisons. Each synthetic role contains two binary sites. One site per role is chosen at random as the hidden target. The four laws are complete independence, AB copy dependence alone, and weak or strong AB dependence accompanied by a three-way parity interaction. Low minor-state frequency reduces support. Corrupting a match replaces the entire C partner in that observation with an independent donor drawn from the same population marginal. Missing sites are sampled independently after substitution. This model tests one identifiable type of partner error. Related paralogs with correlated substitutions, alignment errors and clade-dependent missingness can behave differently.

Each replicate has independent training (N), inner validation (N/2) and outer test (N) observations with the same planted coordinates. Training screens all four AB candidates using native empirical MI and all eight ABC candidates using native IPF connected information with total prior mass 1. At least eight complete observations and both states at every site are required in training. The selected triple’s pair-only maximum-entropy (M2) and saturated (M3) models are fitted to training counts. Validation selects a prior mass of 0.5 or 8 separately for each model by mean log probability. The original training fits are retained. No test rows enter selection, fitting or tuning. Across 4,067 evaluable predictions, validation chose mass 8 for M2 in 2,370 and for M3 in 2,554.

Held-out contingency tables test AB independence and a zero three-way loglinear term, respectively, using two-sided exact tests. The triple test conditions on all pair marginals. Valid tables on a binary fiber differ by integer multiples of the alternating parity contrast and have conditional mass proportional to the inverse product of their cell factorials. Sparse tables can have just one feasible allocation and consequently a p-value of one. We report marginal rejection at α = 0.05, and BH-adjusted decisions at q = 0.05 across the two selected hypotheses, using p = 1 for an unavailable test. Model comparison uses the difference in held-out joint log probability, M3 minus M2 in nats per complete observation of the selected triple.

## Results by setting

Each count has denominator 64. **Null BH** counts global-null repetitions with any BH discovery. **Pair-only false triple** counts raw exact triple rejections when only AB dependence exists. **Weak selected/confirmed** and **strong selected/confirmed** count exact training coordinate recovery followed by correct raw held-out rejection. **Strong gain** is the average held-out M3–M2 gain among evaluable replicates. The full record gives both raw and BH results for each family, Wilson 95% intervals, mean false discovery proportion, uncertainty for predictive gains, support, prior choices and fit counts. [Machine-readable results](../benchmarks/recovery_matrix_results.json)

| Training/test N each | Minor frequency | Missing | Wrong C | Null BH | Pair-only false triple | Weak selected/confirmed | Strong selected/confirmed | Strong gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | .50 | 0 | 0 | 3 | 2 | 29/7 | 60/58 | +.1964 |
| 48 | .50 | 0 | .35 | 1 | 1 | 18/1 | 54/37 | +.0693 |
| 48 | .50 | .20 | 0 | 0 | 0 | 25/1 | 47/35 | +.1555 |
| 48 | .50 | .20 | .35 | 0 | 1 | 12/0 | 37/11 | +.0043 |
| 48 | .08 | 0 | 0 | 0 | 0 | 19/0 | 58/20 | +.0943 |
| 48 | .08 | 0 | .35 | 0 | 0 | 9/0 | 46/6 | +.0076 |
| 48 | .08 | .20 | 0 | 1 | 0 | 19/0 | 34/2 | +.0714 |
| 48 | .08 | .20 | .35 | 0 | 0 | 10/0 | 31/1 | +.0241 |
| 192 | .50 | 0 | 0 | 4 | 4 | 58/55 | 64/64 | +.2070 |
| 192 | .50 | 0 | .35 | 1 | 5 | 41/23 | 64/64 | +.0780 |
| 192 | .50 | .20 | 0 | 2 | 2 | 51/23 | 64/64 | +.2034 |
| 192 | .50 | .20 | .35 | 4 | 2 | 33/10 | 61/58 | +.0754 |
| 192 | .08 | 0 | 0 | 0 | 0 | 47/15 | 64/63 | +.0873 |
| 192 | .08 | 0 | .35 | 1 | 0 | 36/5 | 60/52 | +.0363 |
| 192 | .08 | .20 | 0 | 0 | 0 | 27/2 | 61/45 | +.0768 |
| 192 | .08 | .20 | .35 | 0 | 0 | 23/0 | 43/16 | +.0179 |

Under the strong law, the clean depth-192 balanced condition confirmed 64/64 targets (Wilson 95% interval 94.3–100%). With depth 48, rare states, missingness and incorrect partners, training still found the target in 31/64 runs but independent testing confirmed it in just 1/64 (0.28–8.33%). Increasing depth to 192 with all three stressors gave 16/64 confirmed (16.0–36.8%). The corresponding mean predictive gains were +0.2070 (replicate bootstrap 95% interval +0.1972 to +0.2162), +0.0241 (+0.0015 to +0.0486), and +0.0179 (+0.0075 to +0.0286). The gain concerns whichever triple was selected, whether or not it matched the true coordinate.

The weak law’s mean gain ranged from −0.0615 to +0.0216 across settings and was positive in only seven of sixteen. M3 had lower mean held-out log probability in every independent and pair-only setting. After BH adjustment, correct triple discoveries numbered 138/1,024 for the weak law and 594/1,024 for the strong law. Mean per-replicate false discovery proportions across the equally represented settings were 1.66% for complete independence, 0.83% for pair-only, 0.68% for weak and 0.05% for strong. These estimates characterize the registered sampling design. The dependences between the two tests were not used to establish a general FDR guarantee.

All 4,096 scheduled repetitions are represented. Three could select no eligible pair and six could select no eligible triple. An additional 23 selected triples had fewer than eight jointly observed validation rows, leaving 4,067 evaluable predictive gains. Two individual M2 prior fits failed the solver’s entropy identity check; the remaining prior provided a fitted M2 in both cases. Per-setting binary rates use Wilson intervals. Mean gains and mean per-replicate false discovery proportions use 1,000 resamples of independent simulated repetitions in the same setting. No biological sample uncertainty is inferred from the number of simulations. Separate registered phylogenetic studies cover latent clades, misspecified branch lengths and clade holdout. [Tree calibration](tree-model.md) · [Clade and contamination controls](../benchmarks/phylogeny-controls.md)

## Replay

Use the source checkout or the installed benchmark directory and an environment containing the package. The complete script needs only NumPy and the native inference implementation. Six local workers process different settings. `--jobs 1` uses the same logical RNG streams. The tests check exact fixed-margin probability by independent enumeration, recover the analytic contaminated population law, change outer-test outcomes while keeping training decisions fixed, and check visible denominators for fully failed runs.

```sh
python -m pytest -q tests/test_recovery_matrix.py
python benchmarks/recovery_matrix.py --jobs 6 --out results/recovery-matrix-replay.json
```
