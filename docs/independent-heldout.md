# Does the pair-plus-triple model predict a held-out phylum better?

On the two registered six-site biological panels, adding all eight ABC terms to the pair-only model produced a small average improvement. Both primary 95% intervals included losses as well as gains, and the matched-negative controls produced changes of comparable size. These panels provide insufficient evidence of a reproducible prediction benefit from the added terms. [Registered settings](../benchmarks/independent_evaluation_policy.json) · [Every fold and denominator](../benchmarks/independent_nested_metrics.json)

| Cohort and setting | Complete held-out rows / eligible | Phyla with positive gain / tested | Equal-phylum gain, nats/site (95% interval) | Pooled weighted gain |
| --- | ---: | ---: | ---: | ---: |
| Polymerase, observed uniform | 421 / 424 | 13 / 22 | +0.00019 (−0.01036, +0.00985) | +0.00074 |
| Polymerase, equal phylum weights | 421 / 424 | 14 / 22 | +0.00631 (−0.00617, +0.01764) | +0.00617 |
| Polymerase, within-phylum mismatching | 421 / 424 | 12 / 22 | +0.00208 (−0.01045, +0.01371) | +0.00326 |
| Polymerase, within-phylum site shuffling | 421 / 424 | 12 / 22 | +0.00180 (−0.00932, +0.01196) | +0.00284 |
| Mixed ribosome, observed uniform | 109 / 110 | 5 / 9 | +0.00600 (−0.00735, +0.01948) | +0.00375 |
| Mixed ribosome, equal phylum weights | 109 / 110 | 6 / 9 | +0.00461 (−0.00864, +0.01771) | +0.00465 |
| Mixed ribosome, within-phylum mismatching | 109 / 110 | 5 / 9 | +0.00935 (−0.00991, +0.03043) | +0.00602 |
| Mixed ribosome, within-phylum site shuffling | 109 / 110 | 5 / 9 | +0.00773 (−0.00604, +0.02315) | +0.00523 |

A positive gain means the pair-only held-out conditional log loss exceeds the pair-plus-K loss. Both losses average the six site conditionals over the exact same complete observations with the same weights. Polymerase primary pooled losses were 1.26345 and 1.26271 nats/site. Mixed ribosome primary pooled losses were 1.18516 and 1.18141. Under the equal-phylum sensitivity, fitting used the registered source masses. Normalizing within the training fold cancels their common scaling constant. The JSON also reports an assembly-count-weighted gain for each setting. [Estimand, conditional solver and independent truth controls](conditional-models.md)

## Which observations and sites were eligible?

The DNA-dependent polymerase panel comprises three proteins. The ribosome panel comprises 5S RNA, uL5 and uL18. The positional rule used the full reference width and selected its one-third and two-thirds coordinates before opening any structure labels or conditional losses. Polymerase positions were A109/A219, B447/B894 and C469/C938. Mixed ribosome positions were A40/A80, B59/B119 and C39/C78. There was no site search, substitution after scoring or structural filtering. The two cohorts share 139 assemblies. [Outcome-blind acquisition and reference definitions](independent-cohort-staging.md)

Each outer test fold comprised one complete GTDB phylum with at least eight admitted assemblies and at least 24 outside. The 461 polymerase assemblies covered 40 phyla. Of these, 22 phyla met the test size and the other 18 provided 37 training-only genomes. The 163 ribosome assemblies covered 29 phyla. Nine qualified as test folds and the other 20 provided 53 training-only genomes. Phyla excluded from testing still contributed to training when eligible. On the observed tables, complete training sizes ranged from 434 to 449 in polymerase and from 143 to 150 in ribosome. Three eligible polymerase genomes and one eligible ribosome genome lacked a complete score on the panel. The taxa, frozen split identities, training/test mass, Kish depth, gradient residual, coefficient counts, model digests and excluded-group counts are in the compact result.

Each training fold learned the same six fields and all 15 pair terms for both models. The second model included all eight registered ABC terms. Training states with fewer than eight occurrences among positive, complete training rows were pooled. Both used prespecified L2 penalties of 0.1, L-BFGS tolerance 10⁻⁶ and at most 500 iterations. Held-out rows never set the state dictionary, regularization, graph or reference panel. At least 24 complete training observations and one complete held-out observation were required. All 248 fits across 124 fold/setting combinations converged. The release test changes the held-out signal, checks that both fitted model hashes remain identical and verifies the expected reversal in held-out loss. [Regression tests](../tests/test_registered_nested.py)

## Pairing controls and intervals

The first negative control kept A fixed and reassigned complete B and C rows at fixed offsets within each phylum after sorting versioned assembly IDs by a registered hash. Singleton phyla retained both assignments. Groups of size two retained the C assignment. The second control shuffled canonical residues at each of the six sites independently within phyla while leaving missing cells fixed. Each control is one prespecified disturbance. Its measured gain has no permutation p-value.

Intervals resampled paired **held-out phylum means** 2,000 times with PCG64 seed 731903. Each phylum contributed once. The intervals describe these complete folds conditional on the fitted models. Related phyla, overlapping training sets and shared assemblies across cohorts limit independence. Two systems provide too little replication for a population-of-systems interval. The six positions cover a small part of both complexes and lack fitted effects for omitted mediators.

## Reproduce

Install the `optimize` extra, stage the originally hashed cohort sources and set numerical libraries to one thread to reproduce the measured CPU run. Run from the clone using two new output directories. Fitted NPZs and exact split membership stay in those outputs. Only compact aggregate and fold-level evidence ships with the package.

```shell
python benchmarks/independent_nested.py results/public-experimental/independent-cohorts results/holdout-a
python benchmarks/verify_independent_nested.py results/public-experimental/independent-cohorts results/holdout-a
python benchmarks/independent_nested.py results/public-experimental/independent-cohorts results/holdout-b
```

On Windows, Python 3.13.15, NumPy 2.5.5 and SciPy 1.18.1, the full sequence-only comparison took 8.54 and 8.70 seconds in two single-threaded runs including model/split serialization. Both produced the same compact JSON bytes, SHA-256 `e65a4e4e630dbd6b1f1a727cd80e976ca8a0c5092829f8a513464e9c9c65d3f6`. Independent scalar enumeration read the original FASTA strings and saved energy tensors for all 62 primary fold/model predictions. The largest absolute discrepancy in mean conditional loss was 4.45 × 10⁻¹⁶ nats/site. Original sources are checked against the registered hashes before fitting. [Driver](../benchmarks/independent_nested.py) · [Independent arithmetic](../benchmarks/verify_independent_nested.py)
