# Weighting policies and validation

Each sequence-derived policy compares jointly canonical symbols on the retained alignment. Protein, DNA and RNA use their declared alphabets. For `equal_component`, each component must have at least one shared canonical position and meet the minimum overlap fraction relative to its retained length. The arithmetic mean of component identities must meet the identity cutoff. `component` applies the same test to a named component. `concatenated` first pools the matching and overlapping site counts and divides by the pooled overlap. Its overlap denominator is the sum of the component lengths. Length differences and component-specific missingness can therefore change which pairs qualify. All thresholds are inclusive.

`equal_component`, `component` and `concatenated` assign the inverse number of qualifying neighbors, including the row itself. The equal-component and single-component policies require some canonical sequence in every row of each participating component. The concatenated policy requires some canonical sequence in the combined row. `cluster_representatives` uses the connected components of the equal-component neighbor graph. It chooses the smallest observation ID in Unicode lexicographic order within each connected component, assigns weight one to that observation and zero to its other members. A path through an intermediate observation can join two endpoints whose own identity falls below the cutoff. `weights.tsv` records both the local neighbor count and, for this policy, the representative ID.

`clade_balanced` assigns total weight one per supplied clade and splits that mass equally among its observations. `supplied` preserves the original nonnegative weights and their absolute scale. Both require complete validated observation-ID mappings. Their tables leave the neighbor count blank. Zero-weight observations contribute neither count nor weight to score support. The reported sum of weights measures retained mass on the chosen scale. The Kish count, $(\sum w)^2/\sum w^2$, measures concentration and remains invariant under common rescaling. Neither measures an independently established number of evolutionary degrees of freedom.

Native MI and DCA apply both count and absolute weight thresholds to the shared observations for each pair before APC. Local triple scores use the complete rows for their three sites. The global model applies the thresholds to its shared complete-case fit across all retained sites. The weights used by DCA are normalized for the covariance fit and the original scale is used for its support test. MI normalizes within each pair. In the local model, the original scale also affects the relative influence of its additive pseudocount. Global canonical support can be smaller than the support of a local triple. Scores with too little support are undefined. The complete pair archive retains support for each coordinate. The threebody_support table lists every tested triple, its original columns, count and weight pass flags, model scope, and whether its selected score is finite. Near a weight cutoff, a correctly rounded sum ensures consistent inclusive decisions across numerical reduction orders.

Independent whole-component permutations hold A and its observation identifiers fixed and rematch B and C. The scorer recomputes every sequence-derived weight from the rematched rows, including representative selection. Explicitly supplied weights stay attached to the anchored observation design. Clade weights stay fixed within clades and the scorer checks that every permutation stratum lies within a single supplied clade. A design whose weights are attached to the displaced B or C sequence requires a different null.

## Simulation with biased replication

The fixed generator used 32 independent seeds, 8201 through 8232. Each cohort had 40 founders with a Bernoulli(0.35) outcome. Positive founders had eight descendants and other founders had two. The near-duplicate relationship was imperfectly visible to the estimator. Descendants inherited random founder-specific protein, DNA and RNA sequence templates of 40, 8 and 6 positions, with independent random replacement probabilities of 0.05, 0.12 and 0.18. For each cohort we estimated prevalence from descendant outcomes and measured absolute error against a separate sample of 512 founders. Only the simulated design and group-balanced policies used the true replication factors or founder IDs. The identity and overlap settings were 0.8 and 0.5 throughout.

Mean cohort size was 167. Brackets give 95% percentile intervals for mean absolute error from 4,000 resamples of entire independent simulation replicates, seed 48027.

| Policy | Mean absolute error [95% interval] | Mean weight sum | Mean Kish count |
| --- | ---: | ---: | ---: |
| Unit weights | 0.329 [0.301, 0.357] | 167.00 | 167.00 |
| Equal component | 0.106 [0.079, 0.133] | 62.52 | 102.79 |
| Concatenated | 0.077 [0.063, 0.091] | 42.56 | 107.16 |
| A alone | 0.075 [0.060, 0.091] | 40.41 | 109.97 |
| B alone | 0.137 [0.105, 0.170] | 70.13 | 103.77 |
| C alone | 0.162 [0.131, 0.194] | 62.32 | 102.29 |
| Cluster representatives | 0.088 [0.073, 0.103] | 56.19 | 56.19 |
| Group balanced | 0.074 [0.059, 0.090] | 40.00 | 110.61 |
| Supplied inverse replication | 0.074 [0.059, 0.090] | 40.00 | 110.61 |

Equal-component weighting reduced mean absolute error by 0.223 relative to unit weights, with a paired 95% interval for the reduction of 0.205 to 0.239. Founder balancing and inverse replication give identical row weights in this design. Replication depends on the outcome by construction. Different sampling mechanisms, rates and length ratios can change the results.

## Held-out taxa from GatABC

The script verified the original SHA-256 digests for the three prepared alignments, match table and analysis arguments. The full default fit reproduced the historical sum of **505.49112803812943** on 888 observations, with Kish count **590.9818664111334**.

For the held-out comparison, a fixed SHA-256 partition based solely on the supplied taxon ID put 736 observations in training and 152 observations from 151 distinct taxa in evaluation. Every row of a given taxon was assigned to the same set. The source comprised 884 distinct taxon IDs. Column masks and weights were fitted within training, retaining 480, 474 and 94 sites. Each policy estimated the 20 canonical state probabilities by site with a symmetric half-count per state after rescaling weights to training N. We averaged the held-out negative log probability over the available canonical symbols within each row, then over rows per taxon, then equally over taxa. There were 154,904 canonical test symbols. The group-balanced policy used the supplied taxon IDs. Unit weights exercise the supplied-weight implementation as a control.

| Policy | Held-out bits per symbol | Paired difference vs unit [95% interval] | Training weight sum |
| --- | ---: | ---: | ---: |
| Unit weights | 2.05480 | 0.00000 [0.00000, 0.00000] | 736.00 |
| Equal component | 2.08606 | 0.03127 [0.00937, 0.05299] | 437.62 |
| Concatenated | 2.09106 | 0.03627 [0.01311, 0.05927] | 412.64 |
| A alone | 2.08915 | 0.03435 [0.01135, 0.05711] | 419.13 |
| B alone | 2.09484 | 0.04005 [0.01597, 0.06376] | 378.43 |
| C alone | 2.07973 | 0.02493 [0.00551, 0.04420] | 470.19 |
| Cluster representatives | 2.08912 | 0.03433 [0.01154, 0.05687] | 433.00 |
| Group balanced | 2.05478 | -0.00001 [-0.00030, 0.00033] | 733.00 |

Lower values indicate better prediction of held-out site symbols. The units baseline gave lower mean loss than every sequence-derived policy in this split. The 4,000 paired bootstrap draws resampled held-out taxon IDs, using seed 38427 and keeping the training fit fixed. These intervals describe variation across observed test taxa, conditional on this aligned system and its fitted models. Taxa can share ancestry across the split. Contact recovery, phylogenetic independence and generalization to other molecular systems require separate evidence.

## Resources and reproduction

The exact algorithm packs canonical validity and symbol bits and traverses symmetric row blocks. It uses O(N²L) time in scalar site comparisons and avoids a mandatory dense N-by-N identity matrix. The plan reports unordered distinct pairs, the number of row-pair slots including diagonal tile work, and the word slots across components per uncached pass. When permitted, null refits cache exact per-component match and overlap counts. The default ceiling for that cache is 32 MiB. A specified numerical workspace budget can reduce block size or disable the cache. If even one pair block cannot fit, the planner reports the limitation before allocating comparison arrays. The estimate covers numerical arrays only. Existing inputs, Python objects, other commands and process runtime contribute to measured process memory.

On Windows with Python 3.13.15, NumPy 2.5.5 and single-threaded BLAS, the full 888-row weighting step took 0.195 seconds. Its numerical workspace estimate was 25,046,272 bytes, with 128 rows per block and 393,828 unordered distinct pairs. Both benchmarks together took 3.19 seconds and the process peaked at 109,244,416 bytes. Timings include one host measurement.

Run the synthetic comparison from the repository root. The optional argument adds the hash-checked private taxon benchmark. Both print aggregate JSON.

```bash
python benchmarks/weighting_policies.py
python benchmarks/weighting_policies.py --gatabc-root "$DATA"
```
