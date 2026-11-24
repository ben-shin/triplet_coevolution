# Interpreting matched-alignment scores

Each row is one curated A–B–C observation. The rank tables identify sites in the retained alignment, the original input alignment and the named reference. All reported coordinates start at one. A gap in the reference has an empty reference position. The reference can be outside the matched observations.

## Observations and weights

QC measures canonical coverage, gap fraction and missing fraction in each component before weighting. Each component has its own column mask. The retained sites are listed in `run_summary.json`.

Sequence weighting compares two observations at jointly canonical sites. Every component must meet the overlap cutoff, measured against its retained length, and the arithmetic mean of the three component identities must meet the identity cutoff. A row with k neighbors, counting itself, gets weight 1/k. The sum of these weights is the effective depth. Closely related lineages can still share evolutionary history after weighting.

## Pairwise MI and APC

MI is the weighted empirical mutual information in bits, computed on jointly canonical rows for each site pair. Its denominator is the sum of weights on those rows. `min_valid` applies to their unweighted count. APC subtracts the product of the row and column means divided by the overall mean, separately within AB, AC and BC. Finite cells contribute to the means, a zero global mean gives zero correction, and corrected scores can be negative.

## Pairwise direct coupling analysis

`--dca` fits one mean-field model across all retained sites. Each variable site contributes a one-hot vector for its observed canonical states. A missing or gap value receives the site's weighted mean vector. This makes its centered features zero and keeps the joint covariance positive semidefinite. Constant sites contribute zero coupling.

For centered categorical features X, row weights w, and positive ridge λ, the estimator uses

$$C = X^T \, \mathrm{diag}(w) \, X / \sum_r w_r, \qquad J_{ij} = -[(C + \lambda I)^{-1}]_{ij}.$$

Each off-diagonal state block is centered along both axes to obtain the zero-sum gauge. The raw statistic `dca_frobenius` is the Frobenius norm of that block. The default ranking uses `dca_apc`, the blockwise APC score. Each pair's `n_valid` and `valid_weight` count jointly canonical rows before imputation. Global conditioning can reduce mediated associations. An inferred coupling still depends on the sampled sequences, regularization, missingness and unobserved factors. [Mean-field DCA, Morcos et al.](https://arxiv.org/abs/1110.5223) · [Score selection and migration](scores-and-benchmarks.md)

`--ridge` defaults to 0.1 in covariance units. Write N for the number of matched observations and D for the total observed-state features. With N smaller than D, a Woodbury identity computes exactly the same off-diagonal inverse blocks through an N-by-N factorization. This supports long proteins at moderate depth. The dual path caps its design matrix at 16 million N×D elements and its solve dimension at 4,096. Otherwise a batched feature-space covariance and D-by-D solve are available up to D=4,096. Both paths still produce all inter-component score cells. If both limits are exceeded, choose a biologically justified panel or run a smaller analysis. Changing the panel changes the model's conditioning set.

## Three-site inference

`--threebody` evaluates triples with one site from each component. If all eligible triples fit within `--max-triplets` (default 1,000), all are scored and the report labels the selection `exhaustive`. Otherwise the algorithm draws a `uniform` sample without replacement from the coordinate grid using `--seed`. Sampling requires memory proportional to the candidate budget. A small sample can miss a real interaction. Prespecified candidates are labeled `file` and allow a focused experiment that is repeatable across runs.

Supply a tab-separated file with a header and one row per triple using `--triplet-candidates`. Use one-based **original alignment columns**, including alignment gaps in the numbering. This file selects one triple.

```text
a_original_col	b_original_col	c_original_col
3	1	2
```

The file must refer to retained columns. `--require-reference` also requires numbered reference residues at the selected sites. Duplicate or ineligible triples fail with an input error. Uniform coordinate sampling keeps pure parity interactions eligible even when each pairwise MI is zero.

### Local three-factor Potts interaction

The default `--threebody-model local` fits a saturated log-linear model to each three-site marginal. Use rows with canonical residues at all three sites and positive weight. For each site, the state space is its observed canonical support in these rows. Add `--pseudocount` (default 0.5) to **each cell** in the Cartesian product of these supports.

Take the natural logarithm of the smoothed joint counts and subtract the mean along each of its three state axes in turn. The resulting tensor is the three-factor interaction in the zero-sum gauge. This projection removes the intercept, all single-site fields and all pairwise terms of the fitted marginal. A constant normalizing factor cancels. `threebody_rms` is the root mean square of the resulting tensor entries in natural-log units. A variable with only one observed state has zero three-factor contrast. Sparse tables and the pseudocount can strongly affect this score. Compare `observed_cells`, `possible_cells`, effective support and pseudocount sensitivity. [Hierarchical log-linear interactions, Amari](https://people.csail.mit.edu/jrennie/trg/papers/amari-ig-hierarchy-01.pdf)

The local model marginalizes all other sites. Dependencies at those other sites can produce a marginal three-factor signal. The global option below addresses a different question by conditioning on the retained panel.

### Global three-body Potts model

`--threebody-model global` fits one shared Hamiltonian with fields at every variable site, one pair tensor for **every** pair of variable sites (including within-component pairs), and one three-site tensor for each selected variable ABC triple. The log probability is proportional to the sum of the corresponding fields, pair terms and triple terms. Each tensor sums to zero along each of its state axes.

The optimizer minimizes the sum of weighted negative conditional log probabilities for each site's state given the other retained sites, plus `global_l2 / 2` times the sum of squared coefficients. The weights are normalized over rows with canonical residues at **every retained site**. A shared tensor contributes to each conditional involving its sites. This fits a single joint model without enumerating its joint partition function. The L2 penalty (default 0.1) acts on fields, pairs and triples in the declared gauge. Backtracking gradient descent stops when the largest projected gradient magnitude is at most 10⁻⁶. Failure to converge in 500 iterations raises an error. [Potts pseudolikelihood fitting, Ekeberg et al.](https://arxiv.org/abs/1211.1281)

Here `threebody_rms` summarizes the fitted global triple tensor. The summary records the number and weight of complete rows, variable sites, coefficients, iterations, objective and final gradient. If too few complete rows remain, the table has no estimable ranks and the summary records the reason. Interactions apply to the selected model and sample. Establishing biochemical causality requires experimental evidence.

Start global fitting with a short panel and a small set of triples. The implementation allows up to 64 variable sites, 250,000 field/pair/triple coefficients before gauge reduction, and 2,000,000 row-context indices. Each term contributes one index per complete row per site in that term. Including more candidate triples changes the joint fit.

### Signed three-variable co-information

Both triple modes also report the signed descriptive co-information, in bits, using the unsmoothed weighted empirical table on the same rows as the model. It is the sum of the three single-site entropies, minus the three pair entropies, plus the joint entropy. Balanced binary XOR gives −1 bit even though each pairwise MI is zero. A log-linear interaction and co-information answer different questions and have different units.

## Permutation benchmarks

`--permutations B` calibrates the selected MI statistic and any enabled DCA and three-body ranking statistics. A stays fixed. Each draw independently permutes whole B rows and whole C rows, retaining sites and missingness patterns within each component. With `--strata-column`, draws permute only within the specified matching-table groups. The procedure recomputes triplet sequence weights and refits every enabled model for every draw. Column masks, coordinate candidates and reference eligibility depend only on fixed component properties or the supplied candidate list. `--max-pairs` limits display ranks only.

The null hypothesis is independent cross-component matching conditional on these component alignments and exchangeability groups. For example, a curator can supply taxonomic groups when within-group exchangeability is defensible. A family of closely related isolates may violate this assumption even within a group. The test does not simulate evolution along a phylogenetic tree. The triple permutation p-value evaluates the score under shuffled correspondence. Testing a specific triple against an interacting pairwise-only generative null would require a different calibration.

The upper-tail Monte Carlo p-value is (1 + e)/(1 + B), where e counts null scores at least as large as the observed score. Numerical ties count as exceedances. An undefined null score also counts as an exceedance, conservatively. An undefined observed score contributes p=1 to multiple testing and has no output row. `permutation_*.tsv` or `.csv` includes all estimable prespecified hypotheses in coordinate order with the statistic, p-value, adjusted value and draw count. [Permutation p-values, Phipson and Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)

Benjamini–Hochberg adjustment uses one family for all eligible AB, AC and BC pairs of a method, and one family for the chosen ABC triples. Its usual FDR guarantee assumes independence or suitable positive dependence. Overlapping site tests can be dependent. A candidate list chosen after examining these same cross-component scores needs independent testing data for confirmatory inference. For 99 draws the smallest raw p-value is 0.01, which may be too coarse for a large family. [Benjamini and Hochberg](https://www.stat.cmu.edu/~ryantibs/journalclub/benjamini_1995.pdf)

## Reference-contact benchmarks

`--contacts` accepts a tab-separated file of **explicit positives and negatives** keyed to the named references and their biological positions. The header and two example rows are shown below.

```text
x_component	x_reference_id	x_reference_pos	y_component	y_reference_id	y_reference_pos	contact
A	ref_a	101	B	ref_b	48	1
A	ref_a	102	C	ref_c	17	0
```

One row labels one inter-component pair. Supply the reference offsets used in scoring and a documented structural/contact definition appropriate for the reference complex. The program joins either orientation, rejects duplicate labels and evaluates finite scores over labelled pairs only. `contacts.json` reports total, scored and unscored labels, class counts, tie-aware AUROC and noninterpolated average precision at tied score thresholds. AUROC requires both classes, and average precision requires at least one positive. JSON `null` denotes an unavailable metric. Unlabelled pairs are excluded. The full finite score set is evaluated before display truncation.
