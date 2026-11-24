# Fixed-support local maximum-entropy models

The `maxent` triple model asks how much of a **three-site marginal distribution** remains after matching all three pairwise marginals. The model uses one complete-row table for a prespecified ABC site triple. Other sites, lineages and unobserved variables can influence its result. A latent mixture can produce positive connected information even when its visible sites are independent within each latent class. [Schneidman et al., 2003](https://arxiv.org/abs/physics/0307072)

## Common data and prior

For each candidate, retain rows with positive weight and a valid native symbol at all three positions. Accumulate their weights in a joint table $C$. The default support is the full Cartesian product of the component alphabets, e.g. $20\times4\times4=320$ cells for protein, DNA and RNA. Ambiguity remains missing. An explicit gap option adds one state per axis. Unobserved joint cells remain possible. Prespecified structural zeros are provided to the numerical API as a boolean mask. A positive observation in a forbidden cell fails validation.

Let $W$ be the sum of observed weights, $S$ the number of allowed cells and $A$ the total prior mass. The saturated target on an allowed cell is

```text
p3(x,y,z) = [C(x,y,z) + A/S] / [W + A].
```

The default `prior_mode="total", prior_mass=0` gives the empirical distribution. In total mode the supplied mass is $A$. In `per_cell` mode the same input is the pseudocount for **each** allowed cell, so $A=S\alpha$. A value of 0.5 per cell contributes 160 weight units on the 320-cell mixed support and 4,000 on a three-protein product of 8,000 cells. Input weight scaling therefore affects a fixed nonzero prior. The fit reports $W$, $A$, $A/W$, $A/(A+W)$, actual positive-weight row and occupied-cell counts, and the declared number of allowed cells. Use multiple prior settings to inspect sensitivity. Every setting uses identical rows and encoded counts.

A training-derived support requires explicit training observation IDs and a positive `min_state_count`. At each component/site, count eligible native symbols on those training rows without weights. Each symbol reaching the threshold keeps a separate bin. Every remaining native symbol maps to a shared OTHER bin. Unknown symbols remain missing. Bin meanings and dimensions are then fixed, including when a state disappears from a test or shuffled sample. The metadata records the native symbols, every retained site's integer map, training IDs, threshold and gap policy. Training samples and held-out samples must use the same aligned sites in the same order. The caller keeps evaluation observations disjoint when the design requires a held-out estimate.

## Model definitions

- **M1** is the product of the three single-site marginals of the declared target. An arbitrary structural-zero mask may forbid cells with positive independent-product mass. In that case `m1_kind=support_constrained_singles` identifies a maximum-entropy singles fit on the mask. Such a support constraint can itself induce dependence.
- **M2** maximizes Shannon entropy on the same allowed support while matching the XY, XZ and YZ marginals of $p_3$. All three targets are computed from that single probability cube.
- **M3** is $p_3$, the saturated joint probability on the declared support and prior.

`connected_information_bits` is $D_{KL}(p_3\,\|\,p_2)$ in bits. At matched margins it also equals $H(p_2)-H$p_3$$. The outputs include both estimates and their absolute discrepancy. Redundancy, pairwise association and connected three-site information are distinct. A strictly positive triangular pairwise Gibbs law can have nonzero signed co-information and zero connected information. Balanced binary XOR has connected information 1 bit, signed co-information −1 bit, and pairwise-independent margins.

`co_information_bits` uses the weighted empirical single, pair and triple entropies on the same rows and frozen encoding. `target_co_information_bits` uses the prior-smoothed target. On a full, strictly positive Cartesian target, `threebody_rms` projects natural-log probabilities to zero mean on each axis in turn and computes their root-mean-square. Its units are nats. It is undefined on structural or sampling zeros where a finite log tensor on the full product does not exist. The separate historical local log-linear method retains its observed-state, positive-per-cell prior convention.

## Solver and boundaries

Iterative proportional fitting starts uniformly on all allowed cells compatible with the target's positive margins. A zero pair marginal forces all its participating cells to zero. An unobserved joint cell with three positive pair margins remains in the fit. Each update normalizes a slice of the current table and multiplies by its required marginal. A full cycle fits XY, XZ and YZ in that order. Normalizing conditionally before multiplication avoids forming extreme scale-factor ratios. [Csiszár and Shields, Information Theory and Statistics, §5](https://www.renyi.hu/~csiszar/Publications/Information_Theory_and_Statistics%3A_A_Tutorial.pdf)

Convergence requires maximum absolute marginal error ≤ `tolerance` and maximum relative error over positive marginal entries ≤ its square root. Joint-table fits also check the entropy/KL identity to the greater of the requested tolerance in bits and a size-dependent binary64 roundoff allowance. The defaults are $10^{-9}$, 1,000 cycles and a 100,000-cell workspace limit. The result reports the actual iterations, residuals and number of active cells. A solution on a more complicated boundary can converge arbitrarily slowly. A fit that reaches its iteration budget is reported as unresolved. The API never infers infeasibility solely from the iteration limit. Imported pair margins are checked for normalization, compatible dimensions and matching single-variable marginals.

`no_data`, `insufficient_rows`, `insufficient_weight`, `resource_limit`, inconsistent/incompatible margins, numerical underflow and unresolved solver/identity checks produce no connected ranking value. With zero prior a constant empirical axis yields the well-defined zero information and an explicit constant-axis flag. If a prior alone introduces empirical-axis variation, `prior_only_axis` withholds the ranking value. `target_kl_bits` still gives the fitted smoothed-target functional. Positive data on forbidden support, invalid settings and malformed tensors are input errors.

## Likelihoods, occupancy and uncertainty

Each reported data log likelihood is $\sum C\log p_m$, in natural-log units weighted by the supplied row weights. `deviance_nats` is twice the M3-minus-M2 data log likelihood. With smoothing the fitted target differs from the empirical table. `target_deviance_nats` is the doubled augmented-target likelihood difference, $2(W+A)\ln(2)D_{KL}(p_3\|p_2)$, subject to fit precision. The latter uses the prior as well as data. Neither quantity is an asymptotic chi-square test.

`expected_occupied_m2` is the sum over cells of $1-(1-p_2)^n$, conditional on hypothetical independent draws from the fitted pairwise model. Here $n$ is the **actual integer complete-row count**. A low-level count tensor without that count leaves the diagnostic undefined. Sequence redundancy, ancestry and uncertainty in pairing can make independent-draw expectations inaccurate. Residuals quantify numerical error. Prior sensitivity measures dependence on the chosen regularization. This model fit alone supplies no sampling confidence interval or p-value.

## Numerical API

```python
from triplet_coevolution.inference.maxent import MaxEntOptions, fit_joint

fit = fit_joint(joint_weight_counts, n_valid=complete_rows,
                options=MaxEntOptions(prior_mode="total", prior_mass=1.0))
print(fit.score.status, fit.score.connected_information_bits, fit.score.ipf_residual)
```

`fit.p1`, `fit.p2` and `fit.p3` expose the probability models when available. `fit_maxent_grid` accepts a matched MSA, candidate site tuples, weights, a frozen support and an options sequence. Sites are indexed from zero in the retained alignments. The result is indexed first by setting, then by candidate. Counting is shared across settings and identical options reuse their score. Dense fits are released between candidates. Candidate selection, null hypotheses and coordinate provenance belong to the caller. [Numerical solver](../src/triplet_coevolution/inference/maxent.py) · [Fixed state support](../src/triplet_coevolution/inference/maxent_support.py) · [Exact and failure-case tests](../tests/test_maxent.py)
