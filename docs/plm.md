# Mixed-alphabet pseudolikelihood DCA

The `plm` backend fits a multinomial conditional model for every retained target position. All other sites on the declared panel can act as predictors, including sites within the same component. Joint mode uses A, B and C together. Pairwise mode fits AB, AC and BC separately. Changing the panel can change both the conditioning variables and the set of complete rows. The package also provides the `mf` ridge inverse-covariance estimator. Select the backend explicitly when comparing scores. [Ekeberg et al., 2013](https://doi.org/10.1103/PhysRevE.87.012707) · [Ekeberg, Hartonen and Aurell, 2014](https://arxiv.org/abs/1401.4832)

## Objective and gauge

Let $w_r$ be the nonnegative observation weight, $W$ the total complete positive weight, and $f_j$ the vector of weighted observed frequencies at predictor $j$ on those same rows. The centered categorical features are $x_{rj}=e(s_{rj})-f_j$. For target $i$, the field $h_i$ has one coefficient per native target state. Each directional coefficient table $J_{j\to i}$ has rows indexed by predictor state and columns by target state. The logits are

$$\eta_{ri}=h_i+\sum_{j\ne i}J_{j\to i}^{\mathsf T}x_{rj}.$$

The target probability is the softmax over its full declared alphabet. The solver minimizes the weighted mean negative log conditional probability in nats plus $\lambda_h\lVert h_i\rVert^2/2 + \lambda_J\sum_j\lVert J_{j\to i}\rVert_F^2/2$. Each strictly positive penalty controls its own parameter class. Rescaling all input weights by a common positive factor preserves the optimum once eligibility is unchanged. Centering gives the field its value at the observed predictor means. Separate target fits can produce incompatible conditional distributions. The fitted scores describe conditional associations; these conditionals do not by themselves supply a validated joint simulator.

For each inter-role pair, take the canonical submatrix of each directional coefficient table and subtract its row and column means, then add its grand mean. This equals projecting the full block, removing any gap row or column, and projecting the canonical block again. Transpose the reverse directional estimate to matching axes and average the two projected matrices. The raw coupling score is the Frobenius norm of this average. The sign of an individual coefficient is defined by the zero-sum gauge. APC uses the complete eligible interface block and the analysis's declared coordinate universe. The native archive stores the statistic names `dca_frobenius` and `dca_apc` together with the selected estimator. A Frobenius norm accumulates more terms for a larger alphabet; interpreting cross-interface differences should account for state count, entropy and support.

## Missing observations and capacity

The fixed alphabet has 20 canonical protein states or four DNA/RNA states. If `include_gaps` is true, one additional observed gap category enters each conditional fit. Gap coefficients are excluded from the canonical scoring norm. Ambiguous and missing symbols have no fitted category.

The supported `complete` policy requires every site on the declared conditioning panel to be observed in an eligible row. It uses the same rows, weights and predictor means for both directions. The score support counts positive fit rows with canonical states at **both** endpoints. A constant canonical endpoint, insufficient row count/weight, or a failed directional optimization withholds the pair score. Inspect the complete-row losses, total weight and estimable score count in diagnostics. A wide panel with scattered missing data can have little or no complete support. Select and report a biologically justified panel independently of the scores if a smaller conditioning set is needed. Informative observation loss can change the complete-case distribution.

An exact eight-state binary chain illustrates the masking decision. Its spin fields were $(0.27,-0.24,0.12)$, its signed AB and BC couplings were $0.53$ and $-0.71$ nats, and AC was absent. We enumerated each state and every independent missingness pattern at a per-site missing probability of 0.2. Complete cases preserved 51.2% of the weight and returned AB 0.529995 and AC −0.0000028 with $\lambda_h=\lambda_J=10^{-6}$ and a $2\times10^{-9}$ gradient threshold. A development comparison that assigned every missing predictor its mean logit returned AB 0.4860 and an artificial AC value of −0.0884. A missing mediator can leave information about the target in a remote predictor. The package rejects mean-predictor imputation for PLM. [Regression tests](../tests/test_plm.py)

## Optimization, resources and outputs

The CPU solver uses limited-memory BFGS with eight curvature pairs, a descent check and Armijo backtracking. It computes softmaxes after subtracting the maximum logit. Convergence requires an infinity-norm objective gradient within the declared tolerance. An iteration limit or line-search failure is recorded per target, and pairs requiring that target receive an undefined score. Defaults are 0.01 for each penalty, a $10^{-6}$ gradient threshold, 250 iterations, 256 rows per derivative block and one concurrent target fit. Regularization and derived scores change with the choices. Prespecify parameters or select them with independent held-out observations.

The likelihood evaluates coded states directly. Only a bounded group of targets is optimized concurrently. Reconciliation writes one directional canonical inter-role block per pair to a temporary file and reads it when the other direction is ready. Before fitting, the planner checks declared RAM and temporary-file budgets, each 512 MiB by default. The planned CPU work is proportional to rows, conditioning sites and target alphabet size per objective evaluation, summed across all targets. Line-search evaluations and total iterations multiply that cost. Source strings, Python object overhead and lower-level numerical libraries are outside the numerical workspace estimate. The `workers` setting bounds simultaneous target fits. Thread settings of the NumPy/BLAS distribution can also matter for support calculations and diagnostic SVDs.

Per-target diagnostics include iterations, objective, maximum gradient, sample counts, zero-weight and incomplete-row attrition, total usable weight, entropy, observed/declared states and parameter count. A weighted SVD reports centered predictor rank when there are at most 128 predictor features and 65,536 design entries. Larger designs report a rank upper bound. The penalized Hessian condition bound combines the positive ridge curvature and the categorical covariance bound with the weighted mean squared length of the complete centered design (including its intercept). These are numerical capacity diagnostics. They do not count independent evolutionary observations. [Backend](../src/triplet_coevolution/inference/plm.py) · [Objective and solver](../src/triplet_coevolution/inference/_plm_target.py)

```python
from triplet_coevolution.inference.plm import PLMOptions, fit_plm, plm_preflight

settings = PLMOptions(lambda_h=0.01, lambda_j=0.01, workers=1)
plan = plm_preflight(msa, settings, conditioning="joint")
fit = fit_plm(msa, weights, options=settings, conditioning="joint")
print(fit.diagnostics["status"], fit.diagnostics["finite_scores"])
```

Use `--dca-backend plm` and `--dca-conditioning joint` or `pairwise` in native analysis. A DCA policy JSON can include `{"backend":"plm","include_gaps":true,"plm":{"lambda_h":0.01,"lambda_j":0.01,"missing_policy":"complete"}}`. Output score coordinates remain those of the retained analysis maps.

## Independent published software comparison

An isolated double-precision build of the Marks Lab's `plmc` (MIT license), commit `18c9e55e3bd2f14f4968be19a807b401996c929a`, ran on the same synthetic gap-free protein alignment as the native fit. No external implementation is needed by the production backend. The benchmark generates 2,400 rows across six protein positions with four planted associations, equal single-site frequencies over **all 20 residues**, and unit weights. Its FASTA SHA-256 is `09236c69d2fd073c03537cbb5687b6b49b5e4811f0f9b9e5dd3cd7cf2fed3585`. It reads the external parameter file, then computes canonical zero-sum raw scores under the same protocol for both fits. [Pinned upstream source and executable instructions](https://github.com/debbiemarkslab/plmc/tree/18c9e55e3bd2f14f4968be19a807b401996c929a) · [Upstream license](https://github.com/debbiemarkslab/plmc/blob/18c9e55e3bd2f14f4968be19a807b401996c929a/LICENSE)

Native target penalties were $\lambda_h=0.008$ and $\lambda_J=0.004$. The external joint fit uses a **summed** conditional negative log likelihood and one shared copy of each pair penalty. Under equal directional coefficients, the corresponding `plmc` penalty coefficients are $N\lambda_h/2$ for fields and $N\lambda_J$ for shared pairs, both 9.6 here. We explicitly supplied the 20-letter alphabet and unit weights, disabled the group penalty, and set the external iteration ceiling to 400. External optimization reported success. All six native fits converged below $2\times10^{-8}$ maximum gradient. Target-specific fits and post-fit reconciliation generally yield a different optimum from shared symmetric coefficients.

| Comparison on 12 inter-role pairs | Result |
| --- | ---: |
| Spearman correlation of raw norm ranks | 0.993007 |
| Pearson correlation of raw norms | 0.999839 |
| Relative root-mean-squared norm difference | 1.6071% |
| Maximum absolute norm difference | 0.16407 |
| Maximum absolute signed zero-sum coefficient difference | 0.03991 nats |
| Planted pairs among the first four in each implementation | 4/4 |

The checked setup used Python 3.13.15, NumPy 2.5.5, and GCC 15.2.0 under Ubuntu for the reference program. With a separately built reference executable, run the command below from the repository root. On Windows, add `--wsl Ubuntu` and give the executable's absolute Linux path. The comparison is a small full-alphabet implementation check. Multi-complex biological generalization requires separately curated held-out systems. [Runnable reference comparison](../benchmarks/plm_reference.py)

```console
python benchmarks/plm_reference.py --plmc /absolute/path/to/plmc
```
