# Gibbs bootstrap for a shared pair model

The conditional workflow can test whether adding prespecified three-site terms improves held-out conditional prediction beyond sampling variation under its fitted pair-only law. In its default mode the workflow enumerates the finite law and takes independent draws. An explicit Gibbs policy permits wider conditioning panels that pass the work, memory and mixing checks. Both modes fit the same regularized, shared-tensor categorical models. The generator comes from the fitted training-fold pair Hamiltonian.

Run the bundled synthetic example after creating the mixed analysis directory in the quickstart.

```console
triplet conditional --run results/mixed --config examples/mixed/conditional-gibbs.json --out results/gibbs-holdout
```

The same JSON can be supplied to `analyze --conditional-policy`. Python callers can use `conditional_run(source, destination, config)` from `triplet_coevolution.conditional_workflow`; already matched arrays can use `compare_nested(..., null_draws=99, null_sampler=GibbsOptions(), seed=41)` from `triplet_coevolution.inference.nested`. The bundled illustration uses 19 draws and has p-value resolution 0.05. [Runnable policy](../examples/mixed/conditional-gibbs.json)

| Policy field | Meaning and default |
| --- | --- |
| `method` | `gibbs` selects bounded simulation. Omit the object or use `{"method":"exact"}` for enumeration. |
| `burn_sweeps` | 32 complete sweeps before keeping an endpoint, allowed 1–4,096. One sweep updates every site in the declared source-site order. |
| `chains`, `diagnostic_sweeps` | Four independently seeded pilot chains with 256 post-burn sweeps per chain. Allowed 4–16 chains and an even 64–8,192 recorded sweeps. |
| `max_rhat`, `min_ess` | Maximum split R-hat 1.05 and minimum total ESS 100, applied to each monitored quantity. Allowed R-hat threshold is greater than 1 and at most 1.10. ESS must be at least 20 and within the pilot size. |
| `max_total_tv` | Maximum total variation error budget for all sampled endpoints across the planned draws and folds. The default is 0.001. Values must be positive and at most 0.01. |
| `local_i3` | `false` by default. Enable to compare local marginal connected information at 1–32 selected ABC terms under the same pair-only null. |

## Sampling and mixing

For site $i$ and state $a$, the conditional logit is $h_i(a)+\sum_{j\sim i} J_{ij}(a,x_j)$, with each edge tensor transposed when read from its second end. Subtracting the largest logit before exponentiation gives the conditional probabilities. The fitted model has zero-sum fields and couplings. A deterministic sweep leaves its positive joint categorical law invariant. A fitted directional DCA score cannot supply these shared tensors.

For each training fold the pilot starts one chain entirely in the first state at each site, one in the last, and the remaining chains independently dispersed. It monitors every per-site state indicator, every nonconstant edge's log potential and the total log potential when nonconstant. Each trace is split in half for classical variance-ratio R-hat and a multichain autocorrelation ESS. The ESS estimate truncates consecutive lag-pair sums at their first nonpositive value, applies the initial monotone sequence and caps ESS at the total recorded draws. A monitored trace without within-chain variation fails. The reported observable-wise values can help distinguish rare unvisited states and separated modes. These are raw indicator/energy diagnostics. [Vehtari et al., convergence and ESS](https://arxiv.org/abs/1903.08008)

An additional analytic check bounds bias of the generated **independent endpoint** observations. For an edge, let $D_{ij}$ be the greatest cross difference of two rows and two columns of $J_{ij}$. Changing just that neighbor shifts the target log odds by at most $D_{ij}$. The total variation change of a categorical conditional is at most $C_{ij}=\tanh(D_{ij}/4)$. To see this bound, the maximum/minimum likelihood-ratio quotient is at most $\exp(D_{ij})$. Among two distributions with that ratio constraint the greatest TV is $(\exp(D_{ij}/2)-1)/(\exp(D_{ij}/2)+1)$.

Couple a chain from its initial condition to a stationary chain with maximal conditional couplings. Initialize each site's mismatch probability bound to one. In sweep order, replace $d_i$ by $\min(1,\sum_j C_{ij}d_j)$. After burn-in, $\min(1,\sum_i d_i)$ bounds TV of one whole endpoint in exact arithmetic. The code evaluates this recurrence in binary64. Multiplying by the number of rows and replicates gives a union bound for the fold. Each fold must satisfy its equal share of the total budget. This sufficient bound can be loose for strong couplings. Passing empirical diagnostics alone cannot override the bound. [Dyer, Goldberg and Jerrum, systematic-scan influences](https://www.cambridge.org/core/journals/combinatorics-probability-and-computing/article/abs/dobrushin-conditions-and-systematic-scan/01B3E931E0C9DDE75DA9D77F1CA74FF5)

Production starts a separate independent chain per synthetic observation and keeps its final state. The training and test samples have separate random variates. Pilot trajectories are used only for diagnostics. The root seed, fold-specific RNG namespace, four uint32 seed words per pilot chain, fixed scan order, replicate key and endpoint batch size are recorded. Replicate streams use the package PCG64 scheme. Peak endpoint batch size is 256.

## Estimands, masks and limits

All synthetic rows receive their original missing-cell mask and fixed exogenous weight. A literal gap is a state only when the fitted policy includes it. Training uses exactly the original positive-weight complete training rows. Scoring uses the matching held-out complete rows. The field/pair and field/pair/K models both refit on each synthetic training dataset with the same source columns, graph, K terms, bins, regularization and optimizer. Uniform, independently supplied and fixed clade-balanced design weights are supported. Sequence-derived weights, data-dependent state pooling and outcome-based candidate screening withhold this calibration.

The primary upper-tail statistic is pair-only loss minus pair-plus-K loss, measured in nats per held-out site. An available tail has $p=(1+e)/(1+B)$, with $e$ inclusive exceedances in $B$ replicates. Wilson 95% bounds describe uncertainty in the fitted-law tail frequency due to finite replicates. Generative model misspecification, parameter uncertainty, clade dependence within observations and sampling bias outside the fixed design remain separate considerations. A mixing failure starts zero production draws. An unconverged replicate fit withholds its inferential result. The fold retains status, attempted/valid/invalid counts and diagnostic evidence.

Optional local I3 is computed on those same held-out rows, in bits. For each prespecified candidate, the code adds symmetric total prior mass one over its frozen Cartesian bins, refits the local joint and its pairwise maximum-entropy model, and measures their KL divergence. A global pair law can produce local three-way dependence after marginalizing other sites. The null regenerates that background. Candidates lacking an estimable contrast or a valid refit receive a withheld result. BH and BY cover all requested local candidates within a fold, counting unavailable hypotheses as p=1 internally. The primary gain has its own prespecified comparison. Results from different folds or post hoc comparisons require their own interpretation.

Inspect `conditional_summary.json`, each `fit_*/nested_summary.json` and the fold report. The standard nested solver's coefficient, workspace and refit limits still apply. Gibbs adds a maximum of 300 million planned update/diagnostic work units across folds and an estimated sampler workspace cap of the smaller of 256 MiB and the configured model memory. The optional local comparison has its own 300-million cell-iteration bound. Refits also use the solver workspace. The Gibbs estimate covers the sampler. The diagnostic budget accounts for all declared states. Rare states, large alphabets and tightly coupled multimodal panels can abstain even when an approximate trajectory appears plausible. The calibration command's separate `pair_panel` control still fits its bounded exact all-pair-marginal model. [Exact panel and local nulls](null-models.md)

## Synthetic validation

Run `python benchmarks/gibbs_bootstrap.py` from a checkout with the package installed. The script uses seed 319091, 8,192 independent retained endpoints per sampler control and no biological inputs. On Windows with Python 3.13.15 and NumPy 2.5.5, the following results were observed.

| Control | Result | Elapsed |
| --- | --- | ---: |
| 2×3×2 exact, all pairs | Maximum error across all normalized site conditionals $1.11\times10^{-16}$. Maximum joint frequency deviation 0.00486. Minimum monitored ESS 908, maximum split R-hat 1.0025. | 0.10 s |
| 18-site, two-state open chain, $J=0.23$ | 262,144 configurations. Across 18 one-site, 17 neighboring-agreement and one end-to-end moment, maximum error 0.01758. The simultaneous 95% Hoeffding half-width was 0.02107. Minimum ESS 750, maximum split R-hat 1.0054. | 0.55 s |
| Three-site strongly ferromagnetic triangle, $J=14$ | All ten monitored quantities failed. The run-wide TV bound was one; no endpoints were used for inference. | 0.01 s |

The wide-chain oracle uses equal stationary spin probabilities and $E[s_i s_j]=\tanh(J)^{|i-j|}$, independently of the Gibbs implementation. A separate 72-observation masked four-state fixture exercised the public nested comparison with 47 complete training and 23 complete held-out rows. At three sites the exact and Gibbs runs gave identical observed gain, −0.007379 nats/site. Their 63-draw tails were 0.8125 and 0.796875, with Wilson intervals [0.696, 0.888] and [0.678, 0.875], respectively (0.37 and 0.48 s). With nine sites and 262,144 configurations the 19-draw Gibbs comparison took 0.62 s, returned gain tail 0.85 [0.624, 0.945] and local I3 tail 0.30 [0.118, 0.488]. These small draw counts test implementation and calibration behavior. Larger fixed budgets give finer Monte Carlo resolution.
