# Held-out conditional pair versus pair-plus-K models

A strong planted three-site term improved held-out conditional log loss by an average 0.1091 nats per site. A weaker term gave a mean improvement of 0.0010, with an interval crossing zero. Omitting a shared mediator made a three-site term predictive in some cohorts even though the complete generating model contained fields and pairs alone.

## Model and estimand

For a prespecified panel of L categorical sites, the shared undirected log potential is

~~~text
E(x) = sum_s h_s(x_s) + sum_(s,t in edges) J_st(x_s,x_t)
                      + sum_(s,t,u in selected triples) K_stu(x_s,x_t,x_u).
~~~

An additive scalar intercept cancels from every conditional and is fixed to zero. Each field sums to zero over its states. Each pair or triple tensor sums to zero along every axis. The pair-only fit omits the final sum. The second fit re-estimates all fields and pairs together with its selected K tensors. Each selected triple requires all three of its pair faces in the specified graph. With no K terms the fitted parameters and losses coincide exactly.

For each site s the model predicts its observed state by normalizing the exponentiated potentials over the q_s possible states while keeping every other site at its observed value. Training minimizes the weighted mean negative log conditional probability over complete rows and sites, plus half the sum of squared zero-sum coefficients times the declared penalty for each interaction order. Row weights are normalized by their sum inside this objective. Weight sum and Kish count are reported separately in their original scale. Shared fields and pairs use identical penalties in both fits. Multiplying all supplied row weights by a positive constant preserves the fit given unchanged support eligibility.

The primary difference is the held-out pair-only loss minus pair-plus-K loss in **nats per site**. A positive gain favors the expanded model on this outcome. This is a comparison of conditional predictive performance. A marginal three-site statistic, biochemical interaction energy and the normalized joint likelihood each answer different questions.

## Frozen training and evaluation

The native service accepts matched A/B/C observations, their source-to-reference maps and a verified clade split manifest. Each component of the panel is specified with one-based **original alignment columns**. The engine sorts selected positions in original source order, checks their reference IDs and coordinates, and preserves unnumbered gaps. The default graph uses all pairs on this panel. An explicit edge list may include within-component edges as well as between-component edges. Other aligned positions can carry unmeasured common causes. A sparse graph describes conditioning on its declared terms.

The following library call fits a prespecified panel and one ABC term, retains the common predictions for held-out clades, and writes reusable parameter artifacts. Input variables refer to the user's matched cohort and coordinate maps.

~~~python
from triplet_coevolution.inference.nested import (
    NestedOptions, compare_nested, evaluate_model, load_model, write_nested,
)
from triplet_coevolution.validation.clade_splits import load_split_plan

plan = load_split_plan(split_manifest, [m.observation_id for m in msa.matches])
result = compare_nested(
    msa, maps, plan,
    panel={"A": [12, 37], "B": [9], "C": [22]}, triples=[(12, 9, 22)],
    options=NestedOptions(l2=0.05), bootstrap_draws=1000, seed=91,
)
manifest = write_nested(output_directory, result)
entry = manifest["model_artifacts"][plan.folds[0].fold_id]["pair_plus_k"]
model = load_model(output_directory / entry["path"])
predictions = evaluate_model(model, unseen_matched_rows, unseen_coordinate_maps)
~~~

- Training rows determine sequence-derived weights, screened triples and any rare-state pooling. The source panel, reference mapping, clade assignment and numerical settings must be fixed independently of the evaluation outcome. Uniform weights are the default. Externally supplied weights retain their source observation IDs. Canonical letters use full declared alphabets by default. A literal gap is optional. Ambiguous symbols remain missing. A positive training count threshold pools rare native states into one OTHER bin and freezes that mapping for the test set.
- Both fits use the same positive-weight rows complete across the entire panel, including sites outside the K term. Unseen observations use those same state maps and coordinates. The standalone evaluator returns individual losses for scoreable rows. The clade comparison additionally enforces the requested minimum common test count and weight. It reports training and held-out masks, weight totals, Kish counts and optimization status. A failed fit or insufficient common support withholds the comparison.
- A supplied ABC list is interpreted as prespecified. Coordinate-only uniform or exhaustive candidates and training-only triangle, wedge, entropy and hybrid screens are available. Sparse graph screening records how many proposals lacked pair faces. The selected terms and source-coordinate order are saved. The comparison assumes regularization settings were fixed before opening the test split. Reporting all settings in a declared sensitivity grid preserves that distinction. Choosing the best held-out setting makes that evaluation exploratory.
- The clade interval samples complete held-out blocks with replacement and takes percentiles of the equal-clade mean paired gain. It requires at least three contributing test blocks, 20 draws and evaluable support in every requested test block. Block membership and underlying independence require external evidence. LOCO training sets overlap. The interval conditions on the fitted fold models and gives no estimate of variance from training an entirely new model.
- The optional fitted pair-only parametric test computes a bounded exact joint partition sum, draws independent full configurations for training and test rows, and restores their original per-site missing masks. It then refits both models on each synthetic training set and recomputes the row-weighted held-out gain. It retains fixed uniform, external or clade-balanced weights and a prespecified candidate set. Data-dependent screening, state pooling or sequence-fitted weights make this calibration unavailable. P-values use (1 + conservative exceedances)/(1 + draws), with ties and unresolved fits counted against the alternative. Each fold is a separate prespecified comparison. Multiple-fold searches require an explicit multiplicity plan. This iid generator does not preserve within-clade evolutionary correlation.
- Fitting uses projected NumPy descent with an Armijo line search. Optional SciPy L-BFGS uses the same objective and projected gradient. Both require the maximum absolute projected gradient to meet the requested tolerance before contributing an evaluation. The run records objective, gradient, iteration count, coefficient count, free zero-sum dimensions and workspace estimate. Hard limits include 256 conditioning sites, configurable tensor/workspace limits and at most 65,536 exact joint states over at most 32 sites. An oversized exact law makes the generative test unavailable. Larger fitted panels can still have well-defined site-conditionals.
- Each compressed model stores portable binary64 coefficients, exact sites and bins, graph and K terms, penalties, training design, source and split identities, and convergence diagnostics. Loading verifies the schema, digest, tensor layout, dimensions and zero-sum convention. Evaluation requires the same source columns, reference IDs/positions/symbols and molecule types, while row identities may be new. Pair and expanded models share their training-design digest.

## Controlled repeated study

Seed 377210 generated 64 independent training/test cohorts for each condition. Each cohort had 120 training and 72 fresh test rows. All states were binary, weights uniform and regularization 0.025 for every order. Each fit used the same coordinates, complete rows, edges, single prespecified ABC term, gradient tolerance of 0.00001 and cap of 500 iterations. The conditional loss is averaged equally over the sites of that condition. The exact fitted iid pair-null used 39 draws per cohort and a one-sided prespecified level of 0.05. Rejection intervals are 95% Wilson intervals over independent cohorts. Gain intervals use 2,000 independent-cohort bootstrap samples. All fitted comparisons and all 12,480 null refits of paired models were evaluable.

| Generating condition | Positive gain | Mean gain in nats/site [95% interval] | Pair-null rejections [95% interval for rate] |
|---|---:|---:|---:|
| Pair-only | 23/64 | −0.00312 [−0.00553, −0.00108] | 3/64 [1.61%, 12.90%] |
| Pair plus strong K | 64/64 | 0.10910 [0.09783, 0.12085] | 64/64 [94.34%, 100%] |
| Pair plus weak K | 38/64 | 0.00100 [−0.00164, 0.00340] | 5/64 [3.38%, 17.02%] |
| Pair-only system, mediator omitted | 43/64 | 0.00325 [0.00155, 0.00501] | 10/64 [8.71%, 26.43%] |
| Pair-only system, mediator included | 29/64 | −0.00084 [−0.00150, −0.00026] | 0/64 [0%, 5.66%] |

The base law uses spins a, b and c in {−1,+1} with log potential 0.12a − 0.15b + 0.1c + 0.35ab − 0.28bc + 0.12ac. The weak and strong cases add 0.08abc and 0.55abc respectively. In the mediator cases a fourth spin m adds 1.1m + 0.95am + bm + 1.2cm. The omitted case sums m out of the full distribution. Thus its alternative can improve prediction of the observed mixture without a physical K term in the full system. Condition-specific seeds give independent samples, so the two mediator rows should be read as separate estimates. The absolute improvements also have different site denominators. These simulations provide known ground truth at one depth and regularization setting. Independent experimental validation remains open.

### Transfer to new clades

A second prespecified comparison ran the native clade-split service on 64 independent synthetic cohorts per condition. Four training clades had 30 homologs each. Three unseen clades had 24 homologs each. Each clade drew independent site fields with standard deviation 0.22. All retained the same pair coefficients of 0.25 and −0.20. The common training K coefficient was 0.55. It either persisted or reversed to −0.35 across the held-out clades. The same single triple and penalty of 0.025 were fixed in advance. State bins used a minimum of one occurrence among complete positive-weight training rows. Each cohort supplied 200 paired held-out clade bootstrap draws.

| Held-out K | Mean equal-clade gain in nats/site [95% independent-cohort bootstrap interval] | Cohorts with positive gain [95% Wilson interval] | Cohorts with positive lower three-clade percentile limit |
|---|---:|---:|---:|
| Persistent | 0.11929 [0.10635, 0.13229] | 62/64 [89.3%, 99.1%] | 52/64 |
| Reversed | −0.23749 [−0.25907, −0.21796] | 0/64 [0%, 5.7%] | 0/64 |

Every cohort was evaluable. Three held-out clades provide a coarse within-cohort percentile estimate. The intervals around the reported mean gains use 2,000 bootstrap samples of the 64 independent cohorts. These two levels of uncertainty have different sampling units.

Reproduce from an installed checkout.

~~~powershell
python benchmarks/nested_models.py --repetitions 64 --draws 39 --seed 377210
~~~

The initial iid study took 92.94 seconds with Python 3.13.15 and NumPy 2.5.5. A second run took 95.46 seconds and reproduced every rejection and gain-sign count. The native clade-transfer study took 2.37 seconds separately. The test suite checks every gradient coordinate against central finite differences, all mixed-alphabet site conditionals against direct joint enumeration, and pair-only masses against an independent exact enumerator. It also tests energy-preserving gauge transformations, exact-law mediator inclusion, sparse and fields-only fits, held-out mutation isolation, partial rows, budget failures, checksum-protected reloads and fitted-null support limits. The NumPy and optional SciPy optimizers produce matching mixed-panel conditional losses in an isolated SciPy 1.18.1 run.

