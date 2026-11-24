# Clade-block stability and cross-clade replication

In a controlled clade mixture, an unrestricted shuffle rejected 128 of 128 times even though the two target sites were conditionally independent within each clade. Restricting the shuffle to the generating clades reduced that count to 4 of 128. A genuine signal confined to the training clades passed the pooled association test in every cohort and replicated as the sole top-ranked held-out pair in only 13 cohorts.

## Independent synthetic cohorts

The reproducible study uses seed 724261, 128 independently generated cohorts per condition, eight clades of 16 matched observations, and three binary sites per component. A1 and B1 form the prespecified target. Other sites are independent fair coins. Each of the nine AB pairs competes for a single training rank. The eighth clade was set aside before generation. Every test uses raw MI in bits, uniform weights, 59 randomizations, conservative ties and a fixed cutoff of 0.05. Rejection frequencies refer to this one prespecified pair. Intervals below are two-sided 95% Wilson intervals across independent simulated cohorts.

| Generating law | Unrestricted shuffle | Generating clade shuffle | Alternate tree clade shuffle | Target selected on training | Selected and sole best in held-out clade |
|---|---:|---:|---:|---:|---:|
| Independent fair coins | 6/128 [2.2%, 9.8%] | 7/128 [2.7%, 10.9%] | 7/128 [2.7%, 10.9%] | 16/128 [7.8%, 19.3%] | 0/128 [0%, 2.9%] |
| Clade mixture only | 128/128 [97.1%, 100%] | 4/128 [1.2%, 7.8%] | 128/128 [97.1%, 100%] | 128/128 [97.1%, 100%] | 14/128 [6.6%, 17.5%] |
| Persistent planted association | 128/128 [97.1%, 100%] | 128/128 [97.1%, 100%] | 128/128 [97.1%, 100%] | 128/128 [97.1%, 100%] | 100/128 [70.2%, 84.4%] |
| Association confined to training clades | 128/128 [97.1%, 100%] | 128/128 [97.1%, 100%] | 128/128 [97.1%, 100%] | 128/128 [97.1%, 100%] | 13/128 [6.0%, 16.6%] |

In the mixture, two independent Bernoulli draws share a clade-specific probability of 0.15 or 0.85. Shared composition violates unrestricted exchangeability. In the persistent model, B1 copies A1 with an independent 8% flip probability. In the training-clades-only model, the eighth clade follows the conditional-independence law. The other seven retain the planted signal. Small held-out clades have limited support, leading to rank changes even for persistent effects. The strict replication endpoint counts exact cutoff ties as failures and counts every independently generated cohort in its denominator.

The synthetic organism tree contains eight stem branches and a terminal polytomy per clade. An alternative component tree distributes two observations from each source clade into each alternative clade. Both trees use the same 128 named tips, stem length 1 and terminal length 0.1. Cutting at one topological edge from the root produces the two group assignments. The trees have eight informative unrooted splits each, zero shared informative splits, a Robinson–Foulds distance of 16, a normalized distance of 1 and a branch-score distance of 4 in these synthetic units. This alternative partition failed to control the known generating mixture. The branch lengths here define a sensitivity example. The simulation itself draws from the declared clade mixture. Independent-site branch evolution is tested in a separate study. [Tree model and F81 calibration](../docs/tree-model.md)

A discordant organism or component tree can change exchangeability, inferred conservation and which observations are held out. Horizontal transfer, duplication, differential loss, incomplete lineage sorting, alignment problems and tree estimation error can all contribute. Use an independently supported organism or marker tree and assess plausible alternatives on the same named observations. Record rooting, topology resolution, lengths and clade cuts. Polytomies and branches of zero length can limit the information in a normalized distance. Missing tips require an explicitly reconciled common cohort before comparison. Experimental ground truth and a verified organism tree for the private GatABC dataset were unavailable for this study.

Reproduce from the checkout with the package installed.

~~~powershell
python benchmarks/phylogeny_controls.py --repetitions 128 --draws 59 --seed 724261
~~~

The measured run took 6.51 seconds with Python 3.13.15 and NumPy 2.5.5. The driver reports its own runtime, both source tree hashes, topology metrics and all effect counts. No private sequence or tree was used.

## Native stability analysis

The public service accepts a synchronized source alignment, its three original/reference coordinate maps, and one externally assigned clade for each exact observation ID. The default split holds out each clade once. A prespecified list of held-out clades creates a single combined test cohort. Each block membership is fixed before score estimation. Unknown, blank or missing group IDs and a holdout leaving no training clades are rejected. A one-block LOCO design records insufficient independent blocks. Singleton clades remain represented. With the default minimum valid count of two, a singleton test fold has no eligible pair estimates.

~~~python
from triplet_coevolution.validation.clade_stability import evaluate_stability, write_stability
from triplet_coevolution.validation.clade_splits import load_split_plan

result = evaluate_stability(
    source_msa, source_maps, externally_supplied_clades,
    statistics=("mi_bits", "mi_apc", "dca_apc"),
    weight_options={"policy": "clade_balanced"},
    min_site_coverage=0.8, min_valid=4, top_k=10, bootstrap_draws=200, seed=71,
    source={"axes": "full source alignment", "clades": "prespecified external assignments"},
)
write_stability(output_directory, result)
plan = load_split_plan(output_directory / "split_manifest.json", original_observation_ids)
~~~

- **Training decisions.** Per fold, unweighted canonical coverage and entropy come from training rows only. Gaps and ambiguities stay in the row denominator. A column needs at least one canonical symbol. All three components must retain at least one training site. The workflow applies the frozen masks to the test cohort and to bootstrap samples. Weights are fitted separately within each cohort on its frozen panel. The default is uniform. All native weighting policies are available. Supplied weights need exact source observation keys and retain their absolute scale. A test fold with zero supplied weight yields unavailable scores. An archive already filtered by the full cohort supports a conditional analysis of those axes. An unconditional evaluation requires full source axes or an independently prespecified panel and reference.
- **Score and candidate support.** Raw and corrected MI and DCA use the declared pair support threshold and APC universe. Coordinate filters preserve original columns, reference IDs, offsets and unnumbered reference gaps. Optional local maximum-entropy triples use a uniform, exhaustive, triangle, wedge, entropy or hybrid screen fitted on training rows. Rare-state pooling and the candidate repertoire are frozen before evaluating the held-out cohort. Pair screens inside the candidate engine use canonical raw MI with minimum valid count two. Global model predictive performance is a separate estimand.
- **Ranking and recovery.** Each interface is ranked independently over finite scores. The held-out cohort re-estimates scores on the training-defined axes. Triple ranks use the fixed trained shortlist. The training top K uses deterministic coordinate order at ties. Average ranks define Spearman correlation on the common finite subset. It is unavailable for constant ranks or fewer than two comparable cells. The tables report support counts, common coverage, display ranks and tie sizes. Tie-adjusted recovery grants fractional membership at the held-out cutoff and zero credit for an unscored selected cell. A separate coverage column makes those missing evaluations visible. Per-interface macro summaries weight folds equally and give the contributing-fold count for every metric. Training sets overlap across LOCO folds, so those folds do not support an independent-trials interval.
- **Block uncertainty.** For G training clades each draw samples G labels uniformly with replacement, taking every matched row in each selected block. Unequal blocks give variable row counts. Every block occurrence and role sequence has a unique alias. Clade-balanced weights assign each sampled occurrence total weight one. Supplied row weights follow original observation identities into every copy. Identity-based policies recompute neighbors over the sampled rows and can downweight repeated homologs. The selected-feature interval uses the 2.5th and 97.5th percentiles of the training block refits conditional on the frozen screen. It requires at least three distinct training blocks, 20 draws and finite estimates in every draw. Otherwise the output identifies the unavailable condition and reports the finite count. These thresholds are reporting floors. A small number of genuinely independent clades still limits uncertainty estimates. The bootstrap yields descriptive score uncertainty and top-K stability. It does not generate null p-values or calibrate held-out transfer.
- **Artifacts.** The split manifest binds original observation order, evidence source, block labels, exact fold membership and SHA-256 identity. Reloading reconstructs the partition and checks the digest. The stability summary stores the source alignment and coordinate digests, each training cohort's digest, exact selected source columns, keyed weights, frozen state bins, candidate IDs and settings. The fold, selected-site and bootstrap TSV files contain results joined by fold/feature ID. Each bootstrap draw records its ordered source block labels, original-ID digest, number of distinct source blocks and actual row count. Those fields plus the split manifest reproduce the matched resample.
- **Execution bounds.** By default the source grid may contain one million AB/AC/BC pairs and the plan may require ten thousand cohort fits. The service caps split manifests at two million row assignments, selected rank/interval work at two million entries, and individual bootstrap resamples at one million rows and fifty million aligned symbols. DCA retains its state-space solver guard. The absolute number of distinct clades bounds the biological information available regardless of the number of bootstrap replicates.
