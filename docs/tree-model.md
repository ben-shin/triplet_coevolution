# Independent-site evolution on a supplied tree

The `independent_site_tree` null asks whether a statistic is unusual when each retained site evolves independently but all tips inherit states along the specified genealogy. Shared internal branches can make unrelated sites appear associated at the tips. The model accepts protein, DNA and RNA in any role. Each role uses its own canonical alphabet and may have its own supplied tree.

## Transition law and units

At site $s$, let $\pi_s$ be a normalized vector of stationary state probabilities. Draw its root state from $\pi_s$. For a nonconstant site, define $h_s=1-\sum_i\pi_{si}^2$. On an edge with Newick length $t$ and a prespecified rate multiplier $r$, the probability of moving from $i$ to $j$ is

$$P_{ij}(t)=e^{-rt/h_s}\mathbf1_{i=j}+(1-e^{-rt/h_s})\pi_{sj}.$$

In simulation a descendant inherits its parent or redraws from $\pi_s$ with the indicated probabilities. Sibling branches are conditionally independent given the parent. Site histories and root states are independent across sites and components. The rate normalization gives $r$ expected substitutions per unit branch length at stationarity for every nonconstant site. A point-mass site remains constant and has zero realized substitution rate. A zero-length edge copies the parent exactly. A long edge approaches a fresh stationary draw. The model extends the F81 equal-input process to any finite native alphabet. For amino acids it provides a simple exchangeability assumption, with no empirical amino-acid exchangeability matrix. [Felsenstein, 1981](https://doi.org/10.1007/BF01734359)

Newick files must supply all nonroot edge lengths in compatible expected-substitution units. Clock time, arbitrary cladogram lengths and bootstrap support values require separate justification or conversion. Stationarity and reversibility make the joint tip law invariant to moving the root along an existing edge while preserving total path lengths. The root used to construct a clade partition still matters.

## Frequencies, gaps and weights

| Policy | How the null sets each site's stationary distribution |
| --- | --- |
| `supplied` | Fixed external values, supplied by A/B/C in canonical alphabet order. Each component accepts one vector for every site or a matrix with one row per retained column. Use independent estimates or a prespecified model and document their source. |
| `uniform` | Equal probability for every native canonical state. |
| `empirical` | Count canonical symbols with the observed analysis weights. For alphabet size $q$, add $\alpha/q$ to each count, then divide by the total canonical weight plus $\alpha$. The declared total prior mass defaults to zero. |

Empirical frequencies summarize sampled tips. Closely related observations provide less independent evidence about the stationary law. Tip frequencies do not maximize a phylogenetic likelihood. Using them makes the resulting calibration a plug-in simulation. A column with no positive canonical observation requires externally declared frequencies or positive prior mass. Audits report each site's full frequency vector, alphabet, supporting canonical weight and constant/prior-only or unsupported-data columns.

The simulator generates latent canonical states at **every mapped tip**, including observations whose original analysis weight was zero. Those observations do not contribute to empirical frequency estimates. After simulation it restores the original symbol at every gap, ambiguity and missing-data position. When downstream score settings treat gaps as states, their pattern remains fixed. This conditions the null on the observed masks. It assumes no additional state-dependent missingness mechanism beyond this conditioning. Observation and component row IDs remain unchanged, as do site and reference maps. A sequence-derived weighting method can be recomputed on the resulting matched sequences; independently fixed sampling-design weights stay attached to their observations. [Null models and score directions](null-models.md)

## Mapping and tree uncertainty

Supply exactly one complete semicolon-terminated Newick tree per source. An internal node can be bifurcating, multifurcating or unary. Branch lengths must be finite and nonnegative. Newick uses single quotes around special characters and doubles an apostrophe within a quoted label. Simple bracket annotations and internal support labels are accepted. They do not determine edge lengths. The input reader supports UTF-8 with optional BOM. The Newick file size limit is 64 MiB. [PHYLIP Newick description](https://phylipweb.github.io/phylip/newicktree.html)

All leaf names and matched observation IDs must correspond one-to-one. By default the names are equal and case-sensitive. To map separate names, supply a tab-separated file with header `tip_label` and `observation_id`, or a tip-to-observation mapping in Python. Duplicates, unmapped tips and unmatched observations fail validation. Newick traversal order may differ from the observation order. The result records the source and mapping digests, original tip labels in observation order, minimum and maximum lengths, and counts of zero-length branches and polytomies. A supplied root stem is recorded and ignored under the stationary root model.

Provide one shared species/marker tree or one explicitly identified tree for each of A, B and C. Multiple tree scenarios, including alternative branch scaling, should be prespecified and reported together. The individual tests condition on their particular trees. Provenance should distinguish independently inferred species or marker trees, outcome-derived trees and synthetic inputs. Tree choice, gene duplication, incomplete lineage sorting, horizontal transfer and uncertain branch lengths can change inference. A disagreement between trees identifies a sensitivity to examine; the disagreement alone cannot determine its cause.

Tree comparison uses matched observations in both trees. It reports the symmetric difference in rooted clades, the unrooted Robinson–Foulds count and its normalization by the number of internal splits across the two trees. The additional positive-edge count collapses zero-length resolutions. Branch-score distance is the Euclidean distance between unrooted split-length vectors over their union, including terminal edges. An absent split has zero length. Segments split by a root or unary node are summed. Descendant sets use size plus a commutative SHA-256 fingerprint to avoid quadratic memory on deep trees. An independently justified edge-depth cut can define deterministic observation strata. It depends on the supplied root and on resolution depth, and need not group comparable evolutionary distances. [Robinson and Foulds, 1981](https://doi.org/10.1016/0025-5564%2881%2990043-2) · [Kuhner and Felsenstein, 1994](https://doi.org/10.1093/oxfordjournals.molbev.a040126)

## Using the model

```python
from triplet_coevolution.validation.tree_model import load_tree, prepare_tree_null
import numpy as np

tree = load_tree("marker.nwk", [m.observation_id for m in msa.matches], tip_map="tips.tsv")
model = prepare_tree_null(msa, tree, frequency_policy="empirical", prior_mass=1.0)
simulated_msa = model.draw(np.random.default_rng(42))
```

`triplet calibrate --run RUN --config POLICY.json --out OUT` applies the archived statistic and downstream fitting choices to each simulated dataset. Select `independent_site_tree` in the policy. The `tree` object supplies a source description, origin and Newick path (plus a tip map where required). Source-relative paths in a configuration resolve from that file. Named sensitivity scenarios retain separate provenance and corrections. [Calibration workflow](calibration.md) · [Numerical model](../src/triplet_coevolution/validation/tree_model.py)

The equal-input model assumes independent sites and a common rate multiplier. It conditions on the supplied trees, stationary distributions and observation design. It does not model joint Potts states on branches, codons, indels, nonstationary composition or changing branch-specific substitution processes. A pairwise evolutionary null would require its own stationary joint law, branch transition process and validation. A significant tree-null result need not be direct physical or biochemical coupling.

## Synthetic calibration

```console
python benchmarks/tree_calibration.py --repetitions 320 --draws 99 --seed 982011
```

The study generates fresh, independent two-state site histories at 64 tips. Eight clades each have eight tips. Internal edges have length 0.8 and terminal edges 0.035, with equal stationary state frequencies. Each cohort contributes one prespecified raw AB mutual information. With fixed parameters, independent site columns with identical settings batch the Monte Carlo draws. Each marginal frequency in the empirical condition is estimated anew from the observed cohort. The alternative star has all root-to-tip distances 0.835 and removes the eight shared internal branches. The nominal level is 5% and simulated ties count as exceedances. Intervals are Wilson 95% intervals across the independent cohorts. [Reproduction script](../benchmarks/tree_calibration.py)

| Calibration law | Rejections | Estimated rate and 95% interval |
| --- | ---: | ---: |
| Generating tree and known frequencies | 16/320 | 5.0% (3.1–8.0%) |
| Generating tree and unsmoothed empirical tip frequencies | 21/320 | 6.6% (4.3–9.8%) |
| Misspecified star, known frequencies | 149/320 | 46.6% (41.2–52.0%) |
| Unrestricted matching | 132/320 | 41.3% (36.0–46.7%) |

As a strong positive control, set B's entire history equal to A's, preserving each site's marginal F81 law. The known-tree independent-site null rejected 319/320 cohorts (99.7%, 95% interval 98.3–99.9%). This measures sensitivity to perfect dependence. It gives no power estimate for weak interactions. All five comparisons ran in 5.2 seconds on a Ryzen 7 9800X3D with Python 3.13.15 and NumPy 2.5.5.

These rates describe this finite, symmetric tree and prespecified statistic. The high rates under the star and unrestricted assignment illustrate how removing the generating genealogy changes calibration. An independent numerical test also compares the simulated two-tip joint table across 32,000 sites with $\pi_i P_{ij}(t_1+t_2)$ and checks stationarity, reversibility, the transition composition rule and mean substitution rate directly. [Tree and model tests](../tests/test_tree_model.py)
