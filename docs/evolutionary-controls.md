# Tree-conditioned calibration and clade stability

Three commands answer different questions using the same observation IDs. `triplet calibrate` tests a score against an evolutionary model; `triplet split` assigns entire clades to training and evaluation before reading sequences; `triplet stability` measures reproducibility of scores and ranks on the evaluation clades. The bundled Newick and groups are synthetic.

After the README's mixed `triplet analyze` example, run from the repository root:

```console
triplet calibrate --run results/mixed --config examples/mixed/tree-null.json --out results/tree-null
triplet split --matches examples/mixed/matches.tsv --config examples/mixed/clade-study.json --out results/clade-splits
triplet stability --run results/mixed --config examples/mixed/clade-study.json --out results/clade-study
```

The standalone stability command reads the native analysis's original A/B/C sources and match table, verifies their digests and performs fresh fold-specific fitting. It can compute DCA or local scores even if the earlier analysis exported only MI. `--inputs paths.json` relocates byte-identical source files. For the inline workflow, pass `--null-policy examples/mixed/tree-null.json` and/or `--stability-policy examples/mixed/clade-study.json` to `analyze`; the same keys without dashes are available in project JSON. `triplet plan` inspects supplied leaves, per-worker simulation memory, source cohort and folds. Open the offline `report.html` in each result directory.

## Supply the evolutionary model

A tree policy requires an explicit local Newick, a source description and an exact observation-to-leaf correspondence. File paths are relative to the policy. Tips can be the observation IDs themselves; otherwise supply a TSV whose header is `tip_label<TAB>observation_id`. Matching preserves case and requires a bijection over the fitted cohort. Report exclusions separately before preparing the supplied tree. Every nonroot edge needs a finite, nonnegative branch length. An inferred, uncertain or outcome-selected tree changes the scientific interpretation. Describe its source and set `origin` to `independent`, `outcome_derived`, `synthetic` or `unspecified`. The output flags outcome-derived or unspecified trees as exploratory. A marker/species tree can offer independent evidence about shared lineage. Recombination, transfers and gene-specific history may violate the single-tree assumption.

The simulator draws a stationary root and evolves each site independently under a reversible equal-input continuous-time chain. Branch lengths are expected substitutions per variable site at rate scale one. Protein, DNA and RNA use their native canonical alphabets. `frequency_policy` can be `uniform`, `empirical` (observed positive-weight canonical counts and optional total `prior_mass` per site) or `supplied`. For the last option set `frequencies` to an A/B/C object or a relative JSON file, using one normalized vector per component or one vector per retained site. The state order and fitted frequencies are in the manifest. For empirical parameters the test is a fitted-model simulation conditional on those estimates. A `rate_scale` of zero freezes all edges. Constant parameters or uninformative trees can make a rejection reflect poor fit of the conservation/branch model alone. Use a biological external estimate of substitution scale or compare prespecified settings.

Each simulated replicate keeps exact tip identities, matched A/B/C rows, column masks, missing and ambiguous symbols, original coordinates and the prespecified candidate scope. Sequence-derived weights and requested scores are recomputed on the simulated alignment. Exogenous weights remain observation-keyed. The generator stays fixed within a scenario. Under this null, ancestry can induce apparent association even when sites evolve independently. A rejection indicates a departure from the full supplied independent-site model. Pair-coupled evolution on a tree and an explicit mechanistic three-body interaction require other evidence. [Substitution model, assumptions and repeated-cohort measurements](tree-model.md) · [Named calibration, correction, simulation uncertainty and checkpointing](calibration.md)

To compare genes, replace `tree.newick` and `tree.tip_map` with `tree.components` mapping A, B and C separately to `{"newick":"gene.nwk","tip_map":"roles.tsv"}` objects. All trees must map to the same observation set. Reports record rooted clade differences, unrooted Robinson–Foulds distances, a positive-edge split comparison and branch-score distances. A positive distance measures discordance; its cause requires independent evidence.

Add `alternatives` inside `tree` to prespecify up to 16 additional scenarios. Each supplies `label`, `source`, and its own `newick`/tip map or `components`. Frequency/rate options and origin inherit from the primary scenario unless overridden. Labels appear only as escaped display text. The numbered result directories contain separate manifests and corrected results. `tree_sensitivity.tsv` compares finite p/q ranges by original coordinate across the scenarios. Replicates have shared logical stream keys so identical models give identical estimates; different topologies may consume random variates differently. Monte Carlo variability contributes to range widths. Report all chosen scenarios and their provenance. Checkpoint resumption validates the entire declared design, including alternative tree hashes. [Mixed tree policy](../examples/mixed/tree-null.json)

## Hold out complete clades

Specify either a match-table `clade_column`, a `clade_tree` and integer `tree_depth`, or a previously saved `split_manifest`. Depth counts edges from the supplied root. The depth and root are biological design choices. Fewer than two independent blocks cannot support leave-one-clade-out stability. `test_groups` optionally names a prespecified holdout set. Without it, each group forms one evaluation fold. `split_members.tsv` and `split_manifest.json` preserve source observation identities, group definitions and exact disjoint training/test lists. To consume a saved partition in a later study:

```json
{"split_manifest":"../results/clade-splits/split_manifest.json","statistics":["mi_bits","dca_apc"],"top_k":3,"bootstrap_draws":39,"seed":7}
```

The saved manifest binds membership and source provenance. Phase 7 conditional model comparisons can consume the same authored partition through `validation.clade_splits.load_split_plan`. Choose the scientific evaluations and any held-out groups before viewing evaluation scores. A source panel selected using all samples still conditions the study on that upstream selection.

Stability starts with the unmasked native source columns and all original matched observations. For each fold, it estimates masks, selected candidates and any state pooling on training data alone. It estimates independent held-out scores on those same axes/candidates. Per-interface training ranks, held-out ranks, support, tie-aware top-K recovery and Spearman agreement let you distinguish loss of coverage from rank changes. Training and held-out score estimates characterize replication; predictive log loss is a separate estimand. Each bootstrap samples complete training blocks with replacement and refits weights and scores with the training feature set held fixed. Repeated blocks retain original source identities alongside distinct sampling IDs. This procedure estimates descriptive stability rather than a null p-value. A percentile interval requires at least three training clades, at least 20 draws and finite results throughout.

Options include `statistics`, `top_k`, `bootstrap_draws`, `seed`, `min_valid`, `min_valid_weight`, `min_site_coverage`, `min_site_entropy`, `ridge`, `include_gaps`, `require_reference`, `apc_universe`, `candidate_options`, `maxent_options`, `min_state_count`, `max_pair_cells`, `max_fits` and design/sequence weighting. Local triple scores use the local M1/M2/M3 hierarchy. The output includes fold/coordinate support and reasons for undefined estimates, the bootstrap draw IDs and block composition, and training-panel/selection provenance. Effective information is limited by independent blocks even when each block contains many homologs. [Clade control measurements](../benchmarks/phylogeny-controls.md)

```python
from pathlib import Path
from triplet_coevolution.clade_workflow import split_matches, stability_run

policy = Path("examples/mixed/clade-study.json")
plan = split_matches("examples/mixed/matches.tsv", "results/split-python", policy)
summary = stability_run("results/mixed", "results/study-python", policy)
```
