# Test a local three-site model

Use `triplet local` when the question concerns one site from each of three matched molecular components. The default ranking is connected information in bits, the divergence between the observed or smoothed three-site distribution and its pairwise maximum-entropy fit. A high score can arise from unmodeled population mixtures. The observed three-site joint distribution alone cannot identify a physical mechanism. [Model definitions and numerical status](maxent.md)

From the repository root, run the eight-observation mixed example. It enumerates every retained ABC coordinate, uses the same observations at all three sites, and records the sensitivity to two additional prior settings.

```sh
triplet local --a examples/mixed/protein.fasta --b examples/mixed/dna.fasta --c examples/mixed/rna.fasta --b-type dna --c-type rna --matches examples/mixed/matches.tsv --a-ref prot_ref --b-ref dna_ref --c-ref rna_ref --candidate-mode exhaustive --max-triplets 48 --local-policy examples/mixed/local-model.json --out results/local-mixed
```

At original columns A1, B2, C1 the three pairwise MIs are zero. Four joint states occur among 320 possible canonical states. With no prior, connected information is 1 bit and empirical signed co-information is −1 bit. With one unit of prior mass divided among all 320 cells the fitted connected information is 0.891915 bits. With a pseudocount of 0.5 on each cell, the prior contributes 160 weight units against eight units of data and the result falls to 0.030551 bits. All three settings use the same rows, declared support and observed counts. Other coordinates tie for the leading primary score. [Example policy](../examples/mixed/local-model.json)

Open `report.html` to see leading triples, corresponding pair MIs, fit statuses and selection coverage. Add `--plots` to the command for score heatmaps and a plot of the leading tested triples. For a resource estimate before fitting, use `triplet local --dry-run` with the same inputs and a separate output directory. The plan states whether its candidate count is exact or an upper bound until screening. [Planning and replay](analysis-planning.md)

## Analysis options

`triplet analyze --threebody --threebody-model maxent` is the equivalent native invocation. A raw-database or three-folder project can set `analysis.threebody=true`, `analysis.threebody_model="maxent"`, `analysis.local_policy="local-model.json"`, and the optional candidate settings in its existing JSON configuration. The paths are relative to the project file. `triplet_coevolution.cli.analyze` takes the same keyword arguments with underscores and `pathlib.Path` inputs. The lower-level `fit_joint`, `fit_maxent_triples` and `fit_maxent_grid` APIs support in-memory count tensors or matched alignments. [Numerical Python API](maxent.md)

| Setting | Meaning |
| --- | --- |
| `--threebody-model local` | The existing v1 projected log-linear RMS uses observed state supports and its positive per-cell pseudocount. This remains the default model for `analyze --threebody`. |
| `--threebody-model maxent` | Compare independent or support-constrained M1, pairwise maximum entropy M2 and saturated M3 on a single joint table. The default statistic is `connected_information_bits`. |
| `--threebody-model global` | Fit shared regularized conditional terms across retained sites. The RMS describes the zero-sum triple coefficient in the fitted conditional model. |
| `--threebody-score` | Choose `connected_information_bits` for maxent, `co_information_bits` for empirical signed co-information, or `threebody_rms` for the model's projected/conditional norm. Co-information ranks most negative first. Bits and natural-log RMS have different units. |

In `--local-policy FILE`, declare `prior_mode` (`total` or `per_cell`), nonnegative `prior_mass`, `tolerance`, `max_iterations`, and `max_cells`. Setting zero is primary. The optional `sensitivity` list gives additional objects changing prior mode or strength. Repeated equivalent settings are fitted once. Set `include_gaps` to add a per-component gap state for this model. Missing or ambiguous symbols are excluded from every model in the hierarchy. With `min_state_count` greater than zero, supply an explicit list of `training_ids` from the retained match table. Rare and previously unseen native symbols map into a shared OTHER state for that site. The exported map freezes the bins for all primary and sensitivity fits. For a comparison across gap or pooling policies, use a fresh run directory for each policy so all changes to rows, states and denominators remain visible.

Each optional `structural_zeros` entry has `{"original_columns":[1,2,1],"bins":[0,1,2]}` in A/B/C order. Original columns are one-based. Bin numbers are zero-based, ordered by the native alphabets, or defined by the frozen pooling map. Each entry excludes one **joint state at one selected triple** from every fitted model and the prior. Use only separately justified structural zeros. An empirical zero alone supplies no such justification. Unknown coordinates/bins, duplicate directives, empty support or an observed positive-weight row in an excluded cell generate an input error. Partial results from a failed native command are discarded.

## Select ABC candidates

`--max-triplets` limits the number fitted. `--candidate-mode uniform` uses the existing seeded uniform selection without replacement and returns the full grid when it fits. `exhaustive` requires the whole eligible grid to fit. With `--require-reference`, each selected site needs a numbered reference position. `--site-panel` and QC operate before candidate selection. The per-run selection manifest records retained and eligible grids, coordinates, both coordinate digests, seed, input identities, exclusions, screen settings and whether the cohort influenced selection.

| Mode | Selection and inputs |
| --- | --- |
| `explicit` | Supply `--triplet-candidates FILE`. Use the three original-column fields `a_original_col`, `b_original_col`, `c_original_col` or `a_reference_id`, `a_reference_pos` and the corresponding b/c fields. Positions are one-based. The source TSV must contain one valid, unique candidate per row. |
| `prior` | Supply the same coordinate file and a `prior_source` description in `--candidate-policy FILE`. Use independent evidence to assess recovery after prior-guided selection. |
| `triangle` / `wedge` | Rank the screen pool by its weakest of three finite intercomponent pair scores, or by the second strongest of at least two. Select `mi_bits`, `mi_apc`, or with `--dca` choose `dca_frobenius` or `dca_apc`. The native run reuses the full observed pair matrices for screening and reporting. |
| `entropy` | Screen by the weakest site's coverage-weighted entropy in bits with `min_entropy_bits` and `min_site_coverage`. |
| `hybrid` | Reserve a score-independent quota, then admit supplied prior candidates if present and alternate wedge and entropy proposals. `uniform_quota` is a positive count within the fit budget. |

For scored screens the optional `screen_budget` caps the pool inspected. A smaller budget samples the grid reproducibly before screening. For example, `{"schema_version":1,"screen_budget":10000,"uniform_quota":25,"pair_statistic":"mi_bits"}` selects a hybrid policy for a fit budget of at least 25. A declared `training_ids` subset refits screening weights and computes raw pair MI on the named cohort. The final fit still uses the full QC-retained observations. This overlap matters when interpreting performance. For imported coordinates declare `file_origin` as `prespecified`, `same_data` or `training`. An existing explicit file with no declaration retains the legacy prespecification assumption, recorded in its manifest.

The [candidate-discovery benchmark](../benchmarks/candidate-discovery.md) compares these selection modes at a fixed fit budget, including pure parity, a hidden clade mixture and separately simulated evaluation observations.

The existing whole-component shuffles break pair marginals. They cannot calibrate connected information against an M2 null. A maximum-entropy run with `--permutations` still runs requested pair tests and explicitly withholds triple calibration. Triple tests are also withheld for same-cohort screened candidates, including imported files declared adaptive. Predeclared v1 local/global scores retain their existing independent rematching test. Fit diagnostics, likelihood differences and prior sensitivity do not provide sampling p-values.

## Read the tables

| Artifact | Contents |
| --- | --- |
| `candidate_manifest.json`, `candidate_selection.tsv` | Exact selected source/reference coordinates, stable candidate IDs, sampling or screen rationale and score, eligible and inspected counts, source hashes and assumptions. |
| `local_state_support.json` | Original columns and reference positions associated with each frozen state map, optional training IDs, gap policy and declared joint zeros. |
| `local_model_fits.tsv` | Every selected candidate and distinct setting, including insufficient or nonconverged fits. Contains M1–M3 entropies and weighted data likelihoods, connected information, occupancy, prior contribution, solver residuals and information flags. Undefined table entries are blank. |
| `threebody_abc.tsv`, `threebody_support.tsv` | Ranked finite primary statistic, and the support of every candidate including those without an estimable primary score. Both join to the manifest by candidate ID. |
| `analysis_replay.json`, `run_summary.json` | Reusable invocation, software/input hashes, reference maps, selected scores, calibration scope, runtime and artifact checksums. |

With `--fmt csv` the table files have a `.csv` suffix. Run `triplet report --help` for offline regeneration. A sampled or screened ranking describes only the tested coordinates. The evidence for marginal local fit and the evidence for a direct conditional or physical interaction remain separate.
