# Specification audit at v1.2.0

**Phase 1 starts with a verified ranking defect.** The ridge DCA solver already computes a raw Frobenius norm and an APC-corrected matrix. The advanced runner selects raw scores for top-K, contact metrics and permutation tests. An APC column on the resulting truncated table cannot recover a pair already excluded by raw ranking. The existing solver and raw scores provide a compatibility baseline. [Score selection and export](../src/triplet_coevolution/inference/advanced.py) · [DCA solver](../src/triplet_coevolution/inference/potts.py)

This audit describes commit `4d5ae65b57e20fd93b483c2d8825481076eecd55`, package 1.2.0, against the owner's specification for `c511759810d6a1659ea60b3022913dd646ea5093`, package 1.0.0. Status applies to the audited commit even when later commits implement a requirement. **Present** means an identifiable code path or artifact implements the stated behavior. **Partial** means a named subset exists. **Open** means the requested deliverable or acceptance evidence was missing from the tracked baseline. **Optional** marks a conditional extension. Source and test links identify evidence inspected in the checkout. Historical measurements are attributed to their original run. [Original specification, with private paths removed](implementation-specification.md)

## Sections 0–2. Baseline and comparison case

| Requirement | Finding |
|---|---|
| 0. Compatibility, validation and release authority | **Partial.** The CPU install includes NumPy, Biopython and Typer, with plotting optional. Preserve the existing CLI, Python kernels, one-to-one matching, coordinates and offline examples. The Phase 0 validation record must capture its environment, versions, clean test timings, build/install evidence and phase ledger. The 462 passed and 9 skipped in the specification belong to the historical v1.0.0 run. The owner intends free/open-source distribution and has deferred the license choice. SPDX, citation author metadata and redistribution terms require owner confirmation. [Phase 0 environment and validation](validation.md) · [Packaging](../pyproject.toml) · [CLI](../src/triplet_coevolution/cli.py) |
| 1. Changes since the supplied audit | **Present.** A disk-backed Stockholm family reader, bounded scoring panel with original site masks, synchronized row-support audit, automatic molecule/reference choice, genome-aware providers and cached offline evidence have landed. The exact neighbor and MI kernels have changed. The scientific baselines for DCA, local/global triples, supplied contact labels and stratified shuffles remain. The legacy prepared-alignment reader still constructs whole alignments. [Stockholm reader](../src/triplet_coevolution/io/stockholm.py) · [Automatic workflow](automatic.md) · [Exact-kernel measurements](performance.md) · [Legacy reader](../src/triplet_coevolution/io/alignments.py) |
| 2. GatABC regression | **Owner-supplied historical evidence.** Section 2 of the specification separates the pair-only benchmark, sampled triple fit and 19-draw null run. Its six tables, supplied-label checksum, ranks and coverage are regression targets for the private dataset. The 888 shared-proteome observations describe one selected cohort. Its balanced sampled negatives do not give all-interface prevalence, and its 1,000 triples cover only a sample. During Phase 0 the owner supplied private prepared files and historical checksums separately from Git. The supplied checksums identify the historical label file and six full score tables. The initial three uploads provided prepared inputs rather than those historical output files. Independent coordinate and distance validation requires its own mapping evidence. Consult the benchmark manifest and ledger for the status of each check. [Benchmark manifest](../benchmarks/gatabc_manifest.json) · [Implementation ledger](implementation-ledger.md) · [Supplied provenance, definitions and metrics](implementation-specification.md) |

## Section 3. Ranking, reusable benchmarks and provenance

### 3.1 Central score registry — open

MI supports a raw or blockwise-APC ranking. DCA tables carry both scores but sort on `frobenius`. Triple tables sort on local/global `threebody_rms` and also carry signed co-information. GraphML and the HTML use hard-coded advanced rank identities. PNGs cover MI. Null refits reproduce the currently selected MI statistic and raw DCA. Full-block contact evaluation happens before top-K display truncation. [Advanced runner](../src/triplet_coevolution/inference/advanced.py) · [Network ranking](../src/triplet_coevolution/reporting/network.py) · [MI plots](../src/triplet_coevolution/reporting/plots.py)

**Gate.** Define method/backend, direction, units, correction scope, conditioning, coordinate universe, missingness and interpretation for each statistic. Wire corrected DCA default and explicit raw legacy mode, MI and triple selection through CLI, config, API, tables, plots, report, network and nulls. Compute APC before slicing on the documented retained or alternate reference-eligible domain. Distinguish display order from tied metric ranks. Test a best-APC cell outside raw top-K, negative corrections, ties, zero/NaN blocks, constant sites, masked reference positions and mixed alphabets.

### 3.2 Score persistence and standalone benchmarking — partial

The binary contact TSV requires exact reference IDs, positive positions and explicit labels. Its in-memory join excludes unknowns and nonfinite scores. Tie-aware pooled AUROC and noninterpolated AP have exact small tests. Reported metrics cover the selected MI score and raw DCA. Exported ranked CSV/TSV files can be truncated. The baseline CLI has no standalone complete-score benchmark, reranker or report command. [Contact reader and metrics](../src/triplet_coevolution/validation/structure.py) · [Contact and provenance tests](../tests/test_advanced_methods.py) · [CLI](../src/triplet_coevolution/cli.py)

**Gate.** Persist lossless, versioned, checksummed full scores and reference/mask metadata. Support all-pairs export, validated full v1 TSV import with precision and coverage disclosure, and recomputation-free benchmark/export/report. Compare all four pair statistics on common labels. Add interface, macro and pooled metrics, PR/ROC curves, top-K and L-based recovery, unknown/unscored counts, prevalence-aware enrichment, distances, suitable calibration plots and system/block uncertainty. Check metric parity, degenerate label sets, tie handling, export-cap invariance and exact agreement between in-run and standalone evaluation.

### 3.3 Provenance, failure and resume — partial

Inputs have content hashes, outputs are staged, and runs record software version, settings, reference positions, retained columns, depth and relative replay files. Family runs distinguish provider failure, insufficient evidence/support and partial scoring, including the rows/sites actually used by an automatic reduced panel. Existing cache replay is specific to metadata lookup. [Analysis output](../src/triplet_coevolution/cli.py) · [Automatic outcome handling](../src/triplet_coevolution/automatic_run.py) · [Replay and failure tests](../tests/test_automatic_run.py)

**Gate.** Version each artifact contract and migrate changed meanings. Capture git SHA/dirty state, Python/dependencies/BLAS, seed streams, time, budgets, output hashes, convergence and eligibility. Separate private resolved paths from shareable replay paths. Preserve failed-stage logs, distinguish interruption and untestability, and verify configuration, code and input/output identities before resuming or publishing.

## Section 4. Pair models, QC and effective depth

### 4.1 Alignment and effective support — partial

Canonical residues, gaps and missing states differ. QC computes per-site coverage, missing/gap fractions, entropy and conservation. Pair/triple results expose joint valid counts and weights. Original-column panels preserve reference coordinates. Automatic recovery synchronously excludes observations without usable residues and records passes and exclusions. The exact legacy identity rule has an optimized symmetric packed implementation and a bounded permutation cache. The Python weighting result contains per-row weights and neighbor counts. [QC](../src/triplet_coevolution/msa/qc.py) · [Row-support recovery](../src/triplet_coevolution/automatic_qc.py) · [Weights](../src/triplet_coevolution/inference/weights.py) · [Weight tests](../tests/test_weights.py)

**Gate.** Export per-observation weights/counts and site support tied to IDs. Report duplicate clusters, component/joint coverage, fragment/domain and low-complexity outliers, gap co-patterns, reference attrition and taxonomy/clade depth. Add inspectable per-component row exclusions, explicit iterative QC policy, entropy, canonical-count and minimum valid-weight thresholds. Compare equal-component weights with per-component, length-weighted, cluster, clade and supplied weights. Report both weight sum and Kish effective count with their separate definitions. Neither estimates independent branch count by itself.

### 4.2 Mean-field DCA — partial

The centered observed-state one-hot ridge estimator mean-imputes noncanonical values. All three components jointly condition the intercomponent scores. The primal and Woodbury paths have small equivalence, indirect-chain and wide low-depth tests. The primal solve dimension is capped at 4,096 and the dual design at 16 million elements with solve dimension at most 4,096. [Estimator](../src/triplet_coevolution/inference/potts.py) · [Formulas and limits](methods.md) · [Numerical tests](../tests/test_advanced_methods.py)

**Gate.** Preserve v1 numerical behavior while exposing rank, conditioning and missing/support diagnostics. Independently test fixed-alphabet shrinkage or pseudocount conventions, gap-excluded norms and score sensitivity to state count, entropy and occupancy. Compare pair-specific fits to joint ABC on matched coordinates/support. Prespecify ridge sensitivity grids and separate tuning from held-out evaluation. A change in conditioning can reveal mediation without identifying a physical three-body term.

### 4.3 Dedicated pairwise plmDCA — open

The small global triple model uses conditional softmax and shared tensors, but the package does not expose a dedicated mixed-alphabet pairwise plmDCA backend. Implement fields/all desired couplings, separate regularizers, gauge and missing/gap policies, validated symmetric or reconciled asymmetric fits, bounded optimization, diagnostics and explicit nonconvergence. Verify gradients, fields-only and chain cases, mixed alphabet permutations, known interactions and an independent compatible reference. Compare across held-out complexes. An external protein-only comparison or accelerated path is optional and requires a dependency/license check. [Existing ridge backend](../src/triplet_coevolution/inference/potts.py) · [Existing conditional solver](../src/triplet_coevolution/inference/global_potts.py)

## Section 5. Three-body models and candidates

### 5.1 Local information decomposition — partial

The local implementation measures a zero-sum projection of the log of a smoothed joint table and signed co-information. It uses the observed states of the jointly valid rows and a fixed per-cell prior. It records occupied/possible cells and valid support. Its RMS uses natural-log units. Signed co-information uses bits. Both describe a marginal distribution. [Local scoring](../src/triplet_coevolution/inference/threebody.py) · [Closed-form tests](../tests/test_advanced_methods.py)

**Gate.** Fit independent M1, pairwise maximum-entropy M2 and saturated M3 on the same coherent joint counts and declared support. Add IPF residuals, feasibility/boundary/iteration reporting, connected information in bits, likelihood/deviance and sparsity diagnostics. Validate KL and entropy-difference equivalence when margins match. Compare total versus per-cell prior mass and quantify prior-to-data ratio, entropy, expected occupancy and uncertainty. Freeze support/rare-state pooling on prespecified or training data. Require analytic pair-only and XOR cases, noisy and latent cases, and repeated calibration before claims about irreducible dependence.

### 5.2 Nested conditional models — partial

The global estimator already fits fields, all pairs and selected ABC triple tensors on complete cases. It reports fit diagnostics and guards 64 variable sites, 250,000 coefficients and 2 million row-context indices. The native site-panel option accepts original alignment columns. A nested pair-only versus pair-plus-triple predictive comparison is still open. [Global solver](../src/triplet_coevolution/inference/global_potts.py) · [Panel masks](../src/triplet_coevolution/msa/support.py) · [Analytic global-fit tests](../tests/test_advanced_methods.py)

**Gate.** Serialize both fits and score unseen data on identical panels, weights, support, regularization conventions and train/validation splits. Keep choices training-only, report complete-case losses and held-out per-site conditional loss differences with uncertainty. Check enumerated small systems and held-out clades. Add bounded selected-edge models and validated missing-predictor alternatives, and expose omitted conditioning sites. Pseudolikelihood and predictive improvement require interpretations distinct from joint likelihood and biochemical epistasis.

### 5.3 Candidate selection and discovery — partial

Exhaustive scoring within budget, seeded uniform sampling without replacement and explicit original-coordinate candidates are available. The run records the universe size, selection mode, budget and seed. Uniform sampling leaves pure parity eligible. Its highest-scoring sampled triple has no guaranteed rank in the untested universe. [Candidate logic](../src/triplet_coevolution/inference/advanced.py) · [Sampler](../src/triplet_coevolution/inference/threebody.py) · [Candidate tests](../tests/test_advanced_methods.py)

**Gate.** Add reference-coordinate input, pair-score wedges/triangles, support/entropy, declared external priors and hybrid screening with a score-independent quota. Persist exact candidate IDs, screens, training/selection provenance, exclusions and coverage. Validate deterministic batches, planted-signal recall at fixed compute, parity, sparse and pair-only cases. Repeat adaptive selection within nulls or hold out independent clades. Structural priors used for selection must be distinguished from independent benchmark labels.

## Section 6. Statistical controls and phylogeny

### 6.1 Null hypotheses, calibration and uncertainty

| Control in the specification | Observed status and remaining test |
|---|---|
| 1. Unrestricted matching shuffle | **Present.** A stays fixed while entire B and C rows shuffle independently. Weights and enabled statistics are refit. This tests random correspondence. |
| 2. Within-stratum shuffle | **Partial.** Supplied group labels constrain the same shuffle. An entirely immovable panel fails. Add group sizes, singleton count, movable fraction and permutation entropy. |
| 3. Pair-preserving shuffle | **Open.** Implement each preserved-pair choice with its own third-component hypothesis. |
| 4. Local M2 bootstrap | **Open.** Fit M2 on fixed support, simulate the specified observation/missingness design and refit M2/M3. Exact fixed-margin sampling is optional after independent validation. |
| 5. Global pairwise-only generative null | **Open.** Fit a matched panel, validate sampling against small exact distributions, report mixing/ESS and refit both conditional models. |
| 6. Evolutionary tree controls | **Open.** Validate a supplied tree and taxon map. First test an independent-site branch model, then assess gene-tree discordance and uncertainty. A coupled tree model remains experimental pending validation. |
| 7. Clade/block uncertainty | **Open.** Add block bootstrap and leave-one-clade-out stability with explicit independent units. These estimate uncertainty or generalization. |

Serial fixed-seed nulls count numerical ties and invalid null scores conservatively. The plus-one p-value and BH correction cover full prespecified eligible families, including p=1 for untestable observed cells before output filtering. **Open gates** include per-statistic calibration (especially DCA/APC), test manifests, invalid/eligible counts, family definitions, p-resolution and Monte Carlo intervals, BY or justified max-T, effects and rank stability. For an isolated first-ranked rejection, the BH threshold is α/M and the smallest fixed-draw p is 1/(B+1). This bound is different from guaranteed power. Validate independent or resumed streams, clade-confounded and pair-only simulations, repeated type-I error and power intervals. Any adaptive stopping needs valid sequential inference. [Permutation refits](../src/triplet_coevolution/validation/nulls.py) · [Strata](../src/triplet_coevolution/validation/phylogeny.py) · [BH](../src/triplet_coevolution/validation/fdr.py) · [Tests](../tests/test_advanced_methods.py)

## Section 7. Structure, experimental labels and independent systems — open

| Deliverable | Baseline and acceptance evidence still needed |
|---|---|
| Structure to reference | The input today is a supplied binary reference-coordinate label table. Build local PDB/mmCIF assembly/chain/entity/copy/model selection, sequence-to-polymer alignment and polymer-to-resolved-atom mapping. Track author and label IDs, insertions, missing residues, altlocs, modified bases, symmetry transforms and ambiguous mappings. Accept justified mapping overrides/SIFTS and abstain when unresolved. Remote retrieval is optional. |
| Distances and labels | Produce complete observable intercomponent distance matrices and masks, metric-specific cutoff classes, mixed-polymer atom definitions and chain provenance. Add cutoff, ensemble and hard-negative sensitivity. Check direct brute-force distances on an adversarial fixture. Reconcile supplied 2G5H labels separately from a fresh full map once authorized files arrive. Unknown or unresolved pairs stay outside confirmed negatives. |
| Three-site and experimental evidence | Add wedge/triangle/span measures and controls matched on support, conservation and pair scores. Distinguish deposited experimental structures, predicted structures, synthetic truth and mapped functional assays. Register measured units, assay conditions, uncertainty and the exact perturbation design before interpreting an energetic three-way cycle. Contact geometry alone cannot validate irreducible statistical or energetic coupling. |
| Independent validation and visualization | Curate multiple independently mapped complexes and lineages across protein and nucleic-acid compositions, hard paralogs and negative controls. Prespecify tuning and test partitions, dataset hashes and simple comparator scores. Produce author-residue-aware PyMOL/ChimeraX exports, verify the selection syntax and report independent precision/coverage. Baseline examples are synthetic. |

The existing reader and tie-aware metric routine are reusable checks on explicit labels. The repository contains no independently verified 2G5H atom-distance result or multisystem held-out validation at this revision. [Label schema and metrics](../src/triplet_coevolution/validation/structure.py) · [Synthetic labels](../examples/workflow/synthetic_labels.tsv)

## Section 8. Homolog discovery and biological correspondence

### 8.1 Search and alignment — partial

Existing local backends cover portable reference search/projection, HMMER protein/nucleotide search, Infernal CM search and PSI-BLAST, including PSI-BLAST iteration settings. Commands use argument arrays. The workflow records executable versions, hit/model spans and representative accepted placements. Genuine profile-guided reference projection and a full profile-aware multiple alignment are different deliverables. [Search dispatch and hit selection](../src/triplet_coevolution/search/retrieve.py) · [PSI-BLAST](../src/triplet_coevolution/search/psiblast.py) · [Program checks](../src/triplet_coevolution/search/shared.py) · [External-engine tests](../tests/test_profile_search.py)

**Gate.** Add iterative JACKHMMER with inclusion/target thresholds, iteration diagnostics, stable final hit import, database and tool identity, checkpoints and resource controls. Import existing search tables and A3M with explicit insertion semantics. Add appropriate hmmalign/cmalign alignments and optionally MAFFT. Preserve domain copies, orientation, rejected hits and reasons. Test fragment/length/complexity, cross-family contamination, reference/domain coverage, overextension and database-size effects. Reciprocal evidence is optional.

### 8.2 Genome-aware evidence — substantial partial implementation

Curated matches, explicit exclude/priority handling and conservative genome/neighborhood pairing exist. Stockholm row IDs and domain intervals, protein aliases/secondary accessions, versioned assemblies, verified synonyms, RNA loci, role confidence, candidate-level records and group exclusions are available. Exact row correspondence survives extraction. Direct UniProt, ENA, NCBI and RNAcentral lookup has bounded batches, cached records, provider errors and offline replay. Folder runs expose an offline switch and document when public identifier lookups are enabled. Weaker family-profile evidence is distinguished from orthogroup support. [Evidence rules](evidence.md) · [Resolver](../src/triplet_coevolution/evidence/generate.py) · [Ambiguous placement tests](../tests/test_incomplete_placements.py) · [Nucleotide provider tests](../tests/test_nucleotide_evidence.py)

**Gate.** Complete reusable bulk metadata/annotation import, stronger versioned provenance and configurable network, refresh, progress, retry and resume controls. Direct retrieval is an alternative to asynchronous ID mapping. If async jobs are added, test terminal failure/resubmission, paging, job expiration and checkpoint recovery. Add independent GFF3/strand, synteny/operon, domain architecture, compartment and RNA copy evidence as appropriate, with explicit confidence and abstention. A neighborhood label today is a supplied assertion. Any covariance-assisted paralog assignment needs independent training and validation. Compare the legacy cohort of 888 against supplied matches, and give evidence-derived cohorts separate identities.

## Section 9. Runtime, scalability and operations

### 9.1 Preflight and bounded I/O — partial

The automatic runner estimates dimensions/work, applies explicit model budgets, audits reduced scoring panels and preserves full prepared evidence. The separate Stockholm family reader supports plain/gzip and interleaved records, indexes on disk, validates block structure, preserves relevant GF/GS/GC/GR annotations and selects exact IDs plus the reference. Its documented gzip synthetic test used about 30 MiB process peak for 40,000 rows × 2,048 columns. The legacy FASTA/Stockholm `read_msa` and configuration-driven `prepare` still materialize alignments. [Bounded-reader evidence](stockholm.md) · [Automatic panel](../src/triplet_coevolution/automatic_panel.py) · [Legacy workflow](../src/triplet_coevolution/workflow.py)

**Gate.** Add user-invokable doctor, validation and plan/dry-run, with tools, input formats/sizes, observed versus estimated support/dimensions, allocation/disk/cores, output checks and attainable permutation resolution. Extend bounded selection across supported prep paths including FASTA. Preserve symbol-validation scope, reference coordinates, input changes and interrupted-write behavior. Provide compact/chunked full scores and scalable rendering. Quantify memory and I/O on representative authorized workloads.

### 9.2 Optimization and parity — partial with measured improvements

Packed symmetric exact comparisons, a capped component count cache and batchwise MI are implemented. The checked-in synthetic sweep compares 256, 768, 2,048 and 4,096 matched rows against `c511759`, with exact neighbor, weight, p and q comparison and score tolerance 10⁻¹². Its three-repeat medians at 4,096 rows improved from 2.7822 to 0.5937 seconds for weights and from 16.3904 to 4.6522 seconds for five-draw MI/local-triple calibration. These are historical measurements on a specified Windows host. RSS, workload dimensions and limitations accompany them. [Performance protocol and measurements](performance.md) · [Runner](../benchmarks/matched_scaling.py) · [Parity and runner tests](../tests/test_matched_scaling.py)

**Gate.** Extend profiling to step-specific CPU threads, I/O, throughput and uncertainty, full-score persistence and budget-based ranking, candidate batches/sparse tables and held-out screen recall. Keep exact numeric and coordinate parity. Validate controlled BLAS/dual resource use and optional parallel, compiled or GPU paths before enabling them. This sweep does not measure full GatABC, plmDCA or global-triple performance.

### 9.3 Parallel nulls, resume and HPC — open

Nulls use one sequential seeded generator and online exceedance counters. Metadata caches and staged end results cannot reconstruct an interrupted hypothesis test. Implement immutable stage/replicate RNG identities, budgeted worker pools, safe shared read-only data, atomic checkpoints, exact disjoint-shard merge and validated fixed-plan extensions. Test forced termination, corrupt/mismatched state, duplicate/missing replicates, disk failure and serial/parallel/resumed equivalence. Add non-TTY progress, structured logs, rate/ETA and stage commands. A configurable Slurm array example is optional after local replay and merge are proven. [Null loop](../src/triplet_coevolution/validation/nulls.py) · [Current automatic lifecycle](../src/triplet_coevolution/automatic_run.py)

## Section 10. User experience, APIs and reporting — partial

JSON projects use config-relative paths and reject many unknown or invalid values. The CLI exposes aligned analysis, preparation, configured search/workflows and unattended family runs. Offline HTML links ranked tables and reports evidence, masks, selected scope and omissions. The GraphML exporter already distinguishes pair edges from membership edges to a triple factor node. Optional MI PNGs use reference-aware axes. The Python code has callable scientific kernels, while workflow analysis still invokes the CLI callback. [Configuration](../src/triplet_coevolution/config.py) · [Workflow](../src/triplet_coevolution/workflow.py) · [Report tests](../tests/test_report.py) · [GraphML](../src/triplet_coevolution/reporting/network.py)

**Gate.** Unify typed CLI/config/API declarations and schema migrations, supply reusable scientific stages and templates for each requested molecular composition. Add external-tool timeout and cleanup controls. Extend report filters, search and downloads to complete scores. Distinguish NaN/zero, rank statistic/direction, units, coverage, priors, candidate selection and experimental/predicted label sources. Add raw/corrected maps with appropriate scales, MI–DCA rank plots, structure overlays, ROC/PR/enrichment, I3/nested/clade/null uncertainty and diagnostics. Preserve accessible offline use and GraphML semantics, validate Cytoscape and structure-viewer exports, SVG/PDF/PNG as relevant, and regenerate from artifacts without inference.

## Section 11. Scientific testing and release engineering

### 11.1 Tests beyond smoke tests — partial

Tracked tests cover alignment alphabets, invalid/interleaved Stockholm and gzip, reference/strand numbering, masking, matching ambiguity and secondary IDs, external adapter fixtures, analytic local/global parity, mediated pairs, primal/dual equivalence, large numerical limits, tie-aware metrics, undefined draws, seeded replay, offline reports and evidence-provider mock responses. Existing tests provide a compatibility harness. Their presence alone does not establish repeated-study error rates. [Scientific cases](../tests/test_advanced_methods.py) · [Stockholm tests](../tests/test_stockholm_subset.py) · [Provider tests](../tests/test_uniprot_evidence.py) · [Workflow tests](../tests/test_workflow.py)

**Gate.** Add independent exact/gradient checks, mixed alphabet and state/row/component metamorphic tests, explicit score/coordinate migration and cap invariance, unavailable or corrupt checkpoints, merge tests and structural-distance fixtures. Run prespecified repeated simulations across depth, coupling, ancestry, missingness, paralogs and selection, reporting uncertainty, failures and seeds. Separate fast deterministic CI from slow statistical/performance checks. Record branch coverage, installed artifacts, actual external skips and optional authorized real-data tests.

### 11.2 Engineering and release metadata — partial

Setuptools metadata, CPU core/optional plotting dependencies, CLI entry point/version, source packaging, README, equations, workflow docs and synthetic examples exist. At the audited SHA the tracked tree lacked CI workflows, changelog, contributor guide, citation metadata and license. The package contains numerical, I/O and evidence modules but some orchestration and reporting remain coupled. [Packaging](../pyproject.toml) · [README](../README.md) · [Methods](methods.md)

**Gate.** Add typed service boundaries, manifest contracts, gradual type/format checks, dependencies/security audit, least-privilege CI across supported platforms and Python versions, wheel/editable installs, docs/example tests and scheduled slow checks. Deliver changelog, contribution, API, migration, privacy/security, troubleshooting and reproducibility documentation. Citation metadata must be owner-verified. License/SPDX and third-party data distribution await owner decisions. Lockfiles, environment or container recipes are optional. No automatic publication is authorized.

## Section 12. Phase gates at the audited commit

| Phase | Status and dependency to close |
|---:|---|
| 0 | **Partial at baseline.** Pin SHA, reproduce clean test/build/install results and environment, account for skips, record benchmark hashes, files supplied and any remaining gaps. This audit and the accompanying validation records belong to the subsequent work package. |
| 1 | **Open.** Unify rank semantics and show that corrected top-K includes pairs outside raw top-K. Round-trip complete scores, four statistics, interface/macro metrics and raw legacy outputs. |
| 2 | **Open.** Independently map assembly/chain residues, validate full distances and observable denominators. Gate the separate 2G5H comparison on actual data availability. |
| 3 | **Partial.** Existing original-column panels, row-support exclusions, exact weights and column masks seed a comprehensive QC, weighted support and memory-preflight release. |
| 4 | **Open.** Deliver common-support M2/IPF, I3, priors, selection manifests and independent XOR, latent and pair-only truth cases. |
| 5 | **Partial.** Retain the existing fixed-seed shuffle/BH reference. Implement calibrated statistic selection, distinct hypotheses, resolution warnings, progress, parallelism and replay. |
| 6 | **Open.** Validate clade splits, tree simulation, clade bootstrap and repeated calibration with assumptions and uncertainty. Supplied strata are a starting point. |
| 7 | **Open.** Implement plmDCA and comparable nested predictive models with optimizer and allocation checks, independent tests and held-out evaluation. |
| 8 | **Partial.** Disk-backed Stockholm and genome matching cover important parts. Generalize ingestion and add iterative JACKHMMER, hit reuse, alignment backends and independent pairing validation. |
| 9 | **Partial.** Exact weight/MI speedups and synthetic RSS measurements exist. Prove resource-budgeted restart, deterministic worker/shard merge, monitoring and end-to-end operation. |
| 10 | **Partial.** Extend reports, API, packaging, CI and migration documentation for each validated upstream gate. Owner decisions remain separate release prerequisites. |
| 11 | **Open.** Prespecify independent systems/clades and negative controls, verify label provenance, test calibration and incremental predictive value with uncertainty, and report negative results. |

## Sections 12–14. Release evidence and first work package

Release evidence must cover legacy reproduction and corrected ranks, score definitions across molecular compositions, and calibration of scientific claims. Independent structure mappings require documented observable coverage. Verify mathematical identities on synthetic data and compare predictions on held-out systems. Record resource limits, restart and offline behavior alongside tests, lint, packaging, examples and performance. Record which GatABC hashes, metrics and independent scientific claims were verified, along with the inputs each check used.

Section 13 prioritizes the score registry, full-score persistence and reusable benchmark before structural mapping. Sections 5–6 require trustworthy score/structure infrastructure for comparative claims. Section 14 identifies the original method sources and historical audit provenance. The present audit checks tracked code and artifacts. [Owner's release criteria, implementation order and source list](implementation-specification.md)
