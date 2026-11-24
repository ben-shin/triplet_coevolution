# Calibrating a chosen score

Run the native mixed analysis from the README with `--dca` added and `--out results/mixed-dca`. Its complete archive includes raw and corrected pair scores even when the visible ranked tables show only the top results. From the repository root:

```console
triplet calibrate --run results/mixed-dca --config examples/mixed/null-policy.json --checkpoint results/mixed-null-checkpoint --out results/mixed-null --progress
```

The example shuffles intact component rows and tests MI/APC and DCA/APC in separate AB, AC and BC families. Nineteen draws suffice to exercise the exporter and recovery. For scientific reporting, choose the family, sampling assumptions, desired resolution and final number of draws before running. Read `report.html`, `calibration.tsv` and `calibration.json` in the output directory. The last file records completion, input identities, per-family resolution, fitted model settings and output hashes. The TSV gives every hypothesis in original coordinates, its observed effect in the statistic's native units, tail direction, counts, p, BH and BY q, a 95% Monte Carlo interval and optional maxT p. Rows with untestable observations contribute p=1 to multiplicity correction. Each family includes the eligible coordinate axes and records the source masks. Rank reproducibility across evolutionary groups requires a separate stability analysis.

For inline results, pass `--null-policy examples/mixed/null-policy.json` to `triplet analyze` with the desired observed methods enabled, or add `"null_policy": "null-policy.json"` under project `analysis`. `triplet plan` accepts the same options and reports family sizes and the first-rank BH/BY resolution before fitting. The inline report links its calibration artifacts. Existing `--permutations` runs its original matched-row test and legacy random stream. Choose one calibration policy per analysis. Run the standalone command in separate output directories for comparisons across nulls.

| Null | Question and assumptions |
|---|---|
| `whole_component` | Does cross-component matching explain the observed statistic? A stays anchored and the B and C row assignments are shuffled independently. Unrestricted exchangeability is assumed. |
| `within_stratum` | Uses that shuffle within explicitly named match-table strata. Set `strata_column` to a column present on every fitted observation. The manifest records group sizes, singleton count, movable fraction and both labeled-row and distinct-sequence entropy. |
| `preserve_ab`, `preserve_ac`, `preserve_bc` | Move only the remaining component. An optional `strata_column` restricts its moves. For the preserved interface, joint conditioning or reweighting can change its score even though its rows remain paired. |
| `m2_bootstrap` | Each prespecified ABC candidate has a fitted iid pairwise maximum-entropy null. Sample its joint complete rows on the archived state support, carry over their weights and refit M2 and M3. Finite-sample pair frequencies fluctuate around fitted-null expectations. |
| `pair_panel` | Fit one normalized pairwise law to the entire retained panel, generate joint iid samples, reapply fixed missingness and refit the full panel and local statistics. Exact enumeration supports 3–8 sites and at most 65,536 joint states within the local solver budget. All candidate statistics share the same simulated observations. |

For a local pairwise null use `triplet local` with `--weight-policy uniform`, prespecified coordinates, and the native source arguments. A minimal separate calibration file is:

```json
{"schema_version": 1, "null": "m2_bootstrap", "statistics": ["connected_information_bits"], "draws": 999, "seed": 7}
```

Use an independently justified sampling design for iid draws. An explicit `--weight-policy supplied --weight-column <field>` also qualifies when the policy declares `"weight_design": "fixed_design"` and these weights were fixed independently of outcomes. The same support, structural zeros, prior strength and convergence settings apply to the observed and simulated local fits. The full-panel generator currently requires unrestricted Cartesian structural support. Matching shuffles can violate structural zeros and therefore require unrestricted structural support. A sample used to train a pooled alphabet or adaptive candidate screen must be independent of the evaluation sample. For a held-out panel, supply genuinely prespecified coordinates from the independent training analysis and document that provenance. Unknown chain ancestry, duplicated lineages and a latent common cause can affect all these tests. The numerical assumptions, confounding controls and planted-signal simulations are recorded separately. [Null models](null-models.md)

## Resolution and recovery

The one-sided tail includes ties. The conservative p-value is (1 + finite exceedances + invalid draws)/(1 + executed draws). The Wilson 95% interval estimates simulation uncertainty in the null-tail probability, spanning both assignments of invalid refits. It assumes independent Monte Carlo draws conditional on this fitted null. The minimum p-value is 1/(B+1). At alpha .05, the first-ranked BH threshold across 1,000 hypotheses requires at least 19,999 draws. Multiple low p-values may cross a later BH threshold. BY uses the harmonic factor to permit arbitrary dependence. Optional `max_t: true` uses the most extreme direction-oriented statistic in each joint draw of a family. Strong familywise interpretation additionally needs subset pivotality. Per-candidate M2 sampling uses independent draws and cannot produce a joint maximum.

By default each score has its own family across all three interfaces. Set `family` to `interface` to declare separate AB, AC and BC families per score. Triple candidates form an ABC family. Correction across several separately reported families requires a prospective scientific design. Score choices are `mi_bits`, `mi_apc`, `dca_frobenius`, `dca_apc`, `threebody_rms`, `co_information_bits` (lower, more negative) and `connected_information_bits` (local maxent). Only the selected estimators are refitted. Inputs or scores fitted on a different panel have a different null hypothesis.

`seed`, `draws`, optional `planned_draws`, `workers`, `memory_mib`, `checkpoint_mib`, `checkpoint`, `resume`, `alpha` and `max_t` are supported in the policy file. Relative checkpoint paths are resolved beside that file. Set worker counts and workspace limits for the machine. Numeric workspace estimates exclude input parsing, the interpreter and BLAS allocations. `--progress` emits JSON events to stderr. The output includes `progress.jsonl`. A prefix executed with `--draws` below the prespecified final budget is labeled exploratory. Use a fresh result directory on resume; a hard process termination can leave an unpublished staging directory in the first destination.

```console
triplet calibrate --run results/mixed-dca --config examples/mixed/null-policy.json --draws 5 --checkpoint results/checkpoint-prefix --out results/null-prefix
triplet calibrate --run results/mixed-dca --config examples/mixed/null-policy.json --checkpoint results/checkpoint-prefix --resume --workers 2 --out results/null-complete
```

An interrupted process leaves completed replicate IDs in the checkpoint. Resumption verifies source code, runtime, scientific settings, score values/masks, row and coordinate identities, planned budget and checkpoint integrity before accepting prior draws. Workers and destination may change. A copied native archive can use `--inputs paths.json`, whose optional A/B/C/matches entries point to byte-identical source files relative to that JSON. `--scores results/mixed-dca/scores/score_manifest.json` identifies the same analysis; the sibling native run, matching records and original sequence inputs are required. The v1 TSV-only import supports ranking and benchmarking but contains no recoverable row-level observations. [Hard-interruption reproducibility](../benchmarks/null-recovery.md)

```python
from pathlib import Path
from triplet_coevolution.calibration import calibrate_run
from triplet_coevolution.validation.null_policy import NullPolicy

policy = NullPolicy.from_config("examples/mixed/null-policy.json")
info = calibrate_run(Path("results/mixed-dca"), Path("results/null-python"), policy)
print(info["runner"]["status"], info["families"])
```
