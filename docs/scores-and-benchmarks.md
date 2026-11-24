# Saving and comparing full pair scores

Each `analyze` run writes `scores/score_manifest.json` and `scores/pair_scores.npz`. The manifest includes retained/original/reference coordinate maps, reference identities, strand, units, support and missing-data policies, output schema, model settings and input hashes. The archive stores numeric arrays for each AB, AC and BC block, including observed counts and summed valid weight. A run with DCA enabled stores both DCA statistics alongside both MI statistics. The loader verifies the manifest and payload digests, dimensions, numeric types and score definitions before exposing the data. Transfer both files together.

## Choosing the statistic

| Option | Choices | Default |
|---|---|---|
| `--score` | `mi_bits`, `mi_apc` | `mi_apc` |
| `--dca-score` with `--dca` | `dca_frobenius`, `dca_apc` | `dca_apc` |
| `--threebody-score` with `--threebody` | `threebody_rms`, `co_information_bits` | `threebody_rms` |
| `--apc-universe` | `retained`, `reference` | `retained` |

Tables contain the selected statistic, units in the registry/manifest, direction, one-based display rank and competition rank. Scores tied at positions 1 and 2 both receive competition rank 1, followed by rank 3. Ties keep original coordinate order. Signed co-information ranks **lower first**, emphasizing negative values. Its one-sided null test uses that same tail. Other options rank higher first. Each ranked view can contain up to `--max-pairs` pairs per interface. `--all-pairs` writes every finite eligible pair. Both options leave fitting, correction, calibration and benchmarking on their declared universes.

The default APC uses all retained axes in each interface after base support rules. The reference option computes the correction using numbered axes only and marks corrected cells outside them unavailable. Raw values on the complete fitted block remain accessible in the archive. `--require-reference` independently controls eligibility in ranked and null outputs. Rectangular-block APC subtracts the product of the finite row and column means divided by the finite block mean. The ridge inverse-covariance DCA estimator is unchanged. [Model definitions](methods.md)

## Reusing a run

The commands below use the synthetic three-database example and write to new directories. The benchmark, export and report commands read saved results. They do not fit models or launch searches.

```shell
triplet run --config examples/workflow/run.json --out results/full
triplet benchmark --scores results/full/analysis --labels examples/workflow/synthetic_labels.tsv --merged --out results/benchmark
triplet export --scores results/full/analysis --dca-score dca_frobenius --all-pairs --out results/reranked
triplet report --scores results/full/analysis --dca-score dca_apc --plots --out results/pair-report
triplet report --run results/full/analysis --out results/analysis-report
```

`--scores` accepts the analysis directory, its score directory, or the score manifest itself. `export` generates fresh pair tables. `report --scores` builds a pair report and GraphML from that archive and optionally adds figures and a benchmark against new labels. `report --run` verifies the native output hashes, copies the required local files, and rebuilds the analysis report, including original triple rankings, permutation results and saved figures. The separate parent matching/workflow report stays with the original workflow. Triple scores and calibration tables remain ordinary files in the native analysis directory. Optional figures require the plotting extra.

### Labels, denominators and uncertainty

Provide a tab-separated file with `x_component`, `x_reference_id`, `x_reference_pos`, `y_component`, `y_reference_id`, `y_reference_pos` and `contact`. Positions are biological reference coordinates. Use `1` for positive, `0` for verified negative, or blank for unknown. Optional fields include `source`, `distance_angstrom`, `status`, `cluster_id` and `split`. Resolved/unknown/unresolved/ambiguous/excluded labels are distinguished. Training and validation labels stay outside test metrics. Duplicate or conflicting labels at one pair require prior adjudication.

`benchmark.json` reports the four statistics when DCA was fitted. AB, AC, BC and pooled comparisons share finite, explicitly labeled scores. The macro value averages interfaces with a defined metric and reports the number contributing. AUROC assigns half credit on a tie. AP groups equal thresholds. Single-class cases expose which metrics are defined. Top-K and L-based retrieval include all finite eligible pairs and retain the unknown count. An incomplete label set supports a verified positive lower bound. Natural prevalence, population precision and enrichment require comparable observable denominators. The merged TSV joins raw and corrected scores, competition ranks, percentiles, support and label status. A score is an association statistic, with no automatic probability calibration.

For repeated independent biological units, supply credible `cluster_id` labels and use `triplet benchmark --bootstrap 1000 --seed 7 …` or the native `--benchmark-bootstrap` option. Inspect the reported unit count, valid draws and interval status. Multiple residue pairs from the same structure do not provide independent system replicates. Structural contact and triple ground truth require separate validation. [Scientific release gates](implementation-ledger.md)

Use repeated `--top-k` values on `benchmark`, repeated `--benchmark-k` values on `analyze`, or a JSON list in `analysis.benchmark_k` to change retrieval cutoffs. The default evaluates 10, 25, 50 and 100 pairs together with L, L/2 and L/5. L is the length of the shorter retained, numbered interface axis.

## Migrating from 1.0–1.2

Native manifests and ranked headers now use analysis schema 2. Use `triplet analyze --legacy --dca …` to select v1.0 scalar MI arithmetic, raw Frobenius DCA ranking, retained-axis APC, and RMS triple ranking. This retains legacy numeric semantics and writes the current headers and archive. A normal run uses the batched MI kernel from 1.2. Numerically tiny changes near tied scores can change AP and ranks. Use the recorded kernel and score to interpret comparisons. `--dca-score dca_frobenius` changes just the DCA statistic. Raw and adjusted pair matrices remain available in either mode.

For a complete legacy v1 table export with its original `run_summary.json`, use `triplet export --from-v1 --scores <old-analysis> --all-pairs --out <new-directory>`. This records source hashes, recovery scope and the precision available in serialized tables. Truncated rankings cannot establish the complete universe needed for standalone metrics. Unknown observations, coordinates and labels remain unknown. Use the new artifact for subsequent reranking or benchmarks. Existing prepared inputs and matching evidence can still be passed to `analyze`. JSON configuration accepts `score`, `dca_score`, `threebody_score`, `apc_universe`, `all_pairs`, `legacy` and `benchmark_bootstrap` in its `analysis` object.

## Python

```python
from pathlib import Path
from triplet_coevolution.io.score_store import load_score_store
from triplet_coevolution.validation.benchmark import benchmark_scores

store = load_score_store(Path("results/full/analysis/scores"))
ab = store.scores["dca_apc"]["AB"]
original_a_columns = store.maps[0].original_positions
metrics = benchmark_scores(store, Path("examples/workflow/synthetic_labels.tsv"))
pooled = metrics["statistics"]["dca_apc"]["pooled"]
```
