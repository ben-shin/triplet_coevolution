# Python API and project configurations

Start with `triplet init` to create a complete, offline project. Select the three alphabet types in A/B/C order. Available templates are `protein-protein-protein`, `protein-protein-rna`, `protein-rna-rna`, `rna-rna-rna` and `protein-dna-rna`. Each contains 16 synthetic observations, local matching evidence, a matched table and three aligned Stockholm references. The first columns encode a parity example. The labels in `labels.tsv` are illustrative synthetic labels.

```shell
triplet init --template protein-rna-rna --out demo
triplet run --input demo/input --offline --out demo/matched
triplet score --config demo/analysis.json --dry-run --out demo/plan
triplet score --config demo/analysis.json --out demo/scored
triplet benchmark --scores demo/scored --labels demo/labels.tsv --out demo/check
triplet select --run demo/scored --out demo/selection
triplet report --run demo/scored --out demo/offline
```

Open `demo/offline/report.html` in a browser. Its local explorer includes score distributions and maps, coordinates, full data downloads and available benchmark curves. A report built from `--run` includes the original analysis tables and diagnostics. A report built from `--scores` can choose another stored statistic or evaluate a new label set. For a native structural overlay, add `--structure path/to/structure-map-result`. The exporter checks the supplied physical coordinate identities against the scores. Export SVG without extra dependencies. Install `.[plots]` for PNG and PDF. [Explorer controls and downloads](offline-explorer.md)

## Use the same services in Python

```python
from pathlib import Path
from triplet_coevolution import api
from triplet_coevolution.templates import init_project

project = Path("example-python")
config = init_project(project, "protein-dna-rna")
request = api.AnalysisRequest.from_config(config)
plan = api.analyze(request, project / "plan", dry_run=True)
if plan.summary.get("status") == "blocked":
    raise ValueError(plan.summary)
fit = api.analyze(request, project / "fit")
tests = api.benchmark(fit.directory, project / "labels.tsv", project / "check", seed=7)
selected = api.select(fit.directory, project / "selection")
page = api.report(project / "offline", run=fit.directory, figure_formats=("svg",))
print(fit.manifest, selected.summary["selected"], page.report)
```

| Python entry point | CLI | Input and result |
| --- | --- | --- |
| `api.analyze(request, out, dry_run=False)` | `score --config`, `analyze`, `plan` | Native matched input yields an analysis or resource plan. Read `result.manifest` and `result.summary`. |
| `api.run_folders(directory, out, online=False)` | `run --input … --offline` | Three family folders yield `workflow_summary.json` and a terminal offline report. Use `summary["status"]` and `summary["exit_code"]` for batch handling. |
| `api.run(config, out)` | `prepare --config` or `run --config` | Aligned and database workflows use evidence curated for the three roles. Their schemas describe different input contracts. |
| `api.select(run, out, mode=None)` | `select --run` | Reuses the verified fitted selection when present. An explicit mode creates a separately identified candidate set. |
| `api.benchmark(scores, labels, out)` | `benchmark --scores` | Evaluates stored complete matrices. Supply the same `top_k`, `seed`, bootstrap policy and labels to reproduce in-run metrics. |
| `api.report(out, run=…)` or `api.report(out, scores=…)` | `report --run` or `report --scores` | Creates an offline bundle and returns the local page path. Files, plots and tables derive from archived scores. |

These functions return a `Result` with the destination, primary manifest and its parsed content. A plan can return a documented blocker and a folder run can finish with insufficient evidence. Malformed input, checksum changes and invalid resource settings raise ordinary Python exceptions. Use a new or empty output directory. Lower-level numerical functions operate directly on typed matched alignments and arrays and do not require a command parser. [Models and units](methods.md) · [Conditional models](conditional-models.md) · [Resumable stages](local-runs.md)

To create a native request directly, construct three `ComponentInput` values and `AnalysisRequest(a, b, c, matches, analysis={...})`. Each component needs a local alignment, a declared molecule and a biological reference ID. `reference_row_id` can identify a separate coordinate record in the file. `reference_start` is a positive biological coordinate. `reference_strand="-"` counts downward for a nucleotide reference. Native alignment and site panels retain one-based original alignment columns and biological reference coordinates separately. Gaps have no reference number. [Indexed alignment formats](native-ingestion.md) · [Coordinate and site masks](qc.md)

## Versioned configuration

The `analysis.json` written by `init` is a complete working example. `schema_version: 1` and `kind: "analysis"` select the native score configuration. Specify A/B/C in `components`, the matched table in `matches`, and native settings under `analysis`. Setting keys use underscores, for example `min_valid_weight`, `require_reference`, `dca_score`, `benchmark_k` or `candidate_policy`. Each referenced QC, model, calibration or candidate policy keeps its own validated schema. A string containing a number fails a numeric field, and unknown keys fail validation. Run `triplet analyze --help` for the corresponding switches and defaults.

Native score and configured prepare/search workflows may inherit one earlier configuration with `extends`. Relative inputs and the parent reference resolve against the file that declared them. Objects merge recursively and lists replace inherited lists. Schema versions and kinds must agree, and inheritance rejects cycles or chains longer than 16 files. Save the following as `demo/strict.json` beside the generated native config.

```json
{
  "extends": "analysis.json",
  "analysis": {
    "require_reference": true,
    "benchmark_k": [1, 3, 5],
    "max_pairs": 3
  }
}
```

Run `triplet score --config demo/strict.json --out demo/strict`. The native result records the digest of each layer. Content changes between request construction and scoring invalidate the request. Separate `project.json`, `component.json`, structural definitions and null policies keep their existing schemas and path rules.

## Migration from the released 1.0.0 CLI

Existing v1 alignments, biological references, match tables and raw search configuration version 1 remain valid. The current default selects APC-corrected DCA before top-K truncation in tables, plots, calibration and reporting. Choose `--legacy` or `"legacy": true` for the historical raw Frobenius ranking, scalar MI arithmetic and legacy whole-component permutation protocol. The newer native artifact schema adds fields, checksummed arrays, and portable ranked views. Historical byte hashes pertain to historical schemas. Numerical replay tolerances and the observed near-zero MI tie are recorded below. A complete v1 ranked directory may be imported explicitly with `benchmark --from-v1 --scores ...`; a capped or damaged set cannot recover a full universe.

Search configuration version 2 exposes individual domain hits and their exclusions. The unchanged version 1 search contract retains its historical best-hit behavior. Use `triplet run --offline` for three-folder matching with local metadata. Direct `triplet run` on bare folders may issue public identifier lookups. Read the evidence guide before choosing online mode or sharing downloaded artifacts. [Rank migration and benchmarks](scores-and-benchmarks.md) · [Search schemas](workflows.md) · [Evidence and privacy](evidence.md) · [Measured v1 replay](../benchmarks/gatabc-regression.md)
