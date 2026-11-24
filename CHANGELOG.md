# Changelog

## 1.3.0.dev0 · unreleased

- Native score archives preserve complete MI and direct-coupling matrices, eligibility, coordinates and scoring provenance. Corrected DCA rankings use the declared APC score, and legacy ordering remains available for historical comparison.
- Three-component analyses support weighted mixed-molecule alignments, selected- and exhaustive-candidate policies, local maximum-entropy models and regularized panel-conditional comparisons with held-out evaluation.
- Named null models, clade splits, clustered resampling, deterministic checkpointing and verifiable distributed shards provide distinct controls for matching, ancestry, sampling and model fit.
- Local search and annotated family inputs retain domain alternatives, input hashes, matching uncertainty and original alignment coordinates. Structure workflows map polymer sequences and report observable interface distances and coverage.
- Release checks exercise the source distribution and wheel from fresh environments, bundled examples, offline reports, scientific regressions and adverse input cases. The developer guide describes current scope and evidence.
- The minimum Biopython version is 1.87. This release includes the Bio.Entrez parser fix identified by the dependency audit. [Upstream advisory](https://osv.dev/vulnerability/PYSEC-2026-1221)

For a historical comparison, use the versioned v1 identity and score references recorded in the regression protocol. [GatABC regression](benchmarks/gatabc-regression.md)
