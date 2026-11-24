# Structural mapping and independent labels

Use a local PDB or PDBx/mmCIF file and three named reference sequences. From the project root, run the bundled synthetic protein–protein–RNA example.

```shell
triplet structure-map --config examples/structure/config.json --out results/structure-example
```

Open `report.html` for mappings, interface counts, benchmarks and links. `triplet contacts` is an alias. Exit 0 means every interface has a selected copy and at least one measurable pair. Exit 3 writes a report when an assembly/model choice is still needed or some interfaces are unavailable. Exit 1 means invalid input or a failed operation. A new or empty destination is required. The Python API returns the same result dictionary via `triplet_coevolution.structure.workflow.run_structure(config_path, out_path)`.

## Coordinate sources and reference identity

Config schema 1 accepts paths relative to the JSON file. Set exactly one of `structure` (local PDB/mmCIF, optionally gzip) or `pdb_id`. An optional `sha256` pins the exact bytes, including compression. With `pdb_id`, `--download` permits a request for that specific public coordinate entry. Later calls reuse verified files from `cache` (default `.triplet-cache/structures` beside the config). Retrieval records UTC time, URL, SHA-256 and available ETag/Last-Modified values. Only the PDB identifier is sent. Supply `source_kind` as `experimental`, `predicted`, `synthetic` or `unspecified`. The summary and standalone label source explicitly identify the declared kind. Confidence scores for predicted models require independent interpretation. [RCSB file download services](https://www.rcsb.org/docs/programmatic-access/file-download-services)

Each of `references.A`, `.B`, `.C` specifies `fasta`, `id` and `molecule` (protein, dna, rna). Use `row_id` when the FASTA header token differs from the logical reference ID in a score archive. Default reference coordinates begin at 1 and increase. Set `start` and `strand` for a numbered subsequence or reverse nucleotide reference. The supplied sequence must already run in its biological orientation. Alignment gap characters are removed. Set `organism` to record agreement or disagreement with available deposited source-organism annotations. A homologous source can be evaluated explicitly. [Mixed example](../examples/structure/config.json)

Omitted assembly and model resolve only when the deposition identifies exactly one biological assembly and one model. Choose `assembly` as a string, including explicit `"asu"` when the asymmetric unit is intended. Choose the deposited `model` as an integer. The reader applies biological assembly matrices in their declared composition order and preserves separate transformed copies. It keeps the deposited author and label chain IDs, sequence indices, insertion codes and positive-occupancy atoms from the chosen residue conformer. The source coordinates remain available by checksum. [wwPDB operator-expression definition](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Items/_pdbx_struct_assembly_gen.oper_expression.html) · [PDB biological assembly and missing-residue records](https://www.wwpdb.org/documentation/file-format-content/format33/remarks2.html)

Full polymer sequences, including unobserved residues, are aligned to the three references. The mapping records substitution scheme, affine gap costs, identity, coverage, alternate placements and candidate copies. Defaults require `min_identity=0.3`, `min_coverage=0.7` and positive alignment score. Tied eligible copies require an exact `copy` in the component reference (for example `"P@I"` in the example). A validated local positional override uses `explicit_map`, a TSV with `component, reference_position, polymer_index`, along with that component’s explicit copy. Anchors must be sequence-compatible and monotonic. Unresolved alignments/atoms remain visible in the complete tables. Direct SIFTS downloads are outside this interface. Prepare independently verified correspondences in this explicit format.

## Distances, scores and controls

`distances.tsv` streams the Cartesian product of all three reference interfaces. Each row retains reference and actual structure coordinates, copy, mapping status, three available distance definitions and the chosen label. `heavy` measures the minimum between deposited nonhydrogen atoms, including nucleotide atoms. `ca` uses protein Cα; `cb` uses protein Cβ and Cα for glycine. Missing atoms yield blank measurements. Set `metric` and `close` together (default heavy at 5 Å, inclusive). `distant_metric` defaults to the same metric. `distant` defaults to 25 Å, inclusive, conditional on the chosen close test being measured and negative. The label files have different comparison populations.

| Output | Eligible population and interpretation |
| --- | --- |
| `mapping.tsv` | All reference sites with correspondence, alternate alignments and missing coordinates |
| `labels.tsv` | Every pair with the chosen metric measured. Close is 1, above the cutoff is 0, unmeasurable pairs have a blank label |
| `distant_labels.tsv` | Close versus measured distant controls. Intermediate pairs have a blank label |
| `sampled_labels.tsv` | Optional `negative_ratio` times the positive count per interface, bounded by available distant controls. `seed` fixes reservoir sampling. All positives are retained |
| `cutoff_sensitivity.tsv` | Positive counts, observable denominators and prevalence at the primary cutoff and each supplied `sensitivity_thresholds` value |
| `hard_labels.tsv` | Optional harder controls in the same interface and ordered exposure bins. Set `exposure_bins` to TSV columns `component, reference_id, reference_pos, exposure_bin` and `hard_negative_upper` to an inclusive upper distance above the close cutoff. Missing exposure and strata without controls are counted. The default requested ratio is 1 |
| `structure_summary.json` | Inputs, choices, warnings, candidate/coverage audit, sampling denominators, software identity, output hashes and completion status |

Supply `scores` pointing to a native analysis directory, verified score store or `score_manifest.json`. The pipeline joins by reference identity and physical coordinate, checks retained symbols and strand/molecule metadata, writes every stored pair score and its finite interface rank, and benchmarks each requested control universe. No model fit is required. `benchmark_all_measurable.json` includes per-interface, pooled and macro results and full curves. Curves over large datasets can occupy hundreds of MB. Standalone replay uses the same metric engine.

```shell
triplet benchmark --scores results/analysis/scores --labels results/structure-example/labels.tsv --out results/structure-replay
```

Natural prevalence uses all observable pairs in the stated interface. The scoreable prevalence further conditions on available scores. A balanced distant-control or exposure-matched sample changes prevalence and therefore the interpretation of average precision. One structure supplies no independent-system confidence interval. Compare the entire measurable interface for coverage before interpreting the sampled result. [Supplied 2G5H reconstruction and direct checks](../benchmarks/gatabc-structure.md)

## Triplets and functional measurements

Optional `triples` points to a TSV/CSV containing `a_reference_id, a_reference_pos` and corresponding B/C fields, such as `threebody_abc.tsv`. Keep candidate ranks, scores, cohort tags or predefined control identifiers in the input. `triple_geometry.tsv` preserves them and adds AB/AC/BC distances, count of close and observable edges, minimum edge and span (maximum edge). Geometries with three measured edges distinguish triangles, two-edge wedges, one-edge configurations and separated sites. Missing edges retain a partial count and an unresolved geometry. Matched control cohorts need a separately declared support and conservation design. A geometric pattern alone does not measure thermodynamic epistasis.

`functional_labels` can point to an independent assay TSV with `assay_id, source, evidence_type, assay_context, units, value, uncertainty` plus `a_reference_id, a_reference_pos` and the equivalent B/C pairs. Fill at least one coordinate pair. Leave unreported uncertainty blank. Additional columns can retain source-specific metadata and the uncertainty definition. `functional_evidence.tsv` preserves each reported value and adds actual coordinate IDs and mapping status. It stays separate from structural positive/negative labels. An independently measured three-factor energetic contrast requires all eight states measured under compatible conditions in the same energy units. The conventional contrast is G111 − G110 − G101 − G011 + G100 + G010 + G001 − G000. Propagating its uncertainty requires the measurement uncertainties and covariances. This importer does not infer missing states.

For model/copy sensitivity, the Python iterator `structure.distances.ensemble_distances` takes explicitly selected `MappingResult` alternatives for the same structure digest and references. It reports measured count, minimum/maximum distance, observed close fraction and agreement. Incompletely observed or discordant alternatives have no consensus binary label.

## Viewing actual residue selections

The report links `viewer/viewer.json` and both scripts. The viewer directory contains one transformed mmCIF per selected component. Within that directory open `view.pml` in PyMOL (or run `pymol view.pml`). In ChimeraX use `open view_chimerax.py`. Both scripts use separate copy models and deposited author residue numbers with insertion codes. `viewer_limit` is a per-interface maximum (default 5). With scores, `viewer_score` chooses a stored statistic and defaults to the analysis ranking, preferring DCA when present. Without scores, the viewer selects the nearest measured pairs. The global score ranking and the restricted measurable viewer subset have different denominators. [PyMOL selection API](https://github.com/schrodinger/pymol-open-source/blob/master/modules/pymol/selecting.py) · [ChimeraX open manager](https://github.com/RBVI/ChimeraX/blob/develop/src/bundles/open_command/src/manager.py)
