# Prepare and run workflows

Use `triplet prepare` when your three alignments already exist and sequence rows still need pairing. Use `triplet run` to start with unaligned local FASTA databases, retrieve homolog domains, build reference-anchored alignments, match observations and score them. Both commands require a new or empty output directory.

## Run the examples

From the repository root, run either command with the installed package.

```shell
triplet prepare --config examples/workflow/prepare.json --out results/prepared
triplet run --config examples/workflow/run.json --out results/full
```

The full example includes eight matched groups, a lower-priority protein paralog and one group with missing candidates. One DNA record is on the reverse strand and one protein contains an insertion. The run enables DCA, local three-body analysis and 19 seeded permutations. Its four artificial contact labels exercise the benchmarking output. They have no physical interpretation. Open `results/full/analysis/report.html` for the results. To add PNG heatmaps, install the plotting extra and use `--plots` with a fresh output directory. `--no-plots` overrides a configuration that enables plots. [Run configuration](../examples/workflow/run.json) · [Prepare configuration](../examples/workflow/prepare.json) · [Evidence](../examples/workflow/evidence.tsv)

## Configuration

The configuration is UTF-8 JSON. Relative file paths are resolved from the directory containing the JSON file. Duplicate keys and unrecognized options raise errors. Give each alignment or database its own component entry named `A`, `B` or `C`.

| Top-level key | Meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `components` | An object with exactly the three component entries. |
| `evidence` | A local CSV or TSV of proposed group memberships. |
| `paralogs` | Optional policy for multiple available candidates. Defaults to `exclude`. |
| `analysis` | Optional object for `run`. Uses the analyze option names with underscores, including `min_valid`, `dca`, `threebody`, `threebody_model`, `global_l2`, `max_triplets`, `permutations`, `seed`, `strata_column`, `contacts` and `plots`. Paths to `contacts` and `triplet_candidates` are also config-relative. See `triplet analyze --help` for settings and defaults. |

All components require `molecule` (`protein`, `dna` or `rna`) and `reference_id`. Optional `reference_start` sets the positive 1-based number of the first occupied reference site. Prepare takes an `alignment` in aligned FASTA or Stockholm format. Run takes a `database` in ungapped FASTA format and a single-record ungapped `reference` FASTA with the configured ID. FASTA IDs are unique within each file and end at the first whitespace. Reference IDs may be outside the evidence table.

## Matching and QC

Create one row per proposed component member from your genome or orthology evidence. Each group ID identifies one prospective observation across all three components. For example, a genome ID is sufficient when you expect a single member per family in each genome. If several independent systems share a genome, distinguish their evidence groups explicitly.

| group_id | component | sequence_id | priority | cohort |
| --- | --- | --- | ---: | --- |
| isolate01 | A | a01 | 2 | batch1 |
| isolate01 | A | a01_extra | 1 | batch1 |
| isolate01 | B | b01 | 2 | batch1 |
| isolate01 | C | c01 | 2 | batch1 |

Only `group_id`, `component` and `sequence_id` are required. Each component/sequence combination may occur once across the table. Other columns, apart from optional numeric `priority`, carry group-level annotations. They must agree across every row in the group. The output preserves these annotations, so a column such as `cohort` can be used with `analysis.strata_column`.

1. Candidates qualify if their IDs occur in the input alignments for prepare or in the accepted search hits for run. The audit records unavailable candidates. Supplying a reference sequence for coordinates alone does not add it to the search hits.
2. A group needs a selected member for A, B and C. The default policy excludes a group containing multiple available members in any component.
3. With `"paralogs": "priority"`, multiple available candidates require explicit finite priorities and a unique highest priority. Missing priorities and top ties leave the component ambiguous. Search scores do not resolve a tie in the external orthology evidence.
4. Groups are ordered by group ID. The resulting `matches.tsv` gives selected IDs and group annotations. The alignments retain matched rows and the designated reference, including any reference outside the groups.

Read `matching_qc.json` to inspect each group's candidate list, priorities, availability, selected IDs and exclusion reasons. It also reports sequences outside the evidence table, aligned lengths and coverage, gap and missing fractions across matched rows. Standalone prepare writes QC even when no group qualifies. The `ready_for_analysis` field indicates whether there are matched rows.

## Search and reference coordinates

Set a component's `backend` to select one of the following searches. The default `reference` backend requires only the base Python installation. External searches require the listed executable(s) on the same environment's PATH.

| Backend | Input and executable | Selection |
| --- | --- | --- |
| `reference` | Single reference FASTA and database FASTA, any supported molecule | Local affine-gap alignment. Defaults to `min_identity` 0.5 and `min_coverage` 0.7. |
| `hmmer` | One text HMMER3 `profile` of the declared molecular alphabet. Protein uses `hmmsearch`, nucleotides use `nhmmer`. | Protein uses the reported aligned domain and independent domain E-value. Nucleotide hits include the reported strand. |
| `infernal` | One calibrated text Infernal RNA covariance model in `profile`, searched with `cmsearch`. | Accepted model hits include the reported interval and strand. |
| `psiblast` | Protein. Requires `makeblastdb` and `psiblast`. Provide a reference query or supply a PSI-BLAST checkpoint in `profile` to restart a PSSM. The reference FASTA is required for coordinates in both cases. | Final iteration HSPs. `iterations` defaults to 3. |

External backends use the positive `evalue` threshold, default 0.01. Provide a profile covering the same homologous region as the reference FASTA. Profile calibration and training take place before this workflow. Programs run locally against a supplied FASTA. PSI-BLAST builds a temporary local database and disables usage reporting. The optional adapters were exercised with HMMER 3.4, Infernal 1.1.5 and BLAST+ 2.16.0. [HMMER guide](https://eddylab.org/software/hmmer/CURRENT/Userguide.pdf) · [Infernal guide](https://eddylab.org/infernal/Userguide.pdf) · [BLAST+ manual](https://www.ncbi.nlm.nih.gov/books/NBK279690/)

Reference search tests both strands for DNA and RNA. Local alignment scores a canonical match as +2, mismatch as −1, gap opening as −4 and gap extension as −1. Ambiguous symbols use a neutral substitution score. Identity is the fraction of equal symbols among aligned canonical pairs. Coverage is the number of those pairs divided by the number of canonical sites in the reference. This backend compares every record and is intended for modest domain-sized inputs. Each pairwise alignment is limited to ten million reference-by-target cells.

For multiple qualifying hits on one database record, the workflow selects the highest score, then lowest E-value, forward strand, earliest start and earliest end. The local backend uses Biopython's first optimal traceback per strand. Hit tables record inclusive 1-based coordinates on the original database record and the strand used for retrieval. Nucleotide profile searches independently score the reference against the model to establish its orientation. They require a qualifying hit with an unambiguous best strand. The source program's strand is preserved in `model_strand`, and `strand` gives orientation relative to the supplied reference.

The workflow retrieves each selected interval by exact ID, orients it, and globally aligns it to the reference. Every reference position contributes one MSA column. A deletion appears as a gap. Target residues inserted between reference positions have no anchored column and are omitted from scoring. To study insertion columns, supply a precomputed alignment to `prepare` or `analyze`.

## Output files

| Path in a full run | Contents |
| --- | --- |
| `search/{A,B,C}_hits.tsv` | Selected intervals, strands, scores and backend-specific evidence. |
| `search/{A,B,C}.fasta` | All retrieved hits projected onto their component reference. |
| `prepared/{A,B,C}.fasta`, `prepared/matches.tsv` | Inputs to `triplet analyze`, reusable with other scoring settings. |
| `prepared/matching_qc.json` | Candidate-level audit and matched-row QC. |
| `analysis/` | Ranked tables, optional validation and plots, network, HTML report and analysis manifest. |
| `workflow_summary.json` | Package and Biopython versions, hashes of original input files, effective search settings, group totals and result paths. |

Standalone prepare writes the prepared alignments, match table, QC and workflow summary directly in `--out`. Inspect the QC before using incomplete or ambiguous datasets for analysis.
