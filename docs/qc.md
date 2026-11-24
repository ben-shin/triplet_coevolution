# Matched-alignment quality control

A QC pass decides which **observations** and **source columns** enter scoring. The same retained observations enter A, B and C. Reference coordinates come from each complete input alignment, including a reference row outside the match table. Removing a row or column preserves the original numbering and strand.

By default, the existing column filters run once on all matched rows. A column needs at least one canonical residue, coverage ≥ 0.5, gap fraction ≤ 0.5 and missing fraction ≤ 0.5. The three fraction cutoffs remain configurable. Constant columns pass the default entropy policy. Native analysis preserves its prior error if a retained observation has no canonical residue in one component. The unattended three-family workflow can recover its legacy scoring panel by explicitly repeating a one-canonical-residue rule on all three components. [Mask implementation](../src/triplet_coevolution/msa/masking.py) · [Unattended recovery](../src/triplet_coevolution/automatic_qc.py)

## Choose a policy

Pass a local JSON file with `--qc-policy`. The following example starts by testing each component's sites. It then requires at least two canonical positions in A and canonical coverage on at least half of B's numbered reference sites. It allows conserved positions in A and B and requires two observed states in each retained C column. Group counts come from the named columns in the match table.

```json
{
  "schema_version": 1,
  "order": "columns_first",
  "iteration": "fixed_point",
  "rows": {
    "A": {"min_canonical": 2},
    "B": {"min_reference_coverage": 0.5}
  },
  "sites": {"C": {"min_states": 2}},
  "taxon_column": "taxon_id",
  "clade_column": "clade"
}
```

All thresholds are inclusive. Rule keys are optional. Component names are exactly `A`, `B` and `C`.

| Row rule | Quantity tested |
|---|---|
| `min_canonical`, `max_canonical` | Count of canonical symbols in this component's current row-test columns |
| `min_coverage` | Canonical count divided by the number of row-test columns |
| `min_reference_coverage` | Canonical count on numbered reference sites divided by the number of numbered sites among those columns |
| `min_entropy_bits` | Shannon entropy in bits of this row's observed canonical composition |
| `min_length_ratio`, `max_length_ratio` | Canonical count across all columns in this input row divided by the median full-input canonical count for its component across the original matched cohort |

Site rules accept `min_coverage`, `max_gap_fraction`, `max_missing_fraction`, `min_entropy_bits`, `min_canonical` and `min_states`. An explicit site rule replaces the corresponding global cutoff for that component. Site coverage uses all rows at the column test. Site entropy and state count use only that site's canonical symbols. A row rule requiring an undefined quantity fails the row, for example, requesting minimum reference coverage on a panel containing only reference gaps.

### Order, repetition and convergence

- `columns_first` tests all sites inside the source panel on the current cohort, then tests rows on those sites.
- `rows_first` initially tests rows on the supplied panel, then tests sites on its surviving rows. In a later pass, rows are tested on the preceding pass's columns.
- `single` stops after those two steps. Values reported on the final panel can differ from the values used at the earlier step. The TSV files give the measurements used at decision time.
- `fixed_point` repeats until the rows and resulting column panel stabilize. Removed observations stay excluded. Each column test reconsiders the full original site panel on the surviving cohort, so previously ineligible columns can return. Every pass records rows entering, leaving and retained, the dimensions used for tests and the original columns removed or restored. `max_passes` can set a stricter guard than the internal bound of input depth plus two. Reaching the guard without confirming stability blocks scoring.

## Select source columns

Use one of these TSV headers with `--site-panel`. Each component needs at least one coordinate. Both forms select sites before QC.

```tsv
component	original_column
A	3
B	8
C	2
```

```tsv
component	reference_id	reference_position
A	reference_A	42
B	reference_B	39
C	reference_C	7
```

Positions and source columns start at one. A reference key includes its exact biological ID, including when another row supplies its coordinates. A reverse-strand reference can have decreasing positions across increasing source columns. Duplicate coordinates, reference-ID mismatches, absent positions and out-of-range positions fail validation. A gap in the reference can still be selected by source column. Selecting a domain or interval is a declaration about the supplied panel. The software does not identify protein domains from composition.

## Read the artifacts

**`qc_rows.tsv`** has one record per input observation and component. Excluded observations record the deciding pass, per-component reasons and values on that pass. The three rows of a synchronized exclusion can have different reasons. Retained observations have final-panel measurements, including the number of canonical, gap, missing and reference-covered sites. The decision columns preserve the last row-test measurements. Composition entropy, terminal stretches without canonical symbols and internal gaps describe fragments and low-complexity candidates. A one-state row is flagged. Additional exclusions require an explicit threshold.

**`qc_sites.tsv`** has one row per input site, with source column, retained column, biological position, reference symbol and strand. An empty retained column means exclusion. Each row records the site's panel membership, mask reasons, cohort depth when its mask was computed, and final canonical state histogram, entropy, conservation and occupancy. Protein uses 20 canonical states, DNA uses A/C/G/T, and RNA uses A/C/G/U. The input encoder distinguishes gaps from unknown or ambiguous symbols. Native QC does not transcribe T/U. If an upstream family importer transcribes a declared RNA representation, consult its source audit.

**`qc_summary.json`** records the policy, effective site thresholds, passes and exclusion counts. The SHA-256 hashes for the input panel and final mask cover a UTF-8, LF-terminated canonical TSV with header `component<TAB>original_column` and selected columns in A/B/C, ascending source-column order. They identify coordinate membership independently of source file spelling. The run manifest separately identifies source files.

The component support totals use every retained observation and site. For AB, AC, BC and ABC, joint support gives the number of observation-by-site-combination cells where every required site is canonical. For an observation with canonical counts (a,b,c), its contributions are (ab, ac, bc, abc). Gap co-occurrence uses the same calculation with gap counts. These identities measure the complete eligible Cartesian grid without allocating triple tensors. Gap presence patterns additionally count observations with a gap anywhere in each specified component. The scorer's individual pair/candidate tables give support for a particular combination.

Exact duplicate groups use the retained encoded row per component or the full encoded ABC tuple. Input symbols with identical scoring codes can share a group. First-observation identifiers label these groups and frequencies count their members. These are descriptive counts. Neither exact duplication nor co-occurrence establishes an evolutionary relationship. Taxon/clade summaries use the requested metadata columns or the conventional `taxon_id` and `clade` when present. Missing metadata stay unavailable. Initial and retained counts permit a direct check of group attrition. An explicit missing column is an input error.

All quantities in these QC artifacts are **unweighted**. Weight sums, Kish counts and per-combination weighted support are reported by the weighting and scoring stages. [QC implementation](../src/triplet_coevolution/msa/diagnostics.py) · [Numerical and coordinate tests](../tests/test_qc_diagnostics.py)

## Memory scope

Native analysis validates the input files and builds disk-backed row inventories before it materializes the matched observations and coordinate references. The first budget check counts the sequence symbols to be selected, including raw A3M inserts. Subsequent checks estimate QC and numerical inference workspaces on the synchronized rows. SQLite caches, the largest physical input line, temporary disk space and Python overhead still contribute to resource use. [Native ingestion and measured memory](native-ingestion.md) · [Implementation ledger](implementation-ledger.md)
