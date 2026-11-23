# From three Stockholm families to a matched alignment

Place one folder per component under `input/`. Folder names appear in the audit. Files can use `.sto`, `.sth`, `.stk`, `.stockholm`, `.seed` or `.full`, with or without `.gz`. Hidden directories are ignored. An explicit `alignment` in a component's configuration disambiguates files or accepts another filename. A multi-family file also needs a `family` selector matching its Stockholm ID or accession.

```shell
triplet evidence --input input --out results/prepared
```

Preparation is offline by default. With alignments alone, rows enter the evidence audit. Fully resolved groups produce a match table and synchronized Stockholm subsets. Unresolved mappings, competing paralogs, missing members and competing domains remain available for inspection. Preparing evidence does not fit a model. A project can produce a useful audit with zero matches.

## Folder order and settings

The default assigns A, B and C in ascending, case-sensitive Unicode order of the three folder names. The terminal and workflow manifest show the assignment. To pin the roles across folder renames or distinguish assemblies, add an `input/project.json` like the following. The orthogroup identifiers here are illustrative.

```json
{
  "schema_version": 1,
  "components": {"A": "ATPase", "B": "Partner", "C": "guide_RNA"},
  "expected_orthogroups": {"A": ["demo:A"], "B": ["demo:B"], "C": ["demo:C"]}
}
```

Use `component.json` in each folder for molecule and reference details. Molecules are `protein`, `dna` or `rna`. Leave the molecule undeclared when you only need the offline inventory and lack reliable type information. Supply `--default-molecule protein` when all undeclared components are proteins. Use the biological `reference_id` that should appear in score tables. When that identifier matches multiple domains, supply the exact Stockholm `reference_row_id`. The reference can be outside the matched panel.

Preparation checks every matched sequence and each coordinate reference against the declared scoring alphabet as it extracts their aligned fragments. Lowercase residues, supported ambiguity codes, gaps and `?` are accepted. An incompatible symbol, such as T in a declared RNA row, stops publication and identifies the component, row and fragment position. An undeclared molecule can be inventoried and exported for inspection. Scoring requires its type.

```json
{
  "molecule": "protein",
  "alignment": "family.sto.gz",
  "reference_id": "P12345",
  "reference_row_id": "P12345/42-103"
}
```

A domain suffixed `/42-103` starts at biological residue 42. Its occupied sites must span the inclusive range. A nucleotide interval such as `/103-42` assigns descending positions to its aligned symbols. If a reference has no range, the first occupied residue defaults to 1. Use a positive `reference_start` and, for a reverse-strand nucleotide reference, `"reference_strand":"-"` to supply the known orientation. A contradictory interval or offset raises an error. Original alignment columns, including reference gaps, are preserved.

## Which candidates are resolved?

- **Targeted panel.** Supply one accession, entry name or exact Stockholm row ID per line in a component's `targets.txt`. Blank lines and lines beginning with `#` are ignored. A JSON `targets` array or config-relative target file overrides this default. Absent targets are audited. An accession can expand to several domain copies. An exact domain row explicitly selects that copy.
- **Curated shortlist.** With a root `mappings.tsv` and no target list, each component looks up its manifest identifiers. The manifest describes individual identities, contexts and evidence. You do not need to preassign A/B/C triples.
- **Whole-family discovery.** With neither a shortlist nor a target list, every row of the selected family enters resolution. To explicitly include family rows beyond the manifest, set `"scope":"family"` in its `component.json`. Resolution can be expensive for large online panels. Use a target list or offline mappings when you have a defined population.

Candidates and results are conditional on the selected families and targets. A subset can omit alternative copies or paralogs present elsewhere. The inventory indexes row identifiers on temporary disk. Sequence text for unrelated rows stays outside the synchronized alignment. Selected-row output and resolver metadata scale with the selected panel. Gzip files require sequential decompression for inventory and extraction. Providing stable local files and adequate temporary disk space is important for large collections. The [Stockholm reader guide](stockholm.md) describes annotation handling and includes a user-invoked synthetic benchmark.

## Mapping evidence

An optional UTF-8, tab-separated `input/mappings.tsv` describes curated genome or locus mappings. Required column names are shown in this example. Blank mapping fields can describe an incomplete record. `protein_id` is the stable molecule or locus identifier and is also used for DNA and RNA candidates. Optional fields include `taxon_id`, `proteome_id`, `assembly_id` and `neighborhood_id`. Keep genome/assembly identities specific enough to distinguish strains. Repeat a row to record an alternative mapping or source.

| component | identifier | protein_id | genome_id | orthogroup | source |
| --- | --- | --- | --- | --- | --- |
| A | A1 | locus-A1 | genome1 | demo:A | constructed example |
| B | B1 | locus-B1 | genome1 | demo:B | constructed example |
| C | C1 | locus-C1 | genome1 | demo:C | constructed example |

Role-specific orthogroups use exact, database-qualified identifiers. You can specify `expected_orthogroups` explicitly. A configured reference with unambiguous orthology can also anchor a role. The optional `orthology_scope` restricts automatic OrthoDB anchors to a specified taxonomic scope. Read the per-mapping and per-group reasons when no role can be supported. Taxonomic co-occurrence alone does not establish orthology or a physical complex. The [evidence guide](evidence.md) gives the resolver rules, confidence levels and audit fields.

With `--uniprot`, the resolver may query public UniProt protein and proteome metadata for eligible protein identifiers. Local mappings cover non-protein loci and curated exceptions. The following command declares that all three components are proteins.

```shell
triplet evidence --input input --default-molecule protein --uniprot --out results/live-evidence
```

Accepted lookups are cached in the new output. To reuse the history offline, point `project.json`'s `cache` to that SQLite snapshot, or place a closed cache at `input/uniprot.sqlite`. Run `--offline` with a fresh output directory. Cached absences and unmatched targets remain visible. When extending an input cache in online mode, the workflow writes its new snapshot under the output directory. Absolute paths to source files, SHA-256 checksums and resolver provenance appear in the manifests. The absence of a curated or external mapping limits what can be resolved.

## Inspect and score

| Output | Contents |
| --- | --- |
| `metadata/evidence.tsv` | Comprehensive candidate and alternative evidence, including exclusions and uncertainty. |
| `metadata/groups.tsv` | Group-level decisions, reasons and number of candidates per component. |
| `metadata/selected_evidence.tsv` | Resolved-only membership records in the legacy prepare evidence schema. |
| `metadata/evidence_summary.json` | Resolver provenance, configuration and decision totals. |
| `prepared/matches.tsv` | One row per resolved observation, plus group annotations. |
| `prepared/{A,B,C}.sto` | Strictly synchronized matched domains. Each file has exactly one row per match in the same observation order. With zero matches the table is header-only and the manifest records zero exported rows and no matched alignment. |
| `prepared/*.with_reference.sto` | Optional scoring input for a component whose coordinate reference is outside the matched rows. It adds that reference after the matched observations and preserves their original columns. Replay uses the synchronized file directly when its reference is already a member. |
| `prepared/reference_coordinates.tsv` | Display name, biological ID, source row ID, original columns, symbols and reference positions for each declared reference. |
| `prepared/analyze_args.json` | Native analysis arguments when matched rows, all three types and references are available. |
| `workflow_summary.json` | Input identities, folder assignments, family and row counts, result paths and outstanding scoring requirements. |

Inspect the audit and selected rows first. To score as part of a new invocation, add `--analyze`. It runs only when matched groups, molecule types and reference coordinates are available. Otherwise it leaves the evidence in place and lists the scoring prerequisites. The prepared arguments can also be [replayed independently](workflows.md#output-files).

The [offline synthetic example](../examples/families/) contains mixed protein and RNA folders, three complete genome groups, a group with competing domain copies and a group with only one component. The candidate audit also retains an unrecognized row and an absent target. The following command uses those files in place.

```shell
triplet evidence --input examples/families --out results/example-families
```

For alignments kept in separate locations, use `--config project.json` in place of `--input`. The explicit-path format takes the same root settings, with `components` containing A, B and C objects. Each object accepts the fields from `component.json`, plus `name` for its display label. All paths are relative to the project JSON. Both formats accept optional `analysis` settings as described in the [workflow guide](workflows.md).
