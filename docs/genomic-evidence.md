# Local genomic evidence and lookup controls

Place one Stockholm alignment in each of the three input folders. A project may supply a local GFF3 file per assembly, including gzip-compressed files. Annotations stay local. The command below uses only these files, family metadata and any local lookup snapshots.

```console
triplet run --input input --offline --out results/local-assembly
```

For an annotation-only handoff use `triplet evidence --input input --out results/evidence`. Set the component molecule in `component.json` when the family cannot identify it. The evidence command uses the configured expected orthogroups or a supplied anchor to qualify each role. The unattended run also supports the distinct family-profile tier. [Three-folder workflow](automatic.md) · [Role and identity rules](evidence.md)

The following example shows the annotation entries in `input/project.json`. Specify one source for each distinct assembly. Relative paths start at the project JSON. Omit `assembly_id` if the file has a versioned NCBI `#!genome-build-accession` pragma. When both are present, they must agree.

```json
{
  "schema_version": 1,
  "expected_orthogroups": {
    "A": ["local:family-A"], "B": ["local:family-B"], "C": ["local:family-C"]
  },
  "gff3": [{
    "path": "../annotation/genome.gff3.gz",
    "assembly_id": "GCF_000000001.1",
    "taxon_id": "12345",
    "release": "example-2026-01",
    "orthogroup_attribute": "orthogroup",
    "system_attribute": "operon",
    "system_windows": "../annotation/systems.tsv"
  }]
}
```

This example keeps the annotation directory beside the input project. The optional `components` list limits a source to named roles. The release and taxonomy fields are optional source annotations. A numeric taxon alone cannot identify an assembly. The roles and accessions in the example are illustrative.

The parser reads GFF3 sequence features up to `##FASTA`, including percent-escaped attribute values. It checks the version, ID/strand consistency and 1-based closed intervals, and keeps multipart segment coordinates. It indexes `ID`, `Alias`, `locus_tag`, `protein_id`, `transcript_id` and supported accession-specific Dbxref entries. For a single-segment nucleotide locus, `replicon/start-end` is an exact strand-aware selector, including a descending reverse-strand interval. Matching a protein identifier to a CDS requires an explicitly annotated alias. The physical locus, original alignment row ID, domain suffix and biological accession have separate meanings. [Sequence Ontology GFF3 specification](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md) · [NCBI GFF3 conventions](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/file-formats/annotation-files/about-ncbi-gff3/)

**System assignment.** A named operon can be inherited through a same-replicon, compatible-strand, containing GFF3 parent. A provided named attribute or an explicit window can also place a locus. Each optional window TSV has the exact header below, followed by data rows separated by tabs.

```text
assembly_id	replicon_id	start	end	strand	system_id
```

A window must fully contain the candidate's genomic span. A `+` or `-` window requires that strand. A `.` window admits either strand. Each overlapping system remains an alternative. Several complete independent systems in one genome can be selected separately. A genome-only candidate may compete with a named system and block assignment. Gene adjacency by itself creates no system. A gene-level alias shared by isoforms, repeated alignment domains, RNA copies, conflicting accessions, partial features, pseudogenes and uncertain orientation can all leave groups unresolved. Matching one physical locus in multiple roles precludes selection. Use curated role metadata for compartments or nonadjacent complexes, and supply role-specific exact transcript/protein IDs when independently known. Genomic support records a pairing hypothesis and does not establish a physical interaction.

**Reading the audit.** `metadata/evidence.tsv` records candidate mappings, role/support and exclusion reasons. The GFF3 schema appends `gff3_annotation` with feature ID, typed genomic segments, gene loci, system windows, annotation aliases, limitations and source digest. `groups.tsv` records every candidate group, including incomplete and ambiguous groups. `group_context.tsv` records role confidence and the taxon ID when exactly one distinct supported taxon exists in the group. `evidence_summary.json` counts groups by reason, distinct rows by lookup issue and multiply represented domain intervals. One group may have several reasons. `selected_evidence.tsv` and the synchronized match table contain resolved observations only.

## Public identifier lookups

`triplet run --input input --online` enables identifier-only lookups. A bare unattended run uses online mode unless the project sets `"uniprot": false`. Pass `--offline` to enforce local-only operation. Raw database/profile search uses local executables. The direct UniProt, ENA, RNAcentral and NCBI routes record releases where returned, raw source IDs, retrieval timestamps, and distinguish valid empty responses from errors. Incomplete pages, quota exhaustion and terminal service faults leave an explicit partial-provider status; the unattended run preserves evidence and reports exit 4. [Endpoint schemas and cache identities](evidence.md)

An optional project `providers` object controls online requests. This example uses conservative traffic limits and writes periodic machine-readable progress to standard error.

```json
{"providers": {
  "uniprot_attempts": 4, "nucleotide_attempts": 2, "timeout_seconds": 20,
  "minimum_interval_seconds": 0.5, "max_requests": 10000,
  "max_retry_wait": 15, "jitter_seconds": 0.25, "progress_every": 1000,
  "cache_generation": "2026Q4"
}}
```

The request budget counts network attempts across providers, including retries. The default limit is 100,000. The implementation makes one request at a time, respects provider-specific pacing and stops on an excessive Retry-After. Progress reports row and request counts, elapsed time, and a rough ETA when the complete inventory size is known. A target list that expands to several domains has an unknown final row count.

A completed online lookup batch checkpoints into the shared SQLite cache. To resume after an interrupted run, choose a fresh output directory and reuse the same `--cache-dir` and generation. Each completed run retains its own snapshots under `metadata/`. A new generation creates separate reusable caches and allows refreshed lookups; explicitly configured seed snapshots still contribute their pinned records. The original generation remains available for offline replay. Source manifests include snapshot digests. Raw sequence data, annotations and private project paths never form public provider payloads.
