# Reading and supplying evidence for a three-family project

Start with the [three-family folder instructions](families.md). After preparation, open `metadata/groups.tsv` for one decision per genome or curated neighborhood. Look up its `group_id` in `metadata/evidence.tsv` to inspect the rows, molecules, alternative placements and sources behind that decision. An empty group ID identifies a record whose genome could not be placed.

## Automatic source coverage

An unattended run first checks curated evidence, then reads annotations on the specific Stockholm row. It queries services using supported public identifiers when local evidence is insufficient. The lookup sends identifiers and requests metadata. The alignment and its sequence strings stay local.

| Source | Facts used |
| --- | --- |
| Stockholm row (`#=GS`) | `ASSEMBLY` accepts a versioned GCA or GCF accession. `PROTEOME` accepts a UniProt proteome accession. `LOCUS`, `REPLICON`, `NEIGHBORHOOD` and `ORTHOGROUP` supply a curated row's molecule, chromosome/plasmid, locus cluster and role. Supported `DR` cross-references include Assembly, Proteomes, taxon and scope-qualified numeric OrthoDB groups. `OX` and `OS`, including these fields in `DE`, retain organism context. Multiple conflicting assignments remain visible. |
| ENA Portal | Recognized INSDC genomic accession with a Stockholm interval, including GenBank-style pipe headers. The sequence metadata provides the returned accession version, base count, taxonomy and available assembly IDs. A row with an explicit version must match it. A versionless record takes the version confirmed by the service. |
| NCBI | Genomic RefSeq accessions beginning with NC_, NG_, NT_, NW_ or NZ_ use nuccore ESummary for accession version, sequence length and taxonomy. NCBI Datasets provides version-specific assembly membership for these accessions and for ENA accessions lacking a versioned assembly link. All returned candidates remain in the audit. |
| RNAcentral | A URS identifier with species suffix, or a URS plus an unambiguous row taxon ID, can be queried for genome locations. Each returned chromosome, interval, strand and explicit versioned assembly contributes a candidate. A URS interval in the alignment indexes the RNA. Multiple genomic copies remain distinct alternatives. A missing assembly leaves a location unplaced. |

The Stockholm examples `#=GS record-a ASSEMBLY GCA_987654321.1` and `#=GS record-a LOCUS locus-a` illustrate two row assertions using constructed identifiers. A versioned genomic accession with an oriented Stockholm interval can also identify a locus on a declared assembly. For an archive-based nucleotide placement the endpoints must lie within the confirmed record length. Descending genomic intervals retain the minus strand. RNAcentral genome endpoints retain the provider's values and strand. A bare family name, an organism label or a transcript such as NM_/NR_ without a separate locus mapping supplies insufficient genomic placement.

The audit records `molecule_type`, `nucleotide_accession`, `genomic_start`, `genomic_end`, `genomic_strand`, `placement_source`, raw `stockholm_annotations` and `cross_references`. A verified GCF/GCA pair may join the same group only when the NCBI report explicitly confirms the versioned pair and reports identical sequence content. The original assembly stays in `assembly_id` and the normalized pair appears in `assembly_aliases`. Unverified, changed or multiply placed records retain their separate identities. The [automatic inference guide](automatic-inference.md) describes when family membership supports a role and how the audit distinguishes this from orthology.

Nucleotide metadata requests are deduplicated by public accession. ENA and ESummary support batches. Species-specific RNAcentral locations are paginated and paced. The separate nucleotide and protein SQLite snapshots carry successful results and verified absences. An offline snapshot without the requested accession yields `nucleotide_cache_miss` or `cache_miss`. A completed negative query yields `nucleotide_not_found` or `uniprot_not_found`. Service timeouts, refusal, rate limits beyond the retry window or incomplete responses leave `provider_unavailable` evidence. The run stops further requests to the affected service, preserves completed results, and records the failure under `providers` in `evidence_summary.json`. A later connected run can retry an unavailable lookup.

The ENA field schemas list the five fields requested for sequence annotations. The NCBI OpenAPI definition specifies the sequence-to-assembly and dataset-report routes. RNAcentral documents its API and genome mappings. The test suite exercises supported response formats with constructed records, including reverse intervals, alternate locations, service failures and cache-only replay. [ENA sequence return fields](https://www.ebi.ac.uk/ena/portal/api/returnFields?result=sequence&format=json) · [ENA sequence search fields](https://www.ebi.ac.uk/ena/portal/api/searchFields?result=sequence&format=json) · [NCBI ESummary](https://www.ncbi.nlm.nih.gov/books/NBK25499/) · [NCBI Datasets OpenAPI](https://github.com/ncbi/datasets/blob/master/client/openapi/openapi3_v2.yaml) · [RNAcentral API](https://docs.rnacentral.org/api.html) · [RNAcentral genome mappings](https://docs.rnacentral.org/genome-browser.html)

RNAcentral's public location schema also distinguishes an assembly display name from its optional GCA accession. A display name alone leaves the genomic mapping unresolved. [RNAcentral location schema](https://github.com/RNAcentral/rnacentral-data-schema/blob/master/sections/location.json)

Each RNAcentral page must provide its next-page link or an explicit terminal `null`. The resolver checks consecutive pages for the same species-specific record and rejects repeated locations. When the service supplies a total, it must remain constant and match the number of distinct locations. An incomplete response or interrupted read after bounded retries is recorded as a provider failure and can be retried. Previously completed, independent lookups remain available. Older RNAcentral cache entries require a connected refresh because their pagination was unverified. Other validated source entries in the same snapshot remain usable. [RNAcentral pagination](https://docs.rnacentral.org/api.html#pagination)

## Supply a curated mapping

An existing mapping file can supply facts for local identifiers, unsupported databases or unresolved alternatives.

Put a tab-separated `mappings.tsv` in the input project or configure a manifest path. The six required header names are `component`, `identifier`, `protein_id`, `genome_id`, `orthogroup` and `source`. Each line describes one documented mapping. This small example uses constructed IDs, a noncoding RNA locus and two possible A-family proteins.

```tsv
component	identifier	protein_id	genome_id	orthogroup	source
A	A1	protein-a1	demo-genome-1	demo:ATPase	demo collection v1
A	A2	protein-a2	demo-genome-1	demo:other-ATPase	demo collection v1
B	B1	protein-b1	demo-genome-1	demo:Partner	demo collection v1
C	R1	rna-locus-1	demo-genome-1	demo:Guide	demo collection v1
```

| Field | Meaning |
| --- | --- |
| `component` | A, B or C from the folder assignment in the project. The audit also records the folder's display name. |
| `identifier` | An exact aligned row ID, full domain ID, row accession, entry name or requested alias. An exact domain mapping takes precedence. Otherwise all matching aliases contribute evidence. |
| `protein_id` | A stable identifier for the underlying molecule. For DNA or RNA use the corresponding curated gene, transcript or locus identifier. Use identifiers that distinguish individual loci and remain consistent across aliases. |
| `genome_id` | The specific genome, assembly or proteome supporting co-occurrence. With this cell blank, the resolver uses `assembly_id` when present, followed by `proteome_id`. Assigning the same local ID across rows asserts the same biological genome. |
| `orthogroup` | A database-qualified group or a curated role assignment with recorded evidence. Matching is exact and case-sensitive. Prefix curated assignments with their collection name, such as `demo:Guide`. |
| `source` | Citation, record, versioned collection or other provenance for this line. Required for every mapping. |
| `taxon_id` | Optional taxonomic identifier. Many strains and isolates can share it. It supplies context and consistency checking. |
| `assembly_id`, `proteome_id` | Optional assembly accession with version and proteome accession. Keep both when available. |
| `replicon_id` | Optional placement on a chromosome, plasmid or other replicon within the genome. For UniProt records it is the reported Proteomes `Component` value. |
| `neighborhood_id` | Optional curated locus cluster or system within the genome. Use a shared ID only with evidence that places the components in the same neighborhood. |

Repeat lines with distinct mappings, placements, group memberships or sources to retain alternatives. Leave the molecule, genome or orthogroup cells blank when evidence is missing. Curated records take precedence for genomic features and role assignments. An available snapshot or enabled protein lookup can separately verify the accession behind a curated alias. Keep a versioned manifest for reproducibility. The summary includes its SHA-256.

## Establish each component's expected role

In the example, `expected_orthogroups` in the project JSON would contain `{"A":["demo:ATPase"],"B":["demo:Partner"],"C":["demo:Guide"]}`. On the same genome, A1 supports the A role and A2's different group excludes it. If A2 had another uncharacterized or compatible assignment, the role would remain unresolved.

An explicit component reference can also supply the orthogroup. The resolver looks up the reference, requires one unambiguous underlying molecule and selects its sole group at the requested scope. With `"orthology_scope": "2759"`, eligible automatic anchors have OrthoDB IDs ending in `at2759`. It accepts a unique group across reported memberships when the scope is omitted. Ambiguous or missing anchors appear under `roles` in `evidence_summary.json`. A configured list takes precedence for that role. Scope is significant because an orthogroup is defined at a taxonomic level. [OrthoDB group and level definitions](https://purl.orthodb.org/widoco/index-en.html)

For each genome and optional neighborhood, selection requires an aligned row with a unique placement and a role-supported molecule for each component. A competing row with missing orthology can still be a paralog, so it blocks selection. A group at a different OrthoDB level provides insufficient evidence to exclude a candidate. Shared assignments that cannot distinguish two roles, contradictory source assignments, competing copies of a domain, shared molecules across roles and conflicting taxa or assemblies also prevent selection. Where named neighborhoods and viable candidates lacking neighborhood assignments coexist, the affected genome stays ambiguous. A genome with no locus annotation can still resolve through unique genome-wide role assignments.

Within a role, records that identify the same molecule, genome, neighborhood and known oriented domain interval contribute to a common set of orthology claims. If one source supports the requested group and another assigns the same interval to a different group at the same level, both rows show `orthology_conflict`. Different intervals and records with unknown spans are assessed separately. Multiple concordant annotations for one alignment row count as one candidate. Distinct rows that could both fill the role retain their own IDs because their alignment columns can differ even when their biological coordinates agree. An explicit target containing a full row ID can narrow that row choice.

These decisions apply to the supplied inventory and sources. A target list narrows the pool, and homologues missing from the chosen release may remain undetected. Genome co-occurrence with role support identifies a candidate observation for coevolution analysis. Physical binding requires independent evidence.

A named neighborhood also requires consistent replicon evidence from its placed, role-supported members. For example, `chr1` and `chr2` yield `conflicting_neighborhood_replicons`. Distinct naming systems with no established equivalence yield `unverified_neighborhood_replicons`. The comparison recognizes `chr`/ `chromosome` and Roman chromosome numbers I through IX, so `Chr I` agrees with `Chromosome 1`. Values such as `unplaced scaffold` provide no specific placement. Genome-wide grouping can include different chromosomes when the project makes no neighborhood claim.

## Follow individual mapping decisions

| Output | Contents |
| --- | --- |
| `evidence.tsv` | One row per input identifier and distinct mapping/source alternative. Includes original sequence ID, folder, alignment presence, domain coordinates, aliases requested, lookup ID, molecule, genome, taxon, assembly, proteome, orthogroups, evidence source, releases and confidence. JSON cells retain multiple orthogroups and the original UniProt cross-reference properties. |
| `groups.tsv` | `resolved`, `ambiguous` or `incomplete`, reason codes, and counts of distinct plausible row IDs for A, B and C. A demonstrably different role is excluded from the count. Membership lists stay factored by component, even with thousands of tied combinations. |
| `selected_evidence.tsv` | Exactly three legacy-compatible membership rows per accepted group. The selected IDs are the original domain-specific alignment IDs used by the preparation step. With zero matches the file contains its header. |
| `evidence_summary.json` | Input, audit and selected row counts, role anchors and decisions, resolver mode, source hashes, observed releases and request/cache counts. |

`group_reasons` identifies affected roles, for example `A:multiple_candidates` or `C:absent_from_alignment`. Row-level `role_status` distinguishes `supported`, `orthogroup_mismatch`, `multiple_mappings`, `missing_orthology`, `no_role_anchor`, `missing_genome` and other record-specific outcomes. `confidence` describes the basis for the decision. Its supported categories are `genome_and_role_supported` and `neighborhood_and_role_supported`. Other rows carry `role_excluded` or `uncertain`. These categories have no numerical probability interpretation.

## UniProt metadata and offline replay

The project-driven evidence command supports opt-in live lookup for declared protein components. The unattended run enables lookup by default, and `--offline` selects local annotations and snapshots. Lookup accepts UniProtKB accessions, entry names, `sp|accession|entry` and `tr|accession|entry`. Domain ranges and Stockholm sequence versions are separated from the query and the original row ID is preserved. Isoform accessions retain their isoform number. Pfam and Rfam family accessions identify families and need separate molecule/locus evidence.

The resolver requests active UniProtKB metadata in bounded batches, follows result pages and separately resolves secondary accessions to their possible current entries. The Proteomes service supplies assembly accessions where available. The audit retains each reported proteome and distinct replicon value with its original cross-reference properties. Two different replicon values for a role-compatible molecule yield `multiple_replicon_annotations` until curation resolves the placement. Identical cross-references count once. A chromosome name supplies placement at replicon scale. A shared neighborhood requires independent, finer-scale evidence. OrthoDB group cross-references in the form `OrthoDB:<group>at<level>` can support the corresponding role. Other returned links, including eggNOG and OMA records, remain available in the audit. Requests contain identifiers and retrieve annotation fields. [UniProtKB API](https://www.uniprot.org/api-documentation/uniprotkb) · [Proteomes API](https://www.uniprot.org/api-documentation/proteomes) · [UniProt proteome definition](https://www.uniprot.org/help/proteome)

The SQLite cache retains successful and verified empty lookups with retrieval times and release headers when supplied. `uniprot_not_found` means a completed lookup found no applicable entry. `cache_miss` means the offline snapshot had no record for that ID. Transport or HTTP failures in the project-driven evidence command stop preparation after bounded retries. The unattended run preserves completed evidence and reports lookup failures as described above. Replaying a closed snapshot with the same input and configuration preserves source annotations, including earlier absences. Choose a fresh cache when a new release is needed.

Unsuccessful alias attempts stay visible as individual audit records with an empty group ID. A confirmed mapping through a row's Stockholm accession can establish its molecule and reference orthogroup even when another name on the same row returns no record. Positive hits to different proteins remain competing mappings. When reading older SQLite snapshots the resolver reconstructs replicon alternatives from their preserved raw cross-references. Offline replay leaves the source snapshot untouched.

The evidence audit lists each declared UniProt accession and its origin in `declared_accessions`. An accession in the aligned row name (including a pipe header) or its Stockholm AC annotation must link to the proposed molecule. A biological reference accession imposes the same requirement when it supplies a role. A display name or target selector can help locate records. Inspect `accession_links` for primary-ID agreement, a verified secondary-to-current lookup, or a sourced curated link keyed by the accession or exact domain row. A curated bare mnemonic that points to a different molecule needs separate corroboration. Isoform numbers remain significant.

For example, if the Stockholm row declares accession `P12345` and a mnemonic returns `P23456`, a missing cache entry or verified empty response for `P12345` leaves the proposed mapping unresolved. Its original annotation stays visible with `unlinked_accession` and the missing link in `unlinked_accessions`. An affected automatic anchor reports `unlinked_reference_accession`. These records contribute no orthology to other rows sharing the same interval.

Local GFF3 features, named systems, source hashes, retry limits and cache generations are documented in [Genomic evidence and lookup controls](genomic-evidence.md).
