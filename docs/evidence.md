# Reading and supplying evidence for a three-family project

Start with the [three-family folder instructions](families.md). After preparation, open `metadata/groups.tsv` for one decision per genome or curated neighborhood. Look up its `group_id` in `metadata/evidence.tsv` to inspect the rows, molecules, alternative placements and sources behind that decision. An empty group ID identifies a record whose genome could not be placed.

## Supply a curated mapping

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

Live resolution is opt-in for declared protein components. Lookup accepts UniProtKB accessions, entry names, `sp|accession|entry` and `tr|accession|entry`. Domain ranges and Stockholm sequence versions are separated from the query and the original row ID is preserved. Isoform accessions retain their isoform number. Pfam and Rfam family accessions identify families and need separate molecule/locus evidence.

The resolver requests active UniProtKB metadata in bounded batches, follows result pages and separately resolves secondary accessions to their possible current entries. The Proteomes service supplies assembly accessions where available. The audit retains each reported proteome and distinct replicon value with its original cross-reference properties. Two different replicon values for a role-compatible molecule yield `multiple_replicon_annotations` until curation resolves the placement. Identical cross-references count once. A chromosome name supplies placement at replicon scale. A shared neighborhood requires independent, finer-scale evidence. OrthoDB group cross-references in the form `OrthoDB:<group>at<level>` can support the corresponding role. Other returned links, including eggNOG and OMA records, remain available in the audit. Requests contain identifiers and retrieve annotation fields. [UniProtKB API](https://www.uniprot.org/api-documentation/uniprotkb) · [Proteomes API](https://www.uniprot.org/api-documentation/proteomes) · [UniProt proteome definition](https://www.uniprot.org/help/proteome)

The SQLite cache retains successful and verified empty lookups with retrieval times and release headers when supplied. `uniprot_not_found` means a completed lookup found no applicable entry. `cache_miss` means the offline snapshot had no record for that ID. Transport or HTTP failures stop preparation after bounded retries on transient errors. Replaying a closed snapshot with the same input and configuration preserves source annotations, including earlier absences. Choose a fresh cache when a new release is needed.

Unsuccessful alias attempts stay visible as individual audit records with an empty group ID. A confirmed mapping through a row's Stockholm accession can establish its molecule and reference orthogroup even when another name on the same row returns no record. Positive hits to different proteins remain competing mappings. When reading older SQLite snapshots the resolver reconstructs replicon alternatives from their preserved raw cross-references. Offline replay leaves the source snapshot untouched.

The evidence audit lists each declared UniProt accession and its origin in `declared_accessions`. An accession in the aligned row name (including a pipe header) or its Stockholm AC annotation must link to the proposed molecule. A biological reference accession imposes the same requirement when it supplies a role. A display name or target selector can help locate records. Inspect `accession_links` for primary-ID agreement, a verified secondary-to-current lookup, or a sourced curated link keyed by the accession or exact domain row. A curated bare mnemonic that points to a different molecule needs separate corroboration. Isoform numbers remain significant.

For example, if the Stockholm row declares accession `P12345` and a mnemonic returns `P23456`, a missing cache entry or verified empty response for `P12345` leaves the proposed mapping unresolved. Its original annotation stays visible with `unlinked_accession` and the missing link in `unlinked_accessions`. An affected automatic anchor reports `unlinked_reference_accession`. These records contribute no orthology to other rows sharing the same interval.
