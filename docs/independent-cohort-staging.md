# Public polymerase and ribosome cohorts

Both systems reached the registered minimum of 128 complete RefSeq assemblies and four GTDB phyla in the third 192-assembly batch. The freeze covers sequence data, source provenance and reference correspondence. Experimental scoring is reserved for the independent validation stage.

| Cohort and component order | Assemblies admitted / screened | GTDB phyla | Reference lengths | Sites passing the sequence mask |
| --- | ---: | ---: | --- | --- |
| RNA polymerase, RpoA / RpoB / RpoC | 461 / 576 | 40 | 329 / 1,342 / 1,407 | 304 / 966 / 1,050 |
| Ribosome, 5S rRNA / uL5 / uL18 | 163 / 576 | 29 | 120 / 179 / 117 | 100 / 167 / 108 |

Of 189,801 representatives in the GTDB R232 bacterial marker tree, 150,716 had GenBank-only tips, 30,352 failed the complete-genome requirement, and 353 lacked an exact current RefSeq entry. The frame contained 8,380 eligible representatives from 45 phyla. The protocol ordered genomes within each phylum by seeded accession hash, then sampled successive members of each nonempty phylum in turn. The capped registry contains the first 1,536 genomes. The screen stopped at 576. Both cohorts draw on this frame and share 139 assemblies. The results describe a phylum-stratified sample of complete genomes. [GTDB R232 release](https://data.gtdb.ecogenomic.org/releases/release232/232.0/)

Each accepted genome has one intact, annotated locus for each component on an exact versioned assembly. The screen counts separate loci with identical sequences separately. Proteins match versioned CDS products, while the RNA matches a versioned genomic record, locus tag, interval and strand. Across the mixed cohort screen, 408 genomes had multiple annotated 5S loci and five failed the 5S sequence gate. Exclusion reasons can overlap across components. Annotation completeness and exact source availability limit ascertainment. [NCBI annotation specification](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/file-formats/annotation-files/about-ncbi-gff3/)

## Coordinate references

The polymerase axes are the registered E. coli K-12 canonical sequences P0A7Z4, P0A8V2 and P0A8T7. The deposited 4YG2 sigma-70 holoenzyme includes RpoA, RpoB and RpoC from E. coli O157:H7, and RpoD from K-12. All three tested polymerase entities match the corresponding K-12 sequences at every reference position. Biological assembly 1 contains alpha copies A and B, beta chain C and beta-prime chain D (label asym IDs). Both alpha copies enter the registered copy rule. [4YG2 deposition](https://www.rcsb.org/structure/4YG2)

The 4YBB assembly 1 label asym IDs for 5S, uL5 and uL18 are WB, AC and KC, respectively. The 5S coordinate axis is the 120-nucleotide deposited entity 4YBB_23. The protein axes are K-12 P62399 and P0C018. The deposited protein sequences match all aligned residues. uL5 covers 177 of 179 canonical positions and uL18 covers all 117. The local correspondence file records entity and author labels, coordinate IDs, sequence identity and coverage. Coordinate observability is assessed separately during structural evaluation. [4YBB deposition](https://www.rcsb.org/structure/4YBB)

## Freeze and replay

The prospective protocol was registered before per-assembly acquisition under SHA-256 `8ba73f61cf525a8b5a8880837198da01edee2d8349d30ab9cdfe943d27ada577`. The canonical LF freeze SHA-256 is `df4b9a9ebc547351390e8e5695df3f80326197dddc85894ba9175a7d311958bc`. The freeze pins the frame, screening decisions, reference correspondence, each cohort output and 1,256 per-assembly and reference resources totaling 547,266,092 bytes. The three shared GTDB/RefSeq frame indices are pinned separately. The compact compressed registry, audit and source manifests contain public metadata. Bulk GFF, FASTA and aligned observations live under the ignored `results/public-experimental/independent-cohorts/` directory. [Registration](../benchmarks/independent_cohort_protocol.json) · [Freeze](../benchmarks/independent_cohort_freeze.json) · [Frame indices](../benchmarks/experimental_sources.json)

With the project installed in the active Python environment, reproduce acquisition and run the independent witness check from the repository root.

```sh
python benchmarks/stage_independent_cohorts.py --frozen-frame
python benchmarks/verify_cohort_staging.py
```

Omit `--frozen-frame` to rebuild the candidate registry from the three exact historical indices. NCBI updates its current assembly report and genome annotations. A changed source or output fails digest comparison. Preserve the original ignored cache when exact historical upstream bytes are needed. The freeze used Python 3.13.15 and Biopython 1.88. [NCBI public data policies](https://www.ncbi.nlm.nih.gov/home/about/policies/) · [UniProt terms](https://www.uniprot.org/help/license) · [PDB data usage](https://www.rcsb.org/pages/policies)

Each cohort directory contains the full reference-width FASTAs, synchronized matches, source locus witnesses, one-based sequence-only site masks and predeclared weights. The first FASTA row supplies the reference axis. Only matched assembly rows count as observations. The masks require at least 80% canonical coverage and two canonical states. Uniform assembly weights and equal-total-mass-per-phylum weights are recorded for prospective evaluation. A reference-centred pairwise alignment can miss remote homology and discards insertion columns. The per-locus witness retains query coverage, identity, insertion counts and gap counts.

A second parser checked every admitted protein against the original versioned RefSeq FASTA and GFF interval. It recounted distinct annotated 5S intervals across replicons in all admitted mixed assemblies and checked their strand-specific sequence records. It also reproduced the six alignment row orders, aligned hashes, site masks and weights. These checks covered 1,872 loci across 485 distinct assemblies. The frozen-frame replay produced the same aggregate and file digests.
