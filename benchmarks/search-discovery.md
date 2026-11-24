# Local domain discovery and profile alignment

The synthetic JACKHMMER case found four protein domains across three targets. Two domains occupied positions 9–98 and 110–199 of the same source record. Their range identities remained distinct through matching and alignment. A request for the second domain and an insertion-bearing target produced two observation rows and one coordinate seed. Reusing the verified search for a different selection took 3 milliseconds.

## Run configuration

Current raw workflows accept a typed local reference FASTA, database FASTA and one search backend per role. Use raw schema 2 for domain-preserving discovery. Plain and gzip-compressed FASTA are supported. HMMER and Infernal text profiles may also be gzip-compressed. A reference file must contain exactly one ungapped record matching the declared reference ID and molecule. The configured source ID is kept verbatim, including accession versions. Tool-facing FASTA records receive opaque aliases and a separate source-ID ledger.

~~~json
{
  "molecule": "protein",
  "reference_id": "seed.1",
  "reference": "seed.fasta",
  "database": "targets.fasta.gz",
  "database_release": "local-2026-01",
  "backend": "jackhmmer",
  "iterations": 5,
  "evalue": 0.01,
  "domain_evalue": 0.01,
  "include_evalue": 0.001,
  "include_domain_evalue": 0.001,
  "seed": 42,
  "cpu": 2,
  "timeout_seconds": 600,
  "max_candidates": 100000,
  "cache_dir": "search-cache",
  "alignment_backend": "jackhmmer_stockholm"
}
~~~

JACKHMMER uses a protein sequence seed. The reporting options set per-target and conditional domain E-value thresholds. The inclusion thresholds govern domain admission to the next model and final alignment. Every reported final-round domain remains in the audit, including hits absent from the final inclusion Stockholm. The reported independent domain E-value and conditional domain E-value have separate columns. An E-value depends on the search space. The database release, byte digest and actual record count identify that space. A reference searched separately to locate its region has its own search-space size.

The iteration ceiling, positive random seed, worker count and per-process timeout are explicit. Native HMMER writes its profile and Stockholm checkpoint on each completed round. The run also writes the complete text report, individual round reports, final sequence and domain tables, final Stockholm and a convergence flag read from HMMER's report. An iteration ceiling reached without the convergence message stays distinguishable from convergence. Under POSIX an optional positive integer memory_mb enforces a per-child virtual-address-space limit in MiB. Windows execution supports the timeout; requesting a hard child memory limit requires a POSIX runtime. The external programs must already be installed locally. No sequence is submitted to a remote service.

Protein HMMER runs hmmsearch and nucleotide HMMER runs nhmmer. Infernal accepts RNA covariance models. PSI-BLAST accepts a protein seed or compatible binary PSSM. Its complete multi-iteration table is retained alongside the final-iteration view. Existing-schema external workflows that used best-hit-per-source compatibility retain an audit of competing parsed hits. In the current domain-preserving workflow a bare accession alone cannot identify which repeated source interval a curated match intended.

## Existing search tables

To reuse an external table set backend to import, supply hit_table and choose a format below. Imported result coordinates are one-based and inclusive. Every parsed interval is validated against the named local target sequence, reported lengths and strand. Missing database IDs, malformed data rows, filtered hits and duplicate identical domains have distinct audit outcomes. Raw table text is retained. Replaying an imported result verifies its file and source database identities. The biological provenance of a supplied table still depends on its producer's original query, database and command records.

| hit_format | Interpreted source fields |
|---|---|
| hmmsearch_domtblout | HMMER search/jackhmmer domain table. Target and model identifiers, lengths, domain ordinal/count, aligned and envelope bounds, independent/conditional domain and whole-target E-values. Direction is model-to-sequence search. |
| nhmmer_tblout | nhmmer target table. Target and model identifiers, model interval, strand, aligned and envelope target bounds, sequence length, score and E-value. |
| cmsearch_tblout | Infernal format 1. CM/HMM query, model and target intervals, source orientation, bit score and E-value. |
| blast6 | Standard twelve tab-separated BLAST fields in their documented order, optionally followed by qlen and slen. Reverse query and reverse target strands are retained individually. Their combination determines relative orientation. |

An imported multiquery table requires an exact hit_query. The service makes no inference from taxon IDs, versions stripped from accessions or partial string matches. For a BLAST query matching the coordinate reference, query coordinates can anchor the target fragment. If the query differs, supply reference_hits from the same model/query against that exact reference. HMM/CM model positions are independent of biological reference positions. A profile search of the seed, supplied reference_hits, or a validated profile alignment supplies that correspondence. Missing model lengths remain unknown unless a declared hit_model_length or native profile provides them. An active model-coverage threshold excludes a hit of unknown model coverage. A model named only by an imported accession does not establish experimental identity.

## Filtering and alignment

- Each source interval gets a digest binding the exact source ID, reported interval, strand, query and model interval. Repeated domains use source/start-end row IDs with reverse spans written in descending order. A unique hit can keep its bare source ID for existing evidence or use an explicitly requested range. The domain digest remains stable across those display forms. Identical repeated reports are recorded separately and deduplicated. A source ID resembling a generated range remains distinguishable from that generated row.
- min_model_coverage is the reported model span divided by known model length. min_target_coverage is the aligned source interval divided by full target length. A short domain in a long multidomain sequence can correctly have low target coverage. min_entropy_bits uses the unsmoothed canonical symbol entropy of the actual source fragment. This flags low complexity on the declared threshold. Species and domain architecture still require separate evidence.
- min_reference_coverage and min_domain_coverage apply after placing selected rows. They count aligned positions where both the seed and domain have canonical symbols, divided by the canonical seed and domain lengths respectively. A passing prefilter on an unrequested row leaves alignment-dependent QC explicitly unassessed. Unresolved reference loci and inadequate mapped support remain in the audit. Search scores alone establish neither orthology nor membership of a particular genome or complex.
- JACKHMMER reads its final Stockholm with the original query as coordinate seed. Requested included rows preserve insertions and terminal gaps. The profile alignment alternatives use hmmalign or cmalign on exactly the selected oriented domain fragments and the independent coordinate seed. Tool outputs must reproduce each supplied fragment and seed on removing gaps. For nucleotide profile alignments a positive source orientation is never assumed. The profile must locate the seed unambiguously. Competing reference placements require a five-bit lead for automatic assignment, and the measured margin is reported. A reversed seed is transformed back to the user's input orientation. RNA secondary structure comes from the supplied CM when cmalign is used.
- Profile-mode output retains all aligned columns, records how many lie in reference gaps, and reports insertions and deletions by domain. The portable projection backend instead places fragments onto original reference positions, records discarded insertions, and preserves original numbering. Its heuristic recursively searches residual nonoverlapping source intervals. It cannot establish a complete inventory of overlapping domains. Users needing a model-based inventory should use a declared profile or an audited external hit table.

## Cache, recovery and outputs

Each role has a discovery.json record and hit_audit.tsv. The raw directory contains the native output and a digest inventory. The profile directory, when used, binds stable selected IDs to tool rows. The projected and aligned data can enter the existing native coordinate mapper. Unrelated search hits remain in the evidence without becoming a full in-memory component alignment.

A cache key binds the raw search recipe, exact database/query/model/table bytes, seed sequence, release metadata, execution settings, tool versions and executable SHA-256 digests. Raw data are published by an atomic directory rename after successful completion and input rechecking. A per-key OS lock excludes concurrent writers. A cache read verifies the manifest and every artifact's name, size and digest before copying raw evidence into a new run. Changing selected rows or post-search filtering reuses the verified raw result. Changing database content, executable or raw search settings changes its key. A tool timeout, interruption or exception before publication leaves no completed cache result; retry starts the native search from its input seed. The per-round files are inspection checkpoints. The wrapper does not claim numerical continuation of an interrupted JACKHMMER process at an arbitrary round.

max_candidates bounds reported-hit parsing and the in-memory filtered hit ledger. Raw databases and reports are streamed to and from local disk. The normalized search database, BLAST indexes and command outputs consume scratch space proportional to the inputs and native output. The memory_mb setting governs external processes; Python-side matching/QC and FASTA identifier uniqueness tracking also require memory. A disk or quota failure requires adequate storage before retrying.

## Real executable check

The verification uses only fixed-seed generated sequences and models, four initial protein database records and three RNA records. No biological benchmark or remote homolog database is required. The first protein search ran with a 1,024 MiB child-process bound. The RNA CM was calibrated independently on synthetic sequence before the timed search. Timings are from one local WSL Linux run and include workflow and file overhead, excluding model construction and calibration.

| Run | Versions | Retained domains | Aligned observation rows | Aligned width | Elapsed |
|---|---|---:|---:|---:|---:|
| JACKHMMER, two rounds to convergence | HMMER 3.4 | 4 | 2 | 92 | 70.8 ms |
| Same raw search, different selected row | HMMER 3.4 | 4 | 1 | 92 | 3.0 ms |
| Changed seed, cache miss | HMMER 3.4 | 4 | 1 | 92 | 68.4 ms |
| Changed source database, cache miss | HMMER 3.4 | 4 | 1 | 92 | 68.0 ms |
| hmmsearch plus hmmalign | HMMER 3.4 | 4 | 1 | 92 | 13.6 ms |
| PSI-BLAST, two-round ceiling | BLAST+ 2.16.0+ | 4 | 1 | 90 | 178.1 ms |
| cmsearch plus cmalign, reverse source domain | Infernal 1.1.5 | 4 | 1 | 76 | 201.4 ms |

The 90-residue protein seed acquired two reference-gap insertion columns. JACKHMMER and hmmalign both recovered the complete insertion-bearing target row. The tandem source produced intervals 9–98 and 110–199. The 76-residue RNA recovered the original seed on orienting the reverse-strand source interval. A separate no-hit JACKHMMER probe converged in one round and wrote a seed-only Stockholm. These tests verify coordinate behavior, subprocess execution, evidence and replay on small known inputs. They do not estimate real-database recall, false discovery rate or orthology accuracy.

The RNA stem also produced weaker full-span matches on the opposite strand. Under the chosen reporting threshold the two true source records therefore contributed four retained strand-specific intervals. In the independent search of the coordinate seed the strongest match scored 113.5 bits and the opposite orientation scored 29.4 bits. The 84.1-bit lead selected the seed orientation. Requesting the exact reverse source range selected one corresponding observation for alignment. This example shows why a reported interval or a bare accession can leave biological ambiguity even when the tool finishes successfully.

~~~bash
python benchmarks/search_discovery.py
python -m pytest tests/test_search_discovery.py tests/test_profile_search.py
~~~
