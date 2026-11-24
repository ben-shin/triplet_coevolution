# Planning and resuming local jobs

Use a new workspace for a project. Each invocation writes a journal and returns the path to its JSON receipt. A successful configured run also writes an index linking the matched inputs, native plan, scores, candidate coordinates, calibration, benchmark and report that actually ran.

~~~sh
triplet doctor --scratch results --workers 2 --blas-threads 1
triplet validate-inputs --config examples/workflow/run.json
triplet pipeline --config examples/workflow/run.json --work-dir results/jobs --dry-run
triplet pipeline --config examples/workflow/run.json --work-dir results/jobs --until prepare
triplet pipeline --config examples/workflow/run.json --work-dir results/jobs --until plan
triplet pipeline --config examples/workflow/run.json --work-dir results/jobs --json-log
~~~

An aligned preparation JSON works with the same commands. The usual entry points accept the workspace too: `triplet prepare --config examples/workflow/prepare.json --work-dir results/aligned` stops after matching, and `triplet run --config examples/workflow/run.json --work-dir results/jobs` continues to reporting. A raw search configuration supports `triplet prepare --raw --config examples/workflow/run.json --out results/hits` to publish search and matching independently. Existing `--out` commands continue to require a fresh destination.

For a three-folder project, inspect the sources without any network requests, then run its automatic matching and bounded scoring policy. The second command can be repeated after interruption.

~~~sh
triplet validate-inputs --input input
triplet run --input input --offline --work-dir results/family-jobs
~~~

The folder workflow keeps separate evidence/preparation and scoring/report receipts. Native scoring and report generation bind the installed explorer stylesheet and JavaScript bytes to their cache identities. Changing either reuses eligible input preparation, then republishes the affected HTML under a new receipt. The original report is preserved. Both cache paths verify the asset digests again before publishing. Complete offline input inventories and closed lookup caches can be reused. Network-enabled calls resolve a fresh evidence stage each invocation and can use the normal provider cache for previous queries. The terminal scientific status and exit code retain their usual meaning, including partial, no matches and unavailable providers. Its report includes the specific matching or scoring blockers. A work directory lives outside the input project.

## Before allocating a model

`doctor` prints JSON with Python and package versions, detected BLAS, optional packages, local executable paths/banners, CPU allocation, currently available memory and free disk space. Require a particular tool with `--require-tool hmmsearch`. Requested worker counts, BLAS threads and resource reservations are checked against the current host. Use `--out <directory>` to keep doctor.json. Resource readings are snapshots and can change as other jobs start.

`validate-inputs` reports file byte sizes, SHA-256, formats, reference rows and the matched and excluded group counts for aligned preparation. The raw-database form validates sequences, reference, configured search tools and available evidence, and leaves post-search match counts pending. The folder form reports source selection, row/column counts and molecule inference before evidence lookup. A prepared directory containing analyze_args.json supports `triplet validate-inputs --prepared DIR --out results/preflight`. Its native plan includes row/site QC, coordinate masks, retained dimensions, eligible cells, operation and numeric working memory estimates, approximate disk payload and the declared permutation resolution. Run `triplet plan --help` for direct native inputs.

Configured stages start on fresh child processes with explicit BLAS/OpenMP limits. The default is one stage worker and one BLAS thread. `--workers` bounds configured search CPUs/PLM targets and controls standalone calibration workers. Set `--blas-threads` explicitly when allocating more CPUs. `--memory-mib` bounds estimated native working arrays, and `--disk-mib` reserves free disk before a new stage. The scorer also checks estimated text/matrix output space. Interpreter, external-tool, library and filesystem overhead require additional capacity. Executed stages record wall time, child CPU time, process peak resident bytes including the interpreter and total bytes in published outputs. External child-program RAM is outside that measurement. Cached stages record verification time and bytes.

## Stage order and recovery

| Stage | Result |
|---|---|
| prepare | Synchronized inputs, matched groups, coordinate and matching QC; raw mode also saves discovery audits. |
| plan | Actual retained axes, support and memory/budget diagnostics. A blocked plan stays inspectable. |
| score | Complete native score archive, triple fits when requested, ranked views and run summary. |
| select | Checksummed original fitted candidate selection, or seeded coordinate selection from a pair-only archive. Use `triplet select --run RUN --out SCREEN --mode triangle` for an explicitly new screen. |
| calibrate | Named null refits, replicate evidence, p-values and correction families when a null policy is supplied. |
| benchmark | Independent labels and coverage metrics when a labels file is supplied. |
| report | An offline native report regenerated from the archived analysis. The invocation index also links independent benchmark/calibration stages. |

`--until` selects the final stage. Pass `--null-config null.json` or `--labels contacts.tsv` to add independent calibration and benchmarking. Inline historical `--legacy` contacts stay available in the native scoring output. Changing labels on the normal path reruns benchmarking while reusing the score archive and native report. Changing a ridge or site policy reruns planning and scoring while retaining prepared data. A matching-input, executable or package-source change invalidates its dependent stages.

Typed configuration fields identify each input independently of its filename extension. A tree calibration binds all declared Newick files, tip maps and supplied frequency files for the primary and sensitivity scenarios. Stability and conditional fits bind the clade policy, split manifest, tree sources and site panel they actually read. Changing a branch length creates a new calibration and checkpoint. Source descriptions and observation names remain metadata.

Each completed stage is published by renaming a private attempt into `objects/<stage>/<key>/` on the same filesystem. Its receipt binds the input settings and hashes to every output file, size and SHA-256. Reuse verifies the complete artifact inventory. Source and applicable executable identities are checked across execution. An OS writer lease prevents overlapping workspace publishers. Run logs stay under `runs/<id>/` and include events.jsonl, individual child stdout/stderr and resource receipts. A killed or disk-full attempt has no reusable stage receipt. After restoring space or correcting a failed input, rerun the command against the same workspace. Leave failed attempts available for diagnosis. If an already published artifact is corrupt, retain it and select a fresh workspace. Never edit a verified object to change analysis settings. The workspace is intended for cooperating local processes on a filesystem with reliable locks and same-volume directory rename.

Use `--json-log` on the pipeline for timestamped JSON lines on stderr, `--verbose` for stage events, or `--quiet` to suppress routine output. Command-wide forms before the subcommand, such as `triplet --json-log analyze ...`, also record native publication start and completion. Artifact verification reports measured byte throughput and remaining time. A long fit emits elapsed-time heartbeats; the first fit has no measured throughput history. Named null progress uses completed new draws for rate and remaining time. The pipeline keeps checkpoint journals in a design-hash subdirectory under `checkpoints/`, or under the checkpoint base supplied by the null policy. It resumes durable replicate IDs for the same scientific inputs. Logs require no TTY. Provider credentials are never requested by the runner. Worker stderr and receipts can contain user-provided file names and biological identifiers, so manage access to the workspace accordingly.

For independently scheduled workers, freeze a numerical shard plan, run disjoint replicate IDs and verify a complete merge before using adjusted significance. The Slurm wrapper is optional. [Distributed command examples and recovery](distributed-nulls.md)

The supplied 888-observation GatABC dataset also has an on-machine comparison of estimated workspace, measured process RSS, output bytes and verified reuse. [Resource measurements](../benchmarks/workflow-resources.md)
