# Distributed null calibration

Use a completed native analysis with its inputs and a named null policy. A shard plan fixes the final draw count, statistic definitions and hypothesis families before simulation. Each shard evaluates a disjoint set of zero-based replicate IDs. The merge verifies complete coverage and publishes the same coordinate table, finite-draw counts, confidence bounds, multiple-testing corrections and optional maxT results as `triplet calibrate`.

The example policy uses MI and DCA. From the checkout, create its example source in a new directory.

~~~bash
triplet analyze --a examples/mixed/protein.fasta --b examples/mixed/dna.fasta \
  --c examples/mixed/rna.fasta --b-type dna --c-type rna --matches examples/mixed/matches.tsv \
  --a-ref prot_ref --b-ref dna_ref --c-ref rna_ref --dca --out results/mixed
~~~

## Run on a workstation

The launcher prepares the plan, runs bounded local subprocesses, writes one log per partition and can publish the final calibration. The example allows three concurrent partitions, one worker per partition and 2,048 MiB in total. For a 512 MiB null policy this includes the default 128 MiB reserve per interpreter and coordinator. A smaller plan can use less memory.

~~~bash
python -m triplet_coevolution.validation.shards_local \
  --run results/mixed --config examples/mixed/null-policy.json \
  --plan results/null-plan --shards 5 --jobs 3 --workers 1 \
  --memory-mib 2048 --out results/null-merged
~~~

After a process failure, use the existing plan and output journals. Already completed partitions verify their receipts. A partial partition resumes from its last committed replicate. Choose a new merge destination or the original empty destination. The launcher prints process IDs and exit codes as JSON and saves per-partition draw events under `logs/`.

~~~bash
python -m triplet_coevolution.validation.shards_local \
  --plan results/null-plan --jobs 3 --workers 1 --memory-mib 2048 \
  --resume --out results/null-merged
~~~

The process count times the worker count must fit the detected CPUs. A local child loads the source model once per partition and shares it among its worker threads. Its separate interpreter has a separate copy. The launcher pins six common numerical thread settings to one before NumPy starts, budgets each child's declared numerical memory and reserves interpreter overhead. Actual RSS depends on the interpreter, allocator, libraries and workload. Set process or scheduler limits for a hard memory ceiling. For tiny analyses process startup and filesystem commits can cost more time than the draws save.

## Run stages separately

~~~bash
triplet shard-plan --run results/mixed --config examples/mixed/null-policy.json \
  --shards 3 --out results/null-plan
triplet null-shard --plan results/null-plan --shard-id 0 --progress
triplet null-shard --plan results/null-plan --shard-id 1 --progress
triplet null-shard --plan results/null-plan --shard-id 2 --progress
triplet shard-status --plan results/null-plan
triplet merge-null --plan results/null-plan --out results/null-merged
~~~

Workers may run concurrently or out of order. The default partition at index zero lives under `shards/shard-00000/`. If there are 127 draws in three partitions, their sizes are 43, 42 and 42. A different assignment can use `null-shard --ids ids.json --out DIR` with an explicit JSON array of distinct IDs. Pass every resulting directory through repeated `--shard DIR` options at merge. Granularity is one complete, coherent replicate across all families. Each final family maximum therefore retains the same within-draw dependence as in a serial run.

The declared draw count must equal the final planned count. To distribute a tree-sensitivity study, prepare and label one plan for each prespecified scenario. Serial `calibrate` also supports its multi-scenario comparison report. A relocated matched alignment can be supplied through the regular checksummed `--inputs` mapping. Keep the configured policy and run manifest bytes identical. A plan can move with its source and config at the same relative locations.

## Check integrity and monitor progress

- The scientific header binds the original input hashes, observation order, column coordinates, score archive, candidate selection, modeled null, masks, weight policy, RNG, seed, score directions, finite observations, fixed family and total draws. The final serial and merged headers share the same scientific plan hash. The native plan document also binds its source summary, config bytes, runtime and partitioning.
- A shard checks the package and evaluator source hashes, Python, NumPy and Biopython versions, NumPy's available build metadata, architecture and declared thread environment. Use the same build and numerical environment across workers. These checks do not establish cross-platform bit identity for different BLAS implementations or hardware dispatch.
- Each partition has one OS writer lock and a SQLite rollback journal. Each transaction writes one global ID and its packed exceedance/invalid masks and optional maxima. A failed transaction contributes zero draws. Commit ordering can differ across workers. The receipt commits to the sorted logical evidence, so scheduling order cannot change the digest. Its checksums detect inconsistencies. They do not authenticate a malicious party controlling the software, inputs and all hashes.
- The receipt is written by a flushed atomic rename after the assigned draws finish. The merger reads each journal in one read-only transaction, verifies its chain, identity, receipt and counts, and requires each ID exactly once in the interval from zero through the final draw minus one. Only a successful complete merge publishes calibration.tsv. The source run and partial shards remain available after a failure.
- `shard-status` displays provisional durable counters, the last available timing estimate and missing partitions as JSON. A receipt-present status reports the existence of the marker. The merge performs the full evidence verification. ETA assumes measured workers continue running concurrently. Before enough timed draws finish, or for unstarted partitions, the estimate is null. Each shard also emits a flushed progress line per completed draw with `--progress`.

SQLite and the writer lock require a filesystem with reliable locking, transactions and atomic rename. If cluster policy requires local scratch, give each process its own durable partition or copy the **completed** partition directory to the merge host. Retain the journal and receipt together, stop writers before transfer and check scheduler storage quotas. Requeue from the previous durable location. An incomplete copied journal cannot certify a completed ensemble.

## Optional Slurm array

The array template takes the environment's absolute executable and plan paths. Set the six thread declarations before preparing a plan for it. Submit resource requests appropriate to your cluster. The paths below are examples to set in your shell. Quoting preserves spaces in executable and input paths.

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 GOTO_NUM_THREADS=1
export TRIPLET_BIN="$(command -v triplet)"
export TRIPLET_PLAN="$PWD/results/null-plan"
"$TRIPLET_BIN" shard-plan --run results/mixed --config examples/mixed/null-policy.json \
  --shards 3 --out "$TRIPLET_PLAN"
mkdir -p "$TRIPLET_PLAN/logs"
sbatch --array=0-2 --cpus-per-task=1 --mem=1G --time=00:20:00 \
  --output="$TRIPLET_PLAN/logs/%A_%a.log" --export=ALL examples/hpc/null-array.sh
~~~

The script records the scheduler job ID, array index, worker count and installed program version. It invokes the same resume-capable command used on a workstation. Load the desired modules or environment before preparing and submitting the plan. When all tasks complete, use the same `merge-null` command. The array body has been exercised locally with scheduler variables. Real scheduler deployment depends on the site's filesystem and job policy.

## Python API

`prepare_shard_plan(observed, evaluate, *, draws, seed, null_name, identity, ...)` accepts the same full observed vectors, per-family directions and callback semantics as `run_null`. Provide complete input/candidate/model digests for custom callbacks. Reconstruct resources independently in each process. `shard_ids(draws, shard_id, shards)` supplies a balanced zero-based range. `run_shard(plan, evaluate, directory, *, replicate_ids, ...)` returns a checksummed receipt. `merge_shards(plan, directories, ...)` returns the native `NullRun`. The original observed arrays are copied into read-only plan vectors. The callback receives the global logical ID and its keyed RNG streams. Reusing identical seeds with different hypothesis families does not make those families interchangeable.

The mixed-alphabet validation records timings, process-tree RSS, thread counts, a forced interruption and exact table comparison. [Distributed null benchmark](../benchmarks/distributed-nulls.md)
