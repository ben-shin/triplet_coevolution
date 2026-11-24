# Null execution and interruption recovery

A native calibration with 97 whole-component matching draws produced a byte-identical hypothesis table after an operating-system process kill. The run used the bundled mixed protein, DNA and RNA alignments, estimated equal-component sequence weights for every rematching, scored MI and MI/APC, and computed interface-specific upper-tail tests and maxT results. The parent process terminated the two-worker run after observing committed database records. The checkpoint contained 19 completed draws. A three-worker process restored those records and computed 78 more. A separate serial run computed all 97.

| Checked artifact | SHA-256 |
| --- | --- |
| Complete calibration.tsv | 3ead965c8f7e679eace508ebf08b72e7b1504efa32682b2df8ef21d03af9c33b |
| Finite exceedance and invalid-draw counts | 701fa6f30442fc1ef4c3f95000c85fc06506b82a255e44eb7e67412fa0dc73ff |
| Ordered replicate evidence | e739ac11bdc9bbf577c1c910e28e34f6e012f729f8d782598c57171b7ea7b23f |

The database occupied 40,960 bytes. A second process kill restored 28 committed draws and reproduced all three hashes. A retry with a changed seed failed the identity check. The exact number of records committed before the kill depends on scheduling. The three scientific hashes are invariant across those schedules on the tested platform. Timestamps, process exit codes, progress order and elapsed time describe individual executions.

## Restarting a run

Give the checkpoint a dedicated local directory outside the source analysis and output directories. Set planned_draws and the seed before inspecting results. Execute a prefix with a smaller draws value if required. A prefix is labeled interim_exploratory. A resumed execution may request additional IDs up to the original planned_draws. A changed endpoint beyond that plan requires a new scientific plan.

The SQLite journal stores separate packed finite-exceedance and invalid-draw masks for each family and replicate. A coherent maxT run also records its oriented family maximum in binary64. The journal binds the observed scores and validity masks, family definitions and directions, input and coordinate identities, fitted method settings, seed, RNG version, planned endpoint, and installed package sources. The native CLI supplies the row, mask, coordinate, method and archive provenance. A library caller adds the identities of any custom model, external input or preprocessing step to its identity mapping.

Each replicate and its completion state commit in one SQLite transaction using full synchronous writes. The permanent writer-lock file uses an operating-system file lock, released automatically on process death. Resume verifies every record, payload digest, contiguous commit ordinal, unique logical replicate ID and completion-chain head. A partially completed parallel run can legitimately contain logical IDs out of order. The final set must cover every requested logical ID exactly once. Checkpoints expect local storage with working file locks and SQLite durability semantics.

After a hard kill, the output directory can contain the interrupted staging directory. Point the resumed process at the same checkpoint and a new or empty output directory. The native CLI verifies the original score archive, settings and source files again. The root calibration.json is the published manifest.

## Random streams, counts and resources

The runner derives a fresh PCG64 stream from the seed, null name, zero-based logical replicate ID and a typed logical key using SHA-256 and little-endian entropy words. A matching draw uses one stream across its coherent score vector. Independent local bootstrap candidates each use their stable coordinate ID. Changing thread count, scheduling, candidate evaluation order or a prefix endpoint preserves each logical stream. Python, NumPy, Biopython, evaluator source and package source identities are recorded. Library callbacks supply deterministic scoring code and thread-safe state. Floating-point differences between numerical libraries or hardware can still affect nearly tied fitted scores. The test suite includes fixed reference PCG64 words and a numerical tie rule.

A finite null score counts as an exceedance if its oriented value equals or exceeds the observed value within 64 binary64 epsilons times the greater of one and the observed magnitude. Lower-tail scores are sign-reversed for this comparison. Each nonfinite null score counts conservatively as an exceedance. The reported empirical p-value is (1 + finite exceedances + invalid draws)/(1 + completed draws). An ineligible observation stays in its declared family at p=1. Each result also carries its finite and invalid draw denominators. For joint maxT, one nonfinite member makes that replicate's family maximum unknown, and the draw counts conservatively for every threshold. Empty or unresolved maxima are recorded as NaN. This correction requires jointly simulated statistics in comparable units within a declared family.

The coordinator uses a bounded thread pool over shared input arrays. It holds at most one active or buffered replicate per worker, compact evidence for completed IDs, per-coordinate count vectors and, when requested, a vector of family maxima in logical replicate order. A family of H hypotheses uses two times ceil(H/8) journal payload bytes per draw, plus eight bytes for a requested maximum. SQLite indexing and transaction pages add overhead. The preflight estimates both the runner's working arrays and checkpoint storage. The native command additionally budgets score fitting for all requested workers. The library defaults to a 512 MiB runner workspace envelope. Set BLAS and OpenMP thread counts explicitly when benchmarking multiple workers.

The compact journal supports tails, validity counts and joint maxima. Per-coordinate null means, standard deviations and quantiles require a different data-retention plan. Existing sequential v1 permutation results retain their original random stream in the legacy workflow.

## Reproduction

Run the following from the repository root in the package environment. The process test uses temporary paths and public fixtures. It requires no external datasets or services.

```bash
python benchmarks/null_resume_proof.py
python -m pytest -q tests/test_null_runner.py
```

The runner tests also kill a worker inside an in-flight native MI replicate after exactly five committed draws. They exercise invalid refits, both score directions, numerical ties, altered inputs and masks, writer exclusion, table corruption, a failed SQL state update, an actual SQLite capacity error using a capped test database, and an exception from a worker. The independently supplied serial and reversed-order parallel schedules produce the same ordered evidence and count hashes.

