# Distributed calibration on a known matched alignment

This regression uses the packaged synthetic mixed-protein/DNA/RNA example. It refits MI and MI–APC under whole-component rematching with seed 942, 127 draws, three interfaces per statistic and joint maxT. Six prespecified families contain 80 coordinate tests. The balanced shard sizes are 43, 42 and 42.

One WSL Linux measurement ran the serial job, three partition processes, a merge in partition order 2, 0, 1, a verified replay of all completed partitions, and the unmodified Slurm array script under locally supplied array environment variables. Each numerical process declared one BLAS and OpenMP thread. The local runner used three CPU workers and its coordinator. The suite used preinstalled NumPy, Biopython and the package source in a temporary directory, without external services or private data.

| Stage | Wall seconds | Peak sampled process-tree RSS, MiB | Peak OS threads | Peak processes |
|---|---:|---:|---:|---:|
| Create native source archive | 0.770 | 70.3 | 1 | 1 |
| Serial calibration | 0.527 | 70.0 | 1 | 1 |
| Prepare shard plan | 0.386 | 66.9 | 1 | 1 |
| Evaluate three partitions concurrently | 0.612 | 251.4 | 4 | 4 |
| Verify and merge out of order | 0.386 | 67.5 | 1 | 1 |
| Verify all complete journals through resume | 0.510 | 242.9 | 4 | 4 |
| Replay one array task with the installed command locally | 0.526 | 66.7 | 2 | 2 |

The final serial and merged calibration.tsv files were byte-identical, including coordinate order, observed statistics, finite and invalid counts, two Monte Carlo bounds, both FDR corrections and maxT. Their SHA-256 was `b9a7691afd84319a16654fc0eea25e07aa3c0a20f52ed2181060c50a2e50966b`. The sorted replicate evidence digest was `383f0aaba48fd76c4647a2253f749f8d4cc7258a188ed2dc938a81ac0ae040c9`. The three SQLite evidence files occupied 98,304 bytes in total. Process startup and reload overhead outweighed parallel speedup for this small example. RSS was sampled every 20 ms and sums RSS across processes, including shared pages. It can miss shorter peaks.

Repeating the same design at **4,099 draws** yielded byte-identical final tables with SHA-256 `1657054c0402f7e6dcd4e9074859eca1f320f72607d326c6024d53241cd3dac6` and replicate evidence SHA-256 `09a80be9303d98a91f5b227c199db8572c0687e7c3d4119a59b8f0d9e23d33c2`. Serial calibration took 4.348 s and peaked at 70.9 MiB. Preparing the distributed plan took 0.324 s, evaluating the three partitions took 2.054 s with four OS threads and 253.0 MiB sampled process-tree RSS, and merging took 0.487 s. This gave 2.865 s end to end after the common native fit. The partition journals occupied 749,568 bytes and the complete three-partition resume took 0.488 s. These single-host measurements compare elapsed work and memory under this particular design.

The independent process regression launches three native MI partitions on the same public input, kills one process during its third logical draw and checks the writer lock from a competing process. The partial partition has two durable draws and no completion receipt. A two-worker resume reconstructs the remaining draws, and the merged raw evidence, p-values and maxT agree with the serial calculation. Additional tests cover scattered IDs, nonfinite draws and observed values, lower tails and ties, changed seeds, changed input identities and finite masks, a changed evaluator, missing IDs, duplicate partitions, forged or corrupted receipts and altered SQLite payloads. An actual SQLite capacity error after the first draw and an OS error during receipt publication both leave transactionally recoverable evidence. Live output never uses an interim shard to publish FDR.

Run the measurement and tests from the checkout.

~~~bash
python benchmarks/distributed_nulls.py
python benchmarks/distributed_nulls.py --draws 4099
python -m pytest tests/test_null_shards.py tests/test_null_runner.py tests/test_calibrate_command.py
bash -n examples/hpc/null-array.sh
~~~

The benchmark reports process-tree memory and thread counts on Linux. On another platform the same script verifies numerical output and reports wall time, with unavailable process counters shown as null. Real cluster scheduling, queue latency, distributed filesystem behavior and performance on other datasets require site-specific measurement. [Distributed calibration commands](../docs/distributed-nulls.md)
