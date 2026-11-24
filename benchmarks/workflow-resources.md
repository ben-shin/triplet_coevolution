# Local workflow resource measurements

The supplied GatABC aligned preparation configuration produced 888 matched observations. A local Windows 11 x86-64 run used CPython 3.13.15, NumPy 2.5.5, Biopython 1.88, OpenBLAS 0.3.34 and one BLAS thread. No public lookups, DCA, triplet fits or statistical nulls were involved. Benchmark output was written outside the source dataset.

| Operation | Wall time | Published artifact bytes | Peak worker resident bytes |
|---|---:|---:|---:|
| Prepare and plan in three new workspaces | 4.431, 4.542, 4.395 s | 7,142,683 preparation + 2,489,239 plan | 88,002,560–88,240,128 preparation; 118,095,872–118,173,696 plan |
| Verify/reuse the same preparation and plan | 0.186, 0.184, 0.209 s | 0 new stage bytes | No stage worker |
| Complete native pair-MI score after preparation and planning | 4.569 s in score worker; 9.233 s whole cold workflow | 8,858,587 scoring | 137,641,984 scoring |
| Verify/reuse preparation, plan and score | 0.226 s | 0 new stage bytes | No stage worker |

The native plan estimated 281,670,048 observation–pair visits, 133,251,680 bytes of peak numeric workspace, 10,150,272 bytes of uncompressed matrix payload and 768,000 bytes for primary text tables. The measured peak RSS includes the Python interpreter and library costs. The observed artifact directory includes compressed matrices, tables, metadata and an offline HTML report. The two quantities answer different budgeting questions. The three repeats cover run-to-run variation on this machine; the single score run is an observation with no interval estimate.

Reproduce on local inputs with `python benchmarks/workflow_resources.py --config PATH --until plan --repeats 3` and `--until score --repeats 1`. Each repeat uses a fresh scratch workspace and then verifies reuse. The JSON records all elapsed times, worker RSS and published byte totals. The fixture configuration `examples/workflow/prepare.json` exercises the same commands on eight observations. [Kernel parity and larger measured panels](../docs/kernel-performance.md) · [Distributed null measurements](distributed-nulls.md)
