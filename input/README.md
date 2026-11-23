# Three-family input

Create three folders here and name each after its protein or nucleic acid. Put a Stockholm family alignment in each folder. Files added here are ignored by Git.

```text
input/
  ATPase/family.sto.gz
  Partner/family.sto
  guide_RNA/family.sto.gz
```

Run this from the repository root after installing the package.

```shell
triplet run
```

The workflow reads family annotations, looks up supported public identifiers, pairs unambiguous family members, selects coordinate references and scores supported observations. It benchmarks associations and prints the path to a local HTML report in a new `results/run-…` directory. Each invocation finishes without prompts. Use `triplet run --offline` for local annotations and cached records only.

Recovering a match requires a shared genome or assembly, one identifiable locus for each role and sufficient evidence for family or orthogroup membership. Unplaced sequences, ambiguous copies and incomplete groups appear in the evidence tables. When no triplet qualifies, the report explains the outcome. Optional input files include `mappings.tsv`, reusable lookup snapshots, per-family `targets.txt` and `project.json` for preconfigured overrides. See the [automatic run guide](../docs/automatic.md) or the [strict evidence preparation guide](../docs/families.md).
